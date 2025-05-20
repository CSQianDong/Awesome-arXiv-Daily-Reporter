# CIE: Controlling Language Model Text Generations Using Continuous Signals 

**Authors**: Vinay Samuel, Harshita Diddee, Yiming Zhang, Daphne Ippolito  

**Link**: [PDF](https://arxiv.org/pdf/2505.13448)  

**Abstract**: Aligning language models with user intent is becoming increasingly relevant to enhance user experience. This calls for designing methods that can allow users to control the properties of the language that LMs generate. For example, controlling the length of the generation, the complexity of the language that gets chosen, the sentiment, tone, etc. Most existing work attempts to integrate users' control by conditioning LM generations on natural language prompts or discrete control signals, which are often brittle and hard to scale. In this work, we are interested in \textit{continuous} control signals, ones that exist along a spectrum that can't easily be captured in a natural language prompt or via existing techniques in conditional generation. Through a case study in controlling the precise response-length of generations produced by LMs, we demonstrate how after fine-tuning, behaviors of language models can be controlled via continuous signals -- as vectors that are interpolated between a "low" and a "high" token embedding. Our method more reliably exerts response-length control than in-context learning methods or fine-tuning methods that represent the control signal as a discrete signal. Our full open-sourced code and datasets are available at this https URL. 

---
# ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models 

**Authors**: Liyan Tang, Grace Kim, Xinyu Zhao, Thom Lake, Wenxuan Ding, Fangcong Yin, Prasann Singhal, Manya Wadhwa, Zeyu Leo Liu, Zayne Sprague, Ramya Namuduri, Bodun Hu, Juan Diego Rodriguez, Puyuan Peng, Greg Durrett  

**Link**: [PDF](https://arxiv.org/pdf/2505.13444)  

**Abstract**: Chart understanding presents a unique challenge for large vision-language models (LVLMs), as it requires the integration of sophisticated textual and visual reasoning capabilities. However, current LVLMs exhibit a notable imbalance between these skills, falling short on visual reasoning that is difficult to perform in text. We conduct a case study using a synthetic dataset solvable only through visual reasoning and show that model performance degrades significantly with increasing visual complexity, while human performance remains robust. We then introduce ChartMuseum, a new Chart Question Answering (QA) benchmark containing 1,162 expert-annotated questions spanning multiple reasoning types, curated from real-world charts across 184 sources, specifically built to evaluate complex visual and textual reasoning. Unlike prior chart understanding benchmarks -- where frontier models perform similarly and near saturation -- our benchmark exposes a substantial gap between model and human performance, while effectively differentiating model capabilities: although humans achieve 93% accuracy, the best-performing model Gemini-2.5-Pro attains only 63.0%, and the leading open-source LVLM Qwen2.5-VL-72B-Instruct achieves only 38.5%. Moreover, on questions requiring primarily visual reasoning, all models experience a 35%-55% performance drop from text-reasoning-heavy question performance. Lastly, our qualitative error analysis reveals specific categories of visual reasoning that are challenging for current LVLMs. 

---
# SMOTExT: SMOTE meets Large Language Models 

**Authors**: Mateusz Bystroński, Mikołaj Hołysz, Grzegorz Piotrowski, Nitesh V. Chawla, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2505.13434)  

**Abstract**: Data scarcity and class imbalance are persistent challenges in training robust NLP models, especially in specialized domains or low-resource settings. We propose a novel technique, SMOTExT, that adapts the idea of Synthetic Minority Over-sampling (SMOTE) to textual data. Our method generates new synthetic examples by interpolating between BERT-based embeddings of two existing examples and then decoding the resulting latent point into text with xRAG architecture. By leveraging xRAG's cross-modal retrieval-generation framework, we can effectively turn interpolated vectors into coherent text. While this is preliminary work supported by qualitative outputs only, the method shows strong potential for knowledge distillation and data augmentation in few-shot settings. Notably, our approach also shows promise for privacy-preserving machine learning: in early experiments, training models solely on generated data achieved comparable performance to models trained on the original dataset. This suggests a viable path toward safe and effective learning under data protection constraints. 

---
# Dementia Through Different Eyes: Explainable Modeling of Human and LLM Perceptions for Early Awareness 

**Authors**: Lotem Peled-Cohen, Maya Zadok, Nitay Calderon, Hila Gonen, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2505.13418)  

**Abstract**: Cognitive decline often surfaces in language years before diagnosis. It is frequently non-experts, such as those closest to the patient, who first sense a change and raise concern. As LLMs become integrated into daily communication and used over prolonged periods, it may even be an LLM that notices something is off. But what exactly do they notice--and should be noticing--when making that judgment? This paper investigates how dementia is perceived through language by non-experts. We presented transcribed picture descriptions to non-expert humans and LLMs, asking them to intuitively judge whether each text was produced by someone healthy or with dementia. We introduce an explainable method that uses LLMs to extract high-level, expert-guided features representing these picture descriptions, and use logistic regression to model human and LLM perceptions and compare with clinical diagnoses. Our analysis reveals that human perception of dementia is inconsistent and relies on a narrow, and sometimes misleading, set of cues. LLMs, by contrast, draw on a richer, more nuanced feature set that aligns more closely with clinical patterns. Still, both groups show a tendency toward false negatives, frequently overlooking dementia cases. Through our interpretable framework and the insights it provides, we hope to help non-experts better recognize the linguistic signs that matter. 

---
# AdaptThink: Reasoning Models Can Learn When to Think 

**Authors**: Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13417)  

**Abstract**: Recently, large reasoning models have achieved impressive performance on various tasks by employing human-like deep thinking. However, the lengthy thinking process substantially increases inference overhead, making efficiency a critical bottleneck. In this work, we first demonstrate that NoThinking, which prompts the reasoning model to skip thinking and directly generate the final solution, is a better choice for relatively simple tasks in terms of both performance and efficiency. Motivated by this, we propose AdaptThink, a novel RL algorithm to teach reasoning models to choose the optimal thinking mode adaptively based on problem difficulty. Specifically, AdaptThink features two core components: (1) a constrained optimization objective that encourages the model to choose NoThinking while maintaining the overall performance; (2) an importance sampling strategy that balances Thinking and NoThinking samples during on-policy training, thereby enabling cold start and allowing the model to explore and exploit both thinking modes throughout the training process. Our experiments indicate that AdaptThink significantly reduces the inference costs while further enhancing performance. Notably, on three math datasets, AdaptThink reduces the average response length of DeepSeek-R1-Distill-Qwen-1.5B by 53% and improves its accuracy by 2.4%, highlighting the promise of adaptive thinking-mode selection for optimizing the balance between reasoning quality and efficiency. Our codes and models are available at this https URL. 

---
# Granary: Speech Recognition and Translation Dataset in 25 European Languages 

**Authors**: Nithin Rao Koluguri, Monica Sekoyan, George Zelenfroynd, Sasha Meister, Shuoyang Ding, Sofia Kostandian, He Huang, Nikolay Karpov, Jagadeesh Balam, Vitaly Lavrukhin, Yifan Peng, Sara Papi, Marco Gaido, Alessio Brutti, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2505.13404)  

**Abstract**: Multi-task and multilingual approaches benefit large models, yet speech processing for low-resource languages remains underexplored due to data scarcity. To address this, we present Granary, a large-scale collection of speech datasets for recognition and translation across 25 European languages. This is the first open-source effort at this scale for both transcription and translation. We enhance data quality using a pseudo-labeling pipeline with segmentation, two-pass inference, hallucination filtering, and punctuation restoration. We further generate translation pairs from pseudo-labeled transcriptions using EuroLLM, followed by a data filtration pipeline. Designed for efficiency, our pipeline processes vast amount of data within hours. We assess models trained on processed data by comparing their performance on previously curated datasets for both high- and low-resource languages. Our findings show that these models achieve similar performance using approx. 50% less data. Dataset will be made available at this https URL 

---
# MR. Judge: Multimodal Reasoner as a Judge 

**Authors**: Renjie Pi, Felix Bai, Qibin Chen, Simon Wang, Jiulong Shan, Kieran Liu, Meng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13403)  

**Abstract**: The paradigm of using Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) as evaluative judges has emerged as an effective approach in RLHF and inference-time scaling. In this work, we propose Multimodal Reasoner as a Judge (MR. Judge), a paradigm for empowering general-purpose MLLMs judges with strong reasoning capabilities. Instead of directly assigning scores for each response, we formulate the judgement process as a reasoning-inspired multiple-choice problem. Specifically, the judge model first conducts deliberate reasoning covering different aspects of the responses and eventually selects the best response from them. This reasoning process not only improves the interpretibility of the judgement, but also greatly enhances the performance of MLLM judges. To cope with the lack of questions with scored responses, we propose the following strategy to achieve automatic annotation: 1) Reverse Response Candidates Synthesis: starting from a supervised fine-tuning (SFT) dataset, we treat the original response as the best candidate and prompt the MLLM to generate plausible but flawed negative candidates. 2) Text-based reasoning extraction: we carefully design a data synthesis pipeline for distilling the reasoning capability from a text-based reasoning model, which is adopted to enable the MLLM judges to regain complex reasoning ability via warm up supervised fine-tuning. Experiments demonstrate that our MR. Judge is effective across a wide range of tasks. Specifically, our MR. Judge-7B surpasses GPT-4o by 9.9% on VL-RewardBench, and improves performance on MM-Vet during inference-time scaling by up to 7.7%. 

---
# R3: Robust Rubric-Agnostic Reward Models 

**Authors**: David Anugraha, Zilu Tang, Lester James V. Miranda, Hanyang Zhao, Mohammad Rifqi Farhansyah, Garry Kuwanto, Derry Wijaya, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2505.13388)  

**Abstract**: Reward models are essential for aligning language model outputs with human preferences, yet existing approaches often lack both controllability and interpretability. These models are typically optimized for narrow objectives, limiting their generalizability to broader downstream tasks. Moreover, their scalar outputs are difficult to interpret without contextual reasoning. To address these limitations, we introduce R3, a novel reward modeling framework that is rubric-agnostic, generalizable across evaluation dimensions, and provides interpretable, reasoned score assignments. R3 enables more transparent and flexible evaluation of language models, supporting robust alignment with diverse human values and use cases. Our models, data, and code are available as open source at this https URL 

---
# Thinkless: LLM Learns When to Think 

**Authors**: Gongfan Fang, Xinyin Ma, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13379)  

**Abstract**: Reasoning Language Models, capable of extended chain-of-thought reasoning, have demonstrated remarkable performance on tasks requiring complex logical inference. However, applying elaborate reasoning for all queries often results in substantial computational inefficiencies, particularly when many problems admit straightforward solutions. This motivates an open question: Can LLMs learn when to think? To answer this, we propose Thinkless, a learnable framework that empowers an LLM to adaptively select between short-form and long-form reasoning, based on both task complexity and the model's ability. Thinkless is trained under a reinforcement learning paradigm and employs two control tokens, <short> for concise responses and <think> for detailed reasoning. At the core of our method is a Decoupled Group Relative Policy Optimization (DeGRPO) algorithm, which decomposes the learning objective of hybrid reasoning into two components: (1) a control token loss that governs the selection of the reasoning mode, and (2) a response loss that improves the accuracy of the generated answers. This decoupled formulation enables fine-grained control over the contributions of each objective, stabilizing training and effectively preventing collapse observed in vanilla GRPO. Empirically, on several benchmarks such as Minerva Algebra, MATH-500, and GSM8K, Thinkless is able to reduce the usage of long-chain thinking by 50% - 90%, significantly improving the efficiency of Reasoning Language Models. The code is available at this https URL 

---
# What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts 

**Authors**: Chenyang Yang, Yike Shi, Qianou Ma, Michael Xieyang Liu, Christian Kästner, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13360)  

**Abstract**: Building LLM-powered software requires developers to communicate their requirements through natural language, but developer prompts are frequently underspecified, failing to fully capture many user-important requirements. In this paper, we present an in-depth analysis of prompt underspecification, showing that while LLMs can often (41.1%) guess unspecified requirements by default, such behavior is less robust: Underspecified prompts are 2x more likely to regress over model or prompt changes, sometimes with accuracy drops by more than 20%. We then demonstrate that simply adding more requirements to a prompt does not reliably improve performance, due to LLMs' limited instruction-following capabilities and competing constraints, and standard prompt optimizers do not offer much help. To address this, we introduce novel requirements-aware prompt optimization mechanisms that can improve performance by 4.8% on average over baselines that naively specify everything in the prompt. Beyond prompt optimization, we envision that effectively managing prompt underspecification requires a broader process, including proactive requirements discovery, evaluation, and monitoring. 

---
# Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Reasoning 

**Authors**: Adam Štorek, Mukur Gupta, Samira Hajizadeh, Prashast Srivastava, Suman Jana  

**Link**: [PDF](https://arxiv.org/pdf/2505.13353)  

**Abstract**: Although modern Large Language Models (LLMs) support extremely large contexts, their effectiveness in utilizing long context for code reasoning remains unclear. This paper investigates LLM reasoning ability over code snippets within large repositories and how it relates to their recall ability. Specifically, we differentiate between lexical code recall (verbatim retrieval) and semantic code recall (remembering what the code does). To measure semantic recall, we propose SemTrace, a code reasoning technique where the impact of specific statements on output is attributable and unpredictable. We also present a method to quantify semantic recall sensitivity in existing benchmarks. Our evaluation of state-of-the-art LLMs reveals a significant drop in code reasoning accuracy as a code snippet approaches the middle of the input context, particularly with techniques requiring high semantic recall like SemTrace. Moreover, we find that lexical recall varies by granularity, with models excelling at function retrieval but struggling with line-by-line recall. Notably, a disconnect exists between lexical and semantic recall, suggesting different underlying mechanisms. Finally, our findings indicate that current code reasoning benchmarks may exhibit low semantic recall sensitivity, potentially underestimating LLM challenges in leveraging in-context information. 

---
# Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks 

**Authors**: Narek Maloyan, Bislan Ashinov, Dmitry Namiot  

**Link**: [PDF](https://arxiv.org/pdf/2505.13348)  

**Abstract**: Large Language Models (LLMs) are increasingly employed as evaluators (LLM-as-a-Judge) for assessing the quality of machine-generated text. This paradigm offers scalability and cost-effectiveness compared to human annotation. However, the reliability and security of such systems, particularly their robustness against adversarial manipulations, remain critical concerns. This paper investigates the vulnerability of LLM-as-a-Judge architectures to prompt-injection attacks, where malicious inputs are designed to compromise the judge's decision-making process. We formalize two primary attack strategies: Comparative Undermining Attack (CUA), which directly targets the final decision output, and Justification Manipulation Attack (JMA), which aims to alter the model's generated reasoning. Using the Greedy Coordinate Gradient (GCG) optimization method, we craft adversarial suffixes appended to one of the responses being compared. Experiments conducted on the MT-Bench Human Judgments dataset with open-source instruction-tuned LLMs (Qwen2.5-3B-Instruct and Falcon3-3B-Instruct) demonstrate significant susceptibility. The CUA achieves an Attack Success Rate (ASR) exceeding 30\%, while JMA also shows notable effectiveness. These findings highlight substantial vulnerabilities in current LLM-as-a-Judge systems, underscoring the need for robust defense mechanisms and further research into adversarial evaluation and trustworthiness in LLM-based assessment frameworks. 

---
# J4R: Learning to Judge with Equivalent Initial State Group Relative Preference Optimization 

**Authors**: Austin Xu, Yilun Zhou, Xuan-Phi Nguyen, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2505.13346)  

**Abstract**: To keep pace with the increasing pace of large language models (LLM) development, model output evaluation has transitioned away from time-consuming human evaluation to automatic evaluation, where LLMs themselves are tasked with assessing and critiquing other model outputs. LLM-as-judge models are a class of generative evaluators that excel in evaluating relatively simple domains, like chat quality, but struggle in reasoning intensive domains where model responses contain more substantive and challenging content. To remedy existing judge shortcomings, we explore training judges with reinforcement learning (RL). We make three key contributions: (1) We propose the Equivalent Initial State Group Relative Policy Optimization (EIS-GRPO) algorithm, which allows us to train our judge to be robust to positional biases that arise in more complex evaluation settings. (2) We introduce ReasoningJudgeBench, a benchmark that evaluates judges in diverse reasoning settings not covered by prior work. (3) We train Judge for Reasoning (J4R), a 7B judge trained with EIS-GRPO that outperforms GPT-4o and the next best small judge by 6.7% and 9%, matching or exceeding the performance of larger GRPO-trained judges on both JudgeBench and ReasoningJudgeBench. 

---
# Contextual Paralinguistic Data Creation for Multi-Modal Speech-LLM: Data Condensation and Spoken QA Generation 

**Authors**: Qiongqiong Wang, Hardik B. Sailor, Tianchi Liu, Ai Ti Aw  

**Link**: [PDF](https://arxiv.org/pdf/2505.13338)  

**Abstract**: Current speech-LLMs exhibit limited capability in contextual reasoning alongside paralinguistic understanding, primarily due to the lack of Question-Answer (QA) datasets that cover both aspects. We propose a novel framework for dataset generation from in-the-wild speech data, that integrates contextual reasoning with paralinguistic information. It consists of a pseudo paralinguistic label-based data condensation of in-the-wild speech and LLM-based Contextual Paralinguistic QA (CPQA) generation. The effectiveness is validated by a strong correlation in evaluations of the Qwen2-Audio-7B-Instruct model on a dataset created by our framework and human-generated CPQA dataset. The results also reveal the speech-LLM's limitations in handling empathetic reasoning tasks, highlighting the need for such datasets and more robust models. The proposed framework is first of its kind and has potential in training more robust speech-LLMs with paralinguistic reasoning capabilities. 

---
# Rethinking Stateful Tool Use in Multi-Turn Dialogues: Benchmarks and Challenges 

**Authors**: Hongru Wang, Wenyu Huang, Yufei Wang, Yuanhao Xi, Jianqiao Lu, Huan Zhang, Nan Hu, Zeming Liu, Jeff Z. Pan, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2505.13328)  

**Abstract**: Existing benchmarks that assess Language Models (LMs) as Language Agents (LAs) for tool use primarily focus on stateless, single-turn interactions or partial evaluations, such as tool selection in a single turn, overlooking the inherent stateful nature of interactions in multi-turn applications. To fulfill this gap, we propose \texttt{DialogTool}, a multi-turn dialogue dataset with stateful tool interactions considering the whole life cycle of tool use, across six key tasks in three stages: 1) \textit{tool creation}; 2) \textit{tool utilization}: tool awareness, tool selection, tool execution; and 3) \textit{role-consistent response}: response generation and role play. Furthermore, we build \texttt{VirtualMobile} -- an embodied virtual mobile evaluation environment to simulate API calls and assess the robustness of the created APIs\footnote{We will use tools and APIs alternatively, there are no significant differences between them in this paper.}. Taking advantage of these artifacts, we conduct comprehensive evaluation on 13 distinct open- and closed-source LLMs and provide detailed analysis at each stage, revealing that the existing state-of-the-art LLMs still cannot perform well to use tools over long horizons. 

---
# GUARD: Generation-time LLM Unlearning via Adaptive Restriction and Detection 

**Authors**: Zhijie Deng, Chris Yuhao Liu, Zirui Pang, Xinlei He, Lei Feng, Qi Xuan, Zhaowei Zhu, Jiaheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.13312)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in memorizing vast amounts of knowledge across diverse domains. However, the ability to selectively forget specific knowledge is critical for ensuring the safety and compliance of deployed models. Existing unlearning efforts typically fine-tune the model with resources such as forget data, retain data, and a calibration model. These additional gradient steps blur the decision boundary between forget and retain knowledge, making unlearning often at the expense of overall performance. To avoid the negative impact of fine-tuning, it would be better to unlearn solely at inference time by safely guarding the model against generating responses related to the forget target, without destroying the fluency of text generation. In this work, we propose Generation-time Unlearning via Adaptive Restriction and Detection (GUARD), a framework that enables dynamic unlearning during LLM generation. Specifically, we first employ a prompt classifier to detect unlearning targets and extract the corresponding forbidden token. We then dynamically penalize and filter candidate tokens during generation using a combination of token matching and semantic matching, effectively preventing the model from leaking the forgotten content. Experimental results on copyright content unlearning tasks over the Harry Potter dataset and the MUSE benchmark, as well as entity unlearning tasks on the TOFU dataset, demonstrate that GUARD achieves strong forget quality across various tasks while causing almost no degradation to the LLM's general capabilities, striking an excellent trade-off between forgetting and utility. 

---
# RBF++: Quantifying and Optimizing Reasoning Boundaries across Measurable and Unmeasurable Capabilities for Chain-of-Thought Reasoning 

**Authors**: Qiguang Chen, Libo Qin, Jinhao Liu, Yue Liao, Jiaqi Wang, Jingxuan Zhou, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2505.13307)  

**Abstract**: Chain-of-Thought (CoT) reasoning has proven effective in enhancing large language models (LLMs) on complex tasks, spurring research into its underlying mechanisms. However, two primary challenges remain for real-world applications: (1) the lack of quantitative metrics and actionable guidelines for evaluating and optimizing measurable boundaries of CoT capability, and (2) the absence of methods to assess boundaries of unmeasurable CoT capability, such as multimodal perception. To address these gaps, we introduce the Reasoning Boundary Framework++ (RBF++). To tackle the first challenge, we define the reasoning boundary (RB) as the maximum limit of CoT performance. We also propose a combination law for RBs, enabling quantitative analysis and offering actionable guidance across various CoT tasks. For the second challenge, particularly in multimodal scenarios, we introduce a constant assumption, which replaces unmeasurable RBs with scenario-specific constants. Additionally, we propose the reasoning boundary division mechanism, which divides unmeasurable RBs into two sub-boundaries, facilitating the quantification and optimization of both unmeasurable domain knowledge and multimodal perception capabilities. Extensive experiments involving 38 models across 13 tasks validate the feasibility of our framework in cross-modal settings. Additionally, we evaluate 10 CoT strategies, offer insights into optimization and decay from two complementary perspectives, and expand evaluation benchmarks for measuring RBs in LLM reasoning. We hope this work advances the understanding of RBs and optimization strategies in LLMs. Code and data are available at this https URL. 

---
# I'll believe it when I see it: Images increase misinformation sharing in Vision-Language Models 

**Authors**: Alice Plebe, Timothy Douglas, Diana Riazi, R. Maria del Rio-Chanona  

**Link**: [PDF](https://arxiv.org/pdf/2505.13302)  

**Abstract**: Large language models are increasingly integrated into news recommendation systems, raising concerns about their role in spreading misinformation. In humans, visual content is known to boost credibility and shareability of information, yet its effect on vision-language models (VLMs) remains unclear. We present the first study examining how images influence VLMs' propensity to reshare news content, whether this effect varies across model families, and how persona conditioning and content attributes modulate this behavior. To support this analysis, we introduce two methodological contributions: a jailbreaking-inspired prompting strategy that elicits resharing decisions from VLMs while simulating users with antisocial traits and political alignments; and a multimodal dataset of fact-checked political news from PolitiFact, paired with corresponding images and ground-truth veracity labels. Experiments across model families reveal that image presence increases resharing rates by 4.8% for true news and 15.0% for false news. Persona conditioning further modulates this effect: Dark Triad traits amplify resharing of false news, whereas Republican-aligned profiles exhibit reduced veracity sensitivity. Of all the tested models, only Claude-3-Haiku demonstrates robustness to visual misinformation. These findings highlight emerging risks in multimodal model behavior and motivate the development of tailored evaluation frameworks and mitigation strategies for personalized AI systems. Code and dataset are available at: this https URL 

---
# $\textit{Rank, Chunk and Expand}$: Lineage-Oriented Reasoning for Taxonomy Expansion 

**Authors**: Sahil Mishra, Kumar Arjun, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2505.13282)  

**Abstract**: Taxonomies are hierarchical knowledge graphs crucial for recommendation systems, and web applications. As data grows, expanding taxonomies is essential, but existing methods face key challenges: (1) discriminative models struggle with representation limits and generalization, while (2) generative methods either process all candidates at once, introducing noise and exceeding context limits, or discard relevant entities by selecting noisy candidates. We propose LORex ($\textbf{L}$ineage-$\textbf{O}$riented $\textbf{Re}$asoning for Taxonomy E$\textbf{x}$pansion), a plug-and-play framework that combines discriminative ranking and generative reasoning for efficient taxonomy expansion. Unlike prior methods, LORex ranks and chunks candidate terms into batches, filtering noise and iteratively refining selections by reasoning candidates' hierarchy to ensure contextual efficiency. Extensive experiments across four benchmarks and twelve baselines show that LORex improves accuracy by 12% and Wu & Palmer similarity by 5% over state-of-the-art methods. 

---
# CSC-SQL: Corrective Self-Consistency in Text-to-SQL via Reinforcement Learning 

**Authors**: Lei Sheng, Shuai-Shuai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13271)  

**Abstract**: Large language models (LLMs) have demonstrated strong capabilities in translating natural language questions about relational databases into SQL queries. In particular, test-time scaling techniques such as Self-Consistency and Self-Correction can enhance SQL generation accuracy by increasing computational effort during inference. However, these methods have notable limitations: Self-Consistency may select suboptimal outputs despite majority votes, while Self-Correction typically addresses only syntactic errors. To leverage the strengths of both approaches, we propose CSC-SQL, a novel method that integrates Self-Consistency and Self-Correction. CSC-SQL selects the two most frequently occurring outputs from parallel sampling and feeds them into a merge revision model for correction. Additionally, we employ the Group Relative Policy Optimization (GRPO) algorithm to fine-tune both the SQL generation and revision models via reinforcement learning, significantly enhancing output quality. Experimental results confirm the effectiveness and generalizability of CSC-SQL. On the BIRD development set, our 3B model achieves 65.28% execution accuracy, while the 7B model achieves 69.19%. The code will be open sourced at this https URL. 

---
# Representation of perceived prosodic similarity of conversational feedback 

**Authors**: Livia Qian, Carol Figueroa, Gabriel Skantze  

**Link**: [PDF](https://arxiv.org/pdf/2505.13268)  

**Abstract**: Vocal feedback (e.g., `mhm', `yeah', `okay') is an important component of spoken dialogue and is crucial to ensuring common ground in conversational systems. The exact meaning of such feedback is conveyed through both lexical and prosodic form. In this work, we investigate the perceived prosodic similarity of vocal feedback with the same lexical form, and to what extent existing speech representations reflect such similarities. A triadic comparison task with recruited participants is used to measure perceived similarity of feedback responses taken from two different datasets. We find that spectral and self-supervised speech representations encode prosody better than extracted pitch features, especially in the case of feedback from the same speaker. We also find that it is possible to further condense and align the representations to human perception through contrastive learning. 

---
# From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery 

**Authors**: Tianshi Zheng, Zheye Deng, Hong Ting Tsang, Weiqi Wang, Jiaxin Bai, Zihao Wang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.13259)  

**Abstract**: Large Language Models (LLMs) are catalyzing a paradigm shift in scientific discovery, evolving from task-specific automation tools into increasingly autonomous agents and fundamentally redefining research processes and human-AI collaboration. This survey systematically charts this burgeoning field, placing a central focus on the changing roles and escalating capabilities of LLMs in science. Through the lens of the scientific method, we introduce a foundational three-level taxonomy-Tool, Analyst, and Scientist-to delineate their escalating autonomy and evolving responsibilities within the research lifecycle. We further identify pivotal challenges and future research trajectories such as robotic automation, self-improvement, and ethical governance. Overall, this survey provides a conceptual architecture and strategic foresight to navigate and shape the future of AI-driven scientific discovery, fostering both rapid innovation and responsible advancement. Github Repository: this https URL. 

---
# Effective and Transparent RAG: Adaptive-Reward Reinforcement Learning for Decision Traceability 

**Authors**: Jingyi Ren, Yekun Xu, Xiaolong Wang, Weitao Li, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13258)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly improved the performance of large language models (LLMs) on knowledge-intensive domains. However, although RAG achieved successes across distinct domains, there are still some unsolved challenges: 1) Effectiveness. Existing research mainly focuses on developing more powerful RAG retrievers, but how to enhance the generator's (LLM's) ability to utilize the retrieved information for reasoning and generation? 2) Transparency. Most RAG methods ignore which retrieved content actually contributes to the reasoning process, resulting in a lack of interpretability and visibility. To address this, we propose ARENA (Adaptive-Rewarded Evidence Navigation Agent), a transparent RAG generator framework trained via reinforcement learning (RL) with our proposed rewards. Based on the structured generation and adaptive reward calculation, our RL-based training enables the model to identify key evidence, perform structured reasoning, and generate answers with interpretable decision traces. Applied to Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct, abundant experiments with various RAG baselines demonstrate that our model achieves 10-30% improvements on all multi-hop QA datasets, which is comparable with the SOTA Commercially-developed LLMs (e.g., OpenAI-o1, DeepSeek-R1). Further analyses show that ARENA has strong flexibility to be adopted on new datasets without extra training. Our models and codes are publicly released. 

---
# WikiPersonas: What Can We Learn From Personalized Alignment to Famous People? 

**Authors**: Zilu Tang, Afra Feyza Akyürek, Ekin Akyürek, Derry Wijaya  

**Link**: [PDF](https://arxiv.org/pdf/2505.13257)  

**Abstract**: Preference alignment has become a standard pipeline in finetuning models to follow \emph{generic} human preferences. Majority of work seeks to optimize model to produce responses that would be preferable \emph{on average}, simplifying the diverse and often \emph{contradicting} space of human preferences. While research has increasingly focused on personalized alignment: adapting models to individual user preferences, there is a lack of personalized preference dataset which focus on nuanced individual-level preferences. To address this, we introduce WikiPersona: the first fine-grained personalization using well-documented, famous individuals. Our dataset challenges models to align with these personas through an interpretable process: generating verifiable textual descriptions of a persona's background and preferences in addition to alignment. We systematically evaluate different personalization approaches and find that as few-shot prompting with preferences and fine-tuning fail to simultaneously ensure effectiveness and efficiency, using \textit{inferred personal preferences} as prefixes enables effective personalization, especially in topics where preferences clash while leading to more equitable generalization across unseen personas. 

---
# HeteroSpec: Leveraging Contextual Heterogeneity for Efficient Speculative Decoding 

**Authors**: Siran Liu, Yang Ye, Qianchao Zhu, Zheng Cao, Yongchao He  

**Link**: [PDF](https://arxiv.org/pdf/2505.13254)  

**Abstract**: Autoregressive decoding, the standard approach for Large Language Model (LLM) inference, remains a significant bottleneck due to its sequential nature. While speculative decoding algorithms mitigate this inefficiency through parallel verification, they fail to exploit the inherent heterogeneity in linguistic complexity, a key factor leading to suboptimal resource allocation. We address this by proposing HeteroSpec, a heterogeneity-adaptive speculative decoding framework that dynamically optimizes computational resource allocation based on linguistic context complexity. HeteroSpec introduces two key mechanisms: (1) A novel cumulative meta-path Top-$K$ entropy metric for efficiently identifying predictable contexts. (2) A dynamic resource allocation strategy based on data-driven entropy partitioning, enabling adaptive speculative expansion and pruning tailored to local context difficulty. Evaluated on five public benchmarks and four models, HeteroSpec achieves an average speedup of 4.26$\times$. It consistently outperforms state-of-the-art EAGLE-3 across speedup rates, average acceptance length, and verification cost. Notably, HeteroSpec requires no draft model retraining, incurs minimal overhead, and is orthogonal to other acceleration techniques. It demonstrates enhanced acceleration with stronger draft models, establishing a new paradigm for context-aware LLM inference acceleration. 

---
# Natural Language Planning via Coding and Inference Scaling 

**Authors**: Rikhil Amonkar, Ronan Le Bras, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13252)  

**Abstract**: Real-life textual planning tasks such as meeting scheduling have posed much challenge to LLMs especially when the complexity is high. While previous work primarily studied auto-regressive generation of plans with closed-source models, we systematically evaluate both closed- and open-source models, including those that scales output length with complexity during inference, in generating programs, which are executed to output the plan. We consider not only standard Python code, but also the code to a constraint satisfaction problem solver. Despite the algorithmic nature of the task, we show that programming often but not always outperforms planning. Our detailed error analysis also indicates a lack of robustness and efficiency in the generated code that hinders generalization. 

---
# Stronger Together: Unleashing the Social Impact of Hate Speech Research 

**Authors**: Sidney Wong  

**Link**: [PDF](https://arxiv.org/pdf/2505.13251)  

**Abstract**: The advent of the internet has been both a blessing and a curse for once marginalised communities. When used well, the internet can be used to connect and establish communities crossing different intersections; however, it can also be used as a tool to alienate people and communities as well as perpetuate hate, misinformation, and disinformation especially on social media platforms. We propose steering hate speech research and researchers away from pre-existing computational solutions and consider social methods to inform social solutions to address this social problem. In a similar way linguistics research can inform language planning policy, linguists should apply what we know about language and society to mitigate some of the emergent risks and dangers of anti-social behaviour in digital spaces. We argue linguists and NLP researchers can play a principle role in unleashing the social impact potential of linguistics research working alongside communities, advocates, activists, and policymakers to enable equitable digital inclusion and to close the digital divide. 

---
# JNLP at SemEval-2025 Task 11: Cross-Lingual Multi-Label Emotion Detection Using Generative Models 

**Authors**: Jieying Xue, Phuong Minh Nguyen, Minh Le Nguyen, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13244)  

**Abstract**: With the rapid advancement of global digitalization, users from different countries increasingly rely on social media for information exchange. In this context, multilingual multi-label emotion detection has emerged as a critical research area. This study addresses SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection. Our paper focuses on two sub-tracks of this task: (1) Track A: Multi-label emotion detection, and (2) Track B: Emotion intensity. To tackle multilingual challenges, we leverage pre-trained multilingual models and focus on two architectures: (1) a fine-tuned BERT-based classification model and (2) an instruction-tuned generative LLM. Additionally, we propose two methods for handling multi-label classification: the base method, which maps an input directly to all its corresponding emotion labels, and the pairwise method, which models the relationship between the input text and each emotion category individually. Experimental results demonstrate the strong generalization ability of our approach in multilingual emotion recognition. In Track A, our method achieved Top 4 performance across 10 languages, ranking 1st in Hindi. In Track B, our approach also secured Top 5 performance in 7 languages, highlighting its simplicity and effectiveness\footnote{Our code is available at this https URL. 

---
# SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science 

**Authors**: Jie Ying, Zihong Chen, Zhefan Wang, Wanli Jiang, Chenyang Wang, Zhonghang Yuan, Haoyang Su, Huanjun Kong, Fan Yang, Nanqing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.13220)  

**Abstract**: Seed science is essential for modern agriculture, directly influencing crop yields and global food security. However, challenges such as interdisciplinary complexity and high costs with limited returns hinder progress, leading to a shortage of experts and insufficient technological support. While large language models (LLMs) have shown promise across various fields, their application in seed science remains limited due to the scarcity of digital resources, complex gene-trait relationships, and the lack of standardized benchmarks. To address this gap, we introduce SeedBench -- the first multi-task benchmark specifically designed for seed science. Developed in collaboration with domain experts, SeedBench focuses on seed breeding and simulates key aspects of modern breeding processes. We conduct a comprehensive evaluation of 26 leading LLMs, encompassing proprietary, open-source, and domain-specific fine-tuned models. Our findings not only highlight the substantial gaps between the power of LLMs and the real-world seed science problems, but also make a foundational step for research on LLMs for seed design. 

---
# Picturized and Recited with Dialects: A Multimodal Chinese Representation Framework for Sentiment Analysis of Classical Chinese Poetry 

**Authors**: Xiaocong Du, Haoyu Pei, Haipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13210)  

**Abstract**: Classical Chinese poetry is a vital and enduring part of Chinese literature, conveying profound emotional resonance. Existing studies analyze sentiment based on textual meanings, overlooking the unique rhythmic and visual features inherent in poetry,especially since it is often recited and accompanied by Chinese paintings. In this work, we propose a dialect-enhanced multimodal framework for classical Chinese poetry sentiment analysis. We extract sentence-level audio features from the poetry and incorporate audio from multiple dialects,which may retain regional ancient Chinese phonetic features, enriching the phonetic representation. Additionally, we generate sentence-level visual features, and the multimodal features are fused with textual features enhanced by LLM translation through multimodal contrastive representation learning. Our framework outperforms state-of-the-art methods on two public datasets, achieving at least 2.51% improvement in accuracy and 1.63% in macro F1. We open-source the code to facilitate research in this area and provide insights for general multimodal Chinese representation. 

---
# Alignment-Augmented Speculative Decoding with Alignment Sampling and Conditional Verification 

**Authors**: Jikai Wang, Zhenxu Tian, Juntao Li, Qingrong Xia, Xinyu Duan, Zhefeng Wang, Baoxing Huai, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13204)  

**Abstract**: Recent works have revealed the great potential of speculative decoding in accelerating the autoregressive generation process of large language models. The success of these methods relies on the alignment between draft candidates and the sampled outputs of the target model. Existing methods mainly achieve draft-target alignment with training-based methods, e.g., EAGLE, Medusa, involving considerable training costs. In this paper, we present a training-free alignment-augmented speculative decoding algorithm. We propose alignment sampling, which leverages output distribution obtained in the prefilling phase to provide more aligned draft candidates. To further benefit from high-quality but non-aligned draft candidates, we also introduce a simple yet effective flexible verification strategy. Through an adaptive probability threshold, our approach can improve generation accuracy while further improving inference efficiency. Experiments on 8 datasets (including question answering, summarization and code completion tasks) show that our approach increases the average generation score by 3.3 points for the LLaMA3 model. Our method achieves a mean acceptance length up to 2.39 and speed up generation by 2.23. 

---
# Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space 

**Authors**: Zhengrui Ma, Yang Feng, Chenze Shao, Fandong Meng, Jie Zhou, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13181)  

**Abstract**: We introduce SLED, an alternative approach to speech language modeling by encoding speech waveforms into sequences of continuous latent representations and modeling them autoregressively using an energy distance objective. The energy distance offers an analytical measure of the distributional gap by contrasting simulated and target samples, enabling efficient training to capture the underlying continuous autoregressive distribution. By bypassing reliance on residual vector quantization, SLED avoids discretization errors and eliminates the need for the complicated hierarchical architectures common in existing speech language models. It simplifies the overall modeling pipeline while preserving the richness of speech information and maintaining inference efficiency. Empirical results demonstrate that SLED achieves strong performance in both zero-shot and streaming speech synthesis, showing its potential for broader applications in general-purpose speech language models. 

---
# ToolSpectrum : Towards Personalized Tool Utilization for Large Language Models 

**Authors**: Zihao Cheng, Hongru Wang, Zeming Liu, Yuhang Guo, Yuanfang Guo, Yunhong Wang, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13176)  

**Abstract**: While integrating external tools into large language models (LLMs) enhances their ability to access real-time information and domain-specific services, existing approaches focus narrowly on functional tool selection following user instructions, overlooking the context-aware personalization in tool selection. This oversight leads to suboptimal user satisfaction and inefficient tool utilization, particularly when overlapping toolsets require nuanced selection based on contextual factors. To bridge this gap, we introduce ToolSpectrum, a benchmark designed to evaluate LLMs' capabilities in personalized tool utilization. Specifically, we formalize two key dimensions of personalization, user profile and environmental factors, and analyze their individual and synergistic impacts on tool utilization. Through extensive experiments on ToolSpectrum, we demonstrate that personalized tool utilization significantly improves user experience across diverse scenarios. However, even state-of-the-art LLMs exhibit the limited ability to reason jointly about user profiles and environmental factors, often prioritizing one dimension at the expense of the other. Our findings underscore the necessity of context-aware personalization in tool-augmented LLMs and reveal critical limitations for current models. Our data and code are available at this https URL. 

---
# A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs 

**Authors**: V.S.D.S.Mahesh Akavarapu, Hrishikesh Terdalkar, Pramit Bhattacharyya, Shubhangi Agarwal, Vishakha Deulgaonkar, Pralay Manna, Chaitali Dangarikar, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2505.13173)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable generalization capabilities across diverse tasks and languages. In this study, we focus on natural language understanding in three classical languages -- Sanskrit, Ancient Greek and Latin -- to investigate the factors affecting cross-lingual zero-shot generalization. First, we explore named entity recognition and machine translation into English. While LLMs perform equal to or better than fine-tuned baselines on out-of-domain data, smaller models often struggle, especially with niche or abstract entity types. In addition, we concentrate on Sanskrit by presenting a factoid question-answering (QA) dataset and show that incorporating context via retrieval-augmented generation approach significantly boosts performance. In contrast, we observe pronounced performance drops for smaller LLMs across these QA tasks. These results suggest model scale as an important factor influencing cross-lingual generalization. Assuming that models used such as GPT-4o and Llama-3.1 are not instruction fine-tuned on classical languages, our findings provide insights into how LLMs may generalize on these languages and their consequent utility in classical studies. 

---
# Positional Fragility in LLMs: How Offset Effects Reshape Our Understanding of Memorization Risks 

**Authors**: Yixuan Xu, Antoine Bosselut, Imanol Schlag  

**Link**: [PDF](https://arxiv.org/pdf/2505.13171)  

**Abstract**: Large language models are known to memorize parts of their training data, posing risk of copyright violations. To systematically examine this risk, we pretrain language models (1B/3B/8B) from scratch on 83B tokens, mixing web-scale data with public domain books used to simulate copyrighted content at controlled frequencies at lengths at least ten times longer than prior work. We thereby identified the offset effect, a phenomenon characterized by two key findings: (1) verbatim memorization is most strongly triggered by short prefixes drawn from the beginning of the context window, with memorization decreasing counterintuitively as prefix length increases; and (2) a sharp decline in verbatim recall when prefix begins offset from the initial tokens of the context window. We attribute this to positional fragility: models rely disproportionately on the earliest tokens in their context window as retrieval anchors, making them sensitive to even slight shifts. We further observe that when the model fails to retrieve memorized content, it often produces degenerated text. Leveraging these findings, we show that shifting sensitive data deeper into the context window suppresses both extractable memorization and degeneration. Our results suggest that positional offset is a critical and previously overlooked axis for evaluating memorization risks, since prior work implicitly assumed uniformity by probing only from the beginning of training sequences. 

---
# Role-Playing Evaluation for Large Language Models 

**Authors**: Yassine El Boudouri, Walter Nuninger, Julian Alvarez, Yvan Peter  

**Link**: [PDF](https://arxiv.org/pdf/2505.13157)  

**Abstract**: Large Language Models (LLMs) demonstrate a notable capacity for adopting personas and engaging in role-playing. However, evaluating this ability presents significant challenges, as human assessments are resource-intensive and automated evaluations can be biased. To address this, we introduce Role-Playing Eval (RPEval), a novel benchmark designed to assess LLM role-playing capabilities across four key dimensions: emotional understanding, decision-making, moral alignment, and in-character consistency. This article details the construction of RPEval and presents baseline evaluations. Our code and dataset are available at this https URL 

---
# Tianyi: A Traditional Chinese Medicine all-rounder language model and its Real-World Clinical Practice 

**Authors**: Zhi Liu, Tao Yang, Jing Wang, Yexin Chen, Zhan Gao, Jiaxi Yang, Kui Chen, Bingji Lu, Xiaochen Li, Changyong Luo, Yan Li, Xiaohong Gu, Peng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13156)  

**Abstract**: Natural medicines, particularly Traditional Chinese Medicine (TCM), are gaining global recognition for their therapeutic potential in addressing human symptoms and diseases. TCM, with its systematic theories and extensive practical experience, provides abundant resources for healthcare. However, the effective application of TCM requires precise syndrome diagnosis, determination of treatment principles, and prescription formulation, which demand decades of clinical expertise. Despite advancements in TCM-based decision systems, machine learning, and deep learning research, limitations in data and single-objective constraints hinder their practical application. In recent years, large language models (LLMs) have demonstrated potential in complex tasks, but lack specialization in TCM and face significant challenges, such as too big model scale to deploy and issues with hallucination. To address these challenges, we introduce Tianyi with 7.6-billion-parameter LLM, a model scale proper and specifically designed for TCM, pre-trained and fine-tuned on diverse TCM corpora, including classical texts, expert treatises, clinical records, and knowledge graphs. Tianyi is designed to assimilate interconnected and systematic TCM knowledge through a progressive learning manner. Additionally, we establish TCMEval, a comprehensive evaluation benchmark, to assess LLMs in TCM examinations, clinical tasks, domain-specific question-answering, and real-world trials. The extensive evaluations demonstrate the significant potential of Tianyi as an AI assistant in TCM clinical practice and research, bridging the gap between TCM knowledge and practical application. 

---
# What if Deception Cannot be Detected? A Cross-Linguistic Study on the Limits of Deception Detection from Text 

**Authors**: Aswathy Velutharambath, Roman Klinger, Kai Sassenberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.13147)  

**Abstract**: Can deception be detected solely from written text? Cues of deceptive communication are inherently subtle, even more so in text-only communication. Yet, prior studies have reported considerable success in automatic deception detection. We hypothesize that such findings are largely driven by artifacts introduced during data collection and do not generalize beyond specific datasets. We revisit this assumption by introducing a belief-based deception framework, which defines deception as a misalignment between an author's claims and true beliefs, irrespective of factual accuracy, allowing deception cues to be studied in isolation. Based on this framework, we construct three corpora, collectively referred to as DeFaBel, including a German-language corpus of deceptive and non-deceptive arguments and a multilingual version in German and English, each collected under varying conditions to account for belief change and enable cross-linguistic analysis. Using these corpora, we evaluate commonly reported linguistic cues of deception. Across all three DeFaBel variants, these cues show negligible, statistically insignificant correlations with deception labels, contrary to prior work that treats such cues as reliable indicators. We further benchmark against other English deception datasets following similar data collection protocols. While some show statistically significant correlations, effect sizes remain low and, critically, the set of predictive cues is inconsistent across datasets. We also evaluate deception detection using feature-based models, pretrained language models, and instruction-tuned large language models. While some models perform well on established deception datasets, they consistently perform near chance on DeFaBel. Our findings challenge the assumption that deception can be reliably inferred from linguistic cues and call for rethinking how deception is studied and modeled in NLP. 

---
# Understanding Cross-Lingual Inconsistency in Large Language Models 

**Authors**: Zheng Wei Lim, Alham Fikri Aji, Trevor Cohn  

**Link**: [PDF](https://arxiv.org/pdf/2505.13141)  

**Abstract**: Large language models (LLMs) are demonstrably capable of cross-lingual transfer, but can produce inconsistent output when prompted with the same queries written in different languages. To understand how language models are able to generalize knowledge from one language to the others, we apply the logit lens to interpret the implicit steps taken by LLMs to solve multilingual multi-choice reasoning questions. We find LLMs predict inconsistently and are less accurate because they rely on subspaces of individual languages, rather than working in a shared semantic space. While larger models are more multilingual, we show their hidden states are more likely to dissociate from the shared representation compared to smaller models, but are nevertheless more capable of retrieving knowledge embedded across different languages. Finally, we demonstrate that knowledge sharing can be modulated by steering the models' latent processing towards the shared semantic space. We find reinforcing utilization of the shared space improves the models' multilingual reasoning performance, as a result of more knowledge transfer from, and better output consistency with English. 

---
# ModernGBERT: German-only 1B Encoder Model Trained from Scratch 

**Authors**: Anton Ehrmanntraut, Julia Wunderle, Jan Pfister, Fotis Jannidis, Andreas Hotho  

**Link**: [PDF](https://arxiv.org/pdf/2505.13136)  

**Abstract**: Despite the prominence of decoder-only language models, encoders remain crucial for resource-constrained applications. We introduce ModernGBERT (134M, 1B), a fully transparent family of German encoder models trained from scratch, incorporating architectural innovations from ModernBERT. To evaluate the practical trade-offs of training encoders from scratch, we also present LLäMmlein2Vec (120M, 1B, 7B), a family of encoders derived from German decoder-only models via LLM2Vec. We benchmark all models on natural language understanding, text embedding, and long-context reasoning tasks, enabling a controlled comparison between dedicated encoders and converted decoders. Our results show that ModernGBERT 1B outperforms prior state-of-the-art German encoders as well as encoders adapted via LLM2Vec, with regard to performance and parameter-efficiency. All models, training data, checkpoints and code are publicly available, advancing the German NLP ecosystem with transparent, high-performance encoder models. 

---
# Benchmarking and Confidence Evaluation of LALMs For Temporal Reasoning 

**Authors**: Debarpan Bhattacharya, Apoorva Kulkarni, Sriram Ganapathy  

**Link**: [PDF](https://arxiv.org/pdf/2505.13115)  

**Abstract**: The popular success of text-based large language models (LLM) has streamlined the attention of the multimodal community to combine other modalities like vision and audio along with text to achieve similar multimodal capabilities. In this quest, large audio language models (LALMs) have to be evaluated on reasoning related tasks which are different from traditional classification or generation tasks. Towards this goal, we propose a novel dataset called temporal reasoning evaluation of audio (TREA).
We benchmark open-source LALMs and observe that they are consistently behind human capabilities on the tasks in the TREA dataset. While evaluating LALMs, we also propose an uncertainty metric, which computes the invariance of the model to semantically identical perturbations of the input. Our analysis shows that the accuracy and uncertainty metrics are not necessarily correlated and thus, points to a need for wholesome evaluation of LALMs for high-stakes applications. 

---
# The Effect of Language Diversity When Fine-Tuning Large Language Models for Translation 

**Authors**: David Stap, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2505.13090)  

**Abstract**: Prior research diverges on language diversity in LLM fine-tuning: Some studies report benefits while others find no advantages. Through controlled fine-tuning experiments across 132 translation directions, we systematically resolve these disparities. We find that expanding language diversity during fine-tuning improves translation quality for both unsupervised and -- surprisingly -- supervised pairs, despite less diverse models being fine-tuned exclusively on these supervised pairs. However, benefits plateau or decrease beyond a certain diversity threshold. We show that increased language diversity creates more language-agnostic representations. These representational adaptations help explain the improved performance in models fine-tuned with greater diversity. 

---
# Systematic Generalization in Language Models Scales with Information Entropy 

**Authors**: Sondre Wold, Lucas Georges Gabriel Charpentier, Étienne Simon  

**Link**: [PDF](https://arxiv.org/pdf/2505.13089)  

**Abstract**: Systematic generalization remains challenging for current language models, which are known to be both sensitive to semantically similar permutations of the input and to struggle with known concepts presented in novel contexts. Although benchmarks exist for assessing compositional behavior, it is unclear how to measure the difficulty of a systematic generalization problem. In this work, we show how one aspect of systematic generalization can be described by the entropy of the distribution of component parts in the training data. We formalize a framework for measuring entropy in a sequence-to-sequence task and find that the performance of popular model architectures scales with the entropy. Our work connects systematic generalization to information efficiency, and our results indicate that success at high entropy can be achieved even without built-in priors, and that success at low entropy can serve as a target for assessing progress towards robust systematic generalization. 

---
# Advancing Sequential Numerical Prediction in Autoregressive Models 

**Authors**: Xiang Fei, Jinghui Lu, Qi Sun, Hao Feng, Yanjie Wang, Wei Shi, An-Lan Wang, Jingqun Tang, Can Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13077)  

**Abstract**: Autoregressive models have become the de facto choice for sequence generation tasks, but standard approaches treat digits as independent tokens and apply cross-entropy loss, overlooking the coherent structure of numerical sequences. This paper introduces Numerical Token Integrity Loss (NTIL) to address this gap. NTIL operates at two levels: (1) token-level, where it extends the Earth Mover's Distance (EMD) to preserve ordinal relationships between numerical values, and (2) sequence-level, where it penalizes the overall discrepancy between the predicted and actual sequences. This dual approach improves numerical prediction and integrates effectively with LLMs/MLLMs. Extensive experiments show significant performance improvements with NTIL. 

---
# Suicide Risk Assessment Using Multimodal Speech Features: A Study on the SW1 Challenge Dataset 

**Authors**: Ambre Marie, Ilias Maoudj, Guillaume Dardenne, Gwenolé Quellec  

**Link**: [PDF](https://arxiv.org/pdf/2505.13069)  

**Abstract**: The 1st SpeechWellness Challenge conveys the need for speech-based suicide risk assessment in adolescents. This study investigates a multimodal approach for this challenge, integrating automatic transcription with WhisperX, linguistic embeddings from Chinese RoBERTa, and audio embeddings from WavLM. Additionally, handcrafted acoustic features -- including MFCCs, spectral contrast, and pitch-related statistics -- were incorporated. We explored three fusion strategies: early concatenation, modality-specific processing, and weighted attention with mixup regularization. Results show that weighted attention provided the best generalization, achieving 69% accuracy on the development set, though a performance gap between development and test sets highlights generalization challenges. Our findings, strictly tied to the MINI-KID framework, emphasize the importance of refining embedding representations and fusion mechanisms to enhance classification reliability. 

---
# SNAPE-PM: Building and Utilizing Dynamic Partner Models for Adaptive Explanation Generation 

**Authors**: Amelie S. Robrecht, Christoph R. Kowalski, Stefan Kopp  

**Link**: [PDF](https://arxiv.org/pdf/2505.13053)  

**Abstract**: Adapting to the addressee is crucial for successful explanations, yet poses significant challenges for dialogsystems. We adopt the approach of treating explanation generation as a non-stationary decision process, where the optimal strategy varies according to changing beliefs about the explainee and the interaction context. In this paper we address the questions of (1) how to track the interaction context and the relevant listener features in a formally defined computational partner model, and (2) how to utilize this model in the dynamically adjusted, rational decision process that determines the currently best explanation strategy. We propose a Bayesian inference-based approach to continuously update the partner model based on user feedback, and a non-stationary Markov Decision Process to adjust decision-making based on the partner model values. We evaluate an implementation of this framework with five simulated interlocutors, demonstrating its effectiveness in adapting to different partners with constant and even changing feedback behavior. The results show high adaptivity with distinct explanation strategies emerging for different partners, highlighting the potential of our approach to improve explainable AI systems and dialogsystems in general. 

---
# KIT's Offline Speech Translation and Instruction Following Submission for IWSLT 2025 

**Authors**: Sai Koneru, Maike Züfle, Thai-Binh Nguyen, Seymanur Akti, Jan Niehues, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2505.13036)  

**Abstract**: The scope of the International Workshop on Spoken Language Translation (IWSLT) has recently broadened beyond traditional Speech Translation (ST) to encompass a wider array of tasks, including Speech Question Answering and Summarization. This shift is partly driven by the growing capabilities of modern systems, particularly with the success of Large Language Models (LLMs). In this paper, we present the Karlsruhe Institute of Technology's submissions for the Offline ST and Instruction Following (IF) tracks, where we leverage LLMs to enhance performance across all tasks. For the Offline ST track, we propose a pipeline that employs multiple automatic speech recognition systems, whose outputs are fused using an LLM with document-level context. This is followed by a two-step translation process, incorporating additional refinement step to improve translation quality. For the IF track, we develop an end-to-end model that integrates a speech encoder with an LLM to perform a wide range of instruction-following tasks. We complement it with a final document-level refinement stage to further enhance output quality by using contextual information. 

---
# topicwizard -- a Modern, Model-agnostic Framework for Topic Model Visualization and Interpretation 

**Authors**: Márton Kardos, Kenneth C. Enevoldsen, Kristoffer Laigaard Nielbo  

**Link**: [PDF](https://arxiv.org/pdf/2505.13034)  

**Abstract**: Topic models are statistical tools that allow their users to gain qualitative and quantitative insights into the contents of textual corpora without the need for close reading. They can be applied in a wide range of settings from discourse analysis, through pretraining data curation, to text filtering. Topic models are typically parameter-rich, complex models, and interpreting these parameters can be challenging for their users. It is typical practice for users to interpret topics based on the top 10 highest ranking terms on a given topic. This list-of-words approach, however, gives users a limited and biased picture of the content of topics. Thoughtful user interface design and visualizations can help users gain a more complete and accurate understanding of topic models' output. While some visualization utilities do exist for topic models, these are typically limited to a certain type of topic model. We introduce topicwizard, a framework for model-agnostic topic model interpretation, that provides intuitive and interactive tools that help users examine the complex semantic relations between documents, words and topics learned by topic models. 

---
# To Bias or Not to Bias: Detecting bias in News with bias-detector 

**Authors**: Himel Ghosh, Ahmed Mosharafa, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2505.13010)  

**Abstract**: Media bias detection is a critical task in ensuring fair and balanced information dissemination, yet it remains challenging due to the subjectivity of bias and the scarcity of high-quality annotated data. In this work, we perform sentence-level bias classification by fine-tuning a RoBERTa-based model on the expert-annotated BABE dataset. Using McNemar's test and the 5x2 cross-validation paired t-test, we show statistically significant improvements in performance when comparing our model to a domain-adaptively pre-trained DA-RoBERTa baseline. Furthermore, attention-based analysis shows that our model avoids common pitfalls like oversensitivity to politically charged terms and instead attends more meaningfully to contextually relevant tokens. For a comprehensive examination of media bias, we present a pipeline that combines our model with an already-existing bias-type classifier. Our method exhibits good generalization and interpretability, despite being constrained by sentence-level analysis and dataset size because of a lack of larger and more advanced bias corpora. We talk about context-aware modeling, bias neutralization, and advanced bias type classification as potential future directions. Our findings contribute to building more robust, explainable, and socially responsible NLP systems for media bias detection. 

---
# Evaluating the Performance of RAG Methods for Conversational AI in the Airport Domain 

**Authors**: Yuyang Li, Philip J.M. Kerbusch, Raimon H.R. Pruim, Tobias Käfer  

**Link**: [PDF](https://arxiv.org/pdf/2505.13006)  

**Abstract**: Airports from the top 20 in terms of annual passengers are highly dynamic environments with thousands of flights daily, and they aim to increase the degree of automation. To contribute to this, we implemented a Conversational AI system that enables staff in an airport to communicate with flight information systems. This system not only answers standard airport queries but also resolves airport terminology, jargon, abbreviations, and dynamic questions involving reasoning. In this paper, we built three different Retrieval-Augmented Generation (RAG) methods, including traditional RAG, SQL RAG, and Knowledge Graph-based RAG (Graph RAG). Experiments showed that traditional RAG achieved 84.84% accuracy using BM25 + GPT-4 but occasionally produced hallucinations, which is risky to airport safety. In contrast, SQL RAG and Graph RAG achieved 80.85% and 91.49% accuracy respectively, with significantly fewer hallucinations. Moreover, Graph RAG was especially effective for questions that involved reasoning. Based on our observations, we thus recommend SQL RAG and Graph RAG are better for airport environments, due to fewer hallucinations and the ability to handle dynamic questions. 

---
# EffiBench-X: A Multi-Language Benchmark for Measuring Efficiency of LLM-Generated Code 

**Authors**: Yuhao Qing, Boyu Zhu, Mingzhe Du, Zhijiang Guo, Terry Yue Zhuo, Qianru Zhang, Jie M. Zhang, Heming Cui, Siu-Ming Yiu, Dong Huang, See-Kiong Ng, Luu Anh Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13004)  

**Abstract**: Existing code generation benchmarks primarily evaluate functional correctness, with limited focus on code efficiency and often restricted to a single language like Python. To address this gap, we introduce EffiBench-X, the first multi-language benchmark designed to measure the efficiency of LLM-generated code. EffiBench-X supports Python, C++, Java, JavaScript, Ruby, and Golang. It comprises competitive programming tasks with human-expert solutions as efficiency baselines. Evaluating state-of-the-art LLMs on EffiBench-X reveals that while models generate functionally correct code, they consistently underperform human experts in efficiency. Even the most efficient LLM-generated solutions (Qwen3-32B) achieve only around \textbf{62\%} of human efficiency on average, with significant language-specific variations. LLMs show better efficiency in Python, Ruby, and JavaScript than in Java, C++, and Golang. For instance, DeepSeek-R1's Python code is significantly more efficient than its Java code. These results highlight the critical need for research into LLM optimization techniques to improve code efficiency across diverse languages. The dataset and evaluation infrastructure are submitted and available at this https URL and this https URL. 

---
# ExTrans: Multilingual Deep Reasoning Translation via Exemplar-Enhanced Reinforcement Learning 

**Authors**: Jiaan Wang, Fandong Meng, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12996)  

**Abstract**: In recent years, the emergence of large reasoning models (LRMs), such as OpenAI-o1 and DeepSeek-R1, has shown impressive capabilities in complex problems, e.g., mathematics and coding. Some pioneering studies attempt to bring the success of LRMs in neural machine translation (MT). They try to build LRMs with deep reasoning MT ability via reinforcement learning (RL). Despite some progress that has been made, these attempts generally focus on several high-resource languages, e.g., English and Chinese, leaving the performance on other languages unclear. Besides, the reward modeling methods in previous work do not fully unleash the potential of reinforcement learning in MT. In this work, we first design a new reward modeling method that compares the translation results of the policy MT model with a strong LRM (i.e., DeepSeek-R1-671B), and quantifies the comparisons to provide rewards. Experimental results demonstrate the superiority of the reward modeling method. Using Qwen2.5-7B-Instruct as the backbone, the trained model achieves the new state-of-the-art performance in literary translation, and outperforms strong LRMs including OpenAI-o1 and DeepSeeK-R1. Furthermore, we extend our method to the multilingual settings with 11 languages. With a carefully designed lightweight reward modeling in RL, we can simply transfer the strong MT ability from a single direction into multiple (i.e., 90) translation directions and achieve impressive multilingual MT performance. 

---
# An Empirical Study of Many-to-Many Summarization with Large Language Models 

**Authors**: Jiaan Wang, Fandong Meng, Zengkui Sun, Yunlong Liang, Yuxuan Cao, Jiarong Xu, Haoxiang Shi, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12983)  

**Abstract**: Many-to-many summarization (M2MS) aims to process documents in any language and generate the corresponding summaries also in any language. Recently, large language models (LLMs) have shown strong multi-lingual abilities, giving them the potential to perform M2MS in real applications. This work presents a systematic empirical study on LLMs' M2MS ability. Specifically, we first reorganize M2MS data based on eight previous domain-specific datasets. The reorganized data contains 47.8K samples spanning five domains and six languages, which could be used to train and evaluate LLMs. Then, we benchmark 18 LLMs in a zero-shot manner and an instruction-tuning manner. Fine-tuned traditional models (e.g., mBART) are also conducted for comparisons. Our experiments reveal that, zero-shot LLMs achieve competitive results with fine-tuned traditional models. After instruct-tuning, open-source LLMs can significantly improve their M2MS ability, and outperform zero-shot LLMs (including GPT-4) in terms of automatic evaluations. In addition, we demonstrate that this task-specific improvement does not sacrifice the LLMs' general task-solving abilities. However, as revealed by our human evaluation, LLMs still face the factuality issue, and the instruction tuning might intensify the issue. Thus, how to control factual errors becomes the key when building LLM summarizers in real applications, and is worth noting in future research. 

---
# Fast, Not Fancy: Rethinking G2P with Rich Data and Rule-Based Models 

**Authors**: Mahta Fetrat Qharabagh, Zahra Dehghanian, Hamid R. Rabiee  

**Link**: [PDF](https://arxiv.org/pdf/2505.12973)  

**Abstract**: Homograph disambiguation remains a significant challenge in grapheme-to-phoneme (G2P) conversion, especially for low-resource languages. This challenge is twofold: (1) creating balanced and comprehensive homograph datasets is labor-intensive and costly, and (2) specific disambiguation strategies introduce additional latency, making them unsuitable for real-time applications such as screen readers and other accessibility tools. In this paper, we address both issues. First, we propose a semi-automated pipeline for constructing homograph-focused datasets, introduce the HomoRich dataset generated through this pipeline, and demonstrate its effectiveness by applying it to enhance a state-of-the-art deep learning-based G2P system for Persian. Second, we advocate for a paradigm shift - utilizing rich offline datasets to inform the development of fast, rule-based methods suitable for latency-sensitive accessibility applications like screen readers. To this end, we improve one of the most well-known rule-based G2P systems, eSpeak, into a fast homograph-aware version, HomoFast eSpeak. Our results show an approximate 30% improvement in homograph disambiguation accuracy for the deep learning-based and eSpeak systems. 

---
# A Structured Literature Review on Traditional Approaches in Current Natural Language Processing 

**Authors**: Robin Jegan, Andreas Henrich  

**Link**: [PDF](https://arxiv.org/pdf/2505.12970)  

**Abstract**: The continued rise of neural networks and large language models in the more recent past has altered the natural language processing landscape, enabling new approaches towards typical language tasks and achieving mainstream success. Despite the huge success of large language models, many disadvantages still remain and through this work we assess the state of the art in five application scenarios with a particular focus on the future perspectives and sensible application scenarios of traditional and older approaches and techniques.
In this paper we survey recent publications in the application scenarios classification, information and relation extraction, text simplification as well as text summarization. After defining our terminology, i.e., which features are characteristic for traditional techniques in our interpretation for the five scenarios, we survey if such traditional approaches are still being used, and if so, in what way they are used. It turns out that all five application scenarios still exhibit traditional models in one way or another, as part of a processing pipeline, as a comparison/baseline to the core model of the respective paper, or as the main model(s) of the paper. For the complete statistics, see this https URL 

---
# Calm-Whisper: Reduce Whisper Hallucination On Non-Speech By Calming Crazy Heads Down 

**Authors**: Yingzhi Wang, Anas Alhmoud, Saad Alsahly, Muhammad Alqurishi, Mirco Ravanelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.12969)  

**Abstract**: OpenAI's Whisper has achieved significant success in Automatic Speech Recognition. However, it has consistently been found to exhibit hallucination issues, particularly in non-speech segments, which limits its broader application in complex industrial settings.
In this paper, we introduce a novel method to reduce Whisper's hallucination on non-speech segments without using any pre- or post-possessing techniques. Specifically, we benchmark the contribution of each self-attentional head in the Whisper-large-v3 decoder to the hallucination problem by performing a head-wise mask. Our findings reveal that only 3 of the 20 heads account for over 75% of the hallucinations on the UrbanSound dataset. We then fine-tune these three crazy heads using a collection of non-speech data. The results show that our best fine-tuned model, namely Calm-Whisper, achieves over 80% reduction in non-speech hallucination with only less than 0.1% WER degradation on LibriSpeech test-clean and test-other. 

---
# MA-COIR: Leveraging Semantic Search Index and Generative Models for Ontology-Driven Biomedical Concept Recognition 

**Authors**: Shanshan Liu, Noriki Nishida, Rumana Ferdous Munne, Narumi Tokunaga, Yuki Yamagata, Kouji Kozaki, Yuji Matsumoto  

**Link**: [PDF](https://arxiv.org/pdf/2505.12964)  

**Abstract**: Recognizing biomedical concepts in the text is vital for ontology refinement, knowledge graph construction, and concept relationship discovery. However, traditional concept recognition methods, relying on explicit mention identification, often fail to capture complex concepts not explicitly stated in the text. To overcome this limitation, we introduce MA-COIR, a framework that reformulates concept recognition as an indexing-recognition task. By assigning semantic search indexes (ssIDs) to concepts, MA-COIR resolves ambiguities in ontology entries and enhances recognition efficiency. Using a pretrained BART-based model fine-tuned on small datasets, our approach reduces computational requirements to facilitate adoption by domain experts. Furthermore, we incorporate large language models (LLMs)-generated queries and synthetic data to improve recognition in low-resource settings. Experimental results on three scenarios (CDR, HPO, and HOIP) highlight the effectiveness of MA-COIR in recognizing both explicit and implicit concepts without the need for mention-level annotations during inference, advancing ontology-driven concept recognition in biomedical domain applications. Our code and constructed data are available at this https URL. 

---
# GuRE:Generative Query REwriter for Legal Passage Retrieval 

**Authors**: Daehee Kim, Deokhyung Kang, Jonghwi Kim, Sangwon Ryu, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.12950)  

**Abstract**: Legal Passage Retrieval (LPR) systems are crucial as they help practitioners save time when drafting legal arguments. However, it remains an underexplored avenue. One primary reason is the significant vocabulary mismatch between the query and the target passage. To address this, we propose a simple yet effective method, the Generative query REwriter (GuRE). We leverage the generative capabilities of Large Language Models (LLMs) by training the LLM for query rewriting. "Rewritten queries" help retrievers to retrieve target passages by mitigating vocabulary mismatch. Experimental results show that GuRE significantly improves performance in a retriever-agnostic manner, outperforming all baseline methods. Further analysis reveals that different training objectives lead to distinct retrieval behaviors, making GuRE more suitable than direct retriever fine-tuning for real-world applications. Codes are avaiable at this http URL. 

---
# Neural Morphological Tagging for Nguni Languages 

**Authors**: Cael Marquard, Simbarashe Mawere, Francois Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2505.12949)  

**Abstract**: Morphological parsing is the task of decomposing words into morphemes, the smallest units of meaning in a language, and labelling their grammatical roles. It is a particularly challenging task for agglutinative languages, such as the Nguni languages of South Africa, which construct words by concatenating multiple morphemes. A morphological parsing system can be framed as a pipeline with two separate components, a segmenter followed by a tagger. This paper investigates the use of neural methods to build morphological taggers for the four Nguni languages. We compare two classes of approaches: training neural sequence labellers (LSTMs and neural CRFs) from scratch and finetuning pretrained language models. We compare performance across these two categories, as well as to a traditional rule-based morphological parser. Neural taggers comfortably outperform the rule-based baseline and models trained from scratch tend to outperform pretrained models. We also compare parsing results across different upstream segmenters and with varying linguistic input features. Our findings confirm the viability of employing neural taggers based on pre-existing morphological segmenters for the Nguni languages. 

---
# A3 : an Analytical Low-Rank Approximation Framework for Attention 

**Authors**: Jeffrey T. H. Wong, Cheng Zhang, Xinye Cao, Pedro Gimenes, George A. Constantinides, Wayne Luk, Yiren Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12942)  

**Abstract**: Large language models have demonstrated remarkable performance; however, their massive parameter counts make deployment highly expensive. Low-rank approximation offers a promising compression solution, yet existing approaches have two main limitations: (1) They focus on minimizing the output error of individual linear layers, without considering the architectural characteristics of Transformers, and (2) they decompose a large weight matrix into two small low-rank matrices. Consequently, these methods often fall short compared to other compression techniques like pruning and quantization, and introduce runtime overhead such as the extra GEMM kernel launches for decomposed small matrices. To address these limitations, we propose $\tt A^\tt 3$, a post-training low-rank approximation framework. $\tt A^\tt 3$ splits a Transformer layer into three functional components, namely $\tt QK$, $\tt OV$, and $\tt MLP$. For each component, $\tt A^\tt 3$ provides an analytical solution that reduces the hidden dimension size inside each component while minimizing the component's functional loss ($\it i.e.$, error in attention scores, attention outputs, and MLP outputs). This approach directly reduces model sizes, KV cache sizes, and FLOPs without introducing any runtime overheads. In addition, it provides a new narrative in advancing the optimization problem from singular linear layer loss optimization toward improved end-to-end performance. Through extensive experiments, we show that $\tt A^\tt 3$ maintains superior performance compared to SoTAs. For example, under the same reduction budget in computation and memory, our low-rank approximated LLaMA 3.1-70B achieves a perplexity of 4.69 on WikiText-2, outperforming the previous SoTA's 7.87 by 3.18. We also demonstrate the versatility of $\tt A^\tt 3$, including KV cache compression, quantization, and mixed-rank assignments for enhanced performance. 

---
# Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs 

**Authors**: Zhihe Yang, Xufang Luo, Zilong Wang, Dongqi Han, Zhiyuan He, Dongsheng Li, Yunjian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12929)  

**Abstract**: Reinforcement learning (RL) has become a cornerstone for enhancing the reasoning capabilities of large language models (LLMs), with recent innovations such as Group Relative Policy Optimization (GRPO) demonstrating exceptional effectiveness. In this study, we identify a critical yet underexplored issue in RL training: low-probability tokens disproportionately influence model updates due to their large gradient magnitudes. This dominance hinders the effective learning of high-probability tokens, whose gradients are essential for LLMs' performance but are substantially suppressed. To mitigate this interference, we propose two novel methods: Advantage Reweighting and Low-Probability Token Isolation (Lopti), both of which effectively attenuate gradients from low-probability tokens while emphasizing parameter updates driven by high-probability tokens. Our approaches promote balanced updates across tokens with varying probabilities, thereby enhancing the efficiency of RL training. Experimental results demonstrate that they substantially improve the performance of GRPO-trained LLMs, achieving up to a 46.2% improvement in K&K Logic Puzzle reasoning tasks. Our implementation is available at this https URL. 

---
# PyFCG: Fluid Construction Grammar in Python 

**Authors**: Paul Van Eecke, Katrien Beuls  

**Link**: [PDF](https://arxiv.org/pdf/2505.12920)  

**Abstract**: We present PyFCG, an open source software library that ports Fluid Construction Grammar (FCG) to the Python programming language. PyFCG enables its users to seamlessly integrate FCG functionality into Python programs, and to use FCG in combination with other libraries within Python's rich ecosystem. Apart from a general description of the library, this paper provides three walkthrough tutorials that demonstrate example usage of PyFCG in typical use cases of FCG: (i) formalising and testing construction grammar analyses, (ii) learning usage-based construction grammars from corpora, and (iii) implementing agent-based experiments on emergent communication. 

---
# On the Thinking-Language Modeling Gap in Large Language Models 

**Authors**: Chenxi Liu, Yongqiang Chen, Tongliang Liu, James Cheng, Bo Han, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12896)  

**Abstract**: System 2 reasoning is one of the defining characteristics of intelligence, which requires slow and logical thinking. Human conducts System 2 reasoning via the language of thoughts that organizes the reasoning process as a causal sequence of mental language, or thoughts. Recently, it has been observed that System 2 reasoning can be elicited from Large Language Models (LLMs) pre-trained on large-scale natural languages. However, in this work, we show that there is a significant gap between the modeling of languages and thoughts. As language is primarily a tool for humans to share knowledge and thinking, modeling human language can easily absorb language biases into LLMs deviated from the chain of thoughts in minds. Furthermore, we show that the biases will mislead the eliciting of "thoughts" in LLMs to focus only on a biased part of the premise. To this end, we propose a new prompt technique termed Language-of-Thoughts (LoT) to demonstrate and alleviate this gap. Instead of directly eliciting the chain of thoughts from partial information, LoT instructs LLMs to adjust the order and token used for the expressions of all the relevant information. We show that the simple strategy significantly reduces the language modeling biases in LLMs and improves the performance of LLMs across a variety of reasoning tasks. 

---
# GAP: Graph-Assisted Prompts for Dialogue-based Medication Recommendation 

**Authors**: Jialun Zhong, Yanzeng Li, Sen Hu, Yang Zhang, Teng Xu, Lei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12888)  

**Abstract**: Medication recommendations have become an important task in the healthcare domain, especially in measuring the accuracy and safety of medical dialogue systems (MDS). Different from the recommendation task based on electronic health records (EHRs), dialogue-based medication recommendations require research on the interaction details between patients and doctors, which is crucial but may not exist in EHRs. Recent advancements in large language models (LLM) have extended the medical dialogue domain. These LLMs can interpret patients' intent and provide medical suggestions including medication recommendations, but some challenges are still worth attention. During a multi-turn dialogue, LLMs may ignore the fine-grained medical information or connections across the dialogue turns, which is vital for providing accurate suggestions. Besides, LLMs may generate non-factual responses when there is a lack of domain-specific knowledge, which is more risky in the medical domain. To address these challenges, we propose a \textbf{G}raph-\textbf{A}ssisted \textbf{P}rompts (\textbf{GAP}) framework for dialogue-based medication recommendation. It extracts medical concepts and corresponding states from dialogue to construct an explicitly patient-centric graph, which can describe the neglected but important information. Further, combined with external medical knowledge graphs, GAP can generate abundant queries and prompts, thus retrieving information from multiple sources to reduce the non-factual responses. We evaluate GAP on a dialogue-based medication recommendation dataset and further explore its potential in a more difficult scenario, dynamically diagnostic interviewing. Extensive experiments demonstrate its competitive performance when compared with strong baselines. 

---
# LEXam: Benchmarking Legal Reasoning on 340 Law Exams 

**Authors**: Yu Fan, Jingwei Ni, Jakob Merane, Etienne Salimbeni, Yang Tian, Yoan Hermstrüwer, Yinya Huang, Mubashara Akhtar, Florian Geering, Oliver Dreyer, Daniel Brunner, Markus Leippold, Mrinmaya Sachan, Alexander Stremitzer, Christoph Engel, Elliott Ash, Joel Niklaus  

**Link**: [PDF](https://arxiv.org/pdf/2505.12864)  

**Abstract**: Long-form legal reasoning remains a key challenge for large language models (LLMs) in spite of recent advances in test-time scaling. We introduce LEXam, a novel benchmark derived from 340 law exams spanning 116 law school courses across a range of subjects and degree levels. The dataset comprises 4,886 law exam questions in English and German, including 2,841 long-form, open-ended questions and 2,045 multiple-choice questions. Besides reference answers, the open questions are also accompanied by explicit guidance outlining the expected legal reasoning approach such as issue spotting, rule recall, or rule application. Our evaluation on both open-ended and multiple-choice questions present significant challenges for current LLMs; in particular, they notably struggle with open questions that require structured, multi-step legal reasoning. Moreover, our results underscore the effectiveness of the dataset in differentiating between models with varying capabilities. Adopting an LLM-as-a-Judge paradigm with rigorous human expert validation, we demonstrate how model-generated reasoning steps can be evaluated consistently and accurately. Our evaluation setup provides a scalable method to assess legal reasoning quality beyond simple accuracy metrics. Project page: this https URL 

---
# Re-identification of De-identified Documents with Autoregressive Infilling 

**Authors**: Lucas Georges Gabriel Charpentier, Pierre Lison  

**Link**: [PDF](https://arxiv.org/pdf/2505.12859)  

**Abstract**: Documents revealing sensitive information about individuals must typically be de-identified. This de-identification is often done by masking all mentions of personally identifiable information (PII), thereby making it more difficult to uncover the identity of the person(s) in question. To investigate the robustness of de-identification methods, we present a novel, RAG-inspired approach that attempts the reverse process of re-identification based on a database of documents representing background knowledge. Given a text in which personal identifiers have been masked, the re-identification proceeds in two steps. A retriever first selects from the background knowledge passages deemed relevant for the re-identification. Those passages are then provided to an infilling model which seeks to infer the original content of each text span. This process is repeated until all masked spans are replaced. We evaluate the re-identification on three datasets (Wikipedia biographies, court rulings and clinical notes). Results show that (1) as many as 80% of de-identified text spans can be successfully recovered and (2) the re-identification accuracy increases along with the level of background knowledge. 

---
# The Hidden Structure -- Improving Legal Document Understanding Through Explicit Text Formatting 

**Authors**: Christian Braun, Alexander Lilienbeck, Daniel Mentjukov  

**Link**: [PDF](https://arxiv.org/pdf/2505.12837)  

**Abstract**: Legal contracts possess an inherent, semantically vital structure (e.g., sections, clauses) that is crucial for human comprehension but whose impact on LLM processing remains under-explored. This paper investigates the effects of explicit input text structure and prompt engineering on the performance of GPT-4o and GPT-4.1 on a legal question-answering task using an excerpt of the CUAD. We compare model exact-match accuracy across various input formats: well-structured plain-text (human-generated from CUAD), plain-text cleaned of line breaks, extracted plain-text from Azure OCR, plain-text extracted by GPT-4o Vision, and extracted (and interpreted) Markdown (MD) from GPT-4o Vision. To give an indication of the impact of possible prompt engineering, we assess the impact of shifting task instructions to the system prompt and explicitly informing the model about the structured nature of the input. Our findings reveal that GPT-4o demonstrates considerable robustness to variations in input structure, but lacks in overall performance. Conversely, GPT-4.1's performance is markedly sensitive; poorly structured inputs yield suboptimal results (but identical with GPT-4o), while well-structured formats (original CUAD text, GPT-4o Vision text and GPT-4o MD) improve exact-match accuracy by ~20 percentage points. Optimizing the system prompt to include task details and an advisory about structured input further elevates GPT-4.1's accuracy by an additional ~10-13 percentage points, with Markdown ultimately achieving the highest performance under these conditions (79 percentage points overall exact-match accuracy). This research empirically demonstrates that while newer models exhibit greater resilience, careful input structuring and strategic prompt design remain critical for optimizing the performance of LLMs, and can significantly affect outcomes in high-stakes legal applications. 

---
# FlightGPT: Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with Vision-Language Models 

**Authors**: Hengxing Cai, Jinhan Dong, Jingjun Tan, Jingcheng Deng, Sihang Li, Zhifeng Gao, Haidong Wang, Zicheng Su, Agachai Sumalee, Renxin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12835)  

**Abstract**: Unmanned Aerial Vehicle (UAV) Vision-and-Language Navigation (VLN) is vital for applications such as disaster response, logistics delivery, and urban inspection. However, existing methods often struggle with insufficient multimodal fusion, weak generalization, and poor interpretability. To address these challenges, we propose FlightGPT, a novel UAV VLN framework built upon Vision-Language Models (VLMs) with powerful multimodal perception capabilities. We design a two-stage training pipeline: first, Supervised Fine-Tuning (SFT) using high-quality demonstrations to improve initialization and structured reasoning; then, Group Relative Policy Optimization (GRPO) algorithm, guided by a composite reward that considers goal accuracy, reasoning quality, and format compliance, to enhance generalization and adaptability. Furthermore, FlightGPT introduces a Chain-of-Thought (CoT)-based reasoning mechanism to improve decision interpretability. Extensive experiments on the city-scale dataset CityNav demonstrate that FlightGPT achieves state-of-the-art performance across all scenarios, with a 9.22\% higher success rate than the strongest baseline in unseen environments. Our implementation is publicly available. 

---
# Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering 

**Authors**: Zifeng Cheng, Zhonghui Wang, Yuchen Fu, Zhiwei Jiang, Yafeng Yin, Cong Wang, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12831)  

**Abstract**: Extracting sentence embeddings from large language models (LLMs) is a practical direction, as it requires neither additional data nor fine-tuning. Previous studies usually focus on prompt engineering to guide LLMs to encode the core semantic information of the sentence into the embedding of the last token. However, the last token in these methods still encodes an excess of non-essential information, such as stop words, limiting its encoding capacity. To this end, we propose a Contrastive Prompting (CP) method that introduces an extra auxiliary prompt to elicit better sentence embedding. By contrasting with the auxiliary prompt, CP can steer existing prompts to encode the core semantics of the sentence, rather than non-essential information. CP is a plug-and-play inference-time intervention method that can be combined with various prompt-based methods. Extensive experiments on Semantic Textual Similarity (STS) tasks and downstream classification tasks demonstrate that our method can improve the performance of existing prompt-based methods across different LLMs. Our code will be released at this https URL. 

---
# SynDec: A Synthesize-then-Decode Approach for Arbitrary Textual Style Transfer via Large Language Models 

**Authors**: Han Sun, Zhen Sun, Zongmin Zhang, Linzhao Jia, Wei Shao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12821)  

**Abstract**: Large Language Models (LLMs) are emerging as dominant forces for textual style transfer. However, for arbitrary style transfer, LLMs face two key challenges: (1) considerable reliance on manually-constructed prompts and (2) rigid stylistic biases inherent in LLMs. In this paper, we propose a novel Synthesize-then-Decode (SynDec) approach, which automatically synthesizes high-quality prompts and amplifies their roles during decoding process. Specifically, our approach synthesizes prompts by selecting representative few-shot samples, conducting a four-dimensional style analysis, and reranking the candidates. At LLM decoding stage, the TST effect is amplified by maximizing the contrast in output probabilities between scenarios with and without the synthesized prompt, as well as between prompts and negative samples. We conduct extensive experiments and the results show that SynDec outperforms existing state-of-the-art LLM-based methods on five out of six benchmarks (e.g., achieving up to a 9\% increase in accuracy for modern-to-Elizabethan English transfer). Detailed ablation studies further validate the effectiveness of SynDec. 

---
# PsyMem: Fine-grained psychological alignment and Explicit Memory Control for Advanced Role-Playing LLMs 

**Authors**: Xilong Cheng, Yunxiao Qin, Yuting Tan, Zhengnan Li, Ye Wang, Hongjiang Xiao, Yuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12814)  

**Abstract**: Existing LLM-based role-playing methods often rely on superficial textual descriptions or simplistic metrics, inadequately modeling both intrinsic and extrinsic character dimensions. Additionally, they typically simulate character memory with implicit model knowledge or basic retrieval augment generation without explicit memory alignment, compromising memory consistency. The two issues weaken reliability of role-playing LLMs in several applications, such as trustworthy social simulation. To address these limitations, we propose PsyMem, a novel framework integrating fine-grained psychological attributes and explicit memory control for role-playing. PsyMem supplements textual descriptions with 26 psychological indicators to detailed model character. Additionally, PsyMem implements memory alignment training, explicitly trains the model to align character's response with memory, thereby enabling dynamic memory-controlled responding during inference. By training Qwen2.5-7B-Instruct on our specially designed dataset (including 5,414 characters and 38,962 dialogues extracted from novels), the resulting model, termed as PsyMem-Qwen, outperforms baseline models in role-playing, achieving the best performance in human-likeness and character fidelity. 

---
# Decentralized Arena: Towards Democratic and Scalable Automatic Evaluation of Language Models 

**Authors**: Yanbin Yin, Kun Zhou, Zhen Wang, Xiangdong Zhang, Yifei Shao, Shibo Hao, Yi Gu, Jieyuan Liu, Somanshu Singla, Tianyang Liu, Eric P. Xing, Zhengzhong Liu, Haojian Jin, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12808)  

**Abstract**: The recent explosion of large language models (LLMs), each with its own general or specialized strengths, makes scalable, reliable benchmarking more urgent than ever. Standard practices nowadays face fundamental trade-offs: closed-ended question-based benchmarks (eg MMLU) struggle with saturation as newer models emerge, while crowd-sourced leaderboards (eg Chatbot Arena) rely on costly and slow human judges. Recently, automated methods (eg LLM-as-a-judge) shed light on the scalability, but risk bias by relying on one or a few "authority" models. To tackle these issues, we propose Decentralized Arena (dearena), a fully automated framework leveraging collective intelligence from all LLMs to evaluate each other. It mitigates single-model judge bias by democratic, pairwise evaluation, and remains efficient at scale through two key components: (1) a coarse-to-fine ranking algorithm for fast incremental insertion of new models with sub-quadratic complexity, and (2) an automatic question selection strategy for the construction of new evaluation dimensions. Across extensive experiments across 66 LLMs, dearena attains up to 97% correlation with human judgements, while significantly reducing the cost. Our code and data will be publicly released on this https URL. 

---
# EAVIT: Efficient and Accurate Human Value Identification from Text data via LLMs 

**Authors**: Wenhao Zhu, Yuhang Xie, Guojie Song, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12792)  

**Abstract**: The rapid evolution of large language models (LLMs) has revolutionized various fields, including the identification and discovery of human values within text data. While traditional NLP models, such as BERT, have been employed for this task, their ability to represent textual data is significantly outperformed by emerging LLMs like GPTs. However, the performance of online LLMs often degrades when handling long contexts required for value identification, which also incurs substantial computational costs. To address these challenges, we propose EAVIT, an efficient and accurate framework for human value identification that combines the strengths of both locally fine-tunable and online black-box LLMs. Our framework employs a value detector - a small, local language model - to generate initial value estimations. These estimations are then used to construct concise input prompts for online LLMs, enabling accurate final value identification. To train the value detector, we introduce explanation-based training and data generation techniques specifically tailored for value identification, alongside sampling strategies to optimize the brevity of LLM input prompts. Our approach effectively reduces the number of input tokens by up to 1/6 compared to directly querying online LLMs, while consistently outperforming traditional NLP methods and other LLM-based strategies. 

---
# A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone 

**Authors**: Jitai Hao, Qiang Huang, Hao Liu, Xinyan Xiao, Zhaochun Ren, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12781)  

**Abstract**: Training high-performing Small Language Models (SLMs) remains costly, even with knowledge distillation and pruning from larger teacher models. Existing work often faces three key challenges: (1) information loss from hard pruning, (2) inefficient alignment of representations, and (3) underutilization of informative activations, particularly from Feed-Forward Networks (FFNs). To address these challenges, we introduce Low-Rank Clone (LRC), an efficient pre-training method that constructs SLMs aspiring to behavioral equivalence with strong teacher models. LRC trains a set of low-rank projection matrices that jointly enable soft pruning by compressing teacher weights, and activation clone by aligning student activations, including FFN signals, with those of the teacher. This unified design maximizes knowledge transfer while removing the need for explicit alignment modules. Extensive experiments with open-source teachers (e.g., Llama-3.2-3B-Instruct, Qwen2.5-3B/7B-Instruct) show that LRC matches or surpasses state-of-the-art models trained on trillions of tokens--while using only 20B tokens, achieving over 1,000x training efficiency. Our codes and model checkpoints are available at this https URL and this https URL. 

---
# ReEx-SQL: Reasoning with Execution-Aware Reinforcement Learning for Text-to-SQL 

**Authors**: Yaxun Dai, Wenxuan Xie, Xialie Zhuang, Tianyu Yang, Yiying Yang, Haiqin Yang, Yuhang Zhao, Pingfu Chao, Wenhao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12768)  

**Abstract**: In Text-to-SQL, execution feedback is essential for guiding large language models (LLMs) to reason accurately and generate reliable SQL queries. However, existing methods treat execution feedback solely as a post-hoc signal for correction or selection, failing to integrate it into the generation process. This limitation hinders their ability to address reasoning errors as they occur, ultimately reducing query accuracy and robustness. To address this issue, we propose ReEx-SQL (Reasoning with Execution-Aware Reinforcement Learning), a framework for Text-to-SQL that enables models to interact with the database during decoding and dynamically adjust their reasoning based on execution feedback. ReEx-SQL introduces an execution-aware reasoning paradigm that interleaves intermediate SQL execution into reasoning paths, facilitating context-sensitive revisions. It achieves this through structured prompts with markup tags and a stepwise rollout strategy that integrates execution feedback into each stage of generation. To supervise policy learning, we develop a composite reward function that includes an exploration reward, explicitly encouraging effective database interaction. Additionally, ReEx-SQL adopts a tree-based decoding strategy to support exploratory reasoning, enabling dynamic expansion of alternative reasoning paths. Notably, ReEx-SQL achieves 88.8% on Spider and 64.9% on BIRD at the 7B scale, surpassing the standard reasoning baseline by 2.7% and 2.6%, respectively. It also shows robustness, achieving 85.2% on Spider-Realistic with leading performance. In addition, its tree-structured decoding improves efficiency and performance over linear decoding, reducing inference time by 51.9% on the BIRD development set. 

---
# What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma 

**Authors**: Han Meng, Yancan Chen, Yunan Li, Yitian Yang, Jungup Lee, Renwen Zhang, Yi-Chieh Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.12727)  

**Abstract**: Mental-health stigma remains a pervasive social problem that hampers treatment-seeking and recovery. Existing resources for training neural models to finely classify such stigma are limited, relying primarily on social-media or synthetic data without theoretical underpinnings. To remedy this gap, we present an expert-annotated, theory-informed corpus of human-chatbot interviews, comprising 4,141 snippets from 684 participants with documented socio-cultural backgrounds. Our experiments benchmark state-of-the-art neural models and empirically unpack the challenges of stigma detection. This dataset can facilitate research on computationally detecting, neutralizing, and counteracting mental-health stigma. 

---
# On-Policy Optimization with Group Equivalent Preference for Multi-Programming Language Understanding 

**Authors**: Haoyuan Wu, Rui Ming, Jilong Gao, Hangyu Zhao, Xueyi Chen, Yikai Yang, Haisheng Zheng, Zhuolun He, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12723)  

**Abstract**: Large language models (LLMs) achieve remarkable performance in code generation tasks. However, a significant performance disparity persists between popular programming languages (e.g., Python, C++) and others. To address this capability gap, we leverage the code translation task to train LLMs, thereby facilitating the transfer of coding proficiency across diverse programming languages. Moreover, we introduce OORL for training, a novel reinforcement learning (RL) framework that integrates on-policy and off-policy strategies. Within OORL, on-policy RL is applied during code translation, guided by a rule-based reward signal derived from unit tests. Complementing this coarse-grained rule-based reward, we propose Group Equivalent Preference Optimization (GEPO), a novel preference optimization method. Specifically, GEPO trains the LLM using intermediate representations (IRs) groups. LLMs can be guided to discern IRs equivalent to the source code from inequivalent ones, while also utilizing signals about the mutual equivalence between IRs within the group. This process allows LLMs to capture nuanced aspects of code functionality. By employing OORL for training with code translation tasks, LLMs improve their recognition of code functionality and their understanding of the relationships between code implemented in different languages. Extensive experiments demonstrate that our OORL for LLMs training with code translation tasks achieves significant performance improvements on code benchmarks across multiple programming languages. 

---
# Automated Bias Assessment in AI-Generated Educational Content Using CEAT Framework 

**Authors**: Jingyang Peng, Wenyuan Shen, Jiarui Rao, Jionghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12718)  

**Abstract**: Recent advances in Generative Artificial Intelligence (GenAI) have transformed educational content creation, particularly in developing tutor training materials. However, biases embedded in AI-generated content--such as gender, racial, or national stereotypes--raise significant ethical and educational concerns. Despite the growing use of GenAI, systematic methods for detecting and evaluating such biases in educational materials remain limited. This study proposes an automated bias assessment approach that integrates the Contextualized Embedding Association Test with a prompt-engineered word extraction method within a Retrieval-Augmented Generation framework. We applied this method to AI-generated texts used in tutor training lessons. Results show a high alignment between the automated and manually curated word sets, with a Pearson correlation coefficient of r = 0.993, indicating reliable and consistent bias assessment. Our method reduces human subjectivity and enhances fairness, scalability, and reproducibility in auditing GenAI-produced educational content. 

---
# ToTRL: Unlock LLM Tree-of-Thoughts Reasoning Potential through Puzzles Solving 

**Authors**: Haoyuan Wu, Xueyi Chen, Rui Ming, Jilong Gao, Shoubo Hu, Zhuolun He, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12717)  

**Abstract**: Large language models (LLMs) demonstrate significant reasoning capabilities, particularly through long chain-of-thought (CoT) processes, which can be elicited by reinforcement learning (RL). However, prolonged CoT reasoning presents limitations, primarily verbose outputs due to excessive introspection. The reasoning process in these LLMs often appears to follow a trial-and-error methodology rather than a systematic, logical deduction. In contrast, tree-of-thoughts (ToT) offers a conceptually more advanced approach by modeling reasoning as an exploration within a tree structure. This reasoning structure facilitates the parallel generation and evaluation of multiple reasoning branches, allowing for the active identification, assessment, and pruning of unproductive paths. This process can potentially lead to improved performance and reduced token costs. Building upon the long CoT capability of LLMs, we introduce tree-of-thoughts RL (ToTRL), a novel on-policy RL framework with a rule-based reward. ToTRL is designed to guide LLMs in developing the parallel ToT strategy based on the sequential CoT strategy. Furthermore, we employ LLMs as players in a puzzle game during the ToTRL training process. Solving puzzle games inherently necessitates exploring interdependent choices and managing multiple constraints, which requires the construction and exploration of a thought tree, providing challenging tasks for cultivating the ToT reasoning capability. Our empirical evaluations demonstrate that our ToTQwen3-8B model, trained with our ToTRL, achieves significant improvement in performance and reasoning efficiency on complex reasoning tasks. 

---
# Shadow-FT: Tuning Instruct via Base 

**Authors**: Taiqiang Wu, Runming Yang, Jiayi Li, Pengfei Hu, Ngai Wong, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12716)  

**Abstract**: Large language models (LLMs) consistently benefit from further fine-tuning on various tasks. However, we observe that directly tuning the INSTRUCT (i.e., instruction tuned) models often leads to marginal improvements and even performance degeneration. Notably, paired BASE models, the foundation for these INSTRUCT variants, contain highly similar weight values (i.e., less than 2% on average for Llama 3.1 8B). Therefore, we propose a novel Shadow-FT framework to tune the INSTRUCT models by leveraging the corresponding BASE models. The key insight is to fine-tune the BASE model, and then directly graft the learned weight updates to the INSTRUCT model. Our proposed Shadow-FT introduces no additional parameters, is easy to implement, and significantly improves performance. We conduct extensive experiments on tuning mainstream LLMs, such as Qwen 3 and Llama 3 series, and evaluate them across 19 benchmarks covering coding, reasoning, and mathematical tasks. Experimental results demonstrate that Shadow-FT consistently outperforms conventional full-parameter and parameter-efficient tuning approaches. Further analyses indicate that Shadow-FT can be applied to multimodal large language models (MLLMs) and combined with direct preference optimization (DPO). Codes and weights are available at \href{this https URL}{Github}. 

---
# Know3-RAG: A Knowledge-aware RAG Framework with Adaptive Retrieval, Generation, and Filtering 

**Authors**: Xukai Liu, Ye Liu, Shiwen Wu, Yanghai Zhang, Yihao Yuan, Kai Zhang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12662)  

**Abstract**: Recent advances in large language models (LLMs) have led to impressive progress in natural language generation, yet their tendency to produce hallucinated or unsubstantiated content remains a critical concern. To improve factual reliability, Retrieval-Augmented Generation (RAG) integrates external knowledge during inference. However, existing RAG systems face two major limitations: (1) unreliable adaptive control due to limited external knowledge supervision, and (2) hallucinations caused by inaccurate or irrelevant references. To address these issues, we propose Know3-RAG, a knowledge-aware RAG framework that leverages structured knowledge from knowledge graphs (KGs) to guide three core stages of the RAG process, including retrieval, generation, and filtering. Specifically, we introduce a knowledge-aware adaptive retrieval module that employs KG embedding to assess the confidence of the generated answer and determine retrieval necessity, a knowledge-enhanced reference generation strategy that enriches queries with KG-derived entities to improve generated reference relevance, and a knowledge-driven reference filtering mechanism that ensures semantic alignment and factual accuracy of references. Experiments on multiple open-domain QA benchmarks demonstrate that Know3-RAG consistently outperforms strong baselines, significantly reducing hallucinations and enhancing answer reliability. 

---
# Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals 

**Authors**: Yuxin Lin, Yinglin Zheng, Ming Zeng, Wangzheng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12654)  

**Abstract**: This paper addresses the gap in predicting turn-taking and backchannel actions in human-machine conversations using multi-modal signals (linguistic, acoustic, and visual). To overcome the limitation of existing datasets, we propose an automatic data collection pipeline that allows us to collect and annotate over 210 hours of human conversation videos. From this, we construct a Multi-Modal Face-to-Face (MM-F2F) human conversation dataset, including over 1.5M words and corresponding turn-taking and backchannel annotations from approximately 20M frames. Additionally, we present an end-to-end framework that predicts the probability of turn-taking and backchannel actions from multi-modal signals. The proposed model emphasizes the interrelation between modalities and supports any combination of text, audio, and video inputs, making it adaptable to a variety of realistic scenarios. Our experiments show that our approach achieves state-of-the-art performance on turn-taking and backchannel prediction tasks, achieving a 10\% increase in F1-score on turn-taking and a 33\% increase on backchannel prediction. Our dataset and code are publicly available online to ease of subsequent research. 

---
# Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing 

**Authors**: Jiakuan Xie, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12636)  

**Abstract**: Knowledge editing, which aims to update the knowledge encoded in language models, can be deceptive. Despite the fact that many existing knowledge editing algorithms achieve near-perfect performance on conventional metrics, the models edited by them are still prone to generating original knowledge. This paper introduces the concept of "superficial editing" to describe this phenomenon. Our comprehensive evaluation reveals that this issue presents a significant challenge to existing algorithms. Through systematic investigation, we identify and validate two key factors contributing to this issue: (1) the residual stream at the last subject position in earlier layers and (2) specific attention modules in later layers. Notably, certain attention heads in later layers, along with specific left singular vectors in their output matrices, encapsulate the original knowledge and exhibit a causal relationship with superficial editing. Furthermore, we extend our analysis to the task of superficial unlearning, where we observe consistent patterns in the behavior of specific attention heads and their corresponding left singular vectors, thereby demonstrating the robustness and broader applicability of our methodology and conclusions. Our code is available here. 

---
# R1dacted: Investigating Local Censorship in DeepSeek's R1 Language Model 

**Authors**: Ali Naseh, Harsh Chaudhari, Jaechul Roh, Mingshi Wu, Alina Oprea, Amir Houmansadr  

**Link**: [PDF](https://arxiv.org/pdf/2505.12625)  

**Abstract**: DeepSeek recently released R1, a high-performing large language model (LLM) optimized for reasoning tasks. Despite its efficient training pipeline, R1 achieves competitive performance, even surpassing leading reasoning models like OpenAI's o1 on several benchmarks. However, emerging reports suggest that R1 refuses to answer certain prompts related to politically sensitive topics in China. While existing LLMs often implement safeguards to avoid generating harmful or offensive outputs, R1 represents a notable shift - exhibiting censorship-like behavior on politically charged queries. In this paper, we investigate this phenomenon by first introducing a large-scale set of heavily curated prompts that get censored by R1, covering a range of politically sensitive topics, but are not censored by other models. We then conduct a comprehensive analysis of R1's censorship patterns, examining their consistency, triggers, and variations across topics, prompt phrasing, and context. Beyond English-language queries, we explore censorship behavior in other languages. We also investigate the transferability of censorship to models distilled from the R1 language model. Finally, we propose techniques for bypassing or removing this censorship. Our findings reveal possible additional censorship integration likely shaped by design choices during training or alignment, raising concerns about transparency, bias, and governance in language model deployment. 

---
# Think Before You Attribute: Improving the Performance of LLMs Attribution Systems 

**Authors**: João Eduardo Batista, Emil Vatai, Mohamed Wahib  

**Link**: [PDF](https://arxiv.org/pdf/2505.12621)  

**Abstract**: Large Language Models (LLMs) are increasingly applied in various science domains, yet their broader adoption remains constrained by a critical challenge: the lack of trustworthy, verifiable outputs. Current LLMs often generate answers without reliable source attribution, or worse, with incorrect attributions, posing a barrier to their use in scientific and high-stakes settings, where traceability and accountability are non-negotiable. To be reliable, attribution systems need high accuracy and retrieve data with short lengths, i.e., attribute to a sentence within a document rather than a whole document. We propose a sentence-level pre-attribution step for Retrieve-Augmented Generation (RAG) systems that classify sentences into three categories: not attributable, attributable to a single quote, and attributable to multiple quotes. By separating sentences before attribution, a proper attribution method can be selected for the type of sentence, or the attribution can be skipped altogether. Our results indicate that classifiers are well-suited for this task. In this work, we propose a pre-attribution step to reduce the computational complexity of attribution, provide a clean version of the HAGRID dataset, and provide an end-to-end attribution system that works out of the box. 

---
# Duluth at SemEval-2025 Task 7: TF-IDF with Optimized Vector Dimensions for Multilingual Fact-Checked Claim Retrieval 

**Authors**: Shujauddin Syed, Ted Pedersen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12616)  

**Abstract**: This paper presents the Duluth approach to the SemEval-2025 Task 7 on Multilingual and Crosslingual Fact-Checked Claim Retrieval. We implemented a TF-IDF-based retrieval system with experimentation on vector dimensions and tokenization strategies. Our best-performing configuration used word-level tokenization with a vocabulary size of 15,000 features, achieving an average success@10 score of 0.78 on the development set and 0.69 on the test set across ten languages. Our system showed stronger performance on higher-resource languages but still lagged significantly behind the top-ranked system, which achieved 0.96 average success@10. Our findings suggest that though advanced neural architectures are increasingly dominant in multilingual retrieval tasks, properly optimized traditional methods like TF-IDF remain competitive baselines, especially in limited compute resource scenarios. 

---
# AD-AGENT: A Multi-agent Framework for End-to-end Anomaly Detection 

**Authors**: Tiankai Yang, Junjun Liu, Wingchun Siu, Jiahang Wang, Zhuangzhuang Qian, Chanjuan Song, Cheng Cheng, Xiyang Hu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12594)  

**Abstract**: Anomaly detection (AD) is essential in areas such as fraud detection, network monitoring, and scientific research. However, the diversity of data modalities and the increasing number of specialized AD libraries pose challenges for non-expert users who lack in-depth library-specific knowledge and advanced programming skills. To tackle this, we present AD-AGENT, an LLM-driven multi-agent framework that turns natural-language instructions into fully executable AD pipelines. AD-AGENT coordinates specialized agents for intent parsing, data preparation, library and model selection, documentation mining, and iterative code generation and debugging. Using a shared short-term workspace and a long-term cache, the agents integrate popular AD libraries like PyOD, PyGOD, and TSLib into a unified workflow. Experiments demonstrate that AD-AGENT produces reliable scripts and recommends competitive models across libraries. The system is open-sourced to support further research and practical applications in AD. 

---
# PromptPrism: A Linguistically-Inspired Taxonomy for Prompts 

**Authors**: Sullam Jeoung, Yueyan Chen, Yi Zhang, Shuai Wang, Haibo Ding, Lin Lee Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12592)  

**Abstract**: Prompts are the interface for eliciting the capabilities of large language models (LLMs). Understanding their structure and components is critical for analyzing LLM behavior and optimizing performance. However, the field lacks a comprehensive framework for systematic prompt analysis and understanding. We introduce PromptPrism, a linguistically-inspired taxonomy that enables prompt analysis across three hierarchical levels: functional structure, semantic component, and syntactic pattern. We show the practical utility of PromptPrism by applying it to three applications: (1) a taxonomy-guided prompt refinement approach that automatically improves prompt quality and enhances model performance across a range of tasks; (2) a multi-dimensional dataset profiling method that extracts and aggregates structural, semantic, and syntactic characteristics from prompt datasets, enabling comprehensive analysis of prompt distributions and patterns; (3) a controlled experimental framework for prompt sensitivity analysis by quantifying the impact of semantic reordering and delimiter modifications on LLM performance. Our experimental results validate the effectiveness of our taxonomy across these applications, demonstrating that PromptPrism provides a foundation for refining, profiling, and analyzing prompts. 

---
# CMLFormer: A Dual Decoder Transformer with Switching Point Learning for Code-Mixed Language Modeling 

**Authors**: Aditeya Baral, Allen George Ajith, Roshan Nayak, Mrityunjay Abhijeet Bhanja  

**Link**: [PDF](https://arxiv.org/pdf/2505.12587)  

**Abstract**: Code-mixed languages, characterized by frequent within-sentence language transitions, present structural challenges that standard language models fail to address. In this work, we propose CMLFormer, an enhanced multi-layer dual-decoder Transformer with a shared encoder and synchronized decoder cross-attention, designed to model the linguistic and semantic dynamics of code-mixed text. CMLFormer is pre-trained on an augmented Hinglish corpus with switching point and translation annotations with multiple new objectives specifically aimed at capturing switching behavior, cross-lingual structure, and code-mixing complexity. Our experiments show that CMLFormer improves F1 score, precision, and accuracy over other approaches on the HASOC-2021 benchmark under select pre-training setups. Attention analyses further show that it can identify and attend to switching points, validating its sensitivity to code-mixed structure. These results demonstrate the effectiveness of CMLFormer's architecture and multi-task pre-training strategy for modeling code-mixed languages. 

---
# Improving Multilingual Language Models by Aligning Representations through Steering 

**Authors**: Omar Mahmoud, Buddhika Laknath Semage, Thommen George Karimpanal, Santu Rana  

**Link**: [PDF](https://arxiv.org/pdf/2505.12584)  

**Abstract**: In this paper, we investigate how large language models (LLMS) process non-English tokens within their layer representations, an open question despite significant advancements in the field. Using representation steering, specifically by adding a learned vector to a single model layer's activations, we demonstrate that steering a single model layer can notably enhance performance. Our analysis shows that this approach achieves results comparable to translation baselines and surpasses state of the art prompt optimization methods. Additionally, we highlight how advanced techniques like supervised fine tuning (\textsc{sft}) and reinforcement learning from human feedback (\textsc{rlhf}) improve multilingual capabilities by altering representation spaces. We further illustrate how these methods align with our approach to reshaping LLMS layer representations. 

---
# Measuring Information Distortion in Hierarchical Ultra long Novel Generation:The Optimal Expansion Ratio 

**Authors**: Hanwen Shen, Ting Ying  

**Link**: [PDF](https://arxiv.org/pdf/2505.12572)  

**Abstract**: Writing novels with Large Language Models (LLMs) raises a critical question: how much human-authored outline is necessary to generate high-quality million-word novels? While frameworks such as DOME, Plan&Write, and Long Writer have improved stylistic coherence and logical consistency, they primarily target shorter novels (10k--100k words), leaving ultra-long generation largely unexplored. Drawing on insights from recent text compression methods like LLMZip and LLM2Vec, we conduct an information-theoretic analysis that quantifies distortion occurring when LLMs compress and reconstruct ultra-long novels under varying compression-expansion ratios. We introduce a hierarchical two-stage generation pipeline (outline -> detailed outline -> manuscript) and find an optimal outline length that balances information preservation with human effort. Through extensive experimentation with Chinese novels, we establish that a two-stage hierarchical outline approach significantly reduces semantic distortion compared to single-stage methods. Our findings provide empirically-grounded guidance for authors and researchers collaborating with LLMs to create million-word novels. 

---
# Enriching Patent Claim Generation with European Patent Dataset 

**Authors**: Lekang Jiang, Chengzu Li, Stephan Goetz  

**Link**: [PDF](https://arxiv.org/pdf/2505.12568)  

**Abstract**: Drafting patent claims is time-intensive, costly, and requires professional skill. Therefore, researchers have investigated large language models (LLMs) to assist inventors in writing claims. However, existing work has largely relied on datasets from the United States Patent and Trademark Office (USPTO). To enlarge research scope regarding various jurisdictions, drafting conventions, and legal standards, we introduce EPD, a European patent dataset. EPD presents rich textual data and structured metadata to support multiple patent-related tasks, including claim generation. This dataset enriches the field in three critical aspects: (1) Jurisdictional diversity: Patents from different offices vary in legal and drafting conventions. EPD fills a critical gap by providing a benchmark for European patents to enable more comprehensive evaluation. (2) Quality improvement: EPD offers high-quality granted patents with finalized and legally approved texts, whereas others consist of patent applications that are unexamined or provisional. Experiments show that LLMs fine-tuned on EPD significantly outperform those trained on previous datasets and even GPT-4o in claim quality and cross-domain generalization. (3) Real-world simulation: We propose a difficult subset of EPD to better reflect real-world challenges of claim generation. Results reveal that all tested LLMs perform substantially worse on these challenging samples, which highlights the need for future research. 

---
# The taggedPBC: Annotating a massive parallel corpus for crosslinguistic investigations 

**Authors**: Hiram Ring  

**Link**: [PDF](https://arxiv.org/pdf/2505.12560)  

**Abstract**: Existing datasets available for crosslinguistic investigations have tended to focus on large amounts of data for a small group of languages or a small amount of data for a large number of languages. This means that claims based on these datasets are limited in what they reveal about universal properties of the human language faculty. While this has begun to change through the efforts of projects seeking to develop tagged corpora for a large number of languages, such efforts are still constrained by limits on resources. The current paper reports on a large automatically tagged parallel dataset which has been developed to partially address this issue. The taggedPBC contains more than 1,800 sentences of pos-tagged parallel text data from over 1,500 languages, representing 133 language families and 111 isolates, dwarfing previously available resources. The accuracy of tags in this dataset is shown to correlate well with both existing SOTA taggers for high-resource languages (SpaCy, Trankit) as well as hand-tagged corpora (Universal Dependencies Treebanks). Additionally, a novel measure derived from this dataset, the N1 ratio, correlates with expert determinations of word order in three typological databases (WALS, Grambank, Autotyp) such that a Gaussian Naive Bayes classifier trained on this feature can accurately identify basic word order for languages not in those databases. While much work is still needed to expand and develop this dataset, the taggedPBC is an important step to enable corpus-based crosslinguistic investigations, and is made available for research and collaboration via GitHub. 

---
# Extracting memorized pieces of (copyrighted) books from open-weight language models 

**Authors**: A. Feder Cooper, Aaron Gokaslan, Amy B. Cyphert, Christopher De Sa, Mark A. Lemley, Daniel E. Ho, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12546)  

**Abstract**: Plaintiffs and defendants in copyright lawsuits over generative AI often make sweeping, opposing claims about the extent to which large language models (LLMs) have memorized plaintiffs' protected expression. Drawing on adversarial ML and copyright law, we show that these polarized positions dramatically oversimplify the relationship between memorization and copyright. To do so, we leverage a recent probabilistic extraction technique to extract pieces of the Books3 dataset from 13 open-weight LLMs. Through numerous experiments, we show that it's possible to extract substantial parts of at least some books from different LLMs. This is evidence that the LLMs have memorized the extracted text; this memorized content is copied inside the model parameters. But the results are complicated: the extent of memorization varies both by model and by book. With our specific experiments, we find that the largest LLMs don't memorize most books -- either in whole or in part. However, we also find that Llama 3.1 70B memorizes some books, like Harry Potter and 1984, almost entirely. We discuss why our results have significant implications for copyright cases, though not ones that unambiguously favor either side. 

---
# Towards Reliable and Interpretable Traffic Crash Pattern Prediction and Safety Interventions Using Customized Large Language Models 

**Authors**: Yang Zhao, Pu Wang, Yibo Zhao, Hongru Du, Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12545)  

**Abstract**: Predicting crash events is crucial for understanding crash distributions and their contributing factors, thereby enabling the design of proactive traffic safety policy interventions. However, existing methods struggle to interpret the complex interplay among various sources of traffic crash data, including numeric characteristics, textual reports, crash imagery, environmental conditions, and driver behavior records. As a result, they often fail to capture the rich semantic information and intricate interrelationships embedded in these diverse data sources, limiting their ability to identify critical crash risk factors. In this research, we propose TrafficSafe, a framework that adapts LLMs to reframe crash prediction and feature attribution as text-based reasoning. A multi-modal crash dataset including 58,903 real-world reports together with belonged infrastructure, environmental, driver, and vehicle information is collected and textualized into TrafficSafe Event Dataset. By customizing and fine-tuning LLMs on this dataset, the TrafficSafe LLM achieves a 42% average improvement in F1-score over baselines. To interpret these predictions and uncover contributing factors, we introduce TrafficSafe Attribution, a sentence-level feature attribution framework enabling conditional risk analysis. Findings show that alcohol-impaired driving is the leading factor in severe crashes, with aggressive and impairment-related behaviors having nearly twice the contribution for severe crashes compared to other driver behaviors. Furthermore, TrafficSafe Attribution highlights pivotal features during model training, guiding strategic crash data collection for iterative performance improvements. The proposed TrafficSafe offers a transformative leap in traffic safety research, providing a blueprint for translating advanced AI technologies into responsible, actionable, and life-saving outcomes. 

---
# Disambiguation in Conversational Question Answering in the Era of LLM: A Survey 

**Authors**: Md Mehrab Tanjim, Yeonjun In, Xiang Chen, Victor S. Bursztyn, Ryan A. Rossi, Sungchul Kim, Guang-Jie Ren, Vaishnavi Muppala, Shun Jiang, Yongsung Kim, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.12543)  

**Abstract**: Ambiguity remains a fundamental challenge in Natural Language Processing (NLP) due to the inherent complexity and flexibility of human language. With the advent of Large Language Models (LLMs), addressing ambiguity has become even more critical due to their expanded capabilities and applications. In the context of Conversational Question Answering (CQA), this paper explores the definition, forms, and implications of ambiguity for language driven systems, particularly in the context of LLMs. We define key terms and concepts, categorize various disambiguation approaches enabled by LLMs, and provide a comparative analysis of their advantages and disadvantages. We also explore publicly available datasets for benchmarking ambiguity detection and resolution techniques and highlight their relevance for ongoing research. Finally, we identify open problems and future research directions, proposing areas for further investigation. By offering a comprehensive review of current research on ambiguities and disambiguation with LLMs, we aim to contribute to the development of more robust and reliable language systems. 

---
# Relation Extraction or Pattern Matching? Unravelling the Generalisation Limits of Language Models for Biographical RE 

**Authors**: Varvara Arzt, Allan Hanbury, Michael Wiegand, Gábor Recski, Terra Blevins  

**Link**: [PDF](https://arxiv.org/pdf/2505.12533)  

**Abstract**: Analysing the generalisation capabilities of relation extraction (RE) models is crucial for assessing whether they learn robust relational patterns or rely on spurious correlations. Our cross-dataset experiments find that RE models struggle with unseen data, even within similar domains. Notably, higher intra-dataset performance does not indicate better transferability, instead often signaling overfitting to dataset-specific artefacts. Our results also show that data quality, rather than lexical similarity, is key to robust transfer, and the choice of optimal adaptation strategy depends on the quality of data available: while fine-tuning yields the best cross-dataset performance with high-quality data, few-shot in-context learning (ICL) is more effective with noisier data. However, even in these cases, zero-shot baselines occasionally outperform all cross-dataset results. Structural issues in RE benchmarks, such as single-relation per sample constraints and non-standardised negative class definitions, further hinder model transferability. 

---
# ESC-Judge: A Framework for Comparing Emotional Support Conversational Agents 

**Authors**: Navid Madani, Rohini Srihari  

**Link**: [PDF](https://arxiv.org/pdf/2505.12531)  

**Abstract**: Large language models (LLMs) increasingly power mental-health chatbots, yet the field still lacks a scalable, theory-grounded way to decide which model is most effective to deploy. We present ESC-Judge, the first end-to-end evaluation framework that (i) grounds head-to-head comparisons of emotional-support LLMs in Clara Hill's established Exploration-Insight-Action counseling model, providing a structured and interpretable view of performance, and (ii) fully automates the evaluation pipeline at scale. ESC-Judge operates in three stages: first, it synthesizes realistic help-seeker roles by sampling empirically salient attributes such as stressors, personality, and life history; second, it has two candidate support agents conduct separate sessions with the same role, isolating model-specific strategies; and third, it asks a specialized judge LLM to express pairwise preferences across rubric-anchored skills that span the Exploration, Insight, and Action spectrum. In our study, ESC-Judge matched PhD-level annotators on 85 percent of Exploration, 83 percent of Insight, and 86 percent of Action decisions, demonstrating human-level reliability at a fraction of the cost. All code, prompts, synthetic roles, transcripts, and judgment scripts are released to promote transparent progress in emotionally supportive AI. 

---
# DS-ProGen: A Dual-Structure Deep Language Model for Functional Protein Design 

**Authors**: Yanting Li, Jiyue Jiang, Zikang Wang, Ziqian Lin, Dongchen He, Yuheng Shan, Yanruisheng Shao, Jiayi Li, Xiangyu Shi, Jiuming Wang, Yanyu Chen, Yimin Fan, Han Li, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12511)  

**Abstract**: Inverse Protein Folding (IPF) is a critical subtask in the field of protein design, aiming to engineer amino acid sequences capable of folding correctly into a specified three-dimensional (3D) conformation. Although substantial progress has been achieved in recent years, existing methods generally rely on either backbone coordinates or molecular surface features alone, which restricts their ability to fully capture the complex chemical and geometric constraints necessary for precise sequence prediction. To address this limitation, we present DS-ProGen, a dual-structure deep language model for functional protein design, which integrates both backbone geometry and surface-level representations. By incorporating backbone coordinates as well as surface chemical and geometric descriptors into a next-amino-acid prediction paradigm, DS-ProGen is able to generate functionally relevant and structurally stable sequences while satisfying both global and local conformational constraints. On the PRIDE dataset, DS-ProGen attains the current state-of-the-art recovery rate of 61.47%, demonstrating the synergistic advantage of multi-modal structural encoding in protein design. Furthermore, DS-ProGen excels in predicting interactions with a variety of biological partners, including ligands, ions, and RNA, confirming its robust functional retention capabilities. 

---
# LM$^2$otifs : An Explainable Framework for Machine-Generated Texts Detection 

**Authors**: Xu Zheng, Zhuomin Chen, Esteban Schafir, Sipeng Chen, Hojat Allah Salehi, Haifeng Chen, Farhad Shirani, Wei Cheng, Dongsheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12507)  

**Abstract**: The impressive ability of large language models to generate natural text across various tasks has led to critical challenges in authorship authentication. Although numerous detection methods have been developed to differentiate between machine-generated texts (MGT) and human-generated texts (HGT), the explainability of these methods remains a significant gap. Traditional explainability techniques often fall short in capturing the complex word relationships that distinguish HGT from MGT. To address this limitation, we present LM$^2$otifs, a novel explainable framework for MGT detection. Inspired by probabilistic graphical models, we provide a theoretical rationale for the effectiveness. LM$^2$otifs utilizes eXplainable Graph Neural Networks to achieve both accurate detection and interpretability. The LM$^2$otifs pipeline operates in three key stages: first, it transforms text into graphs based on word co-occurrence to represent lexical dependencies; second, graph neural networks are used for prediction; and third, a post-hoc explainability method extracts interpretable motifs, offering multi-level explanations from individual words to sentence structures. Extensive experiments on multiple benchmark datasets demonstrate the comparable performance of LM$^2$otifs. The empirical evaluation of the extracted explainable motifs confirms their effectiveness in differentiating HGT and MGT. Furthermore, qualitative analysis reveals distinct and visible linguistic fingerprints characteristic of MGT. 

---
# KG-QAGen: A Knowledge-Graph-Based Framework for Systematic Question Generation and Long-Context LLM Evaluation 

**Authors**: Nikita Tatarinov, Vidhyakshaya Kannan, Haricharana Srinivasa, Arnav Raj, Harpreet Singh Anand, Varun Singh, Aditya Luthra, Ravij Lade, Agam Shah, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2505.12495)  

**Abstract**: The increasing context length of modern language models has created a need for evaluating their ability to retrieve and process information across extensive documents. While existing benchmarks test long-context capabilities, they often lack a structured way to systematically vary question complexity. We introduce KG-QAGen (Knowledge-Graph-based Question-Answer Generation), a framework that (1) extracts QA pairs at multiple complexity levels (2) by leveraging structured representations of financial agreements (3) along three key dimensions -- multi-hop retrieval, set operations, and answer plurality -- enabling fine-grained assessment of model performance across controlled difficulty levels. Using this framework, we construct a dataset of 20,139 QA pairs (the largest number among the long-context benchmarks) and open-source a part of it. We evaluate 13 proprietary and open-source LLMs and observe that even the best-performing models are struggling with set-based comparisons and multi-hop logical inference. Our analysis reveals systematic failure modes tied to semantic misinterpretation and inability to handle implicit relations. 

---
# Enhancing Large Language Models with Reward-guided Tree Search for Knowledge Graph Question and Answering 

**Authors**: Xiao Long, Liansheng Zhuang, Chen Shen, Shaotian Yan, Yifei Li, Shafei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12476)  

**Abstract**: Recently, large language models (LLMs) have demonstrated impressive performance in Knowledge Graph Question Answering (KGQA) tasks, which aim to find answers based on knowledge graphs (KGs) for natural language questions. Existing LLMs-based KGQA methods typically follow the Graph Retrieval-Augmented Generation (GraphRAG) paradigm, which first retrieves reasoning paths from the large KGs, and then generates the answers based on them. However, these methods emphasize the exploration of new optimal reasoning paths in KGs while ignoring the exploitation of historical reasoning paths, which may lead to sub-optimal reasoning paths. Additionally, the complex semantics contained in questions may lead to the retrieval of inaccurate reasoning paths. To address these issues, this paper proposes a novel and training-free framework for KGQA tasks called Reward-guided Tree Search on Graph (RTSoG). RTSoG decomposes an original question into a series of simpler and well-defined sub-questions to handle the complex semantics. Then, a Self-Critic Monte Carlo Tree Search (SC-MCTS) guided by a reward model is introduced to iteratively retrieve weighted reasoning paths as contextual knowledge. Finally, it stacks the weighted reasoning paths according to their weights to generate the final answers. Extensive experiments on four datasets demonstrate the effectiveness of RTSoG. Notably, it achieves 8.7\% and 7.0\% performance improvement over the state-of-the-art method on the GrailQA and the WebQSP respectively. 

---
# What are they talking about? Benchmarking Large Language Models for Knowledge-Grounded Discussion Summarization 

**Authors**: Weixiao Zhou, Junnan Zhu, Gengyao Li, Xianfu Cheng, Xinnian Liang, Feifei Zhai, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12474)  

**Abstract**: In this work, we investigate the performance of LLMs on a new task that requires combining discussion with background knowledge for summarization. This aims to address the limitation of outside observer confusion in existing dialogue summarization systems due to their reliance solely on discussion information. To achieve this, we model the task output as background and opinion summaries and define two standardized summarization patterns. To support assessment, we introduce the first benchmark comprising high-quality samples consistently annotated by human experts and propose a novel hierarchical evaluation framework with fine-grained, interpretable metrics. We evaluate 12 LLMs under structured-prompt and self-reflection paradigms. Our findings reveal: (1) LLMs struggle with background summary retrieval, generation, and opinion summary integration. (2) Even top LLMs achieve less than 69% average performance across both patterns. (3) Current LLMs lack adequate self-evaluation and self-correction capabilities for this task. 

---
# Towards DS-NER: Unveiling and Addressing Latent Noise in Distant Annotations 

**Authors**: Yuyang Ding, Dan Qiao, Juntao Li, Jiajie Xu, Pingfu Chao, Xiaofang Zhou, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12454)  

**Abstract**: Distantly supervised named entity recognition (DS-NER) has emerged as a cheap and convenient alternative to traditional human annotation methods, enabling the automatic generation of training data by aligning text with external resources. Despite the many efforts in noise measurement methods, few works focus on the latent noise distribution between different distant annotation methods. In this work, we explore the effectiveness and robustness of DS-NER by two aspects: (1) distant annotation techniques, which encompasses both traditional rule-based methods and the innovative large language model supervision approach, and (2) noise assessment, for which we introduce a novel framework. This framework addresses the challenges by distinctly categorizing them into the unlabeled-entity problem (UEP) and the noisy-entity problem (NEP), subsequently providing specialized solutions for each. Our proposed method achieves significant improvements on eight real-world distant supervision datasets originating from three different data sources and involving four distinct annotation techniques, confirming its superiority over current state-of-the-art methods. 

---
# Introspective Growth: Automatically Advancing LLM Expertise in Technology Judgment 

**Authors**: Siyang Wu, Honglin Bao, Nadav Kunievsky, James A. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2505.12452)  

**Abstract**: Large language models (LLMs) increasingly demonstrate signs of conceptual understanding, yet much of their internal knowledge remains latent, loosely structured, and difficult to access or evaluate. We propose self-questioning as a lightweight and scalable strategy to improve LLMs' understanding, particularly in domains where success depends on fine-grained semantic distinctions. To evaluate this approach, we introduce a challenging new benchmark of 1.3 million post-2015 computer science patent pairs, characterized by dense technical jargon and strategically complex writing. The benchmark centers on a pairwise differentiation task: can a model distinguish between closely related but substantively different inventions? We show that prompting LLMs to generate and answer their own questions - targeting the background knowledge required for the task - significantly improves performance. These self-generated questions and answers activate otherwise underutilized internal knowledge. Allowing LLMs to retrieve answers from external scientific texts further enhances performance, suggesting that model knowledge is compressed and lacks the full richness of the training data. We also find that chain-of-thought prompting and self-questioning converge, though self-questioning remains more effective for improving understanding of technical concepts. Notably, we uncover an asymmetry in prompting: smaller models often generate more fundamental, more open-ended, better-aligned questions for mid-sized models than large models with better understanding do, revealing a new strategy for cross-model collaboration. Altogether, our findings establish self-questioning as both a practical mechanism for automatically improving LLM comprehension, especially in domains with sparse and underrepresented knowledge, and a diagnostic probe of how internal and external knowledge are organized. 

---
# Learning to Play Like Humans: A Framework for LLM Adaptation in Interactive Fiction Games 

**Authors**: Jinming Zhang, Yunfei Long  

**Link**: [PDF](https://arxiv.org/pdf/2505.12439)  

**Abstract**: Interactive Fiction games (IF games) are where players interact through natural language commands. While recent advances in Artificial Intelligence agents have reignited interest in IF games as a domain for studying decision-making, existing approaches prioritize task-specific performance metrics over human-like comprehension of narrative context and gameplay logic. This work presents a cognitively inspired framework that guides Large Language Models (LLMs) to learn and play IF games systematically. Our proposed **L**earning to **P**lay **L**ike **H**umans (LPLH) framework integrates three key components: (1) structured map building to capture spatial and narrative relationships, (2) action learning to identify context-appropriate commands, and (3) feedback-driven experience analysis to refine decision-making over time. By aligning LLMs-based agents' behavior with narrative intent and commonsense constraints, LPLH moves beyond purely exploratory strategies to deliver more interpretable, human-like performance. Crucially, this approach draws on cognitive science principles to more closely simulate how human players read, interpret, and respond within narrative worlds. As a result, LPLH reframes the IF games challenge as a learning problem for LLMs-based agents, offering a new path toward robust, context-aware gameplay in complex text-based environments. 

---
# PSC: Extending Context Window of Large Language Models via Phase Shift Calibration 

**Authors**: Wenqiao Zhu, Chao Xu, Lulu Wang, Jun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12423)  

**Abstract**: Rotary Position Embedding (RoPE) is an efficient position encoding approach and is widely utilized in numerous large language models (LLMs). Recently, a lot of methods have been put forward to further expand the context window based on RoPE. The core concept of those methods is to predefine or search for a set of factors to rescale the base frequencies of RoPE. Nevertheless, it is quite a challenge for existing methods to predefine an optimal factor due to the exponential search space. In view of this, we introduce PSC (Phase Shift Calibration), a small module for calibrating the frequencies predefined by existing methods. With the employment of PSC, we demonstrate that many existing methods can be further enhanced, like PI, YaRN, and LongRoPE. We conducted extensive experiments across multiple models and tasks. The results demonstrate that (1) when PSC is enabled, the comparative reductions in perplexity increase as the context window size is varied from 16k, to 32k, and up to 64k. (2) Our approach is broadly applicable and exhibits robustness across a variety of models and tasks. The code can be found at this https URL. 

---
# Table-R1: Region-based Reinforcement Learning for Table Understanding 

**Authors**: Zhenhe Wu, Jian Yang, Jiaheng Liu, Xianjie Wu, Changzai Pan, Jie Zhang, Yu Zhao, Shuangyong Song, Yongxiang Li, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12415)  

**Abstract**: Tables present unique challenges for language models due to their structured row-column interactions, necessitating specialized approaches for effective comprehension. While large language models (LLMs) have demonstrated potential in table reasoning through prompting and techniques like chain-of-thought (CoT) and program-of-thought (PoT), optimizing their performance for table question answering remains underexplored. In this paper, we introduce region-based Table-R1, a novel reinforcement learning approach that enhances LLM table understanding by integrating region evidence into reasoning steps. Our method employs Region-Enhanced Supervised Fine-Tuning (RE-SFT) to guide models in identifying relevant table regions before generating answers, incorporating textual, symbolic, and program-based reasoning. Additionally, Table-Aware Group Relative Policy Optimization (TARPO) introduces a mixed reward system to dynamically balance region accuracy and answer correctness, with decaying region rewards and consistency penalties to align reasoning steps. Experiments show that Table-R1 achieves an average performance improvement of 14.36 points across multiple base models on three benchmark datasets, even outperforming baseline models with ten times the parameters, while TARPO reduces response token consumption by 67.5% compared to GRPO, significantly advancing LLM capabilities in efficient tabular reasoning. 

---
# The power of text similarity in identifying AI-LLM paraphrased documents: The case of BBC news articles and ChatGPT 

**Authors**: Konstantinos Xylogiannopoulos, Petros Xanthopoulos, Panagiotis Karampelas, Georgios Bakamitsos  

**Link**: [PDF](https://arxiv.org/pdf/2505.12405)  

**Abstract**: Generative AI paraphrased text can be used for copyright infringement and the AI paraphrased content can deprive substantial revenue from original content creators. Despite this recent surge of malicious use of generative AI, there are few academic publications that research this threat. In this article, we demonstrate the ability of pattern-based similarity detection for AI paraphrased news recognition. We propose an algorithmic scheme, which is not limited to detect whether an article is an AI paraphrase, but, more importantly, to identify that the source of infringement is the ChatGPT. The proposed method is tested with a benchmark dataset specifically created for this task that incorporates real articles from BBC, incorporating a total of 2,224 articles across five different news categories, as well as 2,224 paraphrased articles created with ChatGPT. Results show that our pattern similarity-based method, that makes no use of deep learning, can detect ChatGPT assisted paraphrased articles at percentages 96.23% for accuracy, 96.25% for precision, 96.21% for sensitivity, 96.25% for specificity and 96.23% for F1 score. 

---
# Traversal Verification for Speculative Tree Decoding 

**Authors**: Yepeng Weng, Qiao Hu, Xujie Chen, Li Liu, Dianwen Mei, Huishi Qiu, Jiang Tian, Zhongchao Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12398)  

**Abstract**: Speculative decoding is a promising approach for accelerating large language models. The primary idea is to use a lightweight draft model to speculate the output of the target model for multiple subsequent timesteps, and then verify them in parallel to determine whether the drafted tokens should be accepted or rejected. To enhance acceptance rates, existing frameworks typically construct token trees containing multiple candidates in each timestep. However, their reliance on token-level verification mechanisms introduces two critical limitations: First, the probability distribution of a sequence differs from that of individual tokens, leading to suboptimal acceptance length. Second, current verification schemes begin from the root node and proceed layer by layer in a top-down manner. Once a parent node is rejected, all its child nodes should be discarded, resulting in inefficient utilization of speculative candidates. This paper introduces Traversal Verification, a novel speculative decoding algorithm that fundamentally rethinks the verification paradigm through leaf-to-root traversal. Our approach considers the acceptance of the entire token sequence from the current node to the root, and preserves potentially valid subsequences that would be prematurely discarded by existing methods. We theoretically prove that the probability distribution obtained through Traversal Verification is identical to that of the target model, guaranteeing lossless inference while achieving substantial acceleration gains. Experimental results across different large language models and multiple tasks show that our method consistently improves acceptance length and throughput over existing methods 

---
# SLOT: Sample-specific Language Model Optimization at Test-time 

**Authors**: Yang Hu, Xingyu Zhang, Xueji Fang, Zhiyang Chen, Xiao Wang, Huatian Zhang, Guojun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12392)  

**Abstract**: We propose SLOT (Sample-specific Language Model Optimization at Test-time), a novel and parameter-efficient test-time inference approach that enhances a language model's ability to more accurately respond to individual prompts. Existing Large Language Models (LLMs) often struggle with complex instructions, leading to poor performances on those not well represented among general samples. To address this, SLOT conducts few optimization steps at test-time to update a light-weight sample-specific parameter vector. It is added to the final hidden layer before the output head, and enables efficient adaptation by caching the last layer features during per-sample optimization. By minimizing the cross-entropy loss on the input prompt only, SLOT helps the model better aligned with and follow each given instruction. In experiments, we demonstrate that our method outperforms the compared models across multiple benchmarks and LLMs. For example, Qwen2.5-7B with SLOT achieves an accuracy gain of 8.6% on GSM8K from 57.54% to 66.19%, while DeepSeek-R1-Distill-Llama-70B with SLOT achieves a SOTA accuracy of 68.69% on GPQA among 70B-level models. Our code is available at this https URL. 

---
# From n-gram to Attention: How Model Architectures Learn and Propagate Bias in Language Modeling 

**Authors**: Mohsinul Kabir, Tasfia Tahsin, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12381)  

**Abstract**: Current research on bias in language models (LMs) predominantly focuses on data quality, with significantly less attention paid to model architecture and temporal influences of data. Even more critically, few studies systematically investigate the origins of bias. We propose a methodology grounded in comparative behavioral theory to interpret the complex interaction between training data and model architecture in bias propagation during language modeling. Building on recent work that relates transformers to n-gram LMs, we evaluate how data, model design choices, and temporal dynamics affect bias propagation. Our findings reveal that: (1) n-gram LMs are highly sensitive to context window size in bias propagation, while transformers demonstrate architectural robustness; (2) the temporal provenance of training data significantly affects bias; and (3) different model architectures respond differentially to controlled bias injection, with certain biases (e.g. sexual orientation) being disproportionately amplified. As language models become ubiquitous, our findings highlight the need for a holistic approach -- tracing bias to its origins across both data and model dimensions, not just symptoms, to mitigate harm. 

---
# CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement 

**Authors**: Gauri Kholkar, Ratinder Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2505.12368)  

**Abstract**: Prompt injection remains a major security risk for large language models. However, the efficacy of existing guardrail models in context-aware settings remains underexplored, as they often rely on static attack benchmarks. Additionally, they have over-defense tendencies. We introduce CAPTURE, a novel context-aware benchmark assessing both attack detection and over-defense tendencies with minimal in-domain examples. Our experiments reveal that current prompt injection guardrail models suffer from high false negatives in adversarial cases and excessive false positives in benign scenarios, highlighting critical limitations. 

---
# Wisdom from Diversity: Bias Mitigation Through Hybrid Human-LLM Crowds 

**Authors**: Axel Abels, Tom Lenaerts  

**Link**: [PDF](https://arxiv.org/pdf/2505.12349)  

**Abstract**: Despite their performance, large language models (LLMs) can inadvertently perpetuate biases found in the data they are trained on. By analyzing LLM responses to bias-eliciting headlines, we find that these models often mirror human biases. To address this, we explore crowd-based strategies for mitigating bias through response aggregation. We first demonstrate that simply averaging responses from multiple LLMs, intended to leverage the "wisdom of the crowd", can exacerbate existing biases due to the limited diversity within LLM crowds. In contrast, we show that locally weighted aggregation methods more effectively leverage the wisdom of the LLM crowd, achieving both bias mitigation and improved accuracy. Finally, recognizing the complementary strengths of LLMs (accuracy) and humans (diversity), we demonstrate that hybrid crowds containing both significantly enhance performance and further reduce biases across ethnic and gender-related contexts. 

---
# UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models 

**Authors**: Qizhou Chen, Dakan Wang, Taolin Zhang, Zaoming Yan, Chengsong You, Chengyu Wang, Xiaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2505.12345)  

**Abstract**: Model editing aims to enhance the accuracy and reliability of large language models (LLMs) by efficiently adjusting their internal parameters. Currently, most LLM editing datasets are confined to narrow knowledge domains and cover a limited range of editing evaluation. They often overlook the broad scope of editing demands and the diversity of ripple effects resulting from edits. In this context, we introduce UniEdit, a unified benchmark for LLM editing grounded in open-domain knowledge. First, we construct editing samples by selecting entities from 25 common domains across five major categories, utilizing the extensive triple knowledge available in open-domain knowledge graphs to ensure comprehensive coverage of the knowledge domains. To address the issues of generality and locality in editing, we design an Neighborhood Multi-hop Chain Sampling (NMCS) algorithm to sample subgraphs based on a given knowledge piece to entail comprehensive ripple effects to evaluate. Finally, we employ proprietary LLMs to convert the sampled knowledge subgraphs into natural language text, guaranteeing grammatical accuracy and syntactical diversity. Extensive statistical analysis confirms the scale, comprehensiveness, and diversity of our UniEdit benchmark. We conduct comprehensive experiments across multiple LLMs and editors, analyzing their performance to highlight strengths and weaknesses in editing across open knowledge domains and various evaluation criteria, thereby offering valuable insights for future research endeavors. 

---
# LLMSR@XLLM25: An Empirical Study of LLM for Structural Reasoning 

**Authors**: Xinye Li, Mingqi Wan, Dianbo Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.12328)  

**Abstract**: We present Team asdfo123's submission to the LLMSR@XLLM25 shared task, which evaluates large language models on producing fine-grained, controllable, and interpretable reasoning processes. Systems must extract all problem conditions, decompose a chain of thought into statement-evidence pairs, and verify the logical validity of each pair. Leveraging only the off-the-shelf Meta-Llama-3-8B-Instruct, we craft a concise few-shot, multi-turn prompt that first enumerates all conditions and then guides the model to label, cite, and adjudicate every reasoning step. A lightweight post-processor based on regular expressions normalises spans and enforces the official JSON schema. Without fine-tuning, external retrieval, or ensembling, our method ranks 5th overall, achieving macro F1 scores on par with substantially more complex and resource-consuming pipelines. We conclude by analysing the strengths and limitations of our approach and outlining directions for future research in structural reasoning with LLMs. Our code is available at this https URL. 

---
# ExpertSteer: Intervening in LLMs through Expert Knowledge 

**Authors**: Weixuan Wang, Minghao Wu, Barry Haddow, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2505.12313)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities across various tasks, yet guiding them to follow desired behaviours during inference remains a significant challenge. Activation steering offers a promising method to control the generation process of LLMs by modifying their internal activations. However, existing methods commonly intervene in the model's behaviour using steering vectors generated by the model itself, which constrains their effectiveness to that specific model and excludes the possibility of leveraging powerful external expert models for steering. To address these limitations, we propose ExpertSteer, a novel approach that leverages arbitrary specialized expert models to generate steering vectors, enabling intervention in any LLMs. ExpertSteer transfers the knowledge from an expert model to a target LLM through a cohesive four-step process: first aligning representation dimensions with auto-encoders to enable cross-model transfer, then identifying intervention layer pairs based on mutual information analysis, next generating steering vectors from the expert model using Recursive Feature Machines, and finally applying these vectors on the identified layers during inference to selectively guide the target LLM without updating model parameters. We conduct comprehensive experiments using three LLMs on 15 popular benchmarks across four distinct domains. Experiments demonstrate that ExpertSteer significantly outperforms established baselines across diverse tasks at minimal cost. 

---
# Bidirectional LMs are Better Knowledge Memorizers? A Benchmark for Real-world Knowledge Injection 

**Authors**: Yuwei Zhang, Wenhao Yu, Shangbin Feng, Yifan Zhu, Letian Peng, Jayanth Srinivasa, Gaowen Liu, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12306)  

**Abstract**: Despite significant advances in large language models (LLMs), their knowledge memorization capabilities remain underexplored, due to the lack of standardized and high-quality test ground. In this paper, we introduce a novel, real-world and large-scale knowledge injection benchmark that evolves continuously over time without requiring human intervention. Specifically, we propose WikiDYK, which leverages recently-added and human-written facts from Wikipedia's "Did You Know..." entries. These entries are carefully selected by expert Wikipedia editors based on criteria such as verifiability and clarity. Each entry is converted into multiple question-answer pairs spanning diverse task formats from easy cloze prompts to complex multi-hop questions. WikiDYK contains 12,290 facts and 77,180 questions, which is also seamlessly extensible with future updates from Wikipedia editors. Extensive experiments using continued pre-training reveal a surprising insight: despite their prevalence in modern LLMs, Causal Language Models (CLMs) demonstrate significantly weaker knowledge memorization capabilities compared to Bidirectional Language Models (BiLMs), exhibiting a 23% lower accuracy in terms of reliability. To compensate for the smaller scales of current BiLMs, we introduce a modular collaborative framework utilizing ensembles of BiLMs as external knowledge repositories to integrate with LLMs. Experiment shows that our framework further improves the reliability accuracy by up to 29.1%. 

---
# HBO: Hierarchical Balancing Optimization for Fine-Tuning Large Language Models 

**Authors**: Weixuan Wang, Minghao Wu, Barry Haddow, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2505.12300)  

**Abstract**: Fine-tuning large language models (LLMs) on a mixture of diverse datasets poses challenges due to data imbalance and heterogeneity. Existing methods often address these issues across datasets (globally) but overlook the imbalance and heterogeneity within individual datasets (locally), which limits their effectiveness. We introduce Hierarchical Balancing Optimization (HBO), a novel method that enables LLMs to autonomously adjust data allocation during fine-tuning both across datasets (globally) and within each individual dataset (locally). HBO employs a bilevel optimization strategy with two types of actors: a Global Actor, which balances data sampling across different subsets of the training mixture, and several Local Actors, which optimizes data usage within each subset based on difficulty levels. These actors are guided by reward functions derived from the LLM's training state, which measure learning progress and relative performance improvement. We evaluate HBO on three LLM backbones across nine diverse tasks in multilingual and multitask setups. Results show that HBO consistently outperforms existing baselines, achieving significant accuracy gains. Our in-depth analysis further demonstrates that both the global actor and local actors of HBO effectively adjust data usage during fine-tuning. HBO provides a comprehensive solution to the challenges of data imbalance and heterogeneity in LLM fine-tuning, enabling more effective training across diverse datasets. 

---
# Enhance Mobile Agents Thinking Process Via Iterative Preference Learning 

**Authors**: Kun Huang, Weikai Xu, Yuxuan Liu, Quandong Wang, Pengzhi Gao, Wei Liu, Jian Luan, Bin Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.12299)  

**Abstract**: The Chain of Action-Planning Thoughts (CoaT) paradigm has been shown to improve the reasoning performance of VLM-based mobile agents in GUI tasks. However, the scarcity of diverse CoaT trajectories limits the expressiveness and generalization ability of such agents. While self-training is commonly employed to address data scarcity, existing approaches either overlook the correctness of intermediate reasoning steps or depend on expensive process-level annotations to construct process reward models (PRM). To address the above problems, we propose an Iterative Preference Learning (IPL) that constructs a CoaT-tree through interative sampling, scores leaf nodes using rule-based reward, and backpropagates feedback to derive Thinking-level Direct Preference Optimization (T-DPO) pairs. To prevent overfitting during warm-up supervised fine-tuning, we further introduce a three-stage instruction evolution, which leverages GPT-4o to generate diverse Q\&A pairs based on real mobile UI screenshots, enhancing both generality and layout understanding. Experiments on three standard Mobile GUI-agent benchmarks demonstrate that our agent MobileIPL outperforms strong baselines, including continual pretraining models such as OS-ATLAS and UI-TARS. It achieves state-of-the-art performance across three standard Mobile GUI-Agents benchmarks and shows strong generalization to out-of-domain scenarios. 

---
# The Tower of Babel Revisited: Multilingual Jailbreak Prompts on Closed-Source Large Language Models 

**Authors**: Linghan Huang, Haolin Jin, Zhaoge Bi, Pengyue Yang, Peizhou Zhao, Taozhao Chen, Xiongfei Wu, Lei Ma, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12287)  

**Abstract**: Large language models (LLMs) have seen widespread applications across various domains, yet remain vulnerable to adversarial prompt injections. While most existing research on jailbreak attacks and hallucination phenomena has focused primarily on open-source models, we investigate the frontier of closed-source LLMs under multilingual attack scenarios. We present a first-of-its-kind integrated adversarial framework that leverages diverse attack techniques to systematically evaluate frontier proprietary solutions, including GPT-4o, DeepSeek-R1, Gemini-1.5-Pro, and Qwen-Max. Our evaluation spans six categories of security contents in both English and Chinese, generating 38,400 responses across 32 types of jailbreak attacks. Attack success rate (ASR) is utilized as the quantitative metric to assess performance from three dimensions: prompt design, model architecture, and language environment. Our findings suggest that Qwen-Max is the most vulnerable, while GPT-4o shows the strongest defense. Notably, prompts in Chinese consistently yield higher ASRs than their English counterparts, and our novel Two-Sides attack technique proves to be the most effective across all models. This work highlights a dire need for language-aware alignment and robust cross-lingual defenses in LLMs, and we hope it will inspire researchers, developers, and policymakers toward more robust and inclusive AI systems. 

---
# LLM-Based Evaluation of Low-Resource Machine Translation: A Reference-less Dialect Guided Approach with a Refined Sylheti-English Benchmark 

**Authors**: Md. Atiqur Rahman, Sabrina Islam, Mushfiqul Haque Omi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12273)  

**Abstract**: Evaluating machine translation (MT) for low-resource languages poses a persistent challenge, primarily due to the limited availability of high quality reference translations. This issue is further exacerbated in languages with multiple dialects, where linguistic diversity and data scarcity hinder robust evaluation. Large Language Models (LLMs) present a promising solution through reference-free evaluation techniques; however, their effectiveness diminishes in the absence of dialect-specific context and tailored guidance. In this work, we propose a comprehensive framework that enhances LLM-based MT evaluation using a dialect guided approach. We extend the ONUBAD dataset by incorporating Sylheti-English sentence pairs, corresponding machine translations, and Direct Assessment (DA) scores annotated by native speakers. To address the vocabulary gap, we augment the tokenizer vocabulary with dialect-specific terms. We further introduce a regression head to enable scalar score prediction and design a dialect-guided (DG) prompting strategy. Our evaluation across multiple LLMs shows that the proposed pipeline consistently outperforms existing methods, achieving the highest gain of +0.1083 in Spearman correlation, along with improvements across other evaluation settings. The dataset and the code are available at this https URL. 

---
# $K$-MSHC: Unmasking Minimally Sufficient Head Circuits in Large Language Models with Experiments on Syntactic Classification Tasks 

**Authors**: Pratim Chowdhary  

**Link**: [PDF](https://arxiv.org/pdf/2505.12268)  

**Abstract**: Understanding which neural components drive specific capabilities in mid-sized language models ($\leq$10B parameters) remains a key challenge. We introduce the $(\bm{K}, \epsilon)$-Minimum Sufficient Head Circuit ($K$-MSHC), a methodology to identify minimal sets of attention heads crucial for classification tasks as well as Search-K-MSHC, an efficient algorithm for discovering these circuits. Applying our Search-K-MSHC algorithm to Gemma-9B, we analyze three syntactic task families: grammar acceptability, arithmetic verification, and arithmetic word problems. Our findings reveal distinct task-specific head circuits, with grammar tasks predominantly utilizing early layers, word problems showing pronounced activity in both shallow and deep regions, and arithmetic verification demonstrating a more distributed pattern across the network. We discover non-linear circuit overlap patterns, where different task pairs share computational components at varying levels of importance. While grammar and arithmetic share many "weak" heads, arithmetic and word problems share more consistently critical "strong" heads. Importantly, we find that each task maintains dedicated "super-heads" with minimal cross-task overlap, suggesting that syntactic and numerical competencies emerge from specialized yet partially reusable head circuits. 

---
# Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation 

**Authors**: Chengwei Qin, Wenxuan Zhou, Karthik Abinav Sankararaman, Nanshu Wang, Tengyu Xu, Alexander Radovic, Eryk Helenowski, Arya Talebzadeh, Aditya Tayade, Sinong Wang, Shafiq Joty, Han Fang, Hao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.12265)  

**Abstract**: Hallucination, the generation of factually incorrect information, remains a significant challenge for large language models (LLMs), especially in open-domain long-form generation. Existing approaches for detecting hallucination in long-form tasks either focus on limited domains or rely heavily on external fact-checking tools, which may not always be available.
In this work, we systematically investigate reference-free hallucination detection in open-domain long-form responses. Our findings reveal that internal states (e.g., model's output probability and entropy) alone are insufficient for reliably (i.e., better than random guessing) distinguishing between factual and hallucinated content. To enhance detection, we explore various existing approaches, including prompting-based methods, probing, and fine-tuning, with fine-tuning proving the most effective. To further improve the accuracy, we introduce a new paradigm, named RATE-FT, that augments fine-tuning with an auxiliary task for the model to jointly learn with the main task of hallucination detection. With extensive experiments and analysis using a variety of model families & datasets, we demonstrate the effectiveness and generalizability of our method, e.g., +3% over general fine-tuning methods on LongFact. 

---
# Teach2Eval: An Indirect Evaluation Method for LLM by Judging How It Teaches 

**Authors**: Yuhang Zhou, Xutian Chen, Yixin Cao, Yuchen Ni, Yu He, Siyu Tian, Xiang Liu, Jian Zhang, Chuanjun Ji, Guangnan Ye, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12259)  

**Abstract**: Recent progress in large language models (LLMs) has outpaced the development of effective evaluation methods. Traditional benchmarks rely on task-specific metrics and static datasets, which often suffer from fairness issues, limited scalability, and contamination risks. In this paper, we introduce Teach2Eval, an indirect evaluation framework inspired by the Feynman Technique. Instead of directly testing LLMs on predefined tasks, our method evaluates a model's multiple abilities to teach weaker student models to perform tasks effectively. By converting open-ended tasks into standardized multiple-choice questions (MCQs) through teacher-generated feedback, Teach2Eval enables scalable, automated, and multi-dimensional assessment. Our approach not only avoids data leakage and memorization but also captures a broad range of cognitive abilities that are orthogonal to current benchmarks. Experimental results across 26 leading LLMs show strong alignment with existing human and model-based dynamic rankings, while offering additional interpretability for training guidance. 

---
# Not All Documents Are What You Need for Extracting Instruction Tuning Data 

**Authors**: Chi Zhang, Huaping Zhong, Hongtao Li, Chengliang Chai, Jiawei Hong, Yuhao Deng, Jiacheng Wang, Tian Tan, Yizhou Yan, Jiantao Qiu, Ye Yuan, Guoren Wang, Conghui He, Lei Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12250)  

**Abstract**: Instruction tuning improves the performance of large language models (LLMs), but it heavily relies on high-quality training data. Recently, LLMs have been used to synthesize instruction data using seed question-answer (QA) pairs. However, these synthesized instructions often lack diversity and tend to be similar to the input seeds, limiting their applicability in real-world scenarios. To address this, we propose extracting instruction tuning data from web corpora that contain rich and diverse knowledge. A naive solution is to retrieve domain-specific documents and extract all QA pairs from them, but this faces two key challenges: (1) extracting all QA pairs using LLMs is prohibitively expensive, and (2) many extracted QA pairs may be irrelevant to the downstream tasks, potentially degrading model performance. To tackle these issues, we introduce EQUAL, an effective and scalable data extraction framework that iteratively alternates between document selection and high-quality QA pair extraction to enhance instruction tuning. EQUAL first clusters the document corpus based on embeddings derived from contrastive learning, then uses a multi-armed bandit strategy to efficiently identify clusters that are likely to contain valuable QA pairs. This iterative approach significantly reduces computational cost while boosting model performance. Experiments on AutoMathText and StackOverflow across four downstream tasks show that EQUAL reduces computational costs by 5-10x and improves accuracy by 2.5 percent on LLaMA-3.1-8B and Mistral-7B 

---
# Distribution Prompting: Understanding the Expressivity of Language Models Through the Next-Token Distributions They Can Produce 

**Authors**: Haojin Wang, Zining Zhu, Freda Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12244)  

**Abstract**: Autoregressive neural language models (LMs) generate a probability distribution over tokens at each time step given a prompt. In this work, we attempt to systematically understand the probability distributions that LMs can produce, showing that some distributions are significantly harder to elicit than others. Specifically, for any target next-token distribution over the vocabulary, we attempt to find a prompt that induces the LM to output a distribution as close as possible to the target, using either soft or hard gradient-based prompt tuning. We find that (1) in general, distributions with very low or very high entropy are easier to approximate than those with moderate entropy; (2) among distributions with the same entropy, those containing ''outlier tokens'' are easier to approximate; (3) target distributions generated by LMs -- even LMs with different tokenizers -- are easier to approximate than randomly chosen targets. These results offer insights into the expressiveness of LMs and the challenges of using them as probability distribution proposers. 

---
# PANORAMA: A synthetic PII-laced dataset for studying sensitive data memorization in LLMs 

**Authors**: Sriram Selvam, Anneswa Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2505.12238)  

**Abstract**: The memorization of sensitive and personally identifiable information (PII) by large language models (LLMs) poses growing privacy risks as models scale and are increasingly deployed in real-world applications. Existing efforts to study sensitive and PII data memorization and develop mitigation strategies are hampered by the absence of comprehensive, realistic, and ethically sourced datasets reflecting the diversity of sensitive information found on the web. We introduce PANORAMA - Profile-based Assemblage for Naturalistic Online Representation and Attribute Memorization Analysis, a large-scale synthetic corpus of 384,789 samples derived from 9,674 synthetic profiles designed to closely emulate the distribution, variety, and context of PII and sensitive data as it naturally occurs in online environments. Our data generation pipeline begins with the construction of internally consistent, multi-attribute human profiles using constrained selection to reflect real-world demographics such as education, health attributes, financial status, etc. Using a combination of zero-shot prompting and OpenAI o3-mini, we generate diverse content types - including wiki-style articles, social media posts, forum discussions, online reviews, comments, and marketplace listings - each embedding realistic, contextually appropriate PII and other sensitive information. We validate the utility of PANORAMA by fine-tuning the Mistral-7B model on 1x, 5x, 10x, and 25x data replication rates with a subset of data and measure PII memorization rates - revealing not only consistent increases with repetition but also variation across content types, highlighting PANORAMA's ability to model how memorization risks differ by context. Our dataset and code are publicly available, providing a much-needed resource for privacy risk assessment, model auditing, and the development of privacy-preserving LLMs. 

---
# Bridging Generative and Discriminative Learning: Few-Shot Relation Extraction via Two-Stage Knowledge-Guided Pre-training 

**Authors**: Quanjiang Guo, Jinchuan Zhang, Sijie Wang, Ling Tian, Zhao Kang, Bin Yan, Weidong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12236)  

**Abstract**: Few-Shot Relation Extraction (FSRE) remains a challenging task due to the scarcity of annotated data and the limited generalization capabilities of existing models. Although large language models (LLMs) have demonstrated potential in FSRE through in-context learning (ICL), their general-purpose training objectives often result in suboptimal performance for task-specific relation extraction. To overcome these challenges, we propose TKRE (Two-Stage Knowledge-Guided Pre-training for Relation Extraction), a novel framework that synergistically integrates LLMs with traditional relation extraction models, bridging generative and discriminative learning paradigms. TKRE introduces two key innovations: (1) leveraging LLMs to generate explanation-driven knowledge and schema-constrained synthetic data, addressing the issue of data scarcity; and (2) a two-stage pre-training strategy combining Masked Span Language Modeling (MSLM) and Span-Level Contrastive Learning (SCL) to enhance relational reasoning and generalization. Together, these components enable TKRE to effectively tackle FSRE tasks. Comprehensive experiments on benchmark datasets demonstrate the efficacy of TKRE, achieving new state-of-the-art performance in FSRE and underscoring its potential for broader application in low-resource scenarios. \footnote{The code and data are released on this https URL. 

---
# Examining Linguistic Shifts in Academic Writing Before and After the Launch of ChatGPT: A Study on Preprint Papers 

**Authors**: Tong Bao, Yi Zhao, Jin Mao, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12218)  

**Abstract**: Large Language Models (LLMs), such as ChatGPT, have prompted academic concerns about their impact on academic writing. Existing studies have primarily examined LLM usage in academic writing through quantitative approaches, such as word frequency statistics and probability-based analyses. However, few have systematically examined the potential impact of LLMs on the linguistic characteristics of academic writing. To address this gap, we conducted a large-scale analysis across 823,798 abstracts published in last decade from arXiv dataset. Through the linguistic analysis of features such as the frequency of LLM-preferred words, lexical complexity, syntactic complexity, cohesion, readability and sentiment, the results indicate a significant increase in the proportion of LLM-preferred words in abstracts, revealing the widespread influence of LLMs on academic writing. Additionally, we observed an increase in lexical complexity and sentiment in the abstracts, but a decrease in syntactic complexity, suggesting that LLMs introduce more new vocabulary and simplify sentence structure. However, the significant decrease in cohesion and readability indicates that abstracts have fewer connecting words and are becoming more difficult to read. Moreover, our analysis reveals that scholars with weaker English proficiency were more likely to use the LLMs for academic writing, and focused on improving the overall logic and fluency of the abstracts. Finally, at discipline level, we found that scholars in Computer Science showed more pronounced changes in writing style, while the changes in Mathematics were minimal. 

---
# One-for-All Pruning: A Universal Model for Customized Compression of Large Language Models 

**Authors**: Rongguang Ye, Ming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12216)  

**Abstract**: Existing pruning methods for large language models (LLMs) focus on achieving high compression rates while maintaining model performance. Although these methods have demonstrated satisfactory performance in handling a single user's compression request, their processing time increases linearly with the number of requests, making them inefficient for real-world scenarios with multiple simultaneous requests. To address this limitation, we propose a Univeral Model for Customized Compression (UniCuCo) for LLMs, which introduces a StratNet that learns to map arbitrary requests to their optimal pruning strategy. The challenge in training StratNet lies in the high computational cost of evaluating pruning strategies and the non-differentiable nature of the pruning process, which hinders gradient backpropagation for StratNet updates. To overcome these challenges, we leverage a Gaussian process to approximate the evaluation process. Since the gradient of the Gaussian process is computable, we can use it to approximate the gradient of the non-differentiable pruning process, thereby enabling StratNet updates. Experimental results show that UniCuCo is 28 times faster than baselines in processing 64 requests, while maintaining comparable accuracy to baselines. 

---
# GMSA: Enhancing Context Compression via Group Merging and Layer Semantic Alignment 

**Authors**: Jiwei Tang, Zhicheng Zhang, Shunlong Wu, Jingheng Ye, Lichen Bai, Zitai Wang, Tingwei Lu, Jiaqi Chen, Lin Hai, Hai-Tao Zheng, Hong-Gee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.12215)  

**Abstract**: Large language models (LLMs) have achieved impressive performance in a variety of natural language processing (NLP) tasks. However, when applied to long-context scenarios, they face two challenges, i.e., low computational efficiency and much redundant information. This paper introduces GMSA, a context compression framework based on the encoder-decoder architecture, which addresses these challenges by reducing input sequence length and redundant information. Structurally, GMSA has two key components: Group Merging and Layer Semantic Alignment (LSA). Group merging is used to effectively and efficiently extract summary vectors from the original context. Layer semantic alignment, on the other hand, aligns the high-level summary vectors with the low-level primary input semantics, thus bridging the semantic gap between different layers. In the training process, GMSA first learns soft tokens that contain complete semantics through autoencoder training. To furtherly adapt GMSA to downstream tasks, we propose Knowledge Extraction Fine-tuning (KEFT) to extract knowledge from the soft tokens for downstream tasks. We train GMSA by randomly sampling the compression rate for each sample in the dataset. Under this condition, GMSA not only significantly outperforms the traditional compression paradigm in context restoration but also achieves stable and significantly faster convergence with only a few encoder layers. In downstream question-answering (QA) tasks, GMSA can achieve approximately a 2x speedup in end-to-end inference while outperforming both the original input prompts and various state-of-the-art (SOTA) methods by a large margin. 

---
# Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning 

**Authors**: Shaobo Wang, Ziming Wang, Xiangqi Jin, Jize Wang, Jiajun Zhang, Kaixin Li, Zichen Wen, Zhong Li, Conghui He, Xuming Hu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12212)  

**Abstract**: Fine-tuning large language models (LLMs) on task-specific data is essential for their effective deployment. As dataset sizes grow, efficiently selecting optimal subsets for training becomes crucial to balancing performance and computational costs. Traditional data selection methods often require fine-tuning a scoring model on the target dataset, which is time-consuming and resource-intensive, or rely on heuristics that fail to fully leverage the model's predictive capabilities. To address these challenges, we propose Data Whisperer, an efficient, training-free, attention-based method that leverages few-shot in-context learning with the model to be fine-tuned. Comprehensive evaluations were conducted on both raw and synthetic datasets across diverse tasks and models. Notably, Data Whisperer achieves superior performance compared to the full GSM8K dataset on the Llama-3-8B-Instruct model, using just 10% of the data, and outperforms existing methods with a 3.1-point improvement and a 7.4$\times$ speedup. 

---
# How Reliable is Multilingual LLM-as-a-Judge? 

**Authors**: Xiyan Fu, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12201)  

**Abstract**: LLM-as-a-Judge has emerged as a popular evaluation strategy, where advanced large language models assess generation results in alignment with human instructions. While these models serve as a promising alternative to human annotators, their reliability in multilingual evaluation remains uncertain. To bridge this gap, we conduct a comprehensive analysis of multilingual LLM-as-a-Judge. Specifically, we evaluate five models from different model families across five diverse tasks involving 25 languages. Our findings reveal that LLMs struggle to achieve consistent judgment results across languages, with an average Fleiss' Kappa of approximately 0.3, and some models performing even worse. To investigate the cause of inconsistency, we analyze various influencing factors. We observe that consistency varies significantly across languages, with particularly poor performance in low-resource languages. Additionally, we find that neither training on multilingual data nor increasing model scale directly improves judgment consistency. These findings suggest that LLMs are not yet reliable for evaluating multilingual predictions. We finally propose an ensemble strategy which improves the consistency of the multilingual judge in real-world applications. 

---
# Vectors from Larger Language Models Predict Human Reading Time and fMRI Data More Poorly when Dimensionality Expansion is Controlled 

**Authors**: Yi-Chien Lin, Hongao Zhu, William Schuler  

**Link**: [PDF](https://arxiv.org/pdf/2505.12196)  

**Abstract**: The impressive linguistic abilities of large language models (LLMs) have recommended them as models of human sentence processing, with some conjecturing a positive 'quality-power' relationship (Wilcox et al., 2023), in which language models' (LMs') fit to psychometric data continues to improve as their ability to predict words in context increases. This is important because it suggests that elements of LLM architecture, such as veridical attention to context and a unique objective of predicting upcoming words, reflect the architecture of the human sentence processing faculty, and that any inadequacies in predicting human reading time and brain imaging data may be attributed to insufficient model complexity, which recedes as larger models become available. Recent studies (Oh and Schuler, 2023) have shown this scaling inverts after a point, as LMs become excessively large and accurate, when word prediction probability (as information-theoretic surprisal) is used as a predictor. Other studies propose the use of entire vectors from differently sized LLMs, still showing positive scaling (Schrimpf et al., 2021), casting doubt on the value of surprisal as a predictor, but do not control for the larger number of predictors in vectors from larger LMs. This study evaluates LLM scaling using entire LLM vectors, while controlling for the larger number of predictors in vectors from larger LLMs. Results show that inverse scaling obtains, suggesting that inadequacies in predicting human reading time and brain imaging data may be due to substantial misalignment between LLMs and human sentence processing, which worsens as larger models are used. 

---
# Decoding the Mind of Large Language Models: A Quantitative Evaluation of Ideology and Biases 

**Authors**: Manari Hirose, Masato Uchida  

**Link**: [PDF](https://arxiv.org/pdf/2505.12183)  

**Abstract**: The widespread integration of Large Language Models (LLMs) across various sectors has highlighted the need for empirical research to understand their biases, thought patterns, and societal implications to ensure ethical and effective use. In this study, we propose a novel framework for evaluating LLMs, focusing on uncovering their ideological biases through a quantitative analysis of 436 binary-choice questions, many of which have no definitive answer. By applying our framework to ChatGPT and Gemini, findings revealed that while LLMs generally maintain consistent opinions on many topics, their ideologies differ across models and languages. Notably, ChatGPT exhibits a tendency to change their opinion to match the questioner's opinion. Both models also exhibited problematic biases, unethical or unfair claims, which might have negative societal impacts. These results underscore the importance of addressing both ideological and ethical considerations when evaluating LLMs. The proposed framework offers a flexible, quantitative method for assessing LLM behavior, providing valuable insights for the development of more socially aligned AI systems. 

---
# Truth Neurons 

**Authors**: Haohang Li, Yupeng Cao, Yangyang Yu, Jordan W. Suchow, Zining Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12182)  

**Abstract**: Despite their remarkable success and deployment across diverse workflows, language models sometimes produce untruthful responses. Our limited understanding of how truthfulness is mechanistically encoded within these models jeopardizes their reliability and safety. In this paper, we propose a method for identifying representations of truthfulness at the neuron level. We show that language models contain truth neurons, which encode truthfulness in a subject-agnostic manner. Experiments conducted across models of varying scales validate the existence of truth neurons, confirming that the encoding of truthfulness at the neuron level is a property shared by many language models. The distribution patterns of truth neurons over layers align with prior findings on the geometry of truthfulness. Selectively suppressing the activations of truth neurons found through the TruthfulQA dataset degrades performance both on TruthfulQA and on other benchmarks, showing that the truthfulness mechanisms are not tied to a specific dataset. Our results offer novel insights into the mechanisms underlying truthfulness in language models and highlight potential directions toward improving their trustworthiness and reliability. 

---
# Emotion Recognition for Low-Resource Turkish: Fine-Tuning BERTurk on TREMO and Testing on Xenophobic Political Discourse 

**Authors**: Darmawan Wicaksono, Hasri Akbar Awal Rozaq, Nevfel Boz  

**Link**: [PDF](https://arxiv.org/pdf/2505.12160)  

**Abstract**: Social media platforms like X (formerly Twitter) play a crucial role in shaping public discourse and societal norms. This study examines the term Sessiz Istila (Silent Invasion) on Turkish social media, highlighting the rise of anti-refugee sentiment amidst the Syrian refugee influx. Using BERTurk and the TREMO dataset, we developed an advanced Emotion Recognition Model (ERM) tailored for Turkish, achieving 92.62% accuracy in categorizing emotions such as happiness, fear, anger, sadness, disgust, and surprise. By applying this model to large-scale X data, the study uncovers emotional nuances in Turkish discourse, contributing to computational social science by advancing sentiment analysis in underrepresented languages and enhancing our understanding of global digital discourse and the unique linguistic challenges of Turkish. The findings underscore the transformative potential of localized NLP tools, with our ERM model offering practical applications for real-time sentiment analysis in Turkish-language contexts. By addressing critical areas, including marketing, public relations, and crisis management, these models facilitate improved decision-making through timely and accurate sentiment tracking. This highlights the significance of advancing research that accounts for regional and linguistic nuances. 

---
# The AI Gap: How Socioeconomic Status Affects Language Technology Interactions 

**Authors**: Elisa Bassignana, Amanda Cercas Curry, Dirk Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2505.12158)  

**Abstract**: Socioeconomic status (SES) fundamentally influences how people interact with each other and more recently, with digital technologies like Large Language Models (LLMs). While previous research has highlighted the interaction between SES and language technology, it was limited by reliance on proxy metrics and synthetic data. We survey 1,000 individuals from diverse socioeconomic backgrounds about their use of language technologies and generative AI, and collect 6,482 prompts from their previous interactions with LLMs. We find systematic differences across SES groups in language technology usage (i.e., frequency, performed tasks), interaction styles, and topics. Higher SES entails a higher level of abstraction, convey requests more concisely, and topics like 'inclusivity' and 'travel'. Lower SES correlates with higher anthropomorphization of LLMs (using ''hello'' and ''thank you'') and more concrete language. Our findings suggest that while generative language technologies are becoming more accessible to everyone, socioeconomic linguistic differences still stratify their use to exacerbate the digital divide. These differences underscore the importance of considering SES in developing language technologies to accommodate varying linguistic needs rooted in socioeconomic factors and limit the AI Gap across SES groups. 

---
# A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings 

**Authors**: Fitsum Gaim, Hoyun Song, Huije Lee, Changgeon Ko, Eui Jun Hwang, Jong C. Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.12116)  

**Abstract**: Content moderation research has recently made significant advances, but still fails to serve the majority of the world's languages due to the lack of resources, leaving millions of vulnerable users to online hostility. This work presents a large-scale human-annotated multi-task benchmark dataset for abusive language detection in Tigrinya social media with joint annotations for three tasks: abusiveness, sentiment, and topic classification. The dataset comprises 13,717 YouTube comments annotated by nine native speakers, collected from 7,373 videos with a total of over 1.2 billion views across 51 channels. We developed an iterative term clustering approach for effective data selection. Recognizing that around 64% of Tigrinya social media content uses Romanized transliterations rather than native Ge'ez script, our dataset accommodates both writing systems to reflect actual language use. We establish strong baselines across the tasks in the benchmark, while leaving significant challenges for future contributions. Our experiments reveal that small, specialized multi-task models outperform the current frontier models in the low-resource setting, achieving up to 86% accuracy (+7 points) in abusiveness detection. We make the resources publicly available to promote research on online safety. 

---
# Improving Fairness in LLMs Through Testing-Time Adversaries 

**Authors**: Isabela Pereira Gregio, Ian Pons, Anna Helena Reali Costa, Artur Jordão  

**Link**: [PDF](https://arxiv.org/pdf/2505.12100)  

**Abstract**: Large Language Models (LLMs) push the bound-aries in natural language processing and generative AI, driving progress across various aspects of modern society. Unfortunately, the pervasive issue of bias in LLMs responses (i.e., predictions) poses a significant and open challenge, hindering their application in tasks involving ethical sensitivity and responsible decision-making. In this work, we propose a straightforward, user-friendly and practical method to mitigate such biases, enhancing the reliability and trustworthiness of LLMs. Our method creates multiple variations of a given sentence by modifying specific attributes and evaluates the corresponding prediction behavior compared to the original, unaltered, prediction/sentence. The idea behind this process is that critical ethical predictions often exhibit notable inconsistencies, indicating the presence of bias. Unlike previous approaches, our method relies solely on forward passes (i.e., testing-time adversaries), eliminating the need for training, fine-tuning, or prior knowledge of the training data distribution. Through extensive experiments on the popular Llama family, we demonstrate the effectiveness of our method in improving various fairness metrics, focusing on the reduction of disparities in how the model treats individuals from different racial groups. Specifically, using standard metrics, we improve the fairness in Llama3 in up to 27 percentage points. Overall, our approach significantly enhances fairness, equity, and reliability in LLM-generated results without parameter tuning or training data modifications, confirming its effectiveness in practical scenarios. We believe our work establishes an important step toward enabling the use of LLMs in tasks that require ethical considerations and responsible decision-making. 

---
# Personalized Author Obfuscation with Large Language Models 

**Authors**: Mohammad Shokri, Sarah Ita Levitan, Rivka Levitan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12090)  

**Abstract**: In this paper, we investigate the efficacy of large language models (LLMs) in obfuscating authorship by paraphrasing and altering writing styles. Rather than adopting a holistic approach that evaluates performance across the entire dataset, we focus on user-wise performance to analyze how obfuscation effectiveness varies across individual authors. While LLMs are generally effective, we observe a bimodal distribution of efficacy, with performance varying significantly across users. To address this, we propose a personalized prompting method that outperforms standard prompting techniques and partially mitigates the bimodality issue. 

---
# Model Merging in Pre-training of Large Language Models 

**Authors**: Yunshui Li, Yiyuan Ma, Shen Yan, Chaoyi Zhang, Jing Liu, Jianqiao Lu, Ziwen Xu, Mengzhao Chen, Minrui Wang, Shiyi Zhan, Jin Ma, Xunhao Lai, Yao Luo, Xingyan Bin, Hongbin Ren, Mingji Han, Wenhao Hao, Bairen Yi, LingJun Liu, Bole Ma, Xiaoying Jia, Zhou Xun, Liang Xiang, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12082)  

**Abstract**: Model merging has emerged as a promising technique for enhancing large language models, though its application in large-scale pre-training remains relatively unexplored. In this paper, we present a comprehensive investigation of model merging techniques during the pre-training process. Through extensive experiments with both dense and Mixture-of-Experts (MoE) architectures ranging from millions to over 100 billion parameters, we demonstrate that merging checkpoints trained with constant learning rates not only achieves significant performance improvements but also enables accurate prediction of annealing behavior. These improvements lead to both more efficient model development and significantly lower training costs. Our detailed ablation studies on merging strategies and hyperparameters provide new insights into the underlying mechanisms while uncovering novel applications. Through comprehensive experimental analysis, we offer the open-source community practical pre-training guidelines for effective model merging. 

---
# Do different prompting methods yield a common task representation in language models? 

**Authors**: Guy Davidson, Todd M. Gureckis, Brenden M. Lake, Adina Williams  

**Link**: [PDF](https://arxiv.org/pdf/2505.12075)  

**Abstract**: Demonstrations and instructions are two primary approaches for prompting language models to perform in-context learning (ICL) tasks. Do identical tasks elicited in different ways result in similar representations of the task? An improved understanding of task representation mechanisms would offer interpretability insights and may aid in steering models. We study this through function vectors, recently proposed as a mechanism to extract few-shot ICL task representations. We generalize function vectors to alternative task presentations, focusing on short textual instruction prompts, and successfully extract instruction function vectors that promote zero-shot task accuracy. We find evidence that demonstration- and instruction-based function vectors leverage different model components, and offer several controls to dissociate their contributions to task performance. Our results suggest that different task presentations do not induce a common task representation but elicit different, partly overlapping mechanisms. Our findings offer principled support to the practice of combining textual instructions and task demonstrations, imply challenges in universally monitoring task inference across presentation forms, and encourage further examinations of LLM task inference mechanisms. 

---
# Historical and psycholinguistic perspectives on morphological productivity: A sketch of an integrative approach 

**Authors**: Harald Baayen, Kristian Berg, Maziyah Mohamed  

**Link**: [PDF](https://arxiv.org/pdf/2505.12071)  

**Abstract**: In this study, we approach morphological productivity from two perspectives: a cognitive-computational perspective, and a diachronic perspective zooming in on an actual speaker, Thomas Mann. For developing the first perspective, we make use of a cognitive computational model of the mental lexicon, the discriminative lexicon model. For computational mappings between form and meaning to be productive, in the sense that novel, previously unencountered words, can be understood and produced, there must be systematicities between the form space and the semantic space. If the relation between form and meaning would be truly arbitrary, a model could memorize form and meaning pairings, but there is no way in which the model would be able to generalize to novel test data. For Finnish nominal inflection, Malay derivation, and English compounding, we explore, using the Discriminative Lexicon Model as a computational tool, to trace differences in the degree to which inflectional and word formation patterns are productive. We show that the DLM tends to associate affix-like sublexical units with the centroids of the embeddings of the words with a given affix. For developing the second perspective, we study how the intake and output of one prolific writer, Thomas Mann, changes over time. We show by means of an examination of what Thomas Mann is likely to have read, and what he wrote, that the rate at which Mann produces novel derived words is extremely low. There are far more novel words in his input than in his output. We show that Thomas Mann is less likely to produce a novel derived word with a given suffix the greater the average distance is of the embeddings of all derived words to the corresponding centroid, and discuss the challenges of using speaker-specific embeddings for low-frequency and novel words. 

---
# Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement 

**Authors**: Peng Ding, Jun Kuang, Zongyu Wang, Xuezhi Cao, Xunliang Cai, Jiajun Chen, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12060)  

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities across various tasks but remain vulnerable to meticulously crafted jailbreak attacks. In this paper, we identify a critical safety gap: while LLMs are adept at detecting jailbreak prompts, they often produce unsafe responses when directly processing these inputs. Inspired by this insight, we propose SAGE (Self-Aware Guard Enhancement), a training-free defense strategy designed to align LLMs' strong safety discrimination performance with their relatively weaker safety generation ability. SAGE consists of two core components: a Discriminative Analysis Module and a Discriminative Response Module, enhancing resilience against sophisticated jailbreak attempts through flexible safety discrimination instructions. Extensive experiments demonstrate SAGE's effectiveness and robustness across various open-source and closed-source LLMs of different sizes and architectures, achieving an average 99% defense success rate against numerous complex and covert jailbreak methods while maintaining helpfulness on general benchmarks. We further conduct mechanistic interpretability analysis through hidden states and attention distributions, revealing the underlying mechanisms of this detection-generation discrepancy. Our work thus contributes to developing future LLMs with coherent safety awareness and generation behavior. Our code and datasets are publicly available at this https URL. 

---
# GenderBench: Evaluation Suite for Gender Biases in LLMs 

**Authors**: Matúš Pikuliak  

**Link**: [PDF](https://arxiv.org/pdf/2505.12054)  

**Abstract**: We present GenderBench -- a comprehensive evaluation suite designed to measure gender biases in LLMs. GenderBench includes 14 probes that quantify 19 gender-related harmful behaviors exhibited by LLMs. We release GenderBench as an open-source and extensible library to improve the reproducibility and robustness of benchmarking across the field. We also publish our evaluation of 12 LLMs. Our measurements reveal consistent patterns in their behavior. We show that LLMs struggle with stereotypical reasoning, equitable gender representation in generated texts, and occasionally also with discriminatory behavior in high-stakes scenarios, such as hiring. 

---
# ABoN: Adaptive Best-of-N Alignment 

**Authors**: Vinod Raman, Hilal Asi, Satyen Kale  

**Link**: [PDF](https://arxiv.org/pdf/2505.12050)  

**Abstract**: Recent advances in test-time alignment methods, such as Best-of-N sampling, offer a simple and effective way to steer language models (LMs) toward preferred behaviors using reward models (RM). However, these approaches can be computationally expensive, especially when applied uniformly across prompts without accounting for differences in alignment difficulty. In this work, we propose a prompt-adaptive strategy for Best-of-N alignment that allocates inference-time compute more efficiently. Motivated by latency concerns, we develop a two-stage algorithm: an initial exploratory phase estimates the reward distribution for each prompt using a small exploration budget, and a second stage adaptively allocates the remaining budget using these estimates. Our method is simple, practical, and compatible with any LM/RM combination. Empirical results on the AlpacaEval dataset for 12 LM/RM pairs and 50 different batches of prompts show that our adaptive strategy consistently outperforms the uniform allocation with the same inference budget. Moreover, our experiments show that our adaptive strategy remains competitive against uniform allocations with 20% larger inference budgets and even improves in performance as the batch size grows. 

---
# MoL for LLMs: Dual-Loss Optimization to Enhance Domain Expertise While Preserving General Capabilities 

**Authors**: Jingxue Chen, Qingkun Tang, Qianchun Lu, Siyuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12043)  

**Abstract**: Although LLMs perform well in general tasks, domain-specific applications suffer from hallucinations and accuracy limitations. CPT approaches encounter two key issues: (1) domain-biased data degrades general language skills, and (2) improper corpus-mixture ratios limit effective adaptation. To address these, we propose a novel framework, Mixture of Losses (MoL), which decouples optimization objectives for domain-specific and general corpora. Specifically, cross-entropy (CE) loss is applied to domain data to ensure knowledge acquisition, while Kullback-Leibler (KL) divergence aligns general-corpus training with the base model's foundational capabilities. This dual-loss architecture preserves universal skills while enhancing domain expertise, avoiding catastrophic forgetting. Empirically, we validate that a 1:1 domain-to-general corpus ratio optimally balances training and overfitting without the need for extensive tuning or resource-intensive experiments. Furthermore, our experiments demonstrate significant performance gains compared to traditional CPT approaches, which often suffer from degradation in general language capabilities; our model achieves 27.9% higher accuracy on the Math-500 benchmark in the non-think reasoning mode, and an impressive 83.3% improvement on the challenging AIME25 subset in the think mode, underscoring the effectiveness of our approach. 

---
# Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method 

**Authors**: Yupei Ren, Xinyi Zhou, Ning Zhang, Shangqing Zhao, Man Lan, Xiaopeng Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.12028)  

**Abstract**: Argument mining has garnered increasing attention over the years, with the recent advancement of Large Language Models (LLMs) further propelling this trend. However, current argument relations remain relatively simplistic and foundational, struggling to capture the full scope of argument information, particularly when it comes to representing complex argument structures in real-world scenarios. To address this limitation, we propose 14 fine-grained relation types from both vertical and horizontal dimensions, thereby capturing the intricate interplay between argument components for a thorough understanding of argument structure. On this basis, we conducted extensive experiments on three tasks: argument component detection, relation prediction, and automated essay grading. Additionally, we explored the impact of writing quality on argument component detection and relation prediction, as well as the connections between discourse relations and argumentative features. The findings highlight the importance of fine-grained argumentative annotations for argumentative writing quality assessment and encourage multi-dimensional argument analysis. 

---
# Unveiling Knowledge Utilization Mechanisms in LLM-based Retrieval-Augmented Generation 

**Authors**: Yuhao Wang, Ruiyang Ren, Yucheng Wang, Wayne Xin Zhao, Jing Liu, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11995)  

**Abstract**: Considering the inherent limitations of parametric knowledge in large language models (LLMs), retrieval-augmented generation (RAG) is widely employed to expand their knowledge scope. Since RAG has shown promise in knowledge-intensive tasks like open-domain question answering, its broader application to complex tasks and intelligent assistants has further advanced its utility. Despite this progress, the underlying knowledge utilization mechanisms of LLM-based RAG remain underexplored. In this paper, we present a systematic investigation of the intrinsic mechanisms by which LLMs integrate internal (parametric) and external (retrieved) knowledge in RAG scenarios. Specially, we employ knowledge stream analysis at the macroscopic level, and investigate the function of individual modules at the microscopic level. Drawing on knowledge streaming analyses, we decompose the knowledge utilization process into four distinct stages within LLM layers: knowledge refinement, knowledge elicitation, knowledge expression, and knowledge contestation. We further demonstrate that the relevance of passages guides the streaming of knowledge through these stages. At the module level, we introduce a new method, knowledge activation probability entropy (KAPE) for neuron identification associated with either internal or external knowledge. By selectively deactivating these neurons, we achieve targeted shifts in the LLM's reliance on one knowledge source over the other. Moreover, we discern complementary roles for multi-head attention and multi-layer perceptron layers during knowledge formation. These insights offer a foundation for improving interpretability and reliability in retrieval-augmented LLMs, paving the way for more robust and transparent generative solutions in knowledge-intensive domains. 

---
# An Annotated Corpus of Arabic Tweets for Hate Speech Analysis 

**Authors**: Md. Rafiul Biswas, Wajdi Zaghouani  

**Link**: [PDF](https://arxiv.org/pdf/2505.11969)  

**Abstract**: Identifying hate speech content in the Arabic language is challenging due to the rich quality of dialectal variations. This study introduces a multilabel hate speech dataset in the Arabic language. We have collected 10000 Arabic tweets and annotated each tweet, whether it contains offensive content or not. If a text contains offensive content, we further classify it into different hate speech targets such as religion, gender, politics, ethnicity, origin, and others. A text can contain either single or multiple targets. Multiple annotators are involved in the data annotation task. We calculated the inter-annotator agreement, which was reported to be 0.86 for offensive content and 0.71 for multiple hate speech targets. Finally, we evaluated the data annotation task by employing a different transformers-based model in which AraBERTv2 outperformed with a micro-F1 score of 0.7865 and an accuracy of 0.786. 

---
# CCNU at SemEval-2025 Task 3: Leveraging Internal and External Knowledge of Large Language Models for Multilingual Hallucination Annotation 

**Authors**: Xu Liu, Guanyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11965)  

**Abstract**: We present the system developed by the Central China Normal University (CCNU) team for the Mu-SHROOM shared task, which focuses on identifying hallucinations in question-answering systems across 14 different languages. Our approach leverages multiple Large Language Models (LLMs) with distinct areas of expertise, employing them in parallel to annotate hallucinations, effectively simulating a crowdsourcing annotation process. Furthermore, each LLM-based annotator integrates both internal and external knowledge related to the input during the annotation process. Using the open-source LLM DeepSeek-V3, our system achieves the top ranking (\#1) for Hindi data and secures a Top-5 position in seven other languages. In this paper, we also discuss unsuccessful approaches explored during our development process and share key insights gained from participating in this shared task. 

---
# EmoHopeSpeech: An Annotated Dataset of Emotions and Hope Speech in English 

**Authors**: Md. Rafiul Biswas, Wajdi Zaghouani  

**Link**: [PDF](https://arxiv.org/pdf/2505.11959)  

**Abstract**: This research introduces a bilingual dataset comprising 23,456 entries for Arabic and 10,036 entries for English, annotated for emotions and hope speech, addressing the scarcity of multi-emotion (Emotion and hope) datasets. The dataset provides comprehensive annotations capturing emotion intensity, complexity, and causes, alongside detailed classifications and subcategories for hope speech. To ensure annotation reliability, Fleiss' Kappa was employed, revealing 0.75-0.85 agreement among annotators both for Arabic and English language. The evaluation metrics (micro-F1-Score=0.67) obtained from the baseline model (i.e., using a machine learning model) validate that the data annotations are worthy. This dataset offers a valuable resource for advancing natural language processing in underrepresented languages, fostering better cross-linguistic analysis of emotions and hope speech. 

---
# Counterspeech the ultimate shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning 

**Authors**: Aswini Kumar Padhi, Anil Bandhakavi, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2505.11958)  

**Abstract**: Counterspeech has proven to be a powerful tool to combat hate speech online. Previous studies have focused on generating counterspeech conditioned only on specific intents (single attributed). However, a holistic approach considering multiple attributes simultaneously can yield more nuanced and effective responses. Here, we introduce HiPPrO, Hierarchical Prefix learning with Preference Optimization, a novel two-stage framework that utilizes the effectiveness of attribute-specific prefix embedding spaces hierarchically optimized during the counterspeech generation process in the first phase. Thereafter, we incorporate both reference and reward-free preference optimization to generate more constructive counterspeech. Furthermore, we extend IntentCONANv2 by annotating all 13,973 counterspeech instances with emotion labels by five annotators. HiPPrO leverages hierarchical prefix optimization to integrate these dual attributes effectively. An extensive evaluation demonstrates that HiPPrO achieves a ~38 % improvement in intent conformity and a ~3 %, ~2 %, ~3 % improvement in Rouge-1, Rouge-2, and Rouge-L, respectively, compared to several baseline models. Human evaluations further substantiate the superiority of our approach, highlighting the enhanced relevance and appropriateness of the generated counterspeech. This work underscores the potential of multi-attribute conditioning in advancing the efficacy of counterspeech generation systems. 

---
# ChartEdit: How Far Are MLLMs From Automating Chart Analysis? Evaluating MLLMs' Capability via Chart Editing 

**Authors**: Xuanle Zhao, Xuexin Liu, Haoyue Yang, Xianzhen Luo, Fanhu Zeng, Jianling Li, Qi Shi, Chi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11935)  

**Abstract**: Although multimodal large language models (MLLMs) show promise in generating chart rendering code, chart editing presents a greater challenge. This difficulty stems from its nature as a labor-intensive task for humans that also demands MLLMs to integrate chart understanding, complex reasoning, and precise intent interpretation. While many MLLMs claim such editing capabilities, current assessments typically rely on limited case studies rather than robust evaluation methodologies, highlighting the urgent need for a comprehensive evaluation framework. In this work, we propose ChartEdit, a new high-quality benchmark designed for chart editing tasks. This benchmark comprises $1,405$ diverse editing instructions applied to $233$ real-world charts, with each instruction-chart instance having been manually annotated and validated for accuracy. Utilizing ChartEdit, we evaluate the performance of 10 mainstream MLLMs across two types of experiments, assessing them at both the code and chart levels. The results suggest that large-scale models can generate code to produce images that partially match the reference images. However, their ability to generate accurate edits according to the instructions remains limited. The state-of-the-art (SOTA) model achieves a score of only $59.96$, highlighting significant challenges in precise modification. In contrast, small-scale models, including chart-domain models, struggle both with following editing instructions and generating overall chart images, underscoring the need for further development in this area. Code is available at this https URL. 

---
# Neuro-Symbolic Query Compiler 

**Authors**: Yuyao Zhang, Zhicheng Dou, Xiaoxi Li, Jiajie Jin, Yongkang Wu, Zhonghua Li, Qi Ye, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11932)  

**Abstract**: Precise recognition of search intent in Retrieval-Augmented Generation (RAG) systems remains a challenging goal, especially under resource constraints and for complex queries with nested structures and dependencies. This paper presents QCompiler, a neuro-symbolic framework inspired by linguistic grammar rules and compiler design, to bridge this gap. It theoretically designs a minimal yet sufficient Backus-Naur Form (BNF) grammar $G[q]$ to formalize complex queries. Unlike previous methods, this grammar maintains completeness while minimizing redundancy. Based on this, QCompiler includes a Query Expression Translator, a Lexical Syntax Parser, and a Recursive Descent Processor to compile queries into Abstract Syntax Trees (ASTs) for execution. The atomicity of the sub-queries in the leaf nodes ensures more precise document retrieval and response generation, significantly improving the RAG system's ability to address complex queries. 

---
# An Explanation of Intrinsic Self-Correction via Linear Representations and Latent Concepts 

**Authors**: Yu-Ting Lee, Hui-Ying Shih, Fu-Chieh Chang, Pei-Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11924)  

**Abstract**: We provide an explanation for the performance gains of intrinsic self-correction, a process where a language model iteratively refines its outputs without external feedback. More precisely, we investigate how prompting induces interpretable changes in hidden states and thus affects the output distributions. We hypothesize that each prompt-induced shift lies in a linear span of some linear representation vectors, naturally separating tokens based on individual concept alignment. Building around this idea, we give a mathematical formulation of self-correction and derive a concentration result for output tokens based on alignment magnitudes. Our experiments on text detoxification with zephyr-7b-sft reveal a substantial gap in the inner products of the prompt-induced shifts and the unembeddings of the top-100 most toxic tokens vs. those of the unembeddings of the bottom-100 least toxic tokens, under toxic instructions. This suggests that self-correction prompts enhance a language model's capability of latent concept recognition. Our analysis offers insights into the underlying mechanism of self-correction by characterizing how prompting works explainably. For reproducibility, our code is available. 

---
# Enhancing Complex Instruction Following for Large Language Models with Mixture-of-Contexts Fine-tuning 

**Authors**: Yuheng Lu, ZiMeng Bai, Caixia Yuan, Huixing Jiang, Xiaojie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11922)  

**Abstract**: Large language models (LLMs) exhibit remarkable capabilities in handling natural language tasks; however, they may struggle to consistently follow complex instructions including those involve multiple constraints. Post-training LLMs using supervised fine-tuning (SFT) is a standard approach to improve their ability to follow instructions. In addressing complex instruction following, existing efforts primarily focus on data-driven methods that synthesize complex instruction-output pairs for SFT. However, insufficient attention allocated to crucial sub-contexts may reduce the effectiveness of SFT. In this work, we propose transforming sequentially structured input instruction into multiple parallel instructions containing subcontexts. To support processing this multi-input, we propose MISO (Multi-Input Single-Output), an extension to currently dominant decoder-only transformer-based LLMs. MISO introduces a mixture-of-contexts paradigm that jointly considers the overall instruction-output alignment and the influence of individual sub-contexts to enhance SFT effectiveness. We apply MISO fine-tuning to complex instructionfollowing datasets and evaluate it with standard LLM inference. Empirical results demonstrate the superiority of MISO as a fine-tuning method for LLMs, both in terms of effectiveness in complex instruction-following scenarios and its potential for training efficiency. 

---
# ELITE: Embedding-Less retrieval with Iterative Text Exploration 

**Authors**: Zhangyu Wang, Siyuan Gao, Rong Zhou, Hao Wang, Li Ning  

**Link**: [PDF](https://arxiv.org/pdf/2505.11908)  

**Abstract**: Large Language Models (LLMs) have achieved impressive progress in natural language processing, but their limited ability to retain long-term context constrains performance on document-level or multi-turn tasks. Retrieval-Augmented Generation (RAG) mitigates this by retrieving relevant information from an external corpus. However, existing RAG systems often rely on embedding-based retrieval trained on corpus-level semantic similarity, which can lead to retrieving content that is semantically similar in form but misaligned with the question's true intent. Furthermore, recent RAG variants construct graph- or hierarchy-based structures to improve retrieval accuracy, resulting in significant computation and storage overhead. In this paper, we propose an embedding-free retrieval framework. Our method leverages the logical inferencing ability of LLMs in retrieval using iterative search space refinement guided by our novel importance measure and extend our retrieval results with logically related information without explicit graph construction. Experiments on long-context QA benchmarks, including NovelQA and Marathon, show that our approach outperforms strong baselines while reducing storage and runtime by over an order of magnitude. 

---
# Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data 

**Authors**: Philipp Christmann, Gerhard Weikum  

**Link**: [PDF](https://arxiv.org/pdf/2505.11900)  

**Abstract**: Question answering over mixed sources, like text and tables, has been advanced by verbalizing all contents and encoding it with a language model. A prominent case of such heterogeneous data is personal information: user devices log vast amounts of data every day, such as calendar entries, workout statistics, shopping records, streaming history, and more. Information needs range from simple look-ups to queries of analytical nature. The challenge is to provide humans with convenient access with small footprint, so that all personal data stays on the user devices. We present ReQAP, a novel method that creates an executable operator tree for a given question, via recursive decomposition. Operators are designed to enable seamless integration of structured and unstructured sources, and the execution of the operator tree yields a traceable answer. We further release the PerQA benchmark, with persona-based data and questions, covering a diverse spectrum of realistic user needs. 

---
# RLAP: A Reinforcement Learning Enhanced Adaptive Planning Framework for Multi-step NLP Task Solving 

**Authors**: Zepeng Ding, Dixuan Wang, Ziqin Luo, Guochao Jiang, Deqing Yang, Jiaqing Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11893)  

**Abstract**: Multi-step planning has been widely employed to enhance the performance of large language models (LLMs) on downstream natural language processing (NLP) tasks, which decomposes the original task into multiple subtasks and guide LLMs to solve them sequentially without additional training. When addressing task instances, existing methods either preset the order of steps or attempt multiple paths at each step. However, these methods overlook instances' linguistic features and rely on the intrinsic planning capabilities of LLMs to evaluate intermediate feedback and then select subtasks, resulting in suboptimal outcomes. To better solve multi-step NLP tasks with LLMs, in this paper we propose a Reinforcement Learning enhanced Adaptive Planning framework (RLAP). In our framework, we model an NLP task as a Markov decision process (MDP) and employ an LLM directly into the environment. In particular, a lightweight Actor model is trained to estimate Q-values for natural language sequences consisting of states and actions through reinforcement learning. Therefore, during sequential planning, the linguistic features of each sequence in the MDP can be taken into account, and the Actor model interacts with the LLM to determine the optimal order of subtasks for each task instance. We apply RLAP on three different types of NLP tasks and conduct extensive experiments on multiple datasets to verify RLAP's effectiveness and robustness. 

---
# Mobile-Bench-v2: A More Realistic and Comprehensive Benchmark for VLM-based Mobile Agents 

**Authors**: Weikai Xu, Zhizheng Jiang, Yuxuan Liu, Wei Liu, Jian Luan, Yuanchun Li, Yunxin Liu, Bin Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.11891)  

**Abstract**: VLM-based mobile agents are increasingly popular due to their capabilities to interact with smartphone GUIs and XML-structured texts and to complete daily tasks. However, existing online benchmarks struggle with obtaining stable reward signals due to dynamic environmental changes. Offline benchmarks evaluate the agents through single-path trajectories, which stands in contrast to the inherently multi-solution characteristics of GUI tasks. Additionally, both types of benchmarks fail to assess whether mobile agents can handle noise or engage in proactive interactions due to a lack of noisy apps or overly full instructions during the evaluation process. To address these limitations, we use a slot-based instruction generation method to construct a more realistic and comprehensive benchmark named Mobile-Bench-v2. Mobile-Bench-v2 includes a common task split, with offline multi-path evaluation to assess the agent's ability to obtain step rewards during task execution. It contains a noisy split based on pop-ups and ads apps, and a contaminated split named AITZ-Noise to formulate a real noisy environment. Furthermore, an ambiguous instruction split with preset Q\&A interactions is released to evaluate the agent's proactive interaction capabilities. We conduct evaluations on these splits using the single-agent framework AppAgent-v1, the multi-agent framework Mobile-Agent-v2, as well as other mobile agents such as UI-Tars and OS-Atlas. Code and data are available at this https URL. 

---
# AutoMedEval: Harnessing Language Models for Automatic Medical Capability Evaluation 

**Authors**: Xiechi Zhang, Zetian Ouyang, Linlin Wang, Gerard de Melo, Zhu Cao, Xiaoling Wang, Ya Zhang, Yanfeng Wang, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2505.11887)  

**Abstract**: With the proliferation of large language models (LLMs) in the medical domain, there is increasing demand for improved evaluation techniques to assess their capabilities. However, traditional metrics like F1 and ROUGE, which rely on token overlaps to measure quality, significantly overlook the importance of medical terminology. While human evaluation tends to be more reliable, it can be very costly and may as well suffer from inaccuracies due to limits in human expertise and motivation. Although there are some evaluation methods based on LLMs, their usability in the medical field is limited due to their proprietary nature or lack of expertise. To tackle these challenges, we present AutoMedEval, an open-sourced automatic evaluation model with 13B parameters specifically engineered to measure the question-answering proficiency of medical LLMs. The overarching objective of AutoMedEval is to assess the quality of responses produced by diverse models, aspiring to significantly reduce the dependence on human evaluation. Specifically, we propose a hierarchical training method involving curriculum instruction tuning and an iterative knowledge introspection mechanism, enabling AutoMedEval to acquire professional medical assessment capabilities with limited instructional data. Human evaluations indicate that AutoMedEval surpasses other baselines in terms of correlation with human judgments. 

---
# NAMET: Robust Massive Model Editing via Noise-Aware Memory Optimization 

**Authors**: Yanbo Dai, Zhenlan Ji, Zongjie Li, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11876)  

**Abstract**: Model editing techniques are essential for efficiently updating knowledge in large language models (LLMs). However, the effectiveness of existing approaches degrades in massive editing scenarios, particularly when evaluated with practical metrics or in context-rich settings. We attribute these failures to embedding collisions among knowledge items, which undermine editing reliability at scale. To address this, we propose NAMET (Noise-aware Model Editing in Transformers), a simple yet effective method that introduces noise during memory extraction via a one-line modification to MEMIT. Extensive experiments across six LLMs and three datasets demonstrate that NAMET consistently outperforms existing methods when editing thousands of facts. 

---
# When AI Co-Scientists Fail: SPOT-a Benchmark for Automated Verification of Scientific Research 

**Authors**: Guijin Son, Jiwoo Hong, Honglu Fan, Heejeong Nam, Hyunwoo Ko, Seungwon Lim, Jinyeop Song, Jinha Choi, Gonçalo Paulo, Youngjae Yu, Stella Biderman  

**Link**: [PDF](https://arxiv.org/pdf/2505.11855)  

**Abstract**: Recent advances in large language models (LLMs) have fueled the vision of automated scientific discovery, often called AI Co-Scientists. To date, prior work casts these systems as generative co-authors responsible for crafting hypotheses, synthesizing code, or drafting manuscripts. In this work, we explore a complementary application: using LLMs as verifiers to automate the \textbf{academic verification of scientific manuscripts}. To that end, we introduce SPOT, a dataset of 83 published papers paired with 91 errors significant enough to prompt errata or retraction, cross-validated with actual authors and human annotators. Evaluating state-of-the-art LLMs on SPOT, we find that none surpasses 21.1\% recall or 6.1\% precision (o3 achieves the best scores, with all others near zero). Furthermore, confidence estimates are uniformly low, and across eight independent runs, models rarely rediscover the same errors, undermining their reliability. Finally, qualitative analysis with domain experts reveals that even the strongest models make mistakes resembling student-level misconceptions derived from misunderstandings. These findings highlight the substantial gap between current LLM capabilities and the requirements for dependable AI-assisted academic verification. 

---
# Multilingual Collaborative Defense for Large Language Models 

**Authors**: Hongliang Li, Jinan Xu, Gengping Cui, Changhao Guan, Fengran Mo, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11835)  

**Abstract**: The robustness and security of large language models (LLMs) has become a prominent research area. One notable vulnerability is the ability to bypass LLM safeguards by translating harmful queries into rare or underrepresented languages, a simple yet effective method of "jailbreaking" these models. Despite the growing concern, there has been limited research addressing the safeguarding of LLMs in multilingual scenarios, highlighting an urgent need to enhance multilingual safety. In this work, we investigate the correlation between various attack features across different languages and propose Multilingual Collaborative Defense (MCD), a novel learning method that optimizes a continuous, soft safety prompt automatically to facilitate multilingual safeguarding of LLMs. The MCD approach offers three advantages: First, it effectively improves safeguarding performance across multiple languages. Second, MCD maintains strong generalization capabilities while minimizing false refusal rates. Third, MCD mitigates the language safety misalignment caused by imbalances in LLM training corpora. To evaluate the effectiveness of MCD, we manually construct multilingual versions of commonly used jailbreak benchmarks, such as MaliciousInstruct and AdvBench, to assess various safeguarding methods. Additionally, we introduce these datasets in underrepresented (zero-shot) languages to verify the language transferability of MCD. The results demonstrate that MCD outperforms existing approaches in safeguarding against multilingual jailbreak attempts while also exhibiting strong language transfer capabilities. Our code is available at this https URL. 

---
# Class Distillation with Mahalanobis Contrast: An Efficient Training Paradigm for Pragmatic Language Understanding Tasks 

**Authors**: Chenlu Wang, Weimin Lyu, Ritwik Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.11829)  

**Abstract**: Detecting deviant language such as sexism, or nuanced language such as metaphors or sarcasm, is crucial for enhancing the safety, clarity, and interpretation of online social discourse. While existing classifiers deliver strong results on these tasks, they often come with significant computational cost and high data demands. In this work, we propose \textbf{Cla}ss \textbf{D}istillation (ClaD), a novel training paradigm that targets the core challenge: distilling a small, well-defined target class from a highly diverse and heterogeneous background. ClaD integrates two key innovations: (i) a loss function informed by the structural properties of class distributions, based on Mahalanobis distance, and (ii) an interpretable decision algorithm optimized for class separation. Across three benchmark detection tasks -- sexism, metaphor, and sarcasm -- ClaD outperforms competitive baselines, and even with smaller language models and orders of magnitude fewer parameters, achieves performance comparable to several large language models (LLMs). These results demonstrate ClaD as an efficient tool for pragmatic language understanding tasks that require gleaning a small target class from a larger heterogeneous background. 

---
# Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning 

**Authors**: Yansong Ning, Wei Li, Jun Fang, Naiqiang Tan, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11827)  

**Abstract**: Compressing long chain-of-thought (CoT) from large language models (LLMs) is an emerging strategy to improve the reasoning efficiency of LLMs. Despite its promising benefits, existing studies equally compress all thoughts within a long CoT, hindering more concise and effective reasoning. To this end, we first investigate the importance of different thoughts by examining their effectiveness and efficiency in contributing to reasoning through automatic long CoT chunking and Monte Carlo rollouts. Building upon the insights, we propose a theoretically bounded metric to jointly measure the effectiveness and efficiency of different thoughts. We then propose Long$\otimes$Short, an efficient reasoning framework that enables two LLMs to collaboratively solve the problem: a long-thought LLM for more effectively generating important thoughts, while a short-thought LLM for efficiently generating remaining thoughts. Specifically, we begin by synthesizing a small amount of cold-start data to fine-tune LLMs for long-thought and short-thought reasoning styles, respectively. Furthermore, we propose a synergizing-oriented multi-turn reinforcement learning, focusing on the model self-evolution and collaboration between long-thought and short-thought LLMs. Experimental results show that our method enables Qwen2.5-7B and Llama3.1-8B to achieve comparable performance compared to DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B, while reducing token length by over 80% across the MATH500, AIME24/25, AMC23, and GPQA Diamond benchmarks. Our data and code are available at this https URL. 

---
# Chain-of-Model Learning for Language Model 

**Authors**: Kaitao Song, Xiaohua Wang, Xu Tan, Huiqiang Jiang, Chengruidong Zhang, Yongliang Shen, Cen LU, Zihao Li, Zifan Song, Caihua Shan, Yansen Wang, Kan Ren, Xiaoqing Zheng, Tao Qin, Yuqing Yang, Dongsheng Li, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11820)  

**Abstract**: In this paper, we propose a novel learning paradigm, termed Chain-of-Model (CoM), which incorporates the causal relationship into the hidden states of each layer as a chain style, thereby introducing great scaling efficiency in model training and inference flexibility in deployment. We introduce the concept of Chain-of-Representation (CoR), which formulates the hidden states at each layer as a combination of multiple sub-representations (i.e., chains) at the hidden dimension level. In each layer, each chain from the output representations can only view all of its preceding chains in the input representations. Consequently, the model built upon CoM framework can progressively scale up the model size by increasing the chains based on the previous models (i.e., chains), and offer multiple sub-models at varying sizes for elastic inference by using different chain numbers. Based on this principle, we devise Chain-of-Language-Model (CoLM), which incorporates the idea of CoM into each layer of Transformer architecture. Based on CoLM, we further introduce CoLM-Air by introducing a KV sharing mechanism, that computes all keys and values within the first chain and then shares across all chains. This design demonstrates additional extensibility, such as enabling seamless LM switching, prefilling acceleration and so on. Experimental results demonstrate our CoLM family can achieve comparable performance to the standard Transformer, while simultaneously enabling greater flexiblity, such as progressive scaling to improve training efficiency and offer multiple varying model sizes for elastic inference, paving a a new way toward building language models. Our code will be released in the future at: this https URL. 

---
# BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering 

**Authors**: Taolin Zhang, Dongyang Li, Qizhou Chen, Chengyu Wang, Xiaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2505.11811)  

**Abstract**: Multi-hop question answering (QA) involves finding multiple relevant passages and performing step-by-step reasoning to answer complex questions. Previous works on multi-hop QA employ specific methods from different modeling perspectives based on large language models (LLMs), regardless of the question types. In this paper, we first conduct an in-depth analysis of public multi-hop QA benchmarks, dividing the questions into four types and evaluating five types of cutting-edge methods for multi-hop QA: Chain-of-Thought (CoT), Single-step, Iterative-step, Sub-step, and Adaptive-step. We find that different types of multi-hop questions have varying degrees of sensitivity to different types of methods. Thus, we propose a Bi-levEL muLti-agEnt reasoning (BELLE) framework to address multi-hop QA by specifically focusing on the correspondence between question types and methods, where each type of method is regarded as an ''operator'' by prompting LLMs differently. The first level of BELLE includes multiple agents that debate to obtain an executive plan of combined ''operators'' to address the multi-hop QA task comprehensively. During the debate, in addition to the basic roles of affirmative debater, negative debater, and judge, at the second level, we further leverage fast and slow debaters to monitor whether changes in viewpoints are reasonable. Extensive experiments demonstrate that BELLE significantly outperforms strong baselines in various datasets. Additionally, the model consumption of BELLE is higher cost-effectiveness than that of single models in more complex multi-hop QA scenarios. 

---
# Efficiently Building a Domain-Specific Large Language Model from Scratch: A Case Study of a Classical Chinese Large Language Model 

**Authors**: Shen Li, Renfen Hu, Lijun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11810)  

**Abstract**: General-purpose large language models demonstrate notable capabilities in language comprehension and generation, achieving results that are comparable to, or even surpass, human performance in many language information processing tasks. Nevertheless, when general models are applied to some specific domains, e.g., Classical Chinese texts, their effectiveness is often unsatisfactory, and fine-tuning open-source foundational models similarly struggles to adequately incorporate domain-specific knowledge. To address this challenge, this study developed a large language model, AI Taiyan, specifically designed for understanding and generating Classical Chinese. Experiments show that with a reasonable model design, data processing, foundational training, and fine-tuning, satisfactory results can be achieved with only 1.8 billion parameters. In key tasks related to Classical Chinese information processing such as punctuation, identification of allusions, explanation of word meanings, and translation between ancient and modern Chinese, this model exhibits a clear advantage over both general-purpose large models and domain-specific traditional models, achieving levels close to or surpassing human baselines. This research provides a reference for the efficient construction of specialized domain-specific large language models. Furthermore, the paper discusses the application of this model in fields such as the collation of ancient texts, dictionary editing, and language research, combined with case studies. 

---
# Retrospex: Language Agent Meets Offline Reinforcement Learning Critic 

**Authors**: Yufei Xiang, Yiqun Shen, Yeqin Zhang, Cam-Tu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11807)  

**Abstract**: Large Language Models (LLMs) possess extensive knowledge and commonsense reasoning capabilities, making them valuable for creating powerful agents. However, existing LLM agent frameworks have not fully utilized past experiences for improvement. This work introduces a new LLM-based agent framework called Retrospex, which addresses this challenge by analyzing past experiences in depth. Unlike previous approaches, Retrospex does not directly integrate experiences into the LLM's context. Instead, it combines the LLM's action likelihood with action values estimated by a Reinforcement Learning (RL) Critic, which is trained on past experiences through an offline ''retrospection'' process. Additionally, Retrospex employs a dynamic action rescoring mechanism that increases the importance of experience-based values for tasks that require more interaction with the environment. We evaluate Retrospex in ScienceWorld, ALFWorld and Webshop environments, demonstrating its advantages over strong, contemporary baselines. 

---
# Towards Universal Semantics With Large Language Models 

**Authors**: Raymond Baartmans, Matthew Raffel, Rahul Vikram, Aiden Deringer, Lizhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11764)  

**Abstract**: The Natural Semantic Metalanguage (NSM) is a linguistic theory based on a universal set of semantic primes: simple, primitive word-meanings that have been shown to exist in most, if not all, languages of the world. According to this framework, any word, regardless of complexity, can be paraphrased using these primes, revealing a clear and universally translatable meaning. These paraphrases, known as explications, can offer valuable applications for many natural language processing (NLP) tasks, but producing them has traditionally been a slow, manual process. In this work, we present the first study of using large language models (LLMs) to generate NSM explications. We introduce automatic evaluation methods, a tailored dataset for training and evaluation, and fine-tuned models for this task. Our 1B and 8B models outperform GPT-4o in producing accurate, cross-translatable explications, marking a significant step toward universal semantic representation with LLMs and opening up new possibilities for applications in semantic analysis, translation, and beyond. 

---
# Masking in Multi-hop QA: An Analysis of How Language Models Perform with Context Permutation 

**Authors**: Wenyu Huang, Pavlos Vougiouklis, Mirella Lapata, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11754)  

**Abstract**: Multi-hop Question Answering (MHQA) adds layers of complexity to question answering, making it more challenging. When Language Models (LMs) are prompted with multiple search results, they are tasked not only with retrieving relevant information but also employing multi-hop reasoning across the information sources. Although LMs perform well on traditional question-answering tasks, the causal mask can hinder their capacity to reason across complex contexts. In this paper, we explore how LMs respond to multi-hop questions by permuting search results (retrieved documents) under various configurations. Our study reveals interesting findings as follows: 1) Encoder-decoder models, such as the ones in the Flan-T5 family, generally outperform causal decoder-only LMs in MHQA tasks, despite being significantly smaller in size; 2) altering the order of gold documents reveals distinct trends in both Flan T5 models and fine-tuned decoder-only models, with optimal performance observed when the document order aligns with the reasoning chain order; 3) enhancing causal decoder-only models with bi-directional attention by modifying the causal mask can effectively boost their end performance. In addition to the above, we conduct a thorough investigation of the distribution of LM attention weights in the context of MHQA. Our experiments reveal that attention weights tend to peak at higher values when the resulting answer is correct. We leverage this finding to heuristically improve LMs' performance on this task. Our code is publicly available at this https URL. 

---
# Token Masking Improves Transformer-Based Text Classification 

**Authors**: Xianglong Xu, John Bowen, Rojin Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2505.11746)  

**Abstract**: While transformer-based models achieve strong performance on text classification, we explore whether masking input tokens can further enhance their effectiveness. We propose token masking regularization, a simple yet theoretically motivated method that randomly replaces input tokens with a special [MASK] token at probability p. This introduces stochastic perturbations during training, leading to implicit gradient averaging that encourages the model to capture deeper inter-token dependencies. Experiments on language identification and sentiment analysis -- across diverse models (mBERT, Qwen2.5-0.5B, TinyLlama-1.1B) -- show consistent improvements over standard regularization techniques. We identify task-specific optimal masking rates, with p = 0.1 as a strong general default. We attribute the gains to two key effects: (1) input perturbation reduces overfitting, and (2) gradient-level smoothing acts as implicit ensembling. 

---
# ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training 

**Authors**: Feijiang Han, Xiaodong Yu, Jianheng Tang, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2505.11739)  

**Abstract**: Recently, training-free methods for improving large language models (LLMs) have attracted growing interest, with token-level attention tuning emerging as a promising and interpretable direction. However, existing methods typically rely on auxiliary mechanisms to identify important or irrelevant task-specific tokens, introducing potential bias and limiting applicability. In this paper, we uncover a surprising and elegant alternative: the semantically empty initial token is a powerful and underexplored control point for optimizing model behavior. Through theoretical analysis, we show that tuning the initial token's attention sharpens or flattens the attention distribution over subsequent tokens, and its role as an attention sink amplifies this effect. Empirically, we find that: (1) tuning its attention improves LLM performance more effectively than tuning other task-specific tokens; (2) the effect follows a consistent trend across layers, with earlier layers having greater impact, but varies across attention heads, with different heads showing distinct preferences in how they attend to this token. Based on these findings, we propose ZeroTuning, a training-free approach that improves LLM performance by applying head-specific attention adjustments to this special token. Despite tuning only one token, ZeroTuning achieves higher performance on text classification, multiple-choice, and multi-turn conversation tasks across models such as Llama, Qwen, and DeepSeek. For example, ZeroTuning improves Llama-3.1-8B by 11.71% on classification, 2.64% on QA tasks, and raises its multi-turn score from 7.804 to 7.966. The method is also robust to limited resources, few-shot settings, long contexts, quantization, decoding strategies, and prompt variations. Our work sheds light on a previously overlooked control point in LLMs, offering new insights into both inference-time tuning and model interpretability. 

---
# MedCaseReasoning: Evaluating and learning diagnostic reasoning from clinical case reports 

**Authors**: Kevin Wu, Eric Wu, Rahul Thapa, Kevin Wei, Angela Zhang, Arvind Suresh, Jacqueline J. Tao, Min Woo Sun, Alejandro Lozano, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.11733)  

**Abstract**: Doctors and patients alike increasingly use Large Language Models (LLMs) to diagnose clinical cases. However, unlike domains such as math or coding, where correctness can be objectively defined by the final answer, medical diagnosis requires both the outcome and the reasoning process to be accurate. Currently, widely used medical benchmarks like MedQA and MMLU assess only accuracy in the final answer, overlooking the quality and faithfulness of the clinical reasoning process. To address this limitation, we introduce MedCaseReasoning, the first open-access dataset for evaluating LLMs on their ability to align with clinician-authored diagnostic reasoning. The dataset includes 14,489 diagnostic question-and-answer cases, each paired with detailed reasoning statements derived from open-access medical case reports. We evaluate state-of-the-art reasoning LLMs on MedCaseReasoning and find significant shortcomings in their diagnoses and reasoning: for instance, the top-performing open-source model, DeepSeek-R1, achieves only 48% 10-shot diagnostic accuracy and mentions only 64% of the clinician reasoning statements (recall). However, we demonstrate that fine-tuning LLMs on the reasoning traces derived from MedCaseReasoning significantly improves diagnostic accuracy and clinical reasoning recall by an average relative gain of 29% and 41%, respectively. The open-source dataset, code, and models are available at this https URL. 

---
# Disambiguating Reference in Visually Grounded Dialogues through Joint Modeling of Textual and Multimodal Semantic Structures 

**Authors**: Shun Inadumi, Nobuhiro Ueda, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2505.11726)  

**Abstract**: Multimodal reference resolution, including phrase grounding, aims to understand the semantic relations between mentions and real-world objects. Phrase grounding between images and their captions is a well-established task. In contrast, for real-world applications, it is essential to integrate textual and multimodal reference resolution to unravel the reference relations within dialogue, especially in handling ambiguities caused by pronouns and ellipses. This paper presents a framework that unifies textual and multimodal reference resolution by mapping mention embeddings to object embeddings and selecting mentions or objects based on their similarity. Our experiments show that learning textual reference resolution, such as coreference resolution and predicate-argument structure analysis, positively affects performance in multimodal reference resolution. In particular, our model with coreference resolution performs better in pronoun phrase grounding than representative models for this task, MDETR and GLIP. Our qualitative analysis demonstrates that incorporating textual reference relations strengthens the confidence scores between mentions, including pronouns and predicates, and objects, which can reduce the ambiguities that arise in visually grounded dialogues. 

---
# Hierarchical Bracketing Encodings for Dependency Parsing as Tagging 

**Authors**: Ana Ezquerro, David Vilares, Anssi Yli-Jyrä, Carlos Gómez-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2505.11693)  

**Abstract**: We present a family of encodings for sequence labeling dependency parsing, based on the concept of hierarchical bracketing. We prove that the existing 4-bit projective encoding belongs to this family, but it is suboptimal in the number of labels used to encode a tree. We derive an optimal hierarchical bracketing, which minimizes the number of symbols used and encodes projective trees using only 12 distinct labels (vs. 16 for the 4-bit encoding). We also extend optimal hierarchical bracketing to support arbitrary non-projectivity in a more compact way than previous encodings. Our new encodings yield competitive accuracy on a diverse set of treebanks. 

---
# Automatic Speech Recognition for African Low-Resource Languages: Challenges and Future Directions 

**Authors**: Sukairaj Hafiz Imam, Babangida Sani, Dawit Ketema Gete, Bedru Yimam Ahamed, Ibrahim Said Ahmad, Idris Abdulmumin, Seid Muhie Yimam, Muhammad Yahuza Bello, Shamsuddeen Hassan Muhammad  

**Link**: [PDF](https://arxiv.org/pdf/2505.11690)  

**Abstract**: Automatic Speech Recognition (ASR) technologies have transformed human-computer interaction; however, low-resource languages in Africa remain significantly underrepresented in both research and practical applications. This study investigates the major challenges hindering the development of ASR systems for these languages, which include data scarcity, linguistic complexity, limited computational resources, acoustic variability, and ethical concerns surrounding bias and privacy. The primary goal is to critically analyze these barriers and identify practical, inclusive strategies to advance ASR technologies within the African context. Recent advances and case studies emphasize promising strategies such as community-driven data collection, self-supervised and multilingual learning, lightweight model architectures, and techniques that prioritize privacy. Evidence from pilot projects involving various African languages showcases the feasibility and impact of customized solutions, which encompass morpheme-based modeling and domain-specific ASR applications in sectors like healthcare and education. The findings highlight the importance of interdisciplinary collaboration and sustained investment to tackle the distinct linguistic and infrastructural challenges faced by the continent. This study offers a progressive roadmap for creating ethical, efficient, and inclusive ASR systems that not only safeguard linguistic diversity but also improve digital accessibility and promote socioeconomic participation for speakers of African languages. 

---
# Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation 

**Authors**: Susanna Rücker, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2505.11683)  

**Abstract**: Entity disambiguation (ED) is the task of linking mentions in text to corresponding entries in a knowledge base. Dual Encoders address this by embedding mentions and label candidates in a shared embedding space and applying a similarity metric to predict the correct label. In this work, we focus on evaluating key design decisions for Dual Encoder-based ED, such as its loss function, similarity metric, label verbalization format, and negative sampling strategy. We present the resulting model VerbalizED, a document-level Dual Encoder model that includes contextual label verbalizations and efficient hard negative sampling. Additionally, we explore an iterative prediction variant that aims to improve the disambiguation of challenging data points. Comprehensive experiments on AIDA-Yago validate the effectiveness of our approach, offering insights into impactful design choices that result in a new State-of-the-Art system on the ZELDA benchmark. 

---
# Ambiguity Resolution in Text-to-Structured Data Mapping 

**Authors**: Zhibo Hu, Chen Wang, Yanfeng Shu, Hye-Young Paik, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11679)  

**Abstract**: Ambiguity in natural language is a significant obstacle for achieving accurate text to structured data mapping through large language models (LLMs), which affects the performance of tasks such as mapping text to agentic tool calling and text-to-SQL queries. Existing methods of ambiguity handling either exploit ReACT framework to produce the correct mapping through trial and error, or supervised fine tuning to guide models to produce a biased mapping to improve certain tasks. In this paper, we adopt a different approach that characterizes the representation difference of ambiguous text in the latent space and leverage the difference to identify ambiguity before mapping them to structured data. To detect ambiguity of a sentence, we focused on the relationship between ambiguous questions and their interpretations and what cause the LLM ignore multiple interpretations. Different to the distance calculated by dense embedding vectors, we utilize the observation that ambiguity is caused by concept missing in latent space of LLM to design a new distance measurement, computed through the path kernel by the integral of gradient values for each concepts from sparse-autoencoder (SAE) under each state. We identify patterns to distinguish ambiguous questions with this measurement. Based on our observation, We propose a new framework to improve the performance of LLMs on ambiguous agentic tool calling through missing concepts prediction. 

---
# Multilingual Prompt Engineering in Large Language Models: A Survey Across NLP Tasks 

**Authors**: Shubham Vatsal, Harsh Dubey, Aditi Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11665)  

**Abstract**: Large language models (LLMs) have demonstrated impressive performance across a wide range of Natural Language Processing (NLP) tasks. However, ensuring their effectiveness across multiple languages presents unique challenges. Multilingual prompt engineering has emerged as a key approach to enhance LLMs' capabilities in diverse linguistic settings without requiring extensive parameter re-training or fine-tuning. With growing interest in multilingual prompt engineering over the past two to three years, researchers have explored various strategies to improve LLMs' performance across languages and NLP tasks. By crafting structured natural language prompts, researchers have successfully extracted knowledge from LLMs across different languages, making these techniques an accessible pathway for a broader audience, including those without deep expertise in machine learning, to harness the capabilities of LLMs. In this paper, we survey and categorize different multilingual prompting techniques based on the NLP tasks they address across a diverse set of datasets that collectively span around 250 languages. We further highlight the LLMs employed, present a taxonomy of approaches and discuss potential state-of-the-art (SoTA) methods for specific multilingual datasets. Additionally, we derive a range of insights across language families and resource levels (high-resource vs. low-resource), including analyses such as the distribution of NLP tasks by language resource type and the frequency of prompting methods across different language families. Our survey reviews 36 research papers covering 39 prompting techniques applied to 30 multilingual NLP tasks, with the majority of these studies published in the last two years. 

---
# Can an Easy-to-Hard Curriculum Make Reasoning Emerge in Small Language Models? Evidence from a Four-Stage Curriculum on GPT-2 

**Authors**: Xiang Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11643)  

**Abstract**: We demonstrate that a developmentally ordered curriculum markedly improves reasoning transparency and sample-efficiency in small language models (SLMs). Concretely, we train Cognivolve, a 124 M-parameter GPT-2 model, on a four-stage syllabus that ascends from lexical matching to multi-step symbolic inference and then evaluate it without any task-specific fine-tuning. Cognivolve reaches target accuracy in half the optimization steps of a single-phase baseline, activates an order-of-magnitude more gradient-salient reasoning heads, and shifts those heads toward deeper layers, yielding higher-entropy attention that balances local and long-range context. The same curriculum applied out of order or with optimizer resets fails to reproduce these gains, confirming that progression--not extra compute--drives the effect. We also identify open challenges: final-answer success still lags a conventional run by about 30%, and our saliency probe under-detects verbal-knowledge heads in the hardest stage, suggesting directions for mixed-stage fine-tuning and probe expansion. 

---
# Critique-Guided Distillation: Improving Supervised Fine-tuning via Better Distillation 

**Authors**: Berkcan Kapusuzoglu, Supriyo Chakraborty, Chia-Hsuan Lee, Sambit Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11628)  

**Abstract**: Supervised fine-tuning (SFT) using expert demonstrations often suffer from the imitation problem, where the model learns to reproduce the correct responses without \emph{understanding} the underlying rationale. To address this limitation, we propose \textsc{Critique-Guided Distillation (CGD)}, a novel multi-stage framework that integrates teacher model generated \emph{explanatory critiques} and \emph{refined responses} into the SFT process. A student model is then trained to map the triplet of prompt, teacher critique, and its own initial response to the corresponding refined teacher response, thereby learning both \emph{what} to imitate and \emph{why}. Using entropy-based analysis, we show that \textsc{CGD} reduces refinement uncertainty and can be interpreted as a Bayesian posterior update. We perform extensive empirical evaluation of \textsc{CGD}, on variety of benchmark tasks, and demonstrate significant gains on both math (AMC23 +17.5%) and language understanding tasks (MMLU-Pro +6.3%), while successfully mitigating the format drift issues observed in previous critique fine-tuning (CFT) techniques. 

---
# THELMA: Task Based Holistic Evaluation of Large Language Model Applications-RAG Question Answering 

**Authors**: Udita Patel, Rutu Mulkar, Jay Roberts, Cibi Chakravarthy Senthilkumar, Sujay Gandhi, Xiaofei Zheng, Naumaan Nayyar, Rafael Castrillo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11626)  

**Abstract**: We propose THELMA (Task Based Holistic Evaluation of Large Language Model Applications), a reference free framework for RAG (Retrieval Augmented generation) based question answering (QA) applications. THELMA consist of six interdependent metrics specifically designed for holistic, fine grained evaluation of RAG QA applications. THELMA framework helps developers and application owners evaluate, monitor and improve end to end RAG QA pipelines without requiring labelled sources or reference this http URL also present our findings on the interplay of the proposed THELMA metrics, which can be interpreted to identify the specific RAG component needing improvement in QA applications. 

---
# Steering Risk Preferences in Large Language Models by Aligning Behavioral and Neural Representations 

**Authors**: Jian-Qiao Zhu, Haijiang Yan, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.11615)  

**Abstract**: Changing the behavior of large language models (LLMs) can be as straightforward as editing the Transformer's residual streams using appropriately constructed "steering vectors." These modifications to internal neural activations, a form of representation engineering, offer an effective and targeted means of influencing model behavior without retraining or fine-tuning the model. But how can such steering vectors be systematically identified? We propose a principled approach for uncovering steering vectors by aligning latent representations elicited through behavioral methods (specifically, Markov chain Monte Carlo with LLMs) with their neural counterparts. To evaluate this approach, we focus on extracting latent risk preferences from LLMs and steering their risk-related outputs using the aligned representations as steering vectors. We show that the resulting steering vectors successfully and reliably modulate LLM outputs in line with the targeted behavior. 

---
# MedGUIDE: Benchmarking Clinical Decision-Making in Large Language Models 

**Authors**: Xiaomin Li, Mingye Gao, Yuexing Hao, Taoran Li, Guangya Wan, Zihan Wang, Yijun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11613)  

**Abstract**: Clinical guidelines, typically structured as decision trees, are central to evidence-based medical practice and critical for ensuring safe and accurate diagnostic decision-making. However, it remains unclear whether Large Language Models (LLMs) can reliably follow such structured protocols. In this work, we introduce MedGUIDE, a new benchmark for evaluating LLMs on their ability to make guideline-consistent clinical decisions. MedGUIDE is constructed from 55 curated NCCN decision trees across 17 cancer types and uses clinical scenarios generated by LLMs to create a large pool of multiple-choice diagnostic questions. We apply a two-stage quality selection process, combining expert-labeled reward models and LLM-as-a-judge ensembles across ten clinical and linguistic criteria, to select 7,747 high-quality samples. We evaluate 25 LLMs spanning general-purpose, open-source, and medically specialized models, and find that even domain-specific LLMs often underperform on tasks requiring structured guideline adherence. We also test whether performance can be improved via in-context guideline inclusion or continued pretraining. Our findings underscore the importance of MedGUIDE in assessing whether LLMs can operate safely within the procedural frameworks expected in real-world clinical settings. 

---
# Talk to Your Slides: Efficient Slide Editing Agent with Large Language Models 

**Authors**: Kyudan Jung, Hojun Cho, Jooyeol Yun, Jaehyeok Jang, Jagul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11604)  

**Abstract**: Existing research on large language models (LLMs) for PowerPoint predominantly focuses on slide generation, overlooking the common yet tedious task of editing existing slides. We introduce Talk-to-Your-Slides, an LLM-powered agent that directly edits slides within active PowerPoint sessions through COM communication. Our system employs a two-level approach: (1) high-level processing where an LLM agent interprets instructions and formulates editing plans, and (2) low-level execution where Python scripts directly manipulate PowerPoint objects. Unlike previous methods relying on predefined operations, our approach enables more flexible and contextually-aware editing. To facilitate evaluation, we present TSBench, a human-annotated dataset of 379 diverse editing instructions with corresponding slide variations. Experimental results demonstrate that Talk-to-Your-Slides significantly outperforms baseline methods in execution success rate, instruction fidelity, and editing efficiency. Our code and benchmark are available at this https URL 

---
# Assessing Collective Reasoning in Multi-Agent LLMs via Hidden Profile Tasks 

**Authors**: Yuxuan Li, Aoi Naito, Hirokazu Shirado  

**Link**: [PDF](https://arxiv.org/pdf/2505.11556)  

**Abstract**: Multi-agent systems built on large language models (LLMs) promise enhanced problem-solving through distributed information integration, but also risk replicating collective reasoning failures observed in human groups. Yet, no theory-grounded benchmark exists to systematically evaluate such failures. In this paper, we introduce the Hidden Profile paradigm from social psychology as a diagnostic testbed for multi-agent LLM systems. By distributing critical information asymmetrically across agents, the paradigm reveals how inter-agent dynamics support or hinder collective reasoning. We first formalize the paradigm for multi-agent decision-making under distributed knowledge and instantiate it as a benchmark with nine tasks spanning diverse scenarios, including adaptations from prior human studies. We then conduct experiments with GPT-4.1 and five other leading LLMs, including reasoning-enhanced variants, showing that multi-agent systems across all models fail to match the accuracy of single agents given complete information. While agents' collective performance is broadly comparable to that of human groups, nuanced behavioral differences emerge, such as increased sensitivity to social desirability. Finally, we demonstrate the paradigm's diagnostic utility by exploring a cooperation-contradiction trade-off in multi-agent LLM systems. We find that while cooperative agents are prone to over-coordination in collective settings, increased contradiction impairs group convergence. This work contributes a reproducible framework for evaluating multi-agent LLM systems and motivates future research on artificial collective intelligence and human-AI interaction. 

---
# AI-generated Text Detection: A Multifaceted Approach to Binary and Multiclass Classification 

**Authors**: Harika Abburi, Sanmitra Bhattacharya, Edward Bowen, Nirmala Pudota  

**Link**: [PDF](https://arxiv.org/pdf/2505.11550)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in generating text that closely resembles human writing across a wide range of styles and genres. However, such capabilities are prone to potential misuse, such as fake news generation, spam email creation, and misuse in academic assignments. As a result, accurate detection of AI-generated text and identification of the model that generated it are crucial for maintaining the responsible use of LLMs. In this work, we addressed two sub-tasks put forward by the Defactify workshop under AI-Generated Text Detection shared task at the Association for the Advancement of Artificial Intelligence (AAAI 2025): Task A involved distinguishing between human-authored or AI-generated text, while Task B focused on attributing text to its originating language model. For each task, we proposed two neural architectures: an optimized model and a simpler variant. For Task A, the optimized neural architecture achieved fifth place with $F1$ score of 0.994, and for Task B, the simpler neural architecture also ranked fifth place with $F1$ score of 0.627. 

---
# A Data Synthesis Method Driven by Large Language Models for Proactive Mining of Implicit User Intentions in Tourism 

**Authors**: Jinqiang Wang, Huansheng Ning, Tao Zhu, Jianguo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.11533)  

**Abstract**: In the tourism domain, Large Language Models (LLMs) often struggle to mine implicit user intentions from tourists' ambiguous inquiries and lack the capacity to proactively guide users toward clarifying their needs. A critical bottleneck is the scarcity of high-quality training datasets that facilitate proactive questioning and implicit intention mining. While recent advances leverage LLM-driven data synthesis to generate such datasets and transfer specialized knowledge to downstream models, existing approaches suffer from several shortcomings: (1) lack of adaptation to the tourism domain, (2) skewed distributions of detail levels in initial inquiries, (3) contextual redundancy in the implicit intention mining module, and (4) lack of explicit thinking about tourists' emotions and intention values. Therefore, we propose SynPT (A Data Synthesis Method Driven by LLMs for Proactive Mining of Implicit User Intentions in the Tourism), which constructs an LLM-driven user agent and assistant agent to simulate dialogues based on seed data collected from Chinese tourism websites. This approach addresses the aforementioned limitations and generates SynPT-Dialog, a training dataset containing explicit reasoning. The dataset is utilized to fine-tune a general LLM, enabling it to proactively mine implicit user intentions. Experimental evaluations, conducted from both human and LLM perspectives, demonstrate the superiority of SynPT compared to existing methods. Furthermore, we analyze key hyperparameters and present case studies to illustrate the practical applicability of our method, including discussions on its adaptability to English-language scenarios. All code and data are publicly available. 

---
# Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards 

**Authors**: Xiaoyuan Liu, Tian Liang, Zhiwei He, Jiahao Xu, Wenxuan Wang, Pinjia He, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13445)  

**Abstract**: Large Language Models (LLMs) show great promise in complex reasoning, with Reinforcement Learning with Verifiable Rewards (RLVR) being a key enhancement strategy. However, a prevalent issue is ``superficial self-reflection'', where models fail to robustly verify their own outputs. We introduce RISE (Reinforcing Reasoning with Self-Verification), a novel online RL framework designed to tackle this. RISE explicitly and simultaneously trains an LLM to improve both its problem-solving and self-verification abilities within a single, integrated RL process. The core mechanism involves leveraging verifiable rewards from an outcome verifier to provide on-the-fly feedback for both solution generation and self-verification tasks. In each iteration, the model generates solutions, then critiques its own on-policy generated solutions, with both trajectories contributing to the policy update. Extensive experiments on diverse mathematical reasoning benchmarks show that RISE consistently improves model's problem-solving accuracy while concurrently fostering strong self-verification skills. Our analyses highlight the advantages of online verification and the benefits of increased verification compute. Additionally, RISE models exhibit more frequent and accurate self-verification behaviors during reasoning. These advantages reinforce RISE as a flexible and effective path towards developing more robust and self-aware reasoners. 

---
# Optimizing Anytime Reasoning via Budget Relative Policy Optimization 

**Authors**: Penghui Qi, Zichen Liu, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.13438)  

**Abstract**: Scaling test-time compute is crucial for enhancing the reasoning capabilities of large language models (LLMs). Existing approaches typically employ reinforcement learning (RL) to maximize a verifiable reward obtained at the end of reasoning traces. However, such methods optimize only the final performance under a large and fixed token budget, which hinders efficiency in both training and deployment. In this work, we present a novel framework, AnytimeReasoner, to optimize anytime reasoning performance, which aims to improve token efficiency and the flexibility of reasoning under varying token budget constraints. To achieve this, we truncate the complete thinking process to fit within sampled token budgets from a prior distribution, compelling the model to summarize the optimal answer for each truncated thinking for verification. This introduces verifiable dense rewards into the reasoning process, facilitating more effective credit assignment in RL optimization. We then optimize the thinking and summary policies in a decoupled manner to maximize the cumulative reward. Additionally, we introduce a novel variance reduction technique, Budget Relative Policy Optimization (BRPO), to enhance the robustness and efficiency of the learning process when reinforcing the thinking policy. Empirical results in mathematical reasoning tasks demonstrate that our method consistently outperforms GRPO across all thinking budgets under various prior distributions, enhancing both training and token efficiency. 

---
# Fine-tuning Quantized Neural Networks with Zeroth-order Optimization 

**Authors**: Sifeng Shang, Jiayi Zhou, Chenyu Lin, Minxian Li, Kaiyang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.13430)  

**Abstract**: As the size of large language models grows exponentially, GPU memory has become a bottleneck for adapting these models to downstream tasks. In this paper, we aim to push the limits of memory-efficient training by minimizing memory usage on model weights, gradients, and optimizer states, within a unified framework. Our idea is to eliminate both gradients and optimizer states using zeroth-order optimization, which approximates gradients by perturbing weights during forward passes to identify gradient directions. To minimize memory usage on weights, we employ model quantization, e.g., converting from bfloat16 to int4. However, directly applying zeroth-order optimization to quantized weights is infeasible due to the precision gap between discrete weights and continuous gradients, which would otherwise require de-quantization and re-quantization. To overcome this challenge, we propose Quantized Zeroth-order Optimization (QZO), a novel approach that perturbs the continuous quantization scale for gradient estimation and uses a directional derivative clipping method to stabilize training. QZO is orthogonal to both scalar-based and codebook-based post-training quantization methods. Compared to full-parameter fine-tuning in bfloat16, QZO can reduce the total memory cost by more than 18$\times$ for 4-bit LLMs, and enables fine-tuning Llama-2-13B and Stable Diffusion 3.5 Large within a single 24GB GPU. 

---
# CoT-Kinetics: A Theoretical Modeling Assessing LRM Reasoning Process 

**Authors**: Jinhe Bi, Danqi Yan, Yifan Wang, Wenke Huang, Haokun Chen, Guancheng Wan, Mang Ye, Xun Xiao, Hinrich Schuetze, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.13408)  

**Abstract**: Recent Large Reasoning Models significantly improve the reasoning ability of Large Language Models by learning to reason, exhibiting the promising performance in solving complex tasks. LRMs solve tasks that require complex reasoning by explicitly generating reasoning trajectories together with answers. Nevertheless, judging the quality of such an output answer is not easy because only considering the correctness of the answer is not enough and the soundness of the reasoning trajectory part matters as well. Logically, if the soundness of the reasoning part is poor, even if the answer is correct, the confidence of the derived answer should be low. Existing methods did consider jointly assessing the overall output answer by taking into account the reasoning part, however, their capability is still not satisfactory as the causal relationship of the reasoning to the concluded answer cannot properly reflected. In this paper, inspired by classical mechanics, we present a novel approach towards establishing a CoT-Kinetics energy equation. Specifically, our CoT-Kinetics energy equation formulates the token state transformation process, which is regulated by LRM internal transformer layers, as like a particle kinetics dynamics governed in a mechanical field. Our CoT-Kinetics energy assigns a scalar score to evaluate specifically the soundness of the reasoning phase, telling how confident the derived answer could be given the evaluated reasoning. As such, the LRM's overall output quality can be accurately measured, rather than a coarse judgment (e.g., correct or incorrect) anymore. 

---
# A Minimum Description Length Approach to Regularization in Neural Networks 

**Authors**: Matan Abudy, Orr Well, Emmanuel Chemla, Roni Katzir, Nur Lan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13398)  

**Abstract**: State-of-the-art neural networks can be trained to become remarkable solutions to many problems. But while these architectures can express symbolic, perfect solutions, trained models often arrive at approximations instead. We show that the choice of regularization method plays a crucial role: when trained on formal languages with standard regularization ($L_1$, $L_2$, or none), expressive architectures not only fail to converge to correct solutions but are actively pushed away from perfect initializations. In contrast, applying the Minimum Description Length (MDL) principle to balance model complexity with data fit provides a theoretically grounded regularization method. Using MDL, perfect solutions are selected over approximations, independently of the optimization algorithm. We propose that unlike existing regularization techniques, MDL introduces the appropriate inductive bias to effectively counteract overfitting and promote generalization. 

---
# IG Parser: A Software Package for the Encoding of Institutional Statements using the Institutional Grammar 

**Authors**: Christopher K. Frantz  

**Link**: [PDF](https://arxiv.org/pdf/2505.13393)  

**Abstract**: This article provides an overview of IG Parser, a software that facilitates qualitative content analysis of formal (e.g., legal) rules or informal (e.g., socio-normative) norms, and strategies (such as conventions) -- referred to as \emph{institutions} -- that govern social systems and operate configurally to describe \emph{institutional systems}. To this end, the IG Parser employs a distinctive syntax that ensures rigorous encoding of natural language, while automating the transformation into various formats that support the downstream analysis using diverse analytical techniques. The conceptual core of the IG Parser is an associated syntax, IG Script, that operationalizes the conceptual foundations of the Institutional Grammar, and more specifically Institutional Grammar 2.0, an analytical paradigm for institutional analysis. This article presents the IG Parser, including its conceptual foundations, syntactic specification of IG Script, alongside architectural principles. This introduction is augmented with selective illustrative examples that highlight the use and benefit associated with the tool. 

---
# CompeteSMoE -- Statistically Guaranteed Mixture of Experts Training via Competition 

**Authors**: Nam V. Nguyen, Huy Nguyen, Quang Pham, Van Nguyen, Savitha Ramasamy, Nhat Ho  

**Link**: [PDF](https://arxiv.org/pdf/2505.13380)  

**Abstract**: Sparse mixture of experts (SMoE) offers an appealing solution to scale up the model complexity beyond the mean of increasing the network's depth or width. However, we argue that effective SMoE training remains challenging because of the suboptimal routing process where experts that perform computation do not directly contribute to the routing process. In this work, we propose competition, a novel mechanism to route tokens to experts with the highest neural response. Theoretically, we show that the competition mechanism enjoys a better sample efficiency than the traditional softmax routing. Furthermore, we develop CompeteSMoE, a simple yet effective algorithm to train large language models by deploying a router to learn the competition policy, thus enjoying strong performances at a low training overhead. Our extensive empirical evaluations on both the visual instruction tuning and language pre-training tasks demonstrate the efficacy, robustness, and scalability of CompeteSMoE compared to state-of-the-art SMoE strategies. We have made the implementation available at: this https URL. This work is an improved version of the previous study at arXiv:2402.02526 

---
# Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space 

**Authors**: Hengli Li, Chenxi Li, Tong Wu, Xuekai Zhu, Yuxuan Wang, Zhaoxin Yu, Eric Hanchen Jiang, Song-Chun Zhu, Zixia Jia, Ying Nian Wu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.13308)  

**Abstract**: Reasoning ability, a core component of human intelligence, continues to pose a significant challenge for Large Language Models (LLMs) in the pursuit of AGI. Although model performance has improved under the training scaling law, significant challenges remain, particularly with respect to training algorithms, such as catastrophic forgetting, and the limited availability of novel training data. As an alternative, test-time scaling enhances reasoning performance by increasing test-time computation without parameter updating. Unlike prior methods in this paradigm focused on token space, we propose leveraging latent space for more effective reasoning and better adherence to the test-time scaling law. We introduce LatentSeek, a novel framework that enhances LLM reasoning through Test-Time Instance-level Adaptation (TTIA) within the model's latent space. Specifically, LatentSeek leverages policy gradient to iteratively update latent representations, guided by self-generated reward signals. LatentSeek is evaluated on a range of reasoning benchmarks, including GSM8K, MATH-500, and AIME2024, across multiple LLM architectures. Results show that LatentSeek consistently outperforms strong baselines, such as Chain-of-Thought prompting and fine-tuning-based methods. Furthermore, our analysis demonstrates that LatentSeek is highly efficient, typically converging within a few iterations for problems of average complexity, while also benefiting from additional iterations, thereby highlighting the potential of test-time scaling in the latent space. These findings position LatentSeek as a lightweight, scalable, and effective solution for enhancing the reasoning capabilities of LLMs. 

---
# SAKURA: On the Multi-hop Reasoning of Large Audio-Language Models Based on Speech and Audio Information 

**Authors**: Chih-Kai Yang, Neo Ho, Yen-Ting Piao, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.13237)  

**Abstract**: Large audio-language models (LALMs) extend the large language models with multimodal understanding in speech, audio, etc. While their performances on speech and audio-processing tasks are extensively studied, their reasoning abilities remain underexplored. Particularly, their multi-hop reasoning, the ability to recall and integrate multiple facts, lacks systematic evaluation. Existing benchmarks focus on general speech and audio-processing tasks, conversational abilities, and fairness but overlook this aspect. To bridge this gap, we introduce SAKURA, a benchmark assessing LALMs' multi-hop reasoning based on speech and audio information. Results show that LALMs struggle to integrate speech/audio representations for multi-hop reasoning, even when they extract the relevant information correctly, highlighting a fundamental challenge in multimodal reasoning. Our findings expose a critical limitation in LALMs, offering insights and resources for future research. 

---
# Scaling Computer-Use Grounding via User Interface Decomposition and Synthesis 

**Authors**: Tianbao Xie, Jiaqi Deng, Xiaochuan Li, Junlin Yang, Haoyuan Wu, Jixuan Chen, Wenjing Hu, Xinyuan Wang, Yuhui Xu, Zekun Wang, Yiheng Xu, Junli Wang, Doyen Sahoo, Tao Yu, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.13227)  

**Abstract**: Graphical user interface (GUI) grounding, the ability to map natural language instructions to specific actions on graphical user interfaces, remains a critical bottleneck in computer use agent development. Current benchmarks oversimplify grounding tasks as short referring expressions, failing to capture the complexity of real-world interactions that require software commonsense, layout understanding, and fine-grained manipulation capabilities. To address these limitations, we introduce OSWorld-G, a comprehensive benchmark comprising 564 finely annotated samples across diverse task types including text matching, element recognition, layout understanding, and precise manipulation. Additionally, we synthesize and release the largest computer use grounding dataset Jedi, which contains 4 million examples through multi-perspective decoupling of tasks. Our multi-scale models trained on Jedi demonstrate its effectiveness by outperforming existing approaches on ScreenSpot-v2, ScreenSpot-Pro, and our OSWorld-G. Furthermore, we demonstrate that improved grounding with Jedi directly enhances agentic capabilities of general foundation models on complex computer tasks, improving from 5% to 27% on OSWorld. Through detailed ablation studies, we identify key factors contributing to grounding performance and verify that combining specialized data for different interface elements enables compositional generalization to novel interfaces. All benchmark, data, checkpoints, and code are open-sourced and available at this https URL. 

---
# Efficient Generation of Parameterised Quantum Circuits from Large Texts 

**Authors**: Colin Krawchuk, Nikhil Khatri, Neil John Ortega, Dimitri Kartsaklis  

**Link**: [PDF](https://arxiv.org/pdf/2505.13208)  

**Abstract**: Quantum approaches to natural language processing (NLP) are redefining how linguistic information is represented and processed. While traditional hybrid quantum-classical models rely heavily on classical neural networks, recent advancements propose a novel framework, DisCoCirc, capable of directly encoding entire documents as parameterised quantum circuits (PQCs), besides enjoying some additional interpretability and compositionality benefits. Following these ideas, this paper introduces an efficient methodology for converting large-scale texts into quantum circuits using tree-like representations of pregroup diagrams. Exploiting the compositional parallels between language and quantum mechanics, grounded in symmetric monoidal categories, our approach enables faithful and efficient encoding of syntactic and discourse relationships in long and complex texts (up to 6410 words in our experiments) to quantum circuits. The developed system is provided to the community as part of the augmented open-source quantum NLP package lambeq Gen II. 

---
# Zero-Shot Iterative Formalization and Planning in Partially Observable Environments 

**Authors**: Liancheng Gong, Wang Zhu, Jesse Thomason, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13126)  

**Abstract**: In planning, using LLMs not to predict plans but to formalize an environment into the Planning Domain Definition Language (PDDL) has been shown to greatly improve performance and control. While most work focused on fully observable environments, we tackle the more realistic and challenging partially observable environments where existing methods are incapacitated by the lack of complete information. We propose PDDLego+, a framework to iteratively formalize, plan, grow, and refine PDDL representations in a zero-shot manner, without needing access to any existing trajectories. On two textual simulated environments, we show that PDDLego+ not only achieves superior performance, but also shows robustness against problem complexity. We also show that the domain knowledge captured after a successful trial is interpretable and benefits future tasks. 

---
# FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference 

**Authors**: Guangda Liu, Chengwei Li, Zhenyu Ning, Jing Lin, Yiwu Yao, Danning Ke, Minyi Guo, Jieru Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13109)  

**Abstract**: Large language models (LLMs) have been widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length. While KV cache compression methods are proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks. We propose FreeKV, an algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy. On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy. On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency. Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to 13$\times$ speedup compared to SOTA KV retrieval methods. 

---
# LLM-KG-Bench 3.0: A Compass for SemanticTechnology Capabilities in the Ocean of LLMs 

**Authors**: Lars-Peter Meyer, Johannes Frey, Desiree Heim, Felix Brei, Claus Stadler, Kurt Junghanns, Michael Martin  

**Link**: [PDF](https://arxiv.org/pdf/2505.13098)  

**Abstract**: Current Large Language Models (LLMs) can assist developing program code beside many other things, but can they support working with Knowledge Graphs (KGs) as well? Which LLM is offering the best capabilities in the field of Semantic Web and Knowledge Graph Engineering (KGE)? Is this possible to determine without checking many answers manually? The LLM-KG-Bench framework in Version 3.0 is designed to answer these questions. It consists of an extensible set of tasks for automated evaluation of LLM answers and covers different aspects of working with semantic technologies. In this paper the LLM-KG-Bench framework is presented in Version 3 along with a dataset of prompts, answers and evaluations generated with it and several state-of-the-art LLMs. Significant enhancements have been made to the framework since its initial release, including an updated task API that offers greater flexibility in handling evaluation tasks, revised tasks, and extended support for various open models through the vllm library, among other improvements. A comprehensive dataset has been generated using more than 30 contemporary open and proprietary LLMs, enabling the creation of exemplary model cards that demonstrate the models' capabilities in working with RDF and SPARQL, as well as comparing their performance on Turtle and JSON-LD RDF serialization tasks. 

---
# MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix 

**Authors**: Ziyang Ma, Yinghao Ma, Yanqiao Zhu, Chen Yang, Yi-Wen Chao, Ruiyang Xu, Wenxi Chen, Yuanzhe Chen, Zhuo Chen, Jian Cong, Kai Li, Keliang Li, Siyou Li, Xinfeng Li, Xiquan Li, Zheng Lian, Yuzhe Liang, Minghao Liu, Zhikang Niu, Tianrui Wang, Yuping Wang, Yuxuan Wang, Yihao Wu, Guanrou Yang, Jianwei Yu, Ruibin Yuan, Zhisheng Zheng, Ziya Zhou, Haina Zhu, Wei Xue, Emmanouil Benetos, Kai Yu, Eng-Siong Chng, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13032)  

**Abstract**: We introduce MMAR, a new benchmark designed to evaluate the deep reasoning capabilities of Audio-Language Models (ALMs) across massive multi-disciplinary tasks. MMAR comprises 1,000 meticulously curated audio-question-answer triplets, collected from real-world internet videos and refined through iterative error corrections and quality checks to ensure high quality. Unlike existing benchmarks that are limited to specific domains of sound, music, or speech, MMAR extends them to a broad spectrum of real-world audio scenarios, including mixed-modality combinations of sound, music, and speech. Each question in MMAR is hierarchically categorized across four reasoning layers: Signal, Perception, Semantic, and Cultural, with additional sub-categories within each layer to reflect task diversity and complexity. To further foster research in this area, we annotate every question with a Chain-of-Thought (CoT) rationale to promote future advancements in audio reasoning. Each item in the benchmark demands multi-step deep reasoning beyond surface-level understanding. Moreover, a part of the questions requires graduate-level perceptual and domain-specific knowledge, elevating the benchmark's difficulty and depth. We evaluate MMAR using a broad set of models, including Large Audio-Language Models (LALMs), Large Audio Reasoning Models (LARMs), Omni Language Models (OLMs), Large Language Models (LLMs), and Large Reasoning Models (LRMs), with audio caption inputs. The performance of these models on MMAR highlights the benchmark's challenging nature, and our analysis further reveals critical limitations of understanding and reasoning capabilities among current models. We hope MMAR will serve as a catalyst for future advances in this important but little-explored area. 

---
# Evaluatiing the efficacy of LLM Safety Solutions : The Palit Benchmark Dataset 

**Authors**: Sayon Palit, Daniel Woods  

**Link**: [PDF](https://arxiv.org/pdf/2505.13028)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into critical systems in industries like healthcare and finance. Users can often submit queries to LLM-enabled chatbots, some of which can enrich responses with information retrieved from internal databases storing sensitive data. This gives rise to a range of attacks in which a user submits a malicious query and the LLM-system outputs a response that creates harm to the owner, such as leaking internal data or creating legal liability by harming a third-party. While security tools are being developed to counter these threats, there is little formal evaluation of their effectiveness and usability. This study addresses this gap by conducting a thorough comparative analysis of LLM security tools. We identified 13 solutions (9 closed-source, 4 open-source), but only 7 were evaluated due to a lack of participation by proprietary model this http URL evaluate, we built a benchmark dataset of malicious prompts, and evaluate these tools performance against a baseline LLM model (ChatGPT-3.5-Turbo). Our results show that the baseline model has too many false positives to be used for this task. Lakera Guard and ProtectAI LLM Guard emerged as the best overall tools showcasing the tradeoff between usability and performance. The study concluded with recommendations for greater transparency among closed source providers, improved context-aware detections, enhanced open-source engagement, increased user awareness, and the adoption of more representative performance metrics. 

---
# Fractured Chain-of-Thought Reasoning 

**Authors**: Baohao Liao, Hanze Dong, Yuhui Xu, Doyen Sahoo, Christof Monz, Junnan Li, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12992)  

**Abstract**: Inference-time scaling techniques have significantly bolstered the reasoning capabilities of large language models (LLMs) by harnessing additional computational effort at inference without retraining. Similarly, Chain-of-Thought (CoT) prompting and its extension, Long CoT, improve accuracy by generating rich intermediate reasoning trajectories, but these approaches incur substantial token costs that impede their deployment in latency-sensitive settings. In this work, we first show that truncated CoT, which stops reasoning before completion and directly generates the final answer, often matches full CoT sampling while using dramatically fewer tokens. Building on this insight, we introduce Fractured Sampling, a unified inference-time strategy that interpolates between full CoT and solution-only sampling along three orthogonal axes: (1) the number of reasoning trajectories, (2) the number of final solutions per trajectory, and (3) the depth at which reasoning traces are truncated. Through extensive experiments on five diverse reasoning benchmarks and several model scales, we demonstrate that Fractured Sampling consistently achieves superior accuracy-cost trade-offs, yielding steep log-linear scaling gains in Pass@k versus token budget. Our analysis reveals how to allocate computation across these dimensions to maximize performance, paving the way for more efficient and scalable LLM reasoning. 

---
# Leveraging LLM Inconsistency to Boost Pass@k Performance 

**Authors**: Uri Dalal, Meirav Segal, Zvika Ben-Haim, Dan Lahav, Omer Nevo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12938)  

**Abstract**: Large language models (LLMs) achieve impressive abilities in numerous domains, but exhibit inconsistent performance in response to minor input changes. Rather than view this as a drawback, in this paper we introduce a novel method for leveraging models' inconsistency to boost Pass@k performance. Specifically, we present a "Variator" agent that generates k variants of a given task and submits one candidate solution for each one. Our variant generation approach is applicable to a wide range of domains as it is task agnostic and compatible with free-form inputs. We demonstrate the efficacy of our agent theoretically using a probabilistic model of the inconsistency effect, and show empirically that it outperforms the baseline on the APPS dataset. Furthermore, we establish that inconsistency persists even in frontier reasoning models across coding and cybersecurity domains, suggesting our method is likely to remain relevant for future model generations. 

---
# AutoGEEval: A Multimodal and Automated Framework for Geospatial Code Generation on GEE with Large Language Models 

**Authors**: Shuyang Hou, Zhangxiao Shen, Huayi Wu, Jianyuan Liang, Haoyue Jiao, Yaxian Qing, Xiaopu Zhang, Xu Li, Zhipeng Gui, Xuefeng Guan, Longgang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12900)  

**Abstract**: Geospatial code generation is emerging as a key direction in the integration of artificial intelligence and geoscientific analysis. However, there remains a lack of standardized tools for automatic evaluation in this domain. To address this gap, we propose AutoGEEval, the first multimodal, unit-level automated evaluation framework for geospatial code generation tasks on the Google Earth Engine (GEE) platform powered by large language models (LLMs). Built upon the GEE Python API, AutoGEEval establishes a benchmark suite (AutoGEEval-Bench) comprising 1325 test cases that span 26 GEE data types. The framework integrates both question generation and answer verification components to enable an end-to-end automated evaluation pipeline-from function invocation to execution validation. AutoGEEval supports multidimensional quantitative analysis of model outputs in terms of accuracy, resource consumption, execution efficiency, and error types. We evaluate 18 state-of-the-art LLMs-including general-purpose, reasoning-augmented, code-centric, and geoscience-specialized models-revealing their performance characteristics and potential optimization pathways in GEE code generation. This work provides a unified protocol and foundational resource for the development and assessment of geospatial code generation models, advancing the frontier of automated natural language to domain-specific code translation. 

---
# TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios 

**Authors**: Shaohang Wei, Wei Li, Feifan Song, Wen Luo, Tianyi Zhuang, Haochen Tan, Zhijiang Guo, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12891)  

**Abstract**: Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at this https URL , and the dataset is available at this https URL . 

---
# Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective 

**Authors**: Zhongxiang Sun, Qipeng Wang, Haoyu Wang, Xiao Zhang, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12886)  

**Abstract**: Large Reasoning Models (LRMs) have shown impressive capabilities in multi-step reasoning tasks. However, alongside these successes, a more deceptive form of model error has emerged--Reasoning Hallucination--where logically coherent but factually incorrect reasoning traces lead to persuasive yet faulty conclusions. Unlike traditional hallucinations, these errors are embedded within structured reasoning, making them more difficult to detect and potentially more harmful. In this work, we investigate reasoning hallucinations from a mechanistic perspective. We propose the Reasoning Score, which quantifies the depth of reasoning by measuring the divergence between logits obtained from projecting late layers of LRMs to the vocabulary space, effectively distinguishing shallow pattern-matching from genuine deep reasoning. Using this score, we conduct an in-depth analysis on the ReTruthQA dataset and identify two key reasoning hallucination patterns: early-stage fluctuation in reasoning depth and incorrect backtracking to flawed prior steps. These insights motivate our Reasoning Hallucination Detection (RHD) framework, which achieves state-of-the-art performance across multiple domains. To mitigate reasoning hallucinations, we further introduce GRPO-R, an enhanced reinforcement learning algorithm that incorporates step-level deep reasoning rewards via potential-based shaping. Our theoretical analysis establishes stronger generalization guarantees, and experiments demonstrate improved reasoning quality and reduced hallucination rates. 

---
# Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks? 

**Authors**: Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, Ronghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12871)  

**Abstract**: Low rank adaptation (LoRA) has emerged as a prominent technique for fine-tuning large language models (LLMs) thanks to its superb efficiency gains over previous methods. While extensive studies have examined the performance and structural properties of LoRA, its behavior upon training-time attacks remain underexplored, posing significant security risks. In this paper, we theoretically investigate the security implications of LoRA's low-rank structure during fine-tuning, in the context of its robustness against data poisoning and backdoor attacks. We propose an analytical framework that models LoRA's training dynamics, employs the neural tangent kernel to simplify the analysis of the training process, and applies information theory to establish connections between LoRA's low rank structure and its vulnerability against training-time attacks. Our analysis indicates that LoRA exhibits better robustness to backdoor attacks than full fine-tuning, while becomes more vulnerable to untargeted data poisoning due to its over-simplified information geometry. Extensive experimental evaluations have corroborated our theoretical findings. 

---
# GEM: Gaussian Embedding Modeling for Out-of-Distribution Detection in GUI Agents 

**Authors**: Zheng Wu, Pengzhou Cheng, Zongru Wu, Lingzhong Dong, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12842)  

**Abstract**: Graphical user interface (GUI) agents have recently emerged as an intriguing paradigm for human-computer interaction, capable of automatically executing user instructions to operate intelligent terminal devices. However, when encountering out-of-distribution (OOD) instructions that violate environmental constraints or exceed the current capabilities of agents, GUI agents may suffer task breakdowns or even pose security threats. Therefore, effective OOD detection for GUI agents is essential. Traditional OOD detection methods perform suboptimally in this domain due to the complex embedding space and evolving GUI environments. In this work, we observe that the in-distribution input semantic space of GUI agents exhibits a clustering pattern with respect to the distance from the centroid. Based on the finding, we propose GEM, a novel method based on fitting a Gaussian mixture model over input embedding distances extracted from the GUI Agent that reflect its capability boundary. Evaluated on eight datasets spanning smartphones, computers, and web browsers, our method achieves an average accuracy improvement of 23.70\% over the best-performing baseline. Analysis verifies the generalization ability of our method through experiments on nine different backbones. The codes are available at this https URL. 

---
# Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization 

**Authors**: Sunghwan Kim, Dongjin Kang, Taeyoon Kwon, Hyungjoo Chae, Dongha Lee, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12763)  

**Abstract**: Reward models (RMs) play a crucial role in reinforcement learning from human feedback (RLHF), aligning model behavior with human preferences. However, existing benchmarks for reward models show a weak correlation with the performance of optimized policies, suggesting that they fail to accurately assess the true capabilities of RMs. To bridge this gap, we explore several evaluation designs through the lens of reward overoptimization\textemdash a phenomenon that captures both how well the reward model aligns with human preferences and the dynamics of the learning signal it provides to the policy. The results highlight three key findings on how to construct a reliable benchmark: (i) it is important to minimize differences between chosen and rejected responses beyond correctness, (ii) evaluating reward models requires multiple comparisons across a wide range of chosen and rejected responses, and (iii) given that reward models encounter responses with diverse representations, responses should be sourced from a variety of models. However, we also observe that a extremely high correlation with degree of overoptimization leads to comparatively lower correlation with certain downstream performance. Thus, when designing a benchmark, it is desirable to use the degree of overoptimization as a useful tool, rather than the end goal. 

---
# Bullying the Machine: How Personas Increase LLM Vulnerability 

**Authors**: Ziwei Xu, Udit Sanghi, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2505.12692)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in interactions where they are prompted to adopt personas. This paper investigates whether such persona conditioning affects model safety under bullying, an adversarial manipulation that applies psychological pressures in order to force the victim to comply to the attacker. We introduce a simulation framework in which an attacker LLM engages a victim LLM using psychologically grounded bullying tactics, while the victim adopts personas aligned with the Big Five personality traits. Experiments using multiple open-source LLMs and a wide range of adversarial goals reveal that certain persona configurations -- such as weakened agreeableness or conscientiousness -- significantly increase victim's susceptibility to unsafe outputs. Bullying tactics involving emotional or sarcastic manipulation, such as gaslighting and ridicule, are particularly effective. These findings suggest that persona-driven interaction introduces a novel vector for safety risks in LLMs and highlight the need for persona-aware safety evaluation and alignment strategies. 

---
# Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities 

**Authors**: Haoyu Zhao, Yihan Geng, Shange Tang, Yong Lin, Bohan Lyu, Hongzhou Lin, Chi Jin, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2505.12680)  

**Abstract**: LLM-based formal proof assistants (e.g., in Lean) hold great promise for automating mathematical discovery. But beyond syntactic correctness, do these systems truly understand mathematical structure as humans do? We investigate this question through the lens of mathematical inequalities -- a fundamental tool across many domains. While modern provers can solve basic inequalities, we probe their ability to handle human-intuitive compositionality. We introduce Ineq-Comp, a benchmark built from elementary inequalities through systematic transformations, including variable duplication, algebraic rewriting, and multi-step composition. Although these problems remain easy for humans, we find that most provers -- including Goedel, STP, and Kimina-7B -- struggle significantly. DeepSeek-Prover-V2-7B shows relative robustness -- possibly because it is trained to decompose the problems into sub-problems -- but still suffers a 20\% performance drop (pass@32). Strikingly, performance remains poor for all models even when formal proofs of the constituent parts are provided in context, revealing that the source of weakness is indeed in compositional reasoning. Our results expose a persisting gap between the generalization behavior of current AI provers and human mathematical intuition. 

---
# Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents 

**Authors**: Yunseok Jang, Yeda Song, Sungryull Sohn, Lajanugen Logeswaran, Tiange Luo, Dong-Ki Kim, Kyunghoon Bae, Honglak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.12632)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and Vision-Language Models (VLMs) have sparked significant interest in developing GUI visual agents. We introduce MONDAY (Mobile OS Navigation Task Dataset for Agents from YouTube), a large-scale dataset of 313K annotated frames from 20K instructional videos capturing diverse real-world mobile OS navigation across multiple platforms. Models that include MONDAY in their pre-training phases demonstrate robust cross-platform generalization capabilities, consistently outperforming models trained on existing single OS datasets while achieving an average performance gain of 18.11%p on an unseen mobile OS platform. To enable continuous dataset expansion as mobile platforms evolve, we present an automated framework that leverages publicly available video content to create comprehensive task datasets without manual annotation. Our framework comprises robust OCR-based scene detection (95.04% F1score), near-perfect UI element detection (99.87% hit ratio), and novel multi-step action identification to extract reliable action sequences across diverse interface configurations. We contribute both the MONDAY dataset and our automated collection framework to facilitate future research in mobile OS navigation. 

---
# Enhancing Latent Computation in Transformers with Latent Tokens 

**Authors**: Yuchang Sun, Yanxi Chen, Yaliang Li, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.12629)  

**Abstract**: Augmenting large language models (LLMs) with auxiliary tokens has emerged as a promising strategy for enhancing model performance. In this work, we introduce a lightweight method termed latent tokens; these are dummy tokens that may be non-interpretable in natural language but steer the autoregressive decoding process of a Transformer-based LLM via the attention mechanism. The proposed latent tokens can be seamlessly integrated with a pre-trained Transformer, trained in a parameter-efficient manner, and applied flexibly at inference time, while adding minimal complexity overhead to the existing infrastructure of standard Transformers. We propose several hypotheses about the underlying mechanisms of latent tokens and design synthetic tasks accordingly to verify them. Numerical results confirm that the proposed method noticeably outperforms the baselines, particularly in the out-of-distribution generalization scenarios, highlighting its potential in improving the adaptability of LLMs. 

---
# mCLM: A Function-Infused and Synthesis-Friendly Modular Chemical Language Model 

**Authors**: Carl Edwards, Chi Han, Gawon Lee, Thao Nguyen, Bowen Jin, Chetan Kumar Prasad, Sara Szymkuć, Bartosz A. Grzybowski, Ying Diao, Jiawei Han, Ge Liu, Hao Peng, Martin D. Burke, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.12565)  

**Abstract**: Despite their ability to understand chemical knowledge and accurately generate sequential representations, large language models (LLMs) remain limited in their capacity to propose novel molecules with drug-like properties. In addition, the molecules that LLMs propose can often be challenging to make in the lab. To more effectively enable the discovery of functional small molecules, LLMs need to learn a molecular language. However, LLMs are currently limited by encoding molecules from atoms. In this paper, we argue that just like tokenizing texts into (sub-)word tokens instead of characters, molecules should be decomposed and reassembled at the level of functional building blocks, i.e., parts of molecules that bring unique functions and serve as effective building blocks for real-world automated laboratory synthesis. This motivates us to propose mCLM, a modular Chemical-Language Model tokenizing molecules into building blocks and learning a bilingual language model of both natural language descriptions of functions and molecule building blocks. By reasoning on such functional building blocks, mCLM guarantees to generate efficiently synthesizable molecules thanks to recent progress in block-based chemistry, while also improving the functions of molecules in a principled manner. In experiments on 430 FDA-approved drugs, we find mCLM capable of significantly improving 5 out of 6 chemical functions critical to determining drug potentials. More importantly, mCLM can reason on multiple functions and improve the FDA-rejected drugs (``fallen angels'') over multiple iterations to greatly improve their shortcomings. 

---
# UFO-RL: Uncertainty-Focused Optimization for Efficient Reinforcement Learning Data Selection 

**Authors**: Yang Zhao, Kai Xiong, Xiao Ding, Li Du, YangouOuyang, Zhouhao Sun, Jiannan Guan, Wenbin Zhang, Bin Liu, Dong Hu, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12457)  

**Abstract**: Scaling RL for LLMs is computationally expensive, largely due to multi-sampling for policy optimization and evaluation, making efficient data selection crucial. Inspired by the Zone of Proximal Development (ZPD) theory, we hypothesize LLMs learn best from data within their potential comprehension zone. Addressing the limitation of conventional, computationally intensive multi-sampling methods for data assessment, we introduce UFO-RL. This novel framework uses a computationally efficient single-pass uncertainty estimation to identify informative data instances, achieving up to 185x faster data evaluation. UFO-RL leverages this metric to select data within the estimated ZPD for training. Experiments show that training with just 10% of data selected by UFO-RL yields performance comparable to or surpassing full-data training, reducing overall training time by up to 16x while enhancing stability and generalization. UFO-RL offers a practical and highly efficient strategy for scaling RL fine-tuning of LLMs by focusing learning on valuable data. 

---
# IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems 

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2505.12442)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses. 

---
# MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks 

**Authors**: Yinghao Zhu, Ziyi He, Haoran Hu, Xiaochen Zheng, Xichen Zhang, Zixiang Wang, Junyi Gao, Liantao Ma, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12371)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has stimulated interest in multi-agent collaboration for addressing complex medical tasks. However, the practical advantages of multi-agent collaboration approaches remain insufficiently understood. Existing evaluations often lack generalizability, failing to cover diverse tasks reflective of real-world clinical practice, and frequently omit rigorous comparisons against both single-LLM-based and established conventional methods. To address this critical gap, we introduce MedAgentBoard, a comprehensive benchmark for the systematic evaluation of multi-agent collaboration, single-LLM, and conventional approaches. MedAgentBoard encompasses four diverse medical task categories: (1) medical (visual) question answering, (2) lay summary generation, (3) structured Electronic Health Record (EHR) predictive modeling, and (4) clinical workflow automation, across text, medical images, and structured EHR data. Our extensive experiments reveal a nuanced landscape: while multi-agent collaboration demonstrates benefits in specific scenarios, such as enhancing task completeness in clinical workflow automation, it does not consistently outperform advanced single LLMs (e.g., in textual medical QA) or, critically, specialized conventional methods that generally maintain better performance in tasks like medical VQA and EHR-based prediction. MedAgentBoard offers a vital resource and actionable insights, emphasizing the necessity of a task-specific, evidence-based approach to selecting and developing AI solutions in medicine. It underscores that the inherent complexity and overhead of multi-agent collaboration must be carefully weighed against tangible performance gains. All code, datasets, detailed prompts, and experimental results are open-sourced at this https URL. 

---
# Towards Visuospatial Cognition via Hierarchical Fusion of Visual Experts 

**Authors**: Qi Feng, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2505.12363)  

**Abstract**: While Multimodal Large Language Models (MLLMs) excel at general vision-language tasks, visuospatial cognition - reasoning about spatial layouts, relations, and dynamics - remains a significant challenge. Existing models often lack the necessary architectural components and specialized training data for fine-grained spatial understanding. We introduce ViCA2 (Visuospatial Cognitive Assistant 2), a novel MLLM designed to enhance spatial reasoning. ViCA2 features a dual vision encoder architecture integrating SigLIP for semantics and Hiera for spatial structure, coupled with a token ratio control mechanism for efficiency. We also developed ViCA-322K, a new large-scale dataset with over 322,000 spatially grounded question-answer pairs for targeted instruction tuning. On the challenging VSI-Bench benchmark, our ViCA2-7B model achieves a state-of-the-art average score of 56.8, significantly surpassing larger open-source models (e.g., LLaVA-NeXT-Video-72B, 40.9) and leading proprietary models (Gemini-1.5 Pro, 45.4). This demonstrates the effectiveness of our approach in achieving strong visuospatial intelligence with a compact model. We release ViCA2, its codebase, and the ViCA-322K dataset to facilitate further research. 

---
# Visuospatial Cognitive Assistant 

**Authors**: Qi Feng, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2505.12312)  

**Abstract**: Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence. 

---
# LogicOCR: Do Your Large Multimodal Models Excel at Logical Reasoning on Text-Rich Images? 

**Authors**: Maoyuan Ye, Jing Zhang, Juhua Liu, Bo Du, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12307)  

**Abstract**: Recent advances in Large Multimodal Models (LMMs) have significantly improved their reasoning and Optical Character Recognition (OCR) capabilities. However, their performance on complex logical reasoning tasks involving text-rich images remains underexplored. To bridge this gap, we introduce LogicOCR, a benchmark comprising 1,100 multiple-choice questions designed to evaluate LMMs' logical reasoning abilities on text-rich images, while minimizing reliance on domain-specific knowledge (e.g., mathematics). We construct LogicOCR by curating a text corpus from the Chinese National Civil Servant Examination and develop a scalable, automated pipeline to convert it into multimodal samples. First, we design prompt templates to steer GPT-Image-1 to generate images with diverse backgrounds, interleaved text-illustration layouts, and varied fonts, ensuring contextual relevance and visual realism. Then, the generated images are manually verified, with low-quality examples discarded. We evaluate a range of representative open-source and proprietary LMMs under both Chain-of-Thought (CoT) and direct-answer settings. Our multi-dimensional analysis reveals key insights, such as the impact of test-time scaling, input modality differences, and sensitivity to visual-text orientation. Notably, LMMs still lag in multimodal reasoning compared to text-only inputs, indicating that they have not fully bridged visual reading with reasoning. We hope LogicOCR will serve as a valuable resource for advancing multimodal reasoning research. The dataset is available at this https URL. 

---
# Beyond Single-Point Judgment: Distribution Alignment for LLM-as-a-Judge 

**Authors**: Luyu Chen, Zeyu Zhang, Haoran Tan, Quanyu Dai, Hao Yang, Zhenhua Dong, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12301)  

**Abstract**: LLMs have emerged as powerful evaluators in the LLM-as-a-Judge paradigm, offering significant efficiency and flexibility compared to human judgments. However, previous methods primarily rely on single-point evaluations, overlooking the inherent diversity and uncertainty in human evaluations. This approach leads to information loss and decreases the reliability of evaluations. To address this limitation, we propose a novel training framework that explicitly aligns the LLM-generated judgment distribution with empirical human distributions. Specifically, we propose a distributional alignment objective based on KL divergence, combined with an auxiliary cross-entropy regularization to stabilize the training process. Furthermore, considering that empirical distributions may derive from limited human annotations, we incorporate adversarial training to enhance model robustness against distribution perturbations. Extensive experiments across various LLM backbones and evaluation tasks demonstrate that our framework significantly outperforms existing closed-source LLMs and conventional single-point alignment methods, with improved alignment quality, evaluation accuracy, and robustness. 

---
# Efficient RL Training for Reasoning Models via Length-Aware Optimization 

**Authors**: Danlong Yuan, Tian Xie, Shaohan Huang, Zhuocheng Gong, Huishuai Zhang, Chong Luo, Furu Wei, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12284)  

**Abstract**: Large reasoning models, such as OpenAI o1 or DeepSeek R1, have demonstrated remarkable performance on reasoning tasks but often incur a long reasoning path with significant memory and time costs. Existing methods primarily aim to shorten reasoning paths by introducing additional training data and stages. In this paper, we propose three critical reward designs integrated directly into the reinforcement learning process of large reasoning models, which reduce the response length without extra training stages. Experiments on four settings show that our method significantly decreases response length while maintaining or even improving performance. Specifically, in a logic reasoning setting, we achieve a 40% reduction in response length averaged by steps alongside a 14% gain in performance. For math problems, we reduce response length averaged by steps by 33% while preserving performance. 

---
# Vague Knowledge: Evidence from Analyst Reports 

**Authors**: Kerry Xiao, Amy Zang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12269)  

**Abstract**: People in the real world often possess vague knowledge of future payoffs, for which quantification is not feasible or desirable. We argue that language, with differing ability to convey vague information, plays an important but less known-role in subjective expectations. Empirically, we find that in their reports, analysts include useful information in linguistic expressions but not numerical forecasts. Specifically, the textual tone of analyst reports has predictive power for forecast errors and subsequent revisions in numerical forecasts, and this relation becomes stronger when analyst's language is vaguer, when uncertainty is higher, and when analysts are busier. Overall, our theory and evidence suggest that some useful information is vaguely known and only communicated through language. 

---
# LightRetriever: A LLM-based Hybrid Retrieval Architecture with 1000x Faster Query Inference 

**Authors**: Guangyuan Ma, Yongliang Ma, Xuanrui Gou, Zhenpeng Su, Ming Zhou, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12260)  

**Abstract**: Large Language Models (LLMs)-based hybrid retrieval uses LLMs to encode queries and documents into low-dimensional dense or high-dimensional sparse vectors. It retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based hybrid retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full-sized LLM on an H800 GPU, our approach achieves over a 1000x speedup for query inference with GPU acceleration, and even a 20x speedup without GPU. Experiments on large-scale retrieval benchmarks demonstrate that our method generalizes well across diverse retrieval tasks, retaining an average of 95% full-sized performance. 

---
# Reward Inside the Model: A Lightweight Hidden-State Reward Model for LLM's Best-of-N sampling 

**Authors**: Jizhou Guo, Zhaomin Wu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12225)  

**Abstract**: High-quality reward models are crucial for unlocking the reasoning potential of large language models (LLMs), with best-of-N voting demonstrating significant performance gains. However, current reward models, which typically operate on the textual output of LLMs, are computationally expensive and parameter-heavy, limiting their real-world applications. We introduce the Efficient Linear Hidden State Reward (ELHSR) model - a novel, highly parameter-efficient approach that leverages the rich information embedded in LLM hidden states to address these issues. ELHSR systematically outperform baselines with less than 0.005% of the parameters of baselines, requiring only a few samples for training. ELHSR also achieves orders-of-magnitude efficiency improvement with significantly less time and fewer FLOPs per sample than baseline reward models. Moreover, ELHSR exhibits robust performance even when trained only on logits, extending its applicability to some closed-source LLMs. In addition, ELHSR can also be combined with traditional reward models to achieve additional performance gains. 

---
# Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering 

**Authors**: Marco Valentino, Geonhee Kim, Dhairya Dalal, Zhixue Zhao, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2505.12189)  

**Abstract**: Large language models (LLMs) frequently demonstrate reasoning limitations, often conflating content plausibility (i.e., material inference) with logical validity (i.e., formal inference). This can result in biased inferences, where plausible arguments are incorrectly deemed logically valid or vice versa. Mitigating this limitation is critical, as it undermines the trustworthiness and generalizability of LLMs in applications that demand rigorous logical consistency. This paper investigates the problem of mitigating content biases on formal reasoning through activation steering. Specifically, we curate a controlled syllogistic reasoning dataset to disentangle formal validity from content plausibility. After localising the layers responsible for formal and material inference, we investigate contrastive activation steering methods for test-time interventions. An extensive empirical analysis on different LLMs reveals that contrastive steering consistently supports linear control over content biases. However, we observe that a static approach is insufficient for improving all the tested models. We then leverage the possibility to control content effects by dynamically determining the value of the steering parameters via fine-grained conditional methods. We found that conditional steering is effective on unresponsive models, achieving up to 15% absolute improvement in formal reasoning accuracy with a newly introduced kNN-based method (K-CAST). Finally, additional experiments reveal that steering for content effects is robust to prompt variations, incurs minimal side effects on language modeling capabilities, and can partially generalize to out-of-distribution reasoning tasks. Practically, this paper demonstrates that activation-level interventions can offer a scalable strategy for enhancing the robustness of LLMs, contributing towards more systematic and unbiased formal reasoning. 

---
# EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective 

**Authors**: Sen Fang, Weiyuan Ding, Bowen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12185)  

**Abstract**: Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop. 

---
# LLM-BABYBENCH: Understanding and Evaluating Grounded Planning and Reasoning in LLMs 

**Authors**: Omar Choukrani, Idriss Malek, Daniil Orel, Zhuohan Xie, Zangir Iklassov, Martin Takáč, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12135)  

**Abstract**: Assessing the capacity of Large Language Models (LLMs) to plan and reason within the constraints of interactive environments is crucial for developing capable AI agents. We introduce $\textbf{LLM-BabyBench}$, a new benchmark suite designed specifically for this purpose. Built upon a textual adaptation of the procedurally generated BabyAI grid world, this suite evaluates LLMs on three fundamental aspects of grounded intelligence: (1) predicting the consequences of actions on the environment state ($\textbf{Predict}$ task), (2) generating sequences of low-level actions to achieve specified objectives ($\textbf{Plan}$ task), and (3) decomposing high-level instructions into coherent subgoal sequences ($\textbf{Decompose}$ task). We detail the methodology for generating the three corresponding datasets ($\texttt{LLM-BabyBench-Predict}$, $\texttt{-Plan}$, $\texttt{-Decompose}$) by extracting structured information from an expert agent operating within the text-based environment. Furthermore, we provide a standardized evaluation harness and metrics, including environment interaction for validating generated plans, to facilitate reproducible assessment of diverse LLMs. Initial baseline results highlight the challenges posed by these grounded reasoning tasks. The benchmark suite, datasets, data generation code, and evaluation code are made publicly available ($\href{this https URL}{\text{GitHub}}$, $\href{this https URL}{\text{HuggingFace}}$). 

---
# Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents 

**Authors**: Tiannuo Yang, Zebin Yao, Bowen Jin, Lixiao Cui, Yusen Li, Gang Wang, Xiaoguang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12065)  

**Abstract**: Large Language Model (LLM)-based search agents have shown remarkable capabilities in solving complex tasks by dynamically decomposing problems and addressing them through interleaved reasoning and retrieval. However, this interleaved paradigm introduces substantial efficiency bottlenecks. First, we observe that both highly accurate and overly approximate retrieval methods degrade system efficiency: exact search incurs significant retrieval overhead, while coarse retrieval requires additional reasoning steps during generation. Second, we identify inefficiencies in system design, including improper scheduling and frequent retrieval stalls, which lead to cascading latency -- where even minor delays in retrieval amplify end-to-end inference time. To address these challenges, we introduce SearchAgent-X, a high-efficiency inference framework for LLM-based search agents. SearchAgent-X leverages high-recall approximate retrieval and incorporates two key techniques: priority-aware scheduling and non-stall retrieval. Extensive experiments demonstrate that SearchAgent-X consistently outperforms state-of-the-art systems such as vLLM and HNSW-based retrieval across diverse tasks, achieving up to 3.4$\times$ higher throughput and 5$\times$ lower latency, without compromising generation quality. SearchAgent-X is available at this https URL. 

---
# Tiny QA Benchmark++: Ultra-Lightweight, Synthetic Multilingual Dataset Generation & Smoke-Tests for Continuous LLM Evaluation 

**Authors**: Vincent Koc  

**Link**: [PDF](https://arxiv.org/pdf/2505.12058)  

**Abstract**: Tiny QA Benchmark++ (TQB++) presents an ultra-lightweight, multilingual smoke-test suite designed to give large-language-model (LLM) pipelines a unit-test style safety net dataset that runs in seconds with minimal cost. Born out of the tight feedback-loop demands building the Comet Opik prompt-optimization SDK, where waiting on heavyweight benchmarks breaks developer flow. TQB++ couples a 52-item English gold set (less than 20 kB) with a tiny synthetic-data generator pypi package built on provider-agnostic LiteLLM. The generator lets practitioners mint their own tiny packs in any language, domain, or difficulty, while ten ready-made packs already cover Arabic, Chinese, French, German, Japanese, Korean, Portuguese, Russian, Spanish, and Turkish. Every dataset ships with Croissant metadata and plug-and-play files for OpenAI-Evals, LangChain, and standard CI tools, so teams can drop deterministic micro-benchmarks directly into pull-request gates, prompt-engineering loops, and production dashboards without touching GPU budgets. A complete TQB++ run adds only a few seconds to pipeline latency yet reliably flags prompt-template errors, tokenizer drift, and fine-tuning side-effects long before full-scale suites like MMLU or BIG-Bench would finish configuring. The entire framework is released to accelerate continuous, resource-efficient quality assurance across the generative-AI ecosystem. 

---
# AI-Driven Automation Can Become the Foundation of Next-Era Science of Science Research 

**Authors**: Renqi Chen, Haoyang Su, Shixiang Tang, Zhenfei Yin, Qi Wu, Hui Li, Ye Sun, Nanqing Dong, Wanli Ouyang, Philip Torr  

**Link**: [PDF](https://arxiv.org/pdf/2505.12039)  

**Abstract**: The Science of Science (SoS) explores the mechanisms underlying scientific discovery, and offers valuable insights for enhancing scientific efficiency and fostering innovation. Traditional approaches often rely on simplistic assumptions and basic statistical tools, such as linear regression and rule-based simulations, which struggle to capture the complexity and scale of modern research ecosystems. The advent of artificial intelligence (AI) presents a transformative opportunity for the next generation of SoS, enabling the automation of large-scale pattern discovery and uncovering insights previously unattainable. This paper offers a forward-looking perspective on the integration of Science of Science with AI for automated research pattern discovery and highlights key open challenges that could greatly benefit from AI. We outline the advantages of AI over traditional methods, discuss potential limitations, and propose pathways to overcome them. Additionally, we present a preliminary multi-agent system as an illustrative example to simulate research societies, showcasing AI's ability to replicate real-world research patterns and accelerate progress in Science of Science research. 

---
# Introduction to Analytical Software Engineering Design Paradigm 

**Authors**: Tarik Houichime, Younes El Amrani  

**Link**: [PDF](https://arxiv.org/pdf/2505.11979)  

**Abstract**: As modern software systems expand in scale and complexity, the challenges associated with their modeling and formulation grow increasingly intricate. Traditional approaches often fall short in effectively addressing these complexities, particularly in tasks such as design pattern detection for maintenance and assessment, as well as code refactoring for optimization and long-term sustainability. This growing inadequacy underscores the need for a paradigm shift in how such challenges are approached and resolved. This paper presents Analytical Software Engineering (ASE), a novel design paradigm aimed at balancing abstraction, tool accessibility, compatibility, and scalability. ASE enables effective modeling and resolution of complex software engineering problems. The paradigm is evaluated through two frameworks Behavioral-Structural Sequences (BSS) and Optimized Design Refactoring (ODR), both developed in accordance with ASE principles. BSS offers a compact, language-agnostic representation of codebases to facilitate precise design pattern detection. ODR unifies artifact and solution representations to optimize code refactoring via heuristic algorithms while eliminating iterative computational overhead. By providing a structured approach to software design challenges, ASE lays the groundwork for future research in encoding and analyzing complex software metrics. 

---
# J1: Exploring Simple Test-Time Scaling for LLM-as-a-Judge 

**Authors**: Chi-Min Chan, Chunpu Xu, Jiaming Ji, Zhen Ye, Pengcheng Wen, Chunyang Jiang, Yaodong Yang, Wei Xue, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11875)  

**Abstract**: The current focus of AI research is shifting from emphasizing model training towards enhancing evaluation quality, a transition that is crucial for driving further advancements in AI systems. Traditional evaluation methods typically rely on reward models assigning scalar preference scores to outputs. Although effective, such approaches lack interpretability, leaving users often uncertain about why a reward model rates a particular response as high or low. The advent of LLM-as-a-Judge provides a more scalable and interpretable method of supervision, offering insights into the decision-making process. Moreover, with the emergence of large reasoning models, which consume more tokens for deeper thinking and answer refinement, scaling test-time computation in the LLM-as-a-Judge paradigm presents an avenue for further boosting performance and providing more interpretability through reasoning traces. In this paper, we introduce $\textbf{J1-7B}$, which is first supervised fine-tuned on reflection-enhanced datasets collected via rejection-sampling and subsequently trained using Reinforcement Learning (RL) with verifiable rewards. At inference time, we apply Simple Test-Time Scaling (STTS) strategies for additional performance improvement. Experimental results demonstrate that $\textbf{J1-7B}$ surpasses the previous state-of-the-art LLM-as-a-Judge by $ \textbf{4.8}$\% and exhibits a $ \textbf{5.1}$\% stronger scaling trend under STTS. Additionally, we present three key findings: (1) Existing LLM-as-a-Judge does not inherently exhibit such scaling trend. (2) Model simply fine-tuned on reflection-enhanced datasets continues to demonstrate similarly weak scaling behavior. (3) Significant scaling trend emerges primarily during the RL phase, suggesting that effective STTS capability is acquired predominantly through RL training. 

---
# Fair-PP: A Synthetic Dataset for Aligning LLM with Personalized Preferences of Social Equity 

**Authors**: Qi Zhou, Jie Zhang, Dongxia Wang, Qiang Liu, Tianlin Li, Jin Song Dong, Wenhai Wang, Qing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11861)  

**Abstract**: Human preference plays a crucial role in the refinement of large language models (LLMs). However, collecting human preference feedback is costly and most existing datasets neglect the correlation between personalization and preferences. To address this issue, we introduce Fair-PP, a synthetic dataset of personalized preferences targeting social equity, derived from real-world social survey data, which includes 28 social groups, 98 equity topics, and 5 personal preference dimensions. Leveraging GPT-4o-mini, we engage in role-playing based on seven representative persona portrayals guided by existing social survey data, yielding a total of 238,623 preference records. Through Fair-PP, we also contribute (i) An automated framework for generating preference data, along with a more fine-grained dataset of personalized preferences; (ii) analysis of the positioning of the existing mainstream LLMs across five major global regions within the personalized preference space; and (iii) a sample reweighting method for personalized preference alignment, enabling alignment with a target persona while maximizing the divergence from other personas. Empirical experiments show our method outperforms the baselines. 

---
# Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs 

**Authors**: Xuannan Liu, Zekun Li, Zheqi He, Peipei Li, Shuhan Xia, Xing Cui, Huaibo Huang, Xi Yang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2505.11842)  

**Abstract**: The increasing deployment of Large Vision-Language Models (LVLMs) raises safety concerns under potential malicious inputs. However, existing multimodal safety evaluations primarily focus on model vulnerabilities exposed by static image inputs, ignoring the temporal dynamics of video that may induce distinct safety risks. To bridge this gap, we introduce Video-SafetyBench, the first comprehensive benchmark designed to evaluate the safety of LVLMs under video-text attacks. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, each pairing a synthesized video with either a harmful query, which contains explicit malice, or a benign query, which appears harmless but triggers harmful behavior when interpreted alongside the video. To generate semantically accurate videos for safety evaluation, we design a controllable pipeline that decomposes video semantics into subject images (what is shown) and motion text (how it moves), which jointly guide the synthesis of query-relevant videos. To effectively evaluate uncertain or borderline harmful outputs, we propose RJScore, a novel LLM-based metric that incorporates the confidence of judge models and human-aligned decision threshold calibration. Extensive experiments show that benign-query video composition achieves average attack success rates of 67.2%, revealing consistent vulnerabilities to video-induced attacks. We believe Video-SafetyBench will catalyze future research into video-based safety evaluation and defense strategies. 

---
# VenusX: Unlocking Fine-Grained Functional Understanding of Proteins 

**Authors**: Yang Tan, Wenrui Gou, Bozitao Zhong, Liang Hong, Huiqun Yu, Bingxin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.11812)  

**Abstract**: Deep learning models have driven significant progress in predicting protein function and interactions at the protein level. While these advancements have been invaluable for many biological applications such as enzyme engineering and function annotation, a more detailed perspective is essential for understanding protein functional mechanisms and evaluating the biological knowledge captured by models. To address this demand, we introduce VenusX, the first large-scale benchmark for fine-grained functional annotation and function-based protein pairing at the residue, fragment, and domain levels. VenusX comprises three major task categories across six types of annotations, including residue-level binary classification, fragment-level multi-class classification, and pairwise functional similarity scoring for identifying critical active sites, binding sites, conserved sites, motifs, domains, and epitopes. The benchmark features over 878,000 samples curated from major open-source databases such as InterPro, BioLiP, and SAbDab. By providing mixed-family and cross-family splits at three sequence identity thresholds, our benchmark enables a comprehensive assessment of model performance on both in-distribution and out-of-distribution scenarios. For baseline evaluation, we assess a diverse set of popular and open-source models, including pre-trained protein language models, sequence-structure hybrids, structure-based methods, and alignment-based techniques. Their performance is reported across all benchmark datasets and evaluation settings using multiple metrics, offering a thorough comparison and a strong foundation for future research. Code and data are publicly available at this https URL. 

---
# Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors 

**Authors**: Jing Huang, Junyi Tao, Thomas Icard, Diyi Yang, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2505.11770)  

**Abstract**: Interpretability research now offers a variety of techniques for identifying abstract internal mechanisms in neural networks. Can such techniques be used to predict how models will behave on out-of-distribution examples? In this work, we provide a positive answer to this question. Through a diverse set of language modeling tasks--including symbol manipulation, knowledge retrieval, and instruction following--we show that the most robust features for correctness prediction are those that play a distinctive causal role in the model's behavior. Specifically, we propose two methods that leverage causal mechanisms to predict the correctness of model outputs: counterfactual simulation (checking whether key causal variables are realized) and value probing (using the values of those variables to make predictions). Both achieve high AUC-ROC in distribution and outperform methods that rely on causal-agnostic features in out-of-distribution settings, where predicting model behaviors is more crucial. Our work thus highlights a novel and significant application for internal causal analysis of language models. 

---
# Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders 

**Authors**: David Chanin, Tomáš Dulka, Adrià Garriga-Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2505.11756)  

**Abstract**: It is assumed that sparse autoencoders (SAEs) decompose polysemantic activations into interpretable linear directions, as long as the activations are composed of sparse linear combinations of underlying features. However, we find that if an SAE is more narrow than the number of underlying "true features" on which it is trained, and there is correlation between features, the SAE will merge components of correlated features together, thus destroying monosemanticity. In LLM SAEs, these two conditions are almost certainly true. This phenomenon, which we call feature hedging, is caused by SAE reconstruction loss, and is more severe the narrower the SAE. In this work, we introduce the problem of feature hedging and study it both theoretically in toy models and empirically in SAEs trained on LLMs. We suspect that feature hedging may be one of the core reasons that SAEs consistently underperform supervised baselines. Finally, we use our understanding of feature hedging to propose an improved variant of matryoshka SAEs. Our work shows there remain fundamental issues with SAEs, but we are hopeful that that highlighting feature hedging will catalyze future advances that allow SAEs to achieve their full potential of interpreting LLMs at scale. 

---
# Token-Level Uncertainty Estimation for Large Language Model Reasoning 

**Authors**: Tunyu Zhang, Haizhou Shi, Yibin Wang, Hengyi Wang, Xiaoxiao He, Zhuowei Li, Haoxian Chen, Ligong Han, Kai Xu, Huan Zhang, Dimitris Metaxas, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11737)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive capabilities, their output quality remains inconsistent across various application scenarios, making it difficult to identify trustworthy responses, especially in complex tasks requiring multi-step reasoning. In this paper, we propose a token-level uncertainty estimation framework to enable LLMs to self-assess and self-improve their generation quality in mathematical reasoning. Specifically, we introduce low-rank random weight perturbation to LLM decoding, generating predictive distributions that we use to estimate token-level uncertainties. We then aggregate these uncertainties to reflect semantic uncertainty of the generated sequences. Experiments on mathematical reasoning datasets of varying difficulty demonstrate that our token-level uncertainty metrics strongly correlate with answer correctness and model robustness. Additionally, we explore using uncertainty to directly enhance the model's reasoning performance through multiple generations and the particle filtering algorithm. Our approach consistently outperforms existing uncertainty estimation methods, establishing effective uncertainty estimation as a valuable tool for both evaluating and improving reasoning generation in LLMs. 

---
# Efficient Uncertainty Estimation via Distillation of Bayesian Large Language Models 

**Authors**: Harshil Vejendla, Haizhou Shi, Yibin Wang, Tunyu Zhang, Huan Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11731)  

**Abstract**: Recent advances in uncertainty estimation for Large Language Models (LLMs) during downstream adaptation have addressed key challenges of reliability and simplicity. However, existing Bayesian methods typically require multiple sampling iterations during inference, creating significant efficiency issues that limit practical deployment. In this paper, we investigate the possibility of eliminating the need for test-time sampling for LLM uncertainty estimation. Specifically, when given an off-the-shelf Bayesian LLM, we distill its aligned confidence into a non-Bayesian student LLM by minimizing the divergence between their predictive distributions. Unlike typical calibration methods, our distillation is carried out solely on the training dataset without the need of an additional validation dataset. This simple yet effective approach achieves N-times more efficient uncertainty estimation during testing, where N is the number of samples traditionally required by Bayesian LLMs. Our extensive experiments demonstrate that uncertainty estimation capabilities on training data can successfully generalize to unseen test data through our distillation technique, consistently producing results comparable to (or even better than) state-of-the-art Bayesian LLMs. 

---
# EnvInjection: Environmental Prompt Injection Attack to Multi-modal Web Agents 

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.11717)  

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. Environmental prompt injection attacks manipulate the environment to induce the web agent to perform a specific, attacker-chosen action--referred to as the target action. However, existing attacks suffer from limited effectiveness or stealthiness, or are impractical in real-world settings. In this work, we propose EnvInjection, a new attack that addresses these limitations. Our attack adds a perturbation to the raw pixel values of the rendered webpage, which can be implemented by modifying the webpage's source code. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the target action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple webpage datasets shows that EnvInjection is highly effective and significantly outperforms existing baselines. 

---
# Using Reinforcement Learning to Train Large Language Models to Explain Human Decisions 

**Authors**: Jian-Qiao Zhu, Hanbo Xie, Dilip Arumugam, Robert C. Wilson, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.11614)  

**Abstract**: A central goal of cognitive modeling is to develop models that not only predict human behavior but also provide insight into the underlying cognitive mechanisms. While neural network models trained on large-scale behavioral data often achieve strong predictive performance, they typically fall short in offering interpretable explanations of the cognitive processes they capture. In this work, we explore the potential of pretrained large language models (LLMs) to serve as dual-purpose cognitive models--capable of both accurate prediction and interpretable explanation in natural language. Specifically, we employ reinforcement learning with outcome-based rewards to guide LLMs toward generating explicit reasoning traces for explaining human risky choices. Our findings demonstrate that this approach produces high-quality explanations alongside strong quantitative predictions of human decisions. 

---
# Probing the Vulnerability of Large Language Models to Polysemantic Interventions 

**Authors**: Bofan Gong, Shiyang Lai, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.11611)  

**Abstract**: Polysemanticity -- where individual neurons encode multiple unrelated features -- is a well-known characteristic of large neural networks and remains a central challenge in the interpretability of language models. At the same time, its implications for model safety are also poorly understood. Leveraging recent advances in sparse autoencoders, we investigate the polysemantic structure of two small models (Pythia-70M and GPT-2-Small) and evaluate their vulnerability to targeted, covert interventions at the prompt, feature, token, and neuron levels. Our analysis reveals a consistent polysemantic topology shared across both models. Strikingly, we demonstrate that this structure can be exploited to mount effective interventions on two larger, black-box instruction-tuned models (LLaMA3.1-8B-Instruct and Gemma-2-9B-Instruct). These findings suggest not only the generalizability of the interventions but also point to a stable and transferable polysemantic structure that could potentially persist across architectures and training regimes. 

---
# Spectral Policy Optimization: Coloring your Incorrect Reasoning in GRPO 

**Authors**: Peter Chen, Xiaopeng Li, Ziniu Li, Xi Chen, Tianyi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.11595)  

**Abstract**: Reinforcement learning (RL) has demonstrated significant success in enhancing reasoning capabilities in large language models (LLMs). One of the most widely used RL methods is Group Relative Policy Optimization (GRPO)~\cite{Shao-2024-Deepseekmath}, known for its memory efficiency and success in training DeepSeek-R1~\cite{Guo-2025-Deepseek}. However, GRPO stalls when all sampled responses in a group are incorrect -- referred to as an \emph{all-negative-sample} group -- as it fails to update the policy, hindering learning progress. The contributions of this paper are two-fold. First, we propose a simple yet effective framework that introduces response diversity within all-negative-sample groups in GRPO using AI feedback. We also provide a theoretical analysis, via a stylized model, showing how this diversification improves learning dynamics. Second, we empirically validate our approach, showing the improved performance across various model sizes (7B, 14B, 32B) in both offline and online learning settings with 10 benchmarks, including base and distilled variants. Our findings highlight that learning from all-negative-sample groups is not only feasible but beneficial, advancing recent insights from \citet{Xiong-2025-Minimalist}. 

---
# ASR-FAIRBENCH: Measuring and Benchmarking Equity Across Speech Recognition Systems 

**Authors**: Anand Rai, Satyam Rahangdale, Utkarsh Anand, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.11572)  

**Abstract**: Automatic Speech Recognition (ASR) systems have become ubiquitous in everyday applications, yet significant disparities in performance across diverse demographic groups persist. In this work, we introduce the ASR-FAIRBENCH leaderboard which is designed to assess both the accuracy and equity of ASR models in real-time. Leveraging the Meta's Fair-Speech dataset, which captures diverse demographic characteristics, we employ a mixed-effects Poisson regression model to derive an overall fairness score. This score is integrated with traditional metrics like Word Error Rate (WER) to compute the Fairness Adjusted ASR Score (FAAS), providing a comprehensive evaluation framework. Our approach reveals significant performance disparities in SOTA ASR models across demographic groups and offers a benchmark to drive the development of more inclusive ASR technologies. 

---
# TARGET: Benchmarking Table Retrieval for Generative Tasks 

**Authors**: Xingyu Ji, Parker Glenn, Aditya G. Parameswaran, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2505.11545)  

**Abstract**: The data landscape is rich with structured data, often of high value to organizations, driving important applications in data analysis and machine learning. Recent progress in representation learning and generative models for such data has led to the development of natural language interfaces to structured data, including those leveraging text-to-SQL. Contextualizing interactions, either through conversational interfaces or agentic components, in structured data through retrieval-augmented generation can provide substantial benefits in the form of freshness, accuracy, and comprehensiveness of answers. The key question is: how do we retrieve the right table(s) for the analytical query or task at hand? To this end, we introduce TARGET: a benchmark for evaluating TAble Retrieval for GEnerative Tasks. With TARGET we analyze the retrieval performance of different retrievers in isolation, as well as their impact on downstream tasks. We find that dense embedding-based retrievers far outperform a BM25 baseline which is less effective than it is for retrieval over unstructured text. We also surface the sensitivity of retrievers across various metadata (e.g., missing table titles), and demonstrate a stark variation of retrieval performance across datasets and tasks. TARGET is available at this https URL. 

---
