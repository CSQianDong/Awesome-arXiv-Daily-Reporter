# It's All About In-Context Learning! Teaching Extremely Low-Resource Languages to LLMs 

**Authors**: Yue Li, Zhixue Zhao, Carolina Scarton  

**Link**: [PDF](https://arxiv.org/pdf/2508.19089)  

**Abstract**: Extremely low-resource languages, especially those written in rare scripts, as shown in Figure 1, remain largely unsupported by large language models (LLMs). This is due in part to compounding factors such as the lack of training data. This paper delivers the first comprehensive analysis of whether LLMs can acquire such languages purely via in-context learning (ICL), with or without auxiliary alignment signals, and how these methods compare to parameter-efficient fine-tuning (PEFT). We systematically evaluate 20 under-represented languages across three state-of-the-art multilingual LLMs. Our findings highlight the limitation of PEFT when both language and its script are extremely under-represented by the LLM. In contrast, zero-shot ICL with language alignment is impressively effective on extremely low-resource languages, while few-shot ICL or PEFT is more beneficial for languages relatively better represented by LLMs. For LLM practitioners working on extremely low-resource languages, we summarise guidelines grounded by our results on adapting LLMs to low-resource languages, e.g., avoiding fine-tuning a multilingual model on languages of unseen scripts. 

---
# Do LVLMs Know What They Know? A Systematic Study of Knowledge Boundary Perception in LVLMs 

**Authors**: Zhikai Ding, Shiyu Ni, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2508.19111)  

**Abstract**: Large vision-language models (LVLMs) demonstrate strong visual question answering (VQA) capabilities but are shown to hallucinate. A reliable model should perceive its knowledge boundaries-knowing what it knows and what it does not. This paper investigates LVLMs' perception of their knowledge boundaries by evaluating three types of confidence signals: probabilistic confidence, answer consistency-based confidence, and verbalized confidence. Experiments on three LVLMs across three VQA datasets show that, although LVLMs possess a reasonable perception level, there is substantial room for improvement. Among the three confidences, probabilistic and consistency-based signals are more reliable indicators, while verbalized confidence often leads to overconfidence. To enhance LVLMs' perception, we adapt several established confidence calibration methods from Large Language Models (LLMs) and propose three effective methods. Additionally, we compare LVLMs with their LLM counterparts, finding that jointly processing visual and textual inputs decreases question-answering performance but reduces confidence, resulting in an improved perception level compared to LLMs. 

---
# HiPlan: Hierarchical Planning for LLM-Based Agents with Adaptive Global-Local Guidance 

**Authors**: Ziyue Li, Yuan Chang, Gaihong Yu, Xiaoqiu Le  

**Link**: [PDF](https://arxiv.org/pdf/2508.19076)  

**Abstract**: Large language model (LLM)-based agents have demonstrated remarkable capabilities in decision-making tasks, but struggle significantly with complex, long-horizon planning scenarios. This arises from their lack of macroscopic guidance, causing disorientation and failures in complex tasks, as well as insufficient continuous oversight during execution, rendering them unresponsive to environmental changes and prone to deviations. To tackle these challenges, we introduce HiPlan, a hierarchical planning framework that provides adaptive global-local guidance to boost LLM-based agents'decision-making. HiPlan decomposes complex tasks into milestone action guides for general direction and step-wise hints for detailed actions. During the offline phase, we construct a milestone library from expert demonstrations, enabling structured experience reuse by retrieving semantically similar tasks and milestones. In the execution phase, trajectory segments from past milestones are dynamically adapted to generate step-wise hints that align current observations with the milestone objectives, bridging gaps and correcting deviations. Extensive experiments across two challenging benchmarks demonstrate that HiPlan substantially outperforms strong baselines, and ablation studies validate the complementary benefits of its hierarchical components. 

---
# Empowering Computing Education Researchers Through LLM-Assisted Content Analysis 

**Authors**: Laurie Gale, Sebastian Mateos Nicolajsen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18872)  

**Abstract**: Computing education research (CER) is often instigated by practitioners wanting to improve both their own and the wider discipline's teaching practice. However, the latter is often difficult as many researchers lack the colleagues, resources, or capacity to conduct research that is generalisable or rigorous enough to advance the discipline. As a result, research methods that enable sense-making with larger volumes of qualitative data, while not increasing the burden on the researcher, have significant potential within CER.
In this discussion paper, we propose such a method for conducting rigorous analysis on large volumes of textual data, namely a variation of LLM-assisted content analysis (LACA). This method combines content analysis with the use of large language models, empowering researchers to conduct larger-scale research which they would otherwise not be able to perform. Using a computing education dataset, we illustrate how LACA could be applied in a reproducible and rigorous manner. We believe this method has potential in CER, enabling more generalisable findings from a wider range of research. This, together with the development of similar methods, can help to advance both the practice and research quality of the CER discipline. 

---
# Evaluating the Evaluators: Are readability metrics good measures of readability? 

**Authors**: Isabel Cachola, Daniel Khashabi, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2508.19221)  

**Abstract**: Plain Language Summarization (PLS) aims to distill complex documents into accessible summaries for non-expert audiences. In this paper, we conduct a thorough survey of PLS literature, and identify that the current standard practice for readability evaluation is to use traditional readability metrics, such as Flesch-Kincaid Grade Level (FKGL). However, despite proven utility in other fields, these metrics have not been compared to human readability judgments in PLS. We evaluate 8 readability metrics and show that most correlate poorly with human judgments, including the most popular metric, FKGL. We then show that Language Models (LMs) are better judges of readability, with the best-performing model achieving a Pearson correlation of 0.56 with human judgments. Extending our analysis to PLS datasets, which contain summaries aimed at non-expert audiences, we find that LMs better capture deeper measures of readability, such as required background knowledge, and lead to different conclusions than the traditional metrics. Based on these findings, we offer recommendations for best practices in the evaluation of plain language summaries. We release our analysis code and survey data. 

---
# ReflectivePrompt: Reflective evolution in autoprompting algorithms 

**Authors**: Viktor N. Zhuravlev, Artur R. Khairullin, Ernest A. Dyagin, Alena N. Sitkina, Nikita I. Kulin  

**Link**: [PDF](https://arxiv.org/pdf/2508.18870)  

**Abstract**: Autoprompting is the process of automatically selecting optimized prompts for language models, which has been gaining popularity with the rapid advancement of prompt engineering, driven by extensive research in the field of large language models (LLMs). This paper presents ReflectivePrompt - a novel autoprompting method based on evolutionary algorithms that employs a reflective evolution approach for more precise and comprehensive search of optimal prompts. ReflectivePrompt utilizes short-term and long-term reflection operations before crossover and elitist mutation to enhance the quality of the modifications they introduce. This method allows for the accumulation of knowledge obtained throughout the evolution process and updates it at each epoch based on the current population. ReflectivePrompt was tested on 33 datasets for classification and text generation tasks using open-access large language models: t-lite-instruct-0.1 and gemma3-27b-it. The method demonstrates, on average, a significant improvement (e.g., 28% on BBH compared to EvoPrompt) in metrics relative to current state-of-the-art approaches, thereby establishing itself as one of the most effective solutions in evolutionary algorithm-based autoprompting. 

---
# ConfTuner: Training Large Language Models to Express Their Confidence Verbally 

**Authors**: Yibo Li, Miao Xiong, Jiaying Wu, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18847)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-stakes domains such as science, law, and healthcare, where accurate expressions of uncertainty are essential for reliability and trust. However, current LLMs are often observed to generate incorrect answers with high confidence, a phenomenon known as "overconfidence". Recent efforts have focused on calibrating LLMs' verbalized confidence: i.e., their expressions of confidence in text form, such as "I am 80% confident that...". Existing approaches either rely on prompt engineering or fine-tuning with heuristically generated uncertainty estimates, both of which have limited effectiveness and generalizability. Motivated by the notion of proper scoring rules for calibration in classical machine learning models, we introduce ConfTuner, a simple and efficient fine-tuning method that introduces minimal overhead and does not require ground-truth confidence scores or proxy confidence estimates. ConfTuner relies on a new loss function, tokenized Brier score, which we theoretically prove to be a proper scoring rule, intuitively meaning that it "correctly incentivizes the model to report its true probability of being correct". ConfTuner improves calibration across diverse reasoning tasks and generalizes to black-box models such as GPT-4o. Our results further show that better-calibrated confidence enables downstream gains in self-correction and model cascade, advancing the development of trustworthy LLM systems. The code is available at this https URL. 

---
# Arrows of Math Reasoning Data Synthesis for Large Language Models: Diversity, Complexity and Correctness 

**Authors**: Sirui Chen, Changxin Tian, Binbin Hu, Kunlong Chen, Ziqi Liu, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.18824)  

**Abstract**: Enhancing the mathematical reasoning of large language models (LLMs) demands high-quality training data, yet conventional methods face critical challenges in scalability, cost, and data reliability. To address these limitations, we propose a novel program-assisted synthesis framework that systematically generates a high-quality mathematical corpus with guaranteed diversity, complexity, and correctness. This framework integrates mathematical knowledge systems and domain-specific tools to create executable programs. These programs are then translated into natural language problem-solution pairs and vetted by a bilateral validation mechanism that verifies solution correctness against program outputs and ensures program-problem consistency. We have generated 12.3 million such problem-solving triples. Experiments demonstrate that models fine-tuned on our data significantly improve their inference capabilities, achieving state-of-the-art performance on several benchmark datasets and showcasing the effectiveness of our synthesis approach. 

---
# ThinkDial: An Open Recipe for Controlling Reasoning Effort in Large Language Models 

**Authors**: Qianyu He, Siyu Yuan, Xuefeng Li, Mingxuan Wang, Jiangjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18773)  

**Abstract**: Large language models (LLMs) with chain-of-thought reasoning have demonstrated remarkable problem-solving capabilities, but controlling their computational effort remains a significant challenge for practical deployment. Recent proprietary systems like OpenAI's gpt-oss series have introduced discrete operational modes for intuitive reasoning control, but the open-source community has largely failed to achieve such capabilities. In this paper, we introduce ThinkDial, the first open-recipe end-to-end framework that successfully implements gpt-oss-style controllable reasoning through discrete operational modes. Our system enables seamless switching between three distinct reasoning regimes: High mode (full reasoning capability), Medium mode (50 percent token reduction with <10 percent performance degradation), and Low mode (75 percent token reduction with <15 percent performance degradation). We achieve this through an end-to-end training paradigm that integrates budget-mode control throughout the entire pipeline: budget-mode supervised fine-tuning that embeds controllable reasoning capabilities directly into the learning process, and two-phase budget-aware reinforcement learning with adaptive reward shaping. Extensive experiments demonstrate that ThinkDial achieves target compression-performance trade-offs with clear response length reductions while maintaining performance thresholds. The framework also exhibits strong generalization capabilities on out-of-distribution tasks. 

---
# MovieCORE: COgnitive REasoning in Movies 

**Authors**: Gueter Josmy Faure, Min-Hung Chen, Jia-Fong Yeh, Ying Cheng, Hung-Ting Su, Yung-Hao Tang, Shang-Hong Lai, Winston H. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19026)  

**Abstract**: This paper introduces MovieCORE, a novel video question answering (VQA) dataset designed to probe deeper cognitive understanding of movie content. Unlike existing datasets that focus on surface-level comprehension, MovieCORE emphasizes questions that engage System-2 thinking while remaining specific to the video material. We present an innovative agentic brainstorming approach, utilizing multiple large language models (LLMs) as thought agents to generate and refine high-quality question-answer pairs. To evaluate dataset quality, we develop a set of cognitive tests assessing depth, thought-provocation potential, and syntactic complexity. We also propose a comprehensive evaluation scheme for assessing VQA model performance on deeper cognitive tasks. To address the limitations of existing video-language models (VLMs), we introduce an agentic enhancement module, Agentic Choice Enhancement (ACE), which improves model reasoning capabilities post-training by up to 25%. Our work contributes to advancing movie understanding in AI systems and provides valuable insights into the capabilities and limitations of current VQA models when faced with more challenging, nuanced questions about cinematic content. Our project page, dataset and code can be found at this https URL. 

---
# Beyond Quality: Unlocking Diversity in Ad Headline Generation with Large Language Models 

**Authors**: Chang Wang, Siyu Yan, Depeng Yuan, Yuqi Chen, Yanhua Huang, Yuanhang Zheng, Shuhao Li, Yinqi Zhang, Kedi Chen, Mingrui Zhu, Ruiwen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18739)  

**Abstract**: The generation of ad headlines plays a vital role in modern advertising, where both quality and diversity are essential to engage a broad range of audience segments. Current approaches primarily optimize language models for headline quality or click-through rates (CTR), often overlooking the need for diversity and resulting in homogeneous outputs. To address this limitation, we propose DIVER, a novel framework based on large language models (LLMs) that are jointly optimized for both diversity and quality. We first design a semantic- and stylistic-aware data generation pipeline that automatically produces high-quality training pairs with ad content and multiple diverse headlines. To achieve the goal of generating high-quality and diversified ad headlines within a single forward pass, we propose a multi-stage multi-objective optimization framework with supervised fine-tuning (SFT) and reinforcement learning (RL). Experiments on real-world industrial datasets demonstrate that DIVER effectively balances quality and diversity. Deployed on a large-scale content-sharing platform serving hundreds of millions of users, our framework improves advertiser value (ADVV) and CTR by 4.0% and 1.4%. 

---
# Generative Interfaces for Language Models 

**Authors**: Jiaqi Chen, Yanzhe Zhang, Yutong Zhang, Yijia Shao, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.19227)  

**Abstract**: Large language models (LLMs) are increasingly seen as assistants, copilots, and consultants, capable of supporting a wide range of tasks through natural conversation. However, most systems remain constrained by a linear request-response format that often makes interactions inefficient in multi-turn, information-dense, and exploratory tasks. To address these limitations, we propose Generative Interfaces for Language Models, a paradigm in which LLMs respond to user queries by proactively generating user interfaces (UIs) that enable more adaptive and interactive engagement. Our framework leverages structured interface-specific representations and iterative refinements to translate user queries into task-specific UIs. For systematic evaluation, we introduce a multidimensional assessment framework that compares generative interfaces with traditional chat-based ones across diverse tasks, interaction patterns, and query types, capturing functional, interactive, and emotional aspects of user experience. Results show that generative interfaces consistently outperform conversational ones, with humans preferring them in over 70% of cases. These findings clarify when and why users favor generative interfaces, paving the way for future advancements in human-AI interaction. 

---
# LLM-based Contrastive Self-Supervised AMR Learning with Masked Graph Autoencoders for Fake News Detection 

**Authors**: Shubham Gupta, Shraban Kumar Chatterjee, Suman Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18819)  

**Abstract**: The proliferation of misinformation in the digital age has led to significant societal challenges. Existing approaches often struggle with capturing long-range dependencies, complex semantic relations, and the social dynamics influencing news dissemination. Furthermore, these methods require extensive labelled datasets, making their deployment resource-intensive. In this study, we propose a novel self-supervised misinformation detection framework that integrates both complex semantic relations using Abstract Meaning Representation (AMR) and news propagation dynamics. We introduce an LLM-based graph contrastive loss (LGCL) that utilizes negative anchor points generated by a Large Language Model (LLM) to enhance feature separability in a zero-shot manner. To incorporate social context, we employ a multi view graph masked autoencoder, which learns news propagation features from social context graph. By combining these semantic and propagation-based features, our approach effectively differentiates between fake and real news in a self-supervised manner. Extensive experiments demonstrate that our self-supervised framework achieves superior performance compared to other state-of-the-art methodologies, even with limited labelled datasets while improving generalizability. 

---
# Automatic Prompt Optimization with Prompt Distillation 

**Authors**: Viktor N. Zhuravlev, Artur R. Khairullin, Ernest A. Dyagin, Alena N. Sitkina, Nikita I. Kulin  

**Link**: [PDF](https://arxiv.org/pdf/2508.18992)  

**Abstract**: Autoprompting is the process of automatically selecting optimized prompts for language models, which is gaining popularity due to the rapid development of prompt engineering driven by extensive research in the field of large language models (LLMs). This paper presents DistillPrompt -- a novel autoprompting method based on large language models that employs a multi-stage integration of task-specific information into prompts using training data. DistillPrompt utilizes distillation, compression, and aggregation operations to explore the prompt space more thoroughly. The method was tested on different datasets for text classification and generation tasks using the t-lite-instruct-0.1 language model. The results demonstrate a significant average improvement (e.g., 20.12% across the entire dataset compared to Grips) in key metrics over existing methods in the field, establishing DistillPrompt as one of the most effective non-gradient approaches in autoprompting. 

---
# Breaking the Trade-Off Between Faithfulness and Expressiveness for Large Language Models 

**Authors**: Chenxu Yang, Qingyi Si, Zheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.18651)  

**Abstract**: Grounding responses in external knowledge represents an effective strategy for mitigating hallucinations in Large Language Models (LLMs). However, current LLMs struggle to seamlessly integrate knowledge while simultaneously maintaining faithfulness (or fidelity) and expressiveness, capabilities that humans naturally possess. This limitation results in outputs that either lack support from external knowledge, thereby compromising faithfulness, or appear overly verbose and unnatural, thus sacrificing expressiveness. In this work, to break the trade-off between faithfulness and expressiveness, we propose Collaborative Decoding (CoDe), a novel approach that dynamically integrates output probabilities generated with and without external knowledge. This integration is guided by distribution divergence and model confidence, enabling the selective activation of relevant and reliable expressions from the model's internal parameters. Furthermore, we introduce a knowledge-aware reranking mechanism that prevents over-reliance on prior parametric knowledge while ensuring proper utilization of provided external information. Through comprehensive experiments, our plug-and-play CoDe framework demonstrates superior performance in enhancing faithfulness without compromising expressiveness across diverse LLMs and evaluation metrics, validating both its effectiveness and generalizability. 

---
# Demystifying Scientific Problem-Solving in LLMs by Probing Knowledge and Reasoning 

**Authors**: Alan Li, Yixin Liu, Arpan Sarkar, Doug Downey, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2508.19202)  

**Abstract**: Scientific problem solving poses unique challenges for LLMs, requiring both deep domain knowledge and the ability to apply such knowledge through complex reasoning. While automated scientific reasoners hold great promise for assisting human scientists, there is currently no widely adopted holistic benchmark for evaluating scientific reasoning, and few approaches systematically disentangle the distinct roles of knowledge and reasoning in these tasks. To address these gaps, we introduce SciReas, a diverse suite of existing benchmarks for scientific reasoning tasks, and SciReas-Pro, a selective subset that requires more complex reasoning. Our holistic evaluation surfaces insights about scientific reasoning performance that remain hidden when relying on individual benchmarks alone. We then propose KRUX, a probing framework for studying the distinct roles of reasoning and knowledge in scientific tasks. Combining the two, we conduct an in-depth analysis that yields several key findings: (1) Retrieving task-relevant knowledge from model parameters is a critical bottleneck for LLMs in scientific reasoning; (2) Reasoning models consistently benefit from external knowledge added in-context on top of the reasoning enhancement; (3) Enhancing verbalized reasoning improves LLMs' ability to surface task-relevant knowledge. Finally, we conduct a lightweight analysis, comparing our science-focused data composition with concurrent efforts on long CoT SFT, and release SciLit01, a strong 8B baseline for scientific reasoning. 

---
# Filtering for Creativity: Adaptive Prompting for Multilingual Riddle Generation in LLMs 

**Authors**: Duy Le, Kent Ziti, Evan Girard-Sun, Sean O'Brien, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18709)  

**Abstract**: Multilingual riddle generation challenges large language models (LLMs) to balance cultural fluency with creative abstraction. Standard prompting strategies -- zero-shot, few-shot, chain-of-thought -- tend to reuse memorized riddles or perform shallow paraphrasing. We introduce Adaptive Originality Filtering (AOF), a prompting framework that filters redundant generations using cosine-based similarity rejection, while enforcing lexical novelty and cross-lingual fidelity. Evaluated across three LLMs and four language pairs, AOF-enhanced GPT-4o achieves \texttt{0.177} Self-BLEU and \texttt{0.915} Distinct-2 in Japanese, signaling improved lexical diversity and reduced redundancy compared to other prompting methods and language pairs. Our findings show that semantic rejection can guide culturally grounded, creative generation without task-specific fine-tuning. 

---
# EMMM, Explain Me My Model! Explainable Machine Generated Text Detection in Dialogues 

**Authors**: Angela Yifei Yuan, Haoyi Li, Soyeon Caren Han, Christopher Leckie  

**Link**: [PDF](https://arxiv.org/pdf/2508.18715)  

**Abstract**: The rapid adoption of large language models (LLMs) in customer service introduces new risks, as malicious actors can exploit them to conduct large-scale user impersonation through machine-generated text (MGT). Current MGT detection methods often struggle in online conversational settings, reducing the reliability and interpretability essential for trustworthy AI deployment. In customer service scenarios where operators are typically non-expert users, explanation become crucial for trustworthy MGT detection. In this paper, we propose EMMM, an explanation-then-detection framework that balances latency, accuracy, and non-expert-oriented interpretability. Experimental results demonstrate that EMMM provides explanations accessible to non-expert users, with 70\% of human evaluators preferring its outputs, while achieving competitive accuracy compared to state-of-the-art models and maintaining low latency, generating outputs within 1 second. Our code and dataset are open-sourced at this https URL. 

---
# Thinking Before You Speak: A Proactive Test-time Scaling Approach 

**Authors**: Cong Li, Wenchang Chai, Hejun Wu, Yan Pan, Pengxu Wei, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.18648)  

**Abstract**: Large Language Models (LLMs) often exhibit deficiencies with complex reasoning tasks, such as maths, which we attribute to the discrepancy between human reasoning patterns and those presented in the LLMs' training data. When dealing with complex problems, humans tend to think carefully before expressing solutions. However, they often do not articulate their inner thoughts, including their intentions and chosen methodologies. Consequently, critical insights essential for bridging reasoning steps may be absent in training data collected from human sources. To bridge this gap, we proposes inserting \emph{insight}s between consecutive reasoning steps, which review the status and initiate the next reasoning steps. Unlike prior prompting strategies that rely on a single or a workflow of static prompts to facilitate reasoning, \emph{insight}s are \emph{proactively} generated to guide reasoning processes. We implement our idea as a reasoning framework, named \emph{Thinking Before You Speak} (TBYS), and design a pipeline for automatically collecting and filtering in-context examples for the generation of \emph{insight}s, which alleviates human labeling efforts and fine-tuning overheads. Experiments on challenging mathematical datasets verify the effectiveness of TBYS. Project website: this https URL 

---
# Principled Detection of Hallucinations in Large Language Models via Multiple Testing 

**Authors**: Jiawei Li, Akshayaa Magesh, Venugopal V. Veeravalli  

**Link**: [PDF](https://arxiv.org/pdf/2508.18473)  

**Abstract**: While Large Language Models (LLMs) have emerged as powerful foundational models to solve a variety of tasks, they have also been shown to be prone to hallucinations, i.e., generating responses that sound confident but are actually incorrect or even nonsensical. In this work, we formulate the problem of detecting hallucinations as a hypothesis testing problem and draw parallels to the problem of out-of-distribution detection in machine learning models. We propose a multiple-testing-inspired method to solve the hallucination detection problem, and provide extensive experimental results to validate the robustness of our approach against state-of-the-art methods. 

---
# Latent Self-Consistency for Reliable Majority-Set Selection in Short- and Long-Answer Reasoning 

**Authors**: Jeong-seok Oh, Jay-yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.18395)  

**Abstract**: Probabilistic decoding in Large Language Models (LLMs) often yields inconsistent outputs, particularly on complex or long-form questions. Self-Consistency (SC) mitigates this for short-form QA by majority voting over exact strings, whereas Universal Self-Consistency (USC) and Weighted Unigram Consistency Score (WUCS) extend to long-form responses but lose accuracy on short-form benchmarks.
We introduce Latent Self-Consistency (LSC), which selects the most semantically consistent response using learnable token embeddings. A lightweight forward generation of summary tokens increases inference time by less than 1% and requires no changes to the model architecture.
Across 6 short-form and 5 long-form reasoning benchmarks (e.g., MATH, MMLU, TruthfulQA), LSC surpasses SC, USC and WUCS on all short-form and long-form ones on average, while maintaining negligible computational overhead. These results position LSC as a practical consistency-selection method that works reliably across answer formats. Additionally, LSC provides well-calibrated confidence estimates, maintaining low Expected Calibration Error across both answer formats. 

---
# How Reliable are LLMs for Reasoning on the Re-ranking task? 

**Authors**: Nafis Tanveer Islam, Zhiming Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.18444)  

**Abstract**: With the improving semantic understanding capability of Large Language Models (LLMs), they exhibit a greater awareness and alignment with human values, but this comes at the cost of transparency. Although promising results are achieved via experimental analysis, an in-depth understanding of the LLM's internal workings is unavoidable to comprehend the reasoning behind the re-ranking, which provides end users with an explanation that enables them to make an informed decision. Moreover, in newly developed systems with limited user engagement and insufficient ranking data, accurately re-ranking content remains a significant challenge. While various training methods affect the training of LLMs and generate inference, our analysis has found that some training methods exhibit better explainability than others, implying that an accurate semantic understanding has not been learned through all training methods; instead, abstract knowledge has been gained to optimize evaluation, which raises questions about the true reliability of LLMs. Therefore, in this work, we analyze how different training methods affect the semantic understanding of the re-ranking task in LLMs and investigate whether these models can generate more informed textual reasoning to overcome the challenges of transparency or LLMs and limited training data. To analyze the LLMs for re-ranking tasks, we utilize a relatively small ranking dataset from the environment and the Earth science domain to re-rank retrieved content. Furthermore, we also analyze the explainable information to see if the re-ranking can be reasoned using explainability. 

---
# Backprompting: Leveraging Synthetic Production Data for Health Advice Guardrails 

**Authors**: Kellen Tan Cheng, Anna Lisa Gentile, Chad DeLuca, Guang-Jie Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.18384)  

**Abstract**: The pervasiveness of large language models (LLMs) in enterprise settings has also brought forth a significant amount of risks associated with their usage. Guardrails technologies aim to mitigate this risk by filtering LLMs' input/output text through various detectors. However, developing and maintaining robust detectors faces many challenges, one of which is the difficulty in acquiring production-quality labeled data on real LLM outputs prior to deployment. In this work, we propose backprompting, a simple yet intuitive solution to generate production-like labeled data for health advice guardrails development. Furthermore, we pair our backprompting method with a sparse human-in-the-loop clustering technique to label the generated data. Our aim is to construct a parallel corpus roughly representative of the original dataset yet resembling real LLM output. We then infuse existing datasets with our synthetic examples to produce robust training data for our detector. We test our technique in one of the most difficult and nuanced guardrails: the identification of health advice in LLM output, and demonstrate improvement versus other solutions. Our detector is able to outperform GPT-4o by up to 3.73%, despite having 400x less parameters. 

---
# LLMs Can't Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions 

**Authors**: Maojia Song, Tej Deep Pala, Weisheng Jin, Amir Zadeh, Chuan Li, Dorien Herremans, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2508.18321)  

**Abstract**: Large language models (LLMs) are increasingly deployed in multi-agent systems (MAS) as components of collaborative intelligence, where peer interactions dynamically shape individual decision-making. Although prior work has focused on conformity bias, we extend the analysis to examine how LLMs form trust from previous impressions, resist misinformation, and integrate peer input during interaction, key factors for achieving collective intelligence under complex social dynamics. We present KAIROS, a benchmark simulating quiz contests with peer agents of varying reliability, offering fine-grained control over conditions such as expert-novice roles, noisy crowds, and adversarial peers. LLMs receive both historical interactions and current peer responses, allowing systematic investigation into how trust, peer action, and self-confidence influence decisions. As for mitigation strategies, we evaluate prompting, supervised fine-tuning, and reinforcement learning, Group Relative Policy Optimisation (GRPO), across multiple models. Our results reveal that GRPO with multi-agent context combined with outcome-based rewards and unconstrained reasoning achieves the best overall performance, but also decreases the robustness to social influence compared to Base models. The code and datasets are available at: this https URL. 

---
# Emotion Omni: Enabling Empathetic Speech Response Generation through Large Language Models 

**Authors**: Haoyu Wang, Guangyan Zhang, Jiale Chen, Jingyu Li, Yuehai Wang, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.18655)  

**Abstract**: With the development of speech large language models (speech LLMs), users can now interact directly with assistants via speech. However, most existing models simply convert the response content into speech without fully understanding the rich emotional and paralinguistic cues embedded in the user's query. In many cases, the same sentence can have different meanings depending on the emotional expression. Furthermore, emotional understanding is essential for improving user experience in human-machine interaction. Currently, most speech LLMs with empathetic capabilities are trained on massive datasets. This approach requires vast amounts of data and significant computational resources. Therefore, a key challenge lies in how to develop a speech LLM capable of generating empathetic responses with limited data and without the need for large-scale training. To address this challenge, we propose Emotion Omni, a novel model architecture designed to understand the emotional content of user speech input and generate empathetic speech responses. Additionally, we developed a data generation pipeline based on an open-source TTS framework to construct a 200k emotional dialogue dataset, which supports the construction of an empathetic speech assistant. The demos are available at this https URL 

---
# StepWiser: Stepwise Generative Judges for Wiser Reasoning 

**Authors**: Wei Xiong, Wenting Zhao, Weizhe Yuan, Olga Golovneva, Tong Zhang, Jason Weston, Sainbayar Sukhbaatar  

**Link**: [PDF](https://arxiv.org/pdf/2508.19229)  

**Abstract**: As models increasingly leverage multi-step reasoning strategies to solve complex problems, supervising the logical validity of these intermediate steps has become a critical research challenge. Process reward models address this by providing step-by-step feedback, but current approaches have two major drawbacks: they typically function as classifiers without providing explanations, and their reliance on supervised fine-tuning with static datasets limits generalization. Inspired by recent advances, we reframe stepwise reward modeling from a classification task to a reasoning task itself. We thus propose a generative judge that reasons about the policy model's reasoning steps (i.e., meta-reasons), outputting thinking tokens before delivering a final verdict. Our model, StepWiser, is trained by reinforcement learning using relative outcomes of rollouts. We show it provides (i) better judgment accuracy on intermediate steps than existing methods; (ii) can be used to improve the policy model at training time; and (iii) improves inference-time search. 

---
# CAC-CoT: Connector-Aware Compact Chain-of-Thought for Efficient Reasoning Data Synthesis Across Dual-System Cognitive Tasks 

**Authors**: Sunguk Choi, Yonghoon Kwon, Heondeuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.18743)  

**Abstract**: Long chain-of-thought (CoT) prompting helps Large Language Models (LLMs) solve difficult problems, but very long traces often slow or even degrade performance on fast, intuitive "System-1" tasks. We introduce Connector-Aware Compact CoT (CAC-CoT) -- a method that deliberately restricts reasoning to a small, fixed set of connector phrases, steering the model toward concise and well -- structured explanations. Despite its simplicity, our synthetic method with Gemini-2.0-Flash yields a high-quality training quality. CAC-CoT achieves approximately 85% on GSM8K and approximately 40% on GPQA (System-2) while retaining approximately 90% on S1-Bench (System-1). Its reasoning traces average approximately 300 tokens(ART), about one-third the length of baseline traces, delivering higher efficiency without loss of accuracy. 

---
# The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization 

**Authors**: Stephen Meisenbacher, Alexandra Klymenko, Andreea-Elena Bodea, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2508.18976)  

**Abstract**: Differentially private text sanitization refers to the process of privatizing texts under the framework of Differential Privacy (DP), providing provable privacy guarantees while also empirically defending against adversaries seeking to harm privacy. Despite their simplicity, DP text sanitization methods operating at the word level exhibit a number of shortcomings, among them the tendency to leave contextual clues from the original texts due to randomization during sanitization $\unicode{x2013}$ this we refer to as $\textit{contextual vulnerability}$. Given the powerful contextual understanding and inference capabilities of Large Language Models (LLMs), we explore to what extent LLMs can be leveraged to exploit the contextual vulnerability of DP-sanitized texts. We expand on previous work not only in the use of advanced LLMs, but also in testing a broader range of sanitization mechanisms at various privacy levels. Our experiments uncover a double-edged sword effect of LLM-based data reconstruction attacks on privacy and utility: while LLMs can indeed infer original semantics and sometimes degrade empirical privacy protections, they can also be used for good, to improve the quality and privacy of DP-sanitized texts. Based on our findings, we propose recommendations for using LLM data reconstruction as a post-processing step, serving to increase privacy protection by thinking adversarially. 

---
# Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models 

**Authors**: Yi Liu, Xiangyu Liu, Zequn Sun, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18760)  

**Abstract**: Large reasoning models (LRMs) have shown remarkable progress on complex reasoning tasks. However, some questions posed to LRMs are inherently unanswerable, such as math problems lacking sufficient conditions. We find that LRMs continually fail to provide appropriate abstentions when confronted with these unanswerable questions. In this paper, we systematically analyze, investigate, and resolve this issue for trustworthy AI. We first conduct a detailed analysis of the distinct response behaviors of LRMs when facing unanswerable questions. Then, we show that LRMs possess sufficient cognitive capabilities to recognize the flaws in these questions. However, they fail to exhibit appropriate abstention behavior, revealing a misalignment between their internal cognition and external response. Finally, to resolve this issue, we propose a lightweight, two-stage method that combines cognitive monitoring with inference-time intervention. Experimental results demonstrate that our method significantly improves the abstention rate while maintaining the overall reasoning performance. 

---
# Integrating gender inclusivity into large language models via instruction tuning 

**Authors**: Alina Wróblewska, Bartosz Żuk  

**Link**: [PDF](https://arxiv.org/pdf/2508.18466)  

**Abstract**: Imagine a language with masculine, feminine, and neuter grammatical genders, yet, due to historical and political conventions, masculine forms are predominantly used to refer to men, women and mixed-gender groups. This is the reality of contemporary Polish. A social consequence of this unfair linguistic system is that large language models (LLMs) trained on Polish texts inherit and reinforce this masculine bias, generating gender-imbalanced outputs. This study addresses this issue by tuning LLMs using the IPIS dataset, a collection of human-crafted gender-inclusive proofreading in Polish and Polish-to-English translation instructions. Grounded in a theoretical linguistic framework, we design a system prompt with explicit gender-inclusive guidelines for Polish. In our experiments, we IPIS-tune multilingual LLMs (Llama-8B, Mistral-7B and Mistral-Nemo) and Polish-specific LLMs (Bielik and PLLuM). Our approach aims to integrate gender inclusivity as an inherent feature of these models, offering a systematic solution to mitigate gender bias in Polish language generation. 

---
# Beyond Benchmark: LLMs Evaluation with an Anthropomorphic and Value-oriented Roadmap 

**Authors**: Jun Wang, Ninglun Gu, Kailai Zhang, Zijiao Zhang, Yelun Bao, Jin Yang, Xu Yin, Liwei Liu, Yihuan Liu, Pengyong Li, Gary G. Yen, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18646)  

**Abstract**: For Large Language Models (LLMs), a disconnect persists between benchmark performance and real-world utility. Current evaluation frameworks remain fragmented, prioritizing technical metrics while neglecting holistic assessment for deployment. This survey introduces an anthropomorphic evaluation paradigm through the lens of human intelligence, proposing a novel three-dimensional taxonomy: Intelligence Quotient (IQ)-General Intelligence for foundational capacity, Emotional Quotient (EQ)-Alignment Ability for value-based interactions, and Professional Quotient (PQ)-Professional Expertise for specialized proficiency. For practical value, we pioneer a Value-oriented Evaluation (VQ) framework assessing economic viability, social impact, ethical alignment, and environmental sustainability. Our modular architecture integrates six components with an implementation roadmap. Through analysis of 200+ benchmarks, we identify key challenges including dynamic assessment needs and interpretability gaps. It provides actionable guidance for developing LLMs that are technically proficient, contextually relevant, and ethically sound. We maintain a curated repository of open-source evaluation resources at: this https URL. 

---
# A Systematic Approach to Predict the Impact of Cybersecurity Vulnerabilities Using LLMs 

**Authors**: Anders Mølmen Høst, Pierre Lison, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18439)  

**Abstract**: Vulnerability databases, such as the National Vulnerability Database (NVD), offer detailed descriptions of Common Vulnerabilities and Exposures (CVEs), but often lack information on their real-world impact, such as the tactics, techniques, and procedures (TTPs) that adversaries may use to exploit the vulnerability. However, manually linking CVEs to their corresponding TTPs is a challenging and time-consuming task, and the high volume of new vulnerabilities published annually makes automated support desirable.
This paper introduces TRIAGE, a two-pronged automated approach that uses Large Language Models (LLMs) to map CVEs to relevant techniques from the ATT&CK knowledge base. We first prompt an LLM with instructions based on MITRE's CVE Mapping Methodology to predict an initial list of techniques. This list is then combined with the results from a second LLM-based module that uses in-context learning to map a CVE to relevant techniques. This hybrid approach strategically combines rule-based reasoning with data-driven inference. Our evaluation reveals that in-context learning outperforms the individual mapping methods, and the hybrid approach improves recall of exploitation techniques. We also find that GPT-4o-mini performs better than Llama3.3-70B on this task. Overall, our results show that LLMs can be used to automatically predict the impact of cybersecurity vulnerabilities and TRIAGE makes the process of mapping CVEs to ATT&CK more efficient.
Keywords: vulnerability impact, CVE, ATT&CK techniques, large language models, automated mapping. 

---
# Training Language Model Agents to Find Vulnerabilities with CTF-Dojo 

**Authors**: Terry Yue Zhuo, Dingmin Wang, Hantian Ding, Varun Kumar, Zijian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18370)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional capabilities when trained within executable runtime environments, notably excelling at software engineering tasks through verified feedback loops. Yet, scalable and generalizable execution-grounded environments remain scarce, limiting progress in training more capable ML agents. We introduce CTF-Dojo, the first large-scale executable runtime tailored for training LLMs with verifiable feedback, featuring 658 fully functional Capture-The-Flag (CTF)-style challenges containerized in Docker with guaranteed reproducibility. To enable rapid scaling without manual intervention, we develop CTF-Forge, an automated pipeline that transforms publicly available artifacts into ready-to-use execution environments in minutes, eliminating weeks of expert configuration traditionally required. We trained LLM-based agents on just 486 high-quality, execution-verified trajectories from CTF-Dojo, achieving up to 11.6% absolute gains over strong baselines across three competitive benchmarks: InterCode-CTF, NYU CTF Bench, and Cybench. Our best-performing 32B model reaches 31.9% Pass@1, establishing a new open-weight state-of-the-art that rivals frontier models like DeepSeek-V3-0324 and Gemini-2.5-Flash. By framing CTF-style tasks as a benchmark for executable-agent learning, CTF-Dojo demonstrates that execution-grounded training signals are not only effective but pivotal in advancing high-performance ML agents without dependence on costly proprietary systems. 

---
# Text to Query Plans for Question Answering on Large Tables 

**Authors**: Yipeng Zhang, Chen Wang, Yuzhe Zhang, Jacky Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18758)  

**Abstract**: Efficient querying and analysis of large tabular datasets remain significant challenges, especially for users without expertise in programming languages like SQL. Text-to-SQL approaches have shown promising performance on benchmark data; however, they inherit SQL's drawbacks, including inefficiency with large datasets and limited support for complex data analyses beyond basic querying. We propose a novel framework that transforms natural language queries into query plans. Our solution is implemented outside traditional databases, allowing us to support classical SQL commands while avoiding SQL's inherent limitations. Additionally, we enable complex analytical functions, such as principal component analysis and anomaly detection, providing greater flexibility and extensibility than traditional SQL capabilities. We leverage LLMs to iteratively interpret queries and construct operation sequences, addressing computational complexity by incrementally building solutions. By executing operations directly on the data, we overcome context length limitations without requiring the entire dataset to be processed by the model. We validate our framework through experiments on both standard databases and large scientific tables, demonstrating its effectiveness in handling extensive datasets and performing sophisticated data analyses. 

---
# H-PRM: A Pluggable Hotword Pre-Retrieval Module for Various Speech Recognition Systems 

**Authors**: Huangyu Dai, Lingtao Mao, Ben Chen, Zihan Wang, Zihan Liang, Ying Han, Chenyi Lei, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.18295)  

**Abstract**: Hotword customization is crucial in ASR to enhance the accuracy of domain-specific terms. It has been primarily driven by the advancements in traditional models and Audio large language models (LLMs). However, existing models often struggle with large-scale hotwords, as the recognition rate drops dramatically with the number of hotwords increasing. In this paper, we introduce a novel hotword customization system that utilizes a hotword pre-retrieval module (H-PRM) to identify the most relevant hotword candidate by measuring the acoustic similarity between the hotwords and the speech segment. This plug-and-play solution can be easily integrated into traditional models such as SeACo-Paraformer, significantly enhancing hotwords post-recall rate (PRR). Additionally, we incorporate H-PRM into Audio LLMs through a prompt-based approach, enabling seamless customization of hotwords. Extensive testing validates that H-PRM can outperform existing methods, showing a new direction for hotword customization in ASR. 

---
# Membership Inference Attacks on LLM-based Recommender Systems 

**Authors**: Jiajie He, Yuechun Gu, Min-Chun Chen, Keke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18665)  

**Abstract**: Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots. 

---
# FALCON: Autonomous Cyber Threat Intelligence Mining with LLMs for IDS Rule Generation 

**Authors**: Shaswata Mitra, Azim Bazarov, Martin Duclos, Sudip Mittal, Aritran Piplai, Md Rayhanur Rahman, Edward Zieglar, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18684)  

**Abstract**: Signature-based Intrusion Detection Systems (IDS) detect malicious activities by matching network or host activity against predefined rules. These rules are derived from extensive Cyber Threat Intelligence (CTI), which includes attack signatures and behavioral patterns obtained through automated tools and manual threat analysis, such as sandboxing. The CTI is then transformed into actionable rules for the IDS engine, enabling real-time detection and prevention. However, the constant evolution of cyber threats necessitates frequent rule updates, which delay deployment time and weaken overall security readiness. Recent advancements in agentic systems powered by Large Language Models (LLMs) offer the potential for autonomous IDS rule generation with internal evaluation. We introduce FALCON, an autonomous agentic framework that generates deployable IDS rules from CTI data in real-time and evaluates them using built-in multi-phased validators. To demonstrate versatility, we target both network (Snort) and host-based (YARA) mediums and construct a comprehensive dataset of IDS rules with their corresponding CTIs. Our evaluations indicate FALCON excels in automatic rule generation, with an average of 95% accuracy validated by qualitative evaluation with 84% inter-rater agreement among multiple cybersecurity analysts across all metrics. These results underscore the feasibility and effectiveness of LLM-driven data mining for real-time cyber threat mitigation. 

---
# REALM: Recursive Relevance Modeling for LLM-based Document Re-Ranking 

**Authors**: Pinhuan Wang, Zhiqiu Xia, Chunhua Liao, Feiyi Wang, Hang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18379)  

**Abstract**: Large Language Models (LLMs) have shown strong capabilities in document re-ranking, a key component in modern Information Retrieval (IR) systems. However, existing LLM-based approaches face notable limitations, including ranking uncertainty, unstable top-k recovery, and high token cost due to token-intensive prompting. To effectively address these limitations, we propose REALM, an uncertainty-aware re-ranking framework that models LLM-derived relevance as Gaussian distributions and refines them through recursive Bayesian updates. By explicitly capturing uncertainty and minimizing redundant queries, REALM achieves better rankings more efficiently. Experimental results demonstrate that our REALM surpasses state-of-the-art re-rankers while significantly reducing token usage and latency, promoting it as the next-generation re-ranker for modern IR systems. 

---
# MATRIX: Multi-Agent simulaTion fRamework for safe Interactions and conteXtual clinical conversational evaluation 

**Authors**: Ernest Lim, Yajie Vera He, Jared Joselowitz, Kate Preston, Mohita Chowdhury, Louis Williams, Aisling Higham, Katrina Mason, Mariane Melo, Tom Lawton, Yan Jia, Ibrahim Habli  

**Link**: [PDF](https://arxiv.org/pdf/2508.19163)  

**Abstract**: Despite the growing use of large language models (LLMs) in clinical dialogue systems, existing evaluations focus on task completion or fluency, offering little insight into the behavioral and risk management requirements essential for safety-critical systems. This paper presents MATRIX (Multi-Agent simulaTion fRamework for safe Interactions and conteXtual clinical conversational evaluation), a structured, extensible framework for safety-oriented evaluation of clinical dialogue agents.
MATRIX integrates three components: (1) a safety-aligned taxonomy of clinical scenarios, expected system behaviors and failure modes derived through structured safety engineering methods; (2) BehvJudge, an LLM-based evaluator for detecting safety-relevant dialogue failures, validated against expert clinician annotations; and (3) PatBot, a simulated patient agent capable of producing diverse, scenario-conditioned responses, evaluated for realism and behavioral fidelity with human factors expertise, and a patient-preference study.
Across three experiments, we show that MATRIX enables systematic, scalable safety evaluation. BehvJudge with Gemini 2.5-Pro achieves expert-level hazard detection (F1 0.96, sensitivity 0.999), outperforming clinicians in a blinded assessment of 240 dialogues. We also conducted one of the first realism analyses of LLM-based patient simulation, showing that PatBot reliably simulates realistic patient behavior in quantitative and qualitative evaluations. Using MATRIX, we demonstrate its effectiveness in benchmarking five LLM agents across 2,100 simulated dialogues spanning 14 hazard scenarios and 10 clinical domains.
MATRIX is the first framework to unify structured safety engineering with scalable, validated conversational AI evaluation, enabling regulator-aligned safety auditing. We release all evaluation tools, prompts, structured scenarios, and datasets. 

---
# Reasoning LLMs in the Medical Domain: A Literature Survey 

**Authors**: Armin Berger, Sarthak Khanna, David Berghaus, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2508.19097)  

**Abstract**: The emergence of advanced reasoning capabilities in Large Language Models (LLMs) marks a transformative development in healthcare applications. Beyond merely expanding functional capabilities, these reasoning mechanisms enhance decision transparency and explainability-critical requirements in medical contexts. This survey examines the transformation of medical LLMs from basic information retrieval tools to sophisticated clinical reasoning systems capable of supporting complex healthcare decisions. We provide a thorough analysis of the enabling technological foundations, with a particular focus on specialized prompting techniques like Chain-of-Thought and recent breakthroughs in Reinforcement Learning exemplified by DeepSeek-R1. Our investigation evaluates purpose-built medical frameworks while also examining emerging paradigms such as multi-agent collaborative systems and innovative prompting architectures. The survey critically assesses current evaluation methodologies for medical validation and addresses persistent challenges in field interpretation limitations, bias mitigation strategies, patient safety frameworks, and integration of multimodal clinical data. Through this survey, we seek to establish a roadmap for developing reliable LLMs that can serve as effective partners in clinical practice and medical research. 

---
# A Concurrent Modular Agent: Framework for Autonomous LLM Agents 

**Authors**: Norihiro Maruyama, Takahide Yoshida, Hiroki Sato, Atsushi Masumori, Johnsmith, Takashi Ikegami  

**Link**: [PDF](https://arxiv.org/pdf/2508.19042)  

**Abstract**: We introduce the Concurrent Modular Agent (CMA), a framework that orchestrates multiple Large-Language-Model (LLM)-based modules that operate fully asynchronously yet maintain a coherent and fault-tolerant behavioral loop. This framework addresses long-standing difficulties in agent architectures by letting intention emerge from language-mediated interactions among autonomous processes. This approach enables flexible, adaptive, and context-dependent behavior through the combination of concurrently executed modules that offload reasoning to an LLM, inter-module communication, and a single shared global this http URL consider this approach to be a practical realization of Minsky's Society of Mind theory. We demonstrate the viability of our system through two practical use-case studies. The emergent properties observed in our system suggest that complex cognitive phenomena like self-awareness may indeed arise from the organized interaction of simpler processes, supporting Minsky-Society of Mind concept and opening new avenues for artificial intelligence research. The source code for our work is available at: this https URL. 

---
# Sense of Self and Time in Borderline Personality. A Comparative Robustness Study with Generative AI 

**Authors**: Marcin Moskalewicz, Anna Sterna, Marek Pokropski, Paula Flores  

**Link**: [PDF](https://arxiv.org/pdf/2508.19008)  

**Abstract**: This study examines the capacity of large language models (LLMs) to support phenomenological qualitative analysis of first-person experience in Borderline Personality Disorder (BPD), understood as a disorder of temporality and selfhood. Building on a prior human-led thematic analysis of 24 inpatients' life-story interviews, we compared three LLMs (OpenAI GPT-4o, Google Gemini 2.5 Pro, Anthropic Claude Opus 4) prompted to mimic the interpretative style of the original investigators. The models were evaluated with blinded and non-blinded expert judges in phenomenology and clinical psychology. Assessments included semantic congruence, Jaccard coefficients, and multidimensional validity ratings (credibility, coherence, substantiveness, and groundness in data). Results showed variable overlap with the human analysis, from 0 percent in GPT to 42 percent in Claude and 58 percent in Gemini, and a low Jaccard coefficient (0.21-0.28). However, the models recovered themes omitted by humans. Gemini's output most closely resembled the human analysis, with validity scores significantly higher than GPT and Claude (p < 0.0001), and was judged as human by blinded experts. All scores strongly correlated (R > 0.78) with the quantity of text and words per theme, highlighting both the variability and potential of AI-augmented thematic analysis to mitigate human interpretative bias. 

---
# AI Models Exceed Individual Human Accuracy in Predicting Everyday Social Norms 

**Authors**: Pontus Strimling, Simon Karlsson, Irina Vartanova, Kimmo Eriksson  

**Link**: [PDF](https://arxiv.org/pdf/2508.19004)  

**Abstract**: A fundamental question in cognitive science concerns how social norms are acquired and represented. While humans typically learn norms through embodied social experience, we investigated whether large language models can achieve sophisticated norm understanding through statistical learning alone. Across two studies, we systematically evaluated multiple AI systems' ability to predict human social appropriateness judgments for 555 everyday scenarios by examining how closely they predicted the average judgment compared to each human participant. In Study 1, GPT-4.5's accuracy in predicting the collective judgment on a continuous scale exceeded that of every human participant (100th percentile). Study 2 replicated this, with Gemini 2.5 Pro outperforming 98.7% of humans, GPT-5 97.8%, and Claude Sonnet 4 96.0%. Despite this predictive power, all models showed systematic, correlated errors. These findings demonstrate that sophisticated models of social cognition can emerge from statistical learning over linguistic data alone, challenging strong versions of theories emphasizing the exclusive necessity of embodied experience for cultural competence. The systematic nature of AI limitations across different architectures indicates potential boundaries of pattern-based social understanding, while the models' ability to outperform nearly all individual humans in this predictive task suggests that language serves as a remarkably rich repository for cultural knowledge transmission. 

---
# Hybrid Deep Searcher: Integrating Parallel and Sequential Search Reasoning 

**Authors**: Dayoon Ko, Jihyuk Kim, Haeju Park, Sohyeon Kim, Dahyun Lee, Yongrae Jo, Gunhee Kim, Moontae Lee, Kyungjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.19113)  

**Abstract**: Large reasoning models (LRMs) have demonstrated strong performance in complex, multi-step reasoning tasks. Existing methods enhance LRMs by sequentially integrating external knowledge retrieval; models iteratively generate queries, retrieve external information, and progressively reason over this information. However, purely sequential querying increases inference latency and context length, diminishing coherence and potentially reducing accuracy. To address these limitations, we introduce HDS-QA (Hybrid Deep Search QA), a synthetic dataset automatically generated from Natural Questions, explicitly designed to train LRMs to distinguish parallelizable from sequential queries. HDS-QA comprises hybrid-hop questions that combine parallelizable independent subqueries (executable simultaneously) and sequentially dependent subqueries (requiring step-by-step resolution), along with synthetic reasoning-querying-retrieval paths involving parallel queries. We fine-tune an LRM using HDS-QA, naming the model HybridDeepSearcher, which outperforms state-of-the-art baselines across multiple benchmarks, notably achieving +15.9 and +11.5 F1 on FanOutQA and a subset of BrowseComp, respectively, both requiring comprehensive and exhaustive search. Experimental results highlight two key advantages: HybridDeepSearcher reaches comparable accuracy with fewer search turns, significantly reducing inference latency, and it effectively scales as more turns are permitted. These results demonstrate the efficiency, scalability, and effectiveness of explicitly training LRMs to leverage hybrid parallel and sequential querying. 

---
# Interactive Evaluation of Large Language Models for Multi-Requirement Software Engineering Tasks 

**Authors**: Dimitrios Rontogiannis, Maxime Peyrard, Nicolas Baldwin, Martin Josifoski, Robert West, Dimitrios Gunopulos  

**Link**: [PDF](https://arxiv.org/pdf/2508.18905)  

**Abstract**: Standard single-turn, static benchmarks fall short in evaluating the nuanced capabilities of Large Language Models (LLMs) on complex tasks such as software engineering. In this work, we propose a novel interactive evaluation framework that assesses LLMs on multi-requirement programming tasks through structured, feedback-driven dialogue. Each task is modeled as a requirement dependency graph, and an ``interviewer'' LLM, aware of the ground-truth solution, provides minimal, targeted hints to an ``interviewee'' model to help correct errors and fulfill target constraints. This dynamic protocol enables fine-grained diagnostic insights into model behavior, uncovering strengths and systematic weaknesses that static benchmarks fail to measure. We build on DevAI, a benchmark of 55 curated programming tasks, by adding ground-truth solutions and evaluating the relevance and utility of interviewer hints through expert annotation. Our results highlight the importance of dynamic evaluation in advancing the development of collaborative code-generating agents. 

---
# Can Structured Templates Facilitate LLMs in Tackling Harder Tasks? : An Exploration of Scaling Laws by Difficulty 

**Authors**: Zhichao Yang, Zhaoxin Fan, Gen Li, Yuanze Hu, Xinyu Wang, Ye Qiu, Xin Wang, Yifan Sun, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19069)  

**Abstract**: Structured, procedural reasoning is essential for Large Language Models (LLMs), especially in mathematics. While post-training methods have improved LLM performance, they still fall short in capturing deep procedural logic on complex tasks. To tackle the issue, in this paper, we first investigate this limitation and uncover a novel finding: a Scaling Law by Difficulty, which reveals that model performance follows a U-shaped curve with respect to training data complexity -- excessive low-difficulty data impedes abstraction, while high-difficulty data significantly enhances reasoning ability. Motivated by this, we propose the Structured Solution Template (SST) framework, which uses solution templates and a curriculum of varied difficulty to explicitly teach procedural reasoning. Specifically, SST comprises (1) fine-tuning with structured solution-template chains and dynamically weighted loss to prioritize procedural logic, (2) prompt-time injection of solution templates as cognitive scaffolds to guide inference, and (3) integrated curriculum fine-tuning that explicitly teaches the model to self-plan - execute - self-correct. Experiments on GSM8K, AIME24, and new Dynamic En benchmark show that SST significantly improves both accuracy and efficiency, especially on harder problems. 

---
# The Ramon Llull's Thinking Machine for Automated Ideation 

**Authors**: Xinran Zhao, Boyuan Zheng, Chenglei Si, Haofei Yu, Ken Liu, Runlong Zhou, Ruochen Li, Tong Chen, Xiang Li, Yiming Zhang, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19200)  

**Abstract**: This paper revisits Ramon Llull's Ars combinatoria - a medieval framework for generating knowledge through symbolic recombination - as a conceptual foundation for building a modern Llull's thinking machine for research ideation. Our approach defines three compositional axes: Theme (e.g., efficiency, adaptivity), Domain (e.g., question answering, machine translation), and Method (e.g., adversarial training, linear attention). These elements represent high-level abstractions common in scientific work - motivations, problem settings, and technical approaches - and serve as building blocks for LLM-driven exploration. We mine elements from human experts or conference papers and show that prompting LLMs with curated combinations produces research ideas that are diverse, relevant, and grounded in current literature. This modern thinking machine offers a lightweight, interpretable tool for augmenting scientific creativity and suggests a path toward collaborative ideation between humans and AI. 

---
# VISION: Robust and Interpretable Code Vulnerability Detection Leveraging Counterfactual Augmentation 

**Authors**: David Egea, Barproda Halder, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2508.18933)  

**Abstract**: Automated detection of vulnerabilities in source code is an essential cybersecurity challenge, underpinning trust in digital systems and services. Graph Neural Networks (GNNs) have emerged as a promising approach as they can learn structural and logical code relationships in a data-driven manner. However, their performance is severely constrained by training data imbalances and label noise. GNNs often learn 'spurious' correlations from superficial code similarities, producing detectors that fail to generalize well to unseen real-world data. In this work, we propose a unified framework for robust and interpretable vulnerability detection, called VISION, to mitigate spurious correlations by systematically augmenting a counterfactual training dataset. Counterfactuals are samples with minimal semantic modifications but opposite labels. Our framework includes: (i) generating counterfactuals by prompting a Large Language Model (LLM); (ii) targeted GNN training on paired code examples with opposite labels; and (iii) graph-based interpretability to identify the crucial code statements relevant for vulnerability predictions while ignoring spurious ones. We find that VISION reduces spurious learning and enables more robust, generalizable detection, improving overall accuracy (from 51.8% to 97.8%), pairwise contrast accuracy (from 4.5% to 95.8%), and worst-group accuracy (from 0.7% to 85.5%) on the Common Weakness Enumeration (CWE)-20 vulnerability. We further demonstrate gains using proposed metrics: intra-class attribution variance, inter-class attribution distance, and node score dependency. We also release CWE-20-CFA, a benchmark of 27,556 functions (real and counterfactual) from the high-impact CWE-20 category. Finally, VISION advances transparent and trustworthy AI-based cybersecurity systems through interactive visualization for human-in-the-loop analysis. 

---
# SchemaCoder: Automatic Log Schema Extraction Coder with Residual Q-Tree Boosting 

**Authors**: Lily Jiaxin Wan, Chia-Tung Ho, Rongjian Liang, Cunxi Yu, Deming Chen, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.18554)  

**Abstract**: Log schema extraction is the process of deriving human-readable templates from massive volumes of log data, which is essential yet notoriously labor-intensive. Recent studies have attempted to streamline this task by leveraging Large Language Models (LLMs) for automated schema extraction. However, existing methods invariably rely on predefined regular expressions, necessitating human domain expertise and severely limiting productivity gains. To fundamentally address this limitation, we introduce SchemaCoder, the first fully automated schema extraction framework applicable to a wide range of log file formats without requiring human customization within the flow. At its core, SchemaCoder features a novel Residual Question-Tree (Q-Tree) Boosting mechanism that iteratively refines schema extraction through targeted, adaptive queries driven by LLMs. Particularly, our method partitions logs into semantic chunks via context-bounded segmentation, selects representative patterns using embedding-based sampling, and generates schema code through hierarchical Q-Tree-driven LLM queries, iteratively refined by our textual-residual evolutionary optimizer and residual boosting. Experimental validation demonstrates SchemaCoder's superiority on the widely-used LogHub-2.0 benchmark, achieving an average improvement of 21.3% over state-of-the-arts. 

---
# Bias Mitigation Agent: Optimizing Source Selection for Fair and Balanced Knowledge Retrieval 

**Authors**: Karanbir Singh, Deepak Muppiri, William Ngu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18724)  

**Abstract**: Large Language Models (LLMs) have transformed the field of artificial intelligence by unlocking the era of generative applications. Built on top of generative AI capabilities, Agentic AI represents a major shift toward autonomous, goal-driven systems that can reason, retrieve, and act. However, they also inherit the bias present in both internal and external information sources. This significantly affects the fairness and balance of retrieved information, and hence reduces user trust. To address this critical challenge, we introduce a novel Bias Mitigation Agent, a multi-agent system designed to orchestrate the workflow of bias mitigation through specialized agents that optimize the selection of sources to ensure that the retrieved content is both highly relevant and minimally biased to promote fair and balanced knowledge dissemination. The experimental results demonstrate an 81.82\% reduction in bias compared to a baseline naive retrieval strategy. 

---
# Investigating Advanced Reasoning of Large Language Models via Black-Box Interaction 

**Authors**: Congchi Yin, Tianyi Wu, Yankai Shu, Alex Gu, Yunhan Wang, Jun Shao, Xun Jiang, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.19035)  

**Abstract**: Existing tasks fall short in evaluating reasoning ability of Large Language Models (LLMs) in an interactive, unknown environment. This deficiency leads to the isolated assessment of deductive, inductive, and abductive reasoning, neglecting the integrated reasoning process that is indispensable for humans discovery of real world. We introduce a novel evaluation paradigm, \textit{black-box interaction}, to tackle this challenge. A black-box is defined by a hidden function that maps a specific set of inputs to outputs. LLMs are required to unravel the hidden function behind the black-box by interacting with it in given exploration turns, and reasoning over observed input-output pairs. Leveraging this idea, we build the \textsc{Oracle} benchmark which comprises 6 types of black-box task and 96 black-boxes. 19 modern LLMs are benchmarked. o3 ranks first in 5 of the 6 tasks, achieving over 70\% accuracy on most easy black-boxes. But it still struggles with some hard black-box tasks, where its average performance drops below 40\%. Further analysis indicates a universal difficulty among LLMs: They lack the high-level planning capability to develop efficient and adaptive exploration strategies for hypothesis refinement. 

---
# Language Models For Generalised PDDL Planning: Synthesising Sound and Programmatic Policies 

**Authors**: Dillon Z. Chen, Johannes Zenn, Tristan Cinquin, Sheila A. McIlraith  

**Link**: [PDF](https://arxiv.org/pdf/2508.18507)  

**Abstract**: We study the usage of language models (LMs) for planning over world models specified in the Planning Domain Definition Language (PDDL). We prompt LMs to generate Python programs that serve as generalised policies for solving PDDL problems from a given domain. Notably, our approach synthesises policies that are provably sound relative to the PDDL domain without reliance on external verifiers. We conduct experiments on competition benchmarks which show that our policies can solve more PDDL problems than PDDL planners and recent LM approaches within a fixed time and memory constraint. Our approach manifests in the LMPlan planner which can solve planning problems with several hundreds of relevant objects. Surprisingly, we observe that LMs used in our framework sometimes plan more effectively over PDDL problems written in meaningless symbols in place of natural language; e.g. rewriting (at dog kitchen) as (p2 o1 o3). This finding challenges hypotheses that LMs reason over word semantics and memorise solutions from its training corpus, and is worth further exploration. 

---
# Reflection-Enhanced Meta-Optimization Integrating TextGrad-style Prompt Optimization with Memory-Driven Self-Evolution 

**Authors**: Chunlong Wu, Zhibo Qu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18749)  

**Abstract**: Recent advances in prompt optimization, exemplified by methods such as TextGrad, enable automatic, gradient-like refinement of textual prompts to enhance the performance of large language models (LLMs) on specific downstream tasks. However, current approaches are typically stateless and operate independently across optimization runs, lacking mechanisms to preserve and leverage historical optimization experience. Furthermore, they are susceptible to overfitting, often yielding prompt updates that generalize poorly beyond the immediate task context.
To address these limitations, we propose Reflection-Enhanced Meta-Optimization (REMO), a novel framework that integrates (1) a memory-augmented Reflection Retrieval-Augmented Generation (RAG) module - structured as a "mistake notebook" and (2) a Self-Adaptive Optimizer, implemented via an LLM-driven meta-controller that synthesizes epoch-level reflective insights to iteratively improve system-level prompting strategies. This architecture enables not only local, fine-grained prompt tuning akin to TextGrad, but also the systematic accumulation and reuse of cross-run optimization knowledge, thereby supporting continual improvement over time.
We instantiate the REMO framework using Qwen3-32B in standard inference mode - without explicit chain-of-thought prompting - and evaluate its efficacy on the GSM8K benchmark for mathematical reasoning. Experimental results demonstrate that, compared to a TextGrad baseline, REMO achieves more stable and robust generalization, albeit at the cost of increased computational overhead. We provide a detailed exposition of the algorithmic design, conduct a qualitative and quantitative analysis of optimization dynamics, and present a comprehensive ablation study to elucidate the contributions of each component. 

---
# Understanding Tool-Integrated Reasoning 

**Authors**: Heng Lin, Zhongwen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19201)  

**Abstract**: We study why Tool-Integrated Reasoning (TIR) makes Large Language Models (LLMs) more capable. While LLMs integrated with tools like Python code interpreters show great promise, a principled theory explaining why this paradigm is effective has been missing. This work provides the first formal proof that TIR fundamentally expands an LLM's capabilities. We demonstrate that tools enable a strict expansion of the model's empirical and feasible support, breaking the capability ceiling of pure-text models by unlocking problem-solving strategies that are otherwise impossible or intractably verbose. To guide model behavior without compromising training stability and performance, we also introduce Advantage Shaping Policy Optimization (ASPO), a novel algorithm that directly modifies the advantage function to guide the policy behavior. We conduct comprehensive experiments on challenging mathematical benchmarks, leveraging a Python interpreter as the external tool. Our results show that the TIR model decisively outperforms its pure-text counterpart on the pass@k metric. Crucially, this advantage is not confined to computationally-intensive problems but extends to those requiring significant abstract insight. We further identify the emergent cognitive patterns that illustrate how models learn to think with tools. Finally, we report improved tool usage behavior with early code invocation and much more interactive turns with ASPO. Overall, our work provides the first principled explanation for TIR's success, shifting the focus from the mere fact that tools work to why and how they enable more powerful reasoning. 

---
# ZeST: an LLM-based Zero-Shot Traversability Navigation for Unknown Environments 

**Authors**: Shreya Gummadi, Mateus V. Gasparino, Gianluca Capezzuto, Marcelo Becker, Girish Chowdhary  

**Link**: [PDF](https://arxiv.org/pdf/2508.19131)  

**Abstract**: The advancement of robotics and autonomous navigation systems hinges on the ability to accurately predict terrain traversability. Traditional methods for generating datasets to train these prediction models often involve putting robots into potentially hazardous environments, posing risks to equipment and safety. To solve this problem, we present ZeST, a novel approach leveraging visual reasoning capabilities of Large Language Models (LLMs) to create a traversability map in real-time without exposing robots to danger. Our approach not only performs zero-shot traversability and mitigates the risks associated with real-world data collection but also accelerates the development of advanced navigation systems, offering a cost-effective and scalable solution. To support our findings, we present navigation results, in both controlled indoor and unstructured outdoor environments. As shown in the experiments, our method provides safer navigation when compared to other state-of-the-art methods, constantly reaching the final goal. 

---
# An LLM-powered Natural-to-Robotic Language Translation Framework with Correctness Guarantees 

**Authors**: ZhenDong Chen, ZhanShang Nie, ShiXing Wan, JunYi Li, YongTian Cheng, Shuai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.19074)  

**Abstract**: The Large Language Models (LLM) are increasingly being deployed in robotics to generate robot control programs for specific user tasks, enabling embodied intelligence. Existing methods primarily focus on LLM training and prompt design that utilize LLMs to generate executable programs directly from user tasks in natural language. However, due to the inconsistency of the LLMs and the high complexity of the tasks, such best-effort approaches often lead to tremendous programming errors in the generated code, which significantly undermines the effectiveness especially when the light-weight LLMs are applied. This paper introduces a natural-robotic language translation framework that (i) provides correctness verification for generated control programs and (ii) enhances the performance of LLMs in program generation via feedback-based fine-tuning for the programs. To achieve this, a Robot Skill Language (RSL) is proposed to abstract away from the intricate details of the control programs, bridging the natural language tasks with the underlying robot skills. Then, the RSL compiler and debugger are constructed to verify RSL programs generated by the LLM and provide error feedback to the LLM for refining the outputs until being verified by the compiler. This provides correctness guarantees for the LLM-generated programs before being offloaded to the robots for execution, significantly enhancing the effectiveness of LLM-powered robotic applications. Experiments demonstrate NRTrans outperforms the existing method under a range of LLMs and tasks, and achieves a high success rate for light-weight LLMs. 

---
# ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive 

**Authors**: Xinhao Luo, Zihan Liu, Yangjie Zhou, Shihan Fang, Ziyu Huang, Yu Feng, Chen Zhang, Shixuan Sun, Zhenzhe Zheng, Jingwen Leng, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.18850)  

**Abstract**: Large language model (LLM) decoding suffers from high latency due to fragmented execution across operators and heavy reliance on off-chip memory for data exchange and reduction. This execution model limits opportunities for fusion and incurs significant memory traffic and kernel launch overhead. While modern architectures such as NVIDIA Hopper provide distributed shared memory and low-latency intra-cluster interconnects, they expose only low-level data movement instructions, lacking structured abstractions for collective on-chip communication. To bridge this software-hardware gap, we introduce two cluster-level communication primitives, ClusterReduce and ClusterGather, which abstract common communication patterns and enable structured, high-speed data exchange and reduction between thread blocks within a cluster, allowing intermediate results to be on-chip without involving off-chip memory. Building on these abstractions, we design ClusterFusion, an execution framework that schedules communication and computation jointly to expand operator fusion scope by composing decoding stages such as QKV Projection, Attention, and Output Projection into a single fused kernels. Evaluations on H100 GPUs show that ClusterFusion outperforms state-of-the-art inference frameworks by 1.61x on average in end-to-end latency across different models and configurations. The source code is available at this https URL. 

---
# The AI in the Mirror: LLM Self-Recognition in an Iterated Public Goods Game 

**Authors**: Olivia Long, Carter Teplica  

**Link**: [PDF](https://arxiv.org/pdf/2508.18467)  

**Abstract**: As AI agents become increasingly capable of tool use and long-horizon tasks, they have begun to be deployed in settings where multiple agents can interact. However, whereas prior work has mostly focused on human-AI interactions, there is an increasing need to understand AI-AI interactions. In this paper, we adapt the iterated public goods game, a classic behavioral economics game, to analyze the behavior of four reasoning and non-reasoning models across two conditions: models are either told they are playing against "another AI agent" or told their opponents are themselves. We find that, across different settings, telling LLMs that they are playing against themselves significantly changes their tendency to cooperate. While our study is conducted in a toy environment, our results may provide insights into multi-agent settings where agents "unconsciously" discriminating against each other could inexplicably increase or decrease cooperation. 

---
# DrugReasoner: Interpretable Drug Approval Prediction with a Reasoning-augmented Language Model 

**Authors**: Mohammadreza Ghaffarzadeh-Esfahani, Ali Motahharynia, Nahid Yousefian, Navid Mazrouei, Jafar Ghaisari, Yousof Gheisari  

**Link**: [PDF](https://arxiv.org/pdf/2508.18579)  

**Abstract**: Drug discovery is a complex and resource-intensive process, making early prediction of approval outcomes critical for optimizing research investments. While classical machine learning and deep learning methods have shown promise in drug approval prediction, their limited interpretability constraints their impact. Here, we present DrugReasoner, a reasoning-based large language model (LLM) built on the LLaMA architecture and fine-tuned with group relative policy optimization (GRPO) to predict the likelihood of small-molecule approval. DrugReasoner integrates molecular descriptors with comparative reasoning against structurally similar approved and unapproved compounds, generating predictions alongside step-by-step rationales and confidence scores. DrugReasoner achieved robust performance with an AUC of 0.732 and an F1 score of 0.729 on the validation set and 0.725 and 0.718 on the test set, respectively. These results outperformed conventional baselines, including logistic regression, support vector machine, and k-nearest neighbors and had competitive performance relative to XGBoost. On an external independent dataset, DrugReasoner outperformed both baseline and the recently developed ChemAP model, achieving an AUC of 0.728 and an F1-score of 0.774, while maintaining high precision and balanced sensitivity, demonstrating robustness in real-world scenarios. These findings demonstrate that DrugReasoner not only delivers competitive predictive accuracy but also enhances transparency through its reasoning outputs, thereby addressing a key bottleneck in AI-assisted drug discovery. This study highlights the potential of reasoning-augmented LLMs as interpretable and effective tools for pharmaceutical decision-making. 

---
# LaQual: A Novel Framework for Automated Evaluation of LLM App Quality 

**Authors**: Yan Wang, Xinyi Hou, Yanjie Zhao, Weiguo Lin, Haoyu Wang, Junjun Si  

**Link**: [PDF](https://arxiv.org/pdf/2508.18636)  

**Abstract**: LLM app stores are quickly emerging as platforms that gather a wide range of intelligent applications based on LLMs, giving users many choices for content creation, coding support, education, and more. However, the current methods for ranking and recommending apps in these stores mostly rely on static metrics like user activity and favorites, which makes it hard for users to efficiently find high-quality apps. To address these challenges, we propose LaQual, an automated framework for evaluating the quality of LLM apps. LaQual consists of three main stages: first, it labels and classifies LLM apps in a hierarchical way to accurately match them to different scenarios; second, it uses static indicators, such as time-weighted user engagement and functional capability metrics, to filter out low-quality apps; and third, it conducts a dynamic, scenario-adaptive evaluation, where the LLM itself generates scenario-specific evaluation metrics, scoring rules, and tasks for a thorough quality assessment. Experiments on a popular LLM app store show that LaQual is effective. Its automated scores are highly consistent with human judgments (with Spearman's rho of 0.62 and p=0.006 in legal consulting, and rho of 0.60 and p=0.009 in travel planning). By effectively screening, LaQual can reduce the pool of candidate LLM apps by 66.7% to 81.3%. User studies further confirm that LaQual significantly outperforms baseline systems in decision confidence, comparison efficiency (with average scores of 5.45 compared to 3.30), and the perceived value of its evaluation reports (4.75 versus 2.25). Overall, these results demonstrate that LaQual offers a scalable, objective, and user-centered solution for finding and recommending high-quality LLM apps in real-world use cases. 

---
# A Case Study on the Effectiveness of LLMs in Verification with Proof Assistants 

**Authors**: Barış Bayazıt, Yao Li, Xujie Si  

**Link**: [PDF](https://arxiv.org/pdf/2508.18587)  

**Abstract**: Large language models (LLMs) can potentially help with verification using proof assistants by automating proofs. However, it is unclear how effective LLMs are in this task. In this paper, we perform a case study based on two mature Rocq projects: the hs-to-coq tool and Verdi. We evaluate the effectiveness of LLMs in generating proofs by both quantitative and qualitative analysis. Our study finds that: (1) external dependencies and context in the same source file can significantly help proof generation; (2) LLMs perform great on small proofs but can also generate large proofs; (3) LLMs perform differently on different verification projects; and (4) LLMs can generate concise and smart proofs, apply classical techniques to new definitions, but can also make odd mistakes. 

---
# VERIRL: Boosting the LLM-based Verilog Code Generation via Reinforcement Learning 

**Authors**: Fu Teng, Miao Pan, Xuhong Zhang, Zhezhi He, Yiyao Yang, Xinyi Chai, Mengnan Qi, Liqiang Lu, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.18462)  

**Abstract**: Recent advancements in code generation have shown remarkable success across software domains, yet hardware description languages (HDLs) such as Verilog remain underexplored due to their concurrency semantics, syntactic rigidity, and simulation complexity. In this work, we address these challenges by introducing a reinforcement learning (RL) framework tailored for Verilog code generation. We first construct Veribench-53K, a high-quality dataset curated from over 700K Verilog problems, enriched with structured prompts, complexity labels, and diverse testbenches. To tackle the problem of sparse and noisy reward signals, we propose a Trace-back based Rescore mechanism that leverages reasoning paths and iterative refinement to enhance feedback reliability and support reward model training. Furthermore, to mitigate catastrophic forgetting and overfitting during RL fine-tuning, we introduce a sample-balanced weighting strategy that adaptively balances learning dynamics based on reward-probability distributions. These innovations are integrated into an iterative RL pipeline that co-evolves the policy and reward models. In contrast to recent work such as CraftRTL, which relies on large-scale closed-source model distillation, and DeepSeek-style approaches that struggle with sparse feedback, our method demonstrates superior performance using a smaller but high-quality dataset combined with RL optimization. Experiments on Verilog generation tasks demonstrate state-of-the-art performance, with substantial gains in test pass rate, functional correctness, and compilation robustness. Our findings highlight the potential of RL-driven approaches for structured code generation in hardware-centric domains. VERIRL is publicly available at this https URL. 

---
# EAI-Avatar: Emotion-Aware Interactive Talking Head Generation 

**Authors**: Haijie Yang, Zhenyu Zhang, Hao Tang, Jianjun Qian, Jian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18337)  

**Abstract**: Generative models have advanced rapidly, enabling impressive talking head generation that brings AI to life. However, most existing methods focus solely on one-way portrait animation. Even the few that support bidirectional conversational interactions lack precise emotion-adaptive capabilities, significantly limiting their practical applicability. In this paper, we propose EAI-Avatar, a novel emotion-aware talking head generation framework for dyadic interactions. Leveraging the dialogue generation capability of large language models (LLMs, e.g., GPT-4), our method produces temporally consistent virtual avatars with rich emotional variations that seamlessly transition between speaking and listening states. Specifically, we design a Transformer-based head mask generator that learns temporally consistent motion features in a latent mask space, capable of generating arbitrary-length, temporally consistent mask sequences to constrain head motions. Furthermore, we introduce an interactive talking tree structure to represent dialogue state transitions, where each tree node contains information such as child/parent/sibling nodes and the current character's emotional state. By performing reverse-level traversal, we extract rich historical emotional cues from the current node to guide expression synthesis. Extensive experiments demonstrate the superior performance and effectiveness of our method. 

---
# Toward Generalized Autonomous Agents: A Neuro-Symbolic AI Framework for Integrating Social and Technical Support in Education 

**Authors**: Ryan Hare, Ying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18406)  

**Abstract**: One of the enduring challenges in education is how to empower students to take ownership of their learning by setting meaningful goals, tracking their progress, and adapting their strategies when faced with setbacks. Research has shown that this form of leaner-centered learning is best cultivated through structured, supportive environments that promote guided practice, scaffolded inquiry, and collaborative dialogue. In response, educational efforts have increasingly embraced artificial-intelligence (AI)-powered digital learning environments, ranging from educational apps and virtual labs to serious games. Recent advances in large language models (LLMs) and neuro-symbolic systems, meanwhile, offer a transformative opportunity to reimagine how support is delivered in digital learning environments. LLMs are enabling socially interactive learning experiences and scalable, cross-domain learning support that can adapt instructional strategies across varied subjects and contexts. In parallel, neuro-symbolic AI provides new avenues for designing these agents that are not only adaptive but also scalable across domains. Based on these remarks, this paper presents a multi-agent, neuro-symbolic framework designed to resolve the aforementioned challenges. The framework assigns distinct pedagogical roles to specialized agents: an RL-based 'tutor' agent provides authoritative, non-verbal scaffolding, while a proactive, LLM-powered 'peer' agent facilitates the social dimensions of learning. While prior work has explored such agents in isolation, our framework's novelty lies in unifying them through a central educational ontology. Through case studies in both college-level and middle school settings, we demonstrate the framework's adaptability across domains. We conclude by outlining key insights and future directions for advancing AI-driven learning environments. 

---
# ProtoEHR: Hierarchical Prototype Learning for EHR-based Healthcare Predictions 

**Authors**: Zi Cai, Yu Liu, Zhiyao Luo, Tingting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18313)  

**Abstract**: Digital healthcare systems have enabled the collection of mass healthcare data in electronic healthcare records (EHRs), allowing artificial intelligence solutions for various healthcare prediction tasks. However, existing studies often focus on isolated components of EHR data, limiting their predictive performance and interpretability. To address this gap, we propose ProtoEHR, an interpretable hierarchical prototype learning framework that fully exploits the rich, multi-level structure of EHR data to enhance healthcare predictions. More specifically, ProtoEHR models relationships within and across three hierarchical levels of EHRs: medical codes, hospital visits, and patients. We first leverage large language models to extract semantic relationships among medical codes and construct a medical knowledge graph as the knowledge source. Building on this, we design a hierarchical representation learning framework that captures contextualized representations across three levels, while incorporating prototype information within each level to capture intrinsic similarities and improve generalization. To perform a comprehensive assessment, we evaluate ProtoEHR in two public datasets on five clinically significant tasks, including prediction of mortality, prediction of readmission, prediction of length of stay, drug recommendation, and prediction of phenotype. The results demonstrate the ability of ProtoEHR to make accurate, robust, and interpretable predictions compared to baselines in the literature. Furthermore, ProtoEHR offers interpretable insights on code, visit, and patient levels to aid in healthcare prediction. 

---
# Consensus Is All You Need: Gossip-Based Reasoning Among Large Language Models 

**Authors**: Saksham Arora  

**Link**: [PDF](https://arxiv.org/pdf/2508.18292)  

**Abstract**: Large language models have advanced rapidly, but no single model excels in every area -- each has its strengths and weaknesses. Instead of relying on one model alone, we take inspiration from gossip protocols in distributed systems, where information is exchanged with peers until they all come to an agreement. In this setup, models exchange answers and gradually work toward a shared solution. Each LLM acts as a node in a peer-to-peer network, sharing responses and thought processes to reach a collective decision. Our results show that this "gossip-based consensus" leads to robust, resilient, and accurate multi-agent AI reasoning. It helps overcome the weaknesses of individual models and brings out their collective strengths. This approach is similar to how humans build consensus, making AI seem more collaborative and trustworthy instead of just a black-box program. 

---
# Collaborative Intelligence: Topic Modelling of Large Language Model use in Live Cybersecurity Operations 

**Authors**: Martin Lochner, Keegan Keplinger  

**Link**: [PDF](https://arxiv.org/pdf/2508.18488)  

**Abstract**: Objective: This work describes the topic modelling of Security Operations Centre (SOC) use of a large language model (LLM), during live security operations. The goal is to better understand how these specialists voluntarily use this tool.
Background: Human-automation teams have been extensively studied, but transformer-based language models have sparked a new wave of collaboration. SOC personnel at a major cybersecurity provider used an LLM to support live security operations. This study examines how these specialists incorporated the LLM into their work.
Method: Our data set is the result of 10 months of SOC operators accessing GPT-4 over an internally deployed HTTP-based chat application. We performed two topic modelling exercises, first using the established BERTopic model (Grootendorst, 2022), and second, using a novel topic modeling workflow.
Results: Both the BERTopic analysis and novel modelling approach revealed that SOC operators primarily used the LLM to facilitate their understanding of complex text strings. Variations on this use-case accounted for ~40% of SOC LLM usage.
Conclusion: SOC operators are required to rapidly interpret complex commands and similar information. Their natural tendency to leverage LLMs to support this activity indicates that their workflow can be supported and augmented by designing collaborative LLM tools for use in the SOC.
Application: This work can aid in creating next-generation tools for Security Operations Centres. By understanding common use-cases, we can develop workflows supporting SOC task flow. One example is a right-click context menu for executing a command line analysis LLM call directly in the SOC environment. 

---
