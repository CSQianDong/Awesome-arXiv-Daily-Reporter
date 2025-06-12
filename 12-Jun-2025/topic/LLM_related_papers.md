# PGDA-KGQA: A Prompt-Guided Generative Framework with Multiple Data Augmentation Strategies for Knowledge Graph Question Answering 

**Authors**: Xiujun Zhou, Pingjian Zhang, Deyou Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09414)  

**Abstract**: Knowledge Graph Question Answering (KGQA) is a crucial task in natural language processing that requires reasoning over knowledge graphs (KGs) to answer natural language questions. Recent methods utilizing large language models (LLMs) have shown remarkable semantic parsing capabilities but are limited by the scarcity of diverse annotated data and multi-hop reasoning samples. Traditional data augmentation approaches are focus mainly on single-hop questions and prone to semantic distortion, while LLM-based methods primarily address semantic distortion but usually neglect multi-hop reasoning, thus limiting data diversity. The scarcity of multi-hop samples further weakens models' generalization. To address these issues, we propose PGDA-KGQA, a prompt-guided generative framework with multiple data augmentation strategies for KGQA. At its core, PGDA-KGQA employs a unified prompt-design paradigm: by crafting meticulously engineered prompts that integrate the provided textual content, it leverages LLMs to generate large-scale (question, logical form) pairs for model training. Specifically, PGDA-KGQA enriches its training set by: (1) generating single-hop pseudo questions to improve the alignment of question semantics with KG relations; (2) applying semantic-preserving question rewriting to improve robustness against linguistic variations; (3) employing answer-guided reverse path exploration to create realistic multi-hop questions. By adopting an augment-generate-retrieve semantic parsing pipeline, PGDA-KGQA utilizes the augmented data to enhance the accuracy of logical form generation and thus improve answer retrieval performance. Experiments demonstrate that outperforms state-of-the-art methods on standard KGQA datasets, achieving improvements on WebQSP by 2.8%, 1.2%, and 3.1% and on ComplexWebQuestions by 1.8%, 1.1%, and 2.4% in F1, Hits@1, and Accuracy, respectively. 

---
# Learning Efficient and Generalizable Graph Retriever for Knowledge-Graph Question Answering 

**Authors**: Tianjun Yao, Haoxuan Li, Zhiqiang Shen, Pan Li, Tongliang Liu, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09645)  

**Abstract**: Large Language Models (LLMs) have shown strong inductive reasoning ability across various domains, but their reliability is hindered by the outdated knowledge and hallucinations. Retrieval-Augmented Generation mitigates these issues by grounding LLMs with external knowledge; however, most existing RAG pipelines rely on unstructured text, limiting interpretability and structured reasoning. Knowledge graphs, which represent facts as relational triples, offer a more structured and compact alternative. Recent studies have explored integrating knowledge graphs with LLMs for knowledge graph question answering (KGQA), with a significant proportion adopting the retrieve-then-reasoning paradigm. In this framework, graph-based retrievers have demonstrated strong empirical performance, yet they still face challenges in generalization ability. In this work, we propose RAPL, a novel framework for efficient and effective graph retrieval in KGQA. RAPL addresses these limitations through three aspects: (1) a two-stage labeling strategy that combines heuristic signals with parametric models to provide causally grounded supervision; (2) a model-agnostic graph transformation approach to capture both intra- and inter-triple interactions, thereby enhancing representational capacity; and (3) a path-based reasoning strategy that facilitates learning from the injected rational knowledge, and supports downstream reasoner through structured inputs. Empirically, RAPL outperforms state-of-the-art methods by $2.66\%-20.34\%$, and significantly reduces the performance gap between smaller and more powerful LLM-based reasoners, as well as the gap under cross-dataset settings, highlighting its superior retrieval capability and generalizability. Codes are available at: this https URL. 

---
# VerIF: Verification Engineering for Reinforcement Learning in Instruction Following 

**Authors**: Hao Peng, Yunjia Qi, Xiaozhi Wang, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09942)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a key technique for enhancing large language models (LLMs), with verification engineering playing a central role. However, best practices for RL in instruction following remain underexplored. In this work, we explore the verification challenge in RL for instruction following and propose VerIF, a verification method that combines rule-based code verification with LLM-based verification from a large reasoning model (e.g., QwQ-32B). To support this approach, we construct a high-quality instruction-following dataset, VerInstruct, containing approximately 22,000 instances with associated verification signals. We apply RL training with VerIF to two models, achieving significant improvements across several representative instruction-following benchmarks. The trained models reach state-of-the-art performance among models of comparable size and generalize well to unseen constraints. We further observe that their general capabilities remain unaffected, suggesting that RL with VerIF can be integrated into existing RL recipes to enhance overall model performance. We have released our datasets, codes, and models to facilitate future research at this https URL. 

---
# PersonaLens: A Benchmark for Personalization Evaluation in Conversational AI Assistants 

**Authors**: Zheng Zhao, Clara Vania, Subhradeep Kayal, Naila Khan, Shay B. Cohen, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.09902)  

**Abstract**: Large language models (LLMs) have advanced conversational AI assistants. However, systematically evaluating how well these assistants apply personalization--adapting to individual user preferences while completing tasks--remains challenging. Existing personalization benchmarks focus on chit-chat, non-conversational tasks, or narrow domains, failing to capture the complexities of personalized task-oriented assistance. To address this, we introduce PersonaLens, a comprehensive benchmark for evaluating personalization in task-oriented AI assistants. Our benchmark features diverse user profiles equipped with rich preferences and interaction histories, along with two specialized LLM-based agents: a user agent that engages in realistic task-oriented dialogues with AI assistants, and a judge agent that employs the LLM-as-a-Judge paradigm to assess personalization, response quality, and task success. Through extensive experiments with current LLM assistants across diverse tasks, we reveal significant variability in their personalization capabilities, providing crucial insights for advancing conversational AI systems. 

---
# Step-by-step Instructions and a Simple Tabular Output Format Improve the Dependency Parsing Accuracy of LLMs 

**Authors**: Hiroshi Matsuda, Chunpeng Ma, Masayuki Asahara  

**Link**: [PDF](https://arxiv.org/pdf/2506.09983)  

**Abstract**: Recent advances in large language models (LLMs) have enabled impressive performance in various tasks. However, standard prompting often struggles to produce structurally valid and accurate outputs, especially in dependency parsing. We propose a novel step-by-step instruction strategy, where universal part-of-speech tagging precedes the prediction of syntactic heads and dependency labels, and a simplified CoNLL-U like output format, our method achieves state-of-the-art accuracy on Universal Dependencies datasets across 17 languages without hallucination or contamination. We further show that multilingual fine-tuning simultaneously improves cross-language generalization performance. Our results highlight the effectiveness of explicit reasoning steps in LLM-based parsing and offer a scalable, format-consistent alternative to bracket-based approaches. 

---
# Dataset of News Articles with Provenance Metadata for Media Relevance Assessment 

**Authors**: Tomas Peterka, Matyas Bohacek  

**Link**: [PDF](https://arxiv.org/pdf/2506.09847)  

**Abstract**: Out-of-context and misattributed imagery is the leading form of media manipulation in today's misinformation and disinformation landscape. The existing methods attempting to detect this practice often only consider whether the semantics of the imagery corresponds to the text narrative, missing manipulation so long as the depicted objects or scenes somewhat correspond to the narrative at hand. To tackle this, we introduce News Media Provenance Dataset, a dataset of news articles with provenance-tagged images. We formulate two tasks on this dataset, location of origin relevance (LOR) and date and time of origin relevance (DTOR), and present baseline results on six large language models (LLMs). We identify that, while the zero-shot performance on LOR is promising, the performance on DTOR hinders, leaving room for specialized architectures and future work. 

---
# Large Language Models for Toxic Language Detection in Low-Resource Balkan Languages 

**Authors**: Amel Muminovic, Amela Kadric Muminovic  

**Link**: [PDF](https://arxiv.org/pdf/2506.09992)  

**Abstract**: Online toxic language causes real harm, especially in regions with limited moderation tools. In this study, we evaluate how large language models handle toxic comments in Serbian, Croatian, and Bosnian, languages with limited labeled data. We built and manually labeled a dataset of 4,500 YouTube and TikTok comments drawn from videos across diverse categories, including music, politics, sports, modeling, influencer content, discussions of sexism, and general topics. Four models (GPT-3.5 Turbo, GPT-4.1, Gemini 1.5 Pro, and Claude 3 Opus) were tested in two modes: zero-shot and context-augmented. We measured precision, recall, F1 score, accuracy and false positive rates. Including a short context snippet raised recall by about 0.12 on average and improved F1 score by up to 0.10, though it sometimes increased false positives. The best balance came from Gemini in context-augmented mode, reaching an F1 score of 0.82 and accuracy of 0.82, while zero-shot GPT-4.1 led on precision and had the lowest false alarms. We show how adding minimal context can improve toxic language detection in low-resource settings and suggest practical strategies such as improved prompt design and threshold calibration. These results show that prompt design alone can yield meaningful gains in toxicity detection for underserved Balkan language communities. 

---
# Causal Sufficiency and Necessity Improves Chain-of-Thought Reasoning 

**Authors**: Xiangning Yu, Zhuohan Wang, Linyi Yang, Haoxuan Li, Anjie Liu, Xiao Xue, Jun Wang, Mengyue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09853)  

**Abstract**: Chain-of-Thought (CoT) prompting plays an indispensable role in endowing large language models (LLMs) with complex reasoning capabilities. However, CoT currently faces two fundamental challenges: (1) Sufficiency, which ensures that the generated intermediate inference steps comprehensively cover and substantiate the final conclusion; and (2) Necessity, which identifies the inference steps that are truly indispensable for the soundness of the resulting answer. We propose a causal framework that characterizes CoT reasoning through the dual lenses of sufficiency and necessity. Incorporating causal Probability of Sufficiency and Necessity allows us not only to determine which steps are logically sufficient or necessary to the prediction outcome, but also to quantify their actual influence on the final reasoning outcome under different intervention scenarios, thereby enabling the automated addition of missing steps and the pruning of redundant ones. Extensive experimental results on various mathematical and commonsense reasoning benchmarks confirm substantial improvements in reasoning efficiency and reduced token usage without sacrificing accuracy. Our work provides a promising direction for improving LLM reasoning performance and cost-effectiveness. 

---
# Do LLMs Give Psychometrically Plausible Responses in Educational Assessments? 

**Authors**: Andreas Säuberli, Diego Frassinelli, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2506.09796)  

**Abstract**: Knowing how test takers answer items in educational assessments is essential for test development, to evaluate item quality, and to improve test validity. However, this process usually requires extensive pilot studies with human participants. If large language models (LLMs) exhibit human-like response behavior to test items, this could open up the possibility of using them as pilot participants to accelerate test development. In this paper, we evaluate the human-likeness or psychometric plausibility of responses from 18 instruction-tuned LLMs with two publicly available datasets of multiple-choice test items across three subjects: reading, U.S. history, and economics. Our methodology builds on two theoretical frameworks from psychometrics which are commonly used in educational assessment, classical test theory and item response theory. The results show that while larger models are excessively confident, their response distributions can be more human-like when calibrated with temperature scaling. In addition, we find that LLMs tend to correlate better with humans in reading comprehension items compared to other subjects. However, the correlations are not very strong overall, indicating that LLMs should not be used for piloting educational assessments in a zero-shot setting. 

---
# Query-Level Uncertainty in Large Language Models 

**Authors**: Lihu Chen, Gaël Varoquaux  

**Link**: [PDF](https://arxiv.org/pdf/2506.09669)  

**Abstract**: It is important for Large Language Models to be aware of the boundary of their knowledge, the mechanism of identifying known and unknown queries. This type of awareness can help models perform adaptive inference, such as invoking RAG, engaging in slow and deep thinking, or adopting the abstention mechanism, which is beneficial to the development of efficient and trustworthy AI. In this work, we propose a method to detect knowledge boundaries via Query-Level Uncertainty, which aims to determine if the model is able to address a given query without generating any tokens. To this end, we introduce a novel and training-free method called \emph{Internal Confidence}, which leverages self-evaluations across layers and tokens. Empirical results on both factual QA and mathematical reasoning tasks demonstrate that our internal confidence can outperform several baselines. Furthermore, we showcase that our proposed method can be used for efficient RAG and model cascading, which is able to reduce inference costs while maintaining performance. 

---
# From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring 

**Authors**: Yang Li, Qiang Sheng, Yehan Yang, Xueyao Zhang, Juan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09996)  

**Abstract**: Though safety alignment has been applied to most large language models (LLMs), LLM service providers generally deploy a subsequent moderation as the external safety guardrail in real-world products. Existing moderators mainly practice a conventional full detection, which determines the harmfulness based on the complete LLM output, causing high service latency. Recent works pay more attention to partial detection where moderators oversee the generation midway and early stop the output if harmfulness is detected, but they directly apply moderators trained with the full detection paradigm to incomplete outputs, introducing a training-inference gap that lowers the performance. In this paper, we explore how to form a data-and-model solution that natively supports partial detection. For the data, we construct FineHarm, a dataset consisting of 29K prompt-response pairs with fine-grained annotations to provide reasonable supervision for token-level training. Then, we propose the streaming content monitor, which is trained with dual supervision of response- and token-level labels and can follow the output stream of LLM to make a timely judgment of harmfulness. Experiments show that SCM gains 0.95+ in macro F1 score that is comparable to full detection, by only seeing the first 18% of tokens in responses on average. Moreover, the SCM can serve as a pseudo-harmfulness annotator for improving safety alignment and lead to a higher harmlessness score than DPO. 

---
# Attention Head Embeddings with Trainable Deep Kernels for Hallucination Detection in LLMs 

**Authors**: Rodion Oblovatny, Alexandra Bazarova, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2506.09886)  

**Abstract**: We present a novel approach for detecting hallucinations in large language models (LLMs) by analyzing the probabilistic divergence between prompt and response hidden-state distributions. Counterintuitively, we find that hallucinated responses exhibit smaller deviations from their prompts compared to grounded responses, suggesting that hallucinations often arise from superficial rephrasing rather than substantive reasoning. Leveraging this insight, we propose a model-intrinsic detection method that uses distributional distances as principled hallucination scores, eliminating the need for external knowledge or auxiliary models. To enhance sensitivity, we employ deep learnable kernels that automatically adapt to capture nuanced geometric differences between distributions. Our approach outperforms existing baselines, demonstrating state-of-the-art performance on several benchmarks. The method remains competitive even without kernel training, offering a robust, scalable solution for hallucination detection. 

---
# When Detection Fails: The Power of Fine-Tuned Models to Generate Human-Like Social Media Text 

**Authors**: Hillary Dawkins, Kathleen C. Fraser, Svetlana Kiritchenko  

**Link**: [PDF](https://arxiv.org/pdf/2506.09975)  

**Abstract**: Detecting AI-generated text is a difficult problem to begin with; detecting AI-generated text on social media is made even more difficult due to the short text length and informal, idiosyncratic language of the internet. It is nonetheless important to tackle this problem, as social media represents a significant attack vector in online influence campaigns, which may be bolstered through the use of mass-produced AI-generated posts supporting (or opposing) particular policies, decisions, or events. We approach this problem with the mindset and resources of a reasonably sophisticated threat actor, and create a dataset of 505,159 AI-generated social media posts from a combination of open-source, closed-source, and fine-tuned LLMs, covering 11 different controversial topics. We show that while the posts can be detected under typical research assumptions about knowledge of and access to the generating models, under the more realistic assumption that an attacker will not release their fine-tuned model to the public, detectability drops dramatically. This result is confirmed with a human study. Ablation experiments highlight the vulnerability of various detection algorithms to fine-tuned LLMs. This result has implications across all detection domains, since fine-tuning is a generally applicable and realistic LLM use case. 

---
# Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking 

**Authors**: Wuwei Zhang, Fangcong Yin, Howard Yen, Danqi Chen, Xi Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.09944)  

**Abstract**: Recent work has identified retrieval heads (Wu et al., 2025b), a subset of attention heads responsible for retrieving salient information in long-context language models (LMs), as measured by their copy-paste behavior in Needle-in-a-Haystack tasks. In this paper, we introduce QRHEAD (Query-Focused Retrieval Head), an improved set of attention heads that enhance retrieval from long context. We identify QRHEAD by aggregating attention scores with respect to the input query, using a handful of examples from real-world tasks (e.g., long-context QA). We further introduce QR- RETRIEVER, an efficient and effective retriever that uses the accumulated attention mass of QRHEAD as retrieval scores. We use QR- RETRIEVER for long-context reasoning by selecting the most relevant parts with the highest retrieval scores. On multi-hop reasoning tasks LongMemEval and CLIPPER, this yields over 10% performance gains over full context and outperforms strong dense retrievers. We also evaluate QRRETRIEVER as a re-ranker on the BEIR benchmark and find that it achieves strong zero-shot performance, outperforming other LLM-based re-rankers such as RankGPT. Further analysis shows that both the querycontext attention scoring and task selection are crucial for identifying QRHEAD with strong downstream utility. Overall, our work contributes a general-purpose retriever and offers interpretability insights into the long-context capabilities of LMs. 

---
# From Symbolic to Neural and Back: Exploring Knowledge Graph-Large Language Model Synergies 

**Authors**: Blaž Škrlj, Boshko Koloski, Senja Pollak, Nada Lavrač  

**Link**: [PDF](https://arxiv.org/pdf/2506.09566)  

**Abstract**: Integrating structured knowledge from Knowledge Graphs (KGs) into Large Language Models (LLMs) enhances factual grounding and reasoning capabilities. This survey paper systematically examines the synergy between KGs and LLMs, categorizing existing approaches into two main groups: KG-enhanced LLMs, which improve reasoning, reduce hallucinations, and enable complex question answering; and LLM-augmented KGs, which facilitate KG construction, completion, and querying. Through comprehensive analysis, we identify critical gaps and highlight the mutual benefits of structured knowledge integration. Compared to existing surveys, our study uniquely emphasizes scalability, computational efficiency, and data quality. Finally, we propose future research directions, including neuro-symbolic integration, dynamic KG updating, data reliability, and ethical considerations, paving the way for intelligent systems capable of managing more complex real-world knowledge tasks. 

---
# Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA 

**Authors**: Nikolas Evkarpidi, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2506.09657)  

**Abstract**: This paper presents a system developed for SemEval 2025 Task 8: Question Answering (QA) over tabular data. Our approach integrates several key components: text-to-SQL and text-to-code generation modules, a self-correction mechanism, and a retrieval-augmented generation (RAG). Additionally, it includes an end-to-end (E2E) module, all orchestrated by a large language model (LLM). Through ablation studies, we analyzed the effects of different parts of our pipeline and identified the challenges that are still present in this field. During the evaluation phase of the competition, our solution achieved an accuracy of 80%, resulting in a top-13 ranking among the 38 participating teams. Our pipeline demonstrates a significant improvement in accuracy for open-source models and achieves a performance comparable to proprietary LLMs in QA tasks over tables. The code is available at GitHub repository. 

---
# ComfyUI-R1: Exploring Reasoning Models for Workflow Generation 

**Authors**: Zhenran Xu, Yiyu Wang, Xue Yang, Longyue Wang, Weihua Luo, Kaifu Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09790)  

**Abstract**: AI-generated content has evolved from monolithic models to modular workflows, particularly on platforms like ComfyUI, enabling customization in creative pipelines. However, crafting effective workflows requires great expertise to orchestrate numerous specialized components, presenting a steep learning curve for users. To address this challenge, we introduce ComfyUI-R1, the first large reasoning model for automated workflow generation. Starting with our curated dataset of 4K workflows, we construct long chain-of-thought (CoT) reasoning data, including node selection, workflow planning, and code-level workflow representation. ComfyUI-R1 is trained through a two-stage framework: (1) CoT fine-tuning for cold start, adapting models to the ComfyUI domain; (2) reinforcement learning for incentivizing reasoning capability, guided by a fine-grained rule-metric hybrid reward, ensuring format validity, structural integrity, and node-level fidelity. Experiments show that our 7B-parameter model achieves a 97\% format validity rate, along with high pass rate, node-level and graph-level F1 scores, significantly surpassing prior state-of-the-art methods that employ leading closed-source models such as GPT-4o and Claude series. Further analysis highlights the critical role of the reasoning process and the advantage of transforming workflows into code. Qualitative comparison reveals our strength in synthesizing intricate workflows with diverse nodes, underscoring the potential of long CoT reasoning in AI art creation. 

---
# ReasonMed: A 370K Multi-Agent Generated Dataset for Advancing Medical Reasoning 

**Authors**: Yu Sun, Xingyu Qian, Weiwen Xu, Hao Zhang, Chenghao Xiao, Long Li, Yu Rong, Wenbing Huang, Qifeng Bai, Tingyang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09513)  

**Abstract**: Though reasoning-based large language models (LLMs) have excelled in mathematics and programming, their capabilities in knowledge-intensive medical question answering remain underexplored. To address this, we introduce ReasonMed, the largest medical reasoning dataset, comprising 370k high-quality examples distilled from 1.7 million initial reasoning paths generated by various LLMs. ReasonMed is constructed through a \textit{multi-agent verification and refinement process}, where we design an \textit{Error Refiner} to enhance the reasoning paths by identifying and correcting error-prone steps flagged by a verifier. Leveraging ReasonMed, we systematically investigate best practices for training medical reasoning models and find that combining detailed Chain-of-Thought (CoT) reasoning with concise answer summaries yields the most effective fine-tuning strategy. Based on this strategy, we train ReasonMed-7B, which sets a new benchmark for sub-10B models, outperforming the prior best by 4.17\% and even exceeding LLaMA3.1-70B on PubMedQA by 4.60\%. 

---
# Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning 

**Authors**: Jiayi Yuan, Hao Li, Xinheng Ding, Wenya Xie, Yu-Jhe Li, Wentian Zhao, Kun Wan, Jing Shi, Xia Hu, Zirui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09501)  

**Abstract**: Large Language Models (LLMs) are now integral across various domains and have demonstrated impressive performance. Progress, however, rests on the premise that benchmark scores are both accurate and reproducible. We demonstrate that the reproducibility of LLM performance is fragile: changing system configuration such as evaluation batch size, GPU count, and GPU version can introduce significant difference in the generated responses. This issue is especially pronounced in reasoning models, where minor rounding differences in early tokens can cascade into divergent chains of thought, ultimately affecting accuracy. For instance, under bfloat16 precision with greedy decoding, a reasoning model like DeepSeek-R1-Distill-Qwen-7B can exhibit up to 9% variation in accuracy and 9,000 tokens difference in response length due to differences in GPU count, type, and evaluation batch size. We trace the root cause of this variability to the non-associative nature of floating-point arithmetic under limited numerical precision. This work presents the first systematic investigation into how numerical precision affects reproducibility in LLM inference. Through carefully controlled experiments across various hardware, software, and precision settings, we quantify when and how model outputs diverge. Our analysis reveals that floating-point precision -- while critical for reproducibility -- is often neglected in evaluation practices. Inspired by this, we develop a lightweight inference pipeline, dubbed LayerCast, that stores weights in 16-bit precision but performs all computations in FP32, balancing memory efficiency with numerical stability. Code is available at this https URL. 

---
# Inv-Entropy: A Fully Probabilistic Framework for Uncertainty Quantification in Language Models 

**Authors**: Haoyi Song, Ruihan Ji, Naichen Shi, Fan Lai, Raed Al Kontar  

**Link**: [PDF](https://arxiv.org/pdf/2506.09684)  

**Abstract**: Large language models (LLMs) have transformed natural language processing, but their reliable deployment requires effective uncertainty quantification (UQ). Existing UQ methods are often heuristic and lack a probabilistic foundation. This paper begins by providing a theoretical justification for the role of perturbations in UQ for LLMs. We then introduce a dual random walk perspective, modeling input-output pairs as two Markov chains with transition probabilities defined by semantic similarity. Building on this, we propose a fully probabilistic framework based on an inverse model, which quantifies uncertainty by evaluating the diversity of the input space conditioned on a given output through systematic perturbations. Within this framework, we define a new uncertainty measure, Inv-Entropy. A key strength of our framework is its flexibility: it supports various definitions of uncertainty measures, embeddings, perturbation strategies, and similarity metrics. We also propose GAAP, a perturbation algorithm based on genetic algorithms, which enhances the diversity of sampled inputs. In addition, we introduce a new evaluation metric, Temperature Sensitivity of Uncertainty (TSU), which directly assesses uncertainty without relying on correctness as a proxy. Extensive experiments demonstrate that Inv-Entropy outperforms existing semantic UQ methods. The code to reproduce the results can be found at this https URL. 

---
# Hidden in Plain Sight: Evaluation of the Deception Detection Capabilities of LLMs in Multimodal Settings 

**Authors**: Md Messal Monem Miah, Adrita Anika, Xi Shi, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09424)  

**Abstract**: Detecting deception in an increasingly digital world is both a critical and challenging task. In this study, we present a comprehensive evaluation of the automated deception detection capabilities of Large Language Models (LLMs) and Large Multimodal Models (LMMs) across diverse domains. We assess the performance of both open-source and commercial LLMs on three distinct datasets: real life trial interviews (RLTD), instructed deception in interpersonal scenarios (MU3D), and deceptive reviews (OpSpam). We systematically analyze the effectiveness of different experimental setups for deception detection, including zero-shot and few-shot approaches with random or similarity-based in-context example selection. Our results show that fine-tuned LLMs achieve state-of-the-art performance on textual deception detection tasks, while LMMs struggle to fully leverage cross-modal cues. Additionally, we analyze the impact of auxiliary features, such as non-verbal gestures and video summaries, and examine the effectiveness of different prompting strategies, including direct label generation and chain-of-thought reasoning. Our findings provide key insights into how LLMs process and interpret deceptive cues across modalities, highlighting their potential and limitations in real-world deception detection applications. 

---
# UniToMBench: Integrating Perspective-Taking to Improve Theory of Mind in LLMs 

**Authors**: Prameshwar Thiyagarajan, Vaishnavi Parimi, Shamant Sai, Soumil Garg, Zhangir Meirbek, Nitin Yarlagadda, Kevin Zhu, Chris Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.09450)  

**Abstract**: Theory of Mind (ToM), the ability to understand the mental states of oneself and others, remains a challenging area for large language models (LLMs), which often fail to predict human mental states accurately. In this paper, we introduce UniToMBench, a unified benchmark that integrates the strengths of SimToM and TOMBENCH to systematically improve and assess ToM capabilities in LLMs by integrating multi-interaction task designs and evolving story scenarios. Supported by a custom dataset of over 1,000 hand-written scenarios, UniToMBench combines perspective-taking techniques with diverse evaluation metrics to better stimulate social cognition in LLMs. Through evaluation, we observe that while models like GPT-4o and GPT-4o Mini show consistently high accuracy in tasks involving emotional and belief-related scenarios, with results usually above 80%, there is significant variability in their performance across knowledge-based tasks. These results highlight both the strengths and limitations of current LLMs in ToM-related tasks, underscoring the value of UniToMBench as a comprehensive tool for future development. Our code is publicly available here: this https URL. 

---
# Token Constraint Decoding Improves Robustness on Question Answering for Large Language Models 

**Authors**: Jui-Ming Yao, Hao-Yuan Chen, Zi-Xian Tang, Bing-Jia Tan, Sheng-Wei Peng, Bing-Cheng Xie, Shun-Feng Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.09408)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance on multiple-choice question answering (MCQA) benchmarks, yet they remain highly vulnerable to minor input perturbations. In this paper, we introduce and evaluate Token Constraint Decoding (TCD). This simple yet effective inference-time algorithm enforces alignment between token-level predictions to enhance robustness in noisy settings. Through extensive experiments on CommonsenseQA, MMLU, and MMLU-Pro, we show that TCD, especially when paired with prompt engineering (PE) fixes, significantly restores performance degraded by input noise, yielding up to +39\% absolute gains for weaker models like Gemma3 1B. Penalty sweep analyses further reveal that TCD implicitly regularizes overconfident outputs, with different models requiring distinct penalty schedules to maximize resilience. Our findings establish TCD as a practical, model-agnostic approach for improving reasoning stability under real-world imperfections and pave the way for more reliable deployment of LLMs in safety-critical or user-facing applications. 

---
# GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture 

**Authors**: GigaChat team, Mamedov Valentin, Evgenii Kosarev, Gregory Leleytner, Ilya Shchuckin, Valeriy Berezovskiy, Daniil Smirnov, Dmitry Kozlov, Sergei Averkiev, Lukyanenko Ivan, Aleksandr Proshunin, Ainur Israfilova, Ivan Baskov, Artem Chervyakov, Emil Shakirov, Mikhail Kolesov, Daria Khomich, Darya Latortseva, Sergei Porkhun, Yury Fedorov, Oleg Kutuzov, Polina Kudriavtseva, Sofiia Soldatova, Kolodin Egor, Stanislav Pyatkin, Dzmitry Menshykh, Grafov Sergei, Eldar Damirov, Karlov Vladimir, Ruslan Gaitukiev, Arkadiy Shatenov, Alena Fenogenova, Nikita Savushkin, Fedor Minkin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09440)  

**Abstract**: Generative large language models (LLMs) have become crucial for modern NLP research and applications across various languages. However, the development of foundational models specifically tailored to the Russian language has been limited, primarily due to the significant computational resources required. This paper introduces the GigaChat family of Russian LLMs, available in various sizes, including base models and instruction-tuned versions. We provide a detailed report on the model architecture, pre-training process, and experiments to guide design choices. In addition, we evaluate their performance on Russian and English benchmarks and compare GigaChat with multilingual analogs. The paper presents a system demonstration of the top-performing models accessible via an API, a Telegram bot, and a Web interface. Furthermore, we have released three open GigaChat models in open-source (this https URL), aiming to expand NLP research opportunities and support the development of industrial solutions for the Russian language. 

---
# Taming SQL Complexity: LLM-Based Equivalence Evaluation for Text-to-SQL 

**Authors**: Qingyun Zeng, Simin Ma, Arash Niknafs, Ashish Basran, Carol Szabo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09359)  

**Abstract**: The rise of Large Language Models (LLMs) has significantly advanced Text-to-SQL (NL2SQL) systems, yet evaluating the semantic equivalence of generated SQL remains a challenge, especially given ambiguous user queries and multiple valid SQL interpretations. This paper explores using LLMs to assess both semantic and a more practical "weak" semantic equivalence. We analyze common patterns of SQL equivalence and inequivalence, discuss challenges in LLM-based evaluation. 

---
# Bridging Online Behavior and Clinical Insight: A Longitudinal LLM-based Study of Suicidality on YouTube Reveals Novel Digital Markers 

**Authors**: Ilanit Sobol, Shir Lissak, Refael Tikochinski, Tal Nakash, Anat Brunstein Klomek, Eyal Fruchter, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2506.09495)  

**Abstract**: Suicide remains a leading cause of death in Western countries, underscoring the need for new research approaches. As social media becomes central to daily life, digital footprints offer valuable insight into suicidal behavior. Focusing on individuals who attempted suicide while uploading videos to their channels, we investigate: How do suicidal behaviors manifest on YouTube, and how do they differ from expert knowledge? We applied complementary approaches: computational bottom-up, hybrid, and expert-driven top-down, on a novel longitudinal dataset of 181 YouTube channels from individuals with life-threatening attempts, alongside 134 control channels. In the bottom-up approach, we applied LLM-based topic modeling to identify behavioral indicators. Of 166 topics, five were associated with suicide-attempt, with two also showing temporal attempt-related changes ($p<.01$) - Mental Health Struggles ($+0.08$)* and YouTube Engagement ($+0.1$)*. In the hybrid approach, a clinical expert reviewed LLM-derived topics and flagged 19 as suicide-related. However, none showed significant attempt-related temporal effects beyond those identified bottom-up. Notably, YouTube Engagement, a platform-specific indicator, was not flagged by the expert, underscoring the value of bottom-up discovery. In the top-down approach, psychological assessment of suicide attempt narratives revealed that the only significant difference between individuals who attempted before and those attempted during their upload period was the motivation to share this experience: the former aimed to Help Others ($\beta=-1.69$, $p<.01$), while the latter framed it as part of their Personal Recovery ($\beta=1.08$, $p<.01$). By integrating these approaches, we offer a nuanced understanding of suicidality, bridging digital behavior and clinical insights.
* Within-group changes in relation to the suicide attempt. 

---
# CoRT: Code-integrated Reasoning within Thinking 

**Authors**: Chengpeng Li, Zhengyang Tang, Ziniu Li, Mingfeng Xue, Keqin Bao, Tian Ding, Ruoyu Sun, Benyou Wang, Xiang Wang, Junyang Lin, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09820)  

**Abstract**: Large Reasoning Models (LRMs) like o1 and DeepSeek-R1 have shown remarkable progress in natural language reasoning with long chain-of-thought (CoT), yet they remain inefficient or inaccurate when handling complex mathematical operations. Addressing these limitations through computational tools (e.g., computation libraries and symbolic solvers) is promising, but it introduces a technical challenge: Code Interpreter (CI) brings external knowledge beyond the model's internal text representations, thus the direct combination is not efficient. This paper introduces CoRT, a post-training framework for teaching LRMs to leverage CI effectively and efficiently. As a first step, we address the data scarcity issue by synthesizing code-integrated reasoning data through Hint-Engineering, which strategically inserts different hints at appropriate positions to optimize LRM-CI interaction. We manually create 30 high-quality samples, upon which we post-train models ranging from 1.5B to 32B parameters, with supervised fine-tuning, rejection fine-tuning and reinforcement learning. Our experimental results demonstrate that Hint-Engineering models achieve 4\% and 8\% absolute improvements on DeepSeek-R1-Distill-Qwen-32B and DeepSeek-R1-Distill-Qwen-1.5B respectively, across five challenging mathematical reasoning datasets. Furthermore, Hint-Engineering models use about 30\% fewer tokens for the 32B model and 50\% fewer tokens for the 1.5B model compared with the natural language models. The models and code are available at this https URL. 

---
# Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms 

**Authors**: Zeguan Xiao, Yun Chen, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09457)  

**Abstract**: Direct Alignment Algorithms (DAAs), such as Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO), have emerged as efficient alternatives to Reinforcement Learning from Human Feedback (RLHF) algorithms for aligning large language models (LLMs) with human preferences. However, DAAs suffer from a fundamental limitation we identify as the "reward-generation gap" -- a misalignment between optimization objectives during training and actual generation performance during inference. In this paper, we find a contributor to the reward-generation gap is the mismatch between the inherent importance of prefix tokens during the LLM generation process and how this importance is reflected in the implicit reward functions of DAAs. To bridge the gap, we introduce a simple yet effective approach called Prefix-Oriented Equal-length Training (POET), which truncates both preferred and dispreferred responses to match the shorter one's length. Training with POET, where both responses in each sample are truncated to equal length, resulting in diverse truncated lengths across samples, the optimization of DAAs objective is implicitly constrained to converge across all positions, thus paying more attention to prefix tokens than the standard DAAs. We conduct experiments with DPO and SimPO, two representative DAAs, demonstrating that POET improves over their standard implementations, achieving up to 15.6 points in AlpacaEval 2 and overall improvements across downstream tasks. Our results highlight the importance of addressing the misalignment between reward optimization and generation performance in DAAs. 

---
# Comparing human and LLM politeness strategies in free production 

**Authors**: Haoran Zhao, Robert D.Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2506.09391)  

**Abstract**: Polite speech poses a fundamental alignment challenge for large language models (LLMs). Humans deploy a rich repertoire of linguistic strategies to balance informational and social goals -- from positive approaches that build rapport (compliments, expressions of interest) to negative strategies that minimize imposition (hedging, indirectness). We investigate whether LLMs employ a similarly context-sensitive repertoire by comparing human and LLM responses in both constrained and open-ended production tasks. We find that larger models ($\ge$70B parameters) successfully replicate key preferences from the computational pragmatics literature, and human evaluators surprisingly prefer LLM-generated responses in open-ended contexts. However, further linguistic analyses reveal that models disproportionately rely on negative politeness strategies even in positive contexts, potentially leading to misinterpretations. While modern LLMs demonstrate an impressive handle on politeness strategies, these subtle differences raise important questions about pragmatic alignment in AI systems. 

---
# OmniDRCA: Parallel Speech-Text Foundation Model via Dual-Resolution Speech Representations and Contrastive Alignment 

**Authors**: Chao-Hong Tan, Qian Chen, Wen Wang, Chong Deng, Qinglin Zhang, Luyao Cheng, Hai Yu, Xin Zhang, Xiang Lv, Tianyu Zhao, Chong Zhang, Yukun Ma, Yafeng Chen, Hui Wang, Jiaqing Liu, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.09349)  

**Abstract**: Recent studies on end-to-end speech generation with large language models (LLMs) have attracted significant community attention, with multiple works extending text-based LLMs to generate discrete speech tokens. Existing approaches primarily fall into two categories: (1) Methods that generate discrete speech tokens independently without incorporating them into the LLM's autoregressive process, resulting in text generation being unaware of concurrent speech synthesis. (2) Models that generate interleaved or parallel speech-text tokens through joint autoregressive modeling, enabling mutual modality awareness during generation. This paper presents OmniDRCA, a parallel speech-text foundation model based on joint autoregressive modeling, featuring dual-resolution speech representations and contrastive cross-modal alignment. Our approach processes speech and text representations in parallel while enhancing audio comprehension through contrastive alignment. Experimental results on Spoken Question Answering benchmarks demonstrate that OmniDRCA establishes new state-of-the-art (SOTA) performance among parallel joint speech-text modeling based foundation models, and achieves competitive performance compared to interleaved models. Additionally, we explore the potential of extending the framework to full-duplex conversational scenarios. 

---
# DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts 

**Authors**: Yuchen Feng, Bowen Shen, Naibin Gu, Jiaxuan Zhao, Peng Fu, Zheng Lin, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09351)  

**Abstract**: Large language models (LLMs) with the Mixture-of-Experts (MoE) architecture achieve high cost-efficiency by selectively activating a subset of the parameters. Despite the inference efficiency of MoE LLMs, the training of extensive experts from scratch incurs substantial overhead, whereas reconstructing a dense LLM into an MoE LLM significantly reduces the training budget. However, existing reconstruction methods often overlook the diversity among experts, leading to potential redundancy. In this paper, we come up with the observation that a specific LLM exhibits notable diversity after being pruned on different calibration datasets, based on which we present a Diversity-Enhanced reconstruction method named DIVE. The recipe of DIVE includes domain affinity mining, pruning-based expert reconstruction, and efficient retraining. Specifically, the reconstruction includes pruning and reassembly of the feed-forward network (FFN) module. After reconstruction, we efficiently retrain the model on routers, experts and normalization modules. We implement DIVE on Llama-style LLMs with open-source training corpora. Experiments show that DIVE achieves training efficiency with minimal accuracy trade-offs, outperforming existing pruning and MoE reconstruction methods with the same number of activated parameters. 

---
# Towards Efficient and Effective Alignment of Large Language Models 

**Authors**: Yuxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09329)  

**Abstract**: Large language models (LLMs) exhibit remarkable capabilities across diverse tasks, yet aligning them efficiently and effectively with human expectations remains a critical challenge. This thesis advances LLM alignment by introducing novel methodologies in data collection, training, and evaluation. We first address alignment data collection. Existing approaches rely heavily on manually curated datasets or proprietary models. To overcome these limitations, we propose Lion, an adversarial distillation framework that iteratively refines training data by identifying and generating challenging instructions, enabling state-of-the-art zero-shot reasoning. Additionally, we introduce Web Reconstruction (WebR), a fully automated framework that synthesizes instruction-tuning data directly from raw web documents, significantly improving data diversity and scalability over existing synthetic data methods. Next, we enhance alignment training through novel optimization techniques. We develop Learning to Edit (LTE), a framework that enables LLMs to efficiently integrate new knowledge while preserving existing information. LTE leverages meta-learning to improve both real-time and batch knowledge updates. Furthermore, we introduce Bridging and Modeling Correlations (BMC), a refinement of Direct Preference Optimization (DPO) that explicitly captures token-level correlations in preference data, leading to superior alignment across QA and mathematical reasoning tasks. Finally, we tackle the challenge of evaluating alignment. Existing benchmarks emphasize response quality but overlook adherence to specific constraints. To bridge this gap, we introduce FollowBench, a multi-level, fine-grained benchmark assessing LLMs' ability to follow complex constraints across diverse instruction types. Our results expose key weaknesses in current models' constraint adherence, offering insights for future improvements. 

---
# RePO: Replay-Enhanced Policy Optimization 

**Authors**: Siheng Li, Zhanhui Zhou, Wai Lam, Chao Yang, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09340)  

**Abstract**: Reinforcement learning (RL) is vital for optimizing large language models (LLMs). Recent Group Relative Policy Optimization (GRPO) estimates advantages using multiple on-policy outputs per prompt, leading to high computational costs and low data efficiency. To address this, we introduce Replay-Enhanced Policy Optimization (RePO), which leverages diverse replay strategies to retrieve off-policy samples from a replay buffer, allowing policy optimization based on a broader and more diverse set of samples for each prompt. Experiments on five LLMs across seven mathematical reasoning benchmarks demonstrate that RePO achieves absolute average performance gains of $18.4$ and $4.1$ points for Qwen2.5-Math-1.5B and Qwen3-1.7B, respectively, compared to GRPO. Further analysis indicates that RePO increases computational cost by $15\%$ while raising the number of effective optimization steps by $48\%$ for Qwen3-1.7B, with both on-policy and off-policy sample numbers set to $8$. The repository can be accessed at this https URL. 

---
# Did I Faithfully Say What I Thought? Bridging the Gap Between Neural Activity and Self-Explanations in Large Language Models 

**Authors**: Milan Bhan, Jean-Noel Vittaut, Nicolas Chesneau, Sarath Chandar, Marie-Jeanne Lesot  

**Link**: [PDF](https://arxiv.org/pdf/2506.09277)  

**Abstract**: Large Language Models (LLM) have demonstrated the capability of generating free text self Natural Language Explanation (self-NLE) to justify their answers. Despite their logical appearance, self-NLE do not necessarily reflect the LLM actual decision-making process, making such explanations unfaithful. While existing methods for measuring self-NLE faithfulness mostly rely on behavioral tests or computational block identification, none of them examines the neural activity underlying the model's reasoning. This work introduces a novel flexible framework for quantitatively measuring the faithfulness of LLM-generated self-NLE by directly comparing the latter with interpretations of the model's internal hidden states. The proposed framework is versatile and provides deep insights into self-NLE faithfulness by establishing a direct connection between self-NLE and model reasoning. This approach advances the understanding of self-NLE faithfulness and provides building blocks for generating more faithful self-NLE. 

---
# $(RSA)^2$: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding 

**Authors**: Cesare Spinoso-Di Piano, David Austin, Pablo Piantanida, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2506.09301)  

**Abstract**: Figurative language (e.g., irony, hyperbole, understatement) is ubiquitous in human communication, resulting in utterances where the literal and the intended meanings do not match. The Rational Speech Act (RSA) framework, which explicitly models speaker intentions, is the most widespread theory of probabilistic pragmatics, but existing implementations are either unable to account for figurative expressions or require modeling the implicit motivations for using figurative language (e.g., to express joy or annoyance) in a setting-specific way. In this paper, we introduce the Rhetorical-Strategy-Aware RSA $(RSA)^2$ framework which models figurative language use by considering a speaker's employed rhetorical strategy. We show that $(RSA)^2$ enables human-compatible interpretations of non-literal utterances without modeling a speaker's motivations for being non-literal. Combined with LLMs, it achieves state-of-the-art performance on the ironic split of PragMega+, a new irony interpretation dataset introduced in this study. 

---
# Flipping Against All Odds: Reducing LLM Coin Flip Bias via Verbalized Rejection Sampling 

**Authors**: Tim Z. Xiao, Johannes Zenn, Zhen Liu, Weiyang Liu, Robert Bamler, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2506.09998)  

**Abstract**: Large language models (LLMs) can often accurately describe probability distributions using natural language, yet they still struggle to generate faithful samples from them. This mismatch limits their use in tasks requiring reliable stochasticity, such as Monte Carlo methods, agent-based simulations, and randomized decision-making. We investigate this gap between knowledge and sampling in the context of Bernoulli distributions. We introduce Verbalized Rejection Sampling (VRS), a natural-language adaptation of classical rejection sampling that prompts the LLM to reason about and accept or reject proposed samples. Despite relying on the same Bernoulli mechanism internally, VRS substantially reduces sampling bias across models. We provide theoretical analysis showing that, under mild assumptions, VRS improves over direct sampling, with gains attributable to both the algorithm and prompt design. More broadly, our results show how classical probabilistic tools can be verbalized and embedded into LLM workflows to improve reliability, without requiring access to model internals or heavy prompt engineering. 

---
# LLM-as-a-qualitative-judge: automating error analysis in natural language generation 

**Authors**: Nadezhda Chirkova, Tunde Oluwaseyi Ajayi, Seth Aycock, Zain Muhammad Mujahid, Vladana Perlić, Ekaterina Borisova, Markarit Vartampetian  

**Link**: [PDF](https://arxiv.org/pdf/2506.09147)  

**Abstract**: Prompting large language models (LLMs) to evaluate generated text, known as LLM-as-a-judge, has become a standard evaluation approach in natural language generation (NLG), but is primarily used as a quantitative tool, i.e. with numerical scores as main outputs. In this work, we propose LLM-as-a-qualitative-judge, an LLM-based evaluation approach with the main output being a structured report of common issue types in the NLG system outputs. Our approach is targeted at providing developers with meaningful insights on what improvements can be done to a given NLG system and consists of two main steps, namely open-ended per-instance issue analysis and clustering of the discovered issues using an intuitive cumulative algorithm. We also introduce a strategy for evaluating the proposed approach, coupled with ~300 annotations of issues in instances from 12 NLG datasets. Our results show that LLM-as-a-qualitative-judge correctly recognizes instance-specific issues in 2/3 cases and is capable of producing error type reports resembling the reports composed by human annotators. Our code and data are publicly available at this https URL. 

---
# Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform 

**Authors**: Jay Roberts, Kyle Mylonakis, Sidhartha Roy, Kaan Kale  

**Link**: [PDF](https://arxiv.org/pdf/2506.09452)  

**Abstract**: The high cost of ownership of AI compute infrastructure and challenges of robust serving of large language models (LLMs) has led to a surge in managed Model-as-a-service deployments. Even when enterprises choose on-premises deployments, the compute infrastructure is typically shared across many teams in order to maximize the return on investment. In both scenarios the deployed models operate only on plaintext data, and so enterprise data owners must allow their data to appear in plaintext on a shared or multi-tenant compute infrastructure. This results in data owners with private or sensitive data being hesitant or restricted in what data they use with these types of deployments. In this work we introduce the Stained Glass Transform, a learned, stochastic, and sequence dependent transformation of the word embeddings of an LLM which information theoretically provides privacy to the input of the LLM while preserving the utility of model. We theoretically connect a particular class of Stained Glass Transforms to the theory of mutual information of Gaussian Mixture Models. We then calculate a-postiori privacy estimates, based on mutual information, and verify the privacy and utility of instances of transformed embeddings through token level metrics of privacy and standard LLM performance benchmarks. 

---
# Alzheimer's Dementia Detection Using Perplexity from Paired Large Language Models 

**Authors**: Yao Xiao, Heidi Christensen, Stefan Goetze  

**Link**: [PDF](https://arxiv.org/pdf/2506.09315)  

**Abstract**: Alzheimer's dementia (AD) is a neurodegenerative disorder with cognitive decline that commonly impacts language ability. This work extends the paired perplexity approach to detecting AD by using a recent large language model (LLM), the instruction-following version of Mistral-7B. We improve accuracy by an average of 3.33% over the best current paired perplexity method and by 6.35% over the top-ranked method from the ADReSS 2020 challenge benchmark. Our further analysis demonstrates that the proposed approach can effectively detect AD with a clear and interpretable decision boundary in contrast to other methods that suffer from opaque decision-making processes. Finally, by prompting the fine-tuned LLMs and comparing the model-generated responses to human responses, we illustrate that the LLMs have learned the special language patterns of AD speakers, which opens up possibilities for novel methods of model interpretation and data augmentation. 

---
# Multi-Agent Language Models: Advancing Cooperation, Coordination, and Adaptation 

**Authors**: Arjun Vaithilingam Sudhakar  

**Link**: [PDF](https://arxiv.org/pdf/2506.09331)  

**Abstract**: Modern Large Language Models (LLMs) exhibit impressive zero-shot and few-shot generalization capabilities across complex natural language tasks, enabling their widespread use as virtual assistants for diverse applications such as translation and summarization. Despite being trained solely on large corpora of text without explicit supervision on author intent, LLMs appear to infer the underlying meaning of textual interactions. This raises a fundamental question: can LLMs model and reason about the intentions of others, i.e., do they possess a form of theory of mind? Understanding other's intentions is crucial for effective collaboration, which underpins human societal success and is essential for cooperative interactions among multiple agents, including humans and autonomous systems. In this work, we investigate the theory of mind in LLMs through the lens of cooperative multi-agent reinforcement learning (MARL), where agents learn to collaborate via repeated interactions, mirroring human social reasoning. Our approach aims to enhance artificial agent's ability to adapt and cooperate with both artificial and human partners. By leveraging LLM-based agents capable of natural language interaction, we move towards creating hybrid human-AI systems that can foster seamless collaboration, with broad implications for the future of human-artificial interaction. 

---
# Intent Factored Generation: Unleashing the Diversity in Your Language Model 

**Authors**: Eltayeb Ahmed, Uljad Berdica, Martha Elliott, Danijela Horak, Jakob N. Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2506.09659)  

**Abstract**: Obtaining multiple meaningfully diverse, high quality samples from Large Language Models for a fixed prompt remains an open challenge. Current methods for increasing diversity often only operate at the token-level, paraphrasing the same response. This is problematic because it leads to poor exploration on reasoning problems and to unengaging, repetitive conversational agents. To address this we propose Intent Factored Generation (IFG), factorising the sampling process into two stages. First, we sample a semantically dense intent, e.g., a summary or keywords. Second, we sample the final response conditioning on both the original prompt and the intent from the first stage. This allows us to use a higher temperature during the intent step to promote conceptual diversity, and a lower temperature during the final generation to ensure the outputs are coherent and self-consistent. Additionally, we find that prompting the model to explicitly state its intent for each step of the chain-of-thought before generating the step is beneficial for reasoning tasks. We demonstrate our method's effectiveness across a diverse set of tasks. We show this method improves both pass@k and Reinforcement Learning from Verifier Feedback on maths and code tasks. For instruction-tuning, we combine IFG with Direct Preference Optimisation to increase conversational diversity without sacrificing reward. Finally, we achieve higher diversity while maintaining the quality of generations on a general language modelling task, using a new dataset of reader comments and news articles that we collect and open-source. In summary, we present a simple method of increasing the sample diversity of LLMs while maintaining performance. This method can be implemented by changing the prompt and varying the temperature during generation, making it easy to integrate into many algorithms for gains across various applications. 

---
# PHRASED: Phrase Dictionary Biasing for Speech Translation 

**Authors**: Peidong Wang, Jian Xue, Rui Zhao, Junkun Chen, Aswin Shanmugam Subramanian, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09175)  

**Abstract**: Phrases are essential to understand the core concepts in conversations. However, due to their rare occurrence in training data, correct translation of phrases is challenging in speech translation tasks. In this paper, we propose a phrase dictionary biasing method to leverage pairs of phrases mapping from the source language to the target language. We apply the phrase dictionary biasing method to two types of widely adopted models, a transducer-based streaming speech translation model and a multimodal large language model. Experimental results show that the phrase dictionary biasing method outperforms phrase list biasing by 21% relatively for the streaming speech translation model. In addition, phrase dictionary biasing enables multimodal large language models to use external phrase information, achieving 85% relative improvement in phrase recall. 

---
# Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search 

**Authors**: Samuel Holt, Max Ruiz Luyten, Thomas Pouplin, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.09171)  

**Abstract**: Large Language Models (LLMs) are increasingly capable but often require significant guidance or extensive interaction history to perform effectively in complex, interactive environments. Existing methods may struggle with adapting to new information or efficiently utilizing past experiences for multi-step reasoning without fine-tuning. We introduce a novel LLM agent framework that enhances planning capabilities through in-context learning, facilitated by atomic fact augmentation and a recursive lookahead search. Our agent learns to extract task-critical ``atomic facts'' from its interaction trajectories. These facts dynamically augment the prompts provided to LLM-based components responsible for action proposal, latent world model simulation, and state-value estimation. Planning is performed via a depth-limited lookahead search, where the LLM simulates potential trajectories and evaluates their outcomes, guided by the accumulated facts and interaction history. This approach allows the agent to improve its understanding and decision-making online, leveraging its experience to refine its behavior without weight updates. We provide a theoretical motivation linking performance to the quality of fact-based abstraction and LLM simulation accuracy. Empirically, our agent demonstrates improved performance and adaptability on challenging interactive tasks, achieving more optimal behavior as it accumulates experience, showcased in tasks such as TextFrozenLake and ALFWorld. 

---
# An Interpretable N-gram Perplexity Threat Model for Large Language Model Jailbreaks 

**Authors**: Valentyn Boreiko, Alexander Panfilov, Vaclav Voracek, Matthias Hein, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2410.16222)  

**Abstract**: A plethora of jailbreaking attacks have been proposed to obtain harmful responses from safety-tuned LLMs. These methods largely succeed in coercing the target output in their original settings, but their attacks vary substantially in fluency and computational effort. In this work, we propose a unified threat model for the principled comparison of these methods. Our threat model checks if a given jailbreak is likely to occur in the distribution of text. For this, we build an N-gram language model on 1T tokens, which, unlike model-based perplexity, allows for an LLM-agnostic, nonparametric, and inherently interpretable evaluation. We adapt popular attacks to this threat model, and, for the first time, benchmark these attacks on equal footing with it. After an extensive comparison, we find attack success rates against safety-tuned modern models to be lower than previously presented and that attacks based on discrete optimization significantly outperform recent LLM-based attacks. Being inherently interpretable, our threat model allows for a comprehensive analysis and comparison of jailbreak attacks. We find that effective attacks exploit and abuse infrequent bigrams, either selecting the ones absent from real-world text or rare ones, e.g., specific to Reddit or code datasets. 

---
# Too Big to Think: Capacity, Memorization, and Generalization in Pre-Trained Transformers 

**Authors**: Joshua Barron, Devin White  

**Link**: [PDF](https://arxiv.org/pdf/2506.09099)  

**Abstract**: The relationship between memorization and generalization in large language models (LLMs) remains an open area of research, with growing evidence that the two are deeply intertwined. In this work, we investigate this relationship by pre-training a series of capacity-limited Transformer models from scratch on two synthetic character-level tasks designed to separately probe generalization (via arithmetic extrapolation) and memorization (via factual recall). We observe a consistent trade-off: small models extrapolate to unseen arithmetic cases but fail to memorize facts, while larger models memorize but fail to extrapolate. An intermediate-capacity model exhibits a similar shift toward memorization. When trained on both tasks jointly, no model (regardless of size) succeeds at extrapolation. These findings suggest that pre-training may intrinsically favor one learning mode over the other. By isolating these dynamics in a controlled setting, our study offers insight into how model capacity shapes learning behavior and offers broader implications for the design and deployment of small language models. 

---
# Application-Driven Value Alignment in Agentic AI Systems: Survey and Perspectives 

**Authors**: Wei Zeng, Hengshu Zhu, Chuan Qin, Han Wu, Yihang Cheng, Sirui Zhang, Xiaowei Jin, Yinuo Shen, Zhenxing Wang, Feimin Zhong, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.09656)  

**Abstract**: The ongoing evolution of AI paradigms has propelled AI research into the Agentic AI stage. Consequently, the focus of research has shifted from single agents and simple applications towards multi-agent autonomous decision-making and task collaboration in complex environments. As Large Language Models (LLMs) advance, their applications become more diverse and complex, leading to increasingly situational and systemic risks. This has brought significant attention to value alignment for AI agents, which aims to ensure that an agent's goals, preferences, and behaviors align with human values and societal norms. This paper reviews value alignment in agent systems within specific application scenarios. It integrates the advancements in AI driven by large models with the demands of social governance. Our review covers value principles, agent system application scenarios, and agent value alignment evaluation. Specifically, value principles are organized hierarchically from a top-down perspective, encompassing macro, meso, and micro levels. Agent system application scenarios are categorized and reviewed from a general-to-specific viewpoint. Agent value alignment evaluation systematically examines datasets for value alignment assessment and relevant value alignment methods. Additionally, we delve into value coordination among multiple agents within agent systems. Finally, we propose several potential research directions in this field. 

---
# DipLLM: Fine-Tuning LLM for Strategic Decision-making in Diplomacy 

**Authors**: Kaixuan Xu, Jiajun Chai, Sicheng Li, Yuqian Fu, Yuanheng Zhu, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09655)  

**Abstract**: Diplomacy is a complex multiplayer game that requires both cooperation and competition, posing significant challenges for AI systems. Traditional methods rely on equilibrium search to generate extensive game data for training, which demands substantial computational resources. Large Language Models (LLMs) offer a promising alternative, leveraging pre-trained knowledge to achieve strong performance with relatively small-scale fine-tuning. However, applying LLMs to Diplomacy remains challenging due to the exponential growth of possible action combinations and the intricate strategic interactions among players. To address this challenge, we propose DipLLM, a fine-tuned LLM-based agent that learns equilibrium policies for Diplomacy. DipLLM employs an autoregressive factorization framework to simplify the complex task of multi-unit action assignment into a sequence of unit-level decisions. By defining an equilibrium policy within this framework as the learning objective, we fine-tune the model using only 1.5% of the data required by the state-of-the-art Cicero model, surpassing its performance. Our results demonstrate the potential of fine-tuned LLMs for tackling complex strategic decision-making in multiplayer games. 

---
# Beyond Nash Equilibrium: Bounded Rationality of LLMs and humans in Strategic Decision-making 

**Authors**: Kehan Zheng, Jinfeng Zhou, Hongning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09390)  

**Abstract**: Large language models are increasingly used in strategic decision-making settings, yet evidence shows that, like humans, they often deviate from full rationality. In this study, we compare LLMs and humans using experimental paradigms directly adapted from behavioral game-theory research. We focus on two well-studied strategic games, Rock-Paper-Scissors and the Prisoner's Dilemma, which are well known for revealing systematic departures from rational play in human subjects. By placing LLMs in identical experimental conditions, we evaluate whether their behaviors exhibit the bounded rationality characteristic of humans. Our findings show that LLMs reproduce familiar human heuristics, such as outcome-based strategy switching and increased cooperation when future interaction is possible, but they apply these rules more rigidly and demonstrate weaker sensitivity to the dynamic changes in the game environment. Model-level analyses reveal distinctive architectural signatures in strategic behavior, and even reasoning models sometimes struggle to find effective strategies in adaptive situations. These results indicate that current LLMs capture only a partial form of human-like bounded rationality and highlight the need for training methods that encourage flexible opponent modeling and stronger context awareness. 

---
# LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge 

**Authors**: Sahar Abdelnabi, Aideen Fay, Ahmed Salem, Egor Zverev, Kai-Chieh Liao, Chi-Huang Liu, Chun-Chih Kuo, Jannis Weigend, Danyael Manlangit, Alex Apostolov, Haris Umair, João Donato, Masayuki Kawakita, Athar Mahboob, Tran Huu Bach, Tsun-Han Chiang, Myeongjin Cho, Hajin Choi, Byeonghyeon Kim, Hyeonjin Lee, Benjamin Pannell, Conor McCauley, Mark Russinovich, Andrew Paverd, Giovanni Cherubin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09956)  

**Abstract**: Indirect Prompt Injection attacks exploit the inherent limitation of Large Language Models (LLMs) to distinguish between instructions and data in their inputs. Despite numerous defense proposals, the systematic evaluation against adaptive adversaries remains limited, even when successful attacks can have wide security and privacy implications, and many real-world LLM-based applications remain vulnerable. We present the results of LLMail-Inject, a public challenge simulating a realistic scenario in which participants adaptively attempted to inject malicious instructions into emails in order to trigger unauthorized tool calls in an LLM-based email assistant. The challenge spanned multiple defense strategies, LLM architectures, and retrieval configurations, resulting in a dataset of 208,095 unique attack submissions from 839 participants. We release the challenge code, the full dataset of submissions, and our analysis demonstrating how this data can provide new insights into the instruction-data separation problem. We hope this will serve as a foundation for future research towards practical structural solutions to prompt injection. 

---
# Superstudent intelligence in thermodynamics 

**Authors**: Rebecca Loubet, Pascal Zittlau, Marco Hoffmann, Luisa Vollmer, Sophie Fellenz, Heike Leitte, Fabian Jirasek, Johannes Lenhard, Hans Hasse  

**Link**: [PDF](https://arxiv.org/pdf/2506.09822)  

**Abstract**: In this short note, we report and analyze a striking event: OpenAI's large language model o3 has outwitted all students in a university exam on thermodynamics. The thermodynamics exam is a difficult hurdle for most students, where they must show that they have mastered the fundamentals of this important topic. Consequently, the failure rates are very high, A-grades are rare - and they are considered proof of the students' exceptional intellectual abilities. This is because pattern learning does not help in the exam. The problems can only be solved by knowledgeably and creatively combining principles of thermodynamics. We have given our latest thermodynamics exam not only to the students but also to OpenAI's most powerful reasoning model, o3, and have assessed the answers of o3 exactly the same way as those of the students. In zero-shot mode, the model o3 solved all problems correctly, better than all students who took the exam; its overall score was in the range of the best scores we have seen in more than 10,000 similar exams since 1985. This is a turning point: machines now excel in complex tasks, usually taken as proof of human intellectual capabilities. We discuss the consequences this has for the work of engineers and the education of future engineers. 

---
# The Emergence of Abstract Thought in Large Language Models Beyond Any Language 

**Authors**: Yuxin Chen, Yiran Zhao, Yang Zhang, An Zhang, Kenji Kawaguchi, Shafiq Joty, Junnan Li, Tat-Seng Chua, Michael Qizhe Shieh, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09890)  

**Abstract**: As large language models (LLMs) continue to advance, their capacity to function effectively across a diverse range of languages has shown marked improvement. Preliminary studies observe that the hidden activations of LLMs often resemble English, even when responding to non-English prompts. This has led to the widespread assumption that LLMs may "think" in English. However, more recent results showing strong multilingual performance, even surpassing English performance on specific tasks in other languages, challenge this view. In this work, we find that LLMs progressively develop a core language-agnostic parameter space-a remarkably small subset of parameters whose deactivation results in significant performance degradation across all languages. This compact yet critical set of parameters underlies the model's ability to generalize beyond individual languages, supporting the emergence of abstract thought that is not tied to any specific linguistic system. Specifically, we identify language-related neurons-those are consistently activated during the processing of particular languages, and categorize them as either shared (active across multiple languages) or exclusive (specific to one). As LLMs undergo continued development over time, we observe a marked increase in both the proportion and functional importance of shared neurons, while exclusive neurons progressively diminish in influence. These shared neurons constitute the backbone of the core language-agnostic parameter space, supporting the emergence of abstract thought. Motivated by these insights, we propose neuron-specific training strategies tailored to LLMs' language-agnostic levels at different development stages. Experiments across diverse LLM families support our approach. 

---
# Intelligent Design 4.0: Paradigm Evolution Toward the Agentic AI Era 

**Authors**: Shuo Jiang, Min Xie, Frank Youhua Chen, Jian Ma, Jianxi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09755)  

**Abstract**: Research and practice in Intelligent Design (ID) have significantly enhanced engineering innovation, efficiency, quality, and productivity over recent decades, fundamentally reshaping how engineering designers think, behave, and interact with design processes. The recent emergence of Foundation Models (FMs), particularly Large Language Models (LLMs), has demonstrated general knowledge-based reasoning capabilities, and open new paths and avenues for further transformation in engineering design. In this context, this paper introduces Intelligent Design 4.0 (ID 4.0) as an emerging paradigm empowered by agentic AI systems. We review the historical evolution of ID across four distinct stages: rule-based expert systems, task-specific machine learning models, large-scale foundation AI models, and the recent emerging paradigm of multi-agent collaboration. We propose a conceptual framework for ID 4.0 and discuss its potential to support end-to-end automation of engineering design processes through coordinated, autonomous multi-agent-based systems. Furthermore, we discuss future perspectives to enhance and fully realize ID 4.0's potential, including more complex design scenarios, more practical design implementations, novel agent coordination mechanisms, and autonomous design goal-setting with better human value alignment. In sum, these insights lay a foundation for advancing Intelligent Design toward greater adaptivity, autonomy, and effectiveness in addressing increasingly complex design challenges. 

---
# Large Language Models for Design Structure Matrix Optimization 

**Authors**: Shuo Jiang, Min Xie, Jianxi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09749)  

**Abstract**: In complex engineering systems, the interdependencies among components or development activities are often modeled and analyzed using Design Structure Matrix (DSM). Reorganizing elements within a DSM to minimize feedback loops and enhance modularity or process efficiency constitutes a challenging combinatorial optimization (CO) problem in engineering design and operations. As problem sizes increase and dependency networks become more intricate, traditional optimization methods that solely use mathematical heuristics often fail to capture the contextual nuances and struggle to deliver effective solutions. In this study, we explore the potential of Large Language Models (LLMs) for helping solve such CO problems by leveraging their capabilities for advanced reasoning and contextual understanding. We propose a novel LLM-based framework that integrates network topology with contextual domain knowledge for iterative optimization of DSM element sequencing - a common CO problem. Experiments on various DSM cases show that our method consistently achieves faster convergence and superior solution quality compared to both stochastic and deterministic baselines. Notably, we find that incorporating contextual domain knowledge significantly enhances optimization performance regardless of the chosen LLM backbone. These findings highlight the potential of LLMs to solve complex engineering CO problems by combining semantic and mathematical reasoning. This approach paves the way towards a new paradigm in LLM-based engineering design optimization. 

---
# Feature Engineering for Agents: An Adaptive Cognitive Architecture for Interpretable ML Monitoring 

**Authors**: Gusseppe Bravo-Rocca, Peini Liu, Jordi Guitart, Rodrigo M Carrillo-Larco, Ajay Dholakia, David Ellison  

**Link**: [PDF](https://arxiv.org/pdf/2506.09742)  

**Abstract**: Monitoring Machine Learning (ML) models in production environments is crucial, yet traditional approaches often yield verbose, low-interpretability outputs that hinder effective decision-making. We propose a cognitive architecture for ML monitoring that applies feature engineering principles to agents based on Large Language Models (LLMs), significantly enhancing the interpretability of monitoring outputs. Central to our approach is a Decision Procedure module that simulates feature engineering through three key steps: Refactor, Break Down, and Compile. The Refactor step improves data representation to better capture feature semantics, allowing the LLM to focus on salient aspects of the monitoring data while reducing noise and irrelevant information. Break Down decomposes complex information for detailed analysis, and Compile integrates sub-insights into clear, interpretable outputs. This process leads to a more deterministic planning approach, reducing dependence on LLM-generated planning, which can sometimes be inconsistent and overly general. The combination of feature engineering-driven planning and selective LLM utilization results in a robust decision support system, capable of providing highly interpretable and actionable insights. Experiments using multiple LLMs demonstrate the efficacy of our approach, achieving significantly higher accuracy compared to various baselines across several domains. 

---
# TRIDENT: Temporally Restricted Inference via DFA-Enhanced Neural Traversal 

**Authors**: Vincenzo Collura, Karim Tit, Laura Bussi, Eleonora Giunchiglia, Maxime Cordy  

**Link**: [PDF](https://arxiv.org/pdf/2506.09701)  

**Abstract**: Large Language Models (LLMs) and other neural architectures have achieved impressive results across a variety of generative and classification tasks. However, they remain fundamentally ill-equipped to ensure that their outputs satisfy temporal constraints, such as those expressible in Linear Temporal Logic over finite traces (LTLf). In this paper, we introduce TRIDENT: a general and model-agnostic inference-time algorithm that guarantees compliance with such constraints without requiring any retraining. TRIDENT compiles LTLf formulas into a Deterministic Finite Automaton (DFA), which is used to guide a constrained variant of beam search. At each decoding step, transitions that would lead to constraint violations are masked, while remaining paths are dynamically re-ranked based on both the model's probabilities and the DFA's acceptance structure. We formally prove that the resulting sequences are guaranteed to satisfy the given LTLf constraints, and we empirically demonstrate that TRIDENT also improves output quality. We validate our approach on two distinct tasks: temporally constrained image-stream classification and controlled text generation. In both settings, TRIDENT achieves perfect constraint satisfaction, while comparison with the state of the art shows improved efficiency and high standard quality metrics. 

---
# AD^2-Bench: A Hierarchical CoT Benchmark for MLLM in Autonomous Driving under Adverse Conditions 

**Authors**: Zhaoyang Wei, Chenhui Qiang, Bowen Jiang, Xumeng Han, Xuehui Yu, Zhenjun Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.09557)  

**Abstract**: Chain-of-Thought (CoT) reasoning has emerged as a powerful approach to enhance the structured, multi-step decision-making capabilities of Multi-Modal Large Models (MLLMs), is particularly crucial for autonomous driving with adverse weather conditions and complex traffic environments. However, existing benchmarks have largely overlooked the need for rigorous evaluation of CoT processes in these specific and challenging scenarios. To address this critical gap, we introduce AD^2-Bench, the first Chain-of-Thought benchmark specifically designed for autonomous driving with adverse weather and complex scenes. AD^2-Bench is meticulously constructed to fulfill three key criteria: comprehensive data coverage across diverse adverse environments, fine-grained annotations that support multi-step reasoning, and a dedicated evaluation framework tailored for assessing CoT performance. The core contribution of AD^2-Bench is its extensive collection of over 5.4k high-quality, manually annotated CoT instances. Each intermediate reasoning step in these annotations is treated as an atomic unit with explicit ground truth, enabling unprecedented fine-grained analysis of MLLMs' inferential processes under text-level, point-level, and region-level visual prompts. Our comprehensive evaluation of state-of-the-art MLLMs on AD^2-Bench reveals accuracy below 60%, highlighting the benchmark's difficulty and the need to advance robust, interpretable end-to-end autonomous driving systems. AD^2-Bench thus provides a standardized evaluation platform, driving research forward by improving MLLMs' reasoning in autonomous driving, making it an invaluable resource. 

---
# OctoNav: Towards Generalist Embodied Navigation 

**Authors**: Chen Gao, Liankai Jin, Xingyu Peng, Jiazhao Zhang, Yue Deng, Annan Li, He Wang, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09839)  

**Abstract**: Embodied navigation stands as a foundation pillar within the broader pursuit of embodied AI. However, previous navigation research is divided into different tasks/capabilities, e.g., ObjNav, ImgNav and VLN, where they differ in task objectives and modalities, making datasets and methods are designed individually. In this work, we take steps toward generalist navigation agents, which can follow free-form instructions that include arbitrary compounds of multi-modal and multi-capability. To achieve this, we propose a large-scale benchmark and corresponding method, termed OctoNav-Bench and OctoNav-R1. Specifically, OctoNav-Bench features continuous environments and is constructed via a designed annotation pipeline. We thoroughly craft instruction-trajectory pairs, where instructions are diverse in free-form with arbitrary modality and capability. Also, we construct a Think-Before-Action (TBA-CoT) dataset within OctoNav-Bench to provide the thinking process behind actions. For OctoNav-R1, we build it upon MLLMs and adapt it to a VLA-type model, which can produce low-level actions solely based on 2D visual observations. Moreover, we design a Hybrid Training Paradigm (HTP) that consists of three stages, i.e., Action-/TBA-SFT, Nav-GPRO, and Online RL stages. Each stage contains specifically designed learning policies and rewards. Importantly, for TBA-SFT and Nav-GRPO designs, we are inspired by the OpenAI-o1 and DeepSeek-R1, which show impressive reasoning ability via thinking-before-answer. Thus, we aim to investigate how to achieve thinking-before-action in the embodied navigation field, to improve model's reasoning ability toward generalists. Specifically, we propose TBA-SFT to utilize the TBA-CoT dataset to fine-tune the model as a cold-start phrase and then leverage Nav-GPRO to improve its thinking ability. Finally, OctoNav-R1 shows superior performance compared with previous methods. 

---
# SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving 

**Authors**: Xiangchen Li, Dimitrios Spatharakis, Saeid Ghafouri, Jiakun Fan, Dimitrios Nikolopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.09397)  

**Abstract**: Regardless the advancements in device capabilities, efficient inferencing advanced large language models (LLMs) at the edge remains challenging due to limited device memory and power constraints. Existing strategies, such as aggressive quantization, pruning, or remote inference, trade accuracy for efficiency or lead to substantial cost burdens. This position paper introduces a new approach that leverages speculative decoding, previously viewed primarily as a decoding acceleration technique for autoregressive generation of LLMs, as a promising approach specifically adapted for edge computing by orchestrating computation across heterogeneous devices. We propose SLED, a method that allows lightweight edge devices to draft multiple candidate tokens locally using diverse draft models, while a single, shared edge server efficiently batches and verifies the tokens utilizing a more precise target model. This approach supports device heterogeneity and reduces server-side memory footprint by avoiding the need to deploy multiple target models. Our initial experiments with Jetson Orin Nano, Raspberry Pi 5, and an RTX 6000 edge server indicate substantial benefits: significantly reduced latency, improved energy efficiency, and increased concurrent inference sessions, all without sacrificing model accuracy. 

---
# Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation 

**Authors**: Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, Lu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09350)  

**Abstract**: Existing large-scale video generation models are computationally intensive, preventing adoption in real-time and interactive applications. In this work, we propose autoregressive adversarial post-training (AAPT) to transform a pre-trained latent video diffusion model into a real-time, interactive video generator. Our model autoregressively generates a latent frame at a time using a single neural function evaluation (1NFE). The model can stream the result to the user in real time and receive interactive responses as controls to generate the next latent frame. Unlike existing approaches, our method explores adversarial training as an effective paradigm for autoregressive generation. This not only allows us to design an architecture that is more efficient for one-step generation while fully utilizing the KV cache, but also enables training the model in a student-forcing manner that proves to be effective in reducing error accumulation during long video generation. Our experiments demonstrate that our 8B model achieves real-time, 24fps, streaming video generation at 736x416 resolution on a single H100, or 1280x720 on 8xH100 up to a minute long (1440 frames). Visit our research website at this https URL 

---
# Know What You Don't Know: Uncertainty Calibration of Process Reward Models 

**Authors**: Young-Jin Park, Kristjan Greenewald, Kaveh Alim, Hao Wang, Navid Azizan  

**Link**: [PDF](https://arxiv.org/pdf/2506.09338)  

**Abstract**: Process reward models (PRMs) play a central role in guiding inference-time scaling algorithms for large language models (LLMs). However, we observe that even state-of-the-art PRMs can be poorly calibrated and often overestimate success probabilities. To address this, we present a calibration approach, performed via quantile regression, that adjusts PRM outputs to better align with true success probabilities. Leveraging these calibrated success estimates and their associated confidence bounds, we introduce an \emph{instance-adaptive scaling} (IAS) framework that dynamically adjusts the inference budget based on the estimated likelihood that a partial reasoning trajectory will yield a correct final answer. Unlike conventional methods that allocate a fixed number of reasoning trajectories per query, this approach successfully adapts to each instance and reasoning step when using our calibrated PRMs. Experiments on mathematical reasoning benchmarks show that (i) our PRM calibration method successfully achieves small calibration error, outperforming the baseline methods, (ii) calibration is crucial for enabling effective adaptive scaling, and (iii) the proposed IAS strategy reduces inference costs while maintaining final answer accuracy, utilizing less compute on more confident problems as desired. 

---
# "Is This Really a Human Peer Supporter?": Misalignments Between Peer Supporters and Experts in LLM-Supported Interactions 

**Authors**: Kellie Yu Hui Sim, Roy Ka-Wei Lee, Kenny Tsu Wei Choo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09354)  

**Abstract**: Mental health is a growing global concern, prompting interest in AI-driven solutions to expand access to psychosocial support. Peer support, grounded in lived experience, offers a valuable complement to professional care. However, variability in training, effectiveness, and definitions raises concerns about quality, consistency, and safety. Large Language Models (LLMs) present new opportunities to enhance peer support interactions, particularly in real-time, text-based interactions. We present and evaluate an AI-supported system with an LLM-simulated distressed client, context-sensitive LLM-generated suggestions, and real-time emotion visualisations. 2 mixed-methods studies with 12 peer supporters and 5 mental health professionals (i.e., experts) examined the system's effectiveness and implications for practice. Both groups recognised its potential to enhance training and improve interaction quality. However, we found a key tension emerged: while peer supporters engaged meaningfully, experts consistently flagged critical issues in peer supporter responses, such as missed distress cues and premature advice-giving. This misalignment highlights potential limitations in current peer support training, especially in emotionally charged contexts where safety and fidelity to best practices are essential. Our findings underscore the need for standardised, psychologically grounded training, especially as peer support scales globally. They also demonstrate how LLM-supported systems can scaffold this development--if designed with care and guided by expert oversight. This work contributes to emerging conversations on responsible AI integration in mental health and the evolving role of LLMs in augmenting peer-delivered care. 

---
# LLM-ML Teaming: Integrated Symbolic Decoding and Gradient Search for Valid and Stable Generative Feature Transformation 

**Authors**: Xinyuan Wang, Haoyue Bai, Nanxu Gong, Wangyang Ying, Sixun Dong, Xiquan Cui, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09085)  

**Abstract**: Feature transformation enhances data representation by deriving new features from the original data. Generative AI offers potential for this task, but faces challenges in stable generation (consistent outputs) and valid generation (error-free sequences). Existing methods--traditional MLs' low validity and LLMs' instability--fail to resolve both. We find that LLMs ensure valid syntax, while ML's gradient-steered search stabilizes performance. To bridge this gap, we propose a teaming framework combining LLMs' symbolic generation with ML's gradient optimization. This framework includes four steps: (1) golden examples generation, aiming to prepare high-quality samples with the ground knowledge of the teacher LLM; (2) feature transformation sequence embedding and search, intending to uncover potentially superior embeddings within the latent space; (3) student LLM feature transformation, aiming to distill knowledge from the teacher LLM; (4) LLM-ML decoder teaming, dedicating to combine ML and the student LLM probabilities for valid and stable generation. The experiments on various datasets show that the teaming policy can achieve 5\% improvement in downstream performance while reducing nearly half of the error cases. The results also demonstrate the efficiency and robustness of the teaming policy. Additionally, we also have exciting findings on LLMs' capacity to understand the original data. 

---
# EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model 

**Authors**: Alyssa Pinnock, Shakya Jayakody, Kawsher A Roxy, Md Rubel Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.09061)  

**Abstract**: This paper introduces EdgeProfiler, a fast profiling framework designed for evaluating lightweight Large Language Models (LLMs) on edge systems. While LLMs offer remarkable capabilities in natural language understanding and generation, their high computational, memory, and power requirements often confine them to cloud environments. EdgeProfiler addresses these challenges by providing a systematic methodology for assessing LLM performance in resource-constrained edge settings. The framework profiles compact LLMs, including TinyLLaMA, Gemma3.1B, Llama3.2-1B, and DeepSeek-r1-1.5B, using aggressive quantization techniques and strict memory constraints. Analytical modeling is used to estimate latency, FLOPs, and energy consumption. The profiling reveals that 4-bit quantization reduces model memory usage by approximately 60-70%, while maintaining accuracy within 2-5% of full-precision baselines. Inference speeds are observed to improve by 2-3x compared to FP16 baselines across various edge devices. Power modeling estimates a 35-50% reduction in energy consumption for INT4 configurations, enabling practical deployment on hardware such as Raspberry Pi 4/5 and Jetson Orin Nano Super. Our findings emphasize the importance of efficient profiling tailored to lightweight LLMs in edge environments, balancing accuracy, energy efficiency, and computational feasibility. 

---
# Enhanced Whole Page Optimization via Mixed-Grained Reward Mechanism-Adapted Language Models 

**Authors**: Xinyuan Wang, Liang Wu, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09084)  

**Abstract**: Optimizing the presentation of search and recommendation results is crucial to enhancing user experience and engagement. Whole Page Optimization (WPO) plays a pivotal role in this process, as it directly influences how information is surfaced to users. While Pre-trained Large Language Models (LLMs) have demonstrated remarkable capabilities in generating coherent and contextually relevant content, fine-tuning these models for complex tasks like WPO presents challenges. Specifically, the need for extensive human-annotated data to mitigate issues such as hallucinations and model instability can be prohibitively expensive, especially in large-scale systems that interact with millions of items daily. In this work, we address the challenge of fine-tuning LLMs for WPO by using user feedback as the supervision. Unlike manually labeled datasets, user feedback is inherently noisy and less precise. To overcome this, we propose a reward-based fine-tuning approach, PageLLM, which employs a mixed-grained reward mechanism that combines page-level and item-level rewards. The page-level reward evaluates the overall quality and coherence, while the item-level reward focuses on the accuracy and relevance of key recommendations. This dual-reward structure ensures that both the holistic presentation and the critical individual components are optimized. We validate PageLLM on both public and industrial datasets. PageLLM outperforms baselines and achieves a 0.44\% GMV increase in an online A/B test with over 10 million users, demonstrating its real-world impact. 

---
# CUDA-LLM: LLMs Can Write Efficient CUDA Kernels 

**Authors**: Wentao Chen, Jiace Zhu, Qi Fan, Yehan Ma, An Zou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09092)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in general-purpose code generation. However, generating the code which is deeply hardware-specific, architecture-aware, and performance-critical, especially for massively parallel GPUs, remains a complex challenge. In this work, we explore the use of LLMs for the automated generation and optimization of CUDA programs, with the goal of producing high-performance GPU kernels that fully exploit the underlying hardware. To address this challenge, we propose a novel framework called \textbf{Feature Search and Reinforcement (FSR)}. FSR jointly optimizes compilation and functional correctness, as well as the runtime performance, which are validated through extensive and diverse test cases, and measured by actual kernel execution latency on the target GPU, respectively. This approach enables LLMs not only to generate syntactically and semantically correct CUDA code but also to iteratively refine it for efficiency, tailored to the characteristics of the GPU architecture. We evaluate FSR on representative CUDA kernels, covering AI workloads and computational intensive algorithms. Our results show that LLMs augmented with FSR consistently guarantee correctness rates. Meanwhile, the automatically generated kernels can outperform general human-written code by a factor of up to 179$\times$ in execution speeds. These findings highlight the potential of combining LLMs with performance reinforcement to automate GPU programming for hardware-specific, architecture-sensitive, and performance-critical applications. 

---
