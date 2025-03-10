# AceWGS: An LLM-Aided Framework to Accelerate Catalyst Design for Water-Gas Shift Reactions 

**Authors**: Joyjit Chattoraj, Brahim Hamadicharef, Teo Shi Chang, Yingzhi Zeng, Chee Kok Poh, Luwei Chen, Teck Leong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.05607)  

**Abstract**: While the Water-Gas Shift (WGS) reaction plays a crucial role in hydrogen production for fuel cells, finding suitable catalysts to achieve high yields for low-temperature WGS reactions remains a persistent challenge. Artificial Intelligence (AI) has shown promise in accelerating catalyst design by exploring vast candidate spaces, however, two key gaps limit its effectiveness. First, AI models primarily train on numerical data, which fail to capture essential text-based information, such as catalyst synthesis methods. Second, the cross-disciplinary nature of catalyst design requires seamless collaboration between AI, theory, experiments, and numerical simulations, often leading to communication barriers. To address these gaps, we present AceWGS, a Large Language Models (LLMs)-aided framework to streamline WGS catalyst design. AceWGS interacts with researchers through natural language, answering queries based on four features: (i) answering general queries, (ii) extracting information about the database comprising WGS-related journal articles, (iii) comprehending the context described in these articles, and (iv) identifying catalyst candidates using our proposed AI inverse model. We presented a practical case study demonstrating how AceWGS can accelerate the catalyst design process. AceWGS, built with open-source tools, offers an adjustable framework that researchers can readily adapt for a range of AI-accelerated catalyst design applications, supporting seamless integration across cross-disciplinary studies. 

---
# Symbolic Mixture-of-Experts: Adaptive Skill-based Routing for Heterogeneous Reasoning 

**Authors**: Justin Chih-Yao Chen, Sukwon Yun, Elias Stengel-Eskin, Tianlong Chen, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2503.05641)  

**Abstract**: Combining existing pre-trained expert LLMs is a promising avenue for scalably tackling large-scale and diverse tasks. However, selecting experts at the task level is often too coarse-grained, as heterogeneous tasks may require different expertise for each instance. To enable adaptive instance-level mixing of pre-trained LLM experts, we propose Symbolic-MoE, a symbolic, text-based, and gradient-free Mixture-of-Experts framework. Symbolic-MoE takes a fine-grained approach to selection by emphasizing skills, e.g., algebra in math or molecular biology in biomedical reasoning. We propose a skill-based recruiting strategy that dynamically selects the most relevant set of expert LLMs for diverse reasoning tasks based on their strengths. Each selected expert then generates its own reasoning, resulting in k outputs from k experts, which are then synthesized into a final high-quality response by an aggregator chosen based on its ability to integrate diverse reasoning outputs. We show that Symbolic-MoE's instance-level expert selection improves performance by a large margin but -- when implemented naively -- can introduce a high computational overhead due to the need for constant model loading and offloading. To address this, we implement a batch inference strategy that groups instances based on their assigned experts, loading each model only once. This allows us to integrate 16 expert models on 1 GPU with a time cost comparable to or better than prior multi-agent baselines using 4 GPUs. Through extensive evaluations on diverse benchmarks (MMLU-Pro, GPQA, AIME, and MedMCQA), we demonstrate that Symbolic-MoE outperforms strong LLMs like GPT4o-mini, as well as multi-agent approaches, with an absolute average improvement of 8.15% over the best multi-agent baseline. Moreover, Symbolic-MoE removes the need for expensive multi-round discussions, outperforming discussion baselines with less computation. 

---
# Learning LLM Preference over Intra-Dialogue Pairs: A Framework for Utterance-level Understandings 

**Authors**: Xuanqing Liu, Luyang Kong, Wei Niu, Afshin Khashei, Belinda Zeng, Steve Johnson, Jon Jay, Davor Golac, Matt Pope  

**Link**: [PDF](https://arxiv.org/pdf/2503.05620)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in handling complex dialogue tasks without requiring use case-specific fine-tuning. However, analyzing live dialogues in real-time necessitates low-latency processing systems, making it impractical to deploy models with billions of parameters due to latency constraints. As a result, practitioners often prefer smaller models with millions of parameters, trained on high-quality, human-annotated datasets. Yet, curating such datasets is both time-consuming and costly. Consequently, there is a growing need to combine the scalability of LLM-generated labels with the precision of human annotations, enabling fine-tuned smaller models to achieve both higher speed and accuracy comparable to larger models. In this paper, we introduce a simple yet effective framework to address this challenge. Our approach is specifically designed for per-utterance classification problems, which encompass tasks such as intent detection, dialogue state tracking, and more. To mitigate the impact of labeling errors from LLMs -- the primary source of inaccuracies in student models -- we propose a noise-reduced preference learning loss. Experimental results demonstrate that our method significantly improves accuracy across utterance-level dialogue tasks, including sentiment detection (over $2\%$), dialogue act classification (over $1.5\%$), etc. 

---
# Understanding the Limits of Lifelong Knowledge Editing in LLMs 

**Authors**: Lukas Thede, Karsten Roth, Matthias Bethge, Zeynep Akata, Tom Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2503.05683)  

**Abstract**: Keeping large language models factually up-to-date is crucial for deployment, yet costly retraining remains a challenge. Knowledge editing offers a promising alternative, but methods are only tested on small-scale or synthetic edit benchmarks. In this work, we aim to bridge research into lifelong knowledge editing to real-world edits at practically relevant scale. We first introduce WikiBigEdit; a large-scale benchmark of real-world Wikidata edits, built to automatically extend lifelong for future-proof benchmarking. In its first instance, it includes over 500K question-answer pairs for knowledge editing alongside a comprehensive evaluation pipeline. Finally, we use WikiBigEdit to study existing knowledge editing techniques' ability to incorporate large volumes of real-world facts and contrast their capabilities to generic modification techniques such as retrieval augmentation and continual finetuning to acquire a complete picture of the practical extent of current lifelong knowledge editing. 

---
# Chain of Strategy Optimization Makes Large Language Models Better Emotional Supporter 

**Authors**: Weixiang Zhao, Xingyu Sui, Xinyang Han, Yang Deng, Yulin Hu, Jiahe Guo, Libo Qin, Qianyun Du, Shijin Wang, Yanyan Zhao, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05362)  

**Abstract**: The growing emotional stress in modern society has increased the demand for Emotional Support Conversations (ESC). While Large Language Models (LLMs) show promise for ESC, they face two key challenges: (1) low strategy selection accuracy, and (2) preference bias, limiting their adaptability to emotional needs of users. Existing supervised fine-tuning (SFT) struggles to address these issues, as it rigidly trains models on single gold-standard responses without modeling nuanced strategy trade-offs. To overcome these limitations, we propose Chain-of-Strategy Optimization (CSO), a novel approach that optimizes strategy selection preferences at each dialogue turn. We first leverage Monte Carlo Tree Search to construct ESC-Pro, a high-quality preference dataset with turn-level strategy-response pairs. Training on ESC-Pro with CSO improves both strategy accuracy and bias mitigation, enabling LLMs to generate more empathetic and contextually appropriate responses. Experiments on LLaMA-3.1-8B, Gemma-2-9B, and Qwen2.5-7B demonstrate that CSO outperforms standard SFT, highlighting the efficacy of fine-grained, turn-level preference modeling in ESC. 

---
# Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models 

**Authors**: Anar Yeginbergen, Maite Oronoz, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2503.05328)  

**Abstract**: This paper investigates the role of dynamic external knowledge integration in improving counter-argument generation using Large Language Models (LLMs). While LLMs have shown promise in argumentative tasks, their tendency to generate lengthy, potentially unfactual responses highlights the need for more controlled and evidence-based approaches. We introduce a new manually curated dataset of argument and counter-argument pairs specifically designed to balance argumentative complexity with evaluative feasibility. We also propose a new LLM-as-a-Judge evaluation methodology that shows a stronger correlation with human judgments compared to traditional reference-based metrics. Our experimental results demonstrate that integrating dynamic external knowledge from the web significantly improves the quality of generated counter-arguments, particularly in terms of relatedness, persuasiveness, and factuality. The findings suggest that combining LLMs with real-time external knowledge retrieval offers a promising direction for developing more effective and reliable counter-argumentation systems. 

---
# An Empirical Study of Conformal Prediction in LLM with ASP Scaffolds for Robust Reasoning 

**Authors**: Navdeep Kaur, Lachlan McPheat, Alessandra Russo, Anthony G Cohn, Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2503.05439)  

**Abstract**: In this paper, we examine the use of Conformal Language Modelling (CLM) alongside Answer Set Programming (ASP) to enhance the performance of standard open-weight LLMs on complex multi-step reasoning tasks. Using the StepGame dataset, which requires spatial reasoning, we apply CLM to generate sets of ASP programs from an LLM, providing statistical guarantees on the correctness of the outputs. Experimental results show that CLM significantly outperforms baseline models that use standard sampling methods, achieving substantial accuracy improvements across different levels of reasoning complexity. Additionally, the LLM-as-Judge metric enhances CLM's performance, especially in assessing structurally and logically correct ASP outputs. However, calibrating CLM with diverse calibration sets did not improve generalizability for tasks requiring much longer reasoning steps, indicating limitations in handling more complex tasks. 

---
# GEMA-Score: Granular Explainable Multi-Agent Score for Radiology Report Evaluation 

**Authors**: Zhenxuan Zhang, Kinhei Lee, Weihang Deng, Huichi Zhou, Zihao Jin, Jiahao Huang, Zhifan Gao, Dominic C Marshall, Yingying Fang, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05347)  

**Abstract**: Automatic medical report generation supports clinical diagnosis, reduces the workload of radiologists, and holds the promise of improving diagnosis consistency. However, existing evaluation metrics primarily assess the accuracy of key medical information coverage in generated reports compared to human-written reports, while overlooking crucial details such as the location and certainty of reported abnormalities. These limitations hinder the comprehensive assessment of the reliability of generated reports and pose risks in their selection for clinical use. Therefore, we propose a Granular Explainable Multi-Agent Score (GEMA-Score) in this paper, which conducts both objective quantification and subjective evaluation through a large language model-based multi-agent workflow. Our GEMA-Score parses structured reports and employs NER-F1 calculations through interactive exchanges of information among agents to assess disease diagnosis, location, severity, and uncertainty. Additionally, an LLM-based scoring agent evaluates completeness, readability, and clinical terminology while providing explanatory feedback. Extensive experiments validate that GEMA-Score achieves the highest correlation with human expert evaluations on a public dataset, demonstrating its effectiveness in clinical scoring (Kendall coefficient = 0.70 for Rexval dataset and Kendall coefficient = 0.54 for RadEvalX dataset). The anonymous project demo is available at: this https URL. 

---
# MM-StoryAgent: Immersive Narrated Storybook Video Generation with a Multi-Agent Paradigm across Text, Image and Audio 

**Authors**: Xuenan Xu, Jiahao Mei, Chenliang Li, Yuning Wu, Ming Yan, Shaopeng Lai, Ji Zhang, Mengyue Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05242)  

**Abstract**: The rapid advancement of large language models (LLMs) and artificial intelligence-generated content (AIGC) has accelerated AI-native applications, such as AI-based storybooks that automate engaging story production for children. However, challenges remain in improving story attractiveness, enriching storytelling expressiveness, and developing open-source evaluation benchmarks and frameworks. Therefore, we propose and opensource MM-StoryAgent, which creates immersive narrated video storybooks with refined plots, role-consistent images, and multi-channel audio. MM-StoryAgent designs a multi-agent framework that employs LLMs and diverse expert tools (generative models and APIs) across several modalities to produce expressive storytelling videos. The framework enhances story attractiveness through a multi-stage writing pipeline. In addition, it improves the immersive storytelling experience by integrating sound effects with visual, music and narrative assets. MM-StoryAgent offers a flexible, open-source platform for further development, where generative modules can be substituted. Both objective and subjective evaluation regarding textual story quality and alignment between modalities validate the effectiveness of our proposed MM-StoryAgent system. The demo and source code are available. 

---
# AutoIOT: LLM-Driven Automated Natural Language Programming for AIoT Applications 

**Authors**: Leming Shen, Qiang Yang, Yuanqing Zheng, Mo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05346)  

**Abstract**: The advent of Large Language Models (LLMs) has profoundly transformed our lives, revolutionizing interactions with AI and lowering the barrier to AI usage. While LLMs are primarily designed for natural language interaction, the extensive embedded knowledge empowers them to comprehend digital sensor data. This capability enables LLMs to engage with the physical world through IoT sensors and actuators, performing a myriad of AIoT tasks. Consequently, this evolution triggers a paradigm shift in conventional AIoT application development, democratizing its accessibility to all by facilitating the design and development of AIoT applications via natural language. However, some limitations need to be addressed to unlock the full potential of LLMs in AIoT application development. First, existing solutions often require transferring raw sensor data to LLM servers, which raises privacy concerns, incurs high query fees, and is limited by token size. Moreover, the reasoning processes of LLMs are opaque to users, making it difficult to verify the robustness and correctness of inference results. This paper introduces AutoIOT, an LLM-based automated program generator for AIoT applications. AutoIOT enables users to specify their requirements using natural language (input) and automatically synthesizes interpretable programs with documentation (output). AutoIOT automates the iterative optimization to enhance the quality of generated code with minimum user involvement. AutoIOT not only makes the execution of AIoT tasks more explainable but also mitigates privacy concerns and reduces token costs with local execution of synthesized programs. Extensive experiments and user studies demonstrate AutoIOT's remarkable capability in program synthesis for various AIoT tasks. The synthesized programs can match and even outperform some representative baselines. 

---
# Similarity-Based Domain Adaptation with LLMs 

**Authors**: Jie He, Wendi Zhou, Xiang Lorraine Li, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.05281)  

**Abstract**: Unsupervised domain adaptation leverages abundant labeled data from various source domains to generalize onto unlabeled target data. Prior research has primarily focused on learning domain-invariant features across the source and target domains. However, these methods often require training a model using source domain data, which is time-consuming and can limit model usage for applications with different source data. This paper introduces a simple framework that utilizes the impressive generalization capabilities of Large Language Models (LLMs) for target data annotation without the need of source model training, followed by a novel similarity-based knowledge distillation loss. Our extensive experiments on cross-domain text classification reveal that our framework achieves impressive performance, specifically, 2.44\% accuracy improvement when compared to the SOTA method. 

---
# ORANSight-2.0: Foundational LLMs for O-RAN 

**Authors**: Pranshav Gajjar, Vijay K. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2503.05200)  

**Abstract**: Despite the transformative impact of Large Language Models (LLMs) across critical domains such as healthcare, customer service, and business marketing, their integration into Open Radio Access Networks (O-RAN) remains limited. This gap is primarily due to the absence of domain-specific foundational models, with existing solutions often relying on general-purpose LLMs that fail to address the unique challenges and technical intricacies of O-RAN. To bridge this gap, we introduce ORANSight-2.0 (O-RAN Insights), a pioneering initiative aimed at developing specialized foundational LLMs tailored for O-RAN. Built on 18 LLMs spanning five open-source LLM frameworks, ORANSight-2.0 fine-tunes models ranging from 1 to 70B parameters, significantly reducing reliance on proprietary, closed-source models while enhancing performance for O-RAN. At the core of ORANSight-2.0 is RANSTRUCT, a novel Retrieval-Augmented Generation (RAG) based instruction-tuning framework that employs two LLM agents to create high-quality instruction-tuning datasets. The generated dataset is then used to fine-tune the 18 pre-trained open-source LLMs via QLoRA. To evaluate ORANSight-2.0, we introduce srsRANBench, a novel benchmark designed for code generation and codebase understanding in the context of srsRAN, a widely used 5G O-RAN stack. We also leverage ORANBench13K, an existing benchmark for assessing O-RAN-specific knowledge. Our comprehensive evaluations demonstrate that ORANSight-2.0 models outperform general-purpose and closed-source models, such as ChatGPT-4o and Gemini, by 5.421% on ORANBench and 18.465% on srsRANBench, achieving superior performance while maintaining lower computational and energy costs. We also experiment with RAG-augmented variants of ORANSight-2.0 LLMs and thoroughly evaluate their energy characteristics, demonstrating costs for training, standard inference, and RAG-augmented inference. 

---
# Knowledge Updating? No More Model Editing! Just Selective Contextual Reasoning 

**Authors**: Guoxiu He, Xin Song, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.05212)  

**Abstract**: As real-world knowledge evolves, the information embedded within large language models (LLMs) can become outdated, inadequate, or erroneous. Model editing has emerged as a prominent approach for updating LLMs' knowledge with minimal computational costs and parameter changes. This approach typically identifies and adjusts specific model parameters associated with newly acquired knowledge. However, existing methods often underestimate the adverse effects that parameter modifications can have on broadly distributed knowledge. More critically, post-edit LLMs frequently struggle with multi-hop reasoning and continuous knowledge updates. Although various studies have discussed these shortcomings, there is a lack of comprehensive evaluation. In this paper, we provide an evaluation of ten model editing methods along four dimensions: reliability, generalization, locality, and portability. Results confirm that all ten popular model editing methods show significant shortcomings across multiple dimensions, suggesting model editing is less promising. We then propose a straightforward method called Selective Contextual Reasoning (SCR), for knowledge updating. SCR does not modify model parameters but harnesses LLM's inherent contextual reasoning capabilities utilizing the updated knowledge pieces. Under SCR, an LLM first assesses whether an incoming query falls within the scope of an external knowledge base. If it does, the relevant external knowledge texts are contextualized to enhance reasoning; otherwise, the query is answered directly. We evaluate SCR against the ten model editing methods on two counterfactual datasets with three backbone LLMs. Empirical results confirm the effectiveness and efficiency of contextual reasoning for knowledge updating. 

---
# Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching 

**Authors**: Simon A. Aytes, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05179)  

**Abstract**: Recent advances in large language models have demonstrated remarkable reasoning capabilities through Chain of Thought (CoT) prompting, but often at the cost of excessive verbosity in their intermediate outputs, which increases computational overhead. We introduce Sketch-of-Thought (SoT), a novel prompting framework that combines cognitive-inspired reasoning paradigms with linguistic constraints to minimize token usage while preserving reasoning accuracy. SoT is designed as a flexible framework that can incorporate any custom reasoning paradigms based on cognitive science, and we instantiate it with three such paradigms - Conceptual Chaining, Chunked Symbolism, and Expert Lexicons - each tailored to different reasoning tasks and selected dynamically via a lightweight routing model. Through comprehensive evaluation across 15 reasoning datasets with multiple languages and multimodal scenarios, we demonstrate that SoT achieves token reductions of 76% with negligible accuracy impact. In certain domains like mathematical and multi-hop reasoning, it even improves accuracy while using significantly fewer tokens. Our code is publicly available: this https URL. 

---
# RocketEval: Efficient Automated LLM Evaluation via Grading Checklist 

**Authors**: Tianjun Wei, Wei Wen, Ruizhi Qiao, Xing Sun, Jianghong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.05142)  

**Abstract**: Evaluating large language models (LLMs) in diverse and challenging scenarios is essential to align them with human preferences. To mitigate the prohibitive costs associated with human evaluations, utilizing a powerful LLM as a judge has emerged as a favored approach. Nevertheless, this methodology encounters several challenges, including substantial expenses, concerns regarding privacy and security, and reproducibility. In this paper, we propose a straightforward, replicable, and accurate automated evaluation method by leveraging a lightweight LLM as the judge, named RocketEval. Initially, we identify that the performance disparity between lightweight and powerful LLMs in evaluation tasks primarily stems from their ability to conduct comprehensive analyses, which is not easily enhanced through techniques such as chain-of-thought reasoning. By reframing the evaluation task as a multi-faceted Q&A using an instance-specific checklist, we demonstrate that the limited judgment accuracy of lightweight LLMs is largely attributes to high uncertainty and positional bias. To address these challenges, we introduce an automated evaluation process grounded in checklist grading, which is designed to accommodate a variety of scenarios and questions. This process encompasses the creation of checklists, the grading of these checklists by lightweight LLMs, and the reweighting of checklist items to align with the supervised annotations. Our experiments carried out on the automated evaluation benchmarks, MT-Bench and WildBench datasets, reveal that RocketEval, when using Gemma-2-2B as the judge, achieves a high correlation (0.965) with human preferences, which is comparable to GPT-4o. Moreover, RocketEval provides a cost reduction exceeding 50-fold for large-scale evaluation and comparison scenarios. Our code is available at this https URL . 

---
# SpecServe: Efficient and SLO-Aware Large Language Model Serving with Adaptive Speculative Decoding 

**Authors**: Kaiyu Huang, Hao Wu, Zhubo Shi, Han Zou, Minchen Yu, Qingjiang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.05096)  

**Abstract**: Large Language Model (LLM) services often face challenges in achieving low inference latency and meeting Service Level Objectives (SLOs) under dynamic request patterns. Speculative decoding, which exploits lightweight models for drafting and LLMs for verification, has emerged as a compelling technique to accelerate LLM inference. However, existing speculative decoding solutions often fail to adapt to varying workloads and system environments, resulting in performance variability and SLO violations. In this paper, we introduce SpecServe, an efficient LLM inference system that dynamically adjusts speculative strategies according to real-time request loads and system configurations. SpecServe proposes a theoretical model to understand and predict the efficiency of speculative decoding across diverse scenarios. Additionally, it implements intelligent drafting and verification algorithms to guarantee optimal performance while achieving high SLO attainment. Experimental results on real-world LLM traces demonstrate that SpecServe consistently meets SLOs and achieves substantial performance improvements, yielding 1.14$\times$-14.3$\times$ speedups over state-of-the-art speculative inference systems. 

---
# No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding 

**Authors**: Michael Krumdick, Charles Lovering, Varshini Reddy, Seth Ebner, Chris Tanner  

**Link**: [PDF](https://arxiv.org/pdf/2503.05061)  

**Abstract**: LLM-as-a-Judge is a framework that uses an LLM (large language model) to evaluate the quality of natural language text - typically text that is also generated by an LLM. This framework holds great promise due to its relative low-cost, ease of use, and strong correlations with human stylistic preferences. However, LLM Judges have been shown to exhibit biases that can distort their judgments. We evaluate how well LLM Judges can grade whether a given response to a conversational question is correct, an ability crucial to soundly estimating the overall response quality. To do so, we create and publicly release a human-annotated dataset with labels of correctness for 1,200 LLM responses. We source questions from a combination of existing datasets and a novel, challenging benchmark (BFF-Bench) created for this analysis. We demonstrate a strong connection between an LLM's ability to correctly answer a question and grade responses to that question. Although aggregate level statistics might imply a judge has high agreement with human annotators, it will struggle on the subset of questions it could not answer. To address this issue, we recommend a simple solution: provide the judge with a correct, human-written reference answer. We perform an in-depth analysis on how reference quality can affect the performance of an LLM Judge. We show that providing a weaker judge (e.g. Qwen 2.5 7B) with higher quality references reaches better agreement with human annotators than a stronger judge (e.g. GPT-4o) with synthetic references. 

---
# Rewarding Curse: Analyze and Mitigate Reward Modeling Issues for LLM Reasoning 

**Authors**: Jiachun Li, Pengfei Cao, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05188)  

**Abstract**: Chain-of-thought (CoT) prompting demonstrates varying performance under different reasoning tasks. Previous work attempts to evaluate it but falls short in providing an in-depth analysis of patterns that influence the CoT. In this paper, we study the CoT performance from the perspective of effectiveness and faithfulness. For the former, we identify key factors that influence CoT effectiveness on performance improvement, including problem difficulty, information gain, and information flow. For the latter, we interpret the unfaithful CoT issue by conducting a joint analysis of the information interaction among the question, CoT, and answer. The result demonstrates that, when the LLM predicts answers, it can recall correct information missing in the CoT from the question, leading to the problem. Finally, we propose a novel algorithm to mitigate this issue, in which we recall extra information from the question to enhance the CoT generation and evaluate CoTs based on their information gain. Extensive experiments demonstrate that our approach enhances both the faithfulness and effectiveness of CoT. 

---
# A Unified Framework with Novel Metrics for Evaluating the Effectiveness of XAI Techniques in LLMs 

**Authors**: Melkamu Abay Mersha, Mesay Gemeda Yigezu, Hassan shakil, Ali Al shami, Sanghyun Byun, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2503.05050)  

**Abstract**: The increasing complexity of LLMs presents significant challenges to their transparency and interpretability, necessitating the use of eXplainable AI (XAI) techniques to enhance trustworthiness and usability. This study introduces a comprehensive evaluation framework with four novel metrics for assessing the effectiveness of five XAI techniques across five LLMs and two downstream tasks. We apply this framework to evaluate several XAI techniques LIME, SHAP, Integrated Gradients, Layer-wise Relevance Propagation (LRP), and Attention Mechanism Visualization (AMV) using the IMDB Movie Reviews and Tweet Sentiment Extraction datasets. The evaluation focuses on four key metrics: Human-reasoning Agreement (HA), Robustness, Consistency, and Contrastivity. Our results show that LIME consistently achieves high scores across multiple LLMs and evaluation metrics, while AMV demonstrates superior Robustness and near-perfect Consistency. LRP excels in Contrastivity, particularly with more complex models. Our findings provide valuable insights into the strengths and limitations of different XAI methods, offering guidance for developing and selecting appropriate XAI techniques for LLMs. 

---
# Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety 

**Authors**: Yuyou Zhang, Miao Li, William Han, Yihang Yao, Zhepeng Cen, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05021)  

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks that exploit weaknesses in traditional safety alignment, which often relies on rigid refusal heuristics or representation engineering to block harmful outputs. While they are effective for direct adversarial attacks, they fall short of broader safety challenges requiring nuanced, context-aware decision-making. To address this, we propose Reasoning-enhanced Finetuning for interpretable LLM Safety (Rational), a novel framework that trains models to engage in explicit safe reasoning before response. Fine-tuned models leverage the extensive pretraining knowledge in self-generated reasoning to bootstrap their own safety through structured reasoning, internalizing context-sensitive decision-making. Our findings suggest that safety extends beyond refusal, requiring context awareness for more robust, interpretable, and adaptive responses. Reasoning is not only a core capability of LLMs but also a fundamental mechanism for LLM safety. Rational employs reasoning-enhanced fine-tuning, allowing it to reject harmful prompts while providing meaningful and context-aware responses in complex scenarios. 

---
# Biases in Large Language Model-Elicited Text: A Case Study in Natural Language Inference 

**Authors**: Grace Proebsting, Adam Poliak  

**Link**: [PDF](https://arxiv.org/pdf/2503.05047)  

**Abstract**: We test whether NLP datasets created with Large Language Models (LLMs) contain annotation artifacts and social biases like NLP datasets elicited from crowd-source workers. We recreate a portion of the Stanford Natural Language Inference corpus using GPT-4, Llama-2 70b for Chat, and Mistral 7b Instruct. We train hypothesis-only classifiers to determine whether LLM-elicited NLI datasets contain annotation artifacts. Next, we use pointwise mutual information to identify the words in each dataset that are associated with gender, race, and age-related terms. On our LLM-generated NLI datasets, fine-tuned BERT hypothesis-only classifiers achieve between 86-96% accuracy. Our analyses further characterize the annotation artifacts and stereotypical biases in LLM-generated datasets. 

---
# Leveraging Domain Knowledge at Inference Time for LLM Translation: Retrieval versus Generation 

**Authors**: Bryan Li, Jiaming Luo, Eleftheria Briakou, Colin Cherry  

**Link**: [PDF](https://arxiv.org/pdf/2503.05010)  

**Abstract**: While large language models (LLMs) have been increasingly adopted for machine translation (MT), their performance for specialist domains such as medicine and law remains an open challenge. Prior work has shown that LLMs can be domain-adapted at test-time by retrieving targeted few-shot demonstrations or terminologies for inclusion in the prompt. Meanwhile, for general-purpose LLM MT, recent studies have found some success in generating similarly useful domain knowledge from an LLM itself, prior to translation. Our work studies domain-adapted MT with LLMs through a careful prompting setup, finding that demonstrations consistently outperform terminology, and retrieval consistently outperforms generation. We find that generating demonstrations with weaker models can close the gap with larger model's zero-shot performance. Given the effectiveness of demonstrations, we perform detailed analyses to understand their value. We find that domain-specificity is particularly important, and that the popular multi-domain benchmark is testing adaptation to a particular writing style more so than to a specific domain. 

---
# Balcony: A Lightweight Approach to Dynamic Inference of Generative Language Models 

**Authors**: Benyamin Jamialahmadi, Parsa Kavehzadeh, Mehdi Rezagholizadeh, Parsa Farinneya, Hossein Rajabzadeh, Aref Jafari, Boxing Chen, Marzieh Tahaei  

**Link**: [PDF](https://arxiv.org/pdf/2503.05005)  

**Abstract**: Deploying large language models (LLMs) in real-world applications is often hindered by strict computational and latency constraints. While dynamic inference offers the flexibility to adjust model behavior based on varying resource budgets, existing methods are frequently limited by hardware inefficiencies or performance degradation. In this paper, we introduce Balcony, a simple yet highly effective framework for depth-based dynamic inference. By freezing the pretrained LLM and inserting additional transformer layers at selected exit points, Balcony maintains the full model's performance while enabling real-time adaptation to different computational budgets. These additional layers are trained using a straightforward self-distillation loss, aligning the sub-model outputs with those of the full model. This approach requires significantly fewer training tokens and tunable parameters, drastically reducing computational costs compared to prior methods. When applied to the LLaMA3-8B model, using only 0.2% of the original pretraining data, Balcony achieves minimal performance degradation while enabling significant speedups. Remarkably, we show that Balcony outperforms state-of-the-art methods such as Flextron and Layerskip as well as other leading compression techniques on multiple models and at various scales, across a variety of benchmarks. 

---
# DB-Explore: Automated Database Exploration and Instruction Synthesis for Text-to-SQL 

**Authors**: Haoyuan Ma, Yongliang Shen, Hengwei Liu, Wenqi Zhang, Haolei Xu, Qiuying Peng, Jun Wang, Weiming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04959)  

**Abstract**: Recent text-to-SQL systems powered by large language models (LLMs) have demonstrated remarkable performance in translating natural language queries into SQL. However, these systems often struggle with complex database structures and domain-specific queries, as they primarily focus on enhancing logical reasoning and SQL syntax while overlooking the critical need for comprehensive database understanding. To address this limitation, we propose DB-Explore, a novel framework that systematically aligns LLMs with database knowledge through automated exploration and instruction synthesis. DB-Explore constructs database graphs to capture complex relational schemas, leverages GPT-4 to systematically mine structural patterns and semantic knowledge, and synthesizes instructions to distill this knowledge for efficient fine-tuning of LLMs. Our framework enables comprehensive database understanding through diverse sampling strategies and automated instruction generation, bridging the gap between database structures and language models. Experiments conducted on the SPIDER and BIRD benchmarks validate the effectiveness of DB-Explore, achieving an execution accuracy of 52.1% on BIRD and 84.0% on SPIDER. Notably, our open-source implementation, based on the Qwen2.5-coder-7B model, outperforms multiple GPT-4-driven text-to-SQL systems in comparative evaluations, and achieves near state-of-the-art performance with minimal computational cost. 

---
# HILGEN: Hierarchically-Informed Data Generation for Biomedical NER Using Knowledgebases and Large Language Models 

**Authors**: Yao Ge, Yuting Guo, Sudeshna Das, Swati Rajwal, Selen Bozkurt, Abeed Sarker  

**Link**: [PDF](https://arxiv.org/pdf/2503.04930)  

**Abstract**: We present HILGEN, a Hierarchically-Informed Data Generation approach that combines domain knowledge from the Unified Medical Language System (UMLS) with synthetic data generated by large language models (LLMs), specifically GPT-3.5. Our approach leverages UMLS's hierarchical structure to expand training data with related concepts, while incorporating contextual information from LLMs through targeted prompts aimed at automatically generating synthetic examples for sparsely occurring named entities. The performance of the HILGEN approach was evaluated across four biomedical NER datasets (MIMIC III, BC5CDR, NCBI-Disease, and Med-Mentions) using BERT-Large and DANN (Data Augmentation with Nearest Neighbor Classifier) models, applying various data generation strategies, including UMLS, GPT-3.5, and their best ensemble. For the BERT-Large model, incorporating UMLS led to an average F1 score improvement of 40.36%, while using GPT-3.5 resulted in a comparable average increase of 40.52%. The Best-Ensemble approach using BERT-Large achieved the highest improvement, with an average increase of 42.29%. DANN model's F1 score improved by 22.74% on average using the UMLS-only approach. The GPT-3.5-based method resulted in a 21.53% increase, and the Best-Ensemble DANN model showed a more notable improvement, with an average increase of 25.03%. Our proposed HILGEN approach improves NER performance in few-shot settings without requiring additional manually annotated data. Our experiments demonstrate that an effective strategy for optimizing biomedical NER is to combine biomedical knowledge curated in the past, such as the UMLS, and generative LLMs to create synthetic training instances. Our future research will focus on exploring additional innovative synthetic data generation strategies for further improving NER performance. 

---
# Statistical Guarantees of Correctness Coverage for Medical Multiple-Choice Question Answering 

**Authors**: Yusong Ke  

**Link**: [PDF](https://arxiv.org/pdf/2503.05505)  

**Abstract**: Large language models (LLMs) are increasingly deployed in real-world question-answering (QA) applications. However, LLMs have been proven to generate hallucinations and nonfactual information, undermining their trustworthiness in high-stakes medical tasks. Conformal prediction (CP) is well-known to be model-agnostic and distribution-free, which creates statistically rigorous prediction sets in classification tasks. In this work, we for the first time adapt the CP framework to medical multiple-choice question-answering (MCQA) tasks, by correlating the nonconformity score with the frequency score of correct options grounded in self-consistency theory, assuming no access to internal model information. Considering that the adapted CP framework can only control the (mis)coverage rate, we employ a risk control framework, which can manage task-specific metrics by devising a monotonically decreasing loss function. We evaluate our framework on 3 popular medical MCQA datasets utilizing 4 ``off-the-shelf'' LLMs. Empirical results demonstrate that we achieve user-specified average (or marginal) error rates on the test set. Furthermore, we observe that the average prediction set size (APSS) on the test set decreases as the risk level increases, which concludes a promising evaluation metric for the uncertainty of LLMs. 

---
# Beyond RAG: Task-Aware KV Cache Compression for Comprehensive Knowledge Reasoning 

**Authors**: Giulio Corallo, Orion Weller, Fabio Petroni, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2503.04973)  

**Abstract**: Incorporating external knowledge in large language models (LLMs) enhances their utility across diverse applications, but existing methods have trade-offs. Retrieval-Augmented Generation (RAG) fetches evidence via similarity search, but key information may fall outside top ranked results. Long-context models can process multiple documents but are computationally expensive and limited by context window size. Inspired by students condensing study material for open-book exams, we propose task-aware key-value (KV) cache compression, which compresses external knowledge in a zero- or few-shot setup. This enables LLMs to reason efficiently over a compacted representation of all relevant information. Experiments show our approach outperforms both RAG and task-agnostic compression methods. On LongBench v2, it improves accuracy by up to 7 absolute points over RAG with a 30x compression rate, while reducing inference latency from 0.43s to 0.16s. A synthetic dataset highlights that RAG performs well when sparse evidence suffices, whereas task-aware compression is superior for broad knowledge tasks. 

---
# Are Large Language Models Good In-context Learners for Financial Sentiment Analysis? 

**Authors**: Xinyu Wei, Luojia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04873)  

**Abstract**: Recently, large language models (LLMs) with hundreds of billions of parameters have demonstrated the emergent ability, surpassing traditional methods in various domains even without fine-tuning over domain-specific data. However, when it comes to financial sentiment analysis (FSA)$\unicode{x2013}$a fundamental task in financial AI$\unicode{x2013}$these models often encounter various challenges, such as complex financial terminology, subjective human emotions, and ambiguous inclination expressions. In this paper, we aim to answer the fundamental question: whether LLMs are good in-context learners for FSA? Unveiling this question can yield informative insights on whether LLMs can learn to address the challenges by generalizing in-context demonstrations of financial document-sentiment pairs to the sentiment analysis of new documents, given that finetuning these models on finance-specific data is difficult, if not impossible at all. To the best of our knowledge, this is the first paper exploring in-context learning for FSA that covers most modern LLMs (recently released DeepSeek V3 included) and multiple in-context sample selection methods. Comprehensive experiments validate the in-context learning capability of LLMs for FSA. 

---
# Memory Is All You Need: Testing How Model Memory Affects LLM Performance in Annotation Tasks 

**Authors**: Joan C. Timoneda, Sebastián Vallejo Vera  

**Link**: [PDF](https://arxiv.org/pdf/2503.04874)  

**Abstract**: Generative Large Language Models (LLMs) have shown promising results in text annotation using zero-shot and few-shot learning. Yet these approaches do not allow the model to retain information from previous annotations, making each response independent from the preceding ones. This raises the question of whether model memory -- the LLM having knowledge about its own previous annotations in the same task -- affects performance. In this article, using OpenAI's GPT-4o and Meta's Llama 3.1 on two political science datasets, we demonstrate that allowing the model to retain information about its own previous classifications yields significant performance improvements: between 5 and 25\% when compared to zero-shot and few-shot learning. Moreover, memory reinforcement, a novel approach we propose that combines model memory and reinforcement learning, yields additional performance gains in three out of our four tests. These findings have important implications for applied researchers looking to improve performance and efficiency in LLM annotation tasks. 

---
# One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs 

**Authors**: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.04856)  

**Abstract**: Despite extensive safety enhancements in large language models (LLMs), multi-turn "jailbreak" conversations crafted by skilled human adversaries can still breach even the most sophisticated guardrails. However, these multi-turn attacks demand considerable manual effort, limiting their scalability. In this work, we introduce a novel approach called Multi-turn-to-Single-turn (M2S) that systematically converts multi-turn jailbreak prompts into single-turn attacks. Specifically, we propose three conversion strategies - Hyphenize, Numberize, and Pythonize - each preserving sequential context yet packaging it in a single query. Our experiments on the Multi-turn Human Jailbreak (MHJ) dataset show that M2S often increases or maintains high Attack Success Rates (ASRs) compared to original multi-turn conversations. Notably, using a StrongREJECT-based evaluation of harmfulness, M2S achieves up to 95.9% ASR on Mistral-7B and outperforms original multi-turn prompts by as much as 17.5% in absolute improvement on GPT-4o. Further analysis reveals that certain adversarial tactics, when consolidated into a single prompt, exploit structural formatting cues to evade standard policy checks. These findings underscore that single-turn attacks - despite being simpler and cheaper to conduct - can be just as potent, if not more, than their multi-turn counterparts. Our findings underscore the urgent need to reevaluate and reinforce LLM safety strategies, given how adversarial queries can be compacted into a single prompt while still retaining sufficient complexity to bypass existing safety measures. 

---
# Enhancing Collective Intelligence in Large Language Models Through Emotional Integration 

**Authors**: Likith Kadiyala, Ramteja Sajja, Yusuf Sermet, Ibrahim Demir  

**Link**: [PDF](https://arxiv.org/pdf/2503.04849)  

**Abstract**: This research investigates the integration of emotional diversity into Large Language Models (LLMs) to enhance collective intelligence. Inspired by the human wisdom of crowds phenomenon, where group decisions often outperform individual judgments, we fine-tuned the DarkIdol-Llama-3.1-8B model using Google's GoEmotions dataset and Low-Rank Adaptation (LoRA) to simulate emotionally diverse responses. Evaluating the model on a distance estimation task between Fargo, ND, and Seattle, WA, across 15,064 unique persona configurations, we analyzed how emotional states and social attributes influence decision-making. Our findings demonstrate that emotional integration shapes response patterns while maintaining acceptable prediction accuracy, revealing its potential to enhance artificial collective intelligence. This study provides valuable insights into the interplay of emotional diversity and decision-making in LLMs, suggesting pathways for creating emotionally aware AI systems that balance emotional depth with analytical precision. 

---
# TinyR1-32B-Preview: Boosting Accuracy with Branch-Merge Distillation 

**Authors**: Lin Sun, Guangxiang Zhao, Xiaoqi Jian, Yuhan Wu, Weihong Lin, Yongfu Zhu, Change Jia, Linglin Zhang, Jinzhu Wu, Junfeng Ran, Sai-er Hu, Zihan Jiang, Junting Zhou, Wenrui Liu, Bin Cui, Tong Yang, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04872)  

**Abstract**: The challenge of reducing the size of Large Language Models (LLMs) while maintaining their performance has gained significant attention. However, existing methods, such as model distillation and transfer learning, often fail to achieve high accuracy. To address this limitation, we introduce the Branch-Merge distillation approach, which enhances model compression through two phases: (1) the Branch Phase, where knowledge from a large teacher model is \textit{selectively distilled} into specialized student models via domain-specific supervised fine-tuning (SFT); And (2) the Merge Phase, where these student models are merged to enable cross-domain knowledge transfer and improve generalization. We validate our distillation approach using DeepSeek-R1 as the teacher and DeepSeek-R1-Distill-Qwen-32B as the student. The resulting merged model, TinyR1-32B-Preview, outperforms its counterpart DeepSeek-R1-Distill-Qwen-32B across multiple benchmarks, including Mathematics (+5.5 points), Coding (+4.4 points) and Science (+2.9 points), while achieving near-equal performance to DeepSeek-R1 on AIME 2024. The Branch-Merge distillation approach provides a scalable solution for creating smaller, high-performing LLMs with reduced computational cost and time. 

---
# Beyond Next Word Prediction: Developing Comprehensive Evaluation Frameworks for measuring LLM performance on real world applications 

**Authors**: Vishakha Agrawal, Archie Chaudhury, Shreya Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2503.04828)  

**Abstract**: While Large Language Models (LLMs) are fundamentally next-token prediction systems, their practical applications extend far beyond this basic function. From natural language processing and text generation to conversational assistants and software use, LLMs have numerous use-cases, and have already acquired a significant degree of enterprise adoption. To evaluate such models, static evaluation datasets, consisting of a set of prompts and their corresponding ground truths, are often used to benchmark the efficacy of the model for a particular task. In this paper, we provide the basis for a more comprehensive evaluation framework, based upon a traditional game and tool-based architecture that enables a more overarching measurement of a model's capabilities. For simplicity, we provide a generalized foundation that can be extended, without significant alteration, to numerous scenarios, from specific use cases such as supply chain management or financial reasoning, to abstract measurements such as ethics or safety. 

---
# "Only ChatGPT gets me": An Empirical Analysis of GPT versus other Large Language Models for Emotion Detection in Text 

**Authors**: Florian Lecourt, Madalina Croitoru, Konstantin Todorov  

**Link**: [PDF](https://arxiv.org/pdf/2503.04831)  

**Abstract**: This work investigates the capabilities of large language models (LLMs) in detecting and understanding human emotions through text. Drawing upon emotion models from psychology, we adopt an interdisciplinary perspective that integrates computational and affective sciences insights. The main goal is to assess how accurately they can identify emotions expressed in textual interactions and compare different models on this specific task. This research contributes to broader efforts to enhance human-computer interaction, making artificial intelligence technologies more responsive and sensitive to users' emotional nuances. By employing a methodology that involves comparisons with a state-of-the-art model on the GoEmotions dataset, we aim to gauge LLMs' effectiveness as a system for emotional analysis, paving the way for potential applications in various fields that require a nuanced understanding of human language. 

---
# Extrapolation Merging: Keep Improving With Extrapolation and Merging 

**Authors**: Yiguan Lin, Bin Xu, Yinghao Li, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04834)  

**Abstract**: Large Language Models (LLMs) require instruction fine-tuning to perform different downstream tasks. However, the instruction fine-tuning phase still demands significant computational resources and labeled data, lacking a paradigm that can improve model performance without additional computational power and data. Model merging aims to enhance performance by combining the parameters of different models, but the lack of a clear optimization direction during the merging process does not always guarantee improved performance. In this paper, we attempt to provide a clear optimization direction for model merging. We first validate the effectiveness of the model extrapolation method during the instruction fine-tuning phase. Then, we propose Extrapolation Merging, a paradigm that can continue improving model performance without requiring extra computational resources or data. Using the extrapolation method, we provide a clear direction for model merging, achieving local optimization search, and consequently enhancing the merged model's performance. We conduct experiments on seven different tasks, and the results show that our method can consistently improve the model's performance after fine-tuning. 

---
# Prompting Science Report 1: Prompt Engineering is Complicated and Contingent 

**Authors**: Lennart Meincke, Ethan Mollick, Lilach Mollick, Dan Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2503.04818)  

**Abstract**: This is the first of a series of short reports that seek to help business, education, and policy leaders understand the technical details of working with AI through rigorous testing. In this report, we demonstrate two things:
- There is no single standard for measuring whether a Large Language Model (LLM) passes a benchmark, and that choosing a standard has a big impact on how well the LLM does on that benchmark. The standard you choose will depend on your goals for using an LLM in a particular case.
- It is hard to know in advance whether a particular prompting approach will help or harm the LLM's ability to answer any particular question. Specifically, we find that sometimes being polite to the LLM helps performance, and sometimes it lowers performance. We also find that constraining the AI's answers helps performance in some cases, though it may lower performance in other cases.
Taken together, this suggests that benchmarking AI performance is not one-size-fits-all, and also that particular prompting formulas or approaches, like being polite to the AI, are not universally valuable. 

---
# Cite Before You Speak: Enhancing Context-Response Grounding in E-commerce Conversational LLM-Agents 

**Authors**: Jingying Zeng, Hui Liu, Zhenwei Dai, Xianfeng Tang, Chen Luo, Samarth Varshney, Zhen Li, Qi He  

**Link**: [PDF](https://arxiv.org/pdf/2503.04830)  

**Abstract**: With the advancement of conversational large language models (LLMs), several LLM-based Conversational Shopping Agents (CSA) have been developed to help customers answer questions and smooth their shopping journey in e-commerce domain. The primary objective in building a trustworthy CSA is to ensure the agent's responses are accurate and factually grounded, which is essential for building customer trust and encouraging continuous engagement. However, two challenges remain. First, LLMs produce hallucinated or unsupported claims. Such inaccuracies risk spreading misinformation and diminishing customer trust. Second, without providing knowledge source attribution in CSA response, customers struggle to verify LLM-generated information. To address these challenges, we present an easily productionized solution that enables a "citation experience" utilizing In-context Learning (ICL) and Multi-UX-Inference (MUI) to generate responses with citations to attribute its original sources without interfering other existing UX features. With proper UX design, these citation marks can be linked to the related product information and display the source to our customers. In this work, we also build auto-metrics and scalable benchmarks to holistically evaluate LLM's grounding and attribution capabilities. Our experiments demonstrate that incorporating this citation generation paradigm can substantially enhance the grounding of LLM responses by 13.83% on the real-world data. As such, our solution not only addresses the immediate challenges of LLM grounding issues but also adds transparency to conversational AI. 

---
# Memory-augmented Query Reconstruction for LLM-based Knowledge Graph Reasoning 

**Authors**: Mufan Xu, Gewen Liang, Kehai Chen, Wei Wang, Xun Zhou, Muyun Yang, Tiejun Zhao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05193)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance on knowledge graph question answering (KGQA) tasks by planning and interacting with knowledge graphs. However, existing methods often confuse tool utilization with knowledge reasoning, harming readability of model outputs and giving rise to hallucinatory tool invocations, which hinder the advancement of KGQA. To address this issue, we propose Memory-augmented Query Reconstruction for LLM-based Knowledge Graph Reasoning (MemQ) to decouple LLM from tool invocation tasks using LLM-built query memory. By establishing a memory module with explicit descriptions of query statements, the proposed MemQ facilitates the KGQA process with natural language reasoning and memory-augmented query reconstruction. Meanwhile, we design an effective and readable reasoning to enhance the LLM's reasoning capability in KGQA. Experimental results that MemQ achieves state-of-the-art performance on widely used benchmarks WebQSP and CWQ. 

---
# Framing the Game: How Context Shapes LLM Decision-Making 

**Authors**: Isaac Robinson, John Burden  

**Link**: [PDF](https://arxiv.org/pdf/2503.04840)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed across diverse contexts to support decision-making. While existing evaluations effectively probe latent model capabilities, they often overlook the impact of context framing on perceived rational decision-making. In this study, we introduce a novel evaluation framework that systematically varies evaluation instances across key features and procedurally generates vignettes to create highly varied scenarios. By analyzing decision-making patterns across different contexts with the same underlying game structure, we uncover significant contextual variability in LLM responses. Our findings demonstrate that this variability is largely predictable yet highly sensitive to framing effects. Our results underscore the need for dynamic, context-aware evaluation methodologies for real-world deployments. 

---
# Exploring and Evaluating Multimodal Knowledge Reasoning Consistency of Multimodal Large Language Models 

**Authors**: Boyu Jia, Junzhe Zhang, Huixuan Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04801)  

**Abstract**: In recent years, multimodal large language models (MLLMs) have achieved significant breakthroughs, enhancing understanding across text and vision. However, current MLLMs still face challenges in effectively integrating knowledge across these modalities during multimodal knowledge reasoning, leading to inconsistencies in reasoning outcomes. To systematically explore this issue, we propose four evaluation tasks and construct a new dataset. We conduct a series of experiments on this dataset to analyze and compare the extent of consistency degradation in multimodal knowledge reasoning within MLLMs. Based on the experimental results, we identify factors contributing to the observed degradation in consistency. Our research provides new insights into the challenges of multimodal knowledge reasoning and offers valuable guidance for future efforts aimed at improving MLLMs. 

---
# Optimizing Multi-Hop Document Retrieval Through Intermediate Representations 

**Authors**: Jiaen Lin, Jingyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04796)  

**Abstract**: Retrieval-augmented generation (RAG) encounters challenges when addressing complex queries, particularly multi-hop questions. While several methods tackle multi-hop queries by iteratively generating internal queries and retrieving external documents, these approaches are computationally expensive. In this paper, we identify a three-stage information processing pattern in LLMs during layer-by-layer reasoning, consisting of extraction, processing, and subsequent extraction steps. This observation suggests that the representations in intermediate layers contain richer information compared to those in other layers. Building on this insight, we propose Layer-wise RAG (L-RAG). Unlike prior methods that focus on generating new internal queries, L-RAG leverages intermediate representations from the middle layers, which capture next-hop information, to retrieve external knowledge. L-RAG achieves performance comparable to multi-step approaches while maintaining inference overhead similar to that of standard RAG. Experimental results show that L-RAG outperforms existing RAG methods on open-domain multi-hop question-answering datasets, including MuSiQue, HotpotQA, and 2WikiMultiHopQA. The code is available in this https URL 

---
# Sentence-level Reward Model can Generalize Better for Aligning LLM from Human Preference 

**Authors**: Wenjie Qiu, Yi-Chen Li, Xuqin Zhang, Tianyi Zhang, Yihang Zhang, Zongzhang Zhang, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04793)  

**Abstract**: Learning reward models from human preference datasets and subsequently optimizing language models via reinforcement learning has emerged as a fundamental paradigm for aligning LLMs with human preferences. The performance of the reward model plays a crucial role in the effectiveness of alignment. Previous reward models operate at a coarse-grained level, requiring the generation of a complete response to obtain a reward value. The sparse reward may present challenges for downstream reinforcement learning. While recent efforts have attempted to learn token-level reward models, the lack of explicit semantic information makes it difficult to model the credit of every individual token. In this paper, we propose assigning scores to every sentence, introducing an intermediate-grained reward model. By segmenting the complete response into sentences and applying differential operations to reward output at the start and end positions of each sentence, we can effectively model the rewards of sentences. Moreover, a novel attention mechanism is introduced to aggregate the scores of all sentences into a response-level score, which allows it to be trained using the Bradley-Terry model. On common benchmarks, our method outperforms the response-level reward model by 2.7% on RewardBench (for reward modeling evaluation) and surpasses all baselines on AlpacaEval (for alignment evaluation). 

---
# Cross-linguistic disagreement as a conflict of semantic alignment norms in multilingual AI~Linguistic Diversity as a Problem for Philosophy, Cognitive Science, and AI~ 

**Authors**: Masaharu Mizumoto, Dat Tien Nguyen, Justin Sytsma, Mark Alfano, Yu Izumi, Koji Fujita, Nguyen Le Minh  

**Link**: [PDF](https://arxiv.org/pdf/2503.04792)  

**Abstract**: Multilingual large language models (LLMs) face an often-overlooked challenge stemming from intrinsic semantic differences across languages. Linguistic divergence can sometimes lead to cross-linguistic disagreements--disagreements purely due to semantic differences about a relevant concept. This paper identifies such disagreements as conflicts between two fundamental alignment norms in multilingual LLMs: cross-linguistic consistency (CL-consistency), which seeks universal concepts across languages, and consistency with folk judgments (Folk-consistency), which respects language-specific semantic norms. Through examining responses of conversational multilingual AIs in English and Japanese with the cases used in philosophy (cases of knowledge-how attributions), this study demonstrates that even state-of-the-art LLMs provide divergent and internally inconsistent responses. Such findings reveal a novel qualitative limitation in crosslingual knowledge transfer, or conceptual crosslingual knowledge barriers, challenging the assumption that universal representations and cross-linguistic transfer capabilities are inherently desirable. Moreover, they reveal conflicts of alignment policies of their developers, highlighting critical normative questions for LLM researchers and developers. The implications extend beyond technical alignment challenges, raising normative, moral-political, and metaphysical questions about the ideals underlying AI development--questions that are shared with philosophers and cognitive scientists but for which no one yet has definitive answers, inviting a multidisciplinary approach to balance the practical benefits of cross-linguistic consistency and respect for linguistic diversity. 

---
# PanguIR Technical Report for NTCIR-18 AEOLLM Task 

**Authors**: Lang Mei, Chong Chen, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04809)  

**Abstract**: As large language models (LLMs) gain widespread attention in both academia and industry, it becomes increasingly critical and challenging to effectively evaluate their capabilities. Existing evaluation methods can be broadly categorized into two types: manual evaluation and automatic evaluation. Manual evaluation, while comprehensive, is often costly and resource-intensive. Conversely, automatic evaluation offers greater scalability but is constrained by the limitations of its evaluation criteria (dominated by reference-based answers). To address these challenges, NTCIR-18 introduced the AEOLLM (Automatic Evaluation of LLMs) task, aiming to encourage reference-free evaluation methods that can overcome the limitations of existing approaches. In this paper, to enhance the evaluation performance of the AEOLLM task, we propose three key methods to improve the reference-free evaluation: 1) Multi-model Collaboration: Leveraging multiple LLMs to approximate human ratings across various subtasks; 2) Prompt Auto-optimization: Utilizing LLMs to iteratively refine the initial task prompts based on evaluation feedback from training samples; and 3) In-context Learning (ICL) Optimization: Based on the multi-task evaluation feedback, we train a specialized in-context example retrieval model, combined with a semantic relevance retrieval model, to jointly identify the most effective in-context learning examples. Experiments conducted on the final dataset demonstrate that our approach achieves superior performance on the AEOLLM task. 

---
# Ext2Gen: Alignment through Unified Extraction and Generation for Robust Retrieval-Augmented Generation 

**Authors**: Hwanjun Song, Jeonghwan Choi, Minseok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.04789)  

**Abstract**: Retrieval-augmented generation (RAG) enhances LLMs by integrating external knowledge, but generation remains fragile due to the uncertain placement of relevant chunks and retrieval-induced information overload, leading to hallucinations. We propose Ext2Gen, a novel extract-then-generate model that enhances RAG robustness by first extracting query-relevant sentences before generating answers. To optimize this model, we employ preference alignment through pairwise feedback learning, enabling the model to generate robust answers regardless of variations in retrieval results. Extensive experiments demonstrate that Ext2Gen effectively identifies query-relevant sentences with high precision and recall, leading to highly reliable answers. Furthermore, deploying our model in a RAG environment reveals that it not only boosts the performance of the base LLM but also synergizes with advanced retrieval strategies like query expansion. The dataset and model will be released soon. 

---
# Cyber for AI at SemEval-2025 Task 4: Forgotten but Not Lost: The Balancing Act of Selective Unlearning in Large Language Models 

**Authors**: Dinesh Srivasthav P, Bala Mallikarjunarao Garlapati  

**Link**: [PDF](https://arxiv.org/pdf/2503.04795)  

**Abstract**: Large Language Models (LLMs) face significant challenges in maintaining privacy, ethics, and compliance, when sensitive or obsolete data must be selectively removed. Retraining these models from scratch is computationally infeasible, necessitating efficient alternatives. As part of the SemEval 2025 Task 4, this work focuses on the application of selective unlearning in LLMs to address this challenge. In this paper, we present our experiments and findings, primarily leveraging global weight modification to achieve an equilibrium between effectiveness of unlearning, knowledge retention, and target model's post-unlearning utility. We also detail the task-specific evaluation mechanism, results, and challenges. Our algorithms have achieved an aggregate score of 0.409 and 0.389 on the test set for 7B and 1B target models, respectively, demonstrating promising results in verifiable LLM unlearning. 

---
# KunlunBaize: LLM with Multi-Scale Convolution and Multi-Token Prediction Under TransformerX Framework 

**Authors**: Jiexiong Liu, Yixuan Chen, Yanqin Jia, Zhepeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04784)  

**Abstract**: Large language models have demonstrated remarkable performance across various tasks, yet they face challenges such as low computational efficiency, gradient vanishing, and difficulties in capturing complex feature interactions. To address these limitations, a novel framework has been proposed. This framework incorporates a learnable dense residual skip connection mechanism, a TransformerX module a transformer based component integrating multiscale convolution and adaptive activation functions and a multitoken prediction interaction module. The learnable dense residual connections enhance information flow and feature capture across layers. Within the TransformerX module, large convolutional kernels aggregate semantic information from extensive text segments, while smaller convolutions focus on local word order and syntactic structures. The adaptive activation function dynamically adjusts its parameters based on the semantic features of the input text, improving the model's ability to handle diverse semantic expressions and complex relationships. The multitoken prediction module boosts data utilization and accelerates inference by predicting multiple future tokens. These components significantly enhance the performance and efficiency of large language models. 

---
# Comparative Analysis Based on DeepSeek, ChatGPT, and Google Gemini: Features, Techniques, Performance, Future Prospects 

**Authors**: Anichur Rahman, Shahariar Hossain Mahir, Md Tanjum An Tashrif, Airin Afroj Aishi, Md Ahsan Karim, Dipanjali Kundu, Tanoy Debnath, Md. Abul Ala Moududi, MD. Zunead Abedin Eidmum  

**Link**: [PDF](https://arxiv.org/pdf/2503.04783)  

**Abstract**: Nowadays, DeepSeek, ChatGPT, and Google Gemini are the most trending and exciting Large Language Model (LLM) technologies for reasoning, multimodal capabilities, and general linguistic performance worldwide. DeepSeek employs a Mixture-of-Experts (MoE) approach, activating only the parameters most relevant to the task at hand, which makes it especially effective for domain-specific work. On the other hand, ChatGPT relies on a dense transformer model enhanced through reinforcement learning from human feedback (RLHF), and then Google Gemini actually uses a multimodal transformer architecture that integrates text, code, and images into a single framework. However, by using those technologies, people can be able to mine their desired text, code, images, etc, in a cost-effective and domain-specific inference. People may choose those techniques based on the best performance. In this regard, we offer a comparative study based on the DeepSeek, ChatGPT, and Gemini techniques in this research. Initially, we focus on their methods and materials, appropriately including the data selection criteria. Then, we present state-of-the-art features of DeepSeek, ChatGPT, and Gemini based on their applications. Most importantly, we show the technological comparison among them and also cover the dataset analysis for various applications. Finally, we address extensive research areas and future potential guidance regarding LLM-based AI research for the community. 

---
# AgroLLM: Connecting Farmers and Agricultural Practices through Large Language Models for Enhanced Knowledge Transfer and Practical Application 

**Authors**: Dinesh Jackson Samuel, Inna Skarga-Bandurova, David Sikolia, Muhammad Awais  

**Link**: [PDF](https://arxiv.org/pdf/2503.04788)  

**Abstract**: AgroLLM is an AI-powered chatbot designed to enhance knowledge-sharing and education in agriculture using Large Language Models (LLMs) and a Retrieval-Augmented Generation (RAG) framework. By using a comprehensive open-source agricultural database, AgroLLM provides accurate, contextually relevant responses while reducing incorrect information retrieval. The system utilizes the FAISS vector database for efficient similarity searches, ensuring rapid access to agricultural knowledge. A comparative study of three advanced models: Gemini 1.5 Flash, ChatGPT-4o Mini, and Mistral-7B-Instruct-v0.2 was conducted to evaluate performance across four key agricultural domains: Agriculture and Life Sciences, Agricultural Management, Agriculture and Forestry, and Agriculture Business. Key evaluation metrics included embedding quality, search efficiency, and response relevance. Results indicated that ChatGPT-4o Mini with RAG achieved the highest accuracy at 93%. Continuous feedback mechanisms enhance response quality, making AgroLLM a benchmark AI-driven educational tool for farmers, researchers, and professionals, promoting informed decision-making and improved agricultural practices. 

---
# DiMA: An LLM-Powered Ride-Hailing Assistant at DiDi 

**Authors**: Yansong Ning, Shuowei Cai, Wei Li, Jun Fang, Naiqiang Tan, Hua Chai, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04768)  

**Abstract**: On-demand ride-hailing services like DiDi, Uber, and Lyft have transformed urban transportation, offering unmatched convenience and flexibility. In this paper, we introduce DiMA, an LLM-powered ride-hailing assistant deployed in DiDi Chuxing. Its goal is to provide seamless ride-hailing services and beyond through a natural and efficient conversational interface under dynamic and complex spatiotemporal urban contexts. To achieve this, we propose a spatiotemporal-aware order planning module that leverages external tools for precise spatiotemporal reasoning and progressive order planning. Additionally, we develop a cost-effective dialogue system that integrates multi-type dialog repliers with cost-aware LLM configurations to handle diverse conversation goals and trade-off response quality and latency. Furthermore, we introduce a continual fine-tuning scheme that utilizes real-world interactions and simulated dialogues to align the assistant's behavior with human preferred decision-making processes. Since its deployment in the DiDi application, DiMA has demonstrated exceptional performance, achieving 93% accuracy in order planning and 92% in response generation during real-world interactions. Offline experiments further validate DiMA capabilities, showing improvements of up to 70.23% in order planning and 321.27% in response generation compared to three state-of-the-art agent frameworks, while reducing latency by $0.72\times$ to $5.47\times$. These results establish DiMA as an effective, efficient, and intelligent mobile assistant for ride-hailing services. 

---
# WinClick: GUI Grounding with Multimodal Large Language Models 

**Authors**: Zheng Hui, Yinheng Li, Dan zhao, Tianyi Chen, Colby Banbury, Kazuhito Koishida  

**Link**: [PDF](https://arxiv.org/pdf/2503.04730)  

**Abstract**: Graphical User Interface (GUI) tasks are vital for automating workflows such as software testing, user interface navigation. For users, the GUI is the most intuitive platform for interacting with a computer. Previous work identified a key challenge in developing visual GUI agents: GUI grounding - the ability to accurately locate screen elements based on instructions. However, most existing GUI agents rely on structured data formats like DOM or HTML files in training or inferencing, which are inaccessible across all applications, particular in a general desktop environments such as Windows OS. To address this, we introduce WinClick, a novel visual GUI agent developed in Windows platform. WinClick leverages screenshots to detect actionable regions. To overcome the challenge of GUI grounding, we enhance WinClick with GUI grounding pre-training and propose an LLM-based method for aligning GUI grounding data. Additionally, we introduce WinSpot, the first comprehensive benchmark for GUI grounding on Windows. Our experiments demonstrate that WinClick, combined with GUI grounding pre-training, significantly outperforms existing baselines, offering a scalable solution for GUI automation in desktop environments. WinSpot is publicly available at this https URL. 

---
# Leveraging Large Language Models For Optimized Item Categorization using UNSPSC Taxonomy 

**Authors**: Anmolika Singh, Yuhang Diao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04728)  

**Abstract**: Effective item categorization is vital for businesses, enabling the transformation of unstructured datasets into organized categories that streamline inventory management. Despite its importance, item categorization remains highly subjective and lacks a uniform standard across industries and businesses. The United Nations Standard Products and Services Code (UNSPSC) provides a standardized system for cataloguing inventory, yet employing UNSPSC categorizations often demands significant manual effort. This paper investigates the deployment of Large Language Models (LLMs) to automate the classification of inventory data into UNSPSC codes based on Item Descriptions. We evaluate the accuracy and efficiency of LLMs in categorizing diverse datasets, exploring their language processing capabilities and their potential as a tool for standardizing inventory classification. Our findings reveal that LLMs can substantially diminish the manual labor involved in item categorization while maintaining high accuracy, offering a scalable solution for businesses striving to enhance their inventory management practices. 

---
# Towards Anthropomorphic Conversational AI Part I: A Practical Framework 

**Authors**: Fei Wei, Yaliang Li, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.04787)  

**Abstract**: Large language models (LLMs), due to their advanced natural language capabilities, have seen significant success in applications where the user interface is usually a conversational artificial intelligence (AI) agent and engages the user through multi-round conversations. However, many scenarios require the agents to exhibit stronger social and conversational intelligence and demonstrate more human-like (anthropomorphic) reactions. This is an aspect that foundational LLMs have yet to fully address such that a single call of foundational models might be insufficient.
To bridge this gap, we propose a two-stage solution. In this work, we focus on the first stage, introducing a multi-module framework designed to replicate the key aspects of human intelligence involved in conversations. This framework comprises thinking modules for reasoning, resource modules for managing knowledge and external information, and response modules for generating contextually appropriate interactions. With all the modules cooperating, the framework would empower the agents to provide a better human-like conversation experience. In the second stage of our approach, these conversational data, after filtering and labeling, can serve as training and testing data for reinforcement learning, enabling AI to better capture human preferences. This stage is left for future work.
In our experiments, volunteers engaged in over 3000 rounds of conversation with the same AI character powered by a standalone LLM and our framework which integrates the same LLM. A separate group of evaluators rated the conversation samples, revealing that our framework significantly enhanced the social and conversational intelligence, even without fine-tuning the LLM. 

---
# MV-CLAM: Multi-View Molecular Interpretation with Cross-Modal Projection via Language Model 

**Authors**: Sumin Ha, Jun Hyeong Kim, Yinhua Piao, Sun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.04780)  

**Abstract**: Human expertise in chemistry and biomedicine relies on contextual molecular understanding, a capability that large language models (LLMs) can extend through fine-grained alignment between molecular structures and text. Recent multimodal learning advances focus on cross-modal alignment, but existing molecule-text models ignore complementary information in different molecular views and rely on single-view representations, limiting molecular understanding. Moreover, naïve multi-view alignment strategies face two challenges: (1) separate aligned spaces with inconsistent mappings between molecule and text embeddings, and that (2) existing loss objectives fail to preserve complementary information for fine-grained alignment. This can limit the LLM's ability to fully understand the molecular properties. To address these issues, we propose MV-CLAM, a novel framework that aligns multi-view molecular representations into a unified textual space using a multi-query transformer (MQ-Former). Our approach ensures cross-view consistency while a token-level contrastive loss preserves diverse molecular features across textual queries. MV-CLAM enhances molecular reasoning, improving retrieval and captioning accuracy. The source code of MV-CLAM is available in this https URL. 

---
# Pi-GPS: Enhancing Geometry Problem Solving by Unleashing the Power of Diagrammatic Information 

**Authors**: Junbo Zhao, Ting Zhang, Jiayu Sun, Mi Tian, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05543)  

**Abstract**: Geometry problem solving has garnered increasing attention due to its potential applications in intelligent education field. Inspired by the observation that text often introduces ambiguities that diagrams can clarify, this paper presents Pi-GPS, a novel framework that unleashes the power of diagrammatic information to resolve textual ambiguities, an aspect largely overlooked in prior research. Specifically, we design a micro module comprising a rectifier and verifier: the rectifier employs MLLMs to disambiguate text based on the diagrammatic context, while the verifier ensures the rectified output adherence to geometric rules, mitigating model hallucinations. Additionally, we explore the impact of LLMs in theorem predictor based on the disambiguated formal language. Empirical results demonstrate that Pi-GPS surpasses state-of-the-art models, achieving a nearly 10\% improvement on Geometry3K over prior neural-symbolic approaches. We hope this work highlights the significance of resolving textual ambiguity in multimodal mathematical reasoning, a crucial factor limiting performance. 

---
# Cognitive Bias Detection Using Advanced Prompt Engineering 

**Authors**: Frederic Lemieux, Aisha Behr, Clara Kellermann-Bryant, Zaki Mohammed  

**Link**: [PDF](https://arxiv.org/pdf/2503.05516)  

**Abstract**: Cognitive biases, systematic deviations from rationality in judgment, pose significant challenges in generating objective content. This paper introduces a novel approach for real-time cognitive bias detection in user-generated text using large language models (LLMs) and advanced prompt engineering techniques. The proposed system analyzes textual data to identify common cognitive biases such as confirmation bias, circular reasoning, and hidden assumption. By designing tailored prompts, the system effectively leverages LLMs' capabilities to both recognize and mitigate these biases, improving the quality of human-generated content (e.g., news, media, reports). Experimental results demonstrate the high accuracy of our approach in identifying cognitive biases, offering a valuable tool for enhancing content objectivity and reducing the risks of biased decision-making. 

---
# Evaluating open-source Large Language Models for automated fact-checking 

**Authors**: Nicolo' Fontana, Francesco Corso, Enrico Zuccolotto, Francesco Pierri  

**Link**: [PDF](https://arxiv.org/pdf/2503.05565)  

**Abstract**: The increasing prevalence of online misinformation has heightened the demand for automated fact-checking solutions. Large Language Models (LLMs) have emerged as potential tools for assisting in this task, but their effectiveness remains uncertain. This study evaluates the fact-checking capabilities of various open-source LLMs, focusing on their ability to assess claims with different levels of contextual information. We conduct three key experiments: (1) evaluating whether LLMs can identify the semantic relationship between a claim and a fact-checking article, (2) assessing models' accuracy in verifying claims when given a related fact-checking article, and (3) testing LLMs' fact-checking abilities when leveraging data from external knowledge sources such as Google and Wikipedia. Our results indicate that LLMs perform well in identifying claim-article connections and verifying fact-checked stories but struggle with confirming factual news, where they are outperformed by traditional fine-tuned models such as RoBERTa. Additionally, the introduction of external knowledge does not significantly enhance LLMs' performance, calling for more tailored approaches. Our findings highlight both the potential and limitations of LLMs in automated fact-checking, emphasizing the need for further refinements before they can reliably replace human fact-checkers. 

---
# Benchmarking LLMs in Recommendation Tasks: A Comparative Evaluation with Conventional Recommenders 

**Authors**: Qijiong Liu, Jieming Zhu, Lu Fan, Kun Wang, Hengchang Hu, Wei Guo, Yong Liu, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05493)  

**Abstract**: In recent years, integrating large language models (LLMs) into recommender systems has created new opportunities for improving recommendation quality. However, a comprehensive benchmark is needed to thoroughly evaluate and compare the recommendation capabilities of LLMs with traditional recommender systems. In this paper, we introduce RecBench, which systematically investigates various item representation forms (including unique identifier, text, semantic embedding, and semantic identifier) and evaluates two primary recommendation tasks, i.e., click-through rate prediction (CTR) and sequential recommendation (SeqRec). Our extensive experiments cover up to 17 large models and are conducted across five diverse datasets from fashion, news, video, books, and music domains. Our findings indicate that LLM-based recommenders outperform conventional recommenders, achieving up to a 5% AUC improvement in the CTR scenario and up to a 170% NDCG@10 improvement in the SeqRec scenario. However, these substantial performance gains come at the expense of significantly reduced inference efficiency, rendering the LLM-as-RS paradigm impractical for real-time recommendation environments. We aim for our findings to inspire future research, including recommendation-specific model acceleration methods. We will release our code, data, configurations, and platform to enable other researchers to reproduce and build upon our experimental results. 

---
# WritingBench: A Comprehensive Benchmark for Generative Writing 

**Authors**: Yuning Wu, Jiahao Mei, Ming Yan, Chenliang Li, SHaopeng Lai, Yuran Ren, Zijia Wang, Ji Zhang, Mengyue Wu, Qin Jin, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05244)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly enhanced text generation capabilities, yet evaluating their performance in generative writing remains a challenge. Existing benchmarks primarily focus on generic text generation or limited in writing tasks, failing to capture the diverse requirements of high-quality written contents across various domains. To bridge this gap, we present WritingBench, a comprehensive benchmark designed to evaluate LLMs across 6 core writing domains and 100 subdomains, encompassing creative, persuasive, informative, and technical writing. We further propose a query-dependent evaluation framework that empowers LLMs to dynamically generate instance-specific assessment criteria. This framework is complemented by a fine-tuned critic model for criteria-aware scoring, enabling evaluations in style, format and length. The framework's validity is further demonstrated by its data curation capability, which enables 7B-parameter models to approach state-of-the-art (SOTA) performance. We open-source the benchmark, along with evaluation tools and modular framework components, to advance the development of LLMs in writing. 

---
# Shifting Perspectives: Steering Vector Ensembles for Robust Bias Mitigation in LLMs 

**Authors**: Zara Siddique, Irtaza Khalid, Liam D. Turner, Luis Espinosa-Anke  

**Link**: [PDF](https://arxiv.org/pdf/2503.05371)  

**Abstract**: We present a novel approach to bias mitigation in large language models (LLMs) by applying steering vectors to modify model activations in forward passes. We employ Bayesian optimization to systematically identify effective contrastive pair datasets across nine bias axes. When optimized on the BBQ dataset, our individually tuned steering vectors achieve average improvements of 12.2%, 4.7%, and 3.2% over the baseline for Mistral, Llama, and Qwen, respectively. Building on these promising results, we introduce Steering Vector Ensembles (SVE), a method that averages multiple individually optimized steering vectors, each targeting a specific bias axis such as age, race, or gender. By leveraging their collective strength, SVE outperforms individual steering vectors in both bias reduction and maintaining model performance. The work presents the first systematic investigation of steering vectors for bias mitigation, and we demonstrate that SVE is a powerful and computationally efficient strategy for reducing bias in LLMs, with broader implications for enhancing AI safety. 

---
# R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning 

**Authors**: Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.05592)  

**Abstract**: Existing Large Reasoning Models (LRMs) have shown the potential of reinforcement learning (RL) to enhance the complex reasoning capabilities of Large Language Models~(LLMs). While they achieve remarkable performance on challenging tasks such as mathematics and coding, they often rely on their internal knowledge to solve problems, which can be inadequate for time-sensitive or knowledge-intensive questions, leading to inaccuracies and hallucinations. To address this, we propose \textbf{R1-Searcher}, a novel two-stage outcome-based RL approach designed to enhance the search capabilities of LLMs. This method allows LLMs to autonomously invoke external search systems to access additional knowledge during the reasoning process. Our framework relies exclusively on RL, without requiring process rewards or distillation for a cold start. % effectively generalizing to out-of-domain datasets and supporting both Base and Instruct models. Our experiments demonstrate that our method significantly outperforms previous strong RAG methods, even when compared to the closed-source GPT-4o-mini. 

---
# AutoTestForge: A Multidimensional Automated Testing Framework for Natural Language Processing Models 

**Authors**: Hengrui Xing, Cong Tian, Liang Zhao, Zhi Ma, WenSheng Wang, Nan Zhang, Chao Huang, Zhenhua Duan  

**Link**: [PDF](https://arxiv.org/pdf/2503.05102)  

**Abstract**: In recent years, the application of behavioral testing in Natural Language Processing (NLP) model evaluation has experienced a remarkable and substantial growth. However, the existing methods continue to be restricted by the requirements for manual labor and the limited scope of capability assessment. To address these limitations, we introduce AutoTestForge, an automated and multidimensional testing framework for NLP models in this paper. Within AutoTestForge, through the utilization of Large Language Models (LLMs) to automatically generate test templates and instantiate them, manual involvement is significantly reduced. Additionally, a mechanism for the validation of test case labels based on differential testing is implemented which makes use of a multi-model voting system to guarantee the quality of test cases. The framework also extends the test suite across three dimensions, taxonomy, fairness, and robustness, offering a comprehensive evaluation of the capabilities of NLP models. This expansion enables a more in-depth and thorough assessment of the models, providing valuable insights into their strengths and weaknesses. A comprehensive evaluation across sentiment analysis (SA) and semantic textual similarity (STS) tasks demonstrates that AutoTestForge consistently outperforms existing datasets and testing tools, achieving higher error detection rates (an average of $30.89\%$ for SA and $34.58\%$ for STS). Moreover, different generation strategies exhibit stable effectiveness, with error detection rates ranging from $29.03\% - 36.82\%$. 

---
# Invisible Walls in Cities: Leveraging Large Language Models to Predict Urban Segregation Experience with Social Media Content 

**Authors**: Bingbing Fan, Lin Chen, Songwei Li, Jian Yuan, Fengli Xu, Pan Hui, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04773)  

**Abstract**: Understanding experienced segregation in urban daily life is crucial for addressing societal inequalities and fostering inclusivity. The abundance of user-generated reviews on social media encapsulates nuanced perceptions and feelings associated with different places, offering rich insights into segregation. However, leveraging this data poses significant challenges due to its vast volume, ambiguity, and confluence of diverse perspectives. To tackle these challenges, we propose using Large Language Models (LLMs) to automate online review mining for segregation prediction. We design a Reflective LLM Coder to digest social media content into insights consistent with real-world feedback, and eventually produce a codebook capturing key dimensions that signal segregation experience, such as cultural resonance and appeal, accessibility and convenience, and community engagement and local involvement. Guided by the codebook, LLMs can generate both informative review summaries and ratings for segregation prediction. Moreover, we design a REasoning-and-EMbedding (RE'EM) framework, which combines the reasoning and embedding capabilities of language models to integrate multi-channel features for segregation prediction. Experiments on real-world data demonstrate that our framework greatly improves prediction accuracy, with a 22.79% elevation in R2 and a 9.33% reduction in MSE. The derived codebook is generalizable across three different cities, consistently improving prediction this http URL, our user study confirms that the codebook-guided summaries provide cognitive gains for human participants in perceiving POIs' social this http URL study marks an important step toward understanding implicit social barriers and inequalities, demonstrating the great potential of promoting social inclusiveness with AI. 

---
# A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models 

**Authors**: Dong Shu, Xuansheng Wu, Haiyan Zhao, Daking Rai, Ziyu Yao, Ninghao Liu, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2503.05613)  

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, yet their internal mechanisms remain largely opaque. Recently, mechanistic interpretability has attracted significant attention from the research community as a means to understand the inner workings of LLMs. Among various mechanistic interpretability approaches, Sparse Autoencoders (SAEs) have emerged as a particularly promising method due to their ability to disentangle the complex, superimposed features within LLMs into more interpretable components. This paper presents a comprehensive examination of SAEs as a promising approach to interpreting and understanding LLMs. We provide a systematic overview of SAE principles, architectures, and applications specifically tailored for LLM analysis, covering theoretical foundations, implementation strategies, and recent developments in sparsity mechanisms. We also explore how SAEs can be leveraged to explain the internal workings of LLMs, steer model behaviors in desired directions, and develop more transparent training methodologies for future models. Despite the challenges that remain around SAE implementation and scaling, they continue to provide valuable tools for understanding the internal mechanisms of large language models. 

---
# Wanda++: Pruning Large Language Models via Regional Gradients 

**Authors**: Yifan Yang, Kai Zhen, Bhavana Ganesh, Aram Galstyan, Goeric Huybrechts, Markus Müller, Jonas M. Kübler, Rupak Vignesh Swaminathan, Athanasios Mouchtaris, Sravan Babu Bodapati, Nathan Susanj, Zheng Zhang, Jack FitzGerald, Abhishek Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.04992)  

**Abstract**: Large Language Models (LLMs) pruning seeks to remove unimportant weights for inference speedup with minimal performance impact. However, existing methods often suffer from performance loss without full-model sparsity-aware fine-tuning. This paper presents Wanda++, a novel pruning framework that outperforms the state-of-the-art methods by utilizing decoder-block-level \textbf{regional} gradients. Specifically, Wanda++ improves the pruning score with regional gradients for the first time and proposes an efficient regional optimization method to minimize pruning-induced output discrepancies between the dense and sparse decoder output. Notably, Wanda++ improves perplexity by up to 32\% over Wanda in the language modeling task and generalizes effectively to downstream tasks. Further experiments indicate our proposed method is orthogonal to sparsity-aware fine-tuning, where Wanda++ can be combined with LoRA fine-tuning to achieve a similar perplexity improvement as the Wanda method. The proposed method is lightweight, pruning a 7B LLaMA model in under 10 minutes on a single NVIDIA H100 GPU. 

---
# Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks 

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.04833)  

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems. 

---
# Self-Evolved Preference Optimization for Enhancing Mathematical Reasoning in Small Language Models 

**Authors**: Joykirat Singh, Tanmoy Chakraborty, Akshay Nambi  

**Link**: [PDF](https://arxiv.org/pdf/2503.04813)  

**Abstract**: Large language models (LLMs) have significantly improved their reasoning capabilities; however, they still struggle with complex multi-step mathematical problem-solving due to error propagation, lack of self-correction, and limited adaptability to diverse reasoning styles. Existing methods rely on static fine-tuning or prompt engineering, which fail to generalize across problem complexities, while the scarcity of high-quality preference data further hinders reliable reasoning.
We introduce SPHERE, a self-evolving data generation pipeline that enhances reasoning in small language models (SLMs) by iteratively generating, correcting, and diversifying reasoning chains. SPHERE operates in three stages: (i) Self-Generation, where the model autonomously constructs problem-solving steps; (ii) Self-Correction, enabling it to identify and rectify errors; and (iii) Diversity Induction, improving robustness through multiple valid reasoning trajectories. This self-evolution mechanism strengthens mathematical reasoning and enhances model reliability. Evaluations on MATH 500, GSM8K, AIME, AMC, and Olympiad show that SPHERE-trained models achieve significant gains over their base versions and match/surpass GPT-4o on certain benchmarks. Our findings demonstrate that self-evolving models can close the reasoning gap between SLMs and state-of-the-art LLMs, making mathematical AI more reliable, scalable, and efficient. 

---
# What do Large Language Models Say About Animals? Investigating Risks of Animal Harm in Generated Text 

**Authors**: Arturs Kanepajs, Aditi Basu, Sankalpa Ghose, Constance Li, Akshat Mehta, Ronak Mehta, Samuel David Tucker-Davis, Eric Zhou, Bob Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2503.04804)  

**Abstract**: As machine learning systems become increasingly embedded in human society, their impact on the natural world continues to escalate. Technical evaluations have addressed a variety of potential harms from large language models (LLMs) towards humans and the environment, but there is little empirical work regarding harms towards nonhuman animals. Following the growing recognition of animal protection in regulatory and ethical AI frameworks, we present the Animal Harm Assessment (AHA), a novel evaluation of risks of animal harm in LLM-generated text. Our dataset comprises 1,850 curated questions from Reddit post titles and 2,500 synthetic questions based on 50 animal categories (e.g., cats, reptiles) and 50 ethical scenarios, with further 70-30 publi-private split. Scenarios include open-ended questions about how to treat animals, practical scenarios with potential animal harm, and willingness-to-pay measures for the prevention of animal harm. Using the LLM-as-a-judge framework, answers are evaluated for their potential to increase or decrease harm, and evaluations are debiased for the tendency to judge their own outputs more favorably. We show that AHA produces meaningful evaluation results when applied to frontier LLMs, revealing significant differences between models, animal categories, scenarios, and subreddits. We conclude with future directions for technical research and the challenges of building evaluations on complex social and moral topics. 

---
# Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs 

**Authors**: Ling Team, Binwei Zeng, Chao Huang, Chao Zhang, Changxin Tian, Cong Chen, Dingnan Jin, Feng Yu, Feng Zhu, Feng Yuan, Fakang Wang, Gangshan Wang, Guangyao Zhai, Haitao Zhang, Huizhong Li, Jun Zhou, Jia Liu, Junpeng Fang, Junjie Ou, Jun Hu, Ji Luo, Ji Zhang, Jian Liu, Jian Sha, Jianxue Qian, Jiewei Wu, Junping Zhao, Jianguo Li, Jubao Feng, Jingchao Di, Junming Xu, Jinghua Yao, Kuan Xu, Kewei Du, Longfei Li, Lei Liang, Lu Yu, Li Tang, Lin Ju, Peng Xu, Qing Cui, Song Liu, Shicheng Li, Shun Song, Song Yan, Tengwei Cai, Tianyi Chen, Ting Guo, Ting Huang, Tao Feng, Tao Wu, Wei Wu, Xiaolu Zhang, Xueming Yang, Xin Zhao, Xiaobo Hu, Xin Lin, Yao Zhao, Yilong Wang, Yongzhen Guo, Yuanyuan Wang, Yue Yang, Yang Cao, Yuhao Fu, Yi Xiong, Yanzhe Li, Zhe Li, Zhiqiang Zhang, Ziqi Liu, Zhaoxin Huan, Zujie Wen, Zhenhang Sun, Zhuoxuan Du, Zhengyu He  

**Link**: [PDF](https://arxiv.org/pdf/2503.05139)  

**Abstract**: In this technical report, we tackle the challenges of training large-scale Mixture of Experts (MoE) models, focusing on overcoming cost inefficiency and resource limitations prevalent in such systems. To address these issues, we present two differently sized MoE large language models (LLMs), namely Ling-Lite and Ling-Plus (referred to as "Bailing" in Chinese, spelled Bǎilíng in Pinyin). Ling-Lite contains 16.8 billion parameters with 2.75 billion activated parameters, while Ling-Plus boasts 290 billion parameters with 28.8 billion activated parameters. Both models exhibit comparable performance to leading industry benchmarks. This report offers actionable insights to improve the efficiency and accessibility of AI development in resource-constrained settings, promoting more scalable and sustainable technologies. Specifically, to reduce training costs for large-scale MoE models, we propose innovative methods for (1) optimization of model architecture and training processes, (2) refinement of training anomaly handling, and (3) enhancement of model evaluation efficiency. Additionally, leveraging high-quality data generated from knowledge graphs, our models demonstrate superior capabilities in tool use compared to other models. Ultimately, our experimental findings demonstrate that a 300B MoE LLM can be effectively trained on lower-performance devices while achieving comparable performance to models of a similar scale, including dense and MoE models. Compared to high-performance devices, utilizing a lower-specification hardware system during the pre-training phase demonstrates significant cost savings, reducing computing costs by approximately 20%. The models can be accessed at this https URL. 

---
# What can large language models do for sustainable food? 

**Authors**: Anna T. Thomas, Adam Yee, Andrew Mayne, Maya B. Mathur, Dan Jurafsky, Kristina Gligorić  

**Link**: [PDF](https://arxiv.org/pdf/2503.04734)  

**Abstract**: Food systems are responsible for a third of human-caused greenhouse gas emissions. We investigate what Large Language Models (LLMs) can contribute to reducing the environmental impacts of food production. We define a typology of design and prediction tasks based on the sustainable food literature and collaboration with domain experts, and evaluate six LLMs on four tasks in our typology. For example, for a sustainable protein design task, food science experts estimated that collaboration with an LLM can reduce time spent by 45% on average, compared to 22% for collaboration with another expert human food scientist. However, for a sustainable menu design task, LLMs produce suboptimal solutions when instructed to consider both human satisfaction and climate impacts. We propose a general framework for integrating LLMs with combinatorial optimization to improve reasoning capabilities. Our approach decreases emissions of food choices by 79% in a hypothetical restaurant while maintaining participants' satisfaction with their set of choices. Our results demonstrate LLMs' potential, supported by optimization techniques, to accelerate sustainable food development and adoption. 

---
# A Survey of Large Language Model Empowered Agents for Recommendation and Search: Towards Next-Generation Information Retrieval 

**Authors**: Yu Zhang, Shutong Qiao, Jiaqi Zhang, Tzu-Heng Lin, Chen Gao, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05659)  

**Abstract**: Information technology has profoundly altered the way humans interact with information. The vast amount of content created, shared, and disseminated online has made it increasingly difficult to access relevant information. Over the past two decades, search and recommendation systems (collectively referred to as information retrieval systems) have evolved significantly to address these challenges. Recent advances in large language models (LLMs) have demonstrated capabilities that surpass human performance in various language-related tasks and exhibit general understanding, reasoning, and decision-making abilities. This paper explores the transformative potential of large language model agents in enhancing search and recommendation systems. We discuss the motivations and roles of LLM agents, and establish a classification framework to elaborate on the existing research. We highlight the immense potential of LLM agents in addressing current challenges in search and recommendation, providing insights into future research directions. This paper is the first to systematically review and classify the research on LLM agents in these domains, offering a novel perspective on leveraging this advanced AI technology for information retrieval. To help understand the existing works, we list the existing papers on agent-based simulation with large language models at this link: this https URL. 

---
# Ontology Generation using Large Language Models 

**Authors**: Anna Sofia Lippolis, Mohammad Javad Saeedizade, Robin Keskisärkkä, Sara Zuppiroli, Miguel Ceriani, Aldo Gangemi, Eva Blomqvist, Andrea Giovanni Nuzzolese  

**Link**: [PDF](https://arxiv.org/pdf/2503.05388)  

**Abstract**: The ontology engineering process is complex, time-consuming, and error-prone, even for experienced ontology engineers. In this work, we investigate the potential of Large Language Models (LLMs) to provide effective OWL ontology drafts directly from ontological requirements described using user stories and competency questions. Our main contribution is the presentation and evaluation of two new prompting techniques for automated ontology development: Memoryless CQbyCQ and Ontogenia. We also emphasize the importance of three structural criteria for ontology assessment, alongside expert qualitative evaluation, highlighting the need for a multi-dimensional evaluation in order to capture the quality and usability of the generated ontologies. Our experiments, conducted on a benchmark dataset of ten ontologies with 100 distinct CQs and 29 different user stories, compare the performance of three LLMs using the two prompting techniques. The results demonstrate improvements over the current state-of-the-art in LLM-supported ontology engineering. More specifically, the model OpenAI o1-preview with Ontogenia produces ontologies of sufficient quality to meet the requirements of ontology engineers, significantly outperforming novice ontology engineers in modelling ability. However, we still note some common mistakes and variability of result quality, which is important to take into account when using LLMs for ontology authoring support. We discuss these limitations and propose directions for future research. 

---
# Grammar-Based Code Representation: Is It a Worthy Pursuit for LLMs? 

**Authors**: Qingyuan Liang, Zhao Zhang, Zeyu Sun, Zheng Lin, Qi Luo, Yueyi Xiao, Yizhou Chen, Yuqun Zhang, Haotian Zhang, Lu Zhang, Bin Chen, Yingfei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.05507)  

**Abstract**: Grammar serves as a cornerstone in programming languages and software engineering, providing frameworks to define the syntactic space and program structure. Existing research demonstrates the effectiveness of grammar-based code representations in small-scale models, showing their ability to reduce syntax errors and enhance performance. However, as language models scale to the billion level or beyond, syntax-level errors become rare, making it unclear whether grammar information still provides performance benefits. To explore this, we develop a series of billion-scale GrammarCoder models, incorporating grammar rules in the code generation process. Experiments on HumanEval (+) and MBPP (+) demonstrate a notable improvement in code generation accuracy. Further analysis shows that grammar-based representations enhance LLMs' ability to discern subtle code differences, reducing semantic errors caused by minor variations. These findings suggest that grammar-based code representations remain valuable even in billion-scale models, not only by maintaining syntax correctness but also by improving semantic differentiation. 

---
# The Society of HiveMind: Multi-Agent Optimization of Foundation Model Swarms to Unlock the Potential of Collective Intelligence 

**Authors**: Noah Mamie, Susie Xi Rao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05473)  

**Abstract**: Multi-agent systems address issues of accessibility and scalability of artificial intelligence (AI) foundation models, which are often represented by large language models. We develop a framework - the "Society of HiveMind" (SOHM) - that orchestrates the interaction between multiple AI foundation models, imitating the observed behavior of animal swarms in nature by following modern evolutionary theories. On the one hand, we find that the SOHM provides a negligible benefit on tasks that mainly require real-world knowledge. On the other hand, we remark a significant improvement on tasks that require intensive logical reasoning, indicating that multi-agent systems are capable of increasing the reasoning capabilities of the collective compared to the individual agents. Our findings demonstrate the potential of combining a multitude of diverse AI foundation models to form an artificial swarm intelligence capable of self-improvement through interactions with a given environment. 

---
# LLM-based Iterative Approach to Metamodeling in Automotive 

**Authors**: Nenad Petrovic, Fengjunjie Pan, Vahid Zolfaghari, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.05449)  

**Abstract**: In this paper, we introduce an automated approach to domain-specific metamodel construction relying on Large Language Model (LLM). The main focus is adoption in automotive domain. As outcome, a prototype was implemented as web service using Python programming language, while OpenAI's GPT-4o was used as the underlying LLM. Based on the initial experiments, this approach successfully constructs Ecore metamodel based on set of automotive requirements and visualizes it making use of PlantUML notation, so human experts can provide feedback in order to refine the result. Finally, locally deployable solution is also considered, including the limitations and additional steps required. 

---
# Static Program Analysis Guided LLM Based Unit Test Generation 

**Authors**: Sujoy Roychowdhury, Giriprasad Sridhara, A K Raghavan, Joy Bose, Sourav Mazumdar, Hamender Singh, Srinivasan Bajji Sugumaran, Ricardo Britto  

**Link**: [PDF](https://arxiv.org/pdf/2503.05394)  

**Abstract**: We describe a novel approach to automating unit test generation for Java methods using large language models (LLMs). Existing LLM-based approaches rely on sample usage(s) of the method to test (focal method) and/or provide the entire class of the focal method as input prompt and context. The former approach is often not viable due to the lack of sample usages, especially for newly written focal methods. The latter approach does not scale well enough; the bigger the complexity of the focal method and larger associated class, the harder it is to produce adequate test code (due to factors such as exceeding the prompt and context lengths of the underlying LLM). We show that augmenting prompts with \emph{concise} and \emph{precise} context information obtained by program analysis %of the focal method increases the effectiveness of generating unit test code through LLMs. We validate our approach on a large commercial Java project and a popular open-source Java project. 

---
# PromptPex: Automatic Test Generation for Language Model Prompts 

**Authors**: Reshabh K Sharma, Jonathan De Halleux, Shraddha Barke, Benjamin Zorn  

**Link**: [PDF](https://arxiv.org/pdf/2503.05070)  

**Abstract**: Large language models (LLMs) are being used in many applications and prompts for these models are integrated into software applications as code-like artifacts. These prompts behave much like traditional software in that they take inputs, generate outputs, and perform some specific function. However, prompts differ from traditional code in many ways and require new approaches to ensure that they are robust. For example, unlike traditional software the output of a prompt depends on the AI model that interprets it. Also, while natural language prompts are easy to modify, the impact of updates is harder to predict. New approaches to testing, debugging, and modifying prompts with respect to the model running them are required.
To address some of these issues, we developed PromptPex, an LLM-based tool to automatically generate and evaluate unit tests for a given prompt. PromptPex extracts input and output specifications from a prompt and uses them to generate diverse, targeted, and valid unit tests. These tests are instrumental in identifying regressions when a prompt is changed and also serve as a tool to understand how prompts are interpreted by different models. We use PromptPex to generate tests for eight benchmark prompts and evaluate the quality of the generated tests by seeing if they can cause each of four diverse models to produce invalid output. PromptPex consistently creates tests that result in more invalid model outputs than a carefully constructed baseline LLM-based test generator. Furthermore, by extracting concrete specifications from the input prompt, PromptPex allows prompt writers to clearly understand and test specific aspects of their prompts. The source code of PromptPex is available at this https URL. 

---
# Quantifying the Relevance of Youth Research Cited in the US Policy Documents 

**Authors**: Miftahul Jannat Mokarrama, Hamed Alhoori  

**Link**: [PDF](https://arxiv.org/pdf/2503.04977)  

**Abstract**: In recent years, there has been a growing concern and emphasis on conducting research beyond academic or scientific research communities, benefiting society at large. A well-known approach to measuring the impact of research on society is enumerating its policy citation(s). Despite the importance of research in informing policy, there is no concrete evidence to suggest the research's relevance in cited policy documents. This is concerning because it may increase the possibility of evidence used in policy being manipulated by individual, social, or political biases that may lead to inappropriate, fragmented, or archaic research evidence in policy. Therefore, it is crucial to identify the degree of relevance between research articles and citing policy documents. In this paper, we examined the scale of contextual relevance of youth-focused research in the referenced US policy documents using natural language processing techniques, state-of-the-art pre-trained Large Language Models (LLMs), and statistical analysis. Our experiments and analysis concluded that youth-related research articles that get US policy citations are mostly relevant to the citing policy documents. 

---
# A Comprehensive LLM-powered Framework for Driving Intelligence Evaluation 

**Authors**: Shanhe You, Xuewen Luo, Xinhe Liang, Jiashu Yu, Chen Zheng, Jiangtao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.05164)  

**Abstract**: Evaluation methods for autonomous driving are crucial for algorithm optimization. However, due to the complexity of driving intelligence, there is currently no comprehensive evaluation method for the level of autonomous driving intelligence. In this paper, we propose an evaluation framework for driving behavior intelligence in complex traffic environments, aiming to fill this gap. We constructed a natural language evaluation dataset of human professional drivers and passengers through naturalistic driving experiments and post-driving behavior evaluation interviews. Based on this dataset, we developed an LLM-powered driving evaluation framework. The effectiveness of this framework was validated through simulated experiments in the CARLA urban traffic simulator and further corroborated by human assessment. Our research provides valuable insights for evaluating and designing more intelligent, human-like autonomous driving agents. The implementation details of the framework and detailed information about the dataset can be found at Github. 

---
# Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference 

**Authors**: Thanh Le-Cong, Bach Le, Toby Murray  

**Link**: [PDF](https://arxiv.org/pdf/2503.04779)  

**Abstract**: Large Language Models (LLMs) are increasingly being used to automate programming tasks. Yet, LLMs' capabilities in reasoning about program semantics are still inadequately studied, leaving significant potential for further exploration. This paper introduces FormalBench, a comprehensive benchmark designed to evaluate LLMs' reasoning abilities on program semantics, particularly via the task of synthesizing formal program specifications to assist verifying program correctness. This task requires both comprehensive reasoning over all possible program executions (i.e., \textit{completeness}) and the generation of precise, syntactically correct expressions that adhere to formal syntax and semantics (i.e., \textit{consistency}). Using this benchmark, we evaluated the ability of LLMs in synthesizing consistent and complete specifications. Our findings show that LLMs perform well with simple control flows but struggle with more complex structures, especially loops, even with advanced prompting. Additionally, LLMs exhibit limited robustness against semantic-preserving transformations. We also highlight common failure patterns and design self-repair prompts, improving success rates by 25%. 

---
# Generating Millions Of Lean Theorems With Proofs By Exploring State Transition Graphs 

**Authors**: David Yin, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04772)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant potential in generating mathematical proofs. However, a persistent challenge is that LLMs occasionally make mistakes, while even a minor mistake can invalidate an entire proof. Proof assistants like Lean offer a great remedy. They are designed for verifying each step of a proof in a formal language, and in recent years researchers have created AI models to generate proofs in their languages. However, the scarcity of large-scale datasets of Lean proofs restrict the performance of such Automated Theorem Proving (ATP) models.
We developed LeanNavigator, a novel method for generating a large-scale dataset of Lean theorems and proofs by finding new ways to prove existing Lean theorems. By leveraging an interactive Lean client and an efficient method for proof step generation, LeanNavigator efficiently produces new theorems with corresponding proofs. Applying this approach to Mathlib4, we generated 4.7 million theorems totaling 1 billion tokens, surpassing previous datasets by more than an order of magnitude. Using this extensive dataset, we trained an AI model that outperforms the state-of-the-art ReProver model in theorem-proving tasks. These results confirm our hypothesis and demonstrate the critical role of large datasets in improving the performance of automated theorem provers. 

---
