# TRACE Back from the Future: A Probabilistic Reasoning Approach to Controllable Language Generation 

**Authors**: Gwen Yidou Weng, Benjie Wang, Guy Van den Broeck  

**Link**: [PDF](https://arxiv.org/pdf/2504.18535)  

**Abstract**: As large language models (LMs) advance, there is an increasing need to control their outputs to align with human values (e.g., detoxification) or desired attributes (e.g., personalization, topic). However, autoregressive models focus on next-token predictions and struggle with global properties that require looking ahead. Existing solutions either tune or post-train LMs for each new attribute - expensive and inflexible - or approximate the Expected Attribute Probability (EAP) of future sequences by sampling or training, which is slow and unreliable for rare attributes. We introduce TRACE (Tractable Probabilistic Reasoning for Adaptable Controllable gEneration), a novel framework that efficiently computes EAP and adapts to new attributes through tractable probabilistic reasoning and lightweight control. TRACE distills a Hidden Markov Model (HMM) from an LM and pairs it with a small classifier to estimate attribute probabilities, enabling exact EAP computation over the HMM's predicted futures. This EAP is then used to reweigh the LM's next-token probabilities for globally compliant continuations. Empirically, TRACE achieves state-of-the-art results in detoxification with only 10% decoding overhead, adapts to 76 low-resource personalized LLMs within seconds, and seamlessly extends to composite attributes. 

---
# Investigating Co-Constructive Behavior of Large Language Models in Explanation Dialogues 

**Authors**: Leandra Fichtel, Maximilian Spliethöver, Eyke Hüllermeier, Patricia Jimenez, Nils Klowait, Stefan Kopp, Axel-Cyrille Ngonga Ngomo, Amelie Robrecht, Ingrid Scharlau, Lutz Terfloth, Anna-Lisa Vollmer, Henning Wachsmuth  

**Link**: [PDF](https://arxiv.org/pdf/2504.18483)  

**Abstract**: The ability to generate explanations that are understood by explainees is the quintessence of explainable artificial intelligence. Since understanding depends on the explainee's background and needs, recent research has focused on co-constructive explanation dialogues, where the explainer continuously monitors the explainee's understanding and adapts explanations dynamically. We investigate the ability of large language models (LLMs) to engage as explainers in co-constructive explanation dialogues. In particular, we present a user study in which explainees interact with LLMs, of which some have been instructed to explain a predefined topic co-constructively. We evaluate the explainees' understanding before and after the dialogue, as well as their perception of the LLMs' co-constructive behavior. Our results indicate that current LLMs show some co-constructive behaviors, such as asking verification questions, that foster the explainees' engagement and can improve understanding of a topic. However, their ability to effectively monitor the current understanding and scaffold the explanations accordingly remains limited. 

---
# Generative Induction of Dialogue Task Schemas with Streaming Refinement and Simulated Interactions 

**Authors**: James D. Finch, Yasasvi Josyula, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.18474)  

**Abstract**: In task-oriented dialogue (TOD) systems, Slot Schema Induction (SSI) is essential for automatically identifying key information slots from dialogue data without manual intervention. This paper presents a novel state-of-the-art (SoTA) approach that formulates SSI as a text generation task, where a language model incrementally constructs and refines a slot schema over a stream of dialogue data. To develop this approach, we present a fully automatic LLM-based TOD simulation method that creates data with high-quality state labels for novel task domains. Furthermore, we identify issues in SSI evaluation due to data leakage and poor metric alignment with human judgment. We resolve these by creating new evaluation data using our simulation method with human guidance and correction, as well as designing improved evaluation metrics. These contributions establish a foundation for future SSI research and advance the SoTA in dialogue understanding and system development. 

---
# Fast-Slow Thinking for Large Vision-Language Model Reasoning 

**Authors**: Wenyi Xiao, Leilei Gan, Weilong Dai, Wanggui He, Ziwei Huang, Haoyuan Li, Fangxun Shu, Zhelun Yu, Peng Zhang, Hao Jiang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18458)  

**Abstract**: Recent advances in large vision-language models (LVLMs) have revealed an \textit{overthinking} phenomenon, where models generate verbose reasoning across all tasks regardless of questions. To address this issue, we present \textbf{FAST}, a novel \textbf{Fa}st-\textbf{S}low \textbf{T}hinking framework that dynamically adapts reasoning depth based on question characteristics. Through empirical analysis, we establish the feasibility of fast-slow thinking in LVLMs by investigating how response length and data distribution affect performance. We develop FAST-GRPO with three components: model-based metrics for question characterization, an adaptive thinking reward mechanism, and difficulty-aware KL regularization. Experiments across seven reasoning benchmarks demonstrate that FAST achieves state-of-the-art accuracy with over 10\% relative improvement compared to the base model, while reducing token usage by 32.7-67.3\% compared to previous slow-thinking approaches, effectively balancing reasoning length and accuracy. 

---
# PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts 

**Authors**: Yiming Wang, Pei Zhang, Jialong Tang, Haoran Wei, Baosong Yang, Rui Wang, Chenshu Sun, Feitong Sun, Jiran Zhang, Junxuan Wu, Qiqian Cang, Yichang Zhang, Fei Huang, Junyang Lin, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.18428)  

**Abstract**: In this paper, we introduce PolyMath, a multilingual mathematical reasoning benchmark covering 18 languages and 4 easy-to-hard difficulty levels. Our benchmark ensures difficulty comprehensiveness, language diversity, and high-quality translation, making it a highly discriminative multilingual mathematical benchmark in the era of reasoning LLMs. We conduct a comprehensive evaluation for advanced LLMs and find that even Deepseek-R1-671B and Qwen-QwQ-32B, achieve only 43.4 and 41.8 benchmark scores, with less than 30% accuracy under the highest level. From a language perspective, our benchmark reveals several key challenges of LLMs in multilingual reasoning: (1) Reasoning performance varies widely across languages for current LLMs; (2) Input-output language consistency is low in reasoning LLMs and may be correlated with performance; (3) The thinking length differs significantly by language for current LLMs. Additionally, we demonstrate that controlling the output language in the instructions has the potential to affect reasoning performance, especially for some low-resource languages, suggesting a promising direction for improving multilingual capabilities in LLMs. 

---
# BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs 

**Authors**: Hongyu Wang, Shuming Ma, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2504.18415)  

**Abstract**: Efficient deployment of 1-bit Large Language Models (LLMs) is hindered by activation outliers, which complicate quantization to low bit-widths. We introduce BitNet v2, a novel framework enabling native 4-bit activation quantization for 1-bit LLMs. To tackle outliers in attention and feed-forward network activations, we propose H-BitLinear, a module applying an online Hadamard transformation prior to activation quantization. This transformation smooths sharp activation distributions into more Gaussian-like forms, suitable for low-bit representation. Experiments show BitNet v2 trained from scratch with 8-bit activations matches BitNet b1.58 performance. Crucially, BitNet v2 achieves minimal performance degradation when trained with native 4-bit activations, significantly reducing memory footprint and computational cost for batched inference. 

---
# Expressing stigma and inappropriate responses prevents LLMs from safely replacing mental health providers 

**Authors**: Jared Moore, Declan Grabb, William Agnew, Kevin Klyman, Stevie Chancellor, Desmond C. Ong, Nick Haber  

**Link**: [PDF](https://arxiv.org/pdf/2504.18412)  

**Abstract**: Should a large language model (LLM) be used as a therapist? In this paper, we investigate the use of LLMs to *replace* mental health providers, a use case promoted in the tech startup and research space. We conduct a mapping review of therapy guides used by major medical institutions to identify crucial aspects of therapeutic relationships, such as the importance of a therapeutic alliance between therapist and client. We then assess the ability of LLMs to reproduce and adhere to these aspects of therapeutic relationships by conducting several experiments investigating the responses of current LLMs, such as `gpt-4o`. Contrary to best practices in the medical community, LLMs 1) express stigma toward those with mental health conditions and 2) respond inappropriately to certain common (and critical) conditions in naturalistic therapy settings -- e.g., LLMs encourage clients' delusional thinking, likely due to their sycophancy. This occurs even with larger and newer LLMs, indicating that current safety practices may not address these gaps. Furthermore, we note foundational and practical barriers to the adoption of LLMs as therapists, such as that a therapeutic alliance requires human characteristics (e.g., identity and stakes). For these reasons, we conclude that LLMs should not replace therapists, and we discuss alternative roles for LLMs in clinical therapy. 

---
# HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding? 

**Authors**: Yusen Zhang, Wenliang Zheng, Aashrith Madasu, Peng Shi, Ryo Kamoi, Hao Zhou, Zhuoyang Zou, Shu Zhao, Sarkar Snigdha Sarathi Das, Vipul Gupta, Xiaoxin Lu, Nan Zhang, Ranran Haoran Zhang, Avitej Iyer, Renze Lou, Wenpeng Yin, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18406)  

**Abstract**: High-resolution image (HRI) understanding aims to process images with a large number of pixels, such as pathological images and agricultural aerial images, both of which can exceed 1 million pixels. Vision Large Language Models (VLMs) can allegedly handle HRIs, however, there is a lack of a comprehensive benchmark for VLMs to evaluate HRI understanding. To address this gap, we introduce HRScene, a novel unified benchmark for HRI understanding with rich scenes. HRScene incorporates 25 real-world datasets and 2 synthetic diagnostic datasets with resolutions ranging from 1,024 $\times$ 1,024 to 35,503 $\times$ 26,627. HRScene is collected and re-annotated by 10 graduate-level annotators, covering 25 scenarios, ranging from microscopic to radiology images, street views, long-range pictures, and telescope images. It includes HRIs of real-world objects, scanned documents, and composite multi-image. The two diagnostic evaluation datasets are synthesized by combining the target image with the gold answer and distracting images in different orders, assessing how well models utilize regions in HRI. We conduct extensive experiments involving 28 VLMs, including Gemini 2.0 Flash and GPT-4o. Experiments on HRScene show that current VLMs achieve an average accuracy of around 50% on real-world tasks, revealing significant gaps in HRI understanding. Results on synthetic datasets reveal that VLMs struggle to effectively utilize HRI regions, showing significant Regional Divergence and lost-in-middle, shedding light on future research. 

---
# A UD Treebank for Bohairic Coptic 

**Authors**: Amir Zeldes, Nina Speransky, Nicholas Wagner, Caroline T. Schroeder  

**Link**: [PDF](https://arxiv.org/pdf/2504.18386)  

**Abstract**: Despite recent advances in digital resources for other Coptic dialects, especially Sahidic, Bohairic Coptic, the main Coptic dialect for pre-Mamluk, late Byzantine Egypt, and the contemporary language of the Coptic Church, remains critically under-resourced. This paper presents and evaluates the first syntactically annotated corpus of Bohairic Coptic, sampling data from a range of works, including Biblical text, saints' lives and Christian ascetic writing. We also explore some of the main differences we observe compared to the existing UD treebank of Sahidic Coptic, the classical dialect of the language, and conduct joint and cross-dialect parsing experiments, revealing the unique nature of Bohairic as a related, but distinct variety from the more often studied Sahidic. 

---
# Pushing the boundary on Natural Language Inference 

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho  

**Link**: [PDF](https://arxiv.org/pdf/2504.18376)  

**Abstract**: Natural Language Inference (NLI) is a central task in natural language understanding with applications in fact-checking, question answering, and information retrieval. Despite its importance, current NLI systems heavily rely on supervised learning with datasets that often contain annotation artifacts and biases, limiting generalization and real-world applicability. In this work, we apply a reinforcement learning-based approach using Group Relative Policy Optimization (GRPO) for Chain-of-Thought (CoT) learning in NLI, eliminating the need for labeled rationales and enabling this type of training on more challenging datasets such as ANLI. We fine-tune 7B, 14B, and 32B language models using parameter-efficient techniques (LoRA and QLoRA), demonstrating strong performance across standard and adversarial NLI benchmarks. Our 32B AWQ-quantized model surpasses state-of-the-art results on 7 out of 11 adversarial sets$\unicode{x2013}$or on all of them considering our replication$\unicode{x2013}$within a 22GB memory footprint, showing that robust reasoning can be retained under aggressive quantization. This work provides a scalable and practical framework for building robust NLI systems without sacrificing inference quality. 

---
# Auto-SLURP: A Benchmark Dataset for Evaluating Multi-Agent Frameworks in Smart Personal Assistant 

**Authors**: Lei Shen, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.18373)  

**Abstract**: In recent years, multi-agent frameworks powered by large language models (LLMs) have advanced rapidly. Despite this progress, there is still a notable absence of benchmark datasets specifically tailored to evaluate their performance. To bridge this gap, we introduce Auto-SLURP, a benchmark dataset aimed at evaluating LLM-based multi-agent frameworks in the context of intelligent personal assistants. Auto-SLURP extends the original SLURP dataset -- initially developed for natural language understanding tasks -- by relabeling the data and integrating simulated servers and external services. This enhancement enables a comprehensive end-to-end evaluation pipeline, covering language understanding, task execution, and response generation. Our experiments demonstrate that Auto-SLURP presents a significant challenge for current state-of-the-art frameworks, highlighting that truly reliable and intelligent multi-agent personal assistants remain a work in progress. The dataset and related code are available at this https URL. 

---
# Comparing Uncertainty Measurement and Mitigation Methods for Large Language Models: A Systematic Review 

**Authors**: Toghrul Abbasli, Kentaroh Toyoda, Yuan Wang, Leon Witt, Muhammad Asif Ali, Yukai Miao, Dan Li, Qingsong Wei  

**Link**: [PDF](https://arxiv.org/pdf/2504.18346)  

**Abstract**: Large Language Models (LLMs) have been transformative across many domains. However, hallucination -- confidently outputting incorrect information -- remains one of the leading challenges for LLMs. This raises the question of how to accurately assess and quantify the uncertainty of LLMs. Extensive literature on traditional models has explored Uncertainty Quantification (UQ) to measure uncertainty and employed calibration techniques to address the misalignment between uncertainty and accuracy. While some of these methods have been adapted for LLMs, the literature lacks an in-depth analysis of their effectiveness and does not offer a comprehensive benchmark to enable insightful comparison among existing solutions. In this work, we fill this gap via a systematic survey of representative prior works on UQ and calibration for LLMs and introduce a rigorous benchmark. Using two widely used reliability datasets, we empirically evaluate six related methods, which justify the significant findings of our review. Finally, we provide outlooks for key future directions and outline open challenges. To the best of our knowledge, this survey is the first dedicated study to review the calibration methods and relevant metrics for LLMs. 

---
# TextTIGER: Text-based Intelligent Generation with Entity Prompt Refinement for Text-to-Image Generation 

**Authors**: Shintaro Ozaki, Kazuki Hayashi, Yusuke Sakai, Jingun Kwon, Hidetaka Kamigaito, Katsuhiko Hayashi, Manabu Okumura, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2504.18269)  

**Abstract**: Generating images from prompts containing specific entities requires models to retain as much entity-specific knowledge as possible. However, fully memorizing such knowledge is impractical due to the vast number of entities and their continuous emergence. To address this, we propose Text-based Intelligent Generation with Entity prompt Refinement (TextTIGER), which augments knowledge on entities included in the prompts and then summarizes the augmented descriptions using Large Language Models (LLMs) to mitigate performance degradation from longer inputs. To evaluate our method, we introduce WiT-Cub (WiT with Captions and Uncomplicated Background-explanations), a dataset comprising captions, images, and an entity list. Experiments on four image generation models and five LLMs show that TextTIGER improves image generation performance in standard metrics (IS, FID, and CLIPScore) compared to caption-only prompts. Additionally, multiple annotators' evaluation confirms that the summarized descriptions are more informative, validating LLMs' ability to generate concise yet rich descriptions. These findings demonstrate that refining prompts with augmented and summarized entity-related descriptions enhances image generation capabilities. The code and dataset will be available upon acceptance. 

---
# MAGI: Multi-Agent Guided Interview for Psychiatric Assessment 

**Authors**: Guanqun Bi, Zhuang Chen, Zhoufu Liu, Hongkai Wang, Xiyao Xiao, Yuqiang Xie, Wen Zhang, Yongkang Huang, Yuxuan Chen, Libiao Peng, Yi Feng, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18260)  

**Abstract**: Automating structured clinical interviews could revolutionize mental healthcare accessibility, yet existing large language models (LLMs) approaches fail to align with psychiatric diagnostic protocols. We present MAGI, the first framework that transforms the gold-standard Mini International Neuropsychiatric Interview (MINI) into automatic computational workflows through coordinated multi-agent collaboration. MAGI dynamically navigates clinical logic via four specialized agents: 1) an interview tree guided navigation agent adhering to the MINI's branching structure, 2) an adaptive question agent blending diagnostic probing, explaining, and empathy, 3) a judgment agent validating whether the response from participants meet the node, and 4) a diagnosis Agent generating Psychometric Chain-of- Thought (PsyCoT) traces that explicitly map symptoms to clinical criteria. Experimental results on 1,002 real-world participants covering depression, generalized anxiety, social anxiety and suicide shows that MAGI advances LLM- assisted mental health assessment by combining clinical rigor, conversational adaptability, and explainable reasoning. 

---
# Efficient Single-Pass Training for Multi-Turn Reasoning 

**Authors**: Ritesh Goru, Shanay Mehta, Prateek Jain  

**Link**: [PDF](https://arxiv.org/pdf/2504.18246)  

**Abstract**: Training Large Language Models ( LLMs) to generate explicit reasoning before they produce an answer has been shown to improve their performance across various tasks such as mathematics and coding. However, fine-tuning LLMs on multi-turn reasoning datasets presents a unique challenge: LLMs must generate reasoning tokens that are excluded from subsequent inputs to the LLM. This discrepancy prevents us from processing an entire conversation in a single forward pass-an optimization readily available when we fine-tune on a multi-turn non-reasoning dataset. This paper proposes a novel approach that overcomes this limitation through response token duplication and a custom attention mask that enforces appropriate visibility constraints. Our approach significantly reduces the training time and allows efficient fine-tuning on multi-turn reasoning datasets. 

---
# Even Small Reasoners Should Quote Their Sources: Introducing the Pleias-RAG Model Family 

**Authors**: Pierre-Carl Langlais, Pavel Chizhov, Mattia Nee, Carlos Rosas Hinostroza, Matthieu Delsart, Irène Girard, Othman Hicheur, Anastasia Stasenko, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2504.18225)  

**Abstract**: We introduce a new generation of small reasoning models for RAG, search, and source summarization. Pleias-RAG-350m and Pleias-RAG-1B are mid-trained on a large synthetic dataset emulating the retrieval of a wide variety of multilingual open sources from the Common Corpus. They provide native support for citation and grounding with literal quotes and reintegrate multiple features associated with RAG workflows, such as query routing, query reformulation, and source reranking. Pleias-RAG-350m and Pleias-RAG-1B outperform SLMs below 4 billion parameters on standardized RAG benchmarks (HotPotQA, 2wiki) and are competitive with popular larger models, including Qwen-2.5-7B, Llama-3.1-8B, and Gemma-3-4B. They are the only SLMs to date maintaining consistent RAG performance across leading European languages and ensuring systematic reference grounding for statements. Due to their size and ease of deployment on constrained infrastructure and higher factuality by design, the models unlock a range of new use cases for generative AI. 

---
# Optimising ChatGPT for creativity in literary translation: A case study from English into Dutch, Chinese, Catalan and Spanish 

**Authors**: Shuxiang Du, Ana Guerberof Arenas, Antonio Toral, Kyo Gerrits, Josep Marco Borillo  

**Link**: [PDF](https://arxiv.org/pdf/2504.18221)  

**Abstract**: This study examines the variability of Chat-GPT machine translation (MT) outputs across six different configurations in four languages,with a focus on creativity in a literary text. We evaluate GPT translations in different text granularity levels, temperature settings and prompting strategies with a Creativity Score formula. We found that prompting ChatGPT with a minimal instruction yields the best creative translations, with "Translate the following text into [TG] creatively" at the temperature of 1.0 outperforming other configurations and DeepL in Spanish, Dutch, and Chinese. Nonetheless, ChatGPT consistently underperforms compared to human translation (HT). 

---
# Aligning Language Models for Icelandic Legal Text Summarization 

**Authors**: Þórir Hrafn Harðarson, Hrafn Loftsson, Stefán Ólafsson  

**Link**: [PDF](https://arxiv.org/pdf/2504.18180)  

**Abstract**: The integration of language models in the legal domain holds considerable promise for streamlining processes and improving efficiency in managing extensive workloads. However, the specialized terminology, nuanced language, and formal style of legal texts can present substantial challenges. This study examines whether preference-based training techniques, specifically Reinforcement Learning from Human Feedback and Direct Preference Optimization, can enhance models' performance in generating Icelandic legal summaries that align with domain-specific language standards and user preferences. We compare models fine-tuned with preference training to those using conventional supervised learning. Results indicate that preference training improves the legal accuracy of generated summaries over standard fine-tuning but does not significantly enhance the overall quality of Icelandic language usage. Discrepancies between automated metrics and human evaluations further underscore the importance of qualitative assessment in developing language models for the legal domain. 

---
# EDU-NER-2025: Named Entity Recognition in Urdu Educational Texts using XLM-RoBERTa with X (formerly Twitter) 

**Authors**: Fida Ullah, Muhammad Ahmad, Muhammad Tayyab Zamir, Muhammad Arif, Grigori sidorov, Edgardo Manuel Felipe Riverón, Alexander Gelbukh  

**Link**: [PDF](https://arxiv.org/pdf/2504.18142)  

**Abstract**: Named Entity Recognition (NER) plays a pivotal role in various Natural Language Processing (NLP) tasks by identifying and classifying named entities (NEs) from unstructured data into predefined categories such as person, organization, location, date, and time. While extensive research exists for high-resource languages and general domains, NER in Urdu particularly within domain-specific contexts like education remains significantly underexplored. This is Due to lack of annotated datasets for educational content which limits the ability of existing models to accurately identify entities such as academic roles, course names, and institutional terms, underscoring the urgent need for targeted resources in this domain. To the best of our knowledge, no dataset exists in the domain of the Urdu language for this purpose. To achieve this objective this study makes three key contributions. Firstly, we created a manually annotated dataset in the education domain, named EDU-NER-2025, which contains 13 unique most important entities related to education domain. Second, we describe our annotation process and guidelines in detail and discuss the challenges of labelling EDU-NER-2025 dataset. Third, we addressed and analyzed key linguistic challenges, such as morphological complexity and ambiguity, which are prevalent in formal Urdu texts. 

---
# Temporal Entailment Pretraining for Clinical Language Models over EHR Data 

**Authors**: Tatsunori Tanaka, Fi Zheng, Kai Sato, Zhifeng Li, Yuanyun Zhang, Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.18128)  

**Abstract**: Clinical language models have achieved strong performance on downstream tasks by pretraining on domain specific corpora such as discharge summaries and medical notes. However, most approaches treat the electronic health record as a static document, neglecting the temporally-evolving and causally entwined nature of patient trajectories. In this paper, we introduce a novel temporal entailment pretraining objective for language models in the clinical domain. Our method formulates EHR segments as temporally ordered sentence pairs and trains the model to determine whether a later state is entailed by, contradictory to, or neutral with respect to an earlier state. Through this temporally structured pretraining task, models learn to perform latent clinical reasoning over time, improving their ability to generalize across forecasting and diagnosis tasks. We pretrain on a large corpus derived from MIMIC IV and demonstrate state of the art results on temporal clinical QA, early warning prediction, and disease progression modeling. 

---
# Evaluating Evaluation Metrics -- The Mirage of Hallucination Detection 

**Authors**: Atharva Kulkarni, Yuan Zhang, Joel Ruben Antony Moniz, Xiou Ge, Bo-Hsiang Tseng, Dhivya Piraviperumal, Swabha Swayamdipta, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18114)  

**Abstract**: Hallucinations pose a significant obstacle to the reliability and widespread adoption of language models, yet their accurate measurement remains a persistent challenge. While many task- and domain-specific metrics have been proposed to assess faithfulness and factuality concerns, the robustness and generalization of these metrics are still untested. In this paper, we conduct a large-scale empirical evaluation of 6 diverse sets of hallucination detection metrics across 4 datasets, 37 language models from 5 families, and 5 decoding methods. Our extensive investigation reveals concerning gaps in current hallucination evaluation: metrics often fail to align with human judgments, take an overtly myopic view of the problem, and show inconsistent gains with parameter scaling. Encouragingly, LLM-based evaluation, particularly with GPT-4, yields the best overall results, and mode-seeking decoding methods seem to reduce hallucinations, especially in knowledge-grounded settings. These findings underscore the need for more robust metrics to understand and quantify hallucinations, and better strategies to mitigate them. 

---
# Comparative Study on the Discourse Meaning of Chinese and English Media in the Paris Olympics Based on LDA Topic Modeling Technology and LLM Prompt Engineering 

**Authors**: Yinglong Yu, Zhaopu Yao, Fang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.18106)  

**Abstract**: This study analyzes Chinese and English media reports on the Paris Olympics using topic modeling, Large Language Model (LLM) prompt engineering, and corpus phraseology methods to explore similarities and differences in discourse construction and attitudinal meanings. Common topics include the opening ceremony, athlete performance, and sponsorship brands. Chinese media focus on specific sports, sports spirit, doping controversies, and new technologies, while English media focus on female athletes, medal wins, and eligibility controversies. Chinese reports show more frequent prepositional co-occurrences and positive semantic prosody in describing the opening ceremony and sports spirit. English reports exhibit positive semantic prosody when covering female athletes but negative prosody in predicting opening ceremony reactions and discussing women's boxing controversies. 

---
# Application and Optimization of Large Models Based on Prompt Tuning for Fact-Check-Worthiness Estimation 

**Authors**: Yinglong Yu, Hao Shen, Zhengyi Lyu, Qi He  

**Link**: [PDF](https://arxiv.org/pdf/2504.18104)  

**Abstract**: In response to the growing problem of misinformation in the context of globalization and informatization, this paper proposes a classification method for fact-check-worthiness estimation based on prompt tuning. We construct a model for fact-check-worthiness estimation at the methodological level using prompt tuning. By applying designed prompt templates to large language models, we establish in-context learning and leverage prompt tuning technology to improve the accuracy of determining whether claims have fact-check-worthiness, particularly when dealing with limited or unlabeled data. Through extensive experiments on public datasets, we demonstrate that the proposed method surpasses or matches multiple baseline methods in the classification task of fact-check-worthiness estimation assessment, including classical pre-trained models such as BERT, as well as recent popular large models like GPT-3.5 and GPT-4. Experiments show that the prompt tuning-based method proposed in this study exhibits certain advantages in evaluation metrics such as F1 score and accuracy, thereby effectively validating its effectiveness and advancement in the task of fact-check-worthiness estimation. 

---
# Random-Set Large Language Models 

**Authors**: Muhammad Mubashar, Shireen Kudukkil Manchingal, Fabio Cuzzolin  

**Link**: [PDF](https://arxiv.org/pdf/2504.18085)  

**Abstract**: Large Language Models (LLMs) are known to produce very high-quality tests and responses to our queries. But how much can we trust this generated text? In this paper, we study the problem of uncertainty quantification in LLMs. We propose a novel Random-Set Large Language Model (RSLLM) approach which predicts finite random sets (belief functions) over the token space, rather than probability vectors as in classical LLMs. In order to allow so efficiently, we also present a methodology based on hierarchical clustering to extract and use a budget of "focal" subsets of tokens upon which the belief prediction is defined, rather than using all possible collections of tokens, making the method scalable yet effective. RS-LLMs encode the epistemic uncertainty induced in their generation process by the size and diversity of its training set via the size of the credal sets associated with the predicted belief functions. The proposed approach is evaluated on CoQA and OBQA datasets using Llama2-7b, Mistral-7b and Phi-2 models and is shown to outperform the standard model in both datasets in terms of correctness of answer while also showing potential in estimating the second level uncertainty in its predictions and providing the capability to detect when its hallucinating. 

---
# Stabilizing Reasoning in Medical LLMs with Continued Pretraining and Reasoning Preference Optimization 

**Authors**: Wataru Kawakami, Keita Suzuki, Junichiro Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2504.18080)  

**Abstract**: Large Language Models (LLMs) show potential in medicine, yet clinical adoption is hindered by concerns over factual accuracy, language-specific limitations (e.g., Japanese), and critically, their reliability when required to generate reasoning explanations -- a prerequisite for trust. This paper introduces Preferred-MedLLM-Qwen-72B, a 72B-parameter model optimized for the Japanese medical domain to achieve both high accuracy and stable reasoning. We employ a two-stage fine-tuning process on the Qwen2.5-72B base model: first, Continued Pretraining (CPT) on a comprehensive Japanese medical corpus instills deep domain knowledge. Second, Reasoning Preference Optimization (RPO), a preference-based method, enhances the generation of reliable reasoning pathways while preserving high answer accuracy. Evaluations on the Japanese Medical Licensing Exam benchmark (IgakuQA) show Preferred-MedLLM-Qwen-72B achieves state-of-the-art performance (0.868 accuracy), surpassing strong proprietary models like GPT-4o (0.866). Crucially, unlike baseline or CPT-only models which exhibit significant accuracy degradation (up to 11.5\% and 3.8\% respectively on IgakuQA) when prompted for explanations, our model maintains its high accuracy (0.868) under such conditions. This highlights RPO's effectiveness in stabilizing reasoning generation. This work underscores the importance of optimizing for reliable explanations alongside accuracy. We release the Preferred-MedLLM-Qwen-72B model weights to foster research into trustworthy LLMs for specialized, high-stakes applications. 

---
# PropRAG: Guiding Retrieval with Beam Search over Proposition Paths 

**Authors**: Jingjin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18070)  

**Abstract**: Retrieval Augmented Generation (RAG) has become the standard non-parametric approach for equipping Large Language Models (LLMs) with up-to-date knowledge and mitigating catastrophic forgetting common in continual learning. However, standard RAG, relying on independent passage retrieval, fails to capture the interconnected nature of human memory crucial for complex reasoning (associativity) and contextual understanding (sense-making). While structured RAG methods like HippoRAG utilize knowledge graphs (KGs) built from triples, the inherent context loss limits fidelity. We introduce PropRAG, a framework leveraging contextually rich propositions and a novel beam search algorithm over proposition paths to explicitly discover multi-step reasoning chains. Crucially, PropRAG's online retrieval process operates entirely without invoking generative LLMs, relying instead on efficient graph traversal and pre-computed embeddings. This avoids online LLM inference costs and potential inconsistencies during evidence gathering. LLMs are used effectively offline for high-quality proposition extraction and post-retrieval for answer generation. PropRAG achieves state-of-the-art zero-shot Recall@5 results on PopQA (55.3%), 2Wiki (93.7%), HotpotQA (97.0%), and MuSiQue (77.3%), alongside top F1 scores (e.g., 52.4% on MuSiQue). By improving evidence retrieval through richer representation and explicit, LLM-free online path finding, PropRAG advances non-parametric continual learning. 

---
# Exploring Personality-Aware Interactions in Salesperson Dialogue Agents 

**Authors**: Sijia Cheng, Wen-Yu Chang, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.18058)  

**Abstract**: The integration of dialogue agents into the sales domain requires a deep understanding of how these systems interact with users possessing diverse personas. This study explores the influence of user personas, defined using the Myers-Briggs Type Indicator (MBTI), on the interaction quality and performance of sales-oriented dialogue agents. Through large-scale testing and analysis, we assess the pre-trained agent's effectiveness, adaptability, and personalization capabilities across a wide range of MBTI-defined user types. Our findings reveal significant patterns in interaction dynamics, task completion rates, and dialogue naturalness, underscoring the future potential for dialogue agents to refine their strategies to better align with varying personality traits. This work not only provides actionable insights for building more adaptive and user-centric conversational systems in the sales domain but also contributes broadly to the field by releasing persona-defined user simulators. These simulators, unconstrained by domain, offer valuable tools for future research and demonstrate the potential for scaling personalized dialogue systems across diverse applications. 

---
# DREAM: Disentangling Risks to Enhance Safety Alignment in Multimodal Large Language Models 

**Authors**: Jianyu Liu, Hangyu Guo, Ranjie Duan, Xingyuan Bu, Yancheng He, Shilong Li, Hui Huang, Jiaheng Liu, Yucheng Wang, Chenchen Jing, Xingwei Qu, Xiao Zhang, Yingshui Tan, Yanan Wu, Jihao Gu, Yangguang Li, Jianke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18053)  

**Abstract**: Multimodal Large Language Models (MLLMs) pose unique safety challenges due to their integration of visual and textual data, thereby introducing new dimensions of potential attacks and complex risk combinations. In this paper, we begin with a detailed analysis aimed at disentangling risks through step-by-step reasoning within multimodal inputs. We find that systematic multimodal risk disentanglement substantially enhances the risk awareness of MLLMs. Via leveraging the strong discriminative abilities of multimodal risk disentanglement, we further introduce \textbf{DREAM} (\textit{\textbf{D}isentangling \textbf{R}isks to \textbf{E}nhance Safety \textbf{A}lignment in \textbf{M}LLMs}), a novel approach that enhances safety alignment in MLLMs through supervised fine-tuning and iterative Reinforcement Learning from AI Feedback (RLAIF). Experimental results show that DREAM significantly boosts safety during both inference and training phases without compromising performance on normal tasks (namely oversafety), achieving a 16.17\% improvement in the SIUO safe\&effective score compared to GPT-4V. The data and code are available at this https URL. 

---
# RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models 

**Authors**: Bang An, Shiyue Zhang, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2504.18041)  

**Abstract**: Efforts to ensure the safety of large language models (LLMs) include safety fine-tuning, evaluation, and red teaming. However, despite the widespread use of the Retrieval-Augmented Generation (RAG) framework, AI safety work focuses on standard LLMs, which means we know little about how RAG use cases change a model's safety profile. We conduct a detailed comparative analysis of RAG and non-RAG frameworks with eleven LLMs. We find that RAG can make models less safe and change their safety profile. We explore the causes of this change and find that even combinations of safe models with safe documents can cause unsafe generations. In addition, we evaluate some existing red teaming methods for RAG settings and show that they are less effective than when used for non-RAG settings. Our work highlights the need for safety research and red-teaming methods specifically tailored for RAG LLMs. 

---
# Memory Reviving, Continuing Learning and Beyond: Evaluation of Pre-trained Encoders and Decoders for Multimodal Machine Translation 

**Authors**: Zhuang Yu, Shiliang Sun, Jing Zhao, Tengfei Song, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18012)  

**Abstract**: Multimodal Machine Translation (MMT) aims to improve translation quality by leveraging auxiliary modalities such as images alongside textual input. While recent advances in large-scale pre-trained language and vision models have significantly benefited unimodal natural language processing tasks, their effectiveness and role in MMT remain underexplored. In this work, we conduct a systematic study on the impact of pre-trained encoders and decoders in multimodal translation models. Specifically, we analyze how different training strategies, from training from scratch to using pre-trained and partially frozen components, affect translation performance under a unified MMT framework. Experiments are carried out on the Multi30K and CoMMuTE dataset across English-German and English-French translation tasks. Our results reveal that pre-training plays a crucial yet asymmetrical role in multimodal settings: pre-trained decoders consistently yield more fluent and accurate outputs, while pre-trained encoders show varied effects depending on the quality of visual-text alignment. Furthermore, we provide insights into the interplay between modality fusion and pre-trained components, offering guidance for future architecture design in multimodal translation systems. 

---
# Improving LLM Personas via Rationalization with Psychological Scaffolds 

**Authors**: Brihi Joshi, Xiang Ren, Swabha Swayamdipta, Rik Koncel-Kedziorski, Tim Paek  

**Link**: [PDF](https://arxiv.org/pdf/2504.17993)  

**Abstract**: Language models prompted with a user description or persona can predict a user's preferences and opinions, but existing approaches to building personas -- based solely on a user's demographic attributes and/or prior judgments -- fail to capture the underlying reasoning behind said user judgments. We introduce PB&J (Psychology of Behavior and Judgments), a framework that improves LLM personas by incorporating rationales of why a user might make specific judgments. These rationales are LLM-generated, and aim to reason about a user's behavior on the basis of their experiences, personality traits or beliefs. This is done using psychological scaffolds -- structured frameworks grounded in theories such as the Big 5 Personality Traits and Primal World Beliefs -- that help provide structure to the generated rationales. Experiments on public opinion and movie preference prediction tasks demonstrate that LLM personas augmented with PB&J rationales consistently outperform methods using only a user's demographics and/or judgments. Additionally, LLM personas constructed using scaffolds describing user beliefs perform competitively with those using human-written rationales. 

---
# Optimism, Expectation, or Sarcasm? Multi-Class Hope Speech Detection in Spanish and English 

**Authors**: Sabur Butt, Fazlourrahman Balouchzahi, Ahmad Imam Amjad, Maaz Amjad, Hector G. Ceballos, Salud Maria Jimenez-Zafra  

**Link**: [PDF](https://arxiv.org/pdf/2504.17974)  

**Abstract**: Hope is a complex and underexplored emotional state that plays a significant role in education, mental health, and social interaction. Unlike basic emotions, hope manifests in nuanced forms ranging from grounded optimism to exaggerated wishfulness or sarcasm, making it difficult for Natural Language Processing systems to detect accurately. This study introduces PolyHope V2, a multilingual, fine-grained hope speech dataset comprising over 30,000 annotated tweets in English and Spanish. This resource distinguishes between four hope subtypes Generalized, Realistic, Unrealistic, and Sarcastic and enhances existing datasets by explicitly labeling sarcastic instances. We benchmark multiple pretrained transformer models and compare them with large language models (LLMs) such as GPT 4 and Llama 3 under zero-shot and few-shot regimes. Our findings show that fine-tuned transformers outperform prompt-based LLMs, especially in distinguishing nuanced hope categories and sarcasm. Through qualitative analysis and confusion matrices, we highlight systematic challenges in separating closely related hope subtypes. The dataset and results provide a robust foundation for future emotion recognition tasks that demand greater semantic and contextual sensitivity across languages. 

---
# Kimi-Audio Technical Report 

**Authors**: KimiTeam, Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, Zhengtao Wang, Chu Wei, Yifei Xin, Xinran Xu, Jianwei Yu, Yutao Zhang, Xinyu Zhou, Y. Charles, Jun Chen, Yanru Chen, Yulun Du, Weiran He, Zhenxing Hu, Guokun Lai, Qingcheng Li, Yangyang Liu, Weidong Sun, Jianzhou Wang, Yuzhi Wang, Yuefeng Wu, Yuxin Wu, Dongchao Yang, Hao Yang, Ying Yang, Zhilin Yang, Aoxiong Yin, Ruibin Yuan, Yutong Zhang, Zaida Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.18425)  

**Abstract**: We present Kimi-Audio, an open-source audio foundation model that excels in audio understanding, generation, and conversation. We detail the practices in building Kimi-Audio, including model architecture, data curation, training recipe, inference deployment, and evaluation. Specifically, we leverage a 12.5Hz audio tokenizer, design a novel LLM-based architecture with continuous features as input and discrete tokens as output, and develop a chunk-wise streaming detokenizer based on flow matching. We curate a pre-training dataset that consists of more than 13 million hours of audio data covering a wide range of modalities including speech, sound, and music, and build a pipeline to construct high-quality and diverse post-training data. Initialized from a pre-trained LLM, Kimi-Audio is continual pre-trained on both audio and text data with several carefully designed tasks, and then fine-tuned to support a diverse of audio-related tasks. Extensive evaluation shows that Kimi-Audio achieves state-of-the-art performance on a range of audio benchmarks including speech recognition, audio understanding, audio question answering, and speech conversation. We release the codes, model checkpoints, as well as the evaluation toolkits in this https URL. 

---
# Adversarial Attacks on LLM-as-a-Judge Systems: Insights from Prompt Injections 

**Authors**: Narek Maloyan, Dmitry Namiot  

**Link**: [PDF](https://arxiv.org/pdf/2504.18333)  

**Abstract**: LLM as judge systems used to assess text quality code correctness and argument strength are vulnerable to prompt injection attacks. We introduce a framework that separates content author attacks from system prompt attacks and evaluate five models Gemma 3.27B Gemma 3.4B Llama 3.2 3B GPT 4 and Claude 3 Opus on four tasks with various defenses using fifty prompts per condition. Attacks achieved up to seventy three point eight percent success smaller models proved more vulnerable and transferability ranged from fifty point five to sixty two point six percent. Our results contrast with Universal Prompt Injection and AdvPrompter We recommend multi model committees and comparative scoring and release all code and datasets 

---
# Tracking Articulatory Dynamics in Speech with a Fixed-Weight BiLSTM-CNN Architecture 

**Authors**: Leena G Pillai, D. Muhammad Noorul Mubarak, Elizabeth Sherly  

**Link**: [PDF](https://arxiv.org/pdf/2504.18099)  

**Abstract**: Speech production is a complex sequential process which involve the coordination of various articulatory features. Among them tongue being a highly versatile active articulator responsible for shaping airflow to produce targeted speech sounds that are intellectual, clear, and distinct. This paper presents a novel approach for predicting tongue and lip articulatory features involved in a given speech acoustics using a stacked Bidirectional Long Short-Term Memory (BiLSTM) architecture, combined with a one-dimensional Convolutional Neural Network (CNN) for post-processing with fixed weights initialization. The proposed network is trained with two datasets consisting of simultaneously recorded speech and Electromagnetic Articulography (EMA) datasets, each introducing variations in terms of geographical origin, linguistic characteristics, phonetic diversity, and recording equipment. The performance of the model is assessed in Speaker Dependent (SD), Speaker Independent (SI), corpus dependent (CD) and cross corpus (CC) modes. Experimental results indicate that the proposed model with fixed weights approach outperformed the adaptive weights initialization with in relatively minimal number of training epochs. These findings contribute to the development of robust and efficient models for articulatory feature prediction, paving the way for advancements in speech production research and applications. 

---
# SMARTFinRAG: Interactive Modularized Financial RAG Benchmark 

**Authors**: Yiwei Zha  

**Link**: [PDF](https://arxiv.org/pdf/2504.18024)  

**Abstract**: Financial sectors are rapidly adopting language model technologies, yet evaluating specialized RAG systems in this domain remains challenging. This paper introduces SMARTFinRAG, addressing three critical gaps in financial RAG assessment: (1) a fully modular architecture where components can be dynamically interchanged during runtime; (2) a document-centric evaluation paradigm generating domain-specific QA pairs from newly ingested financial documents; and (3) an intuitive interface bridging research-implementation divides. Our evaluation quantifies both retrieval efficacy and response quality, revealing significant performance variations across configurations. The platform's open-source architecture supports transparent, reproducible research while addressing practical deployment challenges faced by financial institutions implementing RAG systems. 

---
# Collaborating Action by Action: A Multi-agent LLM Framework for Embodied Reasoning 

**Authors**: Isadora White, Kolby Nottingham, Ayush Maniar, Max Robinson, Hansen Lillemark, Mehul Maheshwari, Lianhui Qin, Prithviraj Ammanabrolu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17950)  

**Abstract**: Collaboration is ubiquitous and essential in day-to-day life -- from exchanging ideas, to delegating tasks, to generating plans together. This work studies how LLMs can adaptively collaborate to perform complex embodied reasoning tasks. To this end we introduce MINDcraft, an easily extensible platform built to enable LLM agents to control characters in the open-world game of Minecraft; and MineCollab, a benchmark to test the different dimensions of embodied and collaborative reasoning. An experimental study finds that the primary bottleneck in collaborating effectively for current state-of-the-art agents is efficient natural language communication, with agent performance dropping as much as 15% when they are required to communicate detailed task completion plans. We conclude that existing LLM agents are ill-optimized for multi-agent collaboration, especially in embodied scenarios, and highlight the need to employ methods beyond in-context and imitation learning. Our website can be found here: this https URL 

---
# Toward a Human-Centered Evaluation Framework for Trustworthy LLM-Powered GUI Agents 

**Authors**: Chaoran Chen, Zhiping Zhang, Ibrahim Khalilov, Bingcan Guo, Simret A Gebreegziabher, Yanfang Ye, Ziang Xiao, Yaxing Yao, Tianshi Li, Toby Jia-Jun Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.17934)  

**Abstract**: The rise of Large Language Models (LLMs) has revolutionized Graphical User Interface (GUI) automation through LLM-powered GUI agents, yet their ability to process sensitive data with limited human oversight raises significant privacy and security risks. This position paper identifies three key risks of GUI agents and examines how they differ from traditional GUI automation and general autonomous agents. Despite these risks, existing evaluations focus primarily on performance, leaving privacy and security assessments largely unexplored. We review current evaluation metrics for both GUI and general LLM agents and outline five key challenges in integrating human evaluators for GUI agent assessments. To address these gaps, we advocate for a human-centered evaluation framework that incorporates risk assessments, enhances user awareness through in-context consent, and embeds privacy and security considerations into GUI agent design and evaluation. 

---
# CAMU: Context Augmentation for Meme Understanding 

**Authors**: Girish A. Koushik, Diptesh Kanojia, Helen Treharne, Aditya Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2504.17902)  

**Abstract**: Social media memes are a challenging domain for hate detection because they intertwine visual and textual cues into culturally nuanced messages. We introduce a novel framework, CAMU, which leverages large vision-language models to generate more descriptive captions, a caption-scoring neural network to emphasise hate-relevant content, and parameter-efficient fine-tuning of CLIP's text encoder for an improved multimodal understanding of memes. Experiments on publicly available hateful meme datasets show that simple projection layer fine-tuning yields modest gains, whereas selectively tuning deeper text encoder layers significantly boosts performance on all evaluation metrics. Moreover, our approach attains high accuracy (0.807) and F1-score (0.806) on the Hateful Memes dataset, at par with the existing SoTA framework while being much more efficient, offering practical advantages in real-world scenarios that rely on fixed decision thresholds. CAMU also achieves the best F1-score of 0.673 on the MultiOFF dataset for offensive meme identification, demonstrating its generalisability. Additional analyses on benign confounders reveal that robust visual grounding and nuanced text representations are crucial for reliable hate and offence detection. We will publicly release CAMU along with the resultant models for further research.
Disclaimer: This paper includes references to potentially disturbing, hateful, or offensive content due to the nature of the task. 

---
# Token Sequence Compression for Efficient Multimodal Computing 

**Authors**: Yasmine Omri, Parth Shroff, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2504.17892)  

**Abstract**: The exponential growth of Large Multimodal Models (LMMs) has driven advancements in cross-modal reasoning but at significant computational costs. In this work, we focus on visual language models. We highlight the redundancy and inefficiency in current vision encoders, and seek to construct an adaptive compression method for multimodal data. In this work, we characterize a panoply of visual token selection and merging approaches through both benchmarking and qualitative analysis. In particular, we demonstrate that simple cluster-level token aggregation outperforms prior state-of-the-art works in token selection and merging, including merging at the vision encoder level and attention-based approaches. We underline the redundancy in current vision encoders, and shed light on several puzzling trends regarding principles of visual token selection through cross-modal attention visualizations. This work is a first effort towards more effective encoding and processing of high-dimensional data, and paves the way for more scalable and sustainable multimodal systems. 

---
# Unsupervised Corpus Poisoning Attacks in Continuous Space for Dense Retrieval 

**Authors**: Yongkang Li, Panagiotis Eustratiadis, Simon Lupart, Evangelos Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2504.17884)  

**Abstract**: This paper concerns corpus poisoning attacks in dense information retrieval, where an adversary attempts to compromise the ranking performance of a search algorithm by injecting a small number of maliciously generated documents into the corpus. Our work addresses two limitations in the current literature. First, attacks that perform adversarial gradient-based word substitution search do so in the discrete lexical space, while retrieval itself happens in the continuous embedding space. We thus propose an optimization method that operates in the embedding space directly. Specifically, we train a perturbation model with the objective of maintaining the geometric distance between the original and adversarial document embeddings, while also maximizing the token-level dissimilarity between the original and adversarial documents. Second, it is common for related work to have a strong assumption that the adversary has prior knowledge about the queries. In this paper, we focus on a more challenging variant of the problem where the adversary assumes no prior knowledge about the query distribution (hence, unsupervised). Our core contribution is an adversarial corpus attack that is fast and effective. We present comprehensive experimental results on both in- and out-of-domain datasets, focusing on two related tasks: a top-1 attack and a corpus poisoning attack. We consider attacks under both a white-box and a black-box setting. Notably, our method can generate successful adversarial examples in under two minutes per target document; four times faster compared to the fastest gradient-based word substitution methods in the literature with the same hardware. Furthermore, our adversarial generation method generates text that is more likely to occur under the distribution of natural text (low perplexity), and is therefore more difficult to detect. 

---
# VideoVista-CulturalLingo: 360$^\circ$ Horizons-Bridging Cultures, Languages, and Domains in Video Comprehension 

**Authors**: Xinyu Chen, Yunxin Li, Haoyuan Shi, Baotian Hu, Wenhan Luo, Yaowei Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17821)  

**Abstract**: Assessing the video comprehension capabilities of multimodal AI systems can effectively measure their understanding and reasoning abilities. Most video evaluation benchmarks are limited to a single language, typically English, and predominantly feature videos rooted in Western cultural contexts. In this paper, we present VideoVista-CulturalLingo, the first video evaluation benchmark designed to bridge cultural, linguistic, and domain divide in video comprehension. Our work differs from existing benchmarks in the following ways: 1) Cultural diversity, incorporating cultures from China, North America, and Europe; 2) Multi-linguistics, with questions presented in Chinese and English-two of the most widely spoken languages; and 3) Broad domain, featuring videos sourced from hundreds of human-created domains. VideoVista-CulturalLingo contains 1,389 videos and 3,134 QA pairs, and we have evaluated 24 recent open-source or proprietary video large models. From the experiment results, we observe that: 1) Existing models perform worse on Chinese-centric questions than Western-centric ones, particularly those related to Chinese history; 2) Current open-source models still exhibit limitations in temporal understanding, especially in the Event Localization task, achieving a maximum score of only 45.2%; 3) Mainstream models demonstrate strong performance in general scientific questions, while open-source models demonstrate weak performance in mathematics. 

---
