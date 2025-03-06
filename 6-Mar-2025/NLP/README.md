# Process-based Self-Rewarding Language Models 

**Authors**: Shimao Zhang, Xiao Liu, Xin Zhang, Junxiao Liu, Zheheng Luo, Shujian Huang, Yeyun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.03746)  

**Abstract**: Large Language Models have demonstrated outstanding performance across various downstream tasks and have been widely applied in multiple scenarios. Human-annotated preference data is used for training to further improve LLMs' performance, which is constrained by the upper limit of human performance. Therefore, Self-Rewarding method has been proposed, where LLMs generate training data by rewarding their own outputs. However, the existing self-rewarding paradigm is not effective in mathematical reasoning scenarios and may even lead to a decline in performance. In this work, we propose the Process-based Self-Rewarding pipeline for language models, which introduces long-thought reasoning, step-wise LLM-as-a-Judge, and step-wise preference optimization within the self-rewarding paradigm. Our new paradigm successfully enhances the performance of LLMs on multiple mathematical reasoning benchmarks through iterative Process-based Self-Rewarding, demonstrating the immense potential of self-rewarding to achieve LLM reasoning that may surpass human capabilities. 

---
# Improving LLM Safety Alignment with Dual-Objective Optimization 

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.03710)  

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at this https URL 

---
# Effective LLM Knowledge Learning via Model Generalization 

**Authors**: Mingkang Zhu, Xi Chen, Zhongdao Wang, Bei Yu, Hengshuang Zhao, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.03705)  

**Abstract**: Large language models (LLMs) are trained on enormous documents that contain extensive world knowledge. However, it is still not well-understood how knowledge is acquired via autoregressive pre-training. This lack of understanding greatly hinders effective knowledge learning, especially for continued pretraining on up-to-date information, as this evolving information often lacks diverse repetitions like foundational knowledge. In this paper, we focus on understanding and improving LLM knowledge learning. We found and verified that knowledge learning for LLMs can be deemed as an implicit supervised task hidden in the autoregressive pre-training objective. Our findings suggest that knowledge learning for LLMs would benefit from methods designed to improve generalization ability for supervised tasks. Based on our analysis, we propose the formatting-based data augmentation to grow in-distribution samples, which does not present the risk of altering the facts embedded in documents as text paraphrasing. We also introduce sharpness-aware minimization as an effective optimization algorithm to better improve generalization. Moreover, our analysis and method can be readily extended to instruction tuning. Extensive experiment results validate our findings and demonstrate our methods' effectiveness in both continued pre-training and instruction tuning. This paper offers new perspectives and insights to interpret and design effective strategies for LLM knowledge learning. 

---
# SoftMatcha: A Soft and Fast Pattern Matcher for Billion-Scale Corpus Searches 

**Authors**: Hiroyuki Deguchi, Go Kamoda, Yusuke Matsushita, Chihiro Taguchi, Kohei Suenaga, Masaki Waga, Sho Yokoi  

**Link**: [PDF](https://arxiv.org/pdf/2503.03703)  

**Abstract**: Researchers and practitioners in natural language processing and computational linguistics frequently observe and analyze the real language usage in large-scale corpora. For that purpose, they often employ off-the-shelf pattern-matching tools, such as grep, and keyword-in-context concordancers, which is widely used in corpus linguistics for gathering examples. Nonetheless, these existing techniques rely on surface-level string matching, and thus they suffer from the major limitation of not being able to handle orthographic variations and paraphrasing -- notable and common phenomena in any natural language. In addition, existing continuous approaches such as dense vector search tend to be overly coarse, often retrieving texts that are unrelated but share similar topics. Given these challenges, we propose a novel algorithm that achieves \emph{soft} (or semantic) yet efficient pattern matching by relaxing a surface-level matching with word embeddings. Our algorithm is highly scalable with respect to the size of the corpus text utilizing inverted indexes. We have prepared an efficient implementation, and we provide an accessible web tool. Our experiments demonstrate that the proposed method (i) can execute searches on billion-scale corpora in less than a second, which is comparable in speed to surface-level string matching and dense vector search; (ii) can extract harmful instances that semantically match queries from a large set of English and Japanese Wikipedia articles; and (iii) can be effectively applied to corpus-linguistic analyses of Latin, a language with highly diverse inflections. 

---
# Developing and Utilizing a Large-Scale Cantonese Dataset for Multi-Tasking in Large Language Models 

**Authors**: Jiyue Jiang, Alfred Kar Yin Truong, Yanyu Chen, Qinghang Bao, Sheng Wang, Pengan Chen, Jiuming Wang, Lingpeng Kong, Yu Li, Chuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.03702)  

**Abstract**: High-quality data resources play a crucial role in learning large language models (LLMs), particularly for low-resource languages like Cantonese. Despite having more than 85 million native speakers, Cantonese is still considered a low-resource language in the field of natural language processing (NLP) due to factors such as the dominance of Mandarin, lack of cohesion within the Cantonese-speaking community, diversity in character encoding and input methods, and the tendency of overseas Cantonese speakers to prefer using English. In addition, rich colloquial vocabulary of Cantonese, English loanwords, and code-switching characteristics add to the complexity of corpus collection and processing. To address these challenges, we collect Cantonese texts from a variety of sources, including open source corpora, Hong Kong-specific forums, Wikipedia, and Common Crawl data. We conduct rigorous data processing through language filtering, quality filtering, content filtering, and de-duplication steps, successfully constructing a high-quality Cantonese corpus of over 2 billion tokens for training large language models. We further refined the model through supervised fine-tuning (SFT) on curated Cantonese tasks, enhancing its ability to handle specific applications. Upon completion of the training, the model achieves state-of-the-art (SOTA) performance on four Cantonese benchmarks. After training on our dataset, the model also exhibits improved performance on other mainstream language tasks. 

---
# MAS-GPT: Training LLMs to Build LLM-based Multi-Agent Systems 

**Authors**: Rui Ye, Shuo Tang, Rui Ge, Yaxin Du, Zhenfei Yin, Siheng Chen, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2503.03686)  

**Abstract**: LLM-based multi-agent systems (MAS) have shown significant potential in tackling diverse tasks. However, to design effective MAS, existing approaches heavily rely on manual configurations or multiple calls of advanced LLMs, resulting in inadaptability and high inference costs. In this paper, we simplify the process of building an MAS by reframing it as a generative language task, where the input is a user query and the output is a corresponding MAS. To address this novel task, we unify the representation of MAS as executable code and propose a consistency-oriented data construction pipeline to create a high-quality dataset comprising coherent and consistent query-MAS pairs. Using this dataset, we train MAS-GPT, an open-source medium-sized LLM that is capable of generating query-adaptive MAS within a single LLM inference. The generated MAS can be seamlessly applied to process user queries and deliver high-quality responses. Extensive experiments on 9 benchmarks and 5 LLMs show that the proposed MAS-GPT consistently outperforms 10+ baseline MAS methods on diverse settings, indicating MAS-GPT's high effectiveness, efficiency and strong generalization ability. Code will be available at this https URL. 

---
# Quantification of Tenseness in English and Japanese Tense-Lax Vowels: A Lagrangian Model with Indicator $θ_1$ and Force of Tenseness Ftense(t) 

**Authors**: Tatsuya Ishizaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.03681)  

**Abstract**: The concept of vowel tenseness has traditionally been examined through the binary distinction of tense and lax vowels. However, no universally accepted quantitative definition of tenseness has been established in any language. Previous studies, including those by Jakobson, Fant, and Halle (1951) and Chomsky and Halle (1968), have explored the relationship between vowel tenseness and the vocal tract. Building on these foundations, Ishizaki (2019, 2022) proposed an indirect quantification of vowel tenseness using formant angles $\theta_1$ and $\theta_{F1}$ and their first and second derivatives, $d^Z_1(t)/dt = \lim \tan \theta_1(t$) and $d^2 Z_1(t)/dt^2 = d/dt \lim \tan \theta_1(t)$. This study extends this approach by investigating the potential role of a force-related parameter in determining vowel quality. Specifically, we introduce a simplified model based on the Lagrangian equation to describe the dynamic interaction of the tongue and jaw within the oral cavity during the articulation of close vowels. This model provides a theoretical framework for estimating the forces involved in vowel production across different languages, offering new insights into the physical mechanisms underlying vowel articulation. The findings suggest that this force-based perspective warrants further exploration as a key factor in phonetic and phonological studies. 

---
# Attentive Reasoning Queries: A Systematic Method for Optimizing Instruction-Following in Large Language Models 

**Authors**: Bar Karov, Dor Zohar, Yam Marcovitz  

**Link**: [PDF](https://arxiv.org/pdf/2503.03669)  

**Abstract**: We present Attentive Reasoning Queries (ARQs), a novel structured reasoning approach that significantly improves instruction-following in Large Language Models through domain-specialized reasoning blueprints. While LLMs demonstrate remarkable capabilities across diverse tasks, they often fail to maintain adherence to complex, use-case-specific instructions during multi-turn conversations, presenting challenges for business-critical applications. ARQs address this limitation by guiding LLMs through systematic reasoning steps with targeted queries that reinstate critical instructions and facilitate intermediate reasoning throughout the completion process. In extensive testing within Parlant, our framework for reliable customer-facing agents in which ARQs were born out of necessity, they achieved a 90.2% success rate across 87 test scenarios, outperforming both Chain-of-Thought reasoning (86.1%) and direct response generation (81.5%). ARQs showed particular strength in addressing persistent failure modes like guideline re-application and hallucination prevention. Our analysis also revealed that ARQs can potentially be more computationally efficient than free-form reasoning when carefully designed. These findings demonstrate that structured reasoning approaches provide effective mechanisms for controlling how LLMs process information and make decisions in complex scenarios. 

---
# Analogical Reasoning Inside Large Language Models: Concept Vectors and the Limits of Abstraction 

**Authors**: Gustaw Opiełka, Hannes Rosenbusch, Claire E. Stevenson  

**Link**: [PDF](https://arxiv.org/pdf/2503.03666)  

**Abstract**: Analogical reasoning relies on conceptual abstractions, but it is unclear whether Large Language Models (LLMs) harbor such internal representations. We explore distilled representations from LLM activations and find that function vectors (FVs; Todd et al., 2024) - compact representations for in-context learning (ICL) tasks - are not invariant to simple input changes (e.g., open-ended vs. multiple-choice), suggesting they capture more than pure concepts. Using representational similarity analysis (RSA), we localize a small set of attention heads that encode invariant concept vectors (CVs) for verbal concepts like "antonym". These CVs function as feature detectors that operate independently of the final output - meaning that a model may form a correct internal representation yet still produce an incorrect output. Furthermore, CVs can be used to causally guide model behaviour. However, for more abstract concepts like "previous" and "next", we do not observe invariant linear representations, a finding we link to generalizability issues LLMs display within these domains. 

---
# Improving Neutral Point of View Text Generation through Parameter-Efficient Reinforcement Learning and a Small-Scale High-Quality Dataset 

**Authors**: Jessica Hoffmann, Christiane Ahlheim, Zac Yu, Aria Walfrand, Jarvis Jin, Marie Tano, Ahmad Beirami, Erin van Liemt, Nithum Thain, Hakim Sidahmed, Lucas Dixon  

**Link**: [PDF](https://arxiv.org/pdf/2503.03654)  

**Abstract**: This paper describes the construction of a dataset and the evaluation of training methods to improve generative large language models' (LLMs) ability to answer queries on sensitive topics with a Neutral Point of View (NPOV), i.e., to provide significantly more informative, diverse and impartial answers. The dataset, the SHQ-NPOV dataset, comprises 300 high-quality, human-written quadruplets: a query on a sensitive topic, an answer, an NPOV rating, and a set of links to source texts elaborating the various points of view. The first key contribution of this paper is a new methodology to create such datasets through iterative rounds of human peer-critique and annotator training, which we release alongside the dataset. The second key contribution is the identification of a highly effective training regime for parameter-efficient reinforcement learning (PE-RL) to improve NPOV generation. We compare and extensively evaluate PE-RL and multiple baselines-including LoRA finetuning (a strong baseline), SFT and RLHF.
PE-RL not only improves on overall NPOV quality compared to the strongest baseline ($97.06\%\rightarrow 99.08\%$), but also scores much higher on features linguists identify as key to separating good answers from the best answers ($60.25\%\rightarrow 85.21\%$ for presence of supportive details, $68.74\%\rightarrow 91.43\%$ for absence of oversimplification). A qualitative analysis corroborates this. Finally, our evaluation finds no statistical differences between results on topics that appear in the training dataset and those on separated evaluation topics, which provides strong evidence that our approach to training PE-RL exhibits very effective out of topic generalization. 

---
# Token-Level Privacy in Large Language Models 

**Authors**: Re'em Harel, Niv Gilboa, Yuval Pinter  

**Link**: [PDF](https://arxiv.org/pdf/2503.03652)  

**Abstract**: The use of language models as remote services requires transmitting private information to external providers, raising significant privacy concerns. This process not only risks exposing sensitive data to untrusted service providers but also leaves it vulnerable to interception by eavesdroppers. Existing privacy-preserving methods for natural language processing (NLP) interactions primarily rely on semantic similarity, overlooking the role of contextual information. In this work, we introduce dchi-stencil, a novel token-level privacy-preserving mechanism that integrates contextual and semantic information while ensuring strong privacy guarantees under the dchi differential privacy framework, achieving 2epsilon-dchi-privacy. By incorporating both semantic and contextual nuances, dchi-stencil achieves a robust balance between privacy and utility. We evaluate dchi-stencil using state-of-the-art language models and diverse datasets, achieving comparable and even better trade-off between utility and privacy compared to existing methods. This work highlights the potential of dchi-stencil to set a new standard for privacy-preserving NLP in modern, high-risk applications. 

---
# Psy-Copilot: Visual Chain of Thought for Counseling 

**Authors**: Keqi Chen, Zekai Sun, Huijun Lian, Yingming Gao, Ya Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.03645)  

**Abstract**: Large language models (LLMs) are becoming increasingly popular in the field of psychological counseling. However, when human therapists work with LLMs in therapy sessions, it is hard to understand how the model gives the answers. To address this, we have constructed Psy-COT, a graph designed to visualize the thought processes of LLMs during therapy sessions. The Psy-COT graph presents semi-structured counseling conversations alongside step-by-step annotations that capture the reasoning and insights of therapists. Moreover, we have developed Psy-Copilot, which is a conversational AI assistant designed to assist human psychological therapists in their consultations. It can offer traceable psycho-information based on retrieval, including response candidates, similar dialogue sessions, related strategies, and visual traces of results. We have also built an interactive platform for AI-assisted counseling. It has an interface that displays the relevant parts of the retrieval sub-graph. The Psy-Copilot is designed not to replace psychotherapists but to foster collaboration between AI and human therapists, thereby promoting mental health development. Our code and demo are both open-sourced and available for use. 

---
# Psy-Insight: Explainable Multi-turn Bilingual Dataset for Mental Health Counseling 

**Authors**: Keqi Chen, Zekai Sun, Yuhua Wen, Huijun Lian, Yingming Gao, Ya Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.03607)  

**Abstract**: The in-context learning capabilities of large language models (LLMs) show great potential in mental health support. However, the lack of counseling datasets, particularly in Chinese corpora, restricts their application in this field. To address this, we constructed Psy-Insight, the first mental health-oriented explainable multi-task bilingual dataset. We collected face-to-face multi-turn counseling dialogues, which are annotated with multi-task labels and conversation process explanations. Our annotations include psychotherapy, emotion, strategy, and topic labels, as well as turn-level reasoning and session-level guidance. Psy-Insight is not only suitable for tasks such as label recognition but also meets the need for training LLMs to act as empathetic counselors through logical reasoning. Experiments show that training LLMs on Psy-Insight enables the models to not only mimic the conversation style but also understand the underlying strategies and reasoning of counseling. 

---
# Feature-Level Insights into Artificial Text Detection with Sparse Autoencoders 

**Authors**: Kristian Kuznetsov, Laida Kushnareva, Polina Druzhinina, Anton Razzhigaev, Anastasia Voznyuk, Irina Piontkovskaya, Evgeny Burnaev, Serguei Barannikov  

**Link**: [PDF](https://arxiv.org/pdf/2503.03601)  

**Abstract**: Artificial Text Detection (ATD) is becoming increasingly important with the rise of advanced Large Language Models (LLMs). Despite numerous efforts, no single algorithm performs consistently well across different types of unseen text or guarantees effective generalization to new LLMs. Interpretability plays a crucial role in achieving this goal. In this study, we enhance ATD interpretability by using Sparse Autoencoders (SAE) to extract features from Gemma-2-2b residual stream. We identify both interpretable and efficient features, analyzing their semantics and relevance through domain- and model-specific statistics, a steering approach, and manual or LLM-based interpretation. Our methods offer valuable insights into how texts from various models differ from human-written content. We show that modern LLMs have a distinct writing style, especially in information-dense domains, even though they can produce human-like outputs with personalized prompts. 

---
# Small but Mighty: Enhancing Time Series Forecasting with Lightweight LLMs 

**Authors**: Haoran Fan, Bin Li, Yixuan Weng, Shoujun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.03594)  

**Abstract**: While LLMs have demonstrated remarkable potential in time series forecasting, their practical deployment remains constrained by excessive computational demands and memory footprints. Existing LLM-based approaches typically suffer from three critical limitations: Inefficient parameter utilization in handling numerical time series patterns; Modality misalignment between continuous temporal signals and discrete text embeddings; and Inflexibility for real-time expert knowledge integration. We present SMETimes, the first systematic investigation of sub-3B parameter SLMs for efficient and accurate time series forecasting. Our approach centers on three key innovations: A statistically-enhanced prompting mechanism that bridges numerical time series with textual semantics through descriptive statistical features; A adaptive fusion embedding architecture that aligns temporal patterns with language model token spaces through learnable parameters; And a dynamic mixture-of-experts framework enabled by SLMs' computational efficiency, adaptively combining base predictions with domain-specific models. Extensive evaluations across seven benchmark datasets demonstrate that our 3B-parameter SLM achieves state-of-the-art performance on five primary datasets while maintaining 3.8x faster training and 5.2x lower memory consumption compared to 7B-parameter LLM baselines. Notably, the proposed model exhibits better learning capabilities, achieving 12.3% lower MSE than conventional LLM. Ablation studies validate that our statistical prompting and cross-modal fusion modules respectively contribute 15.7% and 18.2% error reduction in long-horizon forecasting tasks. By redefining the efficiency-accuracy trade-off landscape, this work establishes SLMs as viable alternatives to resource-intensive LLMs for practical time series forecasting. Code and models are available at this https URL. 

---
# English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance 

**Authors**: Karl Audun Borgersen  

**Link**: [PDF](https://arxiv.org/pdf/2503.03592)  

**Abstract**: For consumer usage of locally deployed LLMs, the GGUF format and k_quantization are invaluable tools for maintaining the performance of the original model while reducing it to sizes deployable with consumer-grade hardware. The number of bits dedicated to each weight from the original model is reduced based on how important they are thought to be during model inference. This importance is arrived at through the application of an 'importance matrix'-a relatively small text document meant to be representative of the LLM's standard use-cases. In the vast majority of quants available online, this document is primarily written in English. It was therefore an open question whether performance on English language tasks was preserved through the sacrifice of multilingual performance and whether it can be preserved with alternate importance matrices. This article investigates these hypotheses by quantizing Llama3.3 70B on importance matrices written in three languages (English, Norwegian, and Malayalam) and evaluating them on the MixEval dataset in both English and Norwegian. All experiments related to k_quantization yielded non-significant results (In all cases p > 0.237) indicating that current quantization practices do not disproportionately harm multilingual performance. 

---
# PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention 

**Authors**: Lida Chen, Dong Xu, Chenxin An, Xintao Wang, Yikai Zhang, Jiangjie Chen, Zujie Liang, Feng Wei, Jiaqing Liang, Yanghua Xiao, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03588)  

**Abstract**: Large Language Models (LLMs) face efficiency bottlenecks due to the quadratic complexity of the attention mechanism when processing long contexts. Sparse attention methods offer a promising solution, but existing approaches often suffer from incomplete effective context and/or require complex implementation of pipeline. We present a comprehensive analysis of sparse attention for autoregressive LLMs from the respective of receptive field, recognize the suboptimal nature of existing methods for expanding the receptive field, and introduce PowerAttention, a novel sparse attention design that facilitates effective and complete context extension through the theoretical analysis. PowerAttention achieves exponential receptive field growth in $d$-layer LLMs, allowing each output token to attend to $2^d$ tokens, ensuring completeness and continuity of the receptive field. Experiments demonstrate that PowerAttention outperforms existing static sparse attention methods by $5\sim 40\%$, especially on tasks demanding long-range dependencies like Passkey Retrieval and RULER, while maintaining a comparable time complexity to sliding window attention. Efficiency evaluations further highlight PowerAttention's superior speedup in both prefilling and decoding phases compared with dynamic sparse attentions and full attention ($3.0\times$ faster on 128K context), making it a highly effective and user-friendly solution for processing long sequences in LLMs. 

---
# Scaling Crowdsourced Election Monitoring: Construction and Evaluation of Classification Models for Multilingual and Cross-Domain Classification Settings 

**Authors**: Jabez Magomere, Scott Hale  

**Link**: [PDF](https://arxiv.org/pdf/2503.03582)  

**Abstract**: The adoption of crowdsourced election monitoring as a complementary alternative to traditional election monitoring is on the rise. Yet, its reliance on digital response volunteers to manually process incoming election reports poses a significant scaling bottleneck. In this paper, we address the challenge of scaling crowdsourced election monitoring by advancing the task of automated classification of crowdsourced election reports to multilingual and cross-domain classification settings. We propose a two-step classification approach of first identifying informative reports and then categorising them into distinct information types. We conduct classification experiments using multilingual transformer models such as XLM-RoBERTa and multilingual embeddings such as SBERT, augmented with linguistically motivated features. Our approach achieves F1-Scores of 77\% for informativeness detection and 75\% for information type classification. We conduct cross-domain experiments, applying models trained in a source electoral domain to a new target electoral domain in zero-shot and few-shot classification settings. Our results show promising potential for model transfer across electoral domains, with F1-Scores of 59\% in zero-shot and 63\% in few-shot settings. However, our analysis also reveals a performance bias in detecting informative English reports over Swahili, likely due to imbalances in the training data, indicating a need for caution when deploying classification models in real-world election scenarios. 

---
# An Aspect Extraction Framework using Different Embedding Types, Learning Models, and Dependency Structure 

**Authors**: Ali Erkan, Tunga Güngör  

**Link**: [PDF](https://arxiv.org/pdf/2503.03512)  

**Abstract**: Aspect-based sentiment analysis has gained significant attention in recent years due to its ability to provide fine-grained insights for sentiment expressions related to specific features of entities. An important component of aspect-based sentiment analysis is aspect extraction, which involves identifying and extracting aspect terms from text. Effective aspect extraction serves as the foundation for accurate sentiment analysis at the aspect level. In this paper, we propose aspect extraction models that use different types of embeddings for words and part-of-speech tags and that combine several learning models. We also propose tree positional encoding that is based on dependency parsing output to capture better the aspect positions in sentences. In addition, a new aspect extraction dataset is built for Turkish by machine translating an English dataset in a controlled setting. The experiments conducted on two Turkish datasets showed that the proposed models mostly outperform the studies that use the same datasets, and incorporating tree positional encoding increases the performance of the models. 

---
# CURVALID: Geometrically-guided Adversarial Prompt Detection 

**Authors**: Canaan Yung, Hanxun Huang, Sarah Monazam Erfani, Christopher Leckie  

**Link**: [PDF](https://arxiv.org/pdf/2503.03502)  

**Abstract**: Adversarial prompts capable of jailbreaking large language models (LLMs) and inducing undesirable behaviours pose a significant obstacle to their safe deployment. Current mitigation strategies rely on activating built-in defence mechanisms or fine-tuning the LLMs, but the fundamental distinctions between adversarial and benign prompts are yet to be understood. In this work, we introduce CurvaLID, a novel defense framework that efficiently detects adversarial prompts by leveraging their geometric properties. It is agnostic to the type of LLM, offering a unified detection framework across diverse adversarial prompts and LLM architectures. CurvaLID builds on the geometric analysis of text prompts to uncover their underlying differences. We theoretically extend the concept of curvature via the Whewell equation into an $n$-dimensional word embedding space, enabling us to quantify local geometric properties, including semantic shifts and curvature in the underlying manifolds. Additionally, we employ Local Intrinsic Dimensionality (LID) to capture geometric features of text prompts within adversarial subspaces. Our findings reveal that adversarial prompts differ fundamentally from benign prompts in terms of their geometric characteristics. Our results demonstrate that CurvaLID delivers superior detection and rejection of adversarial queries, paving the way for safer LLM deployment. The source code can be found at this https URL 

---
# Deictic Codes, Demonstratives, and Reference: A Step Toward Solving the Grounding Problem 

**Authors**: Athanassios Raftopoulos, Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2503.03495)  

**Abstract**: In this paper we address the issue of grounding for experiential concepts. Given that perceptual demonstratives are a basic form of such concepts, we examine ways of fixing the referents of such demonstratives. To avoid 'encodingism', that is, relating representations to representations, we postulate that the process of reference fixing must be bottom-up and nonconceptual, so that it can break the circle of conceptual content and touch the world. For that purpose, an appropriate causal relation between representations and the world is needed. We claim that this relation is provided by spatial and object-centered attention that leads to the formation of object files through the function of deictic acts. This entire causal process takes place at a pre-conceptual level, meeting the requirement for a solution to the grounding problem. Finally we claim that our account captures fundamental insights in Putnam's and Kripke's work on "new" reference. 

---
# Enhancing Spoken Discourse Modeling in Language Models Using Gestural Cues 

**Authors**: Varsha Suresh, M. Hamza Mughal, Christian Theobalt, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2503.03474)  

**Abstract**: Research in linguistics shows that non-verbal cues, such as gestures, play a crucial role in spoken discourse. For example, speakers perform hand gestures to indicate topic shifts, helping listeners identify transitions in discourse. In this work, we investigate whether the joint modeling of gestures using human motion sequences and language can improve spoken discourse modeling in language models. To integrate gestures into language models, we first encode 3D human motion sequences into discrete gesture tokens using a VQ-VAE. These gesture token embeddings are then aligned with text embeddings through feature alignment, mapping them into the text embedding space. To evaluate the gesture-aligned language model on spoken discourse, we construct text infilling tasks targeting three key discourse cues grounded in linguistic research: discourse connectives, stance markers, and quantifiers. Results show that incorporating gestures enhances marker prediction accuracy across the three tasks, highlighting the complementary information that gestures can offer in modeling spoken discourse. We view this work as an initial step toward leveraging non-verbal cues to advance spoken language modeling in language models. 

---
# Open-Source Large Language Models as Multilingual Crowdworkers: Synthesizing Open-Domain Dialogues in Several Languages With No Examples in Targets and No Machine Translation 

**Authors**: Ahmed Njifenjou, Virgile Sucal, Bassam Jabaian, Fabrice Lefèvre  

**Link**: [PDF](https://arxiv.org/pdf/2503.03462)  

**Abstract**: The prevailing paradigm in the domain of Open-Domain Dialogue agents predominantly focuses on the English language, encompassing both models and datasets. Furthermore, the financial and temporal investments required for crowdsourcing such datasets for finetuning are substantial, particularly when multiple languages are involved. Fortunately, advancements in Large Language Models (LLMs) have unveiled a plethora of possibilities across diverse tasks. Specifically, instruction-tuning has enabled LLMs to execute tasks based on natural language instructions, occasionally surpassing the performance of human crowdworkers. Additionally, these models possess the capability to function in various languages within a single thread. Consequently, to generate new samples in different languages, we propose leveraging these capabilities to replicate the data collection process. We introduce a pipeline for generating Open-Domain Dialogue data in multiple Target Languages using LLMs, with demonstrations provided in a unique Source Language. By eschewing explicit Machine Translation in this approach, we enhance the adherence to language-specific nuances. We apply this methodology to the PersonaChat dataset. To enhance the openness of generated dialogues and mimic real life scenarii, we added the notion of speech events corresponding to the type of conversation the speakers are involved in and also that of common ground which represents the premises of a conversation. 

---
# Visualising Policy-Reward Interplay to Inform Zeroth-Order Preference Optimisation of Large Language Models 

**Authors**: Alessio Galatolo, Zhenbang Dai, Katie Winkle, Meriem Beloucif  

**Link**: [PDF](https://arxiv.org/pdf/2503.03460)  

**Abstract**: Fine-tuning LLMs with first-order methods like back-propagation is computationally intensive. Zeroth-Order (ZO) optimisation, using function evaluations instead of gradients, reduces memory usage but suffers from slow convergence in high-dimensional models. As a result, ZO research in LLMs has mostly focused on classification, overlooking more complex generative tasks. In this paper, we introduce ZOPrO, a novel ZO algorithm designed for \textit{Preference Optimisation} in LLMs. We begin by analysing the interplay between policy and reward models during traditional (first-order) Preference Optimisation, uncovering patterns in their relative updates. Guided by these insights, we adapt Simultaneous Perturbation Stochastic Approximation (SPSA) with a targeted sampling strategy to accelerate convergence. Through experiments on summarisation, machine translation, and conversational assistants, we demonstrate that our method consistently enhances reward signals while achieving convergence times comparable to first-order methods. While it falls short of some state-of-the-art methods, our work is the first to apply Zeroth-Order methods to Preference Optimisation in LLMs, going beyond classification tasks and paving the way for a largely unexplored research direction. Code and visualisations are available at this https URL 

---
# Taxation Perspectives from Large Language Models: A Case Study on Additional Tax Penalties 

**Authors**: Eunkyung Choi, Young Jin Suh, Hun Park, Wonseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03444)  

**Abstract**: How capable are large language models (LLMs) in the domain of taxation? Although numerous studies have explored the legal domain in general, research dedicated to taxation remain scarce. Moreover, the datasets used in these studies are either simplified, failing to reflect the real-world complexities, or unavailable as open source. To address this gap, we introduce PLAT, a new benchmark designed to assess the ability of LLMs to predict the legitimacy of additional tax penalties. PLAT is constructed to evaluate LLMs' understanding of tax law, particularly in cases where resolving the issue requires more than just applying related statutes. Our experiments with six LLMs reveal that their baseline capabilities are limited, especially when dealing with conflicting issues that demand a comprehensive understanding. However, we found that enabling retrieval, self-reasoning, and discussion among multiple agents with specific role assignments, this limitation can be mitigated. 

---
# RASD: Retrieval-Augmented Speculative Decoding 

**Authors**: Guofeng Quan, Wenfeng Feng, Chuzhan Hao, Guochao Jiang, Yuewei Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03434)  

**Abstract**: Speculative decoding accelerates inference in large language models (LLMs) by generating draft tokens for target model verification. Current approaches for obtaining draft tokens rely on lightweight draft models or additional model structures to generate draft tokens and retrieve context from databases. Due to the draft model's small size and limited training data, model-based speculative decoding frequently becomes less effective in out-of-domain scenarios. Additionally, the time cost of the drafting phase results in a low upper limit on acceptance length during the verification step, limiting overall efficiency. This paper proposes RASD (Retrieval-Augmented Speculative Decoding), which adopts retrieval methods to enhance model-based speculative decoding. We introduce tree pruning and tree fusion to achieve this. Specifically, we develop a pruning method based on the draft model's probability distribution to construct the optimal retrieval tree. Second, we employ the longest prefix matching algorithm to merge the tree generated by the draft model with the retrieval tree, resulting in a unified tree for verification. Experimental results demonstrate that RASD achieves state-of-the-art inference acceleration across tasks such as DocQA, Summary, Code, and In-Domain QA. Moreover, RASD exhibits strong scalability, seamlessly integrating with various speculative decoding approaches, including both generation-based and retrieval-based methods. 

---
# When Claims Evolve: Evaluating and Enhancing the Robustness of Embedding Models Against Misinformation Edits 

**Authors**: Jabez Magomere, Emanuele La Malfa, Manuel Tonneau, Ashkan Kazemi, Scott Hale  

**Link**: [PDF](https://arxiv.org/pdf/2503.03417)  

**Abstract**: Online misinformation remains a critical challenge, and fact-checkers increasingly rely on embedding-based methods to retrieve relevant fact-checks. Yet, when debunked claims reappear in edited forms, the performance of these methods is unclear. In this work, we introduce a taxonomy of six common real-world misinformation edits and propose a perturbation framework that generates valid, natural claim variations. Our multi-stage retrieval evaluation reveals that standard embedding models struggle with user-introduced edits, while LLM-distilled embeddings offer improved robustness at a higher computational cost. Although a strong reranker helps mitigate some issues, it cannot fully compensate for first-stage retrieval gaps. Addressing these retrieval gaps, our train- and inference-time mitigation approaches enhance in-domain robustness by up to 17 percentage points and boost out-of-domain generalization by 10 percentage points over baseline models. Overall, our findings provide practical improvements to claim-matching systems, enabling more reliable fact-checking of evolving misinformation. 

---
# The Serendipity of Claude AI: Case of the 13 Low-Resource National Languages of Mali 

**Authors**: Alou Dembele, Nouhoum Souleymane Coulibaly, Michael Leventhal  

**Link**: [PDF](https://arxiv.org/pdf/2503.03380)  

**Abstract**: Recent advances in artificial intelligence (AI) and natural language processing (NLP) have improved the representation of underrepresented languages. However, most languages, including Mali's 13 official national languages, continue to be poorly supported or unsupported by automatic translation and generative AI. This situation appears to have slightly improved with certain recent LLM releases. The study evaluated Claude AI's translation performance on each of the 13 national languages of Mali. In addition to ChrF2 and BLEU scores, human evaluators assessed translation accuracy, contextual consistency, robustness to dialect variations, management of linguistic bias, adaptation to a limited corpus, and ease of understanding. The study found that Claude AI performs robustly for languages with very modest language resources and, while unable to produce understandable and coherent texts for Malian languages with minimal resources, still manages to produce results which demonstrate the ability to mimic some elements of the language. 

---
# EnigmaToM: Improve LLMs' Theory-of-Mind Reasoning Capabilities with Neural Knowledge Base of Entity States 

**Authors**: Hainiu Xu, Siya Qi, Jiazheng Li, Yuxiang Zhou, Jinhua Du, Caroline Catmur, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2503.03340)  

**Abstract**: Theory-of-Mind (ToM), the ability to infer others' perceptions and mental states, is fundamental to human interaction but remains a challenging task for Large Language Models (LLMs). While existing ToM reasoning methods show promise with reasoning via perceptual perspective-taking, they often rely excessively on LLMs, reducing their efficiency and limiting their applicability to high-order ToM reasoning, which requires multi-hop reasoning about characters' beliefs. To address these issues, we present EnigmaToM, a novel neuro-symbolic framework that enhances ToM reasoning by integrating a Neural Knowledge Base of entity states (Enigma) for (1) a psychology-inspired iterative masking mechanism that facilitates accurate perspective-taking and (2) knowledge injection that elicits key entity information. Enigma generates structured representations of entity states, which construct spatial scene graphs -- leveraging spatial information as an inductive bias -- for belief tracking of various ToM orders and enhancing events with fine-grained entity state details. Experimental results on multiple benchmarks, including ToMi, HiToM, and FANToM, show that EnigmaToM significantly improves ToM reasoning across LLMs of varying sizes, particularly excelling in high-order reasoning scenarios. 

---
# iNews: A Multimodal Dataset for Modeling Personalized Affective Responses to News 

**Authors**: Tiancheng Hu, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2503.03335)  

**Abstract**: Current approaches to emotion detection often overlook the inherent subjectivity of affective experiences, instead relying on aggregated labels that mask individual variations in emotional responses. We introduce iNews, a novel large-scale dataset explicitly capturing subjective affective responses to news headlines. Our dataset comprises annotations from 291 demographically diverse UK participants across 2,899 multimodal Facebook news posts from major UK outlets, with an average of 5.18 annotators per sample. For each post, annotators provide multifaceted labels including valence, arousal, dominance, discrete emotions, content relevance judgments, sharing likelihood, and modality importance ratings (text, image, or both). Furthermore, we collect comprehensive annotator persona information covering demographics, personality, media trust, and consumption patterns, which explain 15.2% of annotation variance - higher than existing NLP datasets. Incorporating this information yields a 7% accuracy gain in zero-shot prediction and remains beneficial even with 32-shot. iNews will enhance research in LLM personalization, subjectivity, affective computing, and individual-level behavior simulation. 

---
# The Box is in the Pen: Evaluating Commonsense Reasoning in Neural Machine Translation 

**Authors**: Jie He, Tao Wang, Deyi Xiong, Qun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.03308)  

**Abstract**: Does neural machine translation yield translations that are congenial with common sense? In this paper, we present a test suite to evaluate the commonsense reasoning capability of neural machine translation. The test suite consists of three test sets, covering lexical and contextless/contextual syntactic ambiguity that requires commonsense knowledge to resolve. We manually create 1,200 triples, each of which contain a source sentence and two contrastive translations, involving 7 different common sense types. Language models pretrained on large-scale corpora, such as BERT, GPT-2, achieve a commonsense reasoning accuracy of lower than 72% on target translations of this test suite. We conduct extensive experiments on the test suite to evaluate commonsense reasoning in neural machine translation and investigate factors that have impact on this capability. Our experiments and analyses demonstrate that neural machine translation performs poorly on commonsense reasoning of the three ambiguity types in terms of both reasoning accuracy (60.1%) and reasoning consistency (31%). The built commonsense test suite is available at this https URL. 

---
# SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection 

**Authors**: Yi-Fan Lu, Xian-Ling Mao, Tian Lan, Tong Zhang, Yu-Shi Zhu, Heyan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03303)  

**Abstract**: Automatic evaluation for Open Domain Event Detection (ODED) is a highly challenging task, because ODED is characterized by a vast diversity of un-constrained output labels from various domains. Nearly all existing evaluation methods for ODED usually first construct evaluation benchmarks with limited labels and domain coverage, and then evaluate ODED methods using metrics based on token-level label matching rules. However, this kind of evaluation framework faces two issues: (1) The limited evaluation benchmarks lack representatives of the real world, making it difficult to accurately reflect the performance of various ODED methods in real-world scenarios; (2) Evaluation metrics based on token-level matching rules fail to capture semantic similarity between predictions and golden labels. To address these two problems above, we propose a scalable and reliable Semantic-level Evaluation framework for Open domain Event detection (SEOE) by constructing a more representative evaluation benchmark and introducing a semantic evaluation metric. Specifically, our proposed framework first constructs a scalable evaluation benchmark that currently includes 564 event types covering 7 major domains, with a cost-effective supplementary annotation strategy to ensure the benchmark's representativeness. The strategy also allows for the supplement of new event types and domains in the future. Then, the proposed SEOE leverages large language models (LLMs) as automatic evaluation agents to compute a semantic F1-score, incorporating fine-grained definitions of semantically similar labels to enhance the reliability of the evaluation. Extensive experiments validate the representatives of the benchmark and the reliability of the semantic evaluation metric. Existing ODED methods are thoroughly evaluated, and the error patterns of predictions are analyzed, revealing several insightful findings. 

---
# LexGenie: Automated Generation of Structured Reports for European Court of Human Rights Case Law 

**Authors**: T.Y.S.S Santosh, Mahmoud Aly, Oana Ichim, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2503.03266)  

**Abstract**: Analyzing large volumes of case law to uncover evolving legal principles, across multiple cases, on a given topic is a demanding task for legal professionals. Structured topical reports provide an effective solution by summarizing key issues, principles, and judgments, enabling comprehensive legal analysis on a particular topic. While prior works have advanced query-based individual case summarization, none have extended to automatically generating multi-case structured reports. To address this, we introduce LexGenie, an automated LLM-based pipeline designed to create structured reports using the entire body of case law on user-specified topics within the European Court of Human Rights jurisdiction. LexGenie retrieves, clusters, and organizes relevant passages by topic to generate a structured outline and cohesive content for each section. Expert evaluation confirms LexGenie's utility in producing structured reports that enhance efficient, scalable legal analysis. 

---
# Can Frontier LLMs Replace Annotators in Biomedical Text Mining? Analyzing Challenges and Exploring Solutions 

**Authors**: Yichong Zhao, Susumu Goto  

**Link**: [PDF](https://arxiv.org/pdf/2503.03261)  

**Abstract**: Large language models (LLMs) can perform various natural language processing (NLP) tasks through in-context learning without relying on supervised data. However, multiple previous studies have reported suboptimal performance of LLMs in biological text mining. By analyzing failure patterns in these evaluations, we identified three primary challenges for LLMs in biomedical corpora: (1) LLMs fail to learn implicit dataset-specific nuances from supervised data, (2) The common formatting requirements of discriminative tasks limit the reasoning capabilities of LLMs particularly for LLMs that lack test-time compute, and (3) LLMs struggle to adhere to annotation guidelines and match exact schemas, which hinders their ability to understand detailed annotation requirements which is essential in biomedical annotation workflow. To address these challenges, we experimented with prompt engineering techniques targeted to the above issues, and developed a pipeline that dynamically extracts instructions from annotation guidelines. Our findings show that frontier LLMs can approach or surpass the performance of state-of-the-art (SOTA) BERT-based models with minimal reliance on manually annotated data and without fine-tuning. Furthermore, we performed model distillation on a closed-source LLM, demonstrating that a BERT model trained exclusively on synthetic data annotated by LLMs can also achieve a practical performance. Based on these results, we explored the feasibility of partially replacing manual annotation with LLMs in production scenarios for biomedical text mining. 

---
# FANS -- Formal Answer Selection for Natural Language Math Reasoning Using Lean4 

**Authors**: Jiarui Yao, Ruida Wang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03238)  

**Abstract**: Large Language Models (LLMs) have displayed astonishing abilities in various tasks, especially in text generation, classification, question answering, etc. However, the reasoning ability of LLMs still faces many debates. The inherent ambiguity of Natural Language (NL) limits LLMs' ability to perform verifiable reasoning, making its answers lack coherence and trustworthy support. To tackle the above problems, we propose a novel framework named FANS: Formal ANswer Selection for Natural Language Math Reasoning Using Lean4. To the best of our knowledge, it is the first framework that utilizes Lean4 to enhance LLMs' NL math reasoning ability. In particular, given an NL math question and LLM-generated answers, FANS first translates it into Lean4 theorem statements. Then it tries to prove it using a Lean4 prover and verify it by Lean4. Finally, it uses the FL result to assist in answer selection. It enhances LLMs' NL math ability in providing a computer-verifiable solution for its correct answer and proposes an alternative method for answer selection beyond the reward model. Extensive experiments indicate the effectiveness of our framework. It can improve the accuracy rate of reward model enhanced LLMs in the MATH-500 dataset by at most 1.91% and AMC-23 by at most 8.33% on strong reward-model baselines. In some particular fields like number theory that Lean4 experts in, we can even select all correct solutions. The qualitative analysis also shows our framework can make NL results formally backed by Lean4 proofs. As a pioneering work in the corresponding field, we will open-source all our models and datasets to further boost the development of the field. 

---
# Targeted Distillation for Sentiment Analysis 

**Authors**: Yice Zhang, Guangyu Xie, Jingjie Lin, Jianzhu Bao, Qianlong Wang, Xi Zeng, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.03225)  

**Abstract**: This paper presents a compact model that achieves strong sentiment analysis capabilities through targeted distillation from advanced large language models (LLMs). Our methodology decouples the distillation target into two key components: sentiment-related knowledge and task alignment. To transfer these components, we propose a two-stage distillation framework. The first stage, knowledge-driven distillation (\textsc{KnowDist}), transfers sentiment-related knowledge to enhance fundamental sentiment analysis capabilities. The second stage, in-context learning distillation (\textsc{ICLDist}), transfers task-specific prompt-following abilities to optimize task alignment. For evaluation, we introduce \textsc{SentiBench}, a comprehensive sentiment analysis benchmark comprising 3 task categories across 12 datasets. Experiments on this benchmark demonstrate that our model effectively balances model size and performance, showing strong competitiveness compared to existing small-scale LLMs. 

---
# MA-LoT: Multi-Agent Lean-based Long Chain-of-Thought Reasoning enhances Formal Theorem Proving 

**Authors**: Ruida Wang, Rui Pan, Yuxin Li, Jipeng Zhang, Yizhen Jia, Shizhe Diao, Renjie Pi, Junjie Hu, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03205)  

**Abstract**: Solving mathematical problems using computer-verifiable languages like Lean has significantly impacted mathematical and computer science communities. State-of-the-art methods utilize single Large Language Models (LLMs) as agents or provers to either generate complete proof or perform tree searches. However, single-agent methods inherently lack a structured way to combine high-level reasoning in Natural Language (NL) with Formal Language (FL) verification feedback. To solve these issues, we propose MA-LoT: Multi-Agent Lean-based Long Chain-of-Thought framework, (to the best of our knowledge), the first multi-agent framework for Lean4 theorem proving that balance high-level NL reasoning and FL verification in Long CoT. Using this structured interaction, our approach enables deeper insights and long-term coherence in proof generation, with which past methods struggle. We do this by leveraging emergent formal reasoning ability in Long CoT using our novel LoT-Transfer Learning training-inference pipeline. Extensive experiments show that our framework achieves 54.51% accuracy rate on the Lean4 version of MiniF2F-Test dataset, largely outperforming GPT-4 (22.95%), single-agent tree search (InternLM-Step-Prover, 50.70%), and whole-proof generation (DeepSeek-Prover-v1.5, 48.36%) baselines. Furthermore, our findings highlight the potential of combining Long CoT with formal verification for a more insightful generation in a broader perspective. 

---
# Towards Robust Universal Information Extraction: Benchmark, Evaluation, and Solution 

**Authors**: Jizhao Zhu, Akang Shi, Zixuan Li, Long Bai, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.03201)  

**Abstract**: In this paper, we aim to enhance the robustness of Universal Information Extraction (UIE) by introducing a new benchmark dataset, a comprehensive evaluation, and a feasible solution. Existing robust benchmark datasets have two key limitations: 1) They generate only a limited range of perturbations for a single Information Extraction (IE) task, which fails to evaluate the robustness of UIE models effectively; 2) They rely on small models or handcrafted rules to generate perturbations, often resulting in unnatural adversarial examples. Considering the powerful generation capabilities of Large Language Models (LLMs), we introduce a new benchmark dataset for Robust UIE, called RUIE-Bench, which utilizes LLMs to generate more diverse and realistic perturbations across different IE tasks. Based on this dataset, we comprehensively evaluate existing UIE models and reveal that both LLM-based models and other models suffer from significant performance drops. To improve robustness and reduce training costs, we propose a data-augmentation solution that dynamically selects hard samples for iterative training based on the model's inference loss. Experimental results show that training with only \textbf{15\%} of the data leads to an average \textbf{7.5\%} relative performance improvement across three IE tasks. 

---
# Structured Outputs Enable General-Purpose LLMs to be Medical Experts 

**Authors**: Guangfu Guo, Kai Zhang, Bryan Hoo, Yujun Cai, Xiaoqian Lu, Nanyun Peng, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03194)  

**Abstract**: Medical question-answering (QA) is a critical task for evaluating how effectively large language models (LLMs) encode clinical knowledge and assessing their potential applications in medicine. Despite showing promise on multiple-choice tests, LLMs frequently struggle with open-ended medical questions, producing responses with dangerous hallucinations or lacking comprehensive coverage of critical aspects. Existing approaches attempt to address these challenges through domain-specific fine-tuning, but this proves resource-intensive and difficult to scale across models. To improve the comprehensiveness and factuality of medical responses, we propose a novel approach utilizing structured medical reasoning. Our method guides LLMs through an seven-step cognitive process inspired by clinical diagnosis, enabling more accurate and complete answers without additional training. Experiments on the MedLFQA benchmark demonstrate that our approach achieves the highest Factuality Score of 85.8, surpassing fine-tuned models. Notably, this improvement transfers to smaller models, highlighting the method's efficiency and scalability. Our code and datasets are available. 

---
# Designing Speech Technologies for Australian Aboriginal English: Opportunities, Risks and Participation 

**Authors**: Ben Hutchinson, Celeste Rodríguez Louro, Glenys Collard, Ned Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2503.03186)  

**Abstract**: In Australia, post-contact language varieties, including creoles and local varieties of international languages, emerged as a result of forced contact between Indigenous communities and English speakers. These contact varieties are widely used, yet are poorly supported by language technologies. This gap presents barriers to participation in civil and economic society for Indigenous communities using these varieties, and reproduces minoritisation of contemporary Indigenous sociolinguistic identities. This paper concerns three questions regarding this context. First, can speech technologies support speakers of Australian Aboriginal English, a local indigenised variety of English? Second, what risks are inherent in such a project? Third, what technology development practices are appropriate for this context, and how can researchers integrate meaningful community participation in order to mitigate risks? We argue that opportunities do exist -- as well as risks -- and demonstrate this through a case study exploring design practices in a real-world project aiming to improve speech technologies for Australian Aboriginal English. We discuss how we integrated culturally appropriate and participatory processes throughout the project. We call for increased support for languages used by Indigenous communities, including contact varieties, which provide practical economic and socio-cultural benefits, provided that participatory and culturally safe practices are enacted. 

---
# Intermediate-Task Transfer Learning: Leveraging Sarcasm Detection for Stance Detection 

**Authors**: Gibson Nkhata, Susan Gauch  

**Link**: [PDF](https://arxiv.org/pdf/2503.03172)  

**Abstract**: Stance Detection (SD) on social media has emerged as a prominent area of interest with implications for social business and political applications thereby garnering escalating research attention within NLP. The inherent subtlety and complexity of texts procured from online platforms pose challenges for SD algorithms in accurately discerning the authors stance. Mostly the inclusion of sarcastic and figurative language drastically impacts the performance of SD models. This paper addresses this by employing sarcasm detection intermediate-task transfer learning tailored for SD. The proposed methodology involves the finetuning of BERT and RoBERTa and the concatenation of convolutional BiLSTM and dense layers. Rigorous experiments are conducted on publicly available datasets to evaluate our transfer-learning framework. The performance of the approach is assessed against various State-Of-The-Art baselines for SD providing empirical evidence of its effectiveness. Notably our model outperforms the best SOTA models even prior to sarcasm-detection pretraining. The integration of sarcasm knowledge into the model proves instrumental in mitigating misclassifications of sarcastic textual elements in SD. Our model accurately predicts 85% of texts that were previously misclassified by the model without sarcasm-detection pretraining thereby amplifying the average F1-score of the model. Our experiments also revealed that the success of the transfer-learning framework is contingent upon the correlation of lexical attributes between the intermediate task and the target task. This study represents the first exploration of sarcasm detection as an intermediate transfer-learning task in the context of SD and simultaneously uses the concatenation of BERT or RoBERTa with other deep-learning techniques establishing the proposed approach as a foundational baseline for future research endeavors in this domain. 

---
# DSVD: Dynamic Self-Verify Decoding for Faithful Generation in Large Language Models 

**Authors**: YiQiu Guo, Yuchen Yang, Zhe Chen, Pingjie Wang, Yusheng Liao, Ya Zhang, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03149)  

**Abstract**: The reliability of large language models remains a critical challenge, particularly due to their susceptibility to hallucinations and factual inaccuracies during text generation. Existing solutions either underutilize models' self-correction with preemptive strategies or use costly post-hoc verification. To further explore the potential of real-time self-verification and correction, we present Dynamic Self-Verify Decoding (DSVD), a novel decoding framework that enhances generation reliability through real-time hallucination detection and efficient error correction. DSVD integrates two key components: (1) parallel self-verification architecture for continuous quality assessment, (2) dynamic rollback mechanism for targeted error recovery. Extensive experiments across five benchmarks demonstrate DSVD's effectiveness, achieving significant improvement in truthfulness (Quesetion-Answering) and factual accuracy (FActScore). Results show the DSVD can be further incorporated with existing faithful decoding methods to achieve stronger performance. Our work establishes that real-time self-verification during generation offers a viable path toward more trustworthy language models without sacrificing practical deployability. 

---
# The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models 

**Authors**: Zichao Li, Xueru Wen, Jie Lou, Yuqiu Ji, Yaojie Lu, Xianpei Han, Debing Zhang, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.03122)  

**Abstract**: Multimodal Reward Models (MM-RMs) are crucial for aligning Large Language Models (LLMs) with human preferences, particularly as LLMs increasingly interact with multimodal data. However, we find that MM-RMs trained on existing datasets often struggle to generalize to out-of-distribution data due to their reliance on unimodal spurious correlations, primarily text-only shortcuts within the training distribution, which prevents them from leveraging true multimodal reward functions. To address this, we introduce a Shortcut-aware MM-RM learning algorithm that mitigates this issue by dynamically reweighting training samples, shifting the distribution toward better multimodal understanding, and reducing dependence on unimodal spurious correlations. Our experiments demonstrate significant improvements in generalization, downstream task performance, and scalability, establishing a more robust framework for multimodal reward modeling. 

---
# External Reliable Information-enhanced Multimodal Contrastive Learning for Fake News Detection 

**Authors**: Biwei Cao, Qihang Wu, Jiuxin Cao, Bo Liu, Jie Gui  

**Link**: [PDF](https://arxiv.org/pdf/2503.03107)  

**Abstract**: With the rapid development of the Internet, the information dissemination paradigm has changed and the efficiency has been improved greatly. While this also brings the quick spread of fake news and leads to negative impacts on cyberspace. Currently, the information presentation formats have evolved gradually, with the news formats shifting from texts to multimodal contents. As a result, detecting multimodal fake news has become one of the research hotspots. However, multimodal fake news detection research field still faces two main challenges: the inability to fully and effectively utilize multimodal information for detection, and the low credibility or static nature of the introduced external information, which limits dynamic updates. To bridge the gaps, we propose ERIC-FND, an external reliable information-enhanced multimodal contrastive learning framework for fake news detection. ERIC-FND strengthens the representation of news contents by entity-enriched external information enhancement method. It also enriches the multimodal news information via multimodal semantic interaction method where the multimodal constrative learning is employed to make different modality representations learn from each other. Moreover, an adaptive fusion method is taken to integrate the news representations from different dimensions for the eventual classification. Experiments are done on two commonly used datasets in different languages, X (Twitter) and Weibo. Experiment results demonstrate that our proposed model ERIC-FND outperforms existing state-of-the-art fake news detection methods under the same settings. 

---
# Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation 

**Authors**: Yurui Chang, Bochuan Cao, Lu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.03106)  

**Abstract**: While large language models have demonstrated exceptional performance across a wide range of tasks, they remain susceptible to hallucinations -- generating plausible yet factually incorrect contents. Existing methods to mitigating such risk often rely on sampling multiple full-length generations, which introduces significant response latency and becomes ineffective when the model consistently produces hallucinated outputs with high confidence. To address these limitations, we introduce Monitoring Decoding (MD), a novel framework that dynamically monitors the generation process and selectively applies in-process interventions, focusing on revising crucial tokens responsible for hallucinations. Instead of waiting until completion of multiple full-length generations, we identify hallucination-prone tokens during generation using a monitor function, and further refine these tokens through a tree-based decoding strategy. This approach ensures an enhanced factual accuracy and coherence in the generated output while maintaining efficiency. Experimental results demonstrate that MD consistently outperforms self-consistency-based approaches in both effectiveness and efficiency, achieving higher factual accuracy while significantly reducing computational overhead. 

---
# MuCo-KGC: Multi-Context-Aware Knowledge Graph Completion 

**Authors**: Haji Gul, Ajaz Ahmad Bhat, Abdul Ghani Haji Naim  

**Link**: [PDF](https://arxiv.org/pdf/2503.03091)  

**Abstract**: Knowledge graph completion (KGC) seeks to predict missing entities (e.g., heads or tails) or relationships in knowledge graphs (KGs), which often contain incomplete data. Traditional embedding-based methods, such as TransE and ComplEx, have improved tail entity prediction but struggle to generalize to unseen entities during testing. Textual-based models mitigate this issue by leveraging additional semantic context; however, their reliance on negative triplet sampling introduces high computational overhead, semantic inconsistencies, and data imbalance. Recent approaches, like KG-BERT, show promise but depend heavily on entity descriptions, which are often unavailable in KGs. Critically, existing methods overlook valuable structural information in the KG related to the entities and relationships. To address these challenges, we propose Multi-Context-Aware Knowledge Graph Completion (MuCo-KGC), a novel model that utilizes contextual information from linked entities and relations within the graph to predict tail entities. MuCo-KGC eliminates the need for entity descriptions and negative triplet sampling, significantly reducing computational complexity while enhancing performance. Our experiments on standard datasets, including FB15k-237, WN18RR, CoDEx-S, and CoDEx-M, demonstrate that MuCo-KGC outperforms state-of-the-art methods on three datasets. Notably, MuCo-KGC improves MRR on WN18RR, and CoDEx-S and CoDEx-M datasets by $1.63\%$, and $3.77\%$ and $20.15\%$ respectively, demonstrating its effectiveness for KGC tasks. 

---
# Improving LLM-as-a-Judge Inference with the Judgment Distribution 

**Authors**: Victor Wang, Michael J.Q. Zhang, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.03064)  

**Abstract**: Using language models to scalably approximate human preferences on text quality (LLM-as-a-judge) has become a standard practice applicable to many tasks. A judgment is often extracted from the judge's textual output alone, typically with greedy decoding. However, LLM judges naturally provide distributions over judgment tokens, inviting a breadth of inference methods for extracting fine-grained preferences. We find that taking the mean of the judgment distribution consistently outperforms taking the mode (i.e. greedy decoding) in all evaluation settings (i.e. pointwise, pairwise, and listwise). We further explore novel methods of deriving preferences from judgment distributions, and find that methods incorporating risk aversion often improve performance. Lastly, we analyze LLM-as-a-judge paired with chain-of-thought (CoT) prompting, showing that CoT can collapse the spread of the judgment distribution, often harming performance. Our findings suggest leveraging distributional output can improve LLM-as-a-judge, as opposed to using the text interface alone. 

---
# Semi-Supervised In-Context Learning: A Baseline Study 

**Authors**: Zhengyao Gu, Henry Peng Zou, Yankai Chen, Aiwei Liu, Weizhi Zhang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.03062)  

**Abstract**: Most existing work in data selection for In-Context Learning (ICL) has focused on constructing demonstrations from ground truth annotations, with limited attention given to selecting reliable self-generated annotations. In this work, we propose a three-step semi-supervised ICL framework: annotation generation, demonstration selection, and semi-supervised inference. Our baseline, Naive-SemiICL, which prompts select high-confidence self-generated demonstrations for ICL prompting, outperforms a 16-shot baseline by an average of 9.94% across 16 datasets. We further introduce IterPSD, an annotation approach that refines pseudo-demonstrations iteratively, achieving up to 6.8% additional gains in classification tasks. Lastly, we reveal a scaling law for semi-supervised ICL, where models achieve optimal performance with over 1,000 demonstrations. 

---
# QE4PE: Word-level Quality Estimation for Human Post-Editing 

**Authors**: Gabriele Sarti, Vilém Zouhar, Grzegorz Chrupała, Ana Guerberof-Arenas, Malvina Nissim, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2503.03044)  

**Abstract**: Word-level quality estimation (QE) detects erroneous spans in machine translations, which can direct and facilitate human post-editing. While the accuracy of word-level QE systems has been assessed extensively, their usability and downstream influence on the speed, quality and editing choices of human post-editing remain understudied. Our QE4PE study investigates the impact of word-level QE on machine translation (MT) post-editing in a realistic setting involving 42 professional post-editors across two translation directions. We compare four error-span highlight modalities, including supervised and uncertainty-based word-level QE methods, for identifying potential errors in the outputs of a state-of-the-art neural MT model. Post-editing effort and productivity are estimated by behavioral logs, while quality improvements are assessed by word- and segment-level human annotation. We find that domain, language and editors' speed are critical factors in determining highlights' effectiveness, with modest differences between human-made and automated QE highlights underlining a gap between accuracy and usability in professional workflows. 

---
# SAGE: Steering and Refining Dialog Generation with State-Action Augmentation 

**Authors**: Yizhe Zhang, Navdeep Jaitly  

**Link**: [PDF](https://arxiv.org/pdf/2503.03040)  

**Abstract**: Recent advances in large language models have demonstrated impressive capabilities in task-oriented applications, yet building emotionally intelligent chatbots that can engage in natural, strategic conversations remains a challenge. We present a novel approach called SAGE that uses latent variables to control long-horizon behavior in dialogue generation. At the core of our method is the State-Action Chain (SAC), which augments standard language model fine-tuning by introducing latent variables that encapsulate emotional states and conversational strategies between dialogue turns. During inference, these variables are generated before each response, enabling coarse-grained control over dialogue progression while maintaining natural interaction patterns. We also introduce a self-improvement pipeline that leverages dialogue tree search, LLM-based reward modeling, and targeted fine-tuning to optimize conversational trajectories. Our experimental results show that models trained with this approach demonstrate improved performance in emotional intelligence metrics while maintaining strong capabilities on LLM benchmarks. The discrete nature of our latent variables facilitates search-based strategies and provides a foundation for future applications of reinforcement learning to dialogue systems, where learning can occur at the state level rather than the token level. 

---
# SAFE: A Sparse Autoencoder-Based Framework for Robust Query Enrichment and Hallucination Mitigation in LLMs 

**Authors**: Samir Abdaljalil, Filippo Pallucchini, Andrea Seveso, Hasan Kurban, Fabio Mercorio, Erchin Serpedin  

**Link**: [PDF](https://arxiv.org/pdf/2503.03032)  

**Abstract**: Despite the state-of-the-art performance of Large Language Models (LLMs), these models often suffer from hallucinations, which can undermine their performance in critical applications. In this work, we propose SAFE, a novel method for detecting and mitigating hallucinations by leveraging Sparse Autoencoders (SAEs). While hallucination detection techniques and SAEs have been explored independently, their synergistic application in a comprehensive system, particularly for hallucination-aware query enrichment, has not been fully investigated. To validate the effectiveness of SAFE, we evaluate it on two models with available SAEs across three diverse cross-domain datasets designed to assess hallucination problems. Empirical results demonstrate that SAFE consistently improves query generation accuracy and mitigates hallucinations across all datasets, achieving accuracy improvements of up to 29.45%. 

---
# One Model to Train them All: Hierarchical Self-Distillation for Enhanced Early Layer Embeddings 

**Authors**: Andrea Gurioli, Federico Pennino, João Monteiro, Maurizio Gabbrielli  

**Link**: [PDF](https://arxiv.org/pdf/2503.03008)  

**Abstract**: Deploying language models often requires handling model size vs. performance trade-offs to satisfy downstream latency constraints while preserving the model's usefulness. Model distillation is commonly employed to reduce model size while maintaining acceptable performance. However, distillation can be inefficient since it involves multiple training steps. In this work, we introduce MODULARSTARENCODER, a modular multi-exit encoder with 1B parameters, useful for multiple tasks within the scope of code retrieval. MODULARSTARENCODER is trained with a novel self-distillation mechanism that significantly improves lower-layer representations-allowing different portions of the model to be used while still maintaining a good trade-off in terms of performance. Our architecture focuses on enhancing text-to-code and code-to-code search by systematically capturing syntactic and semantic structures across multiple levels of representation. Specific encoder layers are targeted as exit heads, allowing higher layers to guide earlier layers during training. This self-distillation effect improves intermediate representations, increasing retrieval recall at no extra training cost. In addition to the multi-exit scheme, our approach integrates a repository-level contextual loss that maximally utilizes the training context window, further enhancing the learned representations. We also release a new dataset constructed via code translation, seamlessly expanding traditional text-to-code benchmarks with code-to-code pairs across diverse programming languages. Experimental results highlight the benefits of self-distillation through multi-exit supervision. 

---
# Will I Get Hate Speech Predicting the Volume of Abusive Replies before Posting in Social Media 

**Authors**: Raneem Alharthia, Rajwa Alharthib, Ravi Shekharc, Aiqi Jiangd, Arkaitz Zubiagaa  

**Link**: [PDF](https://arxiv.org/pdf/2503.03005)  

**Abstract**: Despite the growing body of research tackling offensive language in social media, this research is predominantly reactive, determining if content already posted in social media is abusive. There is a gap in predictive approaches, which we address in our study by enabling to predict the volume of abusive replies a tweet will receive after being posted. We formulate the problem from the perspective of a social media user asking: ``if I post a certain message on social media, is it possible to predict the volume of abusive replies it might receive?'' We look at four types of features, namely text, text metadata, tweet metadata, and account features, which also help us understand the extent to which the user or the content helps predict the number of abusive replies. This, in turn, helps us develop a model to support social media users in finding the best way to post content. One of our objectives is also to determine the extent to which the volume of abusive replies that a tweet will get are motivated by the content of the tweet or by the identity of the user posting it. Our study finds that one can build a model that performs competitively by developing a comprehensive set of features derived from the content of the message that is going to be posted. In addition, our study suggests that features derived from the user's identity do not impact model performance, hence suggesting that it is especially the content of a post that triggers abusive replies rather than who the user is. 

---
# Zero-Shot Multi-Label Classification of Bangla Documents: Large Decoders Vs. Classic Encoders 

**Authors**: Souvika Sarkar, Md. Najib Hasan, Santu Karmaker  

**Link**: [PDF](https://arxiv.org/pdf/2503.02993)  

**Abstract**: Bangla, a language spoken by over 300 million native speakers and ranked as the sixth most spoken language worldwide, presents unique challenges in natural language processing (NLP) due to its complex morphological characteristics and limited resources. While recent Large Decoder Based models (LLMs), such as GPT, LLaMA, and DeepSeek, have demonstrated excellent performance across many NLP tasks, their effectiveness in Bangla remains largely unexplored. In this paper, we establish the first benchmark comparing decoder-based LLMs with classic encoder-based models for Zero-Shot Multi-Label Classification (Zero-Shot-MLC) task in Bangla. Our evaluation of 32 state-of-the-art models reveals that, existing so-called powerful encoders and decoders still struggle to achieve high accuracy on the Bangla Zero-Shot-MLC task, suggesting a need for more research and resources for Bangla NLP. 

---
# Effectively Steer LLM To Follow Preference via Building Confident Directions 

**Authors**: Bingqing Song, Boran Han, Shuai Zhang, Hao Wang, Haoyang Fang, Bonan Min, Yuyang Wang, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.02989)  

**Abstract**: Having an LLM that aligns with human preferences is essential for accommodating individual needs, such as maintaining writing style or generating specific topics of interest. The majority of current alignment methods rely on fine-tuning or prompting, which can be either costly or difficult to control. Model steering algorithms, which modify the model output by constructing specific steering directions, are typically easy to implement and optimization-free. However, their capabilities are typically limited to steering the model into one of the two directions (i.e., bidirectional steering), and there has been no theoretical understanding to guarantee their performance. In this work, we propose a theoretical framework to understand and quantify the model steering methods. Inspired by the framework, we propose a confident direction steering method (CONFST) that steers LLMs via modifying their activations at inference time. More specifically, CONFST builds a confident direction that is closely aligned with users' preferences, and this direction is then added to the activations of the LLMs to effectively steer the model output. Our approach offers three key advantages over popular bidirectional model steering methods: 1) It is more powerful, since multiple (i.e. more than two) users' preferences can be aligned simultaneously; 2) It is simple to implement, since there is no need to determine which layer to add the steering vector to; 3) No explicit user instruction is required. We validate our method on GPT-2 XL (1.5B), Mistral (7B) and Gemma-it (9B) models for tasks that require shifting the output of LLMs across various topics and styles, achieving superior performance over competing methods. 

---
# LINGOLY-TOO: Disentangling Memorisation from Reasoning with Linguistic Templatisation and Orthographic Obfuscation 

**Authors**: Jude Khouja, Karolina Korgul, Simi Hellsten, Lingyi Yang, Vlad Neacs, Harry Mayne, Ryan Kearns, Andrew Bean, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2503.02972)  

**Abstract**: Effective evaluation of the reasoning capabilities of large language models (LLMs) are susceptible to overestimation due to data exposure of evaluation benchmarks. We introduce a framework for producing linguistic reasoning problems that reduces the effect of memorisation in model performance estimates and apply this framework to develop LINGOLY-TOO, a challenging evaluation benchmark for linguistic reasoning. By developing orthographic templates, we dynamically obfuscate the writing systems of real languages to generate numerous question variations. These variations preserve the reasoning steps required for each solution while reducing the likelihood of specific problem instances appearing in model training data. Our experiments demonstrate that frontier models, including OpenAI o1-preview and DeepSeem R1, struggle with advanced reasoning. Our analysis also shows that LLMs exhibit noticeable variance in accuracy across permutations of the same problem, and on average perform better on questions appearing in their original orthography. Our findings highlight the opaque nature of response generation in LLMs and provide evidence that prior data exposure contributes to overestimating the reasoning capabilities of frontier models. 

---
# Multilingual Relative Clause Attachment Ambiguity Resolution in Large Language Models 

**Authors**: So Young Lee, Russell Scheinberg, Amber Shore, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2503.02971)  

**Abstract**: This study examines how large language models (LLMs) resolve relative clause (RC) attachment ambiguities and compares their performance to human sentence processing. Focusing on two linguistic factors, namely the length of RCs and the syntactic position of complex determiner phrases (DPs), we assess whether LLMs can achieve human-like interpretations amid the complexities of language. In this study, we evaluated several LLMs, including Claude, Gemini and Llama, in multiple languages: English, Spanish, French, German, Japanese, and Korean. While these models performed well in Indo-European languages (English, Spanish, French, and German), they encountered difficulties in Asian languages (Japanese and Korean), often defaulting to incorrect English translations. The findings underscore the variability in LLMs' handling of linguistic ambiguities and highlight the need for model improvements, particularly for non-European languages. This research informs future enhancements in LLM design to improve accuracy and human-like processing in diverse linguistic environments. 

---
# InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model 

**Authors**: Siqi Ouyang, Xi Xu, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02969)  

**Abstract**: Simultaneous translation of unbounded streaming speech remains a challenging problem due to the need for effectively processing the history speech context and past translations so that quality and latency, including computation overhead, can be balanced. Most prior works assume pre-segmented speech, limiting their real-world applicability. In this paper, we propose InfiniSST, a novel approach that formulates SST as a multi-turn dialogue task, enabling seamless translation of unbounded speech. We construct translation trajectories and robust segments from MuST-C with multi-latency augmentation during training and develop a key-value (KV) cache management strategy to facilitate efficient inference. Experiments on MuST-C En-Es, En-De, and En-Zh demonstrate that InfiniSST reduces computation-aware latency by 0.5 to 1 second while maintaining the same translation quality compared to baselines. Ablation studies further validate the contributions of our data construction and cache management strategy. We release the code at this https URL 

---
# ExpertGenQA: Open-ended QA generation in Specialized Domains 

**Authors**: Haz Sameen Shahgir, Chansong Lim, Jia Chen, Evangelos E. Papalexakis, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2503.02948)  

**Abstract**: Generating high-quality question-answer pairs for specialized technical domains remains challenging, with existing approaches facing a tradeoff between leveraging expert examples and achieving topical diversity. We present ExpertGenQA, a protocol that combines few-shot learning with structured topic and style categorization to generate comprehensive domain-specific QA pairs. Using U.S. Federal Railroad Administration documents as a test bed, we demonstrate that ExpertGenQA achieves twice the efficiency of baseline few-shot approaches while maintaining $94.4\%$ topic coverage. Through systematic evaluation, we show that current LLM-based judges and reward models exhibit strong bias toward superficial writing styles rather than content quality. Our analysis using Bloom's Taxonomy reveals that ExpertGenQA better preserves the cognitive complexity distribution of expert-written questions compared to template-based approaches. When used to train retrieval models, our generated queries improve top-1 accuracy by $13.02\%$ over baseline performance, demonstrating their effectiveness for downstream applications in technical domains. 

---
# The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems 

**Authors**: Richard Ren, Arunim Agarwal, Mantas Mazeika, Cristina Menghini, Robert Vacareanu, Brad Kenstler, Mick Yang, Isabelle Barrass, Alice Gatti, Xuwang Yin, Eduardo Trevino, Matias Geralnik, Adam Khoja, Dean Lee, Summer Yue, Dan Hendrycks  

**Link**: [PDF](https://arxiv.org/pdf/2503.03750)  

**Abstract**: As large language models (LLMs) become more capable and agentic, the requirement for trust in their outputs grows significantly, yet at the same time concerns have been mounting that models may learn to lie in pursuit of their goals. To address these concerns, a body of work has emerged around the notion of "honesty" in LLMs, along with interventions aimed at mitigating deceptive behaviors. However, evaluations of honesty are currently highly limited, with no benchmark combining large scale and applicability to all models. Moreover, many benchmarks claiming to measure honesty in fact simply measure accuracy--the correctness of a model's beliefs--in disguise. In this work, we introduce a large-scale human-collected dataset for measuring honesty directly, allowing us to disentangle accuracy from honesty for the first time. Across a diverse set of LLMs, we find that while larger models obtain higher accuracy on our benchmark, they do not become more honest. Surprisingly, while most frontier LLMs obtain high scores on truthfulness benchmarks, we find a substantial propensity in frontier LLMs to lie when pressured to do so, resulting in low honesty scores on our benchmark. We find that simple methods, such as representation engineering interventions, can improve honesty. These results underscore the growing need for robust evaluations and effective interventions to ensure LLMs remain trustworthy. 

---
# Unified Mind Model: Reimagining Autonomous Agents in the LLM Era 

**Authors**: Pengbo Hu, Xiang Ying  

**Link**: [PDF](https://arxiv.org/pdf/2503.03459)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable capabilities across domains, tasks, and languages (e.g., ChatGPT and GPT-4), reviving the research of general autonomous agents with human-like cognitive this http URL human-level agents require semantic comprehension and instruction-following capabilities, which exactly fall into the strengths of this http URL there have been several initial attempts to build human-level agents based on LLMs, the theoretical foundation remains a challenging open problem. In this paper, we propose a novel theoretical cognitive architecture, the Unified Mind Model (UMM), which offers guidance to facilitate the rapid creation of autonomous agents with human-level cognitive abilities. Specifically, our UMM starts with the global workspace theory and further leverage LLMs to enable the agent with various cognitive abilities, such as multi-modal perception, planning, reasoning, tool use, learning, memory, reflection and motivation. Building upon UMM, we then develop an agent-building engine, MindOS, which allows users to quickly create domain-/task-specific autonomous agents without any programming effort. 

---
# Transformers for molecular property prediction: Domain adaptation efficiently improves performance 

**Authors**: Afnan Sultan, Max Rausch-Dupont, Shahrukh Khan, Olga Kalinina, Andrea Volkamer, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2503.03360)  

**Abstract**: Most of the current transformer-based chemical language models are pre-trained on millions to billions of molecules. However, the improvement from such scaling in dataset size is not confidently linked to improved molecular property prediction. The aim of this study is to investigate and overcome some of the limitations of transformer models in predicting molecular properties. Specifically, we examine the impact of pre-training dataset size and diversity on the performance of transformer models and investigate the use of domain adaptation as a technique for improving model performance. First, our findings indicate that increasing pretraining dataset size beyond 400K molecules from the GuacaMol dataset does not result in a significant improvement on four ADME endpoints, namely, solubility, permeability, microsomal stability, and plasma protein binding. Second, our results demonstrate that using domain adaptation by further training the transformer model on a small set of domain-relevant molecules, i.e., a few hundred to a few thousand, using multi-task regression of physicochemical properties was sufficient to significantly improve performance for three out of the four investigated ADME endpoints (P-value < 0.001). Finally, we observe that a model pre-trained on 400K molecules and domain adopted on a few hundred/thousand molecules performs similarly (P-value > 0.05) to more complicated transformer models like MolBERT(pre-trained on 1.3M molecules) and MolFormer (pre-trained on 100M molecules). A comparison to a random forest model trained on basic physicochemical properties showed similar performance to the examined transformer models. We believe that current transformer models can be improved through further systematic analysis of pre-training and downstream data, pre-training objectives, and scaling laws, ultimately leading to better and more helpful models. 

---
# LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models 

**Authors**: Xi Zhu, Haochen Xue, Ziwei Zhao, Wujiang Xu, Jingyuan Huang, Minghao Guo, Qifan Wang, Kaixiong Zhou, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03313)  

**Abstract**: Text-Attributed Graphs (TAGs), where each node is associated with text descriptions, are ubiquitous in real-world scenarios. They typically exhibit distinctive structure and domain-specific knowledge, motivating the development of a Graph Foundation Model (GFM) that generalizes across diverse graphs and tasks. Despite large efforts to integrate Large Language Models (LLMs) and Graph Neural Networks (GNNs) for TAGs, existing approaches suffer from decoupled architectures with two-stage alignment, limiting their synergistic potential. Even worse, existing methods assign out-of-vocabulary (OOV) tokens to graph nodes, leading to graph-specific semantics, token explosion, and incompatibility with task-oriented prompt templates, which hinders cross-graph and cross-task transferability. To address these challenges, we propose PromptGFM, a versatile GFM for TAGs grounded in graph vocabulary learning. PromptGFM comprises two key components: (1) Graph Understanding Module, which explicitly prompts LLMs to replicate the finest GNN workflow within the text space, facilitating seamless GNN-LLM integration and elegant graph-text alignment; (2) Graph Inference Module, which establishes a language-based graph vocabulary ensuring expressiveness, transferability, and scalability, enabling readable instructions for LLM fine-tuning. Extensive experiments demonstrate our superiority and transferability across diverse graphs and tasks. The code is available at this: this https URL. 

---
# Which books do I like? 

**Authors**: Hannes Rosenbusch, Erdem Ozan Meral  

**Link**: [PDF](https://arxiv.org/pdf/2503.03300)  

**Abstract**: Finding enjoyable fiction books can be challenging, partly because stories are multi-faceted and one's own literary taste might be difficult to ascertain. Here, we introduce the ISAAC method (Introspection-Support, AI-Annotation, and Curation), a pipeline which supports fiction readers in gaining awareness of their literary preferences and finding enjoyable books. ISAAC consists of four steps: a user supplies book ratings, an AI agent researches and annotates the provided books, patterns in book enjoyment are reviewed by the user, and the AI agent recommends new books. In this proof-of-concept self-study, the authors test whether ISAAC can highlight idiosyncratic patterns in their book enjoyment, spark a deeper reflection about their literary tastes, and make accurate, personalized recommendations of enjoyable books and underexplored literary niches. Results highlight substantial advantages of ISAAC over existing methods such as an integration of automation and intuition, accurate and customizable annotations, and explainable book recommendations. Observed disadvantages are that ISAAC's outputs can elicit false self-narratives (if statistical patterns are taken at face value), that books cannot be annotated if their online documentation is lacking, and that people who are new to reading have to rely on assumed book ratings or movie ratings to power the ISAAC pipeline. We discuss additional opportunities of ISAAC-style book annotations for the study of literary trends, and the scientific classification of books and readers. 

---
# Enhancing Abnormality Grounding for Vision Language Models with Knowledge Descriptions 

**Authors**: Jun Li, Che Liu, Wenjia Bai, Rossella Arcucci, Cosmin I. Bercea, Julia A. Schnabel  

**Link**: [PDF](https://arxiv.org/pdf/2503.03278)  

**Abstract**: Visual Language Models (VLMs) have demonstrated impressive capabilities in visual grounding tasks. However, their effectiveness in the medical domain, particularly for abnormality detection and localization within medical images, remains underexplored. A major challenge is the complex and abstract nature of medical terminology, which makes it difficult to directly associate pathological anomaly terms with their corresponding visual features. In this work, we introduce a novel approach to enhance VLM performance in medical abnormality detection and localization by leveraging decomposed medical knowledge. Instead of directly prompting models to recognize specific abnormalities, we focus on breaking down medical concepts into fundamental attributes and common visual patterns. This strategy promotes a stronger alignment between textual descriptions and visual features, improving both the recognition and localization of abnormalities in medical this http URL evaluate our method on the 0.23B Florence-2 base model and demonstrate that it achieves comparable performance in abnormality grounding to significantly larger 7B LLaVA-based medical VLMs, despite being trained on only 1.5% of the data used for such models. Experimental results also demonstrate the effectiveness of our approach in both known and previously unseen abnormalities, suggesting its strong generalization capabilities. 

---
# Towards Understanding Multi-Round Large Language Model Reasoning: Approximability, Learnability and Generalizability 

**Authors**: Chenhui Xu, Dancheng Liu, Jiajie Li, Amir Nassereldine, Zhaohui Li, Jinjun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.03128)  

**Abstract**: Recent advancements in cognitive science and multi-round reasoning techniques for Large Language Models (LLMs) suggest that iterative thinking processes improve problem-solving performance in complex tasks. Inspired by this, approaches like Chain-of-Thought, debating, and self-refinement have been applied to auto-regressive LLMs, achieving significant successes in tasks such as mathematical reasoning, commonsense reasoning, and multi-hop question answering. Despite these successes, the theoretical basis for how multi-round reasoning enhances problem-solving abilities remains underexplored. In this work, we investigate the approximation, learnability, and generalization properties of multi-round auto-regressive models. We show that Transformers with finite context windows are universal approximators for steps of Turing-computable functions and can approximate any Turing-computable sequence-to-sequence function through multi-round reasoning. We extend PAC learning to sequence generation and demonstrate that multi-round generation is learnable even when the sequence length exceeds the model's context window. Finally, we examine how generalization error propagates across rounds, and show how the aforementioned approaches can help constrain this error, ensuring outputs stay within an expectation boundary. This work sheds light on the systemic theoretical foundations of multi-round sequence learning and reasoning, emphasizing its role in inference complexity. 

---
# KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding 

**Authors**: Zhangchen Xu, Yang Liu, Yueqin Yin, Mingyuan Zhou, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2503.02951)  

**Abstract**: We introduce KodCode, a synthetic dataset that addresses the persistent challenge of acquiring high-quality, verifiable training data across diverse difficulties and domains for training Large Language Models for coding. Existing code-focused resources typically fail to ensure either the breadth of coverage (e.g., spanning simple coding tasks to advanced algorithmic problems) or verifiable correctness (e.g., unit tests). In contrast, KodCode comprises question-solution-test triplets that are systematically validated via a self-verification procedure. Our pipeline begins by synthesizing a broad range of coding questions, then generates solutions and test cases with additional attempts allocated to challenging problems. Finally, post-training data synthesis is done by rewriting questions into diverse formats and generating responses under a test-based reject sampling procedure from a reasoning model (DeepSeek R1). This pipeline yields a large-scale, robust and diverse coding dataset. KodCode is suitable for supervised fine-tuning and the paired unit tests also provide great potential for RL tuning. Fine-tuning experiments on coding benchmarks (HumanEval(+), MBPP(+), BigCodeBench, and LiveCodeBench) demonstrate that KodCode-tuned models achieve state-of-the-art performance, surpassing models like Qwen2.5-Coder-32B-Instruct and DeepSeek-R1-Distill-Llama-70B. 

---
# LiteWebAgent: The Open-Source Suite for VLM-Based Web-Agent Applications 

**Authors**: Danqing Zhang, Balaji Rama, Jingyi Ni, Shiying He, Fu Zhao, Kunyu Chen, Arnold Chen, Junyu Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02950)  

**Abstract**: We introduce LiteWebAgent, an open-source suite for VLM-based web agent applications. Our framework addresses a critical gap in the web agent ecosystem with a production-ready solution that combines minimal serverless backend configuration, intuitive user and browser interfaces, and extensible research capabilities in agent planning, memory, and tree search. For the core LiteWebAgent agent framework, we implemented a simple yet effective baseline using recursive function calling, providing with decoupled action generation and action grounding. In addition, we integrate advanced research components such as agent planning, agent workflow memory, and tree search in a modular and extensible manner. We then integrate the LiteWebAgent agent framework with frontend and backend as deployed systems in two formats: (1) a production Vercel-based web application, which provides users with an agent-controlled remote browser, (2) a Chrome extension leveraging LiteWebAgent's API to control an existing Chrome browser via CDP (Chrome DevTools Protocol). The LiteWebAgent framework is available at this https URL, with deployed frontend at this https URL. 

---
# Text2Scenario: Text-Driven Scenario Generation for Autonomous Driving Test 

**Authors**: Xuan Cai, Xuesong Bai, Zhiyong Cui, Danmu Xie, Daocheng Fu, Haiyang Yu, Yilong Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.02911)  

**Abstract**: Autonomous driving (AD) testing constitutes a critical methodology for assessing performance benchmarks prior to product deployment. The creation of segmented scenarios within a simulated environment is acknowledged as a robust and effective strategy; however, the process of tailoring these scenarios often necessitates laborious and time-consuming manual efforts, thereby hindering the development and implementation of AD technologies. In response to this challenge, we introduce Text2Scenario, a framework that leverages a Large Language Model (LLM) to autonomously generate simulation test scenarios that closely align with user specifications, derived from their natural language inputs. Specifically, an LLM, equipped with a meticulously engineered input prompt scheme functions as a text parser for test scenario descriptions, extracting from a hierarchically organized scenario repository the components that most accurately reflect the user's preferences. Subsequently, by exploiting the precedence of scenario components, the process involves sequentially matching and linking scenario representations within a Domain Specific Language corpus, ultimately fabricating executable test scenarios. The experimental results demonstrate that such prompt engineering can meticulously extract the nuanced details of scenario elements embedded within various descriptive formats, with the majority of generated scenarios aligning closely with the user's initial expectations, allowing for the efficient and precise evaluation of diverse AD stacks void of the labor-intensive need for manual scenario configuration. Project page: this https URL. 

---
# "Would You Want an AI Tutor?" Understanding Stakeholder Perceptions of LLM-based Chatbots in the Classroom 

**Authors**: Caterina Fuligni, Daniel Dominguez Figaredo, Julia Stoyanovich  

**Link**: [PDF](https://arxiv.org/pdf/2503.02885)  

**Abstract**: In recent years, Large Language Models (LLMs) rapidly gained popularity across all parts of society, including education. After initial skepticism and bans, many schools have chosen to embrace this new technology by integrating it into their curricula in the form of virtual tutors and teaching assistants. However, neither the companies developing this technology nor the public institutions involved in its implementation have set up a formal system to collect feedback from the stakeholders impacted by them. In this paper, we argue that understanding the perceptions of those directly affected by LLMS in the classroom, such as students and teachers, as well as those indirectly impacted, like parents and school staff, is essential for ensuring responsible use of AI in this critical domain. Our contributions are two-fold. First, we present results of a literature review focusing on the perceptions of LLM-based chatbots in education. We highlight important gaps in the literature, such as the exclusion of key educational agents (e.g., parents or school administrators) when analyzing the role of stakeholders, and the frequent omission of the learning contexts in which the AI systems are implemented. Thus, we present a taxonomy that organizes existing literature on stakeholder perceptions. Second, we propose the Contextualized Perceptions for the Adoption of Chatbots in Education (Co-PACE) framework, which can be used to systematically elicit perceptions and inform whether and how LLM-based chatbots should be designed, developed, and deployed in the classroom. 

---
