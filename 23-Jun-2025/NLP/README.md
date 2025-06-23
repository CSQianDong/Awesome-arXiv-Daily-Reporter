# Fine-Tuning Lowers Safety and Disrupts Evaluation Consistency 

**Authors**: Kathleen C. Fraser, Hillary Dawkins, Isar Nejadgholi, Svetlana Kiritchenko  

**Link**: [PDF](https://arxiv.org/pdf/2506.17209)  

**Abstract**: Fine-tuning a general-purpose large language model (LLM) for a specific domain or task has become a routine procedure for ordinary users. However, fine-tuning is known to remove the safety alignment features of the model, even when the fine-tuning data does not contain any harmful content. We consider this to be a critical failure mode of LLMs due to the widespread uptake of fine-tuning, combined with the benign nature of the "attack". Most well-intentioned developers are likely unaware that they are deploying an LLM with reduced safety. On the other hand, this known vulnerability can be easily exploited by malicious actors intending to bypass safety guardrails. To make any meaningful progress in mitigating this issue, we first need reliable and reproducible safety evaluations. In this work, we investigate how robust a safety benchmark is to trivial variations in the experimental procedure, and the stochastic nature of LLMs. Our initial experiments expose surprising variance in the results of the safety evaluation, even when seemingly inconsequential changes are made to the fine-tuning setup. Our observations have serious implications for how researchers in this field should report results to enable meaningful comparisons in the future. 

---
# Towards AI Search Paradigm 

**Authors**: Yuchen Li, Hengyi Cai, Rui Kong, Xinran Chen, Jiamin Chen, Jun Yang, Haojie Zhang, Jiayi Li, Jiayi Wu, Yiqun Chen, Changle Qu, Keyi Kong, Wenwen Ye, Lixin Su, Xinyu Ma, Long Xia, Daiting Shi, Jiashu Zhao, Haoyi Xiong, Shuaiqiang Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.17188)  

**Abstract**: In this paper, we introduce the AI Search Paradigm, a comprehensive blueprint for next-generation search systems capable of emulating human information processing and decision-making. The paradigm employs a modular architecture of four LLM-powered agents (Master, Planner, Executor and Writer) that dynamically adapt to the full spectrum of information needs, from simple factual queries to complex multi-stage reasoning tasks. These agents collaborate dynamically through coordinated workflows to evaluate query complexity, decompose problems into executable plans, and orchestrate tool usage, task execution, and content synthesis. We systematically present key methodologies for realizing this paradigm, including task planning and tool integration, execution strategies, aligned and robust retrieval-augmented generation, and efficient LLM inference, spanning both algorithmic techniques and infrastructure-level optimizations. By providing an in-depth guide to these foundational components, this work aims to inform the development of trustworthy, adaptive, and scalable AI search systems. 

---
# CLEAR-3K: Assessing Causal Explanatory Capabilities in Language Models 

**Authors**: Naiming Liu, Richard Baraniuk, Shashank Sonkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.17180)  

**Abstract**: We introduce CLEAR-3K, a dataset of 3,000 assertion-reasoning questions designed to evaluate whether language models can determine if one statement causally explains another. Each question present an assertion-reason pair and challenge language models to distinguish between semantic relatedness and genuine causal explanatory relationships. Through comprehensive evaluation of 21 state-of-the-art language models (ranging from 0.5B to 72B parameters), we identify two fundamental findings. First, language models frequently confuse semantic similarity with causality, relying on lexical and semantic overlap instead of inferring actual causal explanatory relationships. Second, as parameter size increases, models tend to shift from being overly skeptical about causal relationships to being excessively permissive in accepting them. Despite this shift, performance measured by the Matthews Correlation Coefficient plateaus at just 0.55, even for the best-performing this http URL, CLEAR-3K provides a crucial benchmark for developing and evaluating genuine causal reasoning in language models, which is an essential capability for applications that require accurate assessment of causal relationships. 

---
# Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs? 

**Authors**: Adithya Bhaskar, Alexander Wettig, Tianyu Gao, Yihe Dong, Danqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17121)  

**Abstract**: Language models handle increasingly long contexts for tasks such as book summarization, but this leads to growing memory costs for the key-value (KV) cache. Many prior works have proposed ways of discarding KVs from memory, but their approaches are tailored to favorable settings, obscuring caveats like high peak memory and performance degradation, and a fair comparison between methods is difficult. In this paper, we propose the *KV footprint* as a unified metric, which accounts for both the amount of KV entries stored and their lifespan in memory. We evaluate methods based on the smallest footprint they attain while preserving performance in both long-context understanding and generation, with context lengths of up to 128K tokens. This metric reveals the high peak memory of prior KV eviction methods. One class of methods -- *post-fill eviction* -- has a high footprint due to being incompatible with eviction during pre-filling. We adapt these methods to be able to evict KVs during pre-filling, achieving substantially lower KV footprints. We then turn to *recency eviction* methods, wherein we propose PruLong, an end-to-end optimization method for learning which attention heads need to retain the full KV cache and which do not. PruLong saves memory while preserving long-context performance, achieving 12% smaller KV footprint than prior methods while retaining performance in challenging recall tasks. Our paper clarifies the complex tangle of long-context inference methods and paves the way for future development to minimize the KV footprint. 

---
# Better Language Model Inversion by Compactly Representing Next-Token Distributions 

**Authors**: Murtaza Nazir, Matthew Finlayson, John X. Morris, Xiang Ren, Swabha Swayamdipta  

**Link**: [PDF](https://arxiv.org/pdf/2506.17090)  

**Abstract**: Language model inversion seeks to recover hidden prompts using only language model outputs. This capability has implications for security and accountability in language model deployments, such as leaking private information from an API-protected language model's system message. We propose a new method -- prompt inversion from logprob sequences (PILS) -- that recovers hidden prompts by gleaning clues from the model's next-token probabilities over the course of multiple generation steps. Our method is enabled by a key insight: The vector-valued outputs of a language model occupy a low-dimensional subspace. This enables us to losslessly compress the full next-token probability distribution over multiple generation steps using a linear map, allowing more output information to be used for inversion. Our approach yields massive gains over previous state-of-the-art methods for recovering hidden prompts, achieving 2--3.5 times higher exact recovery rates across test sets, in one case increasing the recovery rate from 17% to 60%. Our method also exhibits surprisingly good generalization behavior; for instance, an inverter trained on 16 generations steps gets 5--27 points higher prompt recovery when we increase the number of steps to 32 at test time. Furthermore, we demonstrate strong performance of our method on the more challenging task of recovering hidden system messages. We also analyze the role of verbatim repetition in prompt recovery and propose a new method for cross-family model transfer for logit-based inverters. Our findings show that next-token probabilities are a considerably more vulnerable attack surface for inversion attacks than previously known. 

---
# Chain-of-Thought Prompting Obscures Hallucination Cues in Large Language Models: An Empirical Evaluation 

**Authors**: Jiahao Cheng, Tiancheng Su, Jia Yuan, Guoxiu He, Jiawei Liu, Xinqi Tao, Jingwen Xie, Huaxia Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17088)  

**Abstract**: Large Language Models (LLMs) often exhibit \textit{hallucinations}, generating factually incorrect or semantically irrelevant content in response to prompts. Chain-of-Thought (CoT) prompting can mitigate hallucinations by encouraging step-by-step reasoning, but its impact on hallucination detection remains underexplored. To bridge this gap, we conduct a systematic empirical evaluation. We begin with a pilot experiment, revealing that CoT reasoning significantly affects the LLM's internal states and token probability distributions. Building on this, we evaluate the impact of various CoT prompting methods on mainstream hallucination detection methods across both instruction-tuned and reasoning-oriented LLMs. Specifically, we examine three key dimensions: changes in hallucination score distributions, variations in detection accuracy, and shifts in detection confidence. Our findings show that while CoT prompting helps reduce hallucination frequency, it also tends to obscure critical signals used for detection, impairing the effectiveness of various detection methods. Our study highlights an overlooked trade-off in the use of reasoning. Code is publicly available at: this https URL. 

---
# Tower+: Bridging Generality and Translation Specialization in Multilingual LLMs 

**Authors**: Ricardo Rei, Nuno M. Guerreiro, José Pombal, João Alves, Pedro Teixeirinha, Amin Farajian, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2506.17080)  

**Abstract**: Fine-tuning pretrained LLMs has been shown to be an effective strategy for reaching state-of-the-art performance on specific tasks like machine translation. However, this process of adaptation often implies sacrificing general-purpose capabilities, such as conversational reasoning and instruction-following, hampering the utility of the system in real-world applications that require a mixture of skills. In this paper, we introduce Tower+, a suite of models designed to deliver strong performance across both translation and multilingual general-purpose text capabilities. We achieve a Pareto frontier between translation specialization and multilingual general-purpose capabilities by introducing a novel training recipe that builds on Tower (Alves et al., 2024), comprising continued pretraining, supervised fine-tuning, preference optimization, and reinforcement learning with verifiable rewards. At each stage of training, we carefully generate and curate data to strengthen performance on translation as well as general-purpose tasks involving code generation, mathematics problem solving, and general instruction-following. We develop models at multiple scales: 2B, 9B, and 72B. Our smaller models often outperform larger general-purpose open-weight and proprietary LLMs (e.g., Llama 3.3 70B, GPT-4o). Our largest model delivers best-in-class translation performance for high-resource languages and top results in multilingual Arena Hard evaluations and in IF-MT, a benchmark we introduce for evaluating both translation and instruction-following. Our findings highlight that it is possible to rival frontier models in general capabilities, while optimizing for specific business domains, such as translation and localization. 

---
# Simultaneous Translation with Offline Speech and LLM Models in CUNI Submission to IWSLT 2025 

**Authors**: Dominik Macháček, Peter Polák  

**Link**: [PDF](https://arxiv.org/pdf/2506.17077)  

**Abstract**: This paper describes Charles University submission to the Simultaneous Speech Translation Task of the IWSLT 2025. We cover all four language pairs with a direct or cascade approach. The backbone of our systems is the offline Whisper speech model, which we use for both translation and transcription in simultaneous mode with the state-of-the-art simultaneous policy AlignAtt. We further improve the performance by prompting to inject in-domain terminology, and we accommodate context. Our cascaded systems further use EuroLLM for unbounded simultaneous translation. Compared to the Organizers' baseline, our systems improve by 2 BLEU points on Czech to English and 13-22 BLEU points on English to German, Chinese and Japanese on the development sets. Additionally, we also propose a new enhanced measure of speech recognition latency. 

---
# MUCAR: Benchmarking Multilingual Cross-Modal Ambiguity Resolution for Multimodal Large Language Models 

**Authors**: Xiaolong Wang, Zhaolu Kang, Wangyuxuan Zhai, Xinyue Lou, Yunghwei Lai, Ziyue Wang, Yawen Wang, Kaiyu Huang, Yile Wang, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17046)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated significant advances across numerous vision-language tasks. Due to their strong image-text alignment capability, MLLMs can effectively understand image-text pairs with clear meanings. However, effectively resolving the inherent ambiguities in natural language and visual contexts remains challenging. Existing multimodal benchmarks typically overlook linguistic and visual ambiguities, relying mainly on unimodal context for disambiguation and thus failing to exploit the mutual clarification potential between modalities. To bridge this gap, we introduce MUCAR, a novel and challenging benchmark designed explicitly for evaluating multimodal ambiguity resolution across multilingual and cross-modal scenarios. MUCAR includes: (1) a multilingual dataset where ambiguous textual expressions are uniquely resolved by corresponding visual contexts, and (2) a dual-ambiguity dataset that systematically pairs ambiguous images with ambiguous textual contexts, with each combination carefully constructed to yield a single, clear interpretation through mutual disambiguation. Extensive evaluations involving 19 state-of-the-art multimodal models--encompassing both open-source and proprietary architectures--reveal substantial gaps compared to human-level performance, highlighting the need for future research into more sophisticated cross-modal ambiguity comprehension methods, further pushing the boundaries of multimodal reasoning. 

---
# Instituto de Telecomunicações at IWSLT 2025: Aligning Small-Scale Speech and Language Models for Speech-to-Text Learning 

**Authors**: Giuseppe Attanasio, Sonal Sannigrahi, Ben Peters, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2506.17019)  

**Abstract**: This paper presents the IT-IST submission to the IWSLT 2025 Shared Task on Instruction Following Speech Processing. We submit results for the Short Track, i.e., speech recognition, translation, and spoken question answering. Our model is a unified speech-to-text model that integrates a pre-trained continuous speech encoder and text decoder through a first phase of modality alignment and a second phase of instruction fine-tuning. Crucially, we focus on using small-scale language model backbones (< 2B) and restrict to high-quality, CC-BY data along with synthetic data generation to supplement existing resources. 

---
# LLM-Generated Feedback Supports Learning If Learners Choose to Use It 

**Authors**: Danielle R. Thomas, Conrad Borchers, Shambhavi Bhushan, Erin Gatz, Shivang Gupta, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.17006)  

**Abstract**: Large language models (LLMs) are increasingly used to generate feedback, yet their impact on learning remains underexplored, especially compared to existing feedback methods. This study investigates how on-demand LLM-generated explanatory feedback influences learning in seven scenario-based tutor training lessons. Analyzing over 2,600 lesson completions from 885 tutor learners, we compare posttest performance among learners across three groups: learners who received feedback generated by gpt-3.5-turbo, those who declined it, and those without access. All groups received non-LLM corrective feedback. To address potential selection bias-where higher-performing learners may be more inclined to use LLM feedback-we applied propensity scoring. Learners with a higher predicted likelihood of engaging with LLM feedback scored significantly higher at posttest than those with lower propensity. After adjusting for this effect, two out of seven lessons showed statistically significant learning benefits from LLM feedback with standardized effect sizes of 0.28 and 0.33. These moderate effects suggest that the effectiveness of LLM feedback depends on the learners' tendency to seek support. Importantly, LLM feedback did not significantly increase completion time, and learners overwhelmingly rated it as helpful. These findings highlight LLM feedback's potential as a low-cost and scalable way to improve learning on open-ended tasks, particularly in existing systems already providing feedback without LLMs. This work contributes open datasets, LLM prompts, and rubrics to support reproducibility. 

---
# PersonalAI: Towards digital twins in the graph form 

**Authors**: Mikhail Menschikov, Dmitry Evseev, Ruslan Kostoev, Ilya Perepechkin, Ilnaz Salimov, Victoria Dochkina, Petr Anokhin, Evgeny Burnaev, Nikita Semenov  

**Link**: [PDF](https://arxiv.org/pdf/2506.17001)  

**Abstract**: The challenge of personalizing language models, specifically the ability to account for a user's history during interactions, is of significant interest. Despite recent advancements in large language models (LLMs) and Retrieval Augmented Generation that have enhanced the factual base of LLMs, the task of retaining extensive personal information and using it to generate personalized responses remains pertinent. To address this, we propose utilizing external memory in the form of knowledge graphs, which are constructed and updated by the LLM itself. We have expanded upon ideas of AriGraph architecture and for the first time introduced a combined graph featuring both standard edges and two types of hyperedges. Experiments conducted on the TriviaQA, HotpotQA and DiaASQ benchmarks indicates that this approach aids in making the process of graph construction and knowledge extraction unified and robust. Furthermore, we augmented the DiaASQ benchmark by incorporating parameters such as time into dialogues and introducing contradictory statements made by the same speaker at different times. Despite these modifications, the performance of the question-answering system remained robust, demonstrating the proposed architecture's ability to maintain and utilize temporal dependencies. 

---
# TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs 

**Authors**: Sahil Kale, Vijaykant Nadadur  

**Link**: [PDF](https://arxiv.org/pdf/2506.16990)  

**Abstract**: LaTeX's precision and flexibility in typesetting have made it the gold standard for the preparation of scientific documentation. Large Language Models (LLMs) present a promising opportunity for researchers to produce publication-ready material using LaTeX with natural language instructions, yet current benchmarks completely lack evaluation of this ability. By introducing TeXpert, our benchmark dataset with natural language prompts for generating LaTeX code focused on components of scientific documents across multiple difficulty levels, we conduct an in-depth analysis of LLM performance in this regard and identify frequent error types. Our evaluation across open and closed-source LLMs highlights multiple key findings: LLMs excelling on standard benchmarks perform poorly in LaTeX generation with a significant accuracy drop-off as the complexity of tasks increases; open-source models like DeepSeek v3 and DeepSeek Coder strongly rival closed-source counterparts in LaTeX tasks; and formatting and package errors are unexpectedly prevalent, suggesting a lack of diverse LaTeX examples in the training datasets of most LLMs. Our dataset, code, and model evaluations are available at this https URL. 

---
# Language Bottleneck Models: A Framework for Interpretable Knowledge Tracing and Beyond 

**Authors**: Antonin Berthon, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.16982)  

**Abstract**: Accurately assessing student knowledge is critical for effective education, yet traditional Knowledge Tracing (KT) methods rely on opaque latent embeddings, limiting interpretability. Even LLM-based approaches generate direct predictions or summaries that may hallucinate without any accuracy guarantees. We recast KT as an inverse problem: learning the minimum natural-language summary that makes past answers explainable and future answers predictable. Our Language Bottleneck Model (LBM) consists of an encoder LLM that writes an interpretable knowledge summary and a frozen decoder LLM that must reconstruct and predict student responses using only that summary text. By constraining all predictive information to pass through a short natural-language bottleneck, LBMs ensure that the summary contains accurate information while remaining human-interpretable. Experiments on synthetic arithmetic benchmarks and the large-scale Eedi dataset show that LBMs rival the accuracy of state-of-the-art KT and direct LLM methods while requiring orders-of-magnitude fewer student trajectories. We demonstrate that training the encoder with group-relative policy optimization, using downstream decoding accuracy as a reward signal, effectively improves summary quality. 

---
# From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts 

**Authors**: Daniel Christoph, Max Ploner, Patrick Haller, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2506.16912)  

**Abstract**: Sample efficiency is a crucial property of language models with practical implications for training efficiency. In real-world text, information follows a long-tailed distribution. Yet, we expect models to learn and recall frequent and infrequent facts. Sample-efficient models are better equipped to handle this challenge of learning and retaining rare information without requiring excessive exposure. This study analyzes multiple models of varying architectures and sizes, all trained on the same pre-training data. By annotating relational facts with their frequencies in the training corpus, we examine how model performance varies with fact frequency. Our findings show that most models perform similarly on high-frequency facts but differ notably on low-frequency facts. This analysis provides new insights into the relationship between model architecture, size, and factual learning efficiency. 

---
# MIST: Jailbreaking Black-box Large Language Models via Iterative Semantic Tuning 

**Authors**: Muyang Zheng, Yuanzhi Yao, Changting Lin, Rui Wang, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.16792)  

**Abstract**: Despite efforts to align large language models (LLMs) with societal and moral values, these models remain susceptible to jailbreak attacks--methods designed to elicit harmful responses. Jailbreaking black-box LLMs is considered challenging due to the discrete nature of token inputs, restricted access to the target LLM, and limited query budget. To address the issues above, we propose an effective method for jailbreaking black-box large language Models via Iterative Semantic Tuning, named MIST. MIST enables attackers to iteratively refine prompts that preserve the original semantic intent while inducing harmful content. Specifically, to balance semantic similarity with computational efficiency, MIST incorporates two key strategies: sequential synonym search, and its advanced version--order-determining optimization. Extensive experiments across two open-source models and four closed-source models demonstrate that MIST achieves competitive attack success rates and attack transferability compared with other state-of-the-art white-box and black-box jailbreak methods. Additionally, we conduct experiments on computational efficiency to validate the practical viability of MIST. 

---
# DistillNote: LLM-based clinical note summaries improve heart failure diagnosis 

**Authors**: Heloisa Oss Boll, Antonio Oss Boll, Leticia Puttlitz Boll, Ameen Abu Hanna, Iacer Calixto  

**Link**: [PDF](https://arxiv.org/pdf/2506.16777)  

**Abstract**: Large language models (LLMs) offer unprecedented opportunities to generate concise summaries of patient information and alleviate the burden of clinical documentation that overwhelms healthcare providers. We present Distillnote, a framework for LLM-based clinical note summarization, and generate over 64,000 admission note summaries through three techniques: (1) One-step, direct summarization, and a divide-and-conquer approach involving (2) Structured summarization focused on independent clinical insights, and (3) Distilled summarization that further condenses the Structured summaries. We test how useful are the summaries by using them to predict heart failure compared to a model trained on the original notes. Distilled summaries achieve 79% text compression and up to 18.2% improvement in AUPRC compared to an LLM trained on the full notes. We also evaluate the quality of the generated summaries in an LLM-as-judge evaluation as well as through blinded pairwise comparisons with clinicians. Evaluations indicate that one-step summaries are favoured by clinicians according to relevance and clinical actionability, while distilled summaries offer optimal efficiency (avg. 6.9x compression-to-performance ratio) and significantly reduce hallucinations. We release our summaries on PhysioNet to encourage future research. 

---
# Cross-Modal Obfuscation for Jailbreak Attacks on Large Vision-Language Models 

**Authors**: Lei Jiang, Zixun Zhang, Zizhou Wang, Xiaobing Sun, Zhen Li, Liangli Zhen, Xiaohua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16760)  

**Abstract**: Large Vision-Language Models (LVLMs) demonstrate exceptional performance across multimodal tasks, yet remain vulnerable to jailbreak attacks that bypass built-in safety mechanisms to elicit restricted content generation. Existing black-box jailbreak methods primarily rely on adversarial textual prompts or image perturbations, yet these approaches are highly detectable by standard content filtering systems and exhibit low query and computational efficiency. In this work, we present Cross-modal Adversarial Multimodal Obfuscation (CAMO), a novel black-box jailbreak attack framework that decomposes malicious prompts into semantically benign visual and textual fragments. By leveraging LVLMs' cross-modal reasoning abilities, CAMO covertly reconstructs harmful instructions through multi-step reasoning, evading conventional detection mechanisms. Our approach supports adjustable reasoning complexity and requires significantly fewer queries than prior attacks, enabling both stealth and efficiency. Comprehensive evaluations conducted on leading LVLMs validate CAMO's effectiveness, showcasing robust performance and strong cross-model transferability. These results underscore significant vulnerabilities in current built-in safety mechanisms, emphasizing an urgent need for advanced, alignment-aware security and safety solutions in vision-language systems. 

---
# SocialSim: Towards Socialized Simulation of Emotional Support Conversation 

**Authors**: Zhuang Chen, Yaru Cao, Guanqun Bi, Jincenzi Wu, Jinfeng Zhou, Xiyao Xiao, Si Chen, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16756)  

**Abstract**: Emotional support conversation (ESC) helps reduce people's psychological stress and provide emotional value through interactive dialogues. Due to the high cost of crowdsourcing a large ESC corpus, recent attempts use large language models for dialogue augmentation. However, existing approaches largely overlook the social dynamics inherent in ESC, leading to less effective simulations. In this paper, we introduce SocialSim, a novel framework that simulates ESC by integrating key aspects of social interactions: social disclosure and social awareness. On the seeker side, we facilitate social disclosure by constructing a comprehensive persona bank that captures diverse and authentic help-seeking scenarios. On the supporter side, we enhance social awareness by eliciting cognitive reasoning to generate logical and supportive responses. Building upon SocialSim, we construct SSConv, a large-scale synthetic ESC corpus of which quality can even surpass crowdsourced ESC data. We further train a chatbot on SSConv and demonstrate its state-of-the-art performance in both automatic and human evaluations. We believe SocialSim offers a scalable way to synthesize ESC, making emotional care more accessible and practical. 

---
# Language-Informed Synthesis of Rational Agent Models for Grounded Theory-of-Mind Reasoning On-The-Fly 

**Authors**: Lance Ying, Ryan Truong, Katherine M. Collins, Cedegao E. Zhang, Megan Wei, Tyler Brooke-Wilson, Tan Zhi-Xuan, Lionel Wong, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2506.16755)  

**Abstract**: Drawing real world social inferences usually requires taking into account information from multiple modalities. Language is a particularly powerful source of information in social settings, especially in novel situations where language can provide both abstract information about the environment dynamics and concrete specifics about an agent that cannot be easily visually observed. In this paper, we propose Language-Informed Rational Agent Synthesis (LIRAS), a framework for drawing context-specific social inferences that integrate linguistic and visual inputs. LIRAS frames multimodal social reasoning as a process of constructing structured but situation-specific agent and environment representations - leveraging multimodal language models to parse language and visual inputs into unified symbolic representations, over which a Bayesian inverse planning engine can be run to produce granular probabilistic judgments. On a range of existing and new social reasoning tasks derived from cognitive science experiments, we find that our model (instantiated with a comparatively lightweight VLM) outperforms ablations and state-of-the-art models in capturing human judgments across all domains. 

---
# LM-SPT: LM-Aligned Semantic Distillation for Speech Tokenization 

**Authors**: Daejin Jo, Jeeyoung Yun, Byungseok Roh, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.16738)  

**Abstract**: With the rapid progress of speech language models (SLMs), discrete speech tokens have emerged as a core interface between speech and text, enabling unified modeling across modalities. Recent speech tokenization approaches aim to isolate semantic information from low-level acoustics to better align with language models. In particular, previous methods use SSL teachers such as HuBERT to extract semantic representations, which are then distilled into a semantic quantizer to suppress acoustic redundancy as well as capture content-related latent structures. However, they still produce speech token sequences significantly longer than their textual counterparts, creating challenges for efficient speech-language modeling. Reducing the frame rate is a natural solution, but standard techniques, such as rigid average pooling across frames, can distort or dilute the semantic structure required for effective LM alignment. To address this, we propose LM-SPT, a speech tokenization method that introduces a novel semantic distillation. Instead of directly matching teacher and student features via pooling, we reconstruct speech solely from semantic tokens and minimize the discrepancy between the encoded representations of the original and reconstructed waveforms, obtained from a frozen automatic speech recognition (ASR) encoder. This indirect yet data-driven supervision enables the tokenizer to learn discrete units that are more semantically aligned with language models. LM-SPT further incorporates architectural improvements to the encoder and decoder for speech tokenization, and supports multiple frame rates, including 25Hz, 12.5Hz, and 6.25Hz. Experimental results show that LM-SPT achieves superior reconstruction fidelity compared to baselines, and that SLMs trained with LM-SPT tokens achieve competitive performances on speech-to-text and consistently outperform baselines on text-to-speech tasks. 

---
# The Role of Model Confidence on Bias Effects in Measured Uncertainties 

**Authors**: Xinyi Liu, Weiguang Wang, Hangfeng He  

**Link**: [PDF](https://arxiv.org/pdf/2506.16724)  

**Abstract**: With the growing adoption of Large Language Models (LLMs) for open-ended tasks, accurately assessing epistemic uncertainty, which reflects a model's lack of knowledge, has become crucial to ensuring reliable outcomes. However, quantifying epistemic uncertainty in such tasks is challenging due to the presence of aleatoric uncertainty, which arises from multiple valid answers. While bias can introduce noise into epistemic uncertainty estimation, it may also reduce noise from aleatoric uncertainty. To investigate this trade-off, we conduct experiments on Visual Question Answering (VQA) tasks and find that mitigating prompt-introduced bias improves uncertainty quantification in GPT-4o. Building on prior work showing that LLMs tend to copy input information when model confidence is low, we further analyze how these prompt biases affect measured epistemic and aleatoric uncertainty across varying bias-free confidence levels with GPT-4o and Qwen2-VL. We find that all considered biases induce greater changes in both uncertainties when bias-free model confidence is lower. Moreover, lower bias-free model confidence leads to greater underestimation of epistemic uncertainty (i.e. overconfidence) due to bias, whereas it has no significant effect on the direction of changes in aleatoric uncertainty estimation. These distinct effects deepen our understanding of bias mitigation for uncertainty quantification and potentially inform the development of more advanced techniques. 

---
# ReasonGRM: Enhancing Generative Reward Models through Large Reasoning Models 

**Authors**: Bin Chen, Xinzge Gao, Chuanrui Hu, Penghang Yu, Hua Zhang, Bing-Kun Bao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16712)  

**Abstract**: Generative Reward Models (GRMs) provide greater flexibility than scalar reward models in capturing human preferences, but their effectiveness is limited by poor reasoning capabilities. This often results in incomplete or overly speculative reasoning paths, leading to hallucinations or missing key information in complex tasks. We address this challenge with ReasonGRM, a three-stage generative reward modeling framework. In the first stage, Zero-RL is used to generate concise, outcome-directed reasoning paths that reduce the likelihood of critical omissions. In the second stage, we introduce a novel evaluation metric, $R^\star$, which scores reasoning paths based on their generation likelihood. This favors paths that reach correct answers with minimal exploration, helping to reduce hallucination-prone data during training. In the final stage, the model is further refined through reinforcement learning on challenging examples to enhance its preference discrimination capabilities. Experiments on three public benchmarks show that ReasonGRM achieves competitive or state-of-the-art performance, outperforming previous best GRMs by 1.8\% on average and surpassing proprietary models such as GPT-4o by up to 5.6\%. These results demonstrate the effectiveness of reasoning-aware training and highlight the importance of high-quality rationale selection for reliable preference modeling. 

---
# LegiGPT: Party Politics and Transport Policy with Large Language Model 

**Authors**: Hyunsoo Yun, Eun Hak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.16692)  

**Abstract**: Given the significant influence of lawmakers' political ideologies on legislative decision-making, understanding their impact on policymaking is critically important. We introduce a novel framework, LegiGPT, which integrates a large language model (LLM) with explainable artificial intelligence (XAI) to analyze transportation-related legislative proposals. LegiGPT employs a multi-stage filtering and classification pipeline using zero-shot prompting with GPT-4. Using legislative data from South Korea's 21st National Assembly, we identify key factors - including sponsor characteristics, political affiliations, and geographic variables - that significantly influence transportation policymaking. The LLM was used to classify transportation-related bill proposals through a stepwise filtering process based on keywords, phrases, and contextual relevance. XAI techniques were then applied to examine relationships between party affiliation and associated attributes. The results reveal that the number and proportion of conservative and progressive sponsors, along with district size and electoral population, are critical determinants shaping legislative outcomes. These findings suggest that both parties contributed to bipartisan legislation through different forms of engagement, such as initiating or supporting proposals. This integrated approach provides a valuable tool for understanding legislative dynamics and guiding future policy development, with broader implications for infrastructure planning and governance. 

---
# Mechanisms vs. Outcomes: Probing for Syntax Fails to Explain Performance on Targeted Syntactic Evaluations 

**Authors**: Ananth Agarwal, Jasper Jian, Christopher D. Manning, Shikhar Murty  

**Link**: [PDF](https://arxiv.org/pdf/2506.16678)  

**Abstract**: Large Language Models (LLMs) exhibit a robust mastery of syntax when processing and generating text. While this suggests internalized understanding of hierarchical syntax and dependency relations, the precise mechanism by which they represent syntactic structure is an open area within interpretability research. Probing provides one way to identify the mechanism of syntax being linearly encoded in activations, however, no comprehensive study has yet established whether a model's probing accuracy reliably predicts its downstream syntactic performance. Adopting a "mechanisms vs. outcomes" framework, we evaluate 32 open-weight transformer models and find that syntactic features extracted via probing fail to predict outcomes of targeted syntax evaluations across English linguistic phenomena. Our results highlight a substantial disconnect between latent syntactic representations found via probing and observable syntactic behaviors in downstream tasks. 

---
# Arch-Router: Aligning LLM Routing with Human Preferences 

**Authors**: Co Tran, Salman Paracha, Adil Hafeez, Shuguang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.16655)  

**Abstract**: With the rapid proliferation of large language models (LLMs) -- each optimized for different strengths, style, or latency/cost profile -- routing has become an essential technique to operationalize the use of different models. However, existing LLM routing approaches are limited in two key ways: they evaluate performance using benchmarks that often fail to capture human preferences driven by subjective evaluation criteria, and they typically select from a limited pool of models. In this work, we propose a preference-aligned routing framework that guides model selection by matching queries to user-defined domains (e.g., travel) or action types (e.g., image editing) -- offering a practical mechanism to encode preferences in routing decisions. Specifically, we introduce \textbf{Arch-Router}, a compact 1.5B model that learns to map queries to domain-action preferences for model routing decisions. Our approach also supports seamlessly adding new models for routing without requiring retraining or architectural modifications. Experiments on conversational datasets demonstrate that our approach achieves state-of-the-art (SOTA) results in matching queries with human preferences, outperforming top proprietary models. Our approach captures subjective evaluation criteria and makes routing decisions more transparent and flexible. Our model is available at: \texttt{this https URL}. 

---
# Long-Context Generalization with Sparse Attention 

**Authors**: Pavlo Vasylenko, Marcos Treviso, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2506.16640)  

**Abstract**: Transformer-based architectures traditionally employ softmax to compute attention weights, which produces dense distributions over all tokens in a sequence. While effective in many settings, this density has been shown to be detrimental for tasks that demand precise focus on fixed-size patterns: as sequence length increases, non-informative tokens accumulate attention probability mass, leading to dispersion and representational collapse. We show in this paper that sparse attention mechanisms using $\alpha$-entmax can avoid these issues, due to their ability to assign exact zeros to irrelevant tokens. Furthermore, we introduce Adaptive-Scalable Entmax (ASEntmax), which endows $\alpha$-entmax with a learnable temperature parameter, allowing the attention distribution to interpolate between sparse (pattern-focused) and dense (softmax-like) regimes. Finally, we show that the ability to locate and generalize fixed-size patterns can be further improved through a careful design of position encodings, which impacts both dense and sparse attention methods. By integrating ASEntmax into standard transformer layers alongside proper positional encodings, we show that our models greatly outperform softmax, scalable softmax, and fixed-temperature $\alpha$-entmax baselines on long-context generalization. 

---
# GeoGuess: Multimodal Reasoning based on Hierarchy of Visual Information in Street View 

**Authors**: Fenghua Cheng, Jinxiang Wang, Sen Wang, Zi Huang, Xue Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16633)  

**Abstract**: Multimodal reasoning is a process of understanding, integrating and inferring information across different data modalities. It has recently attracted surging academic attention as a benchmark for Artificial Intelligence (AI). Although there are various tasks for evaluating multimodal reasoning ability, they still have limitations. Lack of reasoning on hierarchical visual clues at different levels of granularity, e.g., local details and global context, is of little discussion, despite its frequent involvement in real scenarios. To bridge the gap, we introduce a novel and challenging task for multimodal reasoning, namely GeoGuess. Given a street view image, the task is to identify its location and provide a detailed explanation. A system that succeeds in GeoGuess should be able to detect tiny visual clues, perceive the broader landscape, and associate with vast geographic knowledge. Therefore, GeoGuess would require the ability to reason between hierarchical visual information and geographic knowledge. In this work, we establish a benchmark for GeoGuess by introducing a specially curated dataset GeoExplain which consists of panoramas-geocoordinates-explanation tuples. Additionally, we present a multimodal and multilevel reasoning method, namely SightSense which can make prediction and generate comprehensive explanation based on hierarchy of visual information and external knowledge. Our analysis and experiments demonstrate their outstanding performance in GeoGuess. 

---
# Initial Investigation of LLM-Assisted Development of Rule-Based Clinical NLP System 

**Authors**: Jianlin Shi, Brian T. Bucher  

**Link**: [PDF](https://arxiv.org/pdf/2506.16628)  

**Abstract**: Despite advances in machine learning (ML) and large language models (LLMs), rule-based natural language processing (NLP) systems remain active in clinical settings due to their interpretability and operational efficiency. However, their manual development and maintenance are labor-intensive, particularly in tasks with large linguistic variability. To overcome these limitations, we proposed a novel approach employing LLMs solely during the rule-based systems development phase. We conducted the initial experiments focusing on the first two steps of developing a rule-based NLP pipeline: find relevant snippets from the clinical note; extract informative keywords from the snippets for the rule-based named entity recognition (NER) component. Our experiments demonstrated exceptional recall in identifying clinically relevant text snippets (Deepseek: 0.98, Qwen: 0.99) and 1.0 in extracting key terms for NER. This study sheds light on a promising new direction for NLP development, enabling semi-automated or automated development of rule-based systems with significantly faster, more cost-effective, and transparent execution compared with deep learning model-based solutions. 

---
# Modeling Public Perceptions of Science in Media 

**Authors**: Jiaxin Pei, Dustin Wright, Isabelle Augenstin, David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2506.16622)  

**Abstract**: Effectively engaging the public with science is vital for fostering trust and understanding in our scientific community. Yet, with an ever-growing volume of information, science communicators struggle to anticipate how audiences will perceive and interact with scientific news. In this paper, we introduce a computational framework that models public perception across twelve dimensions, such as newsworthiness, importance, and surprisingness. Using this framework, we create a large-scale science news perception dataset with 10,489 annotations from 2,101 participants from diverse US and UK populations, providing valuable insights into public responses to scientific information across domains. We further develop NLP models that predict public perception scores with a strong performance. Leveraging the dataset and model, we examine public perception of science from two perspectives: (1) Perception as an outcome: What factors affect the public perception of scientific information? (2) Perception as a predictor: Can we use the estimated perceptions to predict public engagement with science? We find that individuals' frequency of science news consumption is the driver of perception, whereas demographic factors exert minimal influence. More importantly, through a large-scale analysis and carefully designed natural experiment on Reddit, we demonstrate that the estimated public perception of scientific information has direct connections with the final engagement pattern. Posts with more positive perception scores receive significantly more comments and upvotes, which is consistent across different scientific information and for the same science, but are framed differently. Overall, this research underscores the importance of nuanced perception modeling in science communication, offering new pathways to predict public interest and engagement with scientific content. 

---
# A Scoping Review of Synthetic Data Generation for Biomedical Research and Applications 

**Authors**: Hanshu Rao, Weisi Liu, Haohan Wang, I-Chan Huang, Zhe He, Xiaolei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16594)  

**Abstract**: Synthetic data generation--mitigating data scarcity, privacy concerns, and data quality challenges in biomedical fields--has been facilitated by rapid advances of large language models (LLMs). This scoping review follows PRISMA-ScR guidelines and synthesizes 59 studies, published between 2020 and 2025 and collected from PubMed, ACM, Web of Science, and Google Scholar. The review systematically examines biomedical research and application trends in synthetic data generation, emphasizing clinical applications, methodologies, and evaluations. Our analysis identifies data modalities of unstructured texts (78.0%), tabular data (13.6%), and multimodal sources (8.4%); generation methods of prompting (72.9%), fine-tuning (22.0%) LLMs and specialized model (5.1%); and heterogeneous evaluations of intrinsic metrics (27.1%), human-in-the-loop assessments (55.9%), and LLM-based evaluations (13.6%). The analysis addresses current limitations in what, where, and how health professionals can leverage synthetic data generation for biomedical domains. Our review also highlights challenges in adaption across clinical domains, resource and model accessibility, and evaluation standardizations. 

---
# Measuring (a Sufficient) World Model in LLMs: A Variance Decomposition Framework 

**Authors**: Nadav Kunievsky, James A. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2506.16584)  

**Abstract**: Understanding whether large language models (LLMs) possess a world model-a structured understanding of the world that supports generalization beyond surface-level patterns-is central to assessing their reliability, especially in high-stakes applications. We propose a formal framework for evaluating whether an LLM exhibits a sufficiently robust world model, defined as producing consistent outputs across semantically equivalent prompts while distinguishing between prompts that express different intents. We introduce a new evaluation approach to measure this that decomposes model response variability into three components: variability due to user purpose, user articulation, and model instability. An LLM with a strong world model should attribute most of the variability in its responses to changes in foundational purpose rather than superficial changes in articulation. This approach allows us to quantify how much of a model's behavior is semantically grounded rather than driven by model instability or alternative wording. We apply this framework to evaluate LLMs across diverse domains. Our results show how larger models attribute a greater share of output variability to changes in user purpose, indicating a more robust world model. This improvement is not uniform, however: larger models do not consistently outperform smaller ones across all domains, and their advantage in robustness is often modest. These findings highlight the importance of moving beyond accuracy-based benchmarks toward semantic diagnostics that more directly assess the structure and stability of a model's internal understanding of the world. 

---
# Streaming Non-Autoregressive Model for Accent Conversion and Pronunciation Improvement 

**Authors**: Tuan-Nam Nguyen, Ngoc-Quan Pham, Seymanur Akti, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2506.16580)  

**Abstract**: We propose a first streaming accent conversion (AC) model that transforms non-native speech into a native-like accent while preserving speaker identity, prosody and improving pronunciation. Our approach enables stream processing by modifying a previous AC architecture with an Emformer encoder and an optimized inference mechanism. Additionally, we integrate a native text-to-speech (TTS) model to generate ideal ground-truth data for efficient training. Our streaming AC model achieves comparable performance to the top AC models while maintaining stable latency, making it the first AC system capable of streaming. 

---
# Weight Factorization and Centralization for Continual Learning in Speech Recognition 

**Authors**: Enes Yavuz Ugan, Ngoc-Quan Pham, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2506.16574)  

**Abstract**: Modern neural network based speech recognition models are required to continually absorb new data without re-training the whole system, especially in downstream applications using foundation models, having no access to the original training data. Continually training the models in a rehearsal-free, multilingual, and language agnostic condition, likely leads to catastrophic forgetting, when a seemingly insignificant disruption to the weights can destructively harm the quality of the models. Inspired by the ability of human brains to learn and consolidate knowledge through the waking-sleeping cycle, we propose a continual learning approach with two distinct phases: factorization and centralization, learning and merging knowledge accordingly. Our experiments on a sequence of varied code-switching datasets showed that the centralization stage can effectively prevent catastrophic forgetting by accumulating the knowledge in multiple scattering low-rank adapters. 

---
# Automatic Speech Recognition Biases in Newcastle English: an Error Analysis 

**Authors**: Dana Serditova, Kevin Tang, Jochen Steffens  

**Link**: [PDF](https://arxiv.org/pdf/2506.16558)  

**Abstract**: Automatic Speech Recognition (ASR) systems struggle with regional dialects due to biased training which favours mainstream varieties. While previous research has identified racial, age, and gender biases in ASR, regional bias remains underexamined. This study investigates ASR performance on Newcastle English, a well-documented regional dialect known to be challenging for ASR. A two-stage analysis was conducted: first, a manual error analysis on a subsample identified key phonological, lexical, and morphosyntactic errors behind ASR misrecognitions; second, a case study focused on the systematic analysis of ASR recognition of the regional pronouns ``yous'' and ``wor''. Results show that ASR errors directly correlate with regional dialectal features, while social factors play a lesser role in ASR mismatches. We advocate for greater dialectal diversity in ASR training data and highlight the value of sociolinguistic analysis in diagnosing and addressing regional biases. 

---
# Relic: Enhancing Reward Model Generalization for Low-Resource Indic Languages with Few-Shot Examples 

**Authors**: Soumya Suvra Ghosal, Vaibhav Singh, Akash Ghosh, Soumyabrata Pal, Subhadip Baidya, Sriparna Saha, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2506.16502)  

**Abstract**: Reward models are essential for aligning large language models (LLMs) with human preferences. However, most open-source multilingual reward models are primarily trained on preference datasets in high-resource languages, resulting in unreliable reward signals for low-resource Indic languages. Collecting large-scale, high-quality preference data for these languages is prohibitively expensive, making preference-based training approaches impractical. To address this challenge, we propose RELIC, a novel in-context learning framework for reward modeling in low-resource Indic languages. RELIC trains a retriever with a pairwise ranking objective to select in-context examples from auxiliary high-resource languages that most effectively highlight the distinction between preferred and less-preferred responses. Extensive experiments on three preference datasets- PKU-SafeRLHF, WebGPT, and HH-RLHF-using state-of-the-art open-source reward models demonstrate that RELIC significantly improves reward model accuracy for low-resource Indic languages, consistently outperforming existing example selection methods. For example, on Bodo-a low-resource Indic language-using a LLaMA-3.2-3B reward model, RELIC achieves a 12.81% and 10.13% improvement in accuracy over zero-shot prompting and state-of-the-art example selection method, respectively. 

---
# Towards Generalizable Generic Harmful Speech Datasets for Implicit Hate Speech Detection 

**Authors**: Saad Almohaimeed, Saleh Almohaimeed, Damla Turgut, Ladislau Bölöni  

**Link**: [PDF](https://arxiv.org/pdf/2506.16476)  

**Abstract**: Implicit hate speech has recently emerged as a critical challenge for social media platforms. While much of the research has traditionally focused on harmful speech in general, the need for generalizable techniques to detect veiled and subtle forms of hate has become increasingly pressing. Based on lexicon analysis, we hypothesize that implicit hate speech is already present in publicly available harmful speech datasets but may not have been explicitly recognized or labeled by annotators. Additionally, crowdsourced datasets are prone to mislabeling due to the complexity of the task and often influenced by annotators' subjective interpretations. In this paper, we propose an approach to address the detection of implicit hate speech and enhance generalizability across diverse datasets by leveraging existing harmful speech datasets. Our method comprises three key components: influential sample identification, reannotation, and augmentation using Llama-3 70B and GPT-4o. Experimental results demonstrate the effectiveness of our approach in improving implicit hate detection, achieving a +12.9-point F1 score improvement compared to the baseline. 

---
# StoryWriter: A Multi-Agent Framework for Long Story Generation 

**Authors**: Haotian Xia, Hao Peng, Yunjia Qi, Xiaozhi Wang, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16445)  

**Abstract**: Long story generation remains a challenge for existing large language models (LLMs), primarily due to two main factors: (1) discourse coherence, which requires plot consistency, logical coherence, and completeness in the long-form generation, and (2) narrative complexity, which requires an interwoven and engaging narrative. To address these challenges, we propose StoryWriter, a multi-agent story generation framework, which consists of three main modules: (1) outline agent, which generates event-based outlines containing rich event plots, character, and event-event relationships. (2) planning agent, which further details events and plans which events should be written in each chapter to maintain an interwoven and engaging story. (3) writing agent, which dynamically compresses the story history based on the current event to generate and reflect new plots, ensuring the coherence of the generated story. We conduct both human and automated evaluation, and StoryWriter significantly outperforms existing story generation baselines in both story quality and length. Furthermore, we use StoryWriter to generate a dataset, which contains about $6,000$ high-quality long stories, with an average length of $8,000$ words. We train the model Llama3.1-8B and GLM4-9B using supervised fine-tuning on LongStory and develop StoryWriter_GLM and StoryWriter_GLM, which demonstrates advanced performance in long story generation. 

---
# REIS: A High-Performance and Energy-Efficient Retrieval System with In-Storage Processing 

**Authors**: Kangqi Chen, Andreas Kosmas Kakolyris, Rakesh Nadig, Manos Frouzakis, Nika Mansouri Ghiasi, Yu Liang, Haiyu Mao, Jisung Park, Mohammad Sadrosadati, Onur Mutlu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16444)  

**Abstract**: Large Language Models (LLMs) face an inherent challenge: their knowledge is confined to the data that they have been trained on. To overcome this issue, Retrieval-Augmented Generation (RAG) complements the static training-derived knowledge of LLMs with an external knowledge repository. RAG consists of three stages: indexing, retrieval, and generation. The retrieval stage of RAG becomes a significant bottleneck in inference pipelines. In this stage, a user query is mapped to an embedding vector and an Approximate Nearest Neighbor Search (ANNS) algorithm searches for similar vectors in the database to identify relevant items. Due to the large database sizes, ANNS incurs significant data movement overheads between the host and the storage system. To alleviate these overheads, prior works propose In-Storage Processing (ISP) techniques that accelerate ANNS by performing computations inside storage. However, existing works that leverage ISP for ANNS (i) employ algorithms that are not tailored to ISP systems, (ii) do not accelerate data retrieval operations for data selected by ANNS, and (iii) introduce significant hardware modifications, limiting performance and hindering their adoption. We propose REIS, the first ISP system tailored for RAG that addresses these limitations with three key mechanisms. First, REIS employs a database layout that links database embedding vectors to their associated documents, enabling efficient retrieval. Second, it enables efficient ANNS by introducing an ISP-tailored data placement technique that distributes embeddings across the planes of the storage system and employs a lightweight Flash Translation Layer. Third, REIS leverages an ANNS engine that uses the existing computational resources inside the storage system. Compared to a server-grade system, REIS improves the performance (energy efficiency) of retrieval by an average of 13x (55x). 

---
# When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework 

**Authors**: Zhen Xu, Shang Zhu, Jue Wang, Junlin Wang, Ben Athiwaratkun, Chi Wang, James Zou, Ce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16411)  

**Abstract**: We investigate the challenge of applying Large Language Models (LLMs) to long texts. We propose a theoretical framework that distinguishes the failure modes of long context tasks into three categories: cross-chunk dependence (task noise), confusion that grows with context size (model noise), and the imperfect integration of partial results (aggregator noise). Under this view, we analyze when it is effective to use multi-agent chunking, i.e., dividing a length sequence into smaller chunks and aggregating the processed results of each chunk. Our experiments on tasks such as retrieval, question answering, and summarization confirm both the theoretical analysis and the conditions that favor multi-agent chunking. By exploring superlinear model noise growth with input length, we also explain why, for large inputs, a weaker model configured with chunk-based processing can surpass a more advanced model like GPT4o applied in a single shot. Overall, we present a principled understanding framework and our results highlight a direct pathway to handling long contexts in LLMs with carefully managed chunking and aggregator strategies. 

---
# NepaliGPT: A Generative Language Model for the Nepali Language 

**Authors**: Shushanta Pudasaini, Aman Shakya, Siddhartha Shrestha, Sahil Bhatta, Sunil Thapa, Sushmita Palikhe  

**Link**: [PDF](https://arxiv.org/pdf/2506.16399)  

**Abstract**: After the release of ChatGPT, Large Language Models (LLMs) have gained huge popularity in recent days and thousands of variants of LLMs have been released. However, there is no generative language model for the Nepali language, due to which other downstream tasks, including fine-tuning, have not been explored yet. To fill this research gap in the Nepali NLP space, this research proposes \textit{NepaliGPT}, a generative large language model tailored specifically for the Nepali language. This research introduces an advanced corpus for the Nepali language collected from several sources, called the Devanagari Corpus. Likewise, the research introduces the first NepaliGPT benchmark dataset comprised of 4,296 question-answer pairs in the Nepali language. The proposed LLM NepaliGPT achieves the following metrics in text generation: Perplexity of 26.32245, ROUGE-1 score of 0.2604, causal coherence of 81.25\%, and causal consistency of 85.41\%. 

---
# OJBench: A Competition Level Code Benchmark For Large Language Models 

**Authors**: Zhexu Wang, Yiping Liu, Yejie Wang, Wenyang He, Bofei Gao, Muxi Diao, Yanxu Chen, Kelin Fu, Flood Sung, Zhilin Yang, Tianyu Liu, Weiran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16395)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated significant progress in math and code reasoning capabilities. However, existing code benchmark are limited in their ability to evaluate the full spectrum of these capabilities, particularly at the competitive level. To bridge this gap, we introduce OJBench, a novel and challenging benchmark designed to assess the competitive-level code reasoning abilities of LLMs. OJBench comprises 232 programming competition problems from NOI and ICPC, providing a more rigorous test of models' reasoning skills. We conducted a comprehensive evaluation using OJBench on 37 models, including both closed-source and open-source models, reasoning-oriented and non-reasoning-oriented models. Our results indicate that even state-of-the-art reasoning-oriented models, such as o4-mini and Gemini-2.5-pro-exp, struggle with highly challenging competition-level problems. This highlights the significant challenges that models face in competitive-level code reasoning. 

---
# From LLM-anation to LLM-orchestrator: Coordinating Small Models for Data Labeling 

**Authors**: Yao Lu, Zhaiyuan Ji, Jiawei Du, Yu Shanqing, Qi Xuan, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.16393)  

**Abstract**: Although the annotation paradigm based on Large Language Models (LLMs) has made significant breakthroughs in recent years, its actual deployment still has two core bottlenecks: first, the cost of calling commercial APIs in large-scale annotation is very expensive; second, in scenarios that require fine-grained semantic understanding, such as sentiment classification and toxicity classification, the annotation accuracy of LLMs is even lower than that of Small Language Models (SLMs) dedicated to this field. To address these problems, we propose a new paradigm of multi-model cooperative annotation and design a fully automatic annotation framework AutoAnnotator based on this. Specifically, AutoAnnotator consists of two layers. The upper-level meta-controller layer uses the generation and reasoning capabilities of LLMs to select SLMs for annotation, automatically generate annotation code and verify difficult samples; the lower-level task-specialist layer consists of multiple SLMs that perform annotation through multi-model voting. In addition, we use the difficult samples obtained by the secondary review of the meta-controller layer as the reinforcement learning set and fine-tune the SLMs in stages through a continual learning strategy, thereby improving the generalization of SLMs. Extensive experiments show that AutoAnnotator outperforms existing open-source/API LLMs in zero-shot, one-shot, CoT, and majority voting settings. Notably, AutoAnnotator reduces the annotation cost by 74.15% compared to directly annotating with GPT-3.5-turbo, while still improving the accuracy by 6.21%. Project page: this https URL. 

---
# RiOT: Efficient Prompt Refinement with Residual Optimization Tree 

**Authors**: Chenyi Zhou, Zhengyan Shi, Yuan Yao, Lei Liang, Huajun Chen, Qiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16389)  

**Abstract**: Recent advancements in large language models (LLMs) have highlighted their potential across a variety of tasks, but their performance still heavily relies on the design of effective prompts. Existing methods for automatic prompt optimization face two challenges: lack of diversity, limiting the exploration of valuable and innovative directions and semantic drift, where optimizations for one task can degrade performance in others. To address these issues, we propose Residual Optimization Tree (RiOT), a novel framework for automatic prompt optimization. RiOT iteratively refines prompts through text gradients, generating multiple semantically diverse candidates at each step, and selects the best prompt using perplexity. Additionally, RiOT incorporates the text residual connection to mitigate semantic drift by selectively retaining beneficial content across optimization iterations. A tree structure efficiently manages the optimization process, ensuring scalability and flexibility. Extensive experiments across five benchmarks, covering commonsense, mathematical, logical, temporal, and semantic reasoning, demonstrate that RiOT outperforms both previous prompt optimization methods and manual prompting. 

---
# HausaNLP at SemEval-2025 Task 11: Advancing Hausa Text-based Emotion Detection 

**Authors**: Sani Abdullahi Sani, Salim Abubakar, Falalu Ibrahim Lawan, Abdulhamid Abubakar, Maryam Bala  

**Link**: [PDF](https://arxiv.org/pdf/2506.16388)  

**Abstract**: This paper presents our approach to multi-label emotion detection in Hausa, a low-resource African language, as part of SemEval Track A. We fine-tuned AfriBERTa, a transformer-based model pre-trained on African languages, to classify Hausa text into six emotions: anger, disgust, fear, joy, sadness, and surprise. Our methodology involved data preprocessing, tokenization, and model fine-tuning using the Hugging Face Trainer API. The system achieved a validation accuracy of 74.00%, with an F1-score of 73.50%, demonstrating the effectiveness of transformer-based models for emotion detection in low-resource languages. 

---
# Large Language Models in Argument Mining: A Survey 

**Authors**: Hao Li, Viktor Schlegel, Yizheng Sun, Riza Batista-Navarro, Goran Nenadic  

**Link**: [PDF](https://arxiv.org/pdf/2506.16383)  

**Abstract**: Argument Mining (AM), a critical subfield of Natural Language Processing (NLP), focuses on extracting argumentative structures from text. The advent of Large Language Models (LLMs) has profoundly transformed AM, enabling advanced in-context learning, prompt-based generation, and robust cross-domain adaptability. This survey systematically synthesizes recent advancements in LLM-driven AM. We provide a concise review of foundational theories and annotation frameworks, alongside a meticulously curated catalog of datasets. A key contribution is our comprehensive taxonomy of AM subtasks, elucidating how contemporary LLM techniques -- such as prompting, chain-of-thought reasoning, and retrieval augmentation -- have reconfigured their execution. We further detail current LLM architectures and methodologies, critically assess evaluation practices, and delineate pivotal challenges including long-context reasoning, interpretability, and annotation bottlenecks. Conclusively, we highlight emerging trends and propose a forward-looking research agenda for LLM-based computational argumentation, aiming to strategically guide researchers in this rapidly evolving domain. 

---
# InstructTTSEval: Benchmarking Complex Natural-Language Instruction Following in Text-to-Speech Systems 

**Authors**: Kexin Huang, Qian Tu, Liwei Fan, Chenchen Yang, Dong Zhang, Shimin Li, Zhaoye Fei, Qinyuan Cheng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16381)  

**Abstract**: In modern speech synthesis, paralinguistic information--such as a speaker's vocal timbre, emotional state, and dynamic prosody--plays a critical role in conveying nuance beyond mere semantics. Traditional Text-to-Speech (TTS) systems rely on fixed style labels or inserting a speech prompt to control these cues, which severely limits flexibility. Recent attempts seek to employ natural-language instructions to modulate paralinguistic features, substantially improving the generalization of instruction-driven TTS models. Although many TTS systems now support customized synthesis via textual description, their actual ability to interpret and execute complex instructions remains largely unexplored. In addition, there is still a shortage of high-quality benchmarks and automated evaluation metrics specifically designed for instruction-based TTS, which hinders accurate assessment and iterative optimization of these models. To address these limitations, we introduce InstructTTSEval, a benchmark for measuring the capability of complex natural-language style control. We introduce three tasks, namely Acoustic-Parameter Specification, Descriptive-Style Directive, and Role-Play, including English and Chinese subsets, each with 1k test cases (6k in total) paired with reference audio. We leverage Gemini as an automatic judge to assess their instruction-following abilities. Our evaluation of accessible instruction-following TTS systems highlights substantial room for further improvement. We anticipate that InstructTTSEval will drive progress toward more powerful, flexible, and accurate instruction-following TTS. 

---
# Can structural correspondences ground real world representational content in Large Language Models? 

**Authors**: Iwan Williams  

**Link**: [PDF](https://arxiv.org/pdf/2506.16370)  

**Abstract**: Large Language Models (LLMs) such as GPT-4 produce compelling responses to a wide range of prompts. But their representational capacities are uncertain. Many LLMs have no direct contact with extra-linguistic reality: their inputs, outputs and training data consist solely of text, raising the questions (1) can LLMs represent anything and (2) if so, what? In this paper, I explore what it would take to answer these questions according to a structural-correspondence based account of representation, and make an initial survey of this evidence. I argue that the mere existence of structural correspondences between LLMs and worldly entities is insufficient to ground representation of those entities. However, if these structural correspondences play an appropriate role - they are exploited in a way that explains successful task performance - then they could ground real world contents. This requires overcoming a challenge: the text-boundedness of LLMs appears, on the face of it, to prevent them engaging in the right sorts of tasks. 

---
# DISCIE -- Discriminative Closed Information Extraction 

**Authors**: Cedric Möller, Ricardo Usbeck  

**Link**: [PDF](https://arxiv.org/pdf/2506.16348)  

**Abstract**: This paper introduces a novel method for closed information extraction. The method employs a discriminative approach that incorporates type and entity-specific information to improve relation extraction accuracy, particularly benefiting long-tail relations. Notably, this method demonstrates superior performance compared to state-of-the-art end-to-end generative models. This is especially evident for the problem of large-scale closed information extraction where we are confronted with millions of entities and hundreds of relations. Furthermore, we emphasize the efficiency aspect by leveraging smaller models. In particular, the integration of type-information proves instrumental in achieving performance levels on par with or surpassing those of a larger generative model. This advancement holds promise for more accurate and efficient information extraction techniques. 

---
# Analyzing the Influence of Knowledge Graph Information on Relation Extraction 

**Authors**: Cedric Möller, Ricardo Usbeck  

**Link**: [PDF](https://arxiv.org/pdf/2506.16343)  

**Abstract**: We examine the impact of incorporating knowledge graph information on the performance of relation extraction models across a range of datasets. Our hypothesis is that the positions of entities within a knowledge graph provide important insights for relation extraction tasks. We conduct experiments on multiple datasets, each varying in the number of relations, training examples, and underlying knowledge graphs. Our results demonstrate that integrating knowledge graph information significantly enhances performance, especially when dealing with an imbalance in the number of training examples for each relation. We evaluate the contribution of knowledge graph-based features by combining established relation extraction methods with graph-aware Neural Bellman-Ford networks. These features are tested in both supervised and zero-shot settings, demonstrating consistent performance improvements across various datasets. 

---
# Generalizability of Media Frames: Corpus creation and analysis across countries 

**Authors**: Agnese Daffara, Sourabh Dattawad, Sebastian Padó, Tanise Ceron  

**Link**: [PDF](https://arxiv.org/pdf/2506.16337)  

**Abstract**: Frames capture aspects of an issue that are emphasized in a debate by interlocutors and can help us understand how political language conveys different perspectives and ultimately shapes people's opinions. The Media Frame Corpus (MFC) is the most commonly used framework with categories and detailed guidelines for operationalizing frames. It is, however, focused on a few salient U.S. news issues, making it unclear how well these frames can capture news issues in other cultural contexts. To explore this, we introduce FrameNews-PT, a dataset of Brazilian Portuguese news articles covering political and economic news and annotate it within the MFC framework. Through several annotation rounds, we evaluate the extent to which MFC frames generalize to the Brazilian debate issues. We further evaluate how fine-tuned and zero-shot models perform on out-of-domain data. Results show that the 15 MFC frames remain broadly applicable with minor revisions of the guidelines. However, some MFC frames are rarely used, and novel news issues are analyzed using general 'fall-back' frames. We conclude that cross-cultural frame use requires careful consideration. 

---
# PL-Guard: Benchmarking Language Model Safety for Polish 

**Authors**: Aleksandra Krasnodębska, Karolina Seweryn, Szymon Łukasik, Wojciech Kusa  

**Link**: [PDF](https://arxiv.org/pdf/2506.16322)  

**Abstract**: Despite increasing efforts to ensure the safety of large language models (LLMs), most existing safety assessments and moderation tools remain heavily biased toward English and other high-resource languages, leaving majority of global languages underexamined. To address this gap, we introduce a manually annotated benchmark dataset for language model safety classification in Polish. We also create adversarially perturbed variants of these samples designed to challenge model robustness. We conduct a series of experiments to evaluate LLM-based and classifier-based models of varying sizes and architectures. Specifically, we fine-tune three models: Llama-Guard-3-8B, a HerBERT-based classifier (a Polish BERT derivative), and PLLuM, a Polish-adapted Llama-8B model. We train these models using different combinations of annotated data and evaluate their performance, comparing it against publicly available guard models. Results demonstrate that the HerBERT-based classifier achieves the highest overall performance, particularly under adversarial conditions. 

---
# Advancing Automated Speaking Assessment Leveraging Multifaceted Relevance and Grammar Information 

**Authors**: Hao-Chien Lu, Jhen-Ke Lin, Hong-Yun Lin, Chung-Chun Wang, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.16285)  

**Abstract**: Current automated speaking assessment (ASA) systems for use in multi-aspect evaluations often fail to make full use of content relevance, overlooking image or exemplar cues, and employ superficial grammar analysis that lacks detailed error types. This paper ameliorates these deficiencies by introducing two novel enhancements to construct a hybrid scoring model. First, a multifaceted relevance module integrates question and the associated image content, exemplar, and spoken response of an L2 speaker for a comprehensive assessment of content relevance. Second, fine-grained grammar error features are derived using advanced grammar error correction (GEC) and detailed annotation to identify specific error categories. Experiments and ablation studies demonstrate that these components significantly improve the evaluation of content relevance, language use, and overall ASA performance, highlighting the benefits of using richer, more nuanced feature sets for holistic speaking assessment. 

---
# End-to-End Speech Translation for Low-Resource Languages Using Weakly Labeled Data 

**Authors**: Aishwarya Pothula, Bhavana Akkiraju, Srihari Bandarupalli, Charan D, Santosh Kesiraju, Anil Kumar Vuppala  

**Link**: [PDF](https://arxiv.org/pdf/2506.16251)  

**Abstract**: The scarcity of high-quality annotated data presents a significant challenge in developing effective end-to-end speech-to-text translation (ST) systems, particularly for low-resource languages. This paper explores the hypothesis that weakly labeled data can be used to build ST models for low-resource language pairs. We constructed speech-to-text translation datasets with the help of bitext mining using state-of-the-art sentence encoders. We mined the multilingual Shrutilipi corpus to build Shrutilipi-anuvaad, a dataset comprising ST data for language pairs Bengali-Hindi, Malayalam-Hindi, Odia-Hindi, and Telugu-Hindi. We created multiple versions of training data with varying degrees of quality and quantity to investigate the effect of quality versus quantity of weakly labeled data on ST model performance. Results demonstrate that ST systems can be built using weakly labeled data, with performance comparable to massive multi-modal multilingual baselines such as SONAR and SeamlessM4T. 

---
# Comparative Analysis of Abstractive Summarization Models for Clinical Radiology Reports 

**Authors**: Anindita Bhattacharya, Tohida Rehman, Debarshi Kumar Sanyal, Samiran Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2506.16247)  

**Abstract**: The findings section of a radiology report is often detailed and lengthy, whereas the impression section is comparatively more compact and captures key diagnostic conclusions. This research explores the use of advanced abstractive summarization models to generate the concise impression from the findings section of a radiology report. We have used the publicly available MIMIC-CXR dataset. A comparative analysis is conducted on leading pre-trained and open-source large language models, including T5-base, BART-base, PEGASUS-x-base, ChatGPT-4, LLaMA-3-8B, and a custom Pointer Generator Network with a coverage mechanism. To ensure a thorough assessment, multiple evaluation metrics are employed, including ROUGE-1, ROUGE-2, ROUGE-L, METEOR, and BERTScore. By analyzing the performance of these models, this study identifies their respective strengths and limitations in the summarization of medical text. The findings of this paper provide helpful information for medical professionals who need automated summarization solutions in the healthcare sector. 

---
# Web(er) of Hate: A Survey on How Hate Speech Is Typed 

**Authors**: Luna Wang, Andrew Caines, Alice Hutchings  

**Link**: [PDF](https://arxiv.org/pdf/2506.16190)  

**Abstract**: The curation of hate speech datasets involves complex design decisions that balance competing priorities. This paper critically examines these methodological choices in a diverse range of datasets, highlighting common themes and practices, and their implications for dataset reliability. Drawing on Max Weber's notion of ideal types, we argue for a reflexive approach in dataset creation, urging researchers to acknowledge their own value judgments during dataset construction, fostering transparency and methodological rigour. 

---
# JETHICS: Japanese Ethics Understanding Evaluation Dataset 

**Authors**: Masashi Takeshita, Rafal Rzepka  

**Link**: [PDF](https://arxiv.org/pdf/2506.16187)  

**Abstract**: In this work, we propose JETHICS, a Japanese dataset for evaluating ethics understanding of AI models. JETHICS contains 78K examples and is built by following the construction methods of the existing English ETHICS dataset. It includes four categories based normative theories and concepts from ethics and political philosophy; and one representing commonsense morality. Our evaluation experiments on non-proprietary large language models (LLMs) and on GPT-4o reveal that even GPT-4o achieves only an average score of about 0.7, while the best-performing Japanese LLM attains around 0.5, indicating a relatively large room for improvement in current LLMs. 

---
# SGIC: A Self-Guided Iterative Calibration Framework for RAG 

**Authors**: Guanhua Chen, Yutong Yao, Lidia S. Chao, Xuebo Liu, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.16172)  

**Abstract**: Recent research in retrieval-augmented generation (RAG) has concentrated on retrieving useful information from candidate documents. However, numerous methodologies frequently neglect the calibration capabilities of large language models (LLMs), which capitalize on their robust in-context reasoning prowess. This work illustrates that providing LLMs with specific cues substantially improves their calibration efficacy, especially in multi-round calibrations. We present a new SGIC: Self-Guided Iterative Calibration Framework that employs uncertainty scores as a tool. Initially, this framework calculates uncertainty scores to determine both the relevance of each document to the query and the confidence level in the responses produced by the LLMs. Subsequently, it reevaluates these scores iteratively, amalgamating them with prior responses to refine calibration. Furthermore, we introduce an innovative approach for constructing an iterative self-calibration training set, which optimizes LLMs to efficiently harness uncertainty scores for capturing critical information and enhancing response accuracy. Our proposed framework significantly improves performance on both closed-source and open-weight LLMs. 

---
# Under the Shadow of Babel: How Language Shapes Reasoning in LLMs 

**Authors**: Chenxi Wang, Yixuan Zhang, Lang Gao, Zixiang Xu, Zirui Song, Yanbo Wang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.16151)  

**Abstract**: Language is not only a tool for communication but also a medium for human cognition and reasoning. If, as linguistic relativity suggests, the structure of language shapes cognitive patterns, then large language models (LLMs) trained on human language may also internalize the habitual logical structures embedded in different languages. To examine this hypothesis, we introduce BICAUSE, a structured bilingual dataset for causal reasoning, which includes semantically aligned Chinese and English samples in both forward and reversed causal forms. Our study reveals three key findings: (1) LLMs exhibit typologically aligned attention patterns, focusing more on causes and sentence-initial connectives in Chinese, while showing a more balanced distribution in English. (2) Models internalize language-specific preferences for causal word order and often rigidly apply them to atypical inputs, leading to degraded performance, especially in Chinese. (3) When causal reasoning succeeds, model representations converge toward semantically aligned abstractions across languages, indicating a shared understanding beyond surface form. Overall, these results suggest that LLMs not only mimic surface linguistic forms but also internalize the reasoning biases shaped by language. Rooted in cognitive linguistic theory, this phenomenon is for the first time empirically verified through structural analysis of model internals. 

---
# FinCoT: Grounding Chain-of-Thought in Expert Financial Reasoning 

**Authors**: Natapong Nitarach, Warit Sirichotedumrong, Panop Pitchayarthorn, Pittawat Taveekitworachai, Potsawee Manakul, Kunat Pipatanakul  

**Link**: [PDF](https://arxiv.org/pdf/2506.16123)  

**Abstract**: This paper presents FinCoT, a structured chain-of-thought (CoT) prompting approach that incorporates insights from domain-specific expert financial reasoning to guide the reasoning traces of large language models. We investigate that there are three main prompting styles in FinNLP: (1) standard prompting--zero-shot prompting; (2) unstructured CoT--CoT prompting without an explicit reasoning structure, such as the use of tags; and (3) structured CoT prompting--CoT prompting with explicit instructions or examples that define structured reasoning steps. Previously, FinNLP has primarily focused on prompt engineering with either standard or unstructured CoT prompting. However, structured CoT prompting has received limited attention in prior work. Furthermore, the design of reasoning structures in structured CoT prompting is often based on heuristics from non-domain experts. In this study, we investigate each prompting approach in FinNLP. We evaluate the three main prompting styles and FinCoT on CFA-style questions spanning ten financial domains. We observe that FinCoT improves performance from 63.2% to 80.5% and Qwen-2.5-7B-Instruct from 69.7% to 74.2%, while reducing generated tokens eight-fold compared to structured CoT prompting. Our findings show that domain-aligned structured prompts not only improve performance and reduce inference costs but also yield more interpretable and expert-aligned reasoning traces. 

---
# Cyberbullying Detection in Hinglish Text Using MURIL and Explainable AI 

**Authors**: Devesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.16066)  

**Abstract**: The growth of digital communication platforms has led to increased cyberbullying incidents worldwide, creating a need for automated detection systems to protect users. The rise of code-mixed Hindi-English (Hinglish) communication on digital platforms poses challenges for existing cyberbullying detection systems, which were designed primarily for monolingual text. This paper presents a framework for cyberbullying detection in Hinglish text using the Multilingual Representations for Indian Languages (MURIL) architecture to address limitations in current approaches. Evaluation across six benchmark datasets -- Bohra \textit{et al.}, BullyExplain, BullySentemo, Kumar \textit{et al.}, HASOC 2021, and Mendeley Indo-HateSpeech -- shows that the MURIL-based approach outperforms existing multilingual models including RoBERTa and IndicBERT, with improvements of 1.36 to 13.07 percentage points and accuracies of 86.97\% on Bohra, 84.62\% on BullyExplain, 86.03\% on BullySentemo, 75.41\% on Kumar datasets, 83.92\% on HASOC 2021, and 94.63\% on Mendeley dataset. The framework includes explainability features through attribution analysis and cross-linguistic pattern recognition. Ablation studies show that selective layer freezing, appropriate classification head design, and specialized preprocessing for code-mixed content improve detection performance, while failure analysis identifies challenges including context-dependent interpretation, cultural understanding, and cross-linguistic sarcasm detection, providing directions for future research in multilingual cyberbullying detection. 

---
# Self-Critique-Guided Curiosity Refinement: Enhancing Honesty and Helpfulness in Large Language Models via In-Context Learning 

**Authors**: Duc Hieu Ho, Chenglin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.16064)  

**Abstract**: Large language models (LLMs) have demonstrated robust capabilities across various natural language tasks. However, producing outputs that are consistently honest and helpful remains an open challenge. To overcome this challenge, this paper tackles the problem through two complementary directions. It conducts a comprehensive benchmark evaluation of ten widely used large language models, including both proprietary and open-weight models from OpenAI, Meta, and Google. In parallel, it proposes a novel prompting strategy, self-critique-guided curiosity refinement prompting. The key idea behind this strategy is enabling models to self-critique and refine their responses without additional training. The proposed method extends the curiosity-driven prompting strategy by incorporating two lightweight in-context steps including self-critique step and refinement step.
The experiment results on the HONESET dataset evaluated using the framework $\mathrm{H}^2$ (honesty and helpfulness), which was executed with GPT-4o as a judge of honesty and helpfulness, show consistent improvements across all models. The approach reduces the number of poor-quality responses, increases high-quality responses, and achieves relative gains in $\mathrm{H}^2$ scores ranging from 1.4% to 4.3% compared to curiosity-driven prompting across evaluated models. These results highlight the effectiveness of structured self-refinement as a scalable and training-free strategy to improve the trustworthiness of LLMs outputs. 

---
# Knee-Deep in C-RASP: A Transformer Depth Hierarchy 

**Authors**: Andy Yang, Michaël Cadilhac, David Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16055)  

**Abstract**: It has been observed that transformers with greater depth (that is, more layers) have more capabilities, but can we establish formally which capabilities are gained with greater depth? We answer this question with a theoretical proof followed by an empirical study. First, we consider transformers that round to fixed precision except inside attention. We show that this subclass of transformers is expressively equivalent to the programming language C-RASP and this equivalence preserves depth. Second, we prove that deeper C-RASP programs are more expressive than shallower C-RASP programs, implying that deeper transformers are more expressive than shallower transformers (within the subclass mentioned above). These results are established by studying a form of temporal logic with counting operators, which was shown equivalent to C-RASP in previous work. Finally, we provide empirical evidence that our theory predicts the depth required for transformers without positional encodings to length-generalize on a family of sequential dependency tasks. 

---
# A Hybrid DeBERTa and Gated Broad Learning System for Cyberbullying Detection in English Text 

**Authors**: Devesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.16052)  

**Abstract**: The proliferation of online communication platforms has created unprecedented opportunities for global connectivity while simultaneously enabling harmful behaviors such as cyberbullying, which affects approximately 54.4\% of teenagers according to recent research. This paper presents a hybrid architecture that combines the contextual understanding capabilities of transformer-based models with the pattern recognition strengths of broad learning systems for effective cyberbullying detection. This approach integrates a modified DeBERTa model augmented with Squeeze-and-Excitation blocks and sentiment analysis capabilities with a Gated Broad Learning System (GBLS) classifier, creating a synergistic framework that outperforms existing approaches across multiple benchmark datasets. The proposed ModifiedDeBERTa + GBLS model achieved good performance on four English datasets: 79.3\% accuracy on HateXplain, 95.41\% accuracy on SOSNet, 91.37\% accuracy on Mendeley-I, and 94.67\% accuracy on Mendeley-II. Beyond performance gains, the framework incorporates comprehensive explainability mechanisms including token-level attribution analysis, LIME-based local interpretations, and confidence calibration, addressing critical transparency requirements in automated content moderation. Ablation studies confirm the meaningful contribution of each architectural component, while failure case analysis reveals specific challenges in detecting implicit bias and sarcastic content, providing valuable insights for future improvements in cyberbullying detection systems. 

---
# DynScaling: Efficient Verifier-free Inference Scaling via Dynamic and Integrated Sampling 

**Authors**: Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2506.16043)  

**Abstract**: Inference-time scaling has proven effective in boosting large language model (LLM) performance through increased test-time computation. Yet, its practical application is often hindered by reliance on external verifiers or a lack of optimization for realistic computational constraints. We propose DynScaling, which addresses these limitations through two primary innovations: an integrated parallel-sequential sampling strategy and a bandit-based dynamic budget allocation framework. The integrated sampling strategy unifies parallel and sequential sampling by constructing synthetic sequential reasoning chains from initially independent parallel responses, promoting diverse and coherent reasoning trajectories. The dynamic budget allocation framework formulates the allocation of computational resources as a multi-armed bandit problem, adaptively distributing the inference budget across queries based on the uncertainty of previously sampled responses, thereby maximizing computational efficiency. By combining these components, DynScaling effectively improves LLM performance under practical resource constraints without the need for external verifiers. Experimental results demonstrate that DynScaling consistently surpasses existing verifier-free inference scaling baselines in both task performance and computational cost. 

---
# Enhancing Document-Level Question Answering via Multi-Hop Retrieval-Augmented Generation with LLaMA 3 

**Authors**: Xinyue Huang, Ziqi Lin, Fang Sun, Wenchao Zhang, Kejian Tong, Yunbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16037)  

**Abstract**: This paper presents a novel Retrieval-Augmented Generation (RAG) framework tailored for complex question answering tasks, addressing challenges in multi-hop reasoning and contextual understanding across lengthy documents. Built upon LLaMA 3, the framework integrates a dense retrieval module with advanced context fusion and multi-hop reasoning mechanisms, enabling more accurate and coherent response generation. A joint optimization strategy combining retrieval likelihood and generation cross-entropy improves the model's robustness and adaptability. Experimental results show that the proposed system outperforms existing retrieval-augmented and generative baselines, confirming its effectiveness in delivering precise, contextually grounded answers. 

---
# EvoLM: In Search of Lost Language Model Training Dynamics 

**Authors**: Zhenting Qi, Fan Nie, Alexandre Alahi, James Zou, Himabindu Lakkaraju, Yilun Du, Eric Xing, Sham Kakade, Hanlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16029)  

**Abstract**: Modern language model (LM) training has been divided into multiple stages, making it difficult for downstream developers to evaluate the impact of design choices made at each stage. We present EvoLM, a model suite that enables systematic and transparent analysis of LMs' training dynamics across pre-training, continued pre-training, supervised fine-tuning, and reinforcement learning. By training over 100 LMs with 1B and 4B parameters from scratch, we rigorously evaluate both upstream (language modeling) and downstream (problem-solving) reasoning capabilities, including considerations of both in-domain and out-of-domain generalization. Key insights highlight the diminishing returns from excessive pre-training and post-training, the importance and practices of mitigating forgetting during domain-specific continued pre-training, the crucial role of continued pre-training in bridging pre-training and post-training phases, and various intricate trade-offs when configuring supervised fine-tuning and reinforcement learning. To facilitate open research and reproducibility, we release all pre-trained and post-trained models, training datasets for all stages, and our entire training and evaluation pipeline. 

---
# From General to Targeted Rewards: Surpassing GPT-4 in Open-Ended Long-Context Generation 

**Authors**: Zhihan Guo, Jiele Wu, Wenqian Cui, Yifei Zhang, Minda Hu, Yufei Wang, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2506.16024)  

**Abstract**: Current research on long-form context in Large Language Models (LLMs) primarily focuses on the understanding of long-contexts, the Open-ended Long Text Generation (Open-LTG) remains insufficiently explored. Training a long-context generation model requires curation of gold standard reference data, which is typically nonexistent for informative Open-LTG tasks. However, previous methods only utilize general assessments as reward signals, which limits accuracy. To bridge this gap, we introduce ProxyReward, an innovative reinforcement learning (RL) based framework, which includes a dataset and a reward signal computation method. Firstly, ProxyReward Dataset generation is accomplished through simple prompts that enables the model to create automatically, obviating extensive labeled data or significant manual effort. Secondly, ProxyReward Signal offers a targeted evaluation of information comprehensiveness and accuracy for specific questions. The experimental results indicate that our method ProxyReward surpasses even GPT-4-Turbo. It can significantly enhance performance by 20% on the Open-LTG task when training widely used open-source models, while also surpassing the LLM-as-a-Judge approach. Our work presents effective methods to enhance the ability of LLMs to address complex open-ended questions posed by human. 

---
# Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion 

**Authors**: Markus Frohmann, Gabriel Meseguer-Brocal, Markus Schedl, Elena V. Epure  

**Link**: [PDF](https://arxiv.org/pdf/2506.15981)  

**Abstract**: The rapid advancement of AI-based music generation tools is revolutionizing the music industry but also posing challenges to artists, copyright holders, and providers alike. This necessitates reliable methods for detecting such AI-generated content. However, existing detectors, relying on either audio or lyrics, face key practical limitations: audio-based detectors fail to generalize to new or unseen generators and are vulnerable to audio perturbations; lyrics-based methods require cleanly formatted and accurate lyrics, unavailable in practice. To overcome these limitations, we propose a novel, practically grounded approach: a multimodal, modular late-fusion pipeline that combines automatically transcribed sung lyrics and speech features capturing lyrics-related information within the audio. By relying on lyrical aspects directly from audio, our method enhances robustness, mitigates susceptibility to low-level artifacts, and enables practical applicability. Experiments show that our method, DE-detect, outperforms existing lyrics-based detectors while also being more robust to audio perturbations. Thus, it offers an effective, robust solution for detecting AI-generated music in real-world scenarios. Our code is available at this https URL. 

---
# A Vietnamese Dataset for Text Segmentation and Multiple Choices Reading Comprehension 

**Authors**: Toan Nguyen Hai, Ha Nguyen Viet, Truong Quan Xuan, Duc Do Minh  

**Link**: [PDF](https://arxiv.org/pdf/2506.15978)  

**Abstract**: Vietnamese, the 20th most spoken language with over 102 million native speakers, lacks robust resources for key natural language processing tasks such as text segmentation and machine reading comprehension (MRC). To address this gap, we present VSMRC, the Vietnamese Text Segmentation and Multiple-Choice Reading Comprehension Dataset. Sourced from Vietnamese Wikipedia, our dataset includes 15,942 documents for text segmentation and 16,347 synthetic multiple-choice question-answer pairs generated with human quality assurance, ensuring a reliable and diverse resource. Experiments show that mBERT consistently outperforms monolingual models on both tasks, achieving an accuracy of 88.01% on MRC test set and an F1 score of 63.15\% on text segmentation test set. Our analysis reveals that multilingual models excel in NLP tasks for Vietnamese, suggesting potential applications to other under-resourced languages. VSMRC is available at HuggingFace 

---
# Reranking-based Generation for Unbiased Perspective Summarization 

**Authors**: Narutatsu Ri, Nicholas Deas, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2506.15925)  

**Abstract**: Generating unbiased summaries in real-world settings such as political perspective summarization remains a crucial application of Large Language Models (LLMs). Yet, existing evaluation frameworks rely on traditional metrics for measuring key attributes such as coverage and faithfulness without verifying their applicability, and efforts to develop improved summarizers are still nascent. We address these gaps by (1) identifying reliable metrics for measuring perspective summary quality, and (2) investigating the efficacy of LLM-based methods beyond zero-shot inference. Namely, we build a test set for benchmarking metric reliability using human annotations and show that traditional metrics underperform compared to language model-based metrics, which prove to be strong evaluators. Using these metrics, we show that reranking-based methods yield strong results, and preference tuning with synthetically generated and reranking-labeled data further boosts performance. Our findings aim to contribute to the reliable evaluation and development of perspective summarization methods. 

---
# From RAG to Agentic: Validating Islamic-Medicine Responses with LLM Agents 

**Authors**: Mohammad Amaan Sayeed, Mohammed Talha Alam, Raza Imam, Shahab Saquib Sohail, Amir Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2506.15911)  

**Abstract**: Centuries-old Islamic medical texts like Avicenna's Canon of Medicine and the Prophetic Tibb-e-Nabawi encode a wealth of preventive care, nutrition, and holistic therapies, yet remain inaccessible to many and underutilized in modern AI systems. Existing language-model benchmarks focus narrowly on factual recall or user preference, leaving a gap in validating culturally grounded medical guidance at scale. We propose a unified evaluation pipeline, Tibbe-AG, that aligns 30 carefully curated Prophetic-medicine questions with human-verified remedies and compares three LLMs (LLaMA-3, Mistral-7B, Qwen2-7B) under three configurations: direct generation, retrieval-augmented generation, and a scientific self-critique filter. Each answer is then assessed by a secondary LLM serving as an agentic judge, yielding a single 3C3H quality score. Retrieval improves factual accuracy by 13%, while the agentic prompt adds another 10% improvement through deeper mechanistic insight and safety considerations. Our results demonstrate that blending classical Islamic texts with retrieval and self-evaluation enables reliable, culturally sensitive medical question-answering. 

---
# Language Models can perform Single-Utterance Self-Correction of Perturbed Reasoning 

**Authors**: Sam Silver, Jimin Sun, Ivan Zhang, Sara Hooker, Eddie Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.15894)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive mathematical reasoning capabilities, yet their performance remains brittle to minor variations in problem description and prompting strategy. Furthermore, reasoning is vulnerable to sampling-induced errors which autoregressive models must primarily address using self-correction via additionally-generated tokens. To better understand self-correction capabilities of recent models, we conduct experiments measuring models' ability to self-correct synthetic perturbations introduced into their Chain of Thought (CoT) reasoning. We observe robust single-utterance intrinsic self-correction behavior across a range of open-weight models and datasets, ranging from subtle, implicit corrections to explicit acknowledgments and corrections of errors. Our findings suggest that LLMs, including those not finetuned for long CoT, may possess stronger intrinsic self-correction capabilities than commonly shown in the literature. The presence of this ability suggests that recent "reasoning" model work involves amplification of traits already meaningfully present in models. 

---
# Entropy-Driven Pre-Tokenization for Byte-Pair Encoding 

**Authors**: Yifan Hu, Frank Liang, Dachuan Zhao, Jonathan Geuter, Varshini Reddy, Craig W. Schmidt, Chris Tanner  

**Link**: [PDF](https://arxiv.org/pdf/2506.15889)  

**Abstract**: Byte-Pair Encoding (BPE) has become a widely adopted subword tokenization method in modern language models due to its simplicity and strong empirical performance across downstream tasks. However, applying BPE to unsegmented languages such as Chinese presents significant challenges, as its frequency-driven merge operation is agnostic to linguistic boundaries. To address this, we propose two entropy-informed pre-tokenization strategies that guide BPE segmentation using unsupervised information-theoretic cues. The first approach uses pointwise mutual information and left/right entropy to identify coherent character spans, while the second leverages predictive entropy derived from a pretrained GPT-2 model to detect boundary uncertainty. We evaluate both methods on a subset of the PKU dataset and demonstrate substantial improvements in segmentation precision, recall, and F1 score compared to standard BPE. Our results suggest that entropy-guided pre-tokenization not only enhances alignment with gold-standard linguistic units but also offers a promising direction for improving tokenization quality in low-resource and multilingual settings. 

---
# Finance Language Model Evaluation (FLaME) 

**Authors**: Glenn Matlin, Mika Okamoto, Huzaifa Pardawala, Yang Yang, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2506.15846)  

**Abstract**: Language Models (LMs) have demonstrated impressive capabilities with core Natural Language Processing (NLP) tasks. The effectiveness of LMs for highly specialized knowledge-intensive tasks in finance remains difficult to assess due to major gaps in the methodologies of existing evaluation frameworks, which have caused an erroneous belief in a far lower bound of LMs' performance on common Finance NLP (FinNLP) tasks. To demonstrate the potential of LMs for these FinNLP tasks, we present the first holistic benchmarking suite for Financial Language Model Evaluation (FLaME). We are the first research paper to comprehensively study LMs against 'reasoning-reinforced' LMs, with an empirical study of 23 foundation LMs over 20 core NLP tasks in finance. We open-source our framework software along with all data and results. 

---
# MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents 

**Authors**: Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15841)  

**Abstract**: Modern language agents must operate over long-horizon, multi-turn interactions, where they retrieve external information, adapt to observations, and answer interdependent queries. Yet, most LLM systems rely on full-context prompting, appending all past turns regardless of their relevance. This leads to unbounded memory growth, increased computational costs, and degraded reasoning performance on out-of-distribution input lengths. We introduce MEM1, an end-to-end reinforcement learning framework that enables agents to operate with constant memory across long multi-turn tasks. At each turn, MEM1 updates a compact shared internal state that jointly supports memory consolidation and reasoning. This state integrates prior memory with new observations from the environment while strategically discarding irrelevant or redundant information. To support training in more realistic and compositional settings, we propose a simple yet effective and scalable approach to constructing multi-turn environments by composing existing datasets into arbitrarily complex task sequences. Experiments across three domains, including internal retrieval QA, open-domain web QA, and multi-turn web shopping, show that MEM1-7B improves performance by 3.5x while reducing memory usage by 3.7x compared to Qwen2.5-14B-Instruct on a 16-objective multi-hop QA task, and generalizes beyond the training horizon. Our results demonstrate the promise of reasoning-driven memory consolidation as a scalable alternative to existing solutions for training long-horizon interactive agents, where both efficiency and performance are optimized. 

---
# Rethinking LLM Training through Information Geometry and Quantum Metrics 

**Authors**: Riccardo Di Sipio  

**Link**: [PDF](https://arxiv.org/pdf/2506.15830)  

**Abstract**: Optimization in large language models (LLMs) unfolds over high-dimensional parameter spaces with non-Euclidean structure. Information geometry frames this landscape using the Fisher information metric, enabling more principled learning via natural gradient descent. Though often impractical, this geometric lens clarifies phenomena such as sharp minima, generalization, and observed scaling laws. We argue that curvature-aware approaches deepen our understanding of LLM training. Finally, we speculate on quantum analogies based on the Fubini-Study metric and Quantum Fisher Information, hinting at efficient optimization in quantum-enhanced systems. 

---
# Veracity: An Open-Source AI Fact-Checking System 

**Authors**: Taylor Lynn Curtis, Maximilian Puelma Touzel, William Garneau, Manon Gruaz, Mike Pinder, Li Wei Wang, Sukanya Krishna, Luda Cohen, Jean-François Godbout, Reihaneh Rabbany, Kellin Pelrine  

**Link**: [PDF](https://arxiv.org/pdf/2506.15794)  

**Abstract**: The proliferation of misinformation poses a significant threat to society, exacerbated by the capabilities of generative AI. This demo paper introduces Veracity, an open-source AI system designed to empower individuals to combat misinformation through transparent and accessible fact-checking. Veracity leverages the synergy between Large Language Models (LLMs) and web retrieval agents to analyze user-submitted claims and provide grounded veracity assessments with intuitive explanations. Key features include multilingual support, numerical scoring of claim veracity, and an interactive interface inspired by familiar messaging applications. This paper will showcase Veracity's ability to not only detect misinformation but also explain its reasoning, fostering media literacy and promoting a more informed society. 

---
# Dissecting the SWE-Bench Leaderboards: Profiling Submitters and Architectures of LLM- and Agent-Based Repair Systems 

**Authors**: Matias Martinez, Xavier Franch  

**Link**: [PDF](https://arxiv.org/pdf/2506.17208)  

**Abstract**: The rapid progress in Automated Program Repair (APR) has been driven by advances in AI, particularly large language models (LLMs) and agent-based systems. SWE-Bench is a recent benchmark designed to evaluate LLM-based repair systems using real issues and pull requests mined from 12 popular open-source Python repositories. Its public leaderboards, SWE-Bench Lite and SWE-Bench Verified, have become central platforms for tracking progress and comparing solutions. However, because the submission process does not require detailed documentation, the architectural design and origin of many solutions remain unclear. In this paper, we present the first comprehensive study of all submissions to the SWE-Bench Lite (68 entries) and Verified (79 entries) leaderboards, analyzing 67 unique approaches across dimensions such as submitter type, product availability, LLM usage, and system architecture. Our findings reveal the dominance of proprietary LLMs (especially Claude 3.5/3.7), the presence of both agentic and non-agentic designs, and a contributor base spanning from individual developers to large tech companies. 

---
# MEXA: Towards General Multimodal Reasoning with Dynamic Multi-Expert Aggregation 

**Authors**: Shoubin Yu, Yue Zhang, Ziyang Wang, Jaehong Yoon, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.17113)  

**Abstract**: Combining pre-trained expert models offers substantial potential for scalable multimodal reasoning, but building a unified framework remains challenging due to the increasing diversity of input modalities and task complexity. For instance, medical diagnosis requires precise reasoning over structured clinical tables, while financial forecasting depends on interpreting plot-based data to make informed predictions. To tackle this challenge, we introduce MEXA, a training-free framework that performs modality- and task-aware aggregation of multiple expert models to enable effective multimodal reasoning across diverse and distinct domains. MEXA dynamically selects expert models based on the input modality and the task-specific reasoning demands (i.e., skills). Each expert model, specialized in a modality task pair, generates interpretable textual reasoning outputs. MEXA then aggregates and reasons over these outputs using a Large Reasoning Model (LRM) to produce the final answer. This modular design allows flexible and transparent multimodal reasoning across diverse domains without additional training overhead. We extensively evaluate our approach on diverse multimodal benchmarks, including Video Reasoning, Audio Reasoning, 3D Understanding, and Medical QA. MEXA consistently delivers performance improvements over strong multimodal baselines, highlighting the effectiveness and broad applicability of our expert-driven selection and aggregation in diverse multimodal reasoning tasks. 

---
# Are Bias Evaluation Methods Biased ? 

**Authors**: Lina Berrayana, Sean Rooney, Luis Garcés-Erice, Ioana Giurgiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17111)  

**Abstract**: The creation of benchmarks to evaluate the safety of Large Language Models is one of the key activities within the trusted AI community. These benchmarks allow models to be compared for different aspects of safety such as toxicity, bias, harmful behavior etc. Independent benchmarks adopt different approaches with distinct data sets and evaluation methods. We investigate how robust such benchmarks are by using different approaches to rank a set of representative models for bias and compare how similar are the overall rankings. We show that different but widely used bias evaluations methods result in disparate model rankings. We conclude with recommendations for the community in the usage of such benchmarks. 

---
# From Concepts to Components: Concept-Agnostic Attention Module Discovery in Transformers 

**Authors**: Jingtong Su, Julia Kempe, Karen Ullrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.17052)  

**Abstract**: Transformers have achieved state-of-the-art performance across language and vision tasks. This success drives the imperative to interpret their internal mechanisms with the dual goals of enhancing performance and improving behavioral control. Attribution methods help advance interpretability by assigning model outputs associated with a target concept to specific model components. Current attribution research primarily studies multi-layer perceptron neurons and addresses relatively simple concepts such as factual associations (e.g., Paris is located in France). This focus tends to overlook the impact of the attention mechanism and lacks a unified approach for analyzing more complex concepts. To fill these gaps, we introduce Scalable Attention Module Discovery (SAMD), a concept-agnostic method for mapping arbitrary, complex concepts to specific attention heads of general transformer models. We accomplish this by representing each concept as a vector, calculating its cosine similarity with each attention head, and selecting the TopK-scoring heads to construct the concept-associated attention module. We then propose Scalar Attention Module Intervention (SAMI), a simple strategy to diminish or amplify the effects of a concept by adjusting the attention module using only a single scalar parameter. Empirically, we demonstrate SAMD on concepts of varying complexity, and visualize the locations of their corresponding modules. Our results demonstrate that module locations remain stable before and after LLM post-training, and confirm prior work on the mechanics of LLM multilingualism. Through SAMI, we facilitate jailbreaking on HarmBench (+72.7%) by diminishing "safety" and improve performance on the GSM8K benchmark (+1.6%) by amplifying "reasoning". Lastly, we highlight the domain-agnostic nature of our approach by suppressing the image classification accuracy of vision transformers on ImageNet. 

---
# Latent Concept Disentanglement in Transformer-based Language Models 

**Authors**: Guan Zhe Hong, Bhavya Vasudeva, Vatsal Sharan, Cyrus Rashtchian, Prabhakar Raghavan, Rina Panigrahy  

**Link**: [PDF](https://arxiv.org/pdf/2506.16975)  

**Abstract**: When large language models (LLMs) use in-context learning (ICL) to solve a new task, they seem to grasp not only the goal of the task but also core, latent concepts in the demonstration examples. This begs the question of whether transformers represent latent structures as part of their computation or whether they take shortcuts to solve the problem. Prior mechanistic work on ICL does not address this question because it does not sufficiently examine the relationship between the learned representation and the latent concept, and the considered problem settings often involve only single-step reasoning. In this work, we examine how transformers disentangle and use latent concepts. We show that in 2-hop reasoning tasks with a latent, discrete concept, the model successfully identifies the latent concept and does step-by-step concept composition. In tasks parameterized by a continuous latent concept, we find low-dimensional subspaces in the representation space where the geometry mimics the underlying parameterization. Together, these results refine our understanding of ICL and the representation of transformers, and they provide evidence for highly localized structures in the model that disentangle latent concepts in ICL tasks. 

---
# Enhancing Step-by-Step and Verifiable Medical Reasoning in MLLMs 

**Authors**: Haoran Sun, Yankai Jiang, Wenjie Lou, Yujie Zhang, Wenjie Li, Lilong Wang, Mianxin Liu, Lei Liu, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16962)  

**Abstract**: Multimodal large language models (MLLMs) have begun to demonstrate robust reasoning capabilities on general tasks, yet their application in the medical domain remains in its early stages. Constructing chain-of-thought (CoT) training data is essential for bolstering the reasoning abilities of medical MLLMs. However, existing approaches exhibit a deficiency in offering a comprehensive framework for searching and evaluating effective reasoning paths towards critical diagnosis. To address this challenge, we propose Mentor-Intern Collaborative Search (MICS), a novel reasoning-path searching scheme to generate rigorous and effective medical CoT data. MICS first leverages mentor models to initialize the reasoning, one step at a time, then prompts each intern model to continue the thinking along those initiated paths, and finally selects the optimal reasoning path according to the overall reasoning performance of multiple intern models. The reasoning performance is determined by an MICS-Score, which assesses the quality of generated reasoning paths. Eventually, we construct MMRP, a multi-task medical reasoning dataset with ranked difficulty, and Chiron-o1, a new medical MLLM devised via a curriculum learning strategy, with robust visual question-answering and generalizable reasoning capabilities. Extensive experiments demonstrate that Chiron-o1, trained on our CoT dataset constructed using MICS, achieves state-of-the-art performance across a list of medical visual question answering and reasoning benchmarks. Codes are available at GitHub - manglu097/Chiron-o1: Enhancing Step-by-Step and Verifiable Medical Reasoning in MLLMs 

---
# Large Language Models as Psychological Simulators: A Methodological Guide 

**Authors**: Zhicheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.16702)  

**Abstract**: Large language models (LLMs) offer emerging opportunities for psychological and behavioral research, but methodological guidance is lacking. This article provides a framework for using LLMs as psychological simulators across two primary applications: simulating roles and personas to explore diverse contexts, and serving as computational models to investigate cognitive processes. For simulation, we present methods for developing psychologically grounded personas that move beyond demographic categories, with strategies for validation against human data and use cases ranging from studying inaccessible populations to prototyping research instruments. For cognitive modeling, we synthesize emerging approaches for probing internal representations, methodological advances in causal interventions, and strategies for relating model behavior to human cognition. We address overarching challenges including prompt sensitivity, temporal limitations from training data cutoffs, and ethical considerations that extend beyond traditional human subjects review. Throughout, we emphasize the need for transparency about model capabilities and constraints. Together, this framework integrates emerging empirical evidence about LLM performance--including systematic biases, cultural limitations, and prompt brittleness--to help researchers wrangle these challenges and leverage the unique capabilities of LLMs in psychological research. 

---
# From Prompts to Constructs: A Dual-Validity Framework for LLM Research in Psychology 

**Authors**: Zhicheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.16697)  

**Abstract**: Large language models (LLMs) are rapidly being adopted across psychology, serving as research tools, experimental subjects, human simulators, and computational models of cognition. However, the application of human measurement tools to these systems can produce contradictory results, raising concerns that many findings are measurement phantoms--statistical artifacts rather than genuine psychological phenomena. In this Perspective, we argue that building a robust science of AI psychology requires integrating two of our field's foundational pillars: the principles of reliable measurement and the standards for sound causal inference. We present a dual-validity framework to guide this integration, which clarifies how the evidence needed to support a claim scales with its scientific ambition. Using an LLM to classify text may require only basic accuracy checks, whereas claiming it can simulate anxiety demands a far more rigorous validation process. Current practice systematically fails to meet these requirements, often treating statistical pattern matching as evidence of psychological phenomena. The same model output--endorsing "I am anxious"--requires different validation strategies depending on whether researchers claim to measure, characterize, simulate, or model psychological constructs. Moving forward requires developing computational analogues of psychological constructs and establishing clear, scalable standards of evidence rather than the uncritical application of human measurement tools. 

---
# Advancing Harmful Content Detection in Organizational Research: Integrating Large Language Models with Elo Rating System 

**Authors**: Mustafa Akben, Aaron Satko  

**Link**: [PDF](https://arxiv.org/pdf/2506.16575)  

**Abstract**: Large language models (LLMs) offer promising opportunities for organizational research. However, their built-in moderation systems can create problems when researchers try to analyze harmful content, often refusing to follow certain instructions or producing overly cautious responses that undermine validity of the results. This is particularly problematic when analyzing organizational conflicts such as microaggressions or hate speech. This paper introduces an Elo rating-based method that significantly improves LLM performance for harmful content analysis In two datasets, one focused on microaggression detection and the other on hate speech, we find that our method outperforms traditional LLM prompting techniques and conventional machine learning models on key measures such as accuracy, precision, and F1 scores. Advantages include better reliability when analyzing harmful content, fewer false positives, and greater scalability for large-scale datasets. This approach supports organizational applications, including detecting workplace harassment, assessing toxic communication, and fostering safer and more inclusive work environments. 

---
# Revela: Dense Retriever Learning via Language Modeling 

**Authors**: Fengyu Cai, Tong Chen, Xinran Zhao, Sihao Chen, Hongming Zhang, Sherry Tongshuang Wu, Iryna Gurevych, Heinz Koeppl  

**Link**: [PDF](https://arxiv.org/pdf/2506.16552)  

**Abstract**: Dense retrievers play a vital role in accessing external and specialized knowledge to augment language models (LMs). Training dense retrievers typically requires annotated query-document pairs, which are costly and hard to obtain in specialized domains such as code-motivating growing interest in self-supervised retriever learning. Since LMs are trained to capture token-level dependencies through a self-supervised learning objective (i.e., next-token prediction), we can analogously cast retrieval as learning dependencies among chunks of tokens. This analogy naturally leads to the question: How can we adapt self-supervised learning objectives in the spirit of language modeling to train retrievers?
To answer this question, we introduce Revela, a unified and scalable training framework for self-supervised retriever learning via language modeling. Revela models semantic dependencies among documents by conditioning next-token prediction on both local and cross-document context through an in-batch attention mechanism. This attention is weighted by retriever-computed similarity scores, enabling the retriever to be optimized as part of language modeling. We evaluate Revela on both general-domain (BEIR) and domain-specific (CoIR) benchmarks across various retriever backbones. At a comparable parameter scale, Revela outperforms the previous best method with absolute improvements of 5.2 % (18.3 % relative) and 5.6 % (14.4 % relative) on NDCG@10, respectively, underscoring its effectiveness. Performance increases with model size, highlighting both the scalability of our approach and its promise for self-supervised retriever learning. 

---
# Do We Talk to Robots Like Therapists, and Do They Respond Accordingly? Language Alignment in AI Emotional Support 

**Authors**: Sophie Chiang, Guy Laban, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2506.16473)  

**Abstract**: As conversational agents increasingly engage in emotionally supportive dialogue, it is important to understand how closely their interactions resemble those in traditional therapy settings. This study investigates whether the concerns shared with a robot align with those shared in human-to-human (H2H) therapy sessions, and whether robot responses semantically mirror those of human therapists. We analyzed two datasets: one of interactions between users and professional therapists (Hugging Face's NLP Mental Health Conversations), and another involving supportive conversations with a social robot (QTrobot from LuxAI) powered by a large language model (LLM, GPT-3.5). Using sentence embeddings and K-means clustering, we assessed cross-agent thematic alignment by applying a distance-based cluster-fitting method that evaluates whether responses from one agent type map to clusters derived from the other, and validated it using Euclidean distances. Results showed that 90.88% of robot conversation disclosures could be mapped to clusters from the human therapy dataset, suggesting shared topical structure. For matched clusters, we compared the subjects as well as therapist and robot responses using Transformer, Word2Vec, and BERT embeddings, revealing strong semantic overlap in subjects' disclosures in both datasets, as well as in the responses given to similar human disclosure themes across agent types (robot vs. human therapist). These findings highlight both the parallels and boundaries of robot-led support conversations and their potential for augmenting mental health interventions. 

---
# Probe before You Talk: Towards Black-box Defense against Backdoor Unalignment for Large Language Models 

**Authors**: Biao Yi, Tiansheng Huang, Sishuo Chen, Tong Li, Zheli Liu, Zhixuan Chu, Yiming Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16447)  

**Abstract**: Backdoor unalignment attacks against Large Language Models (LLMs) enable the stealthy compromise of safety alignment using a hidden trigger while evading normal safety auditing. These attacks pose significant threats to the applications of LLMs in the real-world Large Language Model as a Service (LLMaaS) setting, where the deployed model is a fully black-box system that can only interact through text. Furthermore, the sample-dependent nature of the attack target exacerbates the threat. Instead of outputting a fixed label, the backdoored LLM follows the semantics of any malicious command with the hidden trigger, significantly expanding the target space. In this paper, we introduce BEAT, a black-box defense that detects triggered samples during inference to deactivate the backdoor. It is motivated by an intriguing observation (dubbed the probe concatenate effect), where concatenated triggered samples significantly reduce the refusal rate of the backdoored LLM towards a malicious probe, while non-triggered samples have little effect. Specifically, BEAT identifies whether an input is triggered by measuring the degree of distortion in the output distribution of the probe before and after concatenation with the input. Our method addresses the challenges of sample-dependent targets from an opposite perspective. It captures the impact of the trigger on the refusal signal (which is sample-independent) instead of sample-specific successful attack behaviors. It overcomes black-box access limitations by using multiple sampling to approximate the output distribution. Extensive experiments are conducted on various backdoor attacks and LLMs (including the closed-source GPT-3.5-turbo), verifying the effectiveness and efficiency of our defense. Besides, we also preliminarily verify that BEAT can effectively defend against popular jailbreak attacks, as they can be regarded as 'natural backdoors'. 

---
# Unpacking Generative AI in Education: Computational Modeling of Teacher and Student Perspectives in Social Media Discourse 

**Authors**: Paulina DeVito, Akhil Vallala, Sean Mcmahon, Yaroslav Hinda, Benjamin Thaw, Hanqi Zhuang, Hari Kalva  

**Link**: [PDF](https://arxiv.org/pdf/2506.16412)  

**Abstract**: Generative AI (GAI) technologies are quickly reshaping the educational landscape. As adoption accelerates, understanding how students and educators perceive these tools is essential. This study presents one of the most comprehensive analyses to date of stakeholder discourse dynamics on GAI in education using social media data. Our dataset includes 1,199 Reddit posts and 13,959 corresponding top-level comments. We apply sentiment analysis, topic modeling, and author classification. To support this, we propose and validate a modular framework that leverages prompt-based large language models (LLMs) for analysis of online social discourse, and we evaluate this framework against classical natural language processing (NLP) models. Our GPT-4o pipeline consistently outperforms prior approaches across all tasks. For example, it achieved 90.6% accuracy in sentiment analysis against gold-standard human annotations. Topic extraction uncovered 12 latent topics in the public discourse with varying sentiment and author distributions. Teachers and students convey optimism about GAI's potential for personalized learning and productivity in higher education. However, key differences emerged: students often voice distress over false accusations of cheating by AI detectors, while teachers generally express concern about job security, academic integrity, and institutional pressures to adopt GAI tools. These contrasting perspectives highlight the tension between innovation and oversight in GAI-enabled learning environments. Our findings suggest a need for clearer institutional policies, more transparent GAI integration practices, and support mechanisms for both educators and students. More broadly, this study demonstrates the potential of LLM-based frameworks for modeling stakeholder discourse within online communities. 

---
# IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks 

**Authors**: Xiaoya Lu, Zeren Chen, Xuhao Hu, Yijin Zhou, Weichen Zhang, Dongrui Liu, Lu Sheng, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16402)  

**Abstract**: Flawed planning from VLM-driven embodied agents poses significant safety hazards, hindering their deployment in real-world household tasks. However, existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. Extensive experiments on leading VLMs, including the GPT-4o and Gemini-2.5 series, reveal that current agents lack interactive safety awareness, and that while safety-aware Chain-of-Thought can improve performance, it often compromises task completion. By highlighting these critical limitations, IS-Bench provides a foundation for developing safer and more reliable embodied AI systems. 

---
# GRPO-CARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning 

**Authors**: Yi Chen, Yuying Ge, Rui Wang, Yixiao Ge, Junhao Cheng, Ying Shan, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16141)  

**Abstract**: Recent reinforcement learning approaches, such as outcome-supervised GRPO, have advanced Chain-of-Thought reasoning in large language models (LLMs), yet their adaptation to multimodal LLMs (MLLMs) is unexplored. To address the lack of rigorous evaluation for MLLM post-training methods, we introduce SEED-Bench-R1, a benchmark with complex real-world videos requiring balanced perception and reasoning. It offers a large training set and evaluates generalization across three escalating challenges: in-distribution, cross-environment, and cross-environment-task scenarios. Using SEED-Bench-R1, we find that standard GRPO, while improving answer accuracy, often reduces logical coherence between reasoning steps and answers, with only a 57.9% consistency rate. This stems from reward signals focusing solely on final answers, encouraging shortcuts, and strict KL penalties limiting this http URL address this, we propose GRPO-CARE, a consistency-aware RL framework optimizing both answer correctness and reasoning coherence without explicit supervision. GRPO-CARE introduces a two-tiered reward: (1) a base reward for answer correctness, and (2) an adaptive consistency bonus, computed by comparing the model's reasoning-to-answer likelihood (via a slowly-evolving reference model) against group this http URL dual mechanism amplifies rewards for reasoning paths that are both correct and logically consistent. Replacing KL penalties with this adaptive bonus, GRPO-CARE outperforms standard GRPO on SEED-Bench-R1, achieving a 6.7% performance gain on the hardest evaluation level and a 24.5% improvement in consistency. It also shows strong transferability, improving model performance across diverse video understanding benchmarks. Our work contributes a systematically designed benchmark and a generalizable post-training framework, advancing the development of more interpretable and robust MLLMs. 

---
# Probing the Robustness of Large Language Models Safety to Latent Perturbations 

**Authors**: Tianle Gu, Kexin Huang, Zongqi Wang, Yixu Wang, Jie Li, Yuanqi Yao, Yang Yao, Yujiu Yang, Yan Teng, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16078)  

**Abstract**: Safety alignment is a key requirement for building reliable Artificial General Intelligence. Despite significant advances in safety alignment, we observe that minor latent shifts can still trigger unsafe responses in aligned models. We argue that this stems from the shallow nature of existing alignment methods, which focus on surface-level refusal behaviors without sufficiently altering internal representations. Consequently, small shifts in hidden activations can re-trigger harmful behaviors embedded in the latent space. To explore the robustness of safety alignment to latent perturbations, we introduce a probing method that measures the Negative Log-Likelihood of the original response generated by the model. This probe quantifies local sensitivity in the latent space, serving as a diagnostic tool for identifying vulnerable directions. Based on this signal, we construct effective jailbreak trajectories, giving rise to the Activation Steering Attack (ASA). More importantly, these insights offer a principled foundation for improving alignment robustness. To this end, we introduce Layer-wise Adversarial Patch Training~(LAPT), a fine-tuning strategy that inject controlled perturbations into hidden representations during training. Experimental results highlight that LAPT strengthen alignment robustness without compromising general capabilities. Our findings reveal fundamental flaws in current alignment paradigms and call for representation-level training strategies that move beyond surface-level behavior supervision. Codes and results are available at this https URL. 

---
# Bayesian Epistemology with Weighted Authority: A Formal Architecture for Truth-Promoting Autonomous Scientific Reasoning 

**Authors**: Craig S. Wright  

**Link**: [PDF](https://arxiv.org/pdf/2506.16015)  

**Abstract**: The exponential expansion of scientific literature has surpassed the epistemic processing capabilities of both human experts and current artificial intelligence systems. This paper introduces Bayesian Epistemology with Weighted Authority (BEWA), a formally structured architecture that operationalises belief as a dynamic, probabilistically coherent function over structured scientific claims. Each claim is contextualised, author-attributed, and evaluated through a system of replication scores, citation weighting, and temporal decay. Belief updates are performed via evidence-conditioned Bayesian inference, contradiction processing, and epistemic decay mechanisms. The architecture supports graph-based claim propagation, authorial credibility modelling, cryptographic anchoring, and zero-knowledge audit verification. By formalising scientific reasoning into a computationally verifiable epistemic network, BEWA advances the foundation for machine reasoning systems that promote truth utility, rational belief convergence, and audit-resilient integrity across dynamic scientific domains. 

---
# Multi-use LLM Watermarking and the False Detection Problem 

**Authors**: Zihao Fu, Chris Russell  

**Link**: [PDF](https://arxiv.org/pdf/2506.15975)  

**Abstract**: Digital watermarking is a promising solution for mitigating some of the risks arising from the misuse of automatically generated text. These approaches either embed non-specific watermarks to allow for the detection of any text generated by a particular sampler, or embed specific keys that allow the identification of the LLM user. However, simultaneously using the same embedding for both detection and user identification leads to a false detection problem, whereby, as user capacity grows, unwatermarked text is increasingly likely to be falsely detected as watermarked. Through theoretical analysis, we identify the underlying causes of this phenomenon. Building on these insights, we propose Dual Watermarking which jointly encodes detection and identification watermarks into generated text, significantly reducing false positives while maintaining high detection accuracy. Our experimental results validate our theoretical findings and demonstrate the effectiveness of our approach. 

---
# Exploring Big Five Personality and AI Capability Effects in LLM-Simulated Negotiation Dialogues 

**Authors**: Myke C. Cohen, Zhe Su, Hsien-Te Kao, Daniel Nguyen, Spencer Lynch, Maarten Sap, Svitlana Volkova  

**Link**: [PDF](https://arxiv.org/pdf/2506.15928)  

**Abstract**: This paper presents an evaluation framework for agentic AI systems in mission-critical negotiation contexts, addressing the need for AI agents that can adapt to diverse human operators and stakeholders. Using Sotopia as a simulation testbed, we present two experiments that systematically evaluated how personality traits and AI agent characteristics influence LLM-simulated social negotiation outcomes--a capability essential for a variety of applications involving cross-team coordination and civil-military interactions. Experiment 1 employs causal discovery methods to measure how personality traits impact price bargaining negotiations, through which we found that Agreeableness and Extraversion significantly affect believability, goal achievement, and knowledge acquisition outcomes. Sociocognitive lexical measures extracted from team communications detected fine-grained differences in agents' empathic communication, moral foundations, and opinion patterns, providing actionable insights for agentic AI systems that must operate reliably in high-stakes operational scenarios. Experiment 2 evaluates human-AI job negotiations by manipulating both simulated human personality and AI system characteristics, specifically transparency, competence, adaptability, demonstrating how AI agent trustworthiness impact mission effectiveness. These findings establish a repeatable evaluation methodology for experimenting with AI agent reliability across diverse operator personalities and human-agent team dynamics, directly supporting operational requirements for reliable AI systems. Our work advances the evaluation of agentic AI workflows by moving beyond standard performance metrics to incorporate social dynamics essential for mission success in complex operations. 

---
# Early Attentive Sparsification Accelerates Neural Speech Transcription 

**Authors**: Zifei Xu, Sayeh Sharify, Hesham Mostafa, Tristan Webb, Wanzin Yazar, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15912)  

**Abstract**: Transformer-based neural speech processing has achieved state-of-the-art performance. Since speech audio signals are known to be highly compressible, here we seek to accelerate neural speech transcription by time-domain signal sparsification early in the neural encoding stage, taking advantage of the interpretability of the self-attention mechanism in transformer audio encoders. With the Whisper family of models, we perform a systematic architecture search over the joint space of sparsification stage (a certain encoder layer) and compression ratio (sparsity). We found that the best resulting solutions under 1% accuracy degradation choose to sparsify the hidden state to 40-60% sparsity at an early encoding stage, and thereby achieve up to 1.6x runtime acceleration in English speech transcription tasks on Nvidia GPUs without any fine-tuning. 

---
# Fractional Reasoning via Latent Steering Vectors Improves Inference Time Compute 

**Authors**: Sheng Liu, Tianlang Chen, Pan Lu, Haotian Ye, Yizheng Chen, Lei Xing, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2506.15882)  

**Abstract**: Test-time compute has emerged as a powerful paradigm for improving the performance of large language models (LLMs), where generating multiple outputs or refining individual chains can significantly boost answer accuracy. However, existing methods like Best-of-N, majority voting, and self-reflection typically apply reasoning in a uniform way across inputs, overlooking the fact that different problems may require different levels of reasoning depth. In this work, we propose Fractional Reasoning, a training-free and model-agnostic framework that enables continuous control over reasoning intensity at inference time, going beyond the limitations of fixed instructional prompts. Our method operates by extracting the latent steering vector associated with deeper reasoning and reapplying it with a tunable scaling factor, allowing the model to tailor its reasoning process to the complexity of each input. This supports two key modes of test-time scaling: (1) improving output quality in breadth-based strategies (e.g., Best-of-N, majority voting), and (2) enhancing the correctness of individual reasoning chains in depth-based strategies (e.g., self-reflection). Experiments on GSM8K, MATH500, and GPQA demonstrate that Fractional Reasoning consistently improves performance across diverse reasoning tasks and models. 

---
# MoR: Better Handling Diverse Queries with a Mixture of Sparse, Dense, and Human Retrievers 

**Authors**: Jushaan Singh Kalra, Xinran Zhao, To Eun Kim, Fengyu Cai, Fernando Diaz, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15862)  

**Abstract**: Retrieval-augmented Generation (RAG) is powerful, but its effectiveness hinges on which retrievers we use and how. Different retrievers offer distinct, often complementary signals: BM25 captures lexical matches; dense retrievers, semantic similarity. Yet in practice, we typically fix a single retriever based on heuristics, which fails to generalize across diverse information needs. Can we dynamically select and integrate multiple retrievers for each individual query, without the need for manual selection? In our work, we validate this intuition with quantitative analysis and introduce mixture of retrievers: a zero-shot, weighted combination of heterogeneous retrievers. Extensive experiments show that such mixtures are effective and efficient: Despite totaling just 0.8B parameters, this mixture outperforms every individual retriever and even larger 7B models by +10.8% and +3.9% on average, respectively. Further analysis also shows that this mixture framework can help incorporate specialized non-oracle human information sources as retrievers to achieve good collaboration, with a 58.9% relative performance improvement over simulated humans alone. 

---
# SLR: An Automated Synthesis Framework for Scalable Logical Reasoning 

**Authors**: Lukas Helff, Ahmad Omar, Felix Friedrich, Wolfgang Stammer, Antonia Wüst, Tim Woydt, Rupert Mitchell, Patrick Schramowski, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2506.15787)  

**Abstract**: We introduce SLR, an end-to-end framework for systematic evaluation and training of Large Language Models (LLMs) via Scalable Logical Reasoning. Given a user's task specification, SLR enables scalable, automated synthesis of inductive reasoning tasks with precisely controlled difficulty. For each task, SLR synthesizes (i) a latent ground-truth rule, (ii) an executable validation program used by a symbolic judge to deterministically verify model outputs, and (iii) an instruction prompt for the reasoning task. Using SLR, we create SLR-Bench, a benchmark comprising over 19k prompts spanning 20 curriculum levels that progressively increase in relational, arithmetic, and recursive complexity. Large-scale evaluation reveals that contemporary LLMs readily produce syntactically valid rules, yet often fail at correct logical inference. Recent reasoning LLMs do somewhat better, but incur substantial increases in test-time compute, sometimes exceeding 15k completion tokens. Finally, logic-tuning via SLR doubles Llama-3-8B accuracy on SLR-Bench, achieving parity with Gemini-Flash-Thinking at a fraction of computational cost. SLR is fully automated, requires no human annotation, ensures dataset novelty, and offers a scalable environment for probing and advancing LLMs' reasoning capabilities. 

---
# InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding 

**Authors**: Minsoo Kim, Kyuhong Shim, Jungwook Choi, Simyung Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15745)  

**Abstract**: Modern multimodal large language models (MLLMs) can reason over hour-long video, yet their key-value (KV) cache grows linearly with time--quickly exceeding the fixed memory of phones, AR glasses, and edge robots. Prior compression schemes either assume the whole video and user query are available offline or must first build the full cache, so memory still scales with stream length. InfiniPot-V is the first training-free, query-agnostic framework that enforces a hard, length-independent memory cap for streaming video understanding. During video encoding it monitors the cache and, once a user-set threshold is reached, runs a lightweight compression pass that (i) removes temporally redundant tokens via Temporal-axis Redundancy (TaR) metric and (ii) keeps semantically significant tokens via Value-Norm (VaN) ranking. Across four open-source MLLMs and four long-video and two streaming-video benchmarks, InfiniPot-V cuts peak GPU memory by up to 94%, sustains real-time generation, and matches or surpasses full-cache accuracy--even in multi-turn dialogues. By dissolving the KV cache bottleneck without retraining or query knowledge, InfiniPot-V closes the gap for on-device streaming video assistants. 

---
# OAgents: An Empirical Study of Building Effective Agents 

**Authors**: He Zhu, Tianrui Qin, King Zhu, Heyuan Huang, Yeyi Guan, Jinxiang Xia, Yi Yao, Hanhao Li, Ningning Wang, Pai Liu, Tianhao Peng, Xin Gui, Xiaowan Li, Yuhui Liu, Yuchen Eleanor Jiang, Jun Wang, Changwang Zhang, Xiangru Tang, Ge Zhang, Jian Yang, Minghao Liu, Xitong Gao, Wangchunshu Zhou, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15741)  

**Abstract**: Recently, Agentic AI has become an increasingly popular research field. However, we argue that current agent research practices lack standardization and scientific rigor, making it hard to conduct fair comparisons among methods. As a result, it is still unclear how different design choices in agent frameworks affect effectiveness, and measuring their progress remains challenging. In this work, we conduct a systematic empirical study on GAIA benchmark and BrowseComp to examine the impact of popular design choices in key agent components in a fair and rigorous manner. We find that the lack of a standard evaluation protocol makes previous works, even open-sourced ones, non-reproducible, with significant variance between random runs. Therefore, we introduce a more robust evaluation protocol to stabilize comparisons. Our study reveals which components and designs are crucial for effective agents, while others are redundant, despite seeming logical. Based on our findings, we build and open-source OAgents, a new foundation agent framework that achieves state-of-the-art performance among open-source projects. OAgents offers a modular design for various agent components, promoting future research in Agentic AI. 

---
# MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference 

**Authors**: Kunxi Li, Zhonghua Jiang, Zhouzhou Shen, Zhaode Wang, Chengfei Lv, Shengyu Zhang, Fan Wu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15724)  

**Abstract**: This paper introduces MadaKV, a modality-adaptive key-value (KV) cache eviction strategy designed to enhance the efficiency of multimodal large language models (MLLMs) in long-context inference. In multimodal scenarios, attention heads exhibit varying preferences for different modalities, resulting in significant disparities in modality importance across attention heads. Traditional KV cache eviction methods, which are tailored for unimodal settings, fail to capture modality-specific information, thereby yielding suboptimal performance. MadaKV addresses these challenges through two key components: modality preference adaptation and hierarchical compression compensation. By dynamically sensing modality information within attention heads and adaptively retaining critical tokens, MadaKV achieves substantial reductions in KV cache memory footprint and model inference decoding latency (1.3 to 1.5 times improvement) while maintaining high accuracy across various multimodal long-context tasks. Extensive experiments on representative MLLMs and the MileBench benchmark demonstrate the effectiveness of MadaKV compared to existing KV cache eviction methods. 

---
# daDPO: Distribution-Aware DPO for Distilling Conversational Abilities 

**Authors**: Zhengze Zhang, Shiqi Wang, Yiqun Shen, Simin Guo, Dahua Lin, Xiaoliang Wang, Nguyen Cam-Tu, Fei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15717)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional performance across various applications, but their conversational abilities decline sharply as model size decreases, presenting a barrier to their deployment in resource-constrained environments. Knowledge distillation with Direct Preference Optimization (dDPO) has emerged as a promising approach to enhancing the conversational abilities of smaller models using a larger teacher model. However, current methods primarily focus on 'black-box' KD, which only uses the teacher's responses, overlooking the output distribution offered by the teacher. This paper addresses this gap by introducing daDPO (Distribution-Aware DPO), a unified method for preference optimization and distribution-based distillation. We provide rigorous theoretical analysis and empirical validation, showing that daDPO outperforms existing methods in restoring performance for pruned models and enhancing smaller LLM models. Notably, in in-domain evaluation, our method enables a 20% pruned Vicuna1.5-7B to achieve near-teacher performance (-7.3% preference rate compared to that of dDPO's -31%), and allows Qwen2.5-1.5B to occasionally outperform its 7B teacher model (14.0% win rate). 

---
# Adaptive Two Sided Laplace Transforms: A Learnable, Interpretable, and Scalable Replacement for Self-Attention 

**Authors**: Andrew Kiruluta  

**Link**: [PDF](https://arxiv.org/pdf/2506.15714)  

**Abstract**: We propose an innovative, learnable two-sided short-time Laplace transform (STLT) mechanism to supplant the traditional self attention in transformer-based LLMs. Our STLT introduces trainable parameters for each Laplace node, enabling end-to-end learning of decay rates , oscillatory frequencies, and window bandwidth T. This flexibility allows the model to dynamically adapt token relevance half lives and frequency responses during training. By selecting S learnable nodes and leveraging fast recursive convolution, we achieve an effective complexity of in time and memory. We further incorporate an efficient FFT-based computation of the relevance matrix and an adaptive node allocation mechanism to dynamically adjust the number of active Laplace nodes. Empirical results on language modeling (WikiText\-103, Project Gutenberg), machine translation (WMT'14 En\-De), and long document question answering (NarrativeQA) demonstrate that our learnable STLT achieves perplexities and scores on par with or better than existing efficient transformers while naturally extending to context lengths exceeding 100k tokens or more limited only by available hardware. Ablation studies confirm the importance of learnable parameters and adaptive node allocation. The proposed approach combines interpretability, through explicit decay and frequency parameters, with scalability and robustness, offering a pathway towards ultra-long-sequence language modeling without the computational bottleneck of self-attention. 

---
# Learn from the Past: Fast Sparse Indexing for Large Language Model Decoding 

**Authors**: Feiyu Yao, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15704)  

**Abstract**: As large language models (LLMs) continue to support increasingly longer contexts, the memory demand for key-value (KV) caches during decoding grows rapidly, becoming a critical bottleneck in both GPU memory capacity and PCIe bandwidth. Sparse attention mechanisms alleviate this issue by computing attention weights only for selected key-value pairs. However, their indexing computation typically requires traversing all key vectors, resulting in significant computational and data transfer overhead. To reduce the cost of index retrieval, existing methods often treat each decoding step as an independent process, failing to exploit the temporal correlations embedded in historical decoding information. To this end, we propose LFPS(Learn From the Past for Sparse Indexing), an acceleration method that dynamically constructs sparse indexing candidates based on historical attention patterns. LFPS captures two prevalent trends in decoder attention -vertical patterns (attending to fixed positions) and slash patterns (attending to relative positions) -and incorporates a positional expansion strategy to effectively predict the Top-k indices for the current step. We validate LFPS on challenging long-context benchmarks such as LongBench-RULER, using Llama-3.1-8B-Instruct as the base model. Experimental results show that LFPS achieves up to 22.8$\times$ speedup over full attention and 9.6$\times$ speedup over exact Top-k retrieval on an RTX 4090 GPU and a single CPU core of a Xeon Gold 6430, respectively, while preserving generation accuracy. These results demonstrate that LFPS offers a practical and efficient solution for decoding optimization in long-context LLM inference. 

---
# DeepRTL2: A Versatile Model for RTL-Related Tasks 

**Authors**: Yi Liu, Hongji Zhang, Yunhao Zhou, Zhengyuan Shi, Changran Xu, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15697)  

**Abstract**: The integration of large language models (LLMs) into electronic design automation (EDA) has significantly advanced the field, offering transformative benefits, particularly in register transfer level (RTL) code generation and understanding. While previous studies have demonstrated the efficacy of fine-tuning LLMs for these generation-based tasks, embedding-based tasks, which are equally critical to EDA workflows, have been largely overlooked. These tasks, including natural language code search, RTL code functionality equivalence checking, and performance prediction, are essential for accelerating and optimizing the hardware design process. To address this gap, we present DeepRTL2, a family of versatile LLMs that unifies both generation- and embedding-based tasks related to RTL. By simultaneously tackling a broad range of tasks, DeepRTL2 represents the first model to provide a comprehensive solution to the diverse challenges in EDA. Through extensive experiments, we show that DeepRTL2 achieves state-of-the-art performance across all evaluated tasks. 

---
# BASE-Q: Bias and Asymmetric Scaling Enhanced Rotational Quantization for Large Language Models 

**Authors**: Liulu He, Shenli Zhen, Karwei Sun, Yijiang Liu, Yufei Zhao, Chongkang Tan, Huanrui Yang, Yuan Du, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.15689)  

**Abstract**: Rotations have become essential to state-of-the-art quantization pipelines for large language models (LLMs) by effectively smoothing outliers in weights and activations. However, further optimizing the rotation parameters offers only limited performance gains and introduces significant training overhead: due to rotation parameter sharing, full-model must be loaded simultaneously to enable backpropagation, resulting in substantial memory consumption and limited practical utility. In this work, we identify two fundamental limitations of current rotational quantization methods: (i) rotation fails to align channel means, resulting in wider quantization bounds and increased rounding errors; and (ii) rotation makes the activation distribution more Gaussian-like, increasing energy loss caused by clipping errors. To address these issues, we introduce \textbf{BASE-Q}, a simple yet powerful approach that combines bias correction and asymmetric scaling to effectively reduce rounding and clipping errors. Furthermore, BASE-Q enables blockwise optimization, eliminating the need for memory-intensive full-model backpropagation. Extensive experiments on various LLMs and benchmarks demonstrate the effectiveness of BASE-Q, narrowing the accuracy gap to full-precision models by 50.5\%, 42.9\%, and 29.2\% compared to QuaRot, SpinQuant, and OSTQuant, respectively. The code will be released soon. 

---
# cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree 

**Authors**: Yilin Zhang, Xinran Zhao, Zora Zhiruo Wang, Chenyang Yang, Jiayi Wei, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15655)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become essential for large-scale code generation, grounding predictions in external code corpora to improve actuality. However, a critical yet underexplored aspect of RAG pipelines is chunking -- the process of dividing documents into retrievable units. Existing line-based chunking heuristics often break semantic structures, splitting functions or merging unrelated code, which can degrade generation quality. We propose chunking via Abstract Syntax Trees (\ourwork), a structure-aware method that recursively breaks large AST nodes into smaller chunks and merges sibling nodes while respecting size limits. This approach generates self-contained, semantically coherent units across programming languages and tasks, improving performance on diverse code generation tasks, e.g., boosting Recall@5 by 4.3 points on RepoEval retrieval and Pass@1 by 2.67 points on SWE-bench generation. Our work highlights the importance of structure-aware chunking for scaling retrieval-enhanced code intelligence. 

---
