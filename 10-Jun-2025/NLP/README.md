# Reinforcement Pre-Training 

**Authors**: Qingxiu Dong, Li Dong, Yao Tang, Tianzhu Ye, Yutao Sun, Zhifang Sui, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.08007)  

**Abstract**: In this work, we introduce Reinforcement Pre-Training (RPT) as a new scaling paradigm for large language models and reinforcement learning (RL). Specifically, we reframe next-token prediction as a reasoning task trained using RL, where it receives verifiable rewards for correctly predicting the next token for a given context. RPT offers a scalable method to leverage vast amounts of text data for general-purpose RL, rather than relying on domain-specific annotated answers. By incentivizing the capability of next-token reasoning, RPT significantly improves the language modeling accuracy of predicting the next tokens. Moreover, RPT provides a strong pre-trained foundation for further reinforcement fine-tuning. The scaling curves show that increased training compute consistently improves the next-token prediction accuracy. The results position RPT as an effective and promising scaling paradigm to advance language model pre-training. 

---
# Correlated Errors in Large Language Models 

**Authors**: Elliot Kim, Avi Garg, Kenny Peng, Nikhil Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.07962)  

**Abstract**: Diversity in training data, architecture, and providers is assumed to mitigate homogeneity in LLMs. However, we lack empirical evidence on whether different LLMs differ meaningfully. We conduct a large-scale empirical evaluation on over 350 LLMs overall, using two popular leaderboards and a resume-screening task. We find substantial correlation in model errors -- on one leaderboard dataset, models agree 60% of the time when both models err. We identify factors driving model correlation, including shared architectures and providers. Crucially, however, larger and more accurate models have highly correlated errors, even with distinct architectures and providers. Finally, we show the effects of correlation in two downstream tasks: LLM-as-judge evaluation and hiring -- the latter reflecting theoretical predictions regarding algorithmic monoculture. 

---
# Language Models over Canonical Byte-Pair Encodings 

**Authors**: Tim Vieira, Tianyu Liu, Clemente Pasti, Yahya Emara, Brian DuSell, Benjamin LeBrun, Mario Giulianelli, Juan Luis Gastaldi, Timothy J. O'Donnell, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2506.07956)  

**Abstract**: Modern language models represent probability distributions over character strings as distributions over (shorter) token strings derived via a deterministic tokenizer, such as byte-pair encoding. While this approach is highly effective at scaling up language models to large corpora, its current incarnations have a concerning property: the model assigns nonzero probability mass to an exponential number of $\it{noncanonical}$ token encodings of each character string -- these are token strings that decode to valid character strings but are impossible under the deterministic tokenizer (i.e., they will never be seen in any training corpus, no matter how large). This misallocation is both erroneous, as noncanonical strings never appear in training data, and wasteful, diverting probability mass away from plausible outputs. These are avoidable mistakes! In this work, we propose methods to enforce canonicality in token-level language models, ensuring that only canonical token strings are assigned positive probability. We present two approaches: (1) canonicality by conditioning, leveraging test-time inference strategies without additional training, and (2) canonicality by construction, a model parameterization that guarantees canonical outputs but requires training. We demonstrate that fixing canonicality mistakes improves the likelihood of held-out data for several models and corpora. 

---
# Statistical Hypothesis Testing for Auditing Robustness in Language Models 

**Authors**: Paulius Rauba, Qiyao Wei, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.07947)  

**Abstract**: Consider the problem of testing whether the outputs of a large language model (LLM) system change under an arbitrary intervention, such as an input perturbation or changing the model variant. We cannot simply compare two LLM outputs since they might differ due to the stochastic nature of the system, nor can we compare the entire output distribution due to computational intractability. While existing methods for analyzing text-based outputs exist, they focus on fundamentally different problems, such as measuring bias or fairness. To this end, we introduce distribution-based perturbation analysis, a framework that reformulates LLM perturbation analysis as a frequentist hypothesis testing problem. We construct empirical null and alternative output distributions within a low-dimensional semantic similarity space via Monte Carlo sampling, enabling tractable inference without restrictive distributional assumptions. The framework is (i) model-agnostic, (ii) supports the evaluation of arbitrary input perturbations on any black-box LLM, (iii) yields interpretable p-values; (iv) supports multiple perturbations via controlled error rates; and (v) provides scalar effect sizes. We demonstrate the usefulness of the framework across multiple case studies, showing how we can quantify response changes, measure true/false positive rates, and evaluate alignment with reference models. Above all, we see this as a reliable frequentist hypothesis testing framework for LLM auditing. 

---
# Quantum Graph Transformer for NLP Sentiment Classification 

**Authors**: Shamminuj Aktar, Andreas Bärtschi, Abdel-Hameed A. Badawy, Stephan Eidenbenz  

**Link**: [PDF](https://arxiv.org/pdf/2506.07937)  

**Abstract**: Quantum machine learning is a promising direction for building more efficient and expressive models, particularly in domains where understanding complex, structured data is critical. We present the Quantum Graph Transformer (QGT), a hybrid graph-based architecture that integrates a quantum self-attention mechanism into the message-passing framework for structured language modeling. The attention mechanism is implemented using parameterized quantum circuits (PQCs), which enable the model to capture rich contextual relationships while significantly reducing the number of trainable parameters compared to classical attention mechanisms. We evaluate QGT on five sentiment classification benchmarks. Experimental results show that QGT consistently achieves higher or comparable accuracy than existing quantum natural language processing (QNLP) models, including both attention-based and non-attention-based approaches. When compared with an equivalent classical graph transformer, QGT yields an average accuracy improvement of 5.42% on real-world datasets and 4.76% on synthetic datasets. Additionally, QGT demonstrates improved sample efficiency, requiring nearly 50% fewer labeled samples to reach comparable performance on the Yelp dataset. These results highlight the potential of graph-based QNLP techniques for advancing efficient and scalable language understanding. 

---
# MiniCPM4: Ultra-Efficient LLMs on End Devices 

**Authors**: MiniCPM Team, Chaojun Xiao, Yuxuan Li, Xu Han, Yuzhuo Bai, Jie Cai, Haotian Chen, Wentong Chen, Xin Cong, Ganqu Cui, Ning Ding, Shengdan Fan, Yewei Fang, Zixuan Fu, Wenyu Guan, Yitong Guan, Junshao Guo, Yufeng Han, Bingxiang He, Yuxiang Huang, Cunliang Kong, Qiuzuo Li, Siyuan Li, Wenhao Li, Yanghao Li, Yishan Li, Zhen Li, Dan Liu, Biyuan Lin, Yankai Lin, Xiang Long, Quanyu Lu, Yaxi Lu, Peiyan Luo, Hongya Lyu, Litu Ou, Yinxu Pan, Zekai Qu, Qundong Shi, Zijun Song, Jiayuan Su, Zhou Su, Ao Sun, Xianghui Sun, Peijun Tang, Fangzheng Wang, Feng Wang, Shuo Wang, Yudong Wang, Yesai Wu, Zhenyu Xiao, Jie Xie, Zihao Xie, Yukun Yan, Jiarui Yuan, Kaihuo Zhang, Lei Zhang, Linyue Zhang, Xueren Zhang, Yudi Zhang, Hengyu Zhao, Weilin Zhao, Weilun Zhao, Yuanqian Zhao, Zhi Zheng, Ge Zhou, Jie Zhou, Wei Zhou, Zihan Zhou, Zixuan Zhou, Zhiyuan Liu, Guoyang Zeng, Chao Jia, Dahai Li, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.07900)  

**Abstract**: This paper introduces MiniCPM4, a highly efficient large language model (LLM) designed explicitly for end-side devices. We achieve this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems. Specifically, in terms of model architecture, we propose InfLLM v2, a trainable sparse attention mechanism that accelerates both prefilling and decoding phases for long-context processing. Regarding training data, we propose UltraClean, an efficient and accurate pre-training data filtering and generation strategy, and UltraChat v2, a comprehensive supervised fine-tuning dataset. These datasets enable satisfactory model performance to be achieved using just 8 trillion training tokens. Regarding training algorithms, we propose ModelTunnel v2 for efficient pre-training strategy search, and improve existing post-training methods by introducing chunk-wise rollout for load-balanced reinforcement learning and data-efficient tenary LLM, BitCPM. Regarding inference systems, we propose this http URL that integrates sparse attention, model quantization, and speculative sampling to achieve efficient prefilling and decoding. To meet diverse on-device requirements, MiniCPM4 is available in two versions, with 0.5B and 8B parameters, respectively. Sufficient evaluation results show that MiniCPM4 outperforms open-source models of similar size across multiple benchmarks, highlighting both its efficiency and effectiveness. Notably, MiniCPM4-8B demonstrates significant speed improvements over Qwen3-8B when processing long sequences. Through further adaptation, MiniCPM4 successfully powers diverse applications, including trustworthy survey generation and tool use with model context protocol, clearly showcasing its broad usability. 

---
# MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs 

**Authors**: Ke Wang, Yiming Qin, Nikolaos Dimitriadis, Alessandro Favero, Pascal Frossard  

**Link**: [PDF](https://arxiv.org/pdf/2506.07899)  

**Abstract**: Language models deployed in real-world systems often require post-hoc updates to incorporate new or corrected knowledge. However, editing such models efficiently and reliably - without retraining or forgetting previous information - remains a major challenge. Existing methods for lifelong model editing either compromise generalization, interfere with past edits, or fail to scale to long editing sequences. We propose MEMOIR, a novel scalable framework that injects knowledge through a residual memory, i.e., a dedicated parameter module, while preserving the core capabilities of the pre-trained model. By sparsifying input activations through sample-dependent masks, MEMOIR confines each edit to a distinct subset of the memory parameters, minimizing interference among edits. At inference, it identifies relevant edits by comparing the sparse activation patterns of new queries to those stored during editing. This enables generalization to rephrased queries by activating only the relevant knowledge while suppressing unnecessary memory activation for unrelated prompts. Experiments on question answering, hallucination correction, and out-of-distribution generalization benchmarks across LLaMA-3 and Mistral demonstrate that MEMOIR achieves state-of-the-art performance across reliability, generalization, and locality metrics, scaling to thousands of sequential edits with minimal forgetting. 

---
# Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning 

**Authors**: Yiju Guo, Wenkai Yang, Zexu Sun, Ning Ding, Zhiyuan Liu, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.07851)  

**Abstract**: Large language models (LLMs) have demonstrated significant improvements in contextual understanding. However, their ability to attend to truly critical information during long-context reasoning and generation still falls behind the pace. Specifically, our preliminary experiments reveal that certain distracting patterns can misdirect the model's attention during inference, and removing these patterns substantially improves reasoning accuracy and generation quality. We attribute this phenomenon to spurious correlations in the training data, which obstruct the model's capacity to infer authentic causal instruction-response relationships. This phenomenon may induce redundant reasoning processes, potentially resulting in significant inference overhead and, more critically, the generation of erroneous or suboptimal responses. To mitigate this, we introduce a two-stage framework called Learning to Focus (LeaF) leveraging intervention-based inference to disentangle confounding factors. In the first stage, LeaF employs gradient-based comparisons with an advanced teacher to automatically identify confounding tokens based on causal relationships in the training corpus. Then, in the second stage, it prunes these tokens during distillation to enact intervention, aligning the student's attention with the teacher's focus distribution on truly critical context tokens. Experimental results demonstrate that LeaF not only achieves an absolute improvement in various mathematical reasoning and code generation benchmarks but also effectively suppresses attention to confounding tokens during inference, yielding a more interpretable and reliable reasoning model. 

---
# WebUIBench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models in WebUI-to-Code 

**Authors**: Zhiyu Lin, Zhengda Zhou, Zhiyuan Zhao, Tianrui Wan, Yilun Ma, Junyu Gao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.07818)  

**Abstract**: With the rapid advancement of Generative AI technology, Multimodal Large Language Models(MLLMs) have the potential to act as AI software engineers capable of executing complex web application development. Considering that the model requires a confluence of multidimensional sub-capabilities to address the challenges of various development phases, constructing a multi-view evaluation framework is crucial for accurately guiding the enhancement of development efficiency. However, existing benchmarks usually fail to provide an assessment of sub-capabilities and focus solely on webpage generation outcomes. In this work, we draw inspiration from the principles of software engineering and further propose WebUIBench, a benchmark systematically designed to evaluate MLLMs in four key areas: WebUI Perception, HTML Programming,WebUI-HTML Understanding, and WebUI-to-Code. WebUIBench comprises 21K high-quality question-answer pairs derived from over 0.7K real-world websites. The extensive evaluation of 29 mainstream MLLMs uncovers the skill characteristics and various weakness that models encountered during the development process. 

---
# MultiMatch: Multihead Consistency Regularization Matching for Semi-Supervised Text Classification 

**Authors**: Iustin Sirbu, Robert-Adrian Popovici, Cornelia Caragea, Stefan Trausan-Matu, Traian Rebedea  

**Link**: [PDF](https://arxiv.org/pdf/2506.07801)  

**Abstract**: We introduce MultiMatch, a novel semi-supervised learning (SSL) algorithm combining the paradigms of co-training and consistency regularization with pseudo-labeling. At its core, MultiMatch features a three-fold pseudo-label weighting module designed for three key purposes: selecting and filtering pseudo-labels based on head agreement and model confidence, and weighting them according to the perceived classification difficulty. This novel module enhances and unifies three existing techniques -- heads agreement from Multihead Co-training, self-adaptive thresholds from FreeMatch, and Average Pseudo-Margins from MarginMatch -- resulting in a holistic approach that improves robustness and performance in SSL settings. Experimental results on benchmark datasets highlight the superior performance of MultiMatch, achieving state-of-the-art results on 9 out of 10 setups from 5 natural language processing datasets and ranking first according to the Friedman test among 19 methods. Furthermore, MultiMatch demonstrates exceptional robustness in highly imbalanced settings, outperforming the second-best approach by 3.26% -- and data imbalance is a key factor for many text classification tasks. 

---
# LLM Unlearning Should Be Form-Independent 

**Authors**: Xiaotian Ye, Mengqi Zhang, Shu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07795)  

**Abstract**: Large Language Model (LLM) unlearning aims to erase or suppress undesirable knowledge within the model, offering promise for controlling harmful or private information to prevent misuse. However, recent studies highlight its limited efficacy in real-world scenarios, hindering practical adoption. In this study, we identify a pervasive issue underlying many downstream failures: the effectiveness of existing unlearning methods heavily depends on the form of training samples and frequently fails to generalize to alternate expressions of the same knowledge. We formally characterize this problem as Form-Dependent Bias and systematically investigate its specific manifestation patterns across various downstream tasks. To quantify its prevalence and support future research, we introduce ORT, a novel benchmark designed to evaluate the robustness of unlearning methods against variations in knowledge expression. Results reveal that Form-Dependent Bias is both widespread and severe among current techniques.
We argue that LLM unlearning should be form-independent to address the endless forms of downstream tasks encountered in real-world security-critical scenarios. Towards this goal, we introduce Rank-one Concept Redirection (ROCR), a novel training-free method, as a promising solution path. ROCR performs unlearning by targeting the invariants in downstream tasks, specifically the activated dangerous concepts. It is capable of modifying model parameters within seconds to redirect the model's perception of a specific unlearning target concept to another harmless concept. Extensive experiments demonstrate that ROCR significantly improves unlearning effectiveness compared to traditional methods while generating highly natural outputs. 

---
# Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking 

**Authors**: Silin Gao, Antoine Bosselut, Samy Bengio, Emmanuel Abbe  

**Link**: [PDF](https://arxiv.org/pdf/2506.07751)  

**Abstract**: Recent studies have shown that large language models (LLMs), especially smaller ones, often lack robustness in their reasoning. I.e., they tend to experience performance drops when faced with distribution shifts, such as changes to numerical or nominal variables, or insertions of distracting clauses. A possible strategy to address this involves generating synthetic data to further "instantiate" reasoning problems on potential variations. In contrast, our approach focuses on "abstracting" reasoning problems. This not only helps counteract distribution shifts but also facilitates the connection to symbolic tools for deriving solutions. We find that this abstraction process is better acquired through reinforcement learning (RL) than just supervised fine-tuning, which often fails to produce faithful abstractions. Our method, AbstraL -- which promotes abstract reasoning in LLMs using RL on granular abstraction data -- significantly mitigates performance degradation on recent GSM perturbation benchmarks. 

---
# Swiss Parliaments Corpus Re-Imagined (SPC_R): Enhanced Transcription with RAG-based Correction and Predicted BLEU 

**Authors**: Vincenzo Timmel, Manfred Vogel, Daniel Perruchoud, Reza Kakooee  

**Link**: [PDF](https://arxiv.org/pdf/2506.07726)  

**Abstract**: This paper presents a new long-form release of the Swiss Parliaments Corpus, converting entire multi-hour Swiss German debate sessions (each aligned with the official session protocols) into high-quality speech-text pairs. Our pipeline starts by transcribing all session audio into Standard German using Whisper Large-v3 under high-compute settings. We then apply a two-step GPT-4o correction process: first, GPT-4o ingests the raw Whisper output alongside the official protocols to refine misrecognitions, mainly named entities. Second, a separate GPT-4o pass evaluates each refined segment for semantic completeness. We filter out any segments whose Predicted BLEU score (derived from Whisper's average token log-probability) and GPT-4o evaluation score fall below a certain threshold. The final corpus contains 801 hours of audio, of which 751 hours pass our quality control. Compared to the original sentence-level SPC release, our long-form dataset achieves a 6-point BLEU improvement, demonstrating the power of combining robust ASR, LLM-based correction, and data-driven filtering for low-resource, domain-specific speech corpora. 

---
# Multilingual Grammatical Error Annotation: Combining Language-Agnostic Framework with Language-Specific Flexibility 

**Authors**: Mengyang Qiu, Tran Minh Nguyen, Zihao Huang, Zelong Li, Yang Gu, Qingyu Gao, Siliang Liu, Jungyeul Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.07719)  

**Abstract**: Grammatical Error Correction (GEC) relies on accurate error annotation and evaluation, yet existing frameworks, such as $\texttt{errant}$, face limitations when extended to typologically diverse languages. In this paper, we introduce a standardized, modular framework for multilingual grammatical error annotation. Our approach combines a language-agnostic foundation with structured language-specific extensions, enabling both consistency and flexibility across languages. We reimplement $\texttt{errant}$ using $\texttt{stanza}$ to support broader multilingual coverage, and demonstrate the framework's adaptability through applications to English, German, Czech, Korean, and Chinese, ranging from general-purpose annotation to more customized linguistic refinements. This work supports scalable and interpretable GEC annotation across languages and promotes more consistent evaluation in multilingual settings. The complete codebase and annotation tools can be accessed at this https URL. 

---
# Through the Valley: Path to Effective Long CoT Training for Small Language Models 

**Authors**: Renjie Luo, Jiaxi Li, Chen Huang, Wei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07712)  

**Abstract**: Long chain-of-thought (CoT) supervision has become a common strategy to enhance reasoning in language models. While effective for large models, we identify a phenomenon we call Long CoT Degradation, in which small language models (SLMs; <=3B parameters) trained on limited long CoT data experience significant performance deterioration. Through extensive experiments on the Qwen2.5, LLaMA3 and Gemma3 families, we demonstrate that this degradation is widespread across SLMs. In some settings, models trained on only 8k long CoT examples lose up to 75% of their original performance before fine-tuning. Strikingly, we further observe that for some particularly small models, even training on 220k long CoT examples fails to recover or surpass their original performance prior to fine-tuning. Our analysis attributes this effect to error accumulation: while longer responses increase the capacity for multi-step reasoning, they also amplify the risk of compounding mistakes. Furthermore, we find that Long CoT Degradation may negatively impacts downstream reinforcement learning (RL), although this can be alleviated by sufficiently scaled supervised fine-tuning (SFT). Our findings challenge common assumptions about the benefits of long CoT training for SLMs and offer practical guidance for building more effective small-scale reasoning models. 

---
# Training Superior Sparse Autoencoders for Instruct Models 

**Authors**: Jiaming Li, Haoran Ye, Yukun Chen, Xinyue Li, Lei Zhang, Hamid Alinejad-Rokny, Jimmy Chih-Hsien Peng, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07691)  

**Abstract**: As large language models (LLMs) grow in scale and capability, understanding their internal mechanisms becomes increasingly critical. Sparse autoencoders (SAEs) have emerged as a key tool in mechanistic interpretability, enabling the extraction of human-interpretable features from LLMs. However, existing SAE training methods are primarily designed for base models, resulting in reduced reconstruction quality and interpretability when applied to instruct models. To bridge this gap, we propose $\underline{\textbf{F}}$inetuning-$\underline{\textbf{a}}$ligned $\underline{\textbf{S}}$equential $\underline{\textbf{T}}$raining ($\textit{FAST}$), a novel training method specifically tailored for instruct models. $\textit{FAST}$ aligns the training process with the data distribution and activation patterns characteristic of instruct models, resulting in substantial improvements in both reconstruction and feature interpretability. On Qwen2.5-7B-Instruct, $\textit{FAST}$ achieves a mean squared error of 0.6468 in token reconstruction, significantly outperforming baseline methods with errors of 5.1985 and 1.5096. In feature interpretability, $\textit{FAST}$ yields a higher proportion of high-quality features, for Llama3.2-3B-Instruct, $21.1\%$ scored in the top range, compared to $7.0\%$ and $10.2\%$ for $\textit{BT(P)}$ and $\textit{BT(F)}$. Surprisingly, we discover that intervening on the activations of special tokens via the SAEs leads to improvements in output quality, suggesting new opportunities for fine-grained control of model behavior. Code, data, and 240 trained SAEs are available at this https URL. 

---
# GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation 

**Authors**: Ionut-Teodor Sorodoc, Leonardo F. R. Ribeiro, Rexhina Blloshmi, Christopher Davis, Adrià de Gispert  

**Link**: [PDF](https://arxiv.org/pdf/2506.07671)  

**Abstract**: We present GaRAGe, a large RAG benchmark with human-curated long-form answers and annotations of each grounding passage, allowing a fine-grained evaluation of whether LLMs can identify relevant grounding when generating RAG answers. Our benchmark contains 2366 questions of diverse complexity, dynamism, and topics, and includes over 35K annotated passages retrieved from both private document sets and the Web, to reflect real-world RAG use cases. This makes it an ideal test bed to evaluate an LLM's ability to identify only the relevant information necessary to compose a response, or provide a deflective response when there is insufficient information. Evaluations of multiple state-of-the-art LLMs on GaRAGe show that the models tend to over-summarise rather than (a) ground their answers strictly on the annotated relevant passages (reaching at most a Relevance-Aware Factuality Score of 60%), or (b) deflect when no relevant grounding is available (reaching at most 31% true positive rate in deflections). The F1 in attribution to relevant sources is at most 58.9%, and we show that performance is particularly reduced when answering time-sensitive questions and when having to draw knowledge from sparser private grounding sources. 

---
# Silencing Empowerment, Allowing Bigotry: Auditing the Moderation of Hate Speech on Twitch 

**Authors**: Prarabdh Shukla, Wei Yin Chong, Yash Patel, Brennan Schaffner, Danish Pruthi, Arjun Bhagoji  

**Link**: [PDF](https://arxiv.org/pdf/2506.07667)  

**Abstract**: To meet the demands of content moderation, online platforms have resorted to automated systems. Newer forms of real-time engagement($\textit{e.g.}$, users commenting on live streams) on platforms like Twitch exert additional pressures on the latency expected of such moderation systems. Despite their prevalence, relatively little is known about the effectiveness of these systems. In this paper, we conduct an audit of Twitch's automated moderation tool ($\texttt{AutoMod}$) to investigate its effectiveness in flagging hateful content. For our audit, we create streaming accounts to act as siloed test beds, and interface with the live chat using Twitch's APIs to send over $107,000$ comments collated from $4$ datasets. We measure $\texttt{AutoMod}$'s accuracy in flagging blatantly hateful content containing misogyny, racism, ableism and homophobia. Our experiments reveal that a large fraction of hateful messages, up to $94\%$ on some datasets, $\textit{bypass moderation}$. Contextual addition of slurs to these messages results in $100\%$ removal, revealing $\texttt{AutoMod}$'s reliance on slurs as a moderation signal. We also find that contrary to Twitch's community guidelines, $\texttt{AutoMod}$ blocks up to $89.5\%$ of benign examples that use sensitive words in pedagogical or empowering contexts. Overall, our audit points to large gaps in $\texttt{AutoMod}$'s capabilities and underscores the importance for such systems to understand context effectively. 

---
# Synthesis by Design: Controlled Data Generation via Structural Guidance 

**Authors**: Lei Xu, Sirui Chen, Yuxuan Huang, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07664)  

**Abstract**: Mathematical reasoning remains challenging for LLMs due to complex logic and the need for precise computation. Existing methods enhance LLM reasoning by synthesizing datasets through problem rephrasing, but face issues with generation quality and problem complexity. To address this, we propose to extract structural information with generated problem-solving code from mathematical reasoning and guide data generation with structured solutions. Applied to MATH and GSM8K, our approach produces 39K problems with labeled intermediate steps and a 6.1K-problem benchmark of higher difficulty. Results on our benchmark show that model performance declines as reasoning length increases. Additionally, we conducted fine-tuning experiments using the proposed training data on a range of LLMs, and the results validate the effectiveness of our dataset. We hope the proposed method and dataset will contribute to future research in enhancing LLM reasoning capabilities. 

---
# Beyond Benchmarks: A Novel Framework for Domain-Specific LLM Evaluation and Knowledge Mapping 

**Authors**: Nitin Sharma, Thomas Wolfers, Çağatay Yıldız  

**Link**: [PDF](https://arxiv.org/pdf/2506.07658)  

**Abstract**: The paper addresses two critical challenges in language model (LM) evaluation: creating reliable domain-specific benchmarks and understanding knowledge representation during domain adaptation. We introduce a deterministic pipeline that converts raw domain corpora into completion-type benchmarks without relying on LMs or human curation, eliminating benchmark contamination issues while enabling evaluation on the latest domain data. Our approach generates domain-specific keywords and related word lists using TF and Term TF-IDF methods and constructs prompt-target pairs. We evaluate models by measuring their ability to complete these prompts with the correct domain-specific targets, providing a direct assessment of domain knowledge with low computational cost. Through comprehensive experiments across multiple models (GPT-2 medium/XL, Llama-2/3.1, OLMo-2, Qwen-2, Mistral) and domains, we demonstrate that our benchmark strongly correlates with expert-generated benchmarks while providing a more accurate measure of domain knowledge than traditional perplexity metrics. We reveal that domain adaptation happens rapidly in smaller models (within 500 steps) and illustrate a new approach to domain knowledge evaluation in base models during training for early stopping. By extending mechanistic analysis to domain adaptation, we discover that initial-to-mid layers are primarily responsible for attribute extraction, while later layers focus on next token prediction. Furthermore, we show that during adaptation, forgetting begins in the middle layers, where attribute extraction happens and is amplified in later layers. Our work provides both a practical evaluation methodology for domain-specific LMs and novel insights into knowledge representation during adaptation, with implications for more efficient fine-tuning strategies and targeted approaches to mitigate catastrophic forgetting. 

---
# Transcript-Prompted Whisper with Dictionary-Enhanced Decoding for Japanese Speech Annotation 

**Authors**: Rui Hu, Xiaolong Lin, Jiawang Liu, Shixi Huang, Zhenpeng Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07646)  

**Abstract**: In this paper, we propose a method for annotating phonemic and prosodic labels on a given audio-transcript pair, aimed at constructing Japanese text-to-speech (TTS) datasets. Our approach involves fine-tuning a large-scale pre-trained automatic speech recognition (ASR) model, conditioned on ground truth transcripts, to simultaneously output phrase-level graphemes and annotation labels. To further correct errors in phonemic labeling, we employ a decoding strategy that utilizes dictionary prior knowledge. The objective evaluation results demonstrate that our proposed method outperforms previous approaches relying solely on text or audio. The subjective evaluation results indicate that the naturalness of speech synthesized by the TTS model, trained with labels annotated using our method, is comparable to that of a model trained with manual annotations. 

---
# Evaluating LLMs Robustness in Less Resourced Languages with Proxy Models 

**Authors**: Maciej Chrabąszcz, Katarzyna Lorenc, Karolina Seweryn  

**Link**: [PDF](https://arxiv.org/pdf/2506.07645)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across various natural language processing (NLP) tasks in recent years. However, their susceptibility to jailbreaks and perturbations necessitates additional evaluations. Many LLMs are multilingual, but safety-related training data contains mainly high-resource languages like English. This can leave them vulnerable to perturbations in low-resource languages such as Polish. We show how surprisingly strong attacks can be cheaply created by altering just a few characters and using a small proxy model for word importance calculation. We find that these character and word-level attacks drastically alter the predictions of different LLMs, suggesting a potential vulnerability that can be used to circumvent their internal safety mechanisms. We validate our attack construction methodology on Polish, a low-resource language, and find potential vulnerabilities of LLMs in this language. Additionally, we show how it can be extended to other languages. We release the created datasets and code for further research. 

---
# TreeReview: A Dynamic Tree of Questions Framework for Deep and Efficient LLM-based Scientific Peer Review 

**Authors**: Yuan Chang, Ziyue Li, Hengyuan Zhang, Yuanbo Kong, Yanru Wu, Zhijiang Guo, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07642)  

**Abstract**: While Large Language Models (LLMs) have shown significant potential in assisting peer review, current methods often struggle to generate thorough and insightful reviews while maintaining efficiency. In this paper, we propose TreeReview, a novel framework that models paper review as a hierarchical and bidirectional question-answering process. TreeReview first constructs a tree of review questions by recursively decomposing high-level questions into fine-grained sub-questions and then resolves the question tree by iteratively aggregating answers from leaf to root to get the final review. Crucially, we incorporate a dynamic question expansion mechanism to enable deeper probing by generating follow-up questions when needed. We construct a benchmark derived from ICLR and NeurIPS venues to evaluate our method on full review generation and actionable feedback comments generation tasks. Experimental results of both LLM-based and human evaluation show that TreeReview outperforms strong baselines in providing comprehensive, in-depth, and expert-aligned review feedback, while reducing LLM token usage by up to 80% compared to computationally intensive approaches. Our code and benchmark dataset are available at this https URL. 

---
# Unblocking Fine-Grained Evaluation of Detailed Captions: An Explaining AutoRater and Critic-and-Revise Pipeline 

**Authors**: Brian Gordon, Yonatan Bitton, Andreea Marzoca, Yasumasa Onoe, Xiao Wang, Daniel Cohen-Or, Idan Szpektor  

**Link**: [PDF](https://arxiv.org/pdf/2506.07631)  

**Abstract**: Large Vision-Language Models (VLMs) now generate highly detailed, paragraphlength image captions, yet evaluating their factual accuracy remains challenging. Current methods often miss fine-grained errors, being designed for shorter texts or lacking datasets with verified inaccuracies. We introduce DOCCI-Critique, a benchmark with 1,400 VLM-generated paragraph captions (100 images, 14 VLMs) featuring over 10,216 sentence-level human annotations of factual correctness and explanatory rationales for errors, all within paragraph context. Building on this, we develop VNLI-Critique, a model for automated sentence-level factuality classification and critique generation. We highlight three key applications: (1) VNLI-Critique demonstrates robust generalization, validated by state-of-the-art performance on the M-HalDetect benchmark and strong results in CHOCOLATE claim verification. (2) The VNLI-Critique driven AutoRater for DOCCI-Critique provides reliable VLM rankings, showing excellent alignment with human factuality judgments (e.g., 0.98 Spearman). (3) An innovative Critic-and-Revise pipeline, where critiques from VNLI-Critique guide LLM-based corrections, achieves substantial improvements in caption factuality (e.g., a 46% gain on DetailCaps-4870). Our work offers a crucial benchmark alongside practical tools, designed to significantly elevate the standards for fine-grained evaluation and foster the improvement of VLM image understanding. Project page: this https URL 

---
# Intent Matters: Enhancing AI Tutoring with Fine-Grained Pedagogical Intent Annotation 

**Authors**: Kseniia Petukhova, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2506.07626)  

**Abstract**: Large language models (LLMs) hold great promise for educational applications, particularly in intelligent tutoring systems. However, effective tutoring requires alignment with pedagogical strategies - something current LLMs lack without task-specific adaptation. In this work, we explore whether fine-grained annotation of teacher intents can improve the quality of LLM-generated tutoring responses. We focus on MathDial, a dialog dataset for math instruction, and apply an automated annotation framework to re-annotate a portion of the dataset using a detailed taxonomy of eleven pedagogical intents. We then fine-tune an LLM using these new annotations and compare its performance to models trained on the original four-category taxonomy. Both automatic and qualitative evaluations show that the fine-grained model produces more pedagogically aligned and effective responses. Our findings highlight the value of intent specificity for controlled text generation in educational settings, and we release our annotated data and code to facilitate further research. 

---
# LoRMA: Low-Rank Multiplicative Adaptation for LLMs 

**Authors**: Harsh Bihany, Shubham Patel, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07621)  

**Abstract**: Large Language Models have shown remarkable capabilities in the NLP domain. Their effectiveness can mainly be attributed to their ability to adapt to an array of downstream tasks. However, generally, full fine-tuning is a computationally expensive job. To mitigate this, many techniques have been developed that prime efficiency, a prominent one being Low-Rank Adaptation (LoRA). However, LoRA and its variants employ re-parametrized additive updates. In this paper, we propose Low-Rank Multiplicative Adaptation (LoRMA), which shifts the paradigm of additive updates to a richer space of matrix multiplicative transformations. We tackle challenges such as computational complexity and rank bottleneck of matrix multiplication by effectively re-ordering operations and introducing rank inflation strategies. We conduct extensive experiments to demonstrate the effectiveness of our approach in terms of various evaluation metrics. 

---
# Vuyko Mistral: Adapting LLMs for Low-Resource Dialectal Translation 

**Authors**: Roman Kyslyi, Yuliia Maksymiuk, Ihor Pysmennyi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07617)  

**Abstract**: In this paper we introduce the first effort to adapt large language models (LLMs) to the Ukrainian dialect (in our case Hutsul), a low-resource and morphologically complex dialect spoken in the Carpathian Highlands. We created a parallel corpus of 9852 dialect-to-standard Ukrainian sentence pairs and a dictionary of 7320 dialectal word mappings. We also addressed data shortage by proposing an advanced Retrieval-Augmented Generation (RAG) pipeline to generate synthetic parallel translation pairs, expanding the corpus with 52142 examples. We have fine-tuned multiple open-source LLMs using LoRA and evaluated them on a standard-to-dialect translation task, also comparing with few-shot GPT-4o translation. In the absence of human annotators, we adopt a multi-metric evaluation strategy combining BLEU, chrF++, TER, and LLM-based judgment (GPT-4o). The results show that even small(7B) finetuned models outperform zero-shot baselines such as GPT-4o across both automatic and LLM-evaluated metrics. All data, models, and code are publicly released at: this https URL 

---
# PolitiSky24: U.S. Political Bluesky Dataset with User Stance Labels 

**Authors**: Peyman Rostami, Vahid Rahimzadeh, Ali Adibi, Azadeh Shakery  

**Link**: [PDF](https://arxiv.org/pdf/2506.07606)  

**Abstract**: Stance detection identifies the viewpoint expressed in text toward a specific target, such as a political figure. While previous datasets have focused primarily on tweet-level stances from established platforms, user-level stance resources, especially on emerging platforms like Bluesky remain scarce. User-level stance detection provides a more holistic view by considering a user's complete posting history rather than isolated posts. We present the first stance detection dataset for the 2024 U.S. presidential election, collected from Bluesky and centered on Kamala Harris and Donald Trump. The dataset comprises 16,044 user-target stance pairs enriched with engagement metadata, interaction graphs, and user posting histories. PolitiSky24 was created using a carefully evaluated pipeline combining advanced information retrieval and large language models, which generates stance labels with supporting rationales and text spans for transparency. The labeling approach achieves 81\% accuracy with scalable LLMs. This resource addresses gaps in political stance analysis through its timeliness, open-data nature, and user-level perspective. The dataset is available at this https URL 

---
# Instructing Large Language Models for Low-Resource Languages: A Systematic Study for Basque 

**Authors**: Oscar Sainz, Naiara Perez, Julen Etxaniz, Joseba Fernandez de Landa, Itziar Aldabe, Iker García-Ferrero, Aimar Zabala, Ekhi Azurmendi, German Rigau, Eneko Agirre, Mikel Artetxe, Aitor Soroa  

**Link**: [PDF](https://arxiv.org/pdf/2506.07597)  

**Abstract**: Instructing language models with user intent requires large instruction datasets, which are only available for a limited set of languages. In this paper, we explore alternatives to conventional instruction adaptation pipelines in low-resource scenarios. We assume a realistic scenario for low-resource languages, where only the following are available: corpora in the target language, existing open-weight multilingual base and instructed backbone LLMs, and synthetically generated instructions sampled from the instructed backbone. We present a comprehensive set of experiments for Basque that systematically study different combinations of these components evaluated on benchmarks and human preferences from 1,680 participants. Our conclusions show that target language corpora are essential, with synthetic instructions yielding robust models, and, most importantly, that using as backbone an instruction-tuned model outperforms using a base non-instructed model, and improved results when scaling up. Using Llama 3.1 instruct 70B as backbone our model comes near frontier models of much larger sizes for Basque, without using any Basque data apart from the 1.2B word corpora. We release code, models, instruction datasets, and human preferences to support full reproducibility in future research on low-resource language adaptation. 

---
# Beyond the Sentence: A Survey on Context-Aware Machine Translation with Large Language Models 

**Authors**: Ramakrishna Appicharla, Baban Gain, Santanu Pal, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2506.07583)  

**Abstract**: Despite the popularity of the large language models (LLMs), their application to machine translation is relatively underexplored, especially in context-aware settings. This work presents a literature review of context-aware translation with LLMs. The existing works utilise prompting and fine-tuning approaches, with few focusing on automatic post-editing and creating translation agents for context-aware machine translation. We observed that the commercial LLMs (such as ChatGPT and Tower LLM) achieved better results than the open-source LLMs (such as Llama and Bloom LLMs), and prompt-based approaches serve as good baselines to assess the quality of translations. Finally, we present some interesting future directions to explore. 

---
# SELT: Self-Evaluation Tree Search for LLMs with Task Decomposition 

**Authors**: Mengsong Wu, Di Zhang, Yuqiang Li, Dongzhan Zhou, Wenliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07557)  

**Abstract**: While Large Language Models (LLMs) have achieved remarkable success in a wide range of applications, their performance often degrades in complex reasoning tasks. In this work, we introduce SELT (Self-Evaluation LLM Tree Search), a novel framework that leverages a modified Monte Carlo Tree Search (MCTS) to enhance LLM reasoning without relying on external reward models. By redefining the Upper Confidence Bound scoring to align with intrinsic self-evaluation capabilities of LLMs and decomposing the inference process into atomic subtasks augmented with semantic clustering at each node, SELT effectively balances exploration and exploitation, reduces redundant reasoning paths, and mitigates hallucination. We validate our approach on challenging benchmarks, including the knowledge-based MMLU and the Tool Learning dataset Seal-Tools, where SELT achieves significant improvements in answer accuracy and reasoning robustness compared to baseline methods. Notably, our framework operates without task-specific fine-tuning, demonstrating strong generalizability across diverse reasoning tasks. Relevant results and code are available at this https URL . 

---
# Bit-level BPE: Below the byte boundary 

**Authors**: Sangwhan Moon, Tatsuya Hiraoka, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.07541)  

**Abstract**: Byte-level fallbacks for subword tokenization have become a common practice in large language models. In particular, it has been demonstrated to be incredibly effective as a pragmatic solution for preventing OOV, especially in the context of larger models. However, breaking a character down to individual bytes significantly increases the sequence length for long-tail tokens in languages such as Chinese, Japanese, and Korean (CJK) and other character-diverse contexts such as emoji. The increased sequence length results in longer computation during both training and inference. In this work, we propose a simple compression technique that reduces the sequence length losslessly. 

---
# Towards Large Language Models with Self-Consistent Natural Language Explanations 

**Authors**: Sahar Admoni, Ofra Amir, Assaf Hallak, Yftah Ziser  

**Link**: [PDF](https://arxiv.org/pdf/2506.07523)  

**Abstract**: Large language models (LLMs) seem to offer an easy path to interpretability: just ask them to explain their decisions. Yet, studies show that these post-hoc explanations often misrepresent the true decision process, as revealed by mismatches in feature importance. Despite growing evidence of this inconsistency, no systematic solutions have emerged, partly due to the high cost of estimating feature importance, which limits evaluations to small datasets. To address this, we introduce the Post-hoc Self-Consistency Bank (PSCB) - a large-scale benchmark of decisions spanning diverse tasks and models, each paired with LLM-generated explanations and corresponding feature importance scores. Analysis of PSCB reveals that self-consistency scores barely differ between correct and incorrect predictions. We also show that the standard metric fails to meaningfully distinguish between explanations. To overcome this limitation, we propose an alternative metric that more effectively captures variation in explanation quality. We use it to fine-tune LLMs via Direct Preference Optimization (DPO), leading to significantly better alignment between explanations and decision-relevant features, even under domain shift. Our findings point to a scalable path toward more trustworthy, self-consistent LLMs. 

---
# DeRAGEC: Denoising Named Entity Candidates with Synthetic Rationale for ASR Error Correction 

**Authors**: Solee Im, Wonjun Lee, Jinmyeong An, Yunsu Kim, Jungseul Ok, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.07510)  

**Abstract**: We present DeRAGEC, a method for improving Named Entity (NE) correction in Automatic Speech Recognition (ASR) systems. By extending the Retrieval-Augmented Generative Error Correction (RAGEC) framework, DeRAGEC employs synthetic denoising rationales to filter out noisy NE candidates before correction. By leveraging phonetic similarity and augmented definitions, it refines noisy retrieved NEs using in-context learning, requiring no additional training. Experimental results on CommonVoice and STOP datasets show significant improvements in Word Error Rate (WER) and NE hit ratio, outperforming baseline ASR and RAGEC methods. Specifically, we achieved a 28% relative reduction in WER compared to ASR without postprocessing. Our source code is publicly available at: this https URL 

---
# What Do Indonesians Really Need from Language Technology? A Nationwide Survey 

**Authors**: Muhammad Dehan Al Kautsar, Lucky Susanto, Derry Wijaya, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2506.07506)  

**Abstract**: There is an emerging effort to develop NLP for Indonesias 700+ local languages, but progress remains costly due to the need for direct engagement with native speakers. However, it is unclear what these language communities truly need from language technology. To address this, we conduct a nationwide survey to assess the actual needs of native speakers in Indonesia. Our findings indicate that addressing language barriers, particularly through machine translation and information retrieval, is the most critical priority. Although there is strong enthusiasm for advancements in language technology, concerns around privacy, bias, and the use of public data for AI training highlight the need for greater transparency and clear communication to support broader AI adoption. 

---
# DEBATE: A Dataset for Disentangling Textual Ambiguity in Mandarin Through Speech 

**Authors**: Haotian Guo, Jing Han, Yongfeng Tu, Shihao Gao, Shengfan Shen, Wulong Xiang, Weihao Gan, Zixing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07502)  

**Abstract**: Despite extensive research on textual and visual disambiguation, disambiguation through speech (DTS) remains underexplored. This is largely due to the lack of high-quality datasets that pair spoken sentences with richly ambiguous text. To address this gap, we present DEBATE, a unique public Chinese speech-text dataset designed to study how speech cues and patterns-pronunciation, pause, stress and intonation-can help resolve textual ambiguity and reveal a speaker's true intent. DEBATE contains 1,001 carefully selected ambiguous utterances, each recorded by 10 native speakers, capturing diverse linguistic ambiguities and their disambiguation through speech. We detail the data collection pipeline and provide rigorous quality analysis. Additionally, we benchmark three state-of-the-art large speech and audio-language models, illustrating clear and huge performance gaps between machine and human understanding of spoken intent. DEBATE represents the first effort of its kind and offers a foundation for building similar DTS datasets across languages and cultures. The dataset and associated code are available at: this https URL. 

---
# A Hybrid GA LLM Framework for Structured Task Optimization 

**Authors**: Berry Feng, Jonas Lin, Patrick Lau  

**Link**: [PDF](https://arxiv.org/pdf/2506.07483)  

**Abstract**: GA LLM is a hybrid framework that combines Genetic Algorithms with Large Language Models to handle structured generation tasks under strict constraints. Each output, such as a plan or report, is treated as a gene, and evolutionary operations like selection, crossover, and mutation are guided by the language model to iteratively improve solutions. The language model provides domain knowledge and creative variation, while the genetic algorithm ensures structural integrity and global optimization. GA LLM has proven effective in tasks such as itinerary planning, academic outlining, and business reporting, consistently producing well structured and requirement satisfying results. Its modular design also makes it easy to adapt to new tasks. Compared to using a language model alone, GA LLM achieves better constraint satisfaction and higher quality solutions by combining the strengths of both components. 

---
# Improving Fairness of Large Language Models in Multi-document Summarization 

**Authors**: Haoyuan Li Yusen Zhang, Snigdha Chaturvedi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07479)  

**Abstract**: Fairness in multi-document summarization (MDS) is crucial for providing comprehensive views across documents with diverse social attribute values, which can significantly impact decision-making. For example, a summarization system that tends to overrepresent negative reviews of products can mislead customers into disregarding good products. Previous works measure fairness in MDS at two levels: summary-level and corpus-level. While summary-level fairness focuses on individual summaries, corpus-level fairness focuses on a corpus of summaries. Recent methods primarily focus on summary-level fairness. We propose FairPO, a preference tuning method that focuses on both summary-level and corpus-level fairness in MDS. To improve summary-level fairness, we propose to generate preference pairs by perturbing document sets. To improve corpus-level fairness, we propose fairness-aware preference tuning by dynamically adjusting the weights of preference pairs. Our experiments show that FairPO outperforms strong baselines while maintaining the critical qualities of summaries. The code is available at this https URL. 

---
# CCI4.0: A Bilingual Pretraining Dataset for Enhancing Reasoning in Large Language Models 

**Authors**: Guang Liu, Liangdong Wang, Jijie Li, Yang Yu, Yao Xu, Jiabei Chen, Yu Bai, Feng Liao, Yonghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.07463)  

**Abstract**: We introduce CCI4.0, a large-scale bilingual pre-training dataset engineered for superior data quality and diverse human-like reasoning trajectory. CCI4.0 occupies roughly $35$ TB of disk space and comprises two sub-datasets: CCI4.0-M2-Base and CCI4.0-M2-CoT. CCI4.0-M2-Base combines a $5.2$ TB carefully curated Chinese web corpus, a $22.5$ TB English subset from Nemotron-CC, and diverse sources from math, wiki, arxiv, and code. Although these data are mostly sourced from well-processed datasets, the quality standards of various domains are dynamic and require extensive expert experience and labor to process. So, we propose a novel pipeline justifying data quality mainly based on models through two-stage deduplication, multiclassifier quality scoring, and domain-aware fluency filtering. We extract $4.5$ billion pieces of CoT(Chain-of-Thought) templates, named CCI4.0-M2-CoT. Differing from the distillation of CoT from larger models, our proposed staged CoT extraction exemplifies diverse reasoning patterns and significantly decreases the possibility of hallucination. Empirical evaluations demonstrate that LLMs pre-trained in CCI4.0 benefit from cleaner, more reliable training signals, yielding consistent improvements in downstream tasks, especially in math and code reflection tasks. Our results underscore the critical role of rigorous data curation and human thinking templates in advancing LLM performance, shedding some light on automatically processing pretraining corpora. 

---
# From Calibration to Collaboration: LLM Uncertainty Quantification Should Be More Human-Centered 

**Authors**: Siddartha Devic, Tejas Srinivasan, Jesse Thomason, Willie Neiswanger, Vatsal Sharan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07461)  

**Abstract**: Large Language Models (LLMs) are increasingly assisting users in the real world, yet their reliability remains a concern. Uncertainty quantification (UQ) has been heralded as a tool to enhance human-LLM collaboration by enabling users to know when to trust LLM predictions. We argue that current practices for uncertainty quantification in LLMs are not optimal for developing useful UQ for human users making decisions in real-world tasks. Through an analysis of 40 LLM UQ methods, we identify three prevalent practices hindering the community's progress toward its goal of benefiting downstream users: 1) evaluating on benchmarks with low ecological validity; 2) considering only epistemic uncertainty; and 3) optimizing metrics that are not necessarily indicative of downstream utility. For each issue, we propose concrete user-centric practices and research directions that LLM UQ researchers should consider. Instead of hill-climbing on unrepresentative tasks using imperfect metrics, we argue that the community should adopt a more human-centered approach to LLM uncertainty quantification. 

---
# KScope: A Framework for Characterizing the Knowledge Status of Language Models 

**Authors**: Yuxin Xiao, Shan Chen, Jack Gallifant, Danielle Bitterman, Thomas Hartvigsen, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07458)  

**Abstract**: Characterizing a large language model's (LLM's) knowledge of a given question is challenging. As a result, prior work has primarily examined LLM behavior under knowledge conflicts, where the model's internal parametric memory contradicts information in the external context. However, this does not fully reflect how well the model knows the answer to the question. In this paper, we first introduce a taxonomy of five knowledge statuses based on the consistency and correctness of LLM knowledge modes. We then propose KScope, a hierarchical framework of statistical tests that progressively refines hypotheses about knowledge modes and characterizes LLM knowledge into one of these five statuses. We apply KScope to nine LLMs across four datasets and systematically establish: (1) Supporting context narrows knowledge gaps across models. (2) Context features related to difficulty, relevance, and familiarity drive successful knowledge updates. (3) LLMs exhibit similar feature preferences when partially correct or conflicted, but diverge sharply when consistently wrong. (4) Context summarization constrained by our feature analysis, together with enhanced credibility, further improves update effectiveness and generalizes across LLMs. 

---
# Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling 

**Authors**: Pritom Saha Akash, Kevin Chen-Chuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07453)  

**Abstract**: Topic modeling plays a vital role in uncovering hidden semantic structures within text corpora, but existing models struggle in low-resource settings where limited target-domain data leads to unstable and incoherent topic inference. We address this challenge by formally introducing domain adaptation for low-resource topic modeling, where a high-resource source domain informs a low-resource target domain without overwhelming it with irrelevant content. We establish a finite-sample generalization bound showing that effective knowledge transfer depends on robust performance in both domains, minimizing latent-space discrepancy, and preventing overfitting to the data. Guided by these insights, we propose DALTA (Domain-Aligned Latent Topic Adaptation), a new framework that employs a shared encoder for domain-invariant features, specialized decoders for domain-specific nuances, and adversarial alignment to selectively transfer relevant information. Experiments on diverse low-resource datasets demonstrate that DALTA consistently outperforms state-of-the-art methods in terms of topic coherence, stability, and transferability. 

---
# LG-ANNA-Embedding technical report 

**Authors**: Jooyoung Choi, Hyun Kim, Hansol Jang, Changwook Jun, Kyunghoon Bae, Hyewon Choi, Stanley Jungkyu Choi, Honglak Lee, Chulmin Yun  

**Link**: [PDF](https://arxiv.org/pdf/2506.07438)  

**Abstract**: This report presents a unified instruction-based framework for learning generalized text embeddings optimized for both information retrieval (IR) and non-IR tasks. Built upon a decoder-only large language model (Mistral-7B), our approach combines in-context learning, soft supervision, and adaptive hard-negative mining to generate context-aware embeddings without task-specific fine-tuning. Structured instructions and few-shot examples are used to guide the model across diverse tasks, enabling strong performance on classification, semantic similarity, clustering, and reranking benchmarks. To improve semantic discrimination, we employ a soft labeling framework where continuous relevance scores, distilled from a high-performance dense retriever and reranker, serve as fine-grained supervision signals. In addition, we introduce adaptive margin-based hard-negative mining, which filters out semantically ambiguous negatives based on their similarity to positive examples, thereby enhancing training stability and retrieval robustness. Our model is evaluated on the newly introduced MTEB (English, v2) benchmark, covering 41 tasks across seven categories. Results show that our method achieves strong generalization and ranks among the top-performing models by Borda score, outperforming several larger or fully fine-tuned baselines. These findings highlight the effectiveness of combining in-context prompting, soft supervision, and adaptive sampling for scalable, high-quality embedding generation. 

---
# Well Begun is Half Done: Low-resource Preference Alignment by Weak-to-Strong Decoding 

**Authors**: Feifan Song, Shaohang Wei, Wen Luo, Yuxuan Fan, Tianyu Liu, Guoyin Wang, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07434)  

**Abstract**: Large Language Models (LLMs) require alignment with human preferences to avoid generating offensive, false, or meaningless content. Recently, low-resource methods for LLM alignment have been popular, while still facing challenges in obtaining both high-quality and aligned content. Motivated by the observation that the difficulty of generating aligned responses is concentrated at the beginning of decoding, we propose a novel framework, Weak-to-Strong Decoding (WSD), to enhance the alignment ability of base models by the guidance of a small aligned model. The small model first drafts well-aligned beginnings, followed by the large base model to continue the rest, controlled by a well-designed auto-switch mechanism. We also collect a new dataset, GenerAlign, to fine-tune a small-sized Pilot-3B as the draft model, which effectively enhances different base models under the WSD framework to outperform all baseline methods, while avoiding degradation on downstream tasks, termed as the alignment tax. Extensive experiments are further conducted to examine the impact of different settings and time efficiency, as well as analyses on the intrinsic mechanisms of WSD in depth. 

---
# Conjoined Predication and Scalar Implicature 

**Authors**: Ratna Kandala  

**Link**: [PDF](https://arxiv.org/pdf/2506.07429)  

**Abstract**: Magri (2016) investigates two puzzles arising from conjunction. Although Magri has proposed a solution to the second puzzle, the first remains unresolved. This first puzzle reveals a hidden interaction among quantification, collective/concurrent interpretation, and contextual updating dimensions that have yet to be explored. In essence, the problem is that certain forms of sentences like "Some Italians come from a warm country," when conjoined as in "(Only) Some Italians come from a warm country and are blond," sound infelicitous, even though no obvious alternative triggers a conflicting scalar implicature. In this paper, we offer a conceptual analysis of Magri's first puzzle by situating it within its original theoretical framework. We argue that the oddness arises from the collective or concurrent reading of the conjunctive predicate: in examples such as "(Only) Some Italians come from a warm country and are blond," this interpretation generates an indirect contextual contradiction. Moreover, we suggest that the pragmatic mechanisms governing scalar implicature generation extend beyond what is captured by exhaustification-based grammatical licensing accounts. 

---
# Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models 

**Authors**: Kyeonghyun Kim, Jinhee Jang, Juhwan Choi, Yoonji Lee, Kyohoon Jin, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07424)  

**Abstract**: Large language models (LLMs) are renowned for their extensive linguistic knowledge and strong generalization capabilities, but their high computational demands make them unsuitable for resource-constrained environments. In contrast, small language models (SLMs) are computationally efficient but often lack the broad generalization capacity of LLMs. To bridge this gap, we propose PiFi, a novel framework that combines the strengths of both LLMs and SLMs to achieve high performance while maintaining efficiency. PiFi integrates a single frozen layer from an LLM into a SLM and fine-tunes the combined model for specific tasks, boosting performance without a significant increase in computational cost. We show that PiFi delivers consistent performance improvements across a range of natural language processing tasks, including both natural language understanding and generation. Moreover, our findings demonstrate PiFi's ability to effectively leverage LLM knowledge, enhancing generalization to unseen domains and facilitating the transfer of linguistic abilities. 

---
# SEED: Enhancing Text-to-SQL Performance and Practical Usability Through Automatic Evidence Generation 

**Authors**: Janghyeon Yun, Sang-goo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.07423)  

**Abstract**: Text-to-SQL enables non-experts to retrieve data from databases by converting natural language queries into SQL. However, state-of-the-art text-to-SQL studies rely on the BIRD dataset, which assumes that evidence is provided along with questions. Although BIRD facilitates research advancements, it assumes that users have expertise and domain knowledge, contradicting the fundamental goal of text-to-SQL. In addition, human-generated evidence in BIRD contains defects, including missing or erroneous evidence, which affects model performance. To address this issue, we propose SEED (System for Evidence Extraction and Domain knowledge generation), an approach that automatically generates evidence to improve performance and practical usability in real-world scenarios. SEED systematically analyzes database schema, description files, and values to extract relevant information. We evaluated SEED on BIRD and Spider, demonstrating that it significantly improves SQL generation accuracy in the no-evidence scenario, and in some cases, even outperforms the setting where BIRD evidence is provided. Our results highlight that SEED-generated evidence not only bridges the gap between research and real-world deployment but also improves the adaptability and robustness of text-to-SQL models. Our code is available at this https URL 

---
# Refusal-Feature-guided Teacher for Safe Finetuning via Data Filtering and Alignment Distillation 

**Authors**: Seokil Ham, Yubin Choi, Seungju Cho, Yujin Yang, Younghun Kim, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07356)  

**Abstract**: Recently, major AI service providers such as Google and OpenAI have introduced Finetuning-as-a-Service, which enables users to customize Large Language Models (LLMs) for specific downstream tasks using their own data. However, this service is vulnerable to degradation of LLM safety-alignment when user data contains harmful prompts. While some prior works address this issue, fundamentally filtering harmful data from user data remains unexplored. Motivated by our observation that a directional representation reflecting refusal behavior (called the refusal feature) obtained from safety-aligned LLMs can inherently distinguish between harmful and harmless prompts, we propose the Refusal-Feature-guided Teacher (ReFT). Our ReFT model is trained to identify harmful prompts based on the similarity between input prompt features and its refusal feature. During finetuning, the ReFT model serves as a teacher that filters harmful prompts from user data and distills alignment knowledge into the base model. Extensive experiments demonstrate that our ReFT-based finetuning strategy effectively minimizes harmful outputs and enhances finetuning accuracy for user-specific tasks, offering a practical solution for secure and reliable deployment of LLMs in Finetuning-as-a-Service. 

---
# Improving LLM Reasoning through Interpretable Role-Playing Steering 

**Authors**: Anyi Wang, Dong Shu, Yifan Wang, Yunpu Ma, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.07335)  

**Abstract**: Role-playing has emerged as an effective technique for enhancing the reasoning capabilities of large language models (LLMs). However, existing methods primarily rely on prompt engineering, which often lacks stability and interpretability. In this paper, we introduce Sparse Autoencoder Role-Playing Steering (SRPS), a novel framework that identifies and manipulates internal model features associated with role-playing behavior. Our approach extracts latent representations from role-play prompts, selects the most relevant features based on activation patterns, and constructs a steering vector that can be injected into the model's residual stream with controllable intensity. Our method enables fine-grained control over role-specific behavior and offers insights into how role information influences internal model activations. Extensive experiments across various reasoning benchmarks and model sizes demonstrate consistent performance gains. Notably, in the zero-shot chain-of-thought (CoT) setting, the accuracy of Llama3.1-8B on CSQA improves from 31.86% to 39.80%, while Gemma2-9B on SVAMP increases from 37.50% to 45.10%. These results highlight the potential of SRPS to enhance reasoning ability in LLMs, providing better interpretability and stability compared to traditional prompt-based role-playing. 

---
# Reward Model Interpretability via Optimal and Pessimal Tokens 

**Authors**: Brian Christian, Hannah Rose Kirk, Jessica A.F. Thompson, Christopher Summerfield, Tsvetomira Dumbalska  

**Link**: [PDF](https://arxiv.org/pdf/2506.07326)  

**Abstract**: Reward modeling has emerged as a crucial component in aligning large language models with human values. Significant attention has focused on using reward models as a means for fine-tuning generative models. However, the reward models themselves -- which directly encode human value judgments by turning prompt-response pairs into scalar rewards -- remain relatively understudied. We present a novel approach to reward model interpretability through exhaustive analysis of their responses across their entire vocabulary space. By examining how different reward models score every possible single-token response to value-laden prompts, we uncover several striking findings: (i) substantial heterogeneity between models trained on similar objectives, (ii) systematic asymmetries in how models encode high- vs low-scoring tokens, (iii) significant sensitivity to prompt framing that mirrors human cognitive biases, and (iv) overvaluation of more frequent tokens. We demonstrate these effects across ten recent open-source reward models of varying parameter counts and architectures. Our results challenge assumptions about the interchangeability of reward models, as well as their suitability as proxies of complex and context-dependent human values. We find that these models can encode concerning biases toward certain identity groups, which may emerge as unintended consequences of harmlessness training -- distortions that risk propagating through the downstream large language models now deployed to millions. 

---
# ConfQA: Answer Only If You Are Confident 

**Authors**: Yin Huang, Yifan Ethan Xu, Kai Sun, Vera Yan, Alicia Sun, Haidar Khan, Jimmy Nguyen, Mohammad Kachuee, Zhaojiang Lin, Yue Liu, Aaron Colak, Anuj Kumar, Wen-tau Yih, Xin Luna Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07309)  

**Abstract**: Can we teach Large Language Models (LLMs) to refrain from hallucinating factual statements? In this paper we present a fine-tuning strategy that we call ConfQA, which can reduce hallucination rate from 20-40% to under 5% across multiple factuality benchmarks. The core idea is simple: when the LLM answers a question correctly, it is trained to continue with the answer; otherwise, it is trained to admit "I am unsure". But there are two key factors that make the training highly effective. First, we introduce a dampening prompt "answer only if you are confident" to explicitly guide the behavior, without which hallucination remains high as 15%-25%. Second, we leverage simple factual statements, specifically attribute values from knowledge graphs, to help LLMs calibrate the confidence, resulting in robust generalization across domains and question types. Building on this insight, we propose the Dual Neural Knowledge framework, which seamlessly select between internally parameterized neural knowledge and externally recorded symbolic knowledge based on ConfQA's confidence. The framework enables potential accuracy gains to beyond 95%, while reducing unnecessary external retrievals by over 30%. 

---
# Subjectivity in the Annotation of Bridging Anaphora 

**Authors**: Lauren Levine, Amir Zeldes  

**Link**: [PDF](https://arxiv.org/pdf/2506.07297)  

**Abstract**: Bridging refers to the associative relationship between inferable entities in a discourse and the antecedents which allow us to understand them, such as understanding what "the door" means with respect to an aforementioned "house". As identifying associative relations between entities is an inherently subjective task, it is difficult to achieve consistent agreement in the annotation of bridging anaphora and their antecedents. In this paper, we explore the subjectivity involved in the annotation of bridging instances at three levels: anaphor recognition, antecedent resolution, and bridging subtype selection. To do this, we conduct an annotation pilot on the test set of the existing GUM corpus, and propose a newly developed classification system for bridging subtypes, which we compare to previously proposed schemes. Our results suggest that some previous resources are likely to be severely under-annotated. We also find that while agreement on the bridging subtype category was moderate, annotator overlap for exhaustively identifying instances of bridging is low, and that many disagreements resulted from subjective understanding of the entities involved. 

---
# Exploring the Impact of Temperature on Large Language Models:Hot or Cold? 

**Authors**: Lujun Li, Lama Sleem, Niccolo' Gentile, Geoffrey Nichil, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2506.07295)  

**Abstract**: The sampling temperature, a critical hyperparameter in large language models (LLMs), modifies the logits before the softmax layer, thereby reshaping the distribution of output tokens. Recent studies have challenged the Stochastic Parrots analogy by demonstrating that LLMs are capable of understanding semantics rather than merely memorizing data and that randomness, modulated by sampling temperature, plays a crucial role in model inference. In this study, we systematically evaluated the impact of temperature in the range of 0 to 2 on data sets designed to assess six different capabilities, conducting statistical analyses on open source models of three different sizes: small (1B--4B), medium (6B--13B), and large (40B--80B). Our findings reveal distinct skill-specific effects of temperature on model performance, highlighting the complexity of optimal temperature selection in practical applications. To address this challenge, we propose a BERT-based temperature selector that takes advantage of these observed effects to identify the optimal temperature for a given prompt. We demonstrate that this approach can significantly improve the performance of small and medium models in the SuperGLUE datasets. Furthermore, our study extends to FP16 precision inference, revealing that temperature effects are consistent with those observed in 4-bit quantized models. By evaluating temperature effects up to 4.0 in three quantized models, we find that the Mutation Temperature -- the point at which significant performance changes occur -- increases with model size. 

---
# Parsing the Switch: LLM-Based UD Annotation for Complex Code-Switched and Low-Resource Languages 

**Authors**: Olga Kellert, Nemika Tyagi, Muhammad Imran, Nelvin Licona-Guevara, Carlos Gómez-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2506.07274)  

**Abstract**: Code-switching presents a complex challenge for syntactic analysis, especially in low-resource language settings where annotated data is scarce. While recent work has explored the use of large language models (LLMs) for sequence-level tagging, few approaches systematically investigate how well these models capture syntactic structure in code-switched contexts. Moreover, existing parsers trained on monolingual treebanks often fail to generalize to multilingual and mixed-language input. To address this gap, we introduce the BiLingua Parser, an LLM-based annotation pipeline designed to produce Universal Dependencies (UD) annotations for code-switched text. First, we develop a prompt-based framework for Spanish-English and Spanish-Guaraní data, combining few-shot LLM prompting with expert review. Second, we release two annotated datasets, including the first Spanish-Guaraní UD-parsed corpus. Third, we conduct a detailed syntactic analysis of switch points across language pairs and communicative contexts. Experimental results show that BiLingua Parser achieves up to 95.29% LAS after expert revision, significantly outperforming prior baselines and multilingual parsers. These results show that LLMs, when carefully guided, can serve as practical tools for bootstrapping syntactic resources in under-resourced, code-switched environments. Data and source code are available at this https URL 

---
# Question Answering under Temporal Conflict: Evaluating and Organizing Evolving Knowledge with LLMs 

**Authors**: Atahan Özer, Çağatay Yıldız  

**Link**: [PDF](https://arxiv.org/pdf/2506.07270)  

**Abstract**: Large language models (LLMs) exhibit remarkable capabilities in question answering and reasoning thanks to their extensive parametric memory. However, their knowledge is inherently limited by the scope of their pre-training data, while real-world information evolves continuously. Updating this knowledge typically requires costly and brittle re-training, or in-context learning (ICL), which becomes impractical at scale given the volume and volatility of modern information. Motivated by these limitations, we investigate how LLMs perform when exposed to temporal text corpora, or documents that reflect evolving knowledge over time, such as sports biographies where facts like a player's "current team" change year by year. To this end, we introduce two new benchmarks: Temporal Wiki, which captures factual drift across historical Wikipedia snapshots, and Unified Clark, which aggregates timestamped news articles to simulate real-world information accumulation. Our analysis reveals that LLMs often struggle to reconcile conflicting or outdated facts and can be misled when multiple versions of a fact appear in context. To address these issues, we propose a lightweight, agentic framework that incrementally builds a structured, external memory from source documents without requiring re-training. This knowledge organization strategy enables models to retrieve and reason over temporally filtered, relevant information at inference time. Empirically, our method outperforms ICL and RAG baselines across both benchmarks, especially on questions requiring more complex reasoning or integration of conflicting facts. 

---
# Bias Attribution in Filipino Language Models: Extending a Bias Interpretability Metric for Application on Agglutinative Languages 

**Authors**: Lance Calvin Lim Gamboa, Yue Feng, Mark Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.07249)  

**Abstract**: Emerging research on bias attribution and interpretability have revealed how tokens contribute to biased behavior in language models processing English texts. We build on this line of inquiry by adapting the information-theoretic bias attribution score metric for implementation on models handling agglutinative languages, particularly Filipino. We then demonstrate the effectiveness of our adapted method by using it on a purely Filipino model and on three multilingual models: one trained on languages worldwide and two on Southeast Asian data. Our results show that Filipino models are driven towards bias by words pertaining to people, objects, and relationships, entity-based themes that stand in contrast to the action-heavy nature of bias-contributing themes in English (i.e., criminal, sexual, and prosocial behaviors). These findings point to differences in how English and non-English models process inputs linked to sociodemographic groups and bias. 

---
# Improving the Efficiency of Long Document Classification using Sentence Ranking Approach 

**Authors**: Prathamesh Kokate, Mitali Sarnaik, Manavi Khopade, Raviraj Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07248)  

**Abstract**: Long document classification poses challenges due to the computational limitations of transformer-based models, particularly BERT, which are constrained by fixed input lengths and quadratic attention complexity. Moreover, using the full document for classification is often redundant, as only a subset of sentences typically carries the necessary information. To address this, we propose a TF-IDF-based sentence ranking method that improves efficiency by selecting the most informative content. Our approach explores fixed-count and percentage-based sentence selection, along with an enhanced scoring strategy combining normalized TF-IDF scores and sentence length. Evaluated on the MahaNews LDC dataset of long Marathi news articles, the method consistently outperforms baselines such as first, last, and random sentence selection. With MahaBERT-v2, we achieve near-identical classification accuracy with just a 0.33 percent drop compared to the full-context baseline, while reducing input size by over 50 percent and inference latency by 43 percent. This demonstrates that significant context reduction is possible without sacrificing performance, making the method practical for real-world long document classification tasks. 

---
# SDE-SQL: Enhancing Text-to-SQL Generation in Large Language Models via Self-Driven Exploration with SQL Probes 

**Authors**: Wenxuan Xie, Yaxun Dai, Wenhao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07245)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly improved performance on the Text-to-SQL task. However, prior approaches typically rely on static, pre-processed database information provided at inference time, which limits the model's ability to fully understand the database contents. Without dynamic interaction, LLMs are constrained to fixed, human-provided context and cannot autonomously explore the underlying data. To address this limitation, we propose SDE-SQL, a framework that enables large language models to perform self-driven exploration of databases during inference. This is accomplished by generating and executing SQL probes, which allow the model to actively retrieve information from the database and iteratively update its understanding of the data. Unlike prior methods, SDE-SQL operates in a zero-shot setting, without relying on any question-SQL pairs as in-context demonstrations. When evaluated on the BIRD benchmark with Qwen2.5-72B-Instruct, SDE-SQL achieves an 8.02% relative improvement in execution accuracy over the vanilla Qwen2.5-72B-Instruct baseline, establishing a new state-of-the-art among methods based on open-source models without supervised fine-tuning (SFT) or model ensembling. Moreover, with SFT, the performance of SDE-SQL can be further enhanced, yielding an additional 0.52% improvement. 

---
# Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs 

**Authors**: Wenrui Zhou, Shu Yang, Qingsong Yang, Zikun Guo, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07180)  

**Abstract**: As video large language models (Video-LLMs) become increasingly integrated into real-world applications that demand grounded multimodal reasoning, ensuring their factual consistency and reliability is of critical importance. However, sycophancy, the tendency of these models to align with user input even when it contradicts the visual evidence, undermines their trustworthiness in such contexts. Current sycophancy research has largely overlooked its specific manifestations in the video-language domain, resulting in a notable absence of systematic benchmarks and targeted evaluations to understand how Video-LLMs respond under misleading user input. To fill this gap, we propose VISE (Video-LLM Sycophancy Benchmarking and Evaluation), the first dedicated benchmark designed to evaluate sycophantic behavior in state-of-the-art Video-LLMs across diverse question formats, prompt biases, and visual reasoning tasks. Specifically, VISE pioneeringly brings linguistic perspectives on sycophancy into the visual domain, enabling fine-grained analysis across multiple sycophancy types and interaction patterns. In addition, we explore key-frame selection as an interpretable, training-free mitigation strategy, which reveals potential paths for reducing sycophantic bias by strengthening visual grounding. 

---
# RULE: Reinforcement UnLEarning Achieves Forget-Retain Pareto Optimality 

**Authors**: Chenlong Zhang, Zhuoran Jin, Hongbang Yuan, Jiaheng Wei, Tong Zhou, Kang Liu, Jun Zhao, Yubo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07171)  

**Abstract**: The widespread deployment of Large Language Models (LLMs) trained on massive, uncurated corpora has raised growing concerns about the inclusion of sensitive, copyrighted, or illegal content. This has led to increasing interest in LLM unlearning: the task of selectively removing specific information from a model without retraining from scratch or degrading overall utility. However, existing methods often rely on large-scale forget and retain datasets, and suffer from unnatural responses, poor generalization, or catastrophic utility loss. In this work, we propose Reinforcement UnLearning (RULE), an efficient framework that formulates unlearning as a refusal boundary optimization problem. RULE is trained with a small portion of the forget set and synthesized boundary queries, using a verifiable reward function that encourages safe refusal on forget--related queries while preserving helpful responses on permissible inputs. We provide both theoretical and empirical evidence demonstrating the effectiveness of RULE in achieving targeted unlearning without compromising model utility. Experimental results show that, with only $12%$ forget set and $8%$ synthesized boundary data, RULE outperforms existing baselines by up to $17.5%$ forget quality and $16.3%$ naturalness response while maintaining general utility, achieving forget--retain Pareto optimality. Remarkably, we further observe that RULE improves the naturalness of model outputs, enhances training efficiency, and exhibits strong generalization ability, generalizing refusal behavior to semantically related but unseen queries. 

---
# CTDGSI: A comprehensive exploitation of instance selection methods for automatic text classification. VII Concurso de Teses, Dissertações e Trabalhos de Graduação em SI -- XXI Simpósio Brasileiro de Sistemas de Informação 

**Authors**: Washington Cunha, Leonardo Rocha, Marcos André Gonçalves  

**Link**: [PDF](https://arxiv.org/pdf/2506.07169)  

**Abstract**: Progress in Natural Language Processing (NLP) has been dictated by the rule of more: more data, more computing power and more complexity, best exemplified by the Large Language Models. However, training (or fine-tuning) large dense models for specific applications usually requires significant amounts of computing resources. This \textbf{Ph.D. dissertation} focuses on an under-investi\-gated NLP data engineering technique, whose potential is enormous in the current scenario known as Instance Selection (IS). The IS goal is to reduce the training set size by removing noisy or redundant instances while maintaining the effectiveness of the trained models and reducing the training process cost. We provide a comprehensive and scientifically sound comparison of IS methods applied to an essential NLP task -- Automatic Text Classification (ATC), considering several classification solutions and many datasets. Our findings reveal a significant untapped potential for IS solutions. We also propose two novel IS solutions that are noise-oriented and redundancy-aware, specifically designed for large datasets and transformer architectures. Our final solution achieved an average reduction of 41\% in training sets, while maintaining the same levels of effectiveness in all datasets. Importantly, our solutions demonstrated speedup improvements of 1.67x (up to 2.46x), making them scalable for datasets with hundreds of thousands of documents. 

---
# GeometryZero: Improving Geometry Solving for LLM with Group Contrastive Policy Optimization 

**Authors**: Yikun Wang, Yibin Wang, Dianyi Wang, Zimian Peng, Qipeng Guo, Dacheng Tao, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07160)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable capabilities across diverse domains, particularly in mathematical reasoning, amid which geometry problem solving remains a challenging area where auxiliary construction plays a enssential role. Existing approaches either achieve suboptimal performance or rely on massive LLMs (e.g., GPT-4o), incurring massive computational costs. We posit that reinforcement learning with verifiable reward (e.g., GRPO) offers a promising direction for training smaller models that effectively combine auxiliary construction with robust geometric reasoning. However, directly applying GRPO to geometric reasoning presents fundamental limitations due to its dependence on unconditional rewards, which leads to indiscriminate and counterproductive auxiliary constructions. To address these challenges, we propose Group Contrastive Policy Optimization (GCPO), a novel reinforcement learning framework featuring two key innovations: (1) Group Contrastive Masking, which adaptively provides positive or negative reward signals for auxiliary construction based on contextual utility, and a (2) length reward that promotes longer reasoning chains. Building on GCPO, we develop GeometryZero, a family of affordable-size geometric reasoning models that judiciously determine when to employ auxiliary construction. Our extensive empirical evaluation across popular geometric benchmarks (Geometry3K, MathVista) demonstrates that GeometryZero models consistently outperform baselines (e.g. GRPO), achieving an average improvement of 4.29% across all benchmarks. 

---
# Syntactic Control of Language Models by Posterior Inference 

**Authors**: Vicky Xefteri, Tim Vieira, Ryan Cotterell, Afra Amini  

**Link**: [PDF](https://arxiv.org/pdf/2506.07154)  

**Abstract**: Controlling the syntactic structure of text generated by language models is valuable for applications requiring clarity, stylistic consistency, or interpretability, yet it remains a challenging task. In this paper, we argue that sampling algorithms based on the posterior inference can effectively enforce a target constituency structure during generation. Our approach combines sequential Monte Carlo, which estimates the posterior distribution by sampling from a proposal distribution, with a syntactic tagger that ensures that each generated token aligns with the desired syntactic structure. Our experiments with GPT2 and Llama3-8B models show that with an appropriate proposal distribution, we can improve syntactic accuracy, increasing the F1 score from $12.31$ (GPT2-large) and $35.33$ (Llama3-8B) to about $93$ in both cases without compromising the language model's fluency. These results underscore both the complexity of syntactic control and the effectiveness of sampling algorithms, offering a promising approach for applications where precise control over syntax is essential. 

---
# Semantic-preserved Augmentation with Confidence-weighted Fine-tuning for Aspect Category Sentiment Analysis 

**Authors**: Yaping Chai, Haoran Xie, Joe S. Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.07148)  

**Abstract**: Large language model (LLM) is an effective approach to addressing data scarcity in low-resource scenarios. Recent existing research designs hand-crafted prompts to guide LLM for data augmentation. We introduce a data augmentation strategy for the aspect category sentiment analysis (ACSA) task that preserves the original sentence semantics and has linguistic diversity, specifically by providing a structured prompt template for an LLM to generate predefined content. In addition, we employ a post-processing technique to further ensure semantic consistency between the generated sentence and the original sentence. The augmented data increases the semantic coverage of the training distribution, enabling the model better to understand the relationship between aspect categories and sentiment polarities, enhancing its inference capabilities. Furthermore, we propose a confidence-weighted fine-tuning strategy to encourage the model to generate more confident and accurate sentiment polarity predictions. Compared with powerful and recent works, our method consistently achieves the best performance on four benchmark datasets over all baselines. 

---
# Prompting Science Report 2: The Decreasing Value of Chain of Thought in Prompting 

**Authors**: Lennart Meincke, Ethan Mollick, Lilach Mollick, Dan Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2506.07142)  

**Abstract**: This is the second in a series of short reports that seek to help business, education, and policy leaders understand the technical details of working with AI through rigorous testing. In this report, we investigate Chain-of-Thought (CoT) prompting, a technique that encourages a large language model (LLM) to "think step by step" (Wei et al., 2022). CoT is a widely adopted method for improving reasoning tasks, however, our findings reveal a more nuanced picture of its effectiveness. We demonstrate two things:
- The effectiveness of Chain-of-Thought prompting can vary greatly depending on the type of task and model. For non-reasoning models, CoT generally improves average performance by a small amount, particularly if the model does not inherently engage in step-by-step processing by default. However, CoT can introduce more variability in answers, sometimes triggering occasional errors in questions the model would otherwise get right. We also found that many recent models perform some form of CoT reasoning even if not asked; for these models, a request to perform CoT had little impact. Performing CoT generally requires far more tokens (increasing cost and time) than direct answers.
- For models designed with explicit reasoning capabilities, CoT prompting often results in only marginal, if any, gains in answer accuracy. However, it significantly increases the time and tokens needed to generate a response. 

---
# Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models 

**Authors**: Samir Abdaljalil, Hasan Kurban, Khalid Qaraqe, Erchin Serpedin  

**Link**: [PDF](https://arxiv.org/pdf/2506.07106)  

**Abstract**: Large language models (LLMs) have shown strong performance across natural language reasoning tasks, yet their reasoning processes remain brittle and difficult to interpret. Prompting techniques like Chain-of-Thought (CoT) enhance reliability by eliciting intermediate reasoning steps or aggregating multiple outputs. However, they lack mechanisms for enforcing logical structure and assessing internal coherence. We introduce Theorem-of-Thought (ToTh), a novel framework that models reasoning as collaboration among three parallel agents, each simulating a distinct mode of inference: abductive, deductive, and inductive. Each agent produces a reasoning trace, which is structured into a formal reasoning graph. To evaluate consistency, we apply Bayesian belief propagation guided by natural language inference (NLI), assigning confidence scores to each step. The most coherent graph is selected to derive the final answer. Experiments on symbolic (WebOfLies) and numerical (MultiArith) reasoning benchmarks show that ToTh consistently outperforms CoT, Self-Consistency, and CoT-Decoding across multiple LLMs, while producing interpretable and logically grounded reasoning chains. Our findings suggest a promising direction for building more robust and cognitively inspired LLM reasoning. The implementation is available at this https URL. 

---
# How Far Are We from Optimal Reasoning Efficiency? 

**Authors**: Jiaxuan Gao, Shu Yan, Qixin Tan, Lu Yang, Shusheng Xu, Wei Fu, Zhiyu Mei, Kaifeng Lyu, Yi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07104)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate remarkable problem-solving capabilities through extended Chain-of-Thought (CoT) reasoning but often produce excessively verbose and redundant reasoning traces. This inefficiency incurs high inference costs and limits practical deployment. While existing fine-tuning methods aim to improve reasoning efficiency, assessing their efficiency gains remains challenging due to inconsistent evaluations. In this work, we introduce the reasoning efficiency frontiers, empirical upper bounds derived from fine-tuning base LRMs across diverse approaches and training configurations. Based on these frontiers, we propose the Reasoning Efficiency Gap (REG), a unified metric quantifying deviations of any fine-tuned LRMs from these frontiers. Systematic evaluation on challenging mathematical benchmarks reveals significant gaps in current methods: they either sacrifice accuracy for short length or still remain inefficient under tight token budgets. To reduce the efficiency gap, we propose REO-RL, a class of Reinforcement Learning algorithms that minimizes REG by targeting a sparse set of token budgets. Leveraging numerical integration over strategically selected budgets, REO-RL approximates the full efficiency objective with low error using a small set of token budgets. Through systematic benchmarking, we demonstrate that our efficiency metric, REG, effectively captures the accuracy-length trade-off, with low-REG methods reducing length while maintaining accuracy. Our approach, REO-RL, consistently reduces REG by >=50 across all evaluated LRMs and matching Qwen3-4B/8B efficiency frontiers under a 16K token budget with minimal accuracy loss. Ablation studies confirm the effectiveness of our exponential token budget strategy. Finally, our findings highlight that fine-tuning LRMs to perfectly align with the efficiency frontiers remains an open challenge. 

---
# Representation Decomposition for Learning Similarity and Contrastness Across Modalities for Affective Computing 

**Authors**: Yuanhe Tian, Pengsen Cheng, Guoqing Jin, Lei Zhang, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.07086)  

**Abstract**: Multi-modal affective computing aims to automatically recognize and interpret human attitudes from diverse data sources such as images and text, thereby enhancing human-computer interaction and emotion understanding. Existing approaches typically rely on unimodal analysis or straightforward fusion of cross-modal information that fail to capture complex and conflicting evidence presented across different modalities. In this paper, we propose a novel LLM-based approach for affective computing that explicitly deconstructs visual and textual representations into shared (modality-invariant) and modality-specific components. Specifically, our approach firstly encodes and aligns input modalities using pre-trained multi-modal encoders, then employs a representation decomposition framework to separate common emotional content from unique cues, and finally integrates these decomposed signals via an attention mechanism to form a dynamic soft prompt for a multi-modal LLM. Extensive experiments on three representative tasks for affective computing, namely, multi-modal aspect-based sentiment analysis, multi-modal emotion analysis, and hateful meme detection, demonstrate the effectiveness of our approach, which consistently outperforms strong baselines and state-of-the-art models. 

---
# Com$^2$: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models 

**Authors**: Kai Xiong, Xiao Ding, Yixin Cao, Yuxiong Yan, Li Du, Yufei Zhang, Jinglong Gao, Jiaqian Liu, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07064)  

**Abstract**: Large language models (LLMs) have mastered abundant simple and explicit commonsense knowledge through pre-training, enabling them to achieve human-like performance in simple commonsense reasoning. Nevertheless, LLMs struggle to reason with complex and implicit commonsense knowledge that is derived from simple ones (such as understanding the long-term effects of certain events), an aspect humans tend to focus on more. Existing works focus on complex tasks like math and code, while complex commonsense reasoning remains underexplored due to its uncertainty and lack of structure. To fill this gap and align with real-world concerns, we propose a benchmark Com$^2$ focusing on complex commonsense reasoning. We first incorporate causal event graphs to serve as structured complex commonsense. Then we adopt causal theory~(e.g., intervention) to modify the causal event graphs and obtain different scenarios that meet human concerns. Finally, an LLM is employed to synthesize examples with slow thinking, which is guided by the logical relationships in the modified causal graphs. Furthermore, we use detective stories to construct a more challenging subset. Experiments show that LLMs struggle in reasoning depth and breadth, while post-training and slow thinking can alleviate this. The code and data are available at this https URL. 

---
# Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning 

**Authors**: LASA Team, Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, Yu Sun, Junao Shen, Chaojun Wang, Jie Tan, Deli Zhao, Tingyang Xu, Hao Zhang, Yu Rong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07044)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in understanding common visual elements, largely due to their large-scale datasets and advanced training strategies. However, their effectiveness in medical applications remains limited due to the inherent discrepancies between data and tasks in medical scenarios and those in the general domain. Concretely, existing medical MLLMs face the following critical limitations: (1) limited coverage of medical knowledge beyond imaging, (2) heightened susceptibility to hallucinations due to suboptimal data curation processes, (3) lack of reasoning capabilities tailored for complex medical scenarios. To address these challenges, we first propose a comprehensive data curation procedure that (1) efficiently acquires rich medical knowledge data not only from medical imaging but also from extensive medical texts and general-domain data; and (2) synthesizes accurate medical captions, visual question answering (VQA), and reasoning samples. As a result, we build a multimodal dataset enriched with extensive medical knowledge. Building on the curated data, we introduce our medical-specialized MLLM: Lingshu. Lingshu undergoes multi-stage training to embed medical expertise and enhance its task-solving capabilities progressively. Besides, we preliminarily explore the potential of applying reinforcement learning with verifiable rewards paradigm to enhance Lingshu's medical reasoning ability. Additionally, we develop MedEvalKit, a unified evaluation framework that consolidates leading multimodal and textual medical benchmarks for standardized, fair, and efficient model assessment. We evaluate the performance of Lingshu on three fundamental medical tasks, multimodal QA, text-based QA, and medical report generation. The results show that Lingshu consistently outperforms the existing open-source multimodal models on most tasks ... 

---
# Reasoning with RAGged events: RAG-Enhanced Event Knowledge Base Construction and reasoning with proof-assistants 

**Authors**: Stergios Chatzikyriakidis  

**Link**: [PDF](https://arxiv.org/pdf/2506.07042)  

**Abstract**: Extracting structured computational representations of historical events from narrative text remains computationally expensive when constructed manually. While RDF/OWL reasoners enable graph-based reasoning, they are limited to fragments of first-order logic, preventing deeper temporal and semantic analysis. This paper addresses both challenges by developing automatic historical event extraction models using multiple LLMs (GPT-4, Claude, Llama 3.2) with three enhancement strategies: pure base generation, knowledge graph enhancement, and Retrieval-Augmented Generation (RAG). We conducted comprehensive evaluations using historical texts from Thucydides. Our findings reveal that enhancement strategies optimize different performance dimensions rather than providing universal improvements. For coverage and historical breadth, base generation achieves optimal performance with Claude and GPT-4 extracting comprehensive events. However, for precision, RAG enhancement improves coordinate accuracy and metadata completeness. Model architecture fundamentally determines enhancement sensitivity: larger models demonstrate robust baseline performance with incremental RAG improvements, while Llama 3.2 shows extreme variance from competitive performance to complete failure. We then developed an automated translation pipeline converting extracted RDF representations into Coq proof assistant specifications, enabling higher-order reasoning beyond RDF capabilities including multi-step causal verification, temporal arithmetic with BC dates, and formal proofs about historical causation. The Coq formalization validates that RAG-discovered event types represent legitimate domain-specific semantic structures rather than ontological violations. 

---
# KG2QA: Knowledge Graph-enhanced Retrieval-Augmented Generation for Communication Standards Question Answering 

**Authors**: Zhongze Luo, Weixuan Wan, Qizhi Zheng, Yanhong Bai, Jingyun Sun, Jian Wang, Dan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07037)  

**Abstract**: There are many types of standards in the field of communication. The traditional consulting model has a long cycle and relies on the knowledge and experience of experts, making it difficult to meet the rapidly developing technological demands. This paper combines the fine-tuning of large language models with the construction of knowledge graphs to implement an intelligent consultation and question-answering system for communication standards. The experimental results show that after LoRA tuning on the constructed dataset of 6,587 questions and answers in the field of communication standards, Qwen2.5-7B-Instruct demonstrates outstanding professional capabilities in the field of communication standards on the test set. BLEU-4 rose from 18.8564 to 66.8993, and evaluation indicators such as ROUGE also increased significantly, outperforming the fine-tuning effect of the comparison model Llama-3-8B-Instruct. Based on the ontology framework containing 6 entity attributes and 10 relation attributes, a knowledge graph of the communication standard domain containing 13,906 entities and 13,524 relations was constructed, showing a relatively good query accuracy rate. The intelligent consultation and question-answering system enables the fine-tuned model on the server side to access the locally constructed knowledge graph and conduct graphical retrieval of key information first, which is conducive to improving the question-answering effect. The evaluation using DeepSeek as the Judge on the test set shows that our RAG framework enables the fine-tuned model to improve the scores at all five angles, with an average score increase of 2.26%. And combined with web services and API interfaces, it has achieved very good results in terms of interaction experience and back-end access, and has very good practical application value. 

---
# A Culturally-diverse Multilingual Multimodal Video Benchmark & Model 

**Authors**: Bhuiyan Sanjid Shafique, Ashmal Vayani, Muhammad Maaz, Hanoona Abdul Rasheed, Dinura Dissanayake, Mohammed Irfan Kurpath, Yahya Hmaiti, Go Inoue, Jean Lahoud, Md. Safirur Rashid, Shadid Intisar Quasem, Maheen Fatima, Franco Vidal, Mykola Maslych, Ketan Pravin More, Sanoojan Baliah, Hasindri Watawana, Yuhao Li, Fabian Farestam, Leon Schaller, Roman Tymtsiv, Simon Weber, Hisham Cholakkal, Ivan Laptev, Shin'ichi Satoh, Michael Felsberg, Mubarak Shah, Salman Khan, Fahad Shahbaz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07032)  

**Abstract**: Large multimodal models (LMMs) have recently gained attention due to their effectiveness to understand and generate descriptions of visual content. Most existing LMMs are in English language. While few recent works explore multilingual image LMMs, to the best of our knowledge, moving beyond the English language for cultural and linguistic inclusivity is yet to be investigated in the context of video LMMs. In pursuit of more inclusive video LMMs, we introduce a multilingual Video LMM benchmark, named ViMUL-Bench, to evaluate Video LMMs across 14 languages, including both low- and high-resource languages: English, Chinese, Spanish, French, German, Hindi, Arabic, Russian, Bengali, Urdu, Sinhala, Tamil, Swedish, and Japanese. Our ViMUL-Bench is designed to rigorously test video LMMs across 15 categories including eight culturally diverse categories, ranging from lifestyles and festivals to foods and rituals and from local landmarks to prominent cultural personalities. ViMUL-Bench comprises both open-ended (short and long-form) and multiple-choice questions spanning various video durations (short, medium, and long) with 8k samples that are manually verified by native language speakers. In addition, we also introduce a machine translated multilingual video training set comprising 1.2 million samples and develop a simple multilingual video LMM, named ViMUL, that is shown to provide a better tradeoff between high-and low-resource languages for video understanding. We hope our ViMUL-Bench and multilingual video LMM along with a large-scale multilingual video training set will help ease future research in developing cultural and linguistic inclusive multilingual video LMMs. Our proposed benchmark, video LMM and training data will be publicly released at this https URL. 

---
# Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text 

**Authors**: Yize Cheng, Vinu Sankar Sadasivan, Mehrdad Saberi, Shoumik Saha, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07001)  

**Abstract**: The increasing capabilities of Large Language Models (LLMs) have raised concerns about their misuse in AI-generated plagiarism and social engineering. While various AI-generated text detectors have been proposed to mitigate these risks, many remain vulnerable to simple evasion techniques such as paraphrasing. However, recent detectors have shown greater robustness against such basic attacks. In this work, we introduce Adversarial Paraphrasing, a training-free attack framework that universally humanizes any AI-generated text to evade detection more effectively. Our approach leverages an off-the-shelf instruction-following LLM to paraphrase AI-generated content under the guidance of an AI text detector, producing adversarial examples that are specifically optimized to bypass detection. Extensive experiments show that our attack is both broadly effective and highly transferable across several detection systems. For instance, compared to simple paraphrasing attack--which, ironically, increases the true positive at 1% false positive (T@1%F) by 8.57% on RADAR and 15.03% on Fast-DetectGPT--adversarial paraphrasing, guided by OpenAI-RoBERTa-Large, reduces T@1%F by 64.49% on RADAR and a striking 98.96% on Fast-DetectGPT. Across a diverse set of detectors--including neural network-based, watermark-based, and zero-shot approaches--our attack achieves an average T@1%F reduction of 87.88% under the guidance of OpenAI-RoBERTa-Large. We also analyze the tradeoff between text quality and attack success to find that our method can significantly reduce detection rates, with mostly a slight degradation in text quality. Our adversarial setup highlights the need for more robust and resilient detection strategies in the light of increasingly sophisticated evasion techniques. 

---
# What makes Reasoning Models Different? Follow the Reasoning Leader for Efficient Decoding 

**Authors**: Ming Li, Zhengyuan Yang, Xiyao Wang, Dianqi Li, Kevin Lin, Tianyi Zhou, Lijuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06998)  

**Abstract**: Large reasoning models (LRMs) achieve strong reasoning performance by emitting long chains of thought. Yet, these verbose traces slow down inference and often drift into unnecessary detail, known as the overthinking phenomenon. To better understand LRMs' behavior, we systematically analyze the token-level misalignment between reasoning and non-reasoning models. While it is expected that their primary difference lies in the stylistic "thinking cues", LRMs uniquely exhibit two pivotal, previously under-explored phenomena: a Global Misalignment Rebound, where their divergence from non-reasoning models persists or even grows as response length increases, and more critically, a Local Misalignment Diminish, where the misalignment concentrates at the "thinking cues" each sentence starts with but rapidly declines in the remaining of the sentence. Motivated by the Local Misalignment Diminish, we propose FoReaL-Decoding, a collaborative fast-slow thinking decoding method for cost-quality trade-off. In FoReaL-Decoding, a Leading model leads the first few tokens for each sentence, and then a weaker draft model completes the following tokens to the end of each sentence. FoReaL-Decoding adopts a stochastic gate to smoothly interpolate between the small and the large model. On four popular math-reasoning benchmarks (AIME24, GPQA-Diamond, MATH500, AMC23), FoReaL-Decoding reduces theoretical FLOPs by 30 to 50% and trims CoT length by up to 40%, while preserving 86 to 100% of model performance. These results establish FoReaL-Decoding as a simple, plug-and-play route to controllable cost-quality trade-offs in reasoning-centric tasks. 

---
# Cultural Bias Matters: A Cross-Cultural Benchmark Dataset and Sentiment-Enriched Model for Understanding Multimodal Metaphors 

**Authors**: Senqi Yang, Dongyu Zhang, Jing Ren, Ziqi Xu, Xiuzhen Zhang, Yiliao Song, Hongfei Lin, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2506.06987)  

**Abstract**: Metaphors are pervasive in communication, making them crucial for natural language processing (NLP). Previous research on automatic metaphor processing predominantly relies on training data consisting of English samples, which often reflect Western European or North American biases. This cultural skew can lead to an overestimation of model performance and contributions to NLP progress. However, the impact of cultural bias on metaphor processing, particularly in multimodal contexts, remains largely unexplored. To address this gap, we introduce MultiMM, a Multicultural Multimodal Metaphor dataset designed for cross-cultural studies of metaphor in Chinese and English. MultiMM consists of 8,461 text-image advertisement pairs, each accompanied by fine-grained annotations, providing a deeper understanding of multimodal metaphors beyond a single cultural domain. Additionally, we propose Sentiment-Enriched Metaphor Detection (SEMD), a baseline model that integrates sentiment embeddings to enhance metaphor comprehension across cultural backgrounds. Experimental results validate the effectiveness of SEMD on metaphor detection and sentiment analysis tasks. We hope this work increases awareness of cultural bias in NLP research and contributes to the development of fairer and more inclusive language models. Our dataset and code are available at this https URL. 

---
# Chain of Methodologies: Scaling Test Time Computation without Training 

**Authors**: Cong Liu, Jie Wu, Weigang Wu, Xu Chen, Liang Lin, Wei-Shi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06982)  

**Abstract**: Large Language Models (LLMs) often struggle with complex reasoning tasks due to insufficient in-depth insights in their training data, which are typically absent in publicly available documents. This paper introduces the Chain of Methodologies (CoM), an innovative and intuitive prompting framework that enhances structured thinking by integrating human methodological insights, enabling LLMs to tackle complex tasks with extended reasoning. CoM leverages the metacognitive abilities of advanced LLMs, activating systematic reasoning throught user-defined methodologies without explicit fine-tuning. Experiments show that CoM surpasses competitive baselines, demonstrating the potential of training-free prompting methods as robust solutions for complex reasoning tasks and bridging the gap toward human-level reasoning through human-like methodological insights. 

---
# Atomic Reasoning for Scientific Table Claim Verification 

**Authors**: Yuji Zhang, Qingyun Wang, Cheng Qian, Jiateng Liu, Chenkai Sun, Denghui Zhang, Tarek Abdelzaher, Chengxiang Zhai, Preslav Nakov, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.06972)  

**Abstract**: Scientific texts often convey authority due to their technical language and complex data. However, this complexity can sometimes lead to the spread of misinformation. Non-experts are particularly susceptible to misleading claims based on scientific tables due to their high information density and perceived credibility. Existing table claim verification models, including state-of-the-art large language models (LLMs), often struggle with precise fine-grained reasoning, resulting in errors and a lack of precision in verifying scientific claims. Inspired by Cognitive Load Theory, we propose that enhancing a model's ability to interpret table-based claims involves reducing cognitive load by developing modular, reusable reasoning components (i.e., atomic skills). We introduce a skill-chaining schema that dynamically composes these skills to facilitate more accurate and generalizable reasoning with a reduced cognitive load. To evaluate this, we create SciAtomicBench, a cross-domain benchmark with fine-grained reasoning annotations. With only 350 fine-tuning examples, our model trained by atomic reasoning outperforms GPT-4o's chain-of-thought method, achieving state-of-the-art results with far less training data. 

---
# Break-The-Chain: Reasoning Failures in LLMs via Adversarial Prompting in Code Generation 

**Authors**: Jaechul Roh, Varun Gandhi, Shivani Anilkumar, Arin Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.06971)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in tasks requiring complex reasoning, such as code generation, mathematical problem solving, and algorithmic synthesis -- especially when aided by reasoning tokens and Chain-of-Thought prompting. Yet, a core question remains: do these models truly reason, or do they merely exploit shallow statistical patterns? In this paper, we systematically investigate the robustness of reasoning LLMs by introducing a suite of semantically faithful yet adversarially structured prompt perturbations. Our evaluation -- spanning 700 perturbed code generations derived from LeetCode-style problems -- applies transformations such as storytelling reframing, irrelevant constraint injection, example reordering, and numeric perturbation. We observe that while certain modifications severely degrade performance (with accuracy drops up to -42.1%), others surprisingly improve model accuracy by up to 35.3%, suggesting sensitivity not only to semantics but also to surface-level prompt dynamics. These findings expose the fragility and unpredictability of current reasoning systems, underscoring the need for more principles approaches to reasoning alignments and prompting robustness. We release our perturbation datasets and evaluation framework to promote further research in trustworthy and resilient LLM reasoning. 

---
# A dependently-typed calculus of event telicity and culminativity 

**Authors**: Pavel Kovalev, Carlo Angiuli  

**Link**: [PDF](https://arxiv.org/pdf/2506.06968)  

**Abstract**: We present a dependently-typed cross-linguistic framework for analyzing the telicity and culminativity of events, accompanied by examples of using our framework to model English sentences. Our framework consists of two parts. In the nominal domain, we model the boundedness of noun phrases and its relationship to subtyping, delimited quantities, and adjectival modification. In the verbal domain we define a dependent event calculus, modeling telic events as those whose undergoer is bounded, culminating events as telic events that achieve their inherent endpoint, and consider adverbial modification. In both domains we pay particular attention to associated entailments. Our framework is defined as an extension of intensional Martin-Löf dependent type theory, and the rules and examples in this paper have been formalized in the Agda proof assistant. 

---
# Learning to Clarify by Reinforcement Learning Through Reward-Weighted Fine-Tuning 

**Authors**: Subhojyoti Mukherjee, Viet Dac Lai, Raghavendra Addanki, Ryan Rossi, Seunghyun Yoon, Trung Bui, Anup Rao, Jayakumar Subramanian, Branislav Kveton  

**Link**: [PDF](https://arxiv.org/pdf/2506.06964)  

**Abstract**: Question answering (QA) agents automatically answer questions posed in natural language. In this work, we learn to ask clarifying questions in QA agents. The key idea in our method is to simulate conversations that contain clarifying questions and learn from them using reinforcement learning (RL). To make RL practical, we propose and analyze offline RL objectives that can be viewed as reward-weighted supervised fine-tuning (SFT) and easily optimized in large language models. Our work stands in a stark contrast to recently proposed methods, based on SFT and direct preference optimization, which have additional hyper-parameters and do not directly optimize rewards. We compare to these methods empirically and report gains in both optimized rewards and language quality. 

---
# BIS Reasoning 1.0: The First Large-Scale Japanese Benchmark for Belief-Inconsistent Syllogistic Reasoning 

**Authors**: Ha-Thanh Nguyen, Chaoran Liu, Hirokazu Kiyomaru, Koichi Takeda, Yusuke Miyao, Maki Matsuda, Yusuke Oda, Pontus Stenetorp, Qianying Liu, Su Myat Noe, Hideyuki Tachibana, Kouta Nakayama, Sadao Kurohashi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06955)  

**Abstract**: We present BIS Reasoning 1.0, the first large-scale Japanese dataset of syllogistic reasoning problems explicitly designed to evaluate belief-inconsistent reasoning in large language models (LLMs). Unlike prior datasets such as NeuBAROCO and JFLD, which focus on general or belief-aligned reasoning, BIS Reasoning 1.0 introduces logically valid yet belief-inconsistent syllogisms to uncover reasoning biases in LLMs trained on human-aligned corpora. We benchmark state-of-the-art models - including GPT models, Claude models, and leading Japanese LLMs - revealing significant variance in performance, with GPT-4o achieving 79.54% accuracy. Our analysis identifies critical weaknesses in current LLMs when handling logically valid but belief-conflicting inputs. These findings have important implications for deploying LLMs in high-stakes domains such as law, healthcare, and scientific literature, where truth must override intuitive belief to ensure integrity and safety. 

---
# What Makes a Good Natural Language Prompt? 

**Authors**: Do Xuan Long, Duy Dinh, Ngoc-Hai Nguyen, Kenji Kawaguchi, Nancy F. Chen, Shafiq Joty, Min-Yen Kan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06950)  

**Abstract**: As large language models (LLMs) have progressed towards more human-like and human--AI communications have become prevalent, prompting has emerged as a decisive component. However, there is limited conceptual consensus on what exactly quantifies natural language prompts. We attempt to address this question by conducting a meta-analysis surveying more than 150 prompting-related papers from leading NLP and AI conferences from 2022 to 2025 and blogs. We propose a property- and human-centric framework for evaluating prompt quality, encompassing 21 properties categorized into six dimensions. We then examine how existing studies assess their impact on LLMs, revealing their imbalanced support across models and tasks, and substantial research gaps. Further, we analyze correlations among properties in high-quality natural language prompts, deriving prompting recommendations. We then empirically explore multi-property prompt enhancements in reasoning tasks, observing that single-property enhancements often have the greatest impact. Finally, we discover that instruction-tuning on property-enhanced prompts can result in better reasoning models. Our findings establish a foundation for property-centric prompt evaluation and optimization, bridging the gaps between human--AI communication and opening new prompting research directions. 

---
# DiscoSum: Discourse-aware News Summarization 

**Authors**: Alexander Spangher, Tenghao Huang, Jialiang Gu, Jiatong Shi, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06930)  

**Abstract**: Recent advances in text summarization have predominantly leveraged large language models to generate concise summaries. However, language models often do not maintain long-term discourse structure, especially in news articles, where organizational flow significantly influences reader engagement. We introduce a novel approach to integrating discourse structure into summarization processes, focusing specifically on news articles across various media. We present a novel summarization dataset where news articles are summarized multiple times in different ways across different social media platforms (e.g. LinkedIn, Facebook, etc.). We develop a novel news discourse schema to describe summarization structures and a novel algorithm, DiscoSum, which employs beam search technique for structure-aware summarization, enabling the transformation of news stories to meet different stylistic and structural demands. Both human and automatic evaluation results demonstrate the efficacy of our approach in maintaining narrative fidelity and meeting structural requirements. 

---
# Hybrid Extractive Abstractive Summarization for Multilingual Sentiment Analysis 

**Authors**: Mikhail Krasitskii, Grigori Sidorov, Olga Kolesnikova, Liliana Chanona Hernandez, Alexander Gelbukh  

**Link**: [PDF](https://arxiv.org/pdf/2506.06929)  

**Abstract**: We propose a hybrid approach for multilingual sentiment analysis that combines extractive and abstractive summarization to address the limitations of standalone methods. The model integrates TF-IDF-based extraction with a fine-tuned XLM-R abstractive module, enhanced by dynamic thresholding and cultural adaptation. Experiments across 10 languages show significant improvements over baselines, achieving 0.90 accuracy for English and 0.84 for low-resource languages. The approach also demonstrates 22% greater computational efficiency than traditional methods. Practical applications include real-time brand monitoring and cross-cultural discourse analysis. Future work will focus on optimization for low-resource languages via 8-bit quantization. 

---
# Automatic Speech Recognition of African American English: Lexical and Contextual Effects 

**Authors**: Hamid Mojarad, Kevin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06888)  

**Abstract**: Automatic Speech Recognition (ASR) models often struggle with the phonetic, phonological, and morphosyntactic features found in African American English (AAE). This study focuses on two key AAE variables: Consonant Cluster Reduction (CCR) and ING-reduction. It examines whether the presence of CCR and ING-reduction increases ASR misrecognition. Subsequently, it investigates whether end-to-end ASR systems without an external Language Model (LM) are more influenced by lexical neighborhood effect and less by contextual predictability compared to systems with an LM. The Corpus of Regional African American Language (CORAAL) was transcribed using wav2vec 2.0 with and without an LM. CCR and ING-reduction were detected using the Montreal Forced Aligner (MFA) with pronunciation expansion. The analysis reveals a small but significant effect of CCR and ING on Word Error Rate (WER) and indicates a stronger presence of lexical neighborhood effect in ASR systems without LMs. 

---
# Mixture of Small and Large Models for Chinese Spelling Check 

**Authors**: Ziheng Qiao, Houquan Zhou, Zhenghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06887)  

**Abstract**: In the era of large language models (LLMs), the Chinese Spelling Check (CSC) task has seen various LLM methods developed, yet their performance remains unsatisfactory. In contrast, fine-tuned BERT-based models, relying on high-quality in-domain data, show excellent performance but suffer from edit pattern overfitting. This paper proposes a novel dynamic mixture approach that effectively combines the probability distributions of small models and LLMs during the beam search decoding phase, achieving a balanced enhancement of precise corrections from small models and the fluency of LLMs. This approach also eliminates the need for fine-tuning LLMs, saving significant time and resources, and facilitating domain adaptation. Comprehensive experiments demonstrate that our mixture approach significantly boosts error correction capabilities, achieving state-of-the-art results across multiple datasets. Our code is available at this https URL. 

---
# Right Is Not Enough: The Pitfalls of Outcome Supervision in Training LLMs for Math Reasoning 

**Authors**: Jiaxing Guo, Wenjie Yang, Shengzhong Zhang, Tongshan Xu, Lun Du, Da Zheng, Zengfeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06877)  

**Abstract**: Outcome-rewarded Large Language Models (LLMs) have demonstrated remarkable success in mathematical problem-solving. However, this success often masks a critical issue: models frequently achieve correct answers through fundamentally unsound reasoning processes, a phenomenon indicative of reward hacking. We introduce MathOlympiadEval, a new dataset with fine-grained annotations, which reveals a significant gap between LLMs' answer correctness and their low process correctness. Existing automated methods like LLM-as-a-judge struggle to reliably detect these reasoning flaws. To address this, we propose ParaStepVerifier, a novel methodology for meticulous, step-by-step verification of mathematical solutions. ParaStepVerifier identifies incorrect reasoning steps. Empirical results demonstrate that ParaStepVerifier substantially improves the accuracy of identifying flawed solutions compared to baselines, especially for complex, multi-step problems. This offers a more robust path towards evaluating and training LLMs with genuine mathematical reasoning. 

---
# Adapt Once, Thrive with Updates: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models 

**Authors**: Naibin Gu, Peng Fu, Xiyu Liu, Ke Ma, Zheng Lin, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06844)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) has become a common method for fine-tuning large language models, where a base model can serve multiple users through PEFT module switching. To enhance user experience, base models require periodic updates. However, once updated, PEFT modules fine-tuned on previous versions often suffer substantial performance degradation on newer versions. Re-tuning these numerous modules to restore performance would incur significant computational costs. Through a comprehensive analysis of the changes that occur during base model updates, we uncover an interesting phenomenon: continual training primarily affects task-specific knowledge stored in Feed-Forward Networks (FFN), while having less impact on the task-specific pattern in the Attention mechanism. Based on these findings, we introduce Trans-PEFT, a novel approach that enhances the PEFT module by focusing on the task-specific pattern while reducing its dependence on certain knowledge in the base model. Further theoretical analysis supports our approach. Extensive experiments across 7 base models and 12 datasets demonstrate that Trans-PEFT trained modules can maintain performance on updated base models without re-tuning, significantly reducing maintenance overhead in real-world applications. 

---
# PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation 

**Authors**: Arkadiusz Modzelewski, Witold Sosnowski, Tiziano Labruna, Adam Wierzbicki, Giovanni Da San Martino  

**Link**: [PDF](https://arxiv.org/pdf/2506.06842)  

**Abstract**: Disinformation detection is a key aspect of media literacy. Psychological studies have shown that knowledge of persuasive fallacies helps individuals detect disinformation. Inspired by these findings, we experimented with large language models (LLMs) to test whether infusing persuasion knowledge enhances disinformation detection. As a result, we introduce the Persuasion-Augmented Chain of Thought (PCoT), a novel approach that leverages persuasion to improve disinformation detection in zero-shot classification. We extensively evaluate PCoT on online news and social media posts. Moreover, we publish two novel, up-to-date disinformation datasets: EUDisinfo and MultiDis. These datasets enable the evaluation of PCoT on content entirely unseen by the LLMs used in our experiments, as the content was published after the models' knowledge cutoffs. We show that, on average, PCoT outperforms competitive methods by 15% across five LLMs and five datasets. These findings highlight the value of persuasion in strengthening zero-shot disinformation detection. 

---
# Can LLMs Generate Reliable Test Case Generators? A Study on Competition-Level Programming Problems 

**Authors**: Yuhan Cao, Zian Chen, Kun Quan, Ziliang Zhang, Yu Wang, Xiaoning Dong, Yeqi Feng, Guanzhong He, Jingcheng Huang, Jianhao Li, Yixuan Tan, Jiafu Tang, Yilin Tang, Junlei Wu, Qianyu Xiao, Can Zheng, Shouchen Zhou, Yuxiang Zhu, Yiming Huang, Tian Xie, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2506.06821)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, capable of tackling complex tasks during inference. However, the extent to which LLMs can be utilized for code checking or debugging through test case generation remains largely unexplored. We investigate this problem from the perspective of competition-level programming (CP) programs and propose TCGBench, a Benchmark for (LLM generation of) Test Case Generators. This benchmark comprises two tasks, aimed at studying the capabilities of LLMs in (1) generating valid test case generators for a given CP problem, and further (2) generating targeted test case generators that expose bugs in human-written code. Experimental results indicate that while state-of-the-art LLMs can generate valid test case generators in most cases, most LLMs struggle to generate targeted test cases that reveal flaws in human code effectively. Especially, even advanced reasoning models (e.g., o3-mini) fall significantly short of human performance in the task of generating targeted generators. Furthermore, we construct a high-quality, manually curated dataset of instructions for generating targeted generators. Analysis demonstrates that the performance of LLMs can be enhanced with the aid of this dataset, by both prompting and fine-tuning. 

---
# Beyond Classification: Towards Speech Emotion Reasoning with Multitask AudioLLMs 

**Authors**: Wenyu Zhang, Yingxu He, Geyu Lin, Zhuohan Liu, Shuo Sun, Bin Wang, Xunlong Zou, Jeremy H. M. Wong, Qiongqiong Wang, Hardik B. Sailor, Nancy F. Chen, Ai Ti Aw  

**Link**: [PDF](https://arxiv.org/pdf/2506.06820)  

**Abstract**: Audio Large Language Models (AudioLLMs) have achieved strong results in semantic tasks like speech recognition and translation, but remain limited in modeling paralinguistic cues such as emotion. Existing approaches often treat emotion understanding as a classification problem, offering little insight into the underlying rationale behind predictions. In this work, we explore emotion reasoning, a strategy that leverages the generative capabilities of AudioLLMs to enhance emotion recognition by producing semantically aligned, evidence-grounded explanations. To support this in multitask AudioLLMs, we introduce a unified framework combining reasoning-augmented data supervision, dual-encoder architecture, and task-alternating training. This approach enables AudioLLMs to effectively learn different tasks while incorporating emotional reasoning. Experiments on IEMOCAP and MELD show that our approach not only improves emotion prediction accuracy but also enhances the coherence and evidential grounding of the generated responses. 

---
# How do datasets, developers, and models affect biases in a low-resourced language? 

**Authors**: Dipto Das, Shion Guha, Bryan Semaan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06816)  

**Abstract**: Sociotechnical systems, such as language technologies, frequently exhibit identity-based biases. These biases exacerbate the experiences of historically marginalized communities and remain understudied in low-resource contexts. While models and datasets specific to a language or with multilingual support are commonly recommended to address these biases, this paper empirically tests the effectiveness of such approaches in the context of gender, religion, and nationality-based identities in Bengali, a widely spoken but low-resourced language. We conducted an algorithmic audit of sentiment analysis models built on mBERT and BanglaBERT, which were fine-tuned using all Bengali sentiment analysis (BSA) datasets from Google Dataset Search. Our analyses showed that BSA models exhibit biases across different identity categories despite having similar semantic content and structure. We also examined the inconsistencies and uncertainties arising from combining pre-trained models and datasets created by individuals from diverse demographic backgrounds. We connected these findings to the broader discussions on epistemic injustice, AI alignment, and methodological decisions in algorithmic audits. 

---
# BTPD: A Multilingual Hand-curated Dataset of Bengali Transnational Political Discourse Across Online Communities 

**Authors**: Dipto Das, Syed Ishtiaque Ahmed, Shion Guha  

**Link**: [PDF](https://arxiv.org/pdf/2506.06813)  

**Abstract**: Understanding political discourse in online spaces is crucial for analyzing public opinion and ideological polarization. While social computing and computational linguistics have explored such discussions in English, such research efforts are significantly limited in major yet under-resourced languages like Bengali due to the unavailability of datasets. In this paper, we present a multilingual dataset of Bengali transnational political discourse (BTPD) collected from three online platforms, each representing distinct community structures and interaction dynamics. Besides describing how we hand-curated the dataset through community-informed keyword-based retrieval, this paper also provides a general overview of its topics and multilingual content. 

---
# Advancing Question Generation with Joint Narrative and Difficulty Control 

**Authors**: Bernardo Leite, Henrique Lopes Cardoso  

**Link**: [PDF](https://arxiv.org/pdf/2506.06812)  

**Abstract**: Question Generation (QG), the task of automatically generating questions from a source input, has seen significant progress in recent years. Difficulty-controllable QG (DCQG) enables control over the difficulty level of generated questions while considering the learner's ability. Additionally, narrative-controllable QG (NCQG) allows control over the narrative aspects embedded in the questions. However, research in QG lacks a focus on combining these two types of control, which is important for generating questions tailored to educational purposes. To address this gap, we propose a strategy for Joint Narrative and Difficulty Control, enabling simultaneous control over these two attributes in the generation of reading comprehension questions. Our evaluation provides preliminary evidence that this approach is feasible, though it is not effective across all instances. Our findings highlight the conditions under which the strategy performs well and discuss the trade-offs associated with its application. 

---
# Not quite Sherlock Holmes: Language model predictions do not reliably differentiate impossible from improbable events 

**Authors**: James A. Michaelov, Reeka Estacio, Zhien Zhang, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06808)  

**Abstract**: Can language models reliably predict that possible events are more likely than merely improbable ones? By teasing apart possibility, typicality, and contextual relatedness, we show that despite the results of previous work, language models' ability to do this is far from robust. In fact, under certain conditions, all models tested - including Llama 3, Gemma 2, and Mistral NeMo - perform at worse-than-chance level, assigning higher probabilities to impossible sentences such as 'the car was given a parking ticket by the brake' than to merely unlikely sentences such as 'the car was given a parking ticket by the explorer'. 

---
# Label-semantics Aware Generative Approach for Domain-Agnostic Multilabel Classification 

**Authors**: Subhendu Khatuya, Shashwat Naidu, Saptarshi Ghosh, Pawan Goyal, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2506.06806)  

**Abstract**: The explosion of textual data has made manual document classification increasingly challenging. To address this, we introduce a robust, efficient domain-agnostic generative model framework for multi-label text classification. Instead of treating labels as mere atomic symbols, our approach utilizes predefined label descriptions and is trained to generate these descriptions based on the input text. During inference, the generated descriptions are matched to the pre-defined labels using a finetuned sentence transformer. We integrate this with a dual-objective loss function, combining cross-entropy loss and cosine similarity of the generated sentences with the predefined target descriptions, ensuring both semantic alignment and accuracy. Our proposed model LAGAMC stands out for its parameter efficiency and versatility across diverse datasets, making it well-suited for practical applications. We demonstrate the effectiveness of our proposed model by achieving new state-of-the-art performances across all evaluated datasets, surpassing several strong baselines. We achieve improvements of 13.94% in Micro-F1 and 24.85% in Macro-F1 compared to the closest baseline across all datasets. 

---
# On the Adaptive Psychological Persuasion of Large Language Models 

**Authors**: Tianjie Ju, Yujia Chen, Hao Fei, Mong-Li Lee, Wynne Hsu, Pengzhou Cheng, Zongru Wu, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06800)  

**Abstract**: Previous work has showcased the intriguing capabilities of Large Language Models (LLMs) in instruction-following and rhetorical fluency. However, systematic exploration of their dual capabilities to autonomously persuade and resist persuasion, particularly in contexts involving psychological rhetoric, remains unexplored. In this paper, we first evaluate four commonly adopted LLMs by tasking them to alternately act as persuaders and listeners in adversarial dialogues. Empirical results show that persuader LLMs predominantly employ repetitive strategies, leading to low success rates. Then we introduce eleven comprehensive psychological persuasion strategies, finding that explicitly instructing LLMs to adopt specific strategies such as Fluency Effect and Repetition Effect significantly improves persuasion success rates. However, no ``one-size-fits-all'' strategy proves universally effective, with performance heavily dependent on contextual counterfactuals. Motivated by these observations, we propose an adaptive framework based on direct preference optimization that trains LLMs to autonomously select optimal strategies by leveraging persuasion results from strategy-specific responses as preference pairs. Experiments on three open-source LLMs confirm that the proposed adaptive psychological persuasion method effectively enables persuader LLMs to select optimal strategies, significantly enhancing their success rates while maintaining general capabilities. Our code is available at this https URL. 

---
# Extending dependencies to the taggedPBC: Word order in transitive clauses 

**Authors**: Hiram Ring  

**Link**: [PDF](https://arxiv.org/pdf/2506.06785)  

**Abstract**: The taggedPBC (Ring 2025a) contains more than 1,800 sentences of pos-tagged parallel text data from over 1,500 languages, representing 133 language families and 111 isolates. While this dwarfs previously available resources, and the POS tags achieve decent accuracy, allowing for predictive crosslinguistic insights (Ring 2025b), the dataset was not initially annotated for dependencies. This paper reports on a CoNLLU-formatted version of the dataset which transfers dependency information along with POS tags to all languages in the taggedPBC. Although there are various concerns regarding the quality of the tags and the dependencies, word order information derived from this dataset regarding the position of arguments and predicates in transitive clauses correlates with expert determinations of word order in three typological databases (WALS, Grambank, Autotyp). This highlights the usefulness of corpus-based typological approaches (as per Baylor et al. 2023; Bjerva 2024) for extending comparisons of discrete linguistic categories, and suggests that important insights can be gained even from noisy data, given sufficient annotation. The dependency-annotated corpora are also made available for research and collaboration via GitHub. 

---
# They want to pretend not to understand: The Limits of Current LLMs in Interpreting Implicit Content of Political Discourse 

**Authors**: Walter Paci, Alessandro Panunzi, Sandro Pezzelle  

**Link**: [PDF](https://arxiv.org/pdf/2506.06775)  

**Abstract**: Implicit content plays a crucial role in political discourse, where speakers systematically employ pragmatic strategies such as implicatures and presuppositions to influence their audiences. Large Language Models (LLMs) have demonstrated strong performance in tasks requiring complex semantic and pragmatic understanding, highlighting their potential for detecting and explaining the meaning of implicit content. However, their ability to do this within political discourse remains largely underexplored. Leveraging, for the first time, the large IMPAQTS corpus, which comprises Italian political speeches with the annotation of manipulative implicit content, we propose methods to test the effectiveness of LLMs in this challenging problem. Through a multiple-choice task and an open-ended generation task, we demonstrate that all tested models struggle to interpret presuppositions and implicatures. We conclude that current LLMs lack the key pragmatic capabilities necessary for accurately interpreting highly implicit language, such as that found in political discourse. At the same time, we highlight promising trends and future directions for enhancing model performance. We release our data and code at this https URL 

---
# Geopolitical biases in LLMs: what are the "good" and the "bad" countries according to contemporary language models 

**Authors**: Mikhail Salnikov, Dmitrii Korzh, Ivan Lazichny, Elvir Karimov, Artyom Iudin, Ivan Oseledets, Oleg Y. Rogov, Alexander Panchenko, Natalia Loukachevitch, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2506.06751)  

**Abstract**: This paper evaluates geopolitical biases in LLMs with respect to various countries though an analysis of their interpretation of historical events with conflicting national perspectives (USA, UK, USSR, and China). We introduce a novel dataset with neutral event descriptions and contrasting viewpoints from different countries. Our findings show significant geopolitical biases, with models favoring specific national narratives. Additionally, simple debiasing prompts had a limited effect in reducing these biases. Experiments with manipulated participant labels reveal models' sensitivity to attribution, sometimes amplifying biases or recognizing inconsistencies, especially with swapped labels. This work highlights national narrative biases in LLMs, challenges the effectiveness of simple debiasing methods, and offers a framework and dataset for future geopolitical bias research. 

---
# C-PATH: Conversational Patient Assistance and Triage in Healthcare System 

**Authors**: Qi Shi, Qiwei Han, Cláudia Soares  

**Link**: [PDF](https://arxiv.org/pdf/2506.06737)  

**Abstract**: Navigating healthcare systems can be complex and overwhelming, creating barriers for patients seeking timely and appropriate medical attention. In this paper, we introduce C-PATH (Conversational Patient Assistance and Triage in Healthcare), a novel conversational AI system powered by large language models (LLMs) designed to assist patients in recognizing symptoms and recommending appropriate medical departments through natural, multi-turn dialogues. C-PATH is fine-tuned on medical knowledge, dialogue data, and clinical summaries using a multi-stage pipeline built on the LLaMA3 architecture. A core contribution of this work is a GPT-based data augmentation framework that transforms structured clinical knowledge from DDXPlus into lay-person-friendly conversations, allowing alignment with patient communication norms. We also implement a scalable conversation history management strategy to ensure long-range coherence. Evaluation with GPTScore demonstrates strong performance across dimensions such as clarity, informativeness, and recommendation accuracy. Quantitative benchmarks show that C-PATH achieves superior performance in GPT-rewritten conversational datasets, significantly outperforming domain-specific baselines. C-PATH represents a step forward in the development of user-centric, accessible, and accurate AI tools for digital health assistance and triage. 

---
# A Survey of Retentive Network 

**Authors**: Haiqi Yang, Zhiyuan Li, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06708)  

**Abstract**: Retentive Network (RetNet) represents a significant advancement in neural network architecture, offering an efficient alternative to the Transformer. While Transformers rely on self-attention to model dependencies, they suffer from high memory costs and limited scalability when handling long sequences due to their quadratic complexity. To mitigate these limitations, RetNet introduces a retention mechanism that unifies the inductive bias of recurrence with the global dependency modeling of attention. This mechanism enables linear-time inference, facilitates efficient modeling of extended contexts, and remains compatible with fully parallelizable training pipelines. RetNet has garnered significant research interest due to its consistently demonstrated cross-domain effectiveness, achieving robust performance across machine learning paradigms including natural language processing, speech recognition, and time-series analysis. However, a comprehensive review of RetNet is still missing from the current literature. This paper aims to fill that gap by offering the first detailed survey of the RetNet architecture, its key innovations, and its diverse applications. We also explore the main challenges associated with RetNet and propose future research directions to support its continued advancement in both academic research and practical deployment. 

---
# DivScore: Zero-Shot Detection of LLM-Generated Text in Specialized Domains 

**Authors**: Zhihui Chen, Kai He, Yucheng Huang, Yunxiao Zhu, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06705)  

**Abstract**: Detecting LLM-generated text in specialized and high-stakes domains like medicine and law is crucial for combating misinformation and ensuring authenticity. However, current zero-shot detectors, while effective on general text, often fail when applied to specialized content due to domain shift. We provide a theoretical analysis showing this failure is fundamentally linked to the KL divergence between human, detector, and source text distributions. To address this, we propose DivScore, a zero-shot detection framework using normalized entropy-based scoring and domain knowledge distillation to robustly identify LLM-generated text in specialized domains. We also release a domain-specific benchmark for LLM-generated text detection in the medical and legal domains. Experiments on our benchmark show that DivScore consistently outperforms state-of-the-art detectors, with 14.4% higher AUROC and 64.0% higher recall (0.1% false positive rate threshold). In adversarial settings, DivScore demonstrates superior robustness than other baselines, achieving on average 22.8% advantage in AUROC and 29.5% in recall. Code and data are publicly available. 

---
# Dynamic and Parametric Retrieval-Augmented Generation 

**Authors**: Weihang Su, Qingyao Ai, Jingtao Zhan, Qian Dong, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06704)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a foundational paradigm for equipping large language models (LLMs) with external knowledge, playing a critical role in information retrieval and knowledge-intensive applications. However, conventional RAG systems typically adopt a static retrieve-then-generate pipeline and rely on in-context knowledge injection, which can be suboptimal for complex tasks that require multihop reasoning, adaptive information access, and deeper integration of external knowledge. Motivated by these limitations, the research community has moved beyond static retrieval and in-context knowledge injection. Among the emerging directions, this tutorial delves into two rapidly growing and complementary research areas on RAG: Dynamic RAG and Parametric RAG. Dynamic RAG adaptively determines when and what to retrieve during the LLM's generation process, enabling real-time adaptation to the LLM's evolving information needs. Parametric RAG rethinks how retrieved knowledge should be injected into LLMs, transitioning from input-level to parameter-level knowledge injection for enhanced efficiency and effectiveness. This tutorial offers a comprehensive overview of recent advances in these emerging research areas. It also shares theoretical foundations and practical insights to support and inspire further research in RAG. 

---
# Learning Distribution-Wise Control in Representation Space for Language Models 

**Authors**: Chunyuan Deng, Ruidi Chang, Hanjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06686)  

**Abstract**: Interventions in language models (LMs) are applied strategically to steer model behavior during the forward pass. Learnable interventions, also known as representation fine-tuning, aim to apply pointwise control within the concept subspace and have proven effective in altering high-level behaviors. In this work, we extend this approach to the distribution level, enabling the model to learn not only pointwise transformations but also the surrounding regions of the concept subspace. We demonstrate that these methods perform effectively in early layers, with larger standard deviations correlating strongly with improved performance. Across eight commonsense reasoning and seven arithmetic reasoning benchmarks, our distribution-wise interventions consistently outperform pointwise interventions in controllability and robustness. These results illustrate that distribution-wise interventions provide a more comprehensive method for steering model behavior and enabling finer-grained control over language models. The code is at: \href{this https URL}{this https URL}. 

---
# Quantile Regression with Large Language Models for Price Prediction 

**Authors**: Nikhita Vedula, Dushyanta Dhyani, Laleh Jalali, Boris Oreshkin, Mohsen Bayati, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06657)  

**Abstract**: Large Language Models (LLMs) have shown promise in structured prediction tasks, including regression, but existing approaches primarily focus on point estimates and lack systematic comparison across different methods. We investigate probabilistic regression using LLMs for unstructured inputs, addressing challenging text-to-distribution prediction tasks such as price estimation where both nuanced text understanding and uncertainty quantification are critical. We propose a novel quantile regression approach that enables LLMs to produce full predictive distributions, improving upon traditional point estimates. Through extensive experiments across three diverse price prediction datasets, we demonstrate that a Mistral-7B model fine-tuned with quantile heads significantly outperforms traditional approaches for both point and distributional estimations, as measured by three established metrics each for prediction accuracy and distributional calibration. Our systematic comparison of LLM approaches, model architectures, training approaches, and data scaling reveals that Mistral-7B consistently outperforms encoder architectures, embedding-based methods, and few-shot learning methods. Our experiments also reveal the effectiveness of LLM-assisted label correction in achieving human-level accuracy without systematic bias. Our curated datasets are made available at this https URL to support future research. 

---
# SafeLawBench: Towards Safe Alignment of Large Language Models 

**Authors**: Chuxue Cao, Han Zhu, Jiaming Ji, Qichao Sun, Zhenghao Zhu, Yinyu Wu, Juntao Dai, Yaodong Yang, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.06636)  

**Abstract**: With the growing prevalence of large language models (LLMs), the safety of LLMs has raised significant concerns. However, there is still a lack of definitive standards for evaluating their safety due to the subjective nature of current safety benchmarks. To address this gap, we conducted the first exploration of LLMs' safety evaluation from a legal perspective by proposing the SafeLawBench benchmark. SafeLawBench categorizes safety risks into three levels based on legal standards, providing a systematic and comprehensive framework for evaluation. It comprises 24,860 multi-choice questions and 1,106 open-domain question-answering (QA) tasks. Our evaluation included 2 closed-source LLMs and 18 open-source LLMs using zero-shot and few-shot prompting, highlighting the safety features of each model. We also evaluated the LLMs' safety-related reasoning stability and refusal behavior. Additionally, we found that a majority voting mechanism can enhance model performance. Notably, even leading SOTA models like Claude-3.5-Sonnet and GPT-4o have not exceeded 80.5% accuracy in multi-choice tasks on SafeLawBench, while the average accuracy of 20 LLMs remains at 68.8\%. We urge the community to prioritize research on the safety of LLMs. 

---
# Psychological Counseling Cannot Be Achieved Overnight: Automated Psychological Counseling Through Multi-Session Conversations 

**Authors**: Junzhe Wang, Bichen Wang, Xing Fu, Yixin Sun, Yanyan Zhao, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.06626)  

**Abstract**: In recent years, Large Language Models (LLMs) have made significant progress in automated psychological counseling. However, current research focuses on single-session counseling, which doesn't represent real-world scenarios. In practice, psychological counseling is a process, not a one-time event, requiring sustained, multi-session engagement to progressively address clients' issues. To overcome this limitation, we introduce a dataset for Multi-Session Psychological Counseling Conversation Dataset (MusPsy-Dataset). Our MusPsy-Dataset is constructed using real client profiles from publicly available psychological case reports. It captures the dynamic arc of counseling, encompassing multiple progressive counseling conversations from the same client across different sessions. Leveraging our dataset, we also developed our MusPsy-Model, which aims to track client progress and adapt its counseling direction over time. Experiments show that our model performs better than baseline models across multiple sessions. 

---
# BriefMe: A Legal NLP Benchmark for Assisting with Legal Briefs 

**Authors**: Jesse Woo, Fateme Hashemi Chaleshtori, Ana Marasović, Kenneth Marino  

**Link**: [PDF](https://arxiv.org/pdf/2506.06619)  

**Abstract**: A core part of legal work that has been under-explored in Legal NLP is the writing and editing of legal briefs. This requires not only a thorough understanding of the law of a jurisdiction, from judgments to statutes, but also the ability to make new arguments to try to expand the law in a new direction and make novel and creative arguments that are persuasive to judges. To capture and evaluate these legal skills in language models, we introduce BRIEFME, a new dataset focused on legal briefs. It contains three tasks for language models to assist legal professionals in writing briefs: argument summarization, argument completion, and case retrieval. In this work, we describe the creation of these tasks, analyze them, and show how current models perform. We see that today's large language models (LLMs) are already quite good at the summarization and guided completion tasks, even beating human-generated headings. Yet, they perform poorly on other tasks in our benchmark: realistic argument completion and retrieving relevant legal cases. We hope this dataset encourages more development in Legal NLP in ways that will specifically aid people in performing legal work. 

---
# Interpretable Depression Detection from Social Media Text Using LLM-Derived Embeddings 

**Authors**: Samuel Kim, Oghenemaro Imieye, Yunting Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.06616)  

**Abstract**: Accurate and interpretable detection of depressive language in social media is useful for early interventions of mental health conditions, and has important implications for both clinical practice and broader public health efforts. In this paper, we investigate the performance of large language models (LLMs) and traditional machine learning classifiers across three classification tasks involving social media data: binary depression classification, depression severity classification, and differential diagnosis classification among depression, PTSD, and anxiety. Our study compares zero-shot LLMs with supervised classifiers trained on both conventional text embeddings and LLM-generated summary embeddings. Our experiments reveal that while zero-shot LLMs demonstrate strong generalization capabilities in binary classification, they struggle with fine-grained ordinal classifications. In contrast, classifiers trained on summary embeddings generated by LLMs demonstrate competitive, and in some cases superior, performance on the classification tasks, particularly when compared to models using traditional text embeddings. Our findings demonstrate the strengths of LLMs in mental health prediction, and suggest promising directions for better utilization of their zero-shot capabilities and context-aware summarization techniques. 

---
# Transferring Features Across Language Models With Model Stitching 

**Authors**: Alan Chen, Jack Merullo, Alessandro Stolfo, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2506.06609)  

**Abstract**: In this work, we demonstrate that affine mappings between residual streams of language models is a cheap way to effectively transfer represented features between models. We apply this technique to transfer the weights of Sparse Autoencoders (SAEs) between models of different sizes to compare their representations. We find that small and large models learn highly similar representation spaces, which motivates training expensive components like SAEs on a smaller model and transferring to a larger model at a FLOPs savings. For example, using a small-to-large transferred SAE as initialization can lead to 50% cheaper training runs when training SAEs on larger models. Next, we show that transferred probes and steering vectors can effectively recover ground truth performance. Finally, we dive deeper into feature-level transferability, finding that semantic and structural features transfer noticeably differently while specific classes of functional features have their roles faithfully mapped. Overall, our findings illustrate similarities and differences in the linear representation spaces of small and large models and demonstrate a method for improving the training efficiency of SAEs. 

---
# Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit 

**Authors**: Charles Goddard, Fernando Fernandes Neto  

**Link**: [PDF](https://arxiv.org/pdf/2506.06607)  

**Abstract**: We present a training-free method to transplant tokenizers in pretrained large language models (LLMs) by reconstructing unseen token embeddings via Orthogonal Matching Pursuit (OMP). Specifically, we approximate each out-of-vocabulary token as a sparse linear combination of shared tokens, in two phases: first, compute each new token's representation in the donor embedding space with a small dictionary of shared anchor tokens, then transfer these same sparse coefficients back into the base model's embedding space.
On two challenging cross-tokenizer tasks--Llama$\to$Mistral NeMo (12B) and Qwen$\to$Llama (1B)--we show that OMP achieves best zero-shot preservation of the base model's performance across multiple benchmarks, while other zero-shot approaches degrade significantly. Compared to baselines (zero-init, mean-init, and existing approaches like WECHSEL, FOCUS, ZETT), OMP consistently achieves the best overall performance, effectively bridging large tokenizer discrepancies without gradient updates. Our analysis further identifies mismatched numerical tokenization schemes as a critical challenge for preserving mathematical reasoning capabilities. This technique enables direct reuse of pretrained model weights with new tokenizers, facilitating cross-tokenizer knowledge distillation, speculative decoding, ensembling, merging, and domain-specific vocabulary adaptations. We integrate our method into the open-source mergekit-tokensurgeon tool for post hoc vocabulary realignment. 

---
# MedCite: Can Language Models Generate Verifiable Text for Medicine? 

**Authors**: Xiao Wang, Mengjue Tan, Qiao Jin, Guangzhi Xiong, Yu Hu, Aidong Zhang, Zhiyong Lu, Minjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06605)  

**Abstract**: Existing LLM-based medical question-answering systems lack citation generation and evaluation capabilities, raising concerns about their adoption in practice. In this work, we introduce \name, the first end-to-end framework that facilitates the design and evaluation of citation generation with LLMs for medical tasks. Meanwhile, we introduce a novel multi-pass retrieval-citation method that generates high-quality citations. Our evaluation highlights the challenges and opportunities of citation generation for medical tasks, while identifying important design choices that have a significant impact on the final citation quality. Our proposed method achieves superior citation precision and recall improvements compared to strong baseline methods, and we show that evaluation results correlate well with annotation results from professional experts. 

---
# Precise Information Control in Long-Form Text Generation 

**Authors**: Jacqueline He, Howard Yen, Margaret Li, Shuyue Stella Li, Zhiyuan Zeng, Weijia Shi, Yulia Tsvetkov, Danqi Chen, Pang Wei Koh, Luke Zettlemoyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.06589)  

**Abstract**: A central challenge in modern language models (LMs) is intrinsic hallucination: the generation of information that is plausible but unsubstantiated relative to input context. To study this problem, we propose Precise Information Control (PIC), a new task formulation that requires models to generate long-form outputs grounded in a provided set of short self-contained statements, known as verifiable claims, without adding any unsupported ones. For comprehensiveness, PIC includes a full setting that tests a model's ability to include exactly all input claims, and a partial setting that requires the model to selectively incorporate only relevant claims. We present PIC-Bench, a benchmark of eight long-form generation tasks (e.g., summarization, biography generation) adapted to the PIC setting, where LMs are supplied with well-formed, verifiable input claims. Our evaluation of a range of open and proprietary LMs on PIC-Bench reveals that, surprisingly, state-of-the-art LMs still intrinsically hallucinate in over 70% of outputs. To alleviate this lack of faithfulness, we introduce a post-training framework, using a weakly supervised preference data construction method, to train an 8B PIC-LM with stronger PIC ability--improving from 69.1% to 91.0% F1 in the full PIC setting. When integrated into end-to-end factual generation pipelines, PIC-LM improves exact match recall by 17.1% on ambiguous QA with retrieval, and factual precision by 30.5% on a birthplace verification task, underscoring the potential of precisely grounded generation. 

---
# LaMP-Cap: Personalized Figure Caption Generation With Multimodal Figure Profiles 

**Authors**: Ho Yin 'Sam' Ng, Ting-Yao Hsu, Aashish Anantha Ramakrishnan, Branislav Kveton, Nedim Lipka, Franck Dernoncourt, Dongwon Lee, Tong Yu, Sungchul Kim, Ryan A. Rossi, Ting-Hao 'Kenneth' Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06561)  

**Abstract**: Figure captions are crucial for helping readers understand and remember a figure's key message. Many models have been developed to generate these captions, helping authors compose better quality captions more easily. Yet, authors almost always need to revise generic AI-generated captions to match their writing style and the domain's style, highlighting the need for personalization. Despite language models' personalization (LaMP) advances, these technologies often focus on text-only settings and rarely address scenarios where both inputs and profiles are multimodal. This paper introduces LaMP-Cap, a dataset for personalized figure caption generation with multimodal figure profiles. For each target figure, LaMP-Cap provides not only the needed inputs, such as figure images, but also up to three other figures from the same document--each with its image, caption, and figure-mentioning paragraphs--as a profile to characterize the context. Experiments with four LLMs show that using profile information consistently helps generate captions closer to the original author-written ones. Ablation studies reveal that images in the profile are more helpful than figure-mentioning paragraphs, highlighting the advantage of using multimodal profiles over text-only ones. 

---
# Beyond Facts: Evaluating Intent Hallucination in Large Language Models 

**Authors**: Yijie Hao, Haofei Yu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2506.06539)  

**Abstract**: When exposed to complex queries containing multiple conditions, today's large language models (LLMs) tend to produce responses that only partially satisfy the query while neglecting certain conditions. We therefore introduce the concept of Intent Hallucination. In this phenomenon, LLMs either omit (neglecting to address certain parts) or misinterpret (responding to invented query parts) elements of the given query, leading to intent hallucinated generation. To systematically evaluate intent hallucination, we introduce FAITHQA, a novel benchmark for intent hallucination that contains 20,068 problems, covering both query-only and retrieval-augmented generation (RAG) setups with varying topics and difficulty. FAITHQA is the first hallucination benchmark that goes beyond factual verification, tailored to identify the fundamental cause of intent hallucination. By evaluating various LLMs on FAITHQA, we find that (1) intent hallucination is a common issue even for state-of-the-art models, and (2) the phenomenon stems from omission or misinterpretation of LLMs. To facilitate future research, we introduce an automatic LLM generation evaluation metric, CONSTRAINT SCORE, for detecting intent hallucination. Human evaluation results demonstrate that CONSTRAINT SCORE is closer to human performance for intent hallucination compared to baselines. 

---
# Fixing It in Post: A Comparative Study of LLM Post-Training Data Quality and Model Performance 

**Authors**: Aladin Djuhera, Swanand Ravindra Kadhe, Syed Zawad, Farhan Ahmed, Heiko Ludwig, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2506.06522)  

**Abstract**: Recent work on large language models (LLMs) has increasingly focused on post-training and alignment with datasets curated to enhance instruction following, world knowledge, and specialized skills. However, most post-training datasets used in leading open- and closed-source LLMs remain inaccessible to the public, with limited information about their construction process. This lack of transparency has motivated the recent development of open-source post-training corpora. While training on these open alternatives can yield performance comparable to that of leading models, systematic comparisons remain challenging due to the significant computational cost of conducting them rigorously at scale, and are therefore largely absent. As a result, it remains unclear how specific samples, task types, or curation strategies influence downstream performance when assessing data quality. In this work, we conduct the first comprehensive side-by-side analysis of two prominent open post-training datasets: Tulu-3-SFT-Mix and SmolTalk. Using the Magpie framework, we annotate each sample with detailed quality metrics, including turn structure (single-turn vs. multi-turn), task category, input quality, and response quality, and we derive statistics that reveal structural and qualitative similarities and differences between the two datasets. Based on these insights, we design a principled curation recipe that produces a new data mixture, TuluTalk, which contains 14% fewer samples than either source dataset while matching or exceeding their performance on key benchmarks. Our findings offer actionable insights for constructing more effective post-training datasets that improve model performance within practical resource limits. To support future research, we publicly release both the annotated source datasets and our curated TuluTalk mixture. 

---
# Biases Propagate in Encoder-based Vision-Language Models: A Systematic Analysis From Intrinsic Measures to Zero-shot Retrieval Outcomes 

**Authors**: Kshitish Ghate, Tessa Charlesworth, Mona Diab, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06506)  

**Abstract**: To build fair AI systems we need to understand how social-group biases intrinsic to foundational encoder-based vision-language models (VLMs) manifest in biases in downstream tasks. In this study, we demonstrate that intrinsic biases in VLM representations systematically ``carry over'' or propagate into zero-shot retrieval tasks, revealing how deeply rooted biases shape a model's outputs. We introduce a controlled framework to measure this propagation by correlating (a) intrinsic measures of bias in the representational space with (b) extrinsic measures of bias in zero-shot text-to-image (TTI) and image-to-text (ITT) retrieval. Results show substantial correlations between intrinsic and extrinsic bias, with an average $\rho$ = 0.83 $\pm$ 0.10. This pattern is consistent across 114 analyses, both retrieval directions, six social groups, and three distinct VLMs. Notably, we find that larger/better-performing models exhibit greater bias propagation, a finding that raises concerns given the trend towards increasingly complex AI models. Our framework introduces baseline evaluation tasks to measure the propagation of group and valence signals. Investigations reveal that underrepresented groups experience less robust propagation, further skewing their model-related outcomes. 

---
# Improving LLM-Powered EDA Assistants with RAFT 

**Authors**: Luyao Shi, Michael Kazda, Charles Schmitter, Hemlata Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.06500)  

**Abstract**: Electronic design engineers often struggle to efficiently access relevant information for tasks like design verification and technology development. While large language models (LLMs) can enhance productivity as conversational agents, pre-trained open-source LLMs lack domain-specific knowledge for Electronic Design Automation (EDA). In a Retrieval-Augmented Generation (RAG) context, LLMs rely on external context but may still produce inaccurate responses. Retrieval-Augmented Fine-Tuning (RAFT) improves LLM performance, but acquiring labeled question/answer (Q/A) data in EDA is difficult. To address this, we propose using synthetic Q/A datasets to enhance LLMs with RAFT. Our results show that RAFT with synthetic data significantly boosts LLM performance for RAG-based EDA tasks. We also investigate the impact of using real user questions as Retrieval-Augmented Few-Shot (RAFS) examples for synthetic data generation. Additionally, we implement secure access control to ensure sensitive information is only accessible to authorized personnel. Finally, we assess the risk of data leakage and unintended memorization during fine-tuning with synthetic data, providing practical insights. 

---
# What Is Seen Cannot Be Unseen: The Disruptive Effect of Knowledge Conflict on Large Language Models 

**Authors**: Kaiser Sun, Fan Bai, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2506.06485)  

**Abstract**: Large language models frequently rely on both contextual input and parametric knowledge to perform tasks. However, these sources can come into conflict, especially when retrieved documents contradict the model's parametric knowledge. We propose a diagnostic framework to systematically evaluate LLM behavior under context-memory conflict, where the contextual information diverges from their parametric beliefs. We construct diagnostic data that elicit these conflicts and analyze model performance across multiple task types. Our findings reveal that (1) knowledge conflict has minimal impact on tasks that do not require knowledge utilization, (2) model performance is consistently higher when contextual and parametric knowledge are aligned, (3) models are unable to fully suppress their internal knowledge even when instructed, and (4) providing rationales that explain the conflict increases reliance on contexts. These insights raise concerns about the validity of model-based evaluation and underscore the need to account for knowledge conflict in the deployment of LLMs. 

---
# Canonical Autoregressive Generation 

**Authors**: Ivi Chatzi, Nina Corvelo Benz, Stratis Tsirtsis, Manuel Gomez-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2506.06446)  

**Abstract**: State of the art large language models are trained using large amounts of tokens derived from raw text using what is called a tokenizer. Crucially, the tokenizer determines the (token) vocabulary a model will use during inference as well as, in principle, the (token) language. This is because, while the token vocabulary may allow for different tokenizations of a string, the tokenizer always maps the string to only one of these tokenizations--the canonical tokenization. However, multiple lines of empirical evidence suggest that large language models do not always generate canonical token sequences, and this comes with several negative consequences. In this work, we first show that, to generate a canonical token sequence, a model needs to generate (partial) canonical token sequences at each step of the autoregressive generation process underpinning its functioning. Building upon this theoretical result, we introduce canonical sampling, a simple and efficient sampling method that precludes a given model from generating non-canonical token sequences. Further, we also show that, in comparison with standard sampling, the distribution of token sequences generated using canonical sampling is provably closer to the true distribution of token sequences used during training. 

---
# SMAR: Soft Modality-Aware Routing Strategy for MoE-based Multimodal Large Language Models Preserving Language Capabilities 

**Authors**: Guoyang Xia, Yifeng Ding, Fengfa Li, Lei Ren, Chen Wei, Fangxiang Feng, Xiaojie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06406)  

**Abstract**: Mixture of Experts (MoE) architectures have become a key approach for scaling large language models, with growing interest in extending them to multimodal tasks. Existing methods to build multimodal MoE models either incur high training costs or suffer from degraded language capabilities when adapting pretrained models. To address this, we propose Soft ModalityAware Routing (SMAR), a novel regularization technique that uses Kullback Leibler divergence to control routing probability distributions across modalities, encouraging expert specialization without modifying model architecture or heavily relying on textual data. Experiments on visual instruction tuning show that SMAR preserves language ability at 86.6% retention with only 2.5% pure text, outperforming baselines while maintaining strong multimodal performance. Our approach offers a practical and efficient solution to balance modality differentiation and language capabilities in multimodal MoE models. 

---
# Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights 

**Authors**: Sooyung Choi, Jaehyeok Lee, Xiaoyuan Yi, Jing Yao, Xing Xie, JinYeong Bak  

**Link**: [PDF](https://arxiv.org/pdf/2506.06404)  

**Abstract**: The application scope of Large Language Models (LLMs) continues to expand, leading to increasing interest in personalized LLMs that align with human values. However, aligning these models with individual values raises significant safety concerns, as certain values may correlate with harmful information. In this paper, we identify specific safety risks associated with value-aligned LLMs and investigate the psychological principles behind these challenges. Our findings reveal two key insights. (1) Value-aligned LLMs are more prone to harmful behavior compared to non-fine-tuned models and exhibit slightly higher risks in traditional safety evaluations than other fine-tuned models. (2) These safety issues arise because value-aligned LLMs genuinely generate text according to the aligned values, which can amplify harmful outcomes. Using a dataset with detailed safety categories, we find significant correlations between value alignment and safety risks, supported by psychological hypotheses. This study offers insights into the "black box" of value alignment and proposes in-context alignment methods to enhance the safety of value-aligned LLMs. 

---
# Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs 

**Authors**: Hongming Yang, Shi Lin, Jun Shao, Changting Lin, Donghai Zhu, Meng Han, Qinglei Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.06401)  

**Abstract**: Lightweight Large Language Models (LwLLMs) are reduced-parameter, optimized models designed to run efficiently on consumer-grade hardware, offering significant advantages in resource efficiency, cost-effectiveness, and data privacy. However, these models often struggle with limited inference and reasoning capabilities, which restrict their performance on complex tasks and limit their practical applicability. Moreover, existing prompt optimization methods typically rely on extensive manual effort or the meta-cognitive abilities of state-of-the-art LLMs, making them less effective for LwLLMs. To address these challenges, we introduce DeBoP, a new Direct Behavior Optimization Paradigm, original from the Chain-of-Thought (CoT) prompting technique. Unlike CoT Prompting, DeBoP is an automatic optimization method, which focuses on the optimization directly on the behavior of LwLLMs. In particular, DeBoP transforms the optimization of complex prompts into the optimization of discrete, quantifiable execution sequences using a gradient-free Monte Carlo Tree Search. We evaluate DeBoP on seven challenging tasks where state-of-the-art LLMs excel but LwLLMs generally underperform. Experimental results demonstrate that DeBoP significantly outperforms recent prompt optimization methods on most tasks. In particular, DeBoP-optimized LwLLMs surpass GPT-3.5 on most tasks while reducing computational time by approximately 60% compared to other automatic prompt optimization methods. 

---
# Natural Language Interaction with Databases on Edge Devices in the Internet of Battlefield Things 

**Authors**: Christopher D. Molek, Roberto Fronteddu, K. Brent Venable, Niranjan Suri  

**Link**: [PDF](https://arxiv.org/pdf/2506.06396)  

**Abstract**: The expansion of the Internet of Things (IoT) in the battlefield, Internet of Battlefield Things (IoBT), gives rise to new opportunities for enhancing situational awareness. To increase the potential of IoBT for situational awareness in critical decision making, the data from these devices must be processed into consumer-ready information objects, and made available to consumers on demand. To address this challenge we propose a workflow that makes use of natural language processing (NLP) to query a database technology and return a response in natural language. Our solution utilizes Large Language Models (LLMs) that are sized for edge devices to perform NLP as well as graphical databases which are well suited for dynamic connected networks which are pervasive in the IoBT. Our architecture employs LLMs for both mapping questions in natural language to Cypher database queries as well as to summarize the database output back to the user in natural language. We evaluate several medium sized LLMs for both of these tasks on a database representing publicly available data from the US Army's Multipurpose Sensing Area (MSA) at the Jornada Range in Las Cruces, NM. We observe that Llama 3.1 (8 billion parameters) outperforms the other models across all the considered metrics. Most importantly, we note that, unlike current methods, our two step approach allows the relaxation of the Exact Match (EM) requirement of the produced Cypher queries with ground truth code and, in this way, it achieves a 19.4% increase in accuracy. Our workflow lays the ground work for deploying LLMs on edge devices to enable natural language interactions with databases containing information objects for critical decision making. 

---
# Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models 

**Authors**: Pengyi Li, Matvey Skripkin, Alexander Zubrey, Andrey Kuznetsov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2506.06395)  

**Abstract**: Large language models (LLMs) excel at reasoning, yet post-training remains critical for aligning their behavior with task goals. Existing reinforcement learning (RL) methods often depend on costly human annotations or external reward models. We propose Reinforcement Learning via Self-Confidence (RLSC), which uses the model's own confidence as reward signals-eliminating the need for labels, preference models, or reward engineering. Applied to Qwen2.5-Math-7B with only 8 samples per question and 4 training epochs, RLSC improves accuracy by +20.10% on AIME2024, +49.40% on MATH500, and +52.50% on AMC23. RLSC offers a simple, scalable post-training method for reasoning models with minimal supervision. 

---
# Detection Method for Prompt Injection by Integrating Pre-trained Model and Heuristic Feature Engineering 

**Authors**: Yi Ji, Runzhi Li, Baolei Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06384)  

**Abstract**: With the widespread adoption of Large Language Models (LLMs), prompt injection attacks have emerged as a significant security threat. Existing defense mechanisms often face critical trade-offs between effectiveness and generalizability. This highlights the urgent need for efficient prompt injection detection methods that are applicable across a wide range of LLMs. To address this challenge, we propose DMPI-PMHFE, a dual-channel feature fusion detection framework. It integrates a pretrained language model with heuristic feature engineering to detect prompt injection attacks. Specifically, the framework employs DeBERTa-v3-base as a feature extractor to transform input text into semantic vectors enriched with contextual information. In parallel, we design heuristic rules based on known attack patterns to extract explicit structural features commonly observed in attacks. Features from both channels are subsequently fused and passed through a fully connected neural network to produce the final prediction. This dual-channel approach mitigates the limitations of relying only on DeBERTa to extract features. Experimental results on diverse benchmark datasets demonstrate that DMPI-PMHFE outperforms existing methods in terms of accuracy, recall, and F1-score. Furthermore, when deployed actually, it significantly reduces attack success rates across mainstream LLMs, including GLM-4, LLaMA 3, Qwen 2.5, and GPT-4o. 

---
# Enhancing Decision-Making of Large Language Models via Actor-Critic 

**Authors**: Heng Dong, Kefei Duan, Chongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06376)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable advancements in natural language processing tasks, yet they encounter challenges in complex decision-making scenarios that require long-term reasoning and alignment with high-level objectives. Existing methods either rely on short-term auto-regressive action generation or face limitations in accurately simulating rollouts and assessing outcomes, leading to sub-optimal decisions. This paper introduces a novel LLM-based Actor-Critic framework, termed LAC, that effectively improves LLM policies with long-term action evaluations in a principled and scalable way. Our approach addresses two key challenges: (1) extracting robust action evaluations by computing Q-values via token logits associated with positive/negative outcomes, enhanced by future trajectory rollouts and reasoning; and (2) enabling efficient policy improvement through a gradient-free mechanism. Experiments across diverse environments -- including high-level decision-making (ALFWorld), low-level action spaces (BabyAI-Text), and large action spaces (WebShop) -- demonstrate the framework's generality and superiority over state-of-the-art methods. Notably, our approach achieves competitive performance using 7B/8B parameter LLMs, even outperforming baseline methods employing GPT-4 in complex tasks. These results underscore the potential of integrating structured policy optimization with LLMs' intrinsic knowledge to advance decision-making capabilities in multi-step environments. 

---
# Relationship Detection on Tabular Data Using Statistical Analysis and Large Language Models 

**Authors**: Panagiotis Koletsis, Christos Panagiotopoulos, Georgios Th. Papadopoulos, Vasilis Efthymiou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06371)  

**Abstract**: Over the past few years, table interpretation tasks have made significant progress due to their importance and the introduction of new technologies and benchmarks in the field. This work experiments with a hybrid approach for detecting relationships among columns of unlabeled tabular data, using a Knowledge Graph (KG) as a reference point, a task known as CPA. This approach leverages large language models (LLMs) while employing statistical analysis to reduce the search space of potential KG relations. The main modules of this approach for reducing the search space are domain and range constraints detection, as well as relation co-appearance analysis. The experimental evaluation on two benchmark datasets provided by the SemTab challenge assesses the influence of each module and the effectiveness of different state-of-the-art LLMs at various levels of quantization. The experiments were performed, as well as at different prompting techniques. The proposed methodology, which is publicly available on github, proved to be competitive with state-of-the-art approaches on these datasets. 

---
# Unified Game Moderation: Soft-Prompting and LLM-Assisted Label Transfer for Resource-Efficient Toxicity Detection 

**Authors**: Zachary Yang, Domenico Tullo, Reihaneh Rabbany  

**Link**: [PDF](https://arxiv.org/pdf/2506.06347)  

**Abstract**: Toxicity detection in gaming communities faces significant scaling challenges when expanding across multiple games and languages, particularly in real-time environments where computational efficiency is crucial. We present two key findings to address these challenges while building upon our previous work on ToxBuster, a BERT-based real-time toxicity detection system. First, we introduce a soft-prompting approach that enables a single model to effectively handle multiple games by incorporating game-context tokens, matching the performance of more complex methods like curriculum learning while offering superior scalability. Second, we develop an LLM-assisted label transfer framework using GPT-4o-mini to extend support to seven additional languages. Evaluations on real game chat data across French, German, Portuguese, and Russian achieve macro F1-scores ranging from 32.96% to 58.88%, with particularly strong performance in German, surpassing the English benchmark of 45.39%. In production, this unified approach significantly reduces computational resources and maintenance overhead compared to maintaining separate models for each game and language combination. At Ubisoft, this model successfully identifies an average of 50 players, per game, per day engaging in sanctionable behavior. 

---
# TESU-LLM: Training Speech-LLMs Without Speech via Unified Encoder Alignment 

**Authors**: Taesoo Kim, Jong Hwan Ko  

**Link**: [PDF](https://arxiv.org/pdf/2506.06343)  

**Abstract**: Recent advances in speech-enabled language models have shown promising results in building intelligent voice assistants. However, most existing approaches rely on large-scale paired speech-text data and extensive computational resources, which pose challenges in terms of scalability and accessibility. In this paper, we present \textbf{TESU-LLM}, a novel framework that enables training speech-capable language models using only text data. Our key insight is to leverage a unified encoder that maps semantically equivalent text and speech inputs to a shared latent space. By aligning the encoder output with the embedding space of a LLM via a lightweight projection network, we enable the model to generalize from text-only supervision to speech-based inference. Despite being trained exclusively on text, TESU-LLM achieves strong performance on various speech-related benchmarks, comparable to baseline methods trained with large-scale multimodal datasets and substantial computational resources. These results highlight the effectiveness and efficiency of our approach, offering a scalable path toward building speech LLMs without speech data. 

---
# How Significant Are the Real Performance Gains? An Unbiased Evaluation Framework for GraphRAG 

**Authors**: Qiming Zeng, Xiao Yan, Hao Luo, Yuhao Lin, Yuxiang Wang, Fangcheng Fu, Bo Du, Quanqing Xu, Jiawei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06331)  

**Abstract**: By retrieving contexts from knowledge graphs, graph-based retrieval-augmented generation (GraphRAG) enhances large language models (LLMs) to generate quality answers for user questions. Many GraphRAG methods have been proposed and reported inspiring performance in answer quality. However, we observe that the current answer evaluation framework for GraphRAG has two critical flaws, i.e., unrelated questions and evaluation biases, which may lead to biased or even wrong conclusions on performance. To tackle the two flaws, we propose an unbiased evaluation framework that uses graph-text-grounded question generation to produce questions that are more related to the underlying dataset and an unbiased evaluation procedure to eliminate the biases in LLM-based answer assessment. We apply our unbiased framework to evaluate 3 representative GraphRAG methods and find that their performance gains are much more moderate than reported previously. Although our evaluation framework may still have flaws, it calls for scientific evaluations to lay solid foundations for GraphRAG research. 

---
# Play to Generalize: Learning to Reason Through Game Play 

**Authors**: Yunfei Xie, Yinsong Ma, Shiyi Lan, Alan Yuille, Junfei Xiao, Chen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.08011)  

**Abstract**: Developing generalizable reasoning capabilities in multimodal large language models (MLLMs) remains challenging. Motivated by cognitive science literature suggesting that gameplay promotes transferable cognitive skills, we propose a novel post-training paradigm, Visual Game Learning, or ViGaL, where MLLMs develop out-of-domain generalization of multimodal reasoning through playing arcade-like games. Specifically, we show that post-training a 7B-parameter MLLM via reinforcement learning (RL) on simple arcade-like games, e.g. Snake, significantly enhances its downstream performance on multimodal math benchmarks like MathVista, and on multi-discipline questions like MMMU, without seeing any worked solutions, equations, or diagrams during RL, suggesting the capture of transferable reasoning skills. Remarkably, our model outperforms specialist models tuned on multimodal reasoning data in multimodal reasoning benchmarks, while preserving the base model's performance on general visual benchmarks, a challenge where specialist models often fall short. Our findings suggest a new post-training paradigm: synthetic, rule-based games can serve as controllable and scalable pre-text tasks that unlock generalizable multimodal reasoning abilities in MLLMs. 

---
# Reparameterized LLM Training via Orthogonal Equivalence Transformation 

**Authors**: Zeju Qiu, Simon Buchholz, Tim Z. Xiao, Maximilian Dax, Bernhard Schölkopf, Weiyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08001)  

**Abstract**: While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs. 

---
# $τ^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment 

**Authors**: Victor Barres, Honghua Dong, Soham Ray, Xujie Si, Karthik Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07982)  

**Abstract**: Existing benchmarks for conversational AI agents simulate single-control environments, where only the AI agent can use tools to interact with the world, while the user remains a passive information provider. This differs from real-world scenarios like technical support, where users need to actively participate in modifying the state of the (shared) world. In order to address this gap, we introduce $\tau^2$-bench, with four key contributions:
1) A novel Telecom dual-control domain modeled as a Dec-POMDP, where both agent and user make use of tools to act in a shared, dynamic environment that tests both agent coordination and communication,
2) A compositional task generator that programmatically creates diverse, verifiable tasks from atomic components, ensuring domain coverage and controlled complexity,
3) A reliable user simulator tightly coupled with the environment, whose behavior is constrained by tools and observable states, improving simulation fidelity,
4) Fine-grained analysis of agent performance through multiple ablations including separating errors arising from reasoning vs communication/coordination.
In particular, our experiments show significant performance drops when agents shift from no-user to dual-control, highlighting the challenges of guiding users. Overall, $\tau^2$-bench provides a controlled testbed for agents that must both reason effectively and guide user actions. 

---
# HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization 

**Authors**: Hongzheng Chen, Yingheng Wang, Yaohui Cai, Hins Hu, Jiajie Li, Shirley Huang, Chenhui Deng, Rongjian Liang, Shufeng Kong, Haoxing Ren, Samitha Samaranayake, Carla P. Gomes, Zhiru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07972)  

**Abstract**: While Large Language Models (LLMs) have demonstrated significant advancements in reasoning and agent-based problem-solving, current evaluation methodologies fail to adequately assess their capabilities: existing benchmarks either rely on closed-ended questions prone to saturation and memorization, or subjective comparisons that lack consistency and rigor. In this work, we introduce HeuriGym, an agentic framework designed for evaluating heuristic algorithms generated by LLMs for combinatorial optimization problems, characterized by clearly defined objectives and expansive solution spaces. HeuriGym empowers LLMs to propose heuristics, receive evaluative feedback via code execution, and iteratively refine their solutions. We evaluate nine state-of-the-art models on nine problems across domains such as computer systems, logistics, and biology, exposing persistent limitations in tool use, planning, and adaptive reasoning. To quantify performance, we propose the Quality-Yield Index (QYI), a metric that captures both solution pass rate and quality. Even top models like GPT-o4-mini-high and Gemini-2.5-Pro attain QYI scores of only 0.6, well below the expert baseline of 1. Our open-source benchmark aims to guide the development of LLMs toward more effective and realistic problem-solving in scientific and engineering domains. 

---
# Reinforcing Multimodal Understanding and Generation with Dual Self-rewards 

**Authors**: Jixiang Hong, Yiran Zhang, Guanzhong Wang, Yi Liu, Ji-Rong Wen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07963)  

**Abstract**: Building upon large language models (LLMs), recent large multimodal models (LMMs) unify cross-model understanding and generation into a single framework. However, LMMs still struggle to achieve accurate image-text alignment, prone to generating text responses contradicting the visual input or failing to follow the text-to-image prompts. Current solutions require external supervision (e.g., human feedback or reward models) and only address unidirectional tasks-either understanding or generation. In this work, based on the observation that understanding and generation are inverse dual tasks, we introduce a self-supervised dual reward mechanism to reinforce the understanding and generation capabilities of LMMs. Specifically, we sample multiple outputs for a given input in one task domain, then reverse the input-output pairs to compute the dual likelihood of the model as self-rewards for optimization. Extensive experimental results on visual understanding and generation benchmarks demonstrate that our method can effectively enhance the performance of the model without any external supervision, especially achieving remarkable improvements in text-to-image tasks. 

---
# ProtocolLLM: RTL Benchmark for SystemVerilog Generation of Communication Protocols 

**Authors**: Arnav Sheth, Ivaxi Sheth, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2506.07945)  

**Abstract**: Recent advances in Large Language Models (LLMs) have shown promising capabilities in generating code for general-purpose programming languages. In contrast, their applicability for hardware description languages, particularly for generating synthesizable and functionally correct designs, remains significantly underexplored. HDLs such as SystemVerilog are logic-oriented and demand strict adherence to timing semantics, concurrency, and synthesizability constraints. Moreover, HDL-based design flows encompass a broad set of tasks beyond structural code generation, including testbench development, assertion-based verification, timing closure, and protocol-level integration for on-chip communication. The objective of our paper is to analyze the capabilities of state-of-the-art LLMs in generating SystemVerilog implementations of standard communication protocols, a core component of embedded and System-on-Chip (SoC) architectures. This paper introduces the first benchmark suite targeting four widely used protocols: SPI, I2C, UART, and AXI. We define code generation tasks that capture varying levels of design abstraction and prompt specificity. The generated designs are assessed for syntactic correctness, synthesizability, and functional fidelity via waveform simulation and test benches. 

---
# Mimicking or Reasoning: Rethinking Multi-Modal In-Context Learning in Vision-Language Models 

**Authors**: Chengyue Huang, Yuchen Zhu, Sichen Zhu, Jingyun Xiao, Moises Andrade, Shivang Chopra, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2506.07936)  

**Abstract**: Vision-language models (VLMs) are widely assumed to exhibit in-context learning (ICL), a property similar to that of their language-only counterparts. While recent work suggests VLMs can perform multimodal ICL (MM-ICL), studies show they often rely on shallow heuristics -- such as copying or majority voting -- rather than true task understanding. We revisit this assumption by evaluating VLMs under distribution shifts, where support examples come from a dataset different from the query. Surprisingly, performance often degrades with more demonstrations, and models tend to copy answers rather than learn from them. To investigate further, we propose a new MM-ICL with Reasoning pipeline that augments each demonstration with a generated rationale alongside the answer. We conduct extensive and comprehensive experiments on both perception- and reasoning-required datasets with open-source VLMs ranging from 3B to 72B and proprietary models such as Gemini 2.0. We conduct controlled studies varying shot count, retrieval method, rationale quality, and distribution. Our results show limited performance sensitivity across these factors, suggesting that current VLMs do not effectively utilize demonstration-level information as intended in MM-ICL. 

---
# Solving Inequality Proofs with Large Language Models 

**Authors**: Jiayi Sheng, Luna Lyu, Jikai Jin, Tony Xia, Alex Gu, James Zou, Pan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07927)  

**Abstract**: Inequality proving, crucial across diverse scientific and mathematical fields, tests advanced reasoning skills such as discovering tight bounds and strategic theorem application. This makes it a distinct, demanding frontier for large language models (LLMs), offering insights beyond general mathematical problem-solving. Progress in this area is hampered by existing datasets that are often scarce, synthetic, or rigidly formal. We address this by proposing an informal yet verifiable task formulation, recasting inequality proving into two automatically checkable subtasks: bound estimation and relation prediction. Building on this, we release IneqMath, an expert-curated dataset of Olympiad-level inequalities, including a test set and training corpus enriched with step-wise solutions and theorem annotations. We also develop a novel LLM-as-judge evaluation framework, combining a final-answer judge with four step-wise judges designed to detect common reasoning flaws. A systematic evaluation of 29 leading LLMs on IneqMath reveals a surprising reality: even top models like o1 achieve less than 10% overall accuracy under step-wise scrutiny; this is a drop of up to 65.5% from their accuracy considering only final answer equivalence. This discrepancy exposes fragile deductive chains and a critical gap for current LLMs between merely finding an answer and constructing a rigorous proof. Scaling model size and increasing test-time computation yield limited gains in overall proof correctness. Instead, our findings highlight promising research directions such as theorem-guided reasoning and self-refinement. Code and data are available at this https URL. 

---
# Uncovering the Functional Roles of Nonlinearity in Memory 

**Authors**: Manuel Brenner, Georgia Koppe  

**Link**: [PDF](https://arxiv.org/pdf/2506.07919)  

**Abstract**: Memory and long-range temporal processing are core requirements for sequence modeling tasks across natural language processing, time-series forecasting, speech recognition, and control. While nonlinear recurrence has long been viewed as essential for enabling such mechanisms, recent work suggests that linear dynamics may often suffice. In this study, we go beyond performance comparisons to systematically dissect the functional role of nonlinearity in recurrent networks--identifying both when it is computationally necessary, and what mechanisms it enables. We use Almost Linear Recurrent Neural Networks (AL-RNNs), which allow fine-grained control over nonlinearity, as both a flexible modeling tool and a probe into the internal mechanisms of memory. Across a range of classic sequence modeling tasks and a real-world stimulus selection task, we find that minimal nonlinearity is not only sufficient but often optimal, yielding models that are simpler, more robust, and more interpretable than their fully nonlinear or linear counterparts. Our results provide a principled framework for selectively introducing nonlinearity, bridging dynamical systems theory with the functional demands of long-range memory and structured computation in recurrent neural networks, with implications for both artificial and biological neural systems. 

---
# LUCIFER: Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement 

**Authors**: Dimitris Panagopoulos, Adolfo Perrusquia, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07915)  

**Abstract**: In dynamic environments, the rapid obsolescence of pre-existing environmental knowledge creates a gap between an agent's internal model and the evolving reality of its operational context. This disparity between prior and updated environmental valuations fundamentally limits the effectiveness of autonomous decision-making. To bridge this gap, the contextual bias of human domain stakeholders, who naturally accumulate insights through direct, real-time observation, becomes indispensable. However, translating their nuanced, and context-rich input into actionable intelligence for autonomous systems remains an open challenge. To address this, we propose LUCIFER (Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement), a domain-agnostic framework that integrates a hierarchical decision-making architecture with reinforcement learning (RL) and large language models (LLMs) into a unified system. This architecture mirrors how humans decompose complex tasks, enabling a high-level planner to coordinate specialised sub-agents, each focused on distinct objectives and temporally interdependent actions. Unlike traditional applications where LLMs are limited to single role, LUCIFER integrates them in two synergistic roles: as context extractors, structuring verbal stakeholder input into domain-aware representations that influence decision-making through an attention space mechanism aligning LLM-derived insights with the agent's learning process, and as zero-shot exploration facilitators guiding the agent's action selection process during exploration. We benchmark various LLMs in both roles and demonstrate that LUCIFER improves exploration efficiency and decision quality, outperforming flat, goal-conditioned policies. Our findings show the potential of context-driven decision-making, where autonomous systems leverage human contextual knowledge for operational success. 

---
# Evaluating Large Language Models on the Frame and Symbol Grounding Problems: A Zero-shot Benchmark 

**Authors**: Shoko Oka  

**Link**: [PDF](https://arxiv.org/pdf/2506.07896)  

**Abstract**: Recent advancements in large language models (LLMs) have revitalized philosophical debates surrounding artificial intelligence. Two of the most fundamental challenges - namely, the Frame Problem and the Symbol Grounding Problem - have historically been viewed as unsolvable within traditional symbolic AI systems. This study investigates whether modern LLMs possess the cognitive capacities required to address these problems. To do so, I designed two benchmark tasks reflecting the philosophical core of each problem, administered them under zero-shot conditions to 13 prominent LLMs (both closed and open-source), and assessed the quality of the models' outputs across five trials each. Responses were scored along multiple criteria, including contextual reasoning, semantic coherence, and information filtering. The results demonstrate that while open-source models showed variability in performance due to differences in model size, quantization, and instruction tuning, several closed models consistently achieved high scores. These findings suggest that select modern LLMs may be acquiring capacities sufficient to produce meaningful and stable responses to these long-standing theoretical challenges. 

---
# Improving large language models with concept-aware fine-tuning 

**Authors**: Michael K. Chen, Xikun Zhang, Jiaxing Huang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07833)  

**Abstract**: Large language models (LLMs) have become the cornerstone of modern AI. However, the existing paradigm of next-token prediction fundamentally limits their ability to form coherent, high-level concepts, making it a critical barrier to human-like understanding and reasoning. Take the phrase "ribonucleic acid" as an example: an LLM will first decompose it into tokens, i.e., artificial text fragments ("rib", "on", ...), then learn each token sequentially, rather than grasping the phrase as a unified, coherent semantic entity. This fragmented representation hinders deeper conceptual understanding and, ultimately, the development of truly intelligent systems. In response, we introduce Concept-Aware Fine-Tuning (CAFT), a novel multi-token training method that redefines how LLMs are fine-tuned. By enabling the learning of sequences that span multiple tokens, this method fosters stronger concept-aware learning. Our experiments demonstrate significant improvements compared to conventional next-token finetuning methods across diverse tasks, including traditional applications like text summarization and domain-specific ones like de novo protein design. Multi-token prediction was previously only possible in the prohibitively expensive pretraining phase; CAFT, to our knowledge, is the first to bring the multi-token setting to the post-training phase, thus effectively democratizing its benefits for the broader community of practitioners and researchers. Finally, the unexpected effectiveness of our proposed method suggests wider implications for the machine learning research community. All code and data are available at this https URL 

---
# E-LDA: Toward Interpretable LDA Topic Models with Strong Guarantees in Logarithmic Parallel Time 

**Authors**: Adam Breuer  

**Link**: [PDF](https://arxiv.org/pdf/2506.07747)  

**Abstract**: In this paper, we provide the first practical algorithms with provable guarantees for the problem of inferring the topics assigned to each document in an LDA topic model. This is the primary inference problem for many applications of topic models in social science, data exploration, and causal inference settings. We obtain this result by showing a novel non-gradient-based, combinatorial approach to estimating topic models. This yields algorithms that converge to near-optimal posterior probability in logarithmic parallel computation time (adaptivity) -- exponentially faster than any known LDA algorithm. We also show that our approach can provide interpretability guarantees such that each learned topic is formally associated with a known keyword. Finally, we show that unlike alternatives, our approach can maintain the independence assumptions necessary to use the learned topic model for downstream causal inference methods that allow researchers to study topics as treatments. In terms of practical performance, our approach consistently returns solutions of higher semantic quality than solutions from state-of-the-art LDA algorithms, neural topic models, and LLM-based topic models across a diverse range of text datasets and evaluation parameters. 

---
# Learning Speaker-Invariant Visual Features for Lipreading 

**Authors**: Yu Li, Feng Xue, Shujie Li, Jinrui Zhang, Shuang Yang, Dan Guo, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07572)  

**Abstract**: Lipreading is a challenging cross-modal task that aims to convert visual lip movements into spoken text. Existing lipreading methods often extract visual features that include speaker-specific lip attributes (e.g., shape, color, texture), which introduce spurious correlations between vision and text. These correlations lead to suboptimal lipreading accuracy and restrict model generalization. To address this challenge, we introduce SIFLip, a speaker-invariant visual feature learning framework that disentangles speaker-specific attributes using two complementary disentanglement modules (Implicit Disentanglement and Explicit Disentanglement) to improve generalization. Specifically, since different speakers exhibit semantic consistency between lip movements and phonetic text when pronouncing the same words, our implicit disentanglement module leverages stable text embeddings as supervisory signals to learn common visual representations across speakers, implicitly decoupling speaker-specific features. Additionally, we design a speaker recognition sub-task within the main lipreading pipeline to filter speaker-specific features, then further explicitly disentangle these personalized visual features from the backbone network via gradient reversal. Experimental results demonstrate that SIFLip significantly enhances generalization performance across multiple public datasets. Experimental results demonstrate that SIFLip significantly improves generalization performance across multiple public datasets, outperforming state-of-the-art methods. 

---
# SAFEFLOW: A Principled Protocol for Trustworthy and Transactional Autonomous Agent Systems 

**Authors**: Peiran Li, Xinkai Zou, Zhuohang Wu, Ruifeng Li, Shuo Xing, Hanwen Zheng, Zhikai Hu, Yuping Wang, Haoxi Li, Qin Yuan, Yingmo Zhang, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07564)  

**Abstract**: Recent advances in large language models (LLMs) and vision-language models (VLMs) have enabled powerful autonomous agents capable of complex reasoning and multi-modal tool use. Despite their growing capabilities, today's agent frameworks remain fragile, lacking principled mechanisms for secure information flow, reliability, and multi-agent coordination. In this work, we introduce SAFEFLOW, a new protocol-level framework for building trustworthy LLM/VLM-based agents. SAFEFLOW enforces fine-grained information flow control (IFC), precisely tracking provenance, integrity, and confidentiality of all the data exchanged between agents, tools, users, and environments. By constraining LLM reasoning to respect these security labels, SAFEFLOW prevents untrusted or adversarial inputs from contaminating high-integrity decisions. To ensure robustness in concurrent multi-agent settings, SAFEFLOW introduces transactional execution, conflict resolution, and secure scheduling over shared state, preserving global consistency across agents. We further introduce mechanisms, including write-ahead logging, rollback, and secure caches, that further enhance resilience against runtime errors and policy violations. To validate the performances, we built SAFEFLOWBENCH, a comprehensive benchmark suite designed to evaluate agent reliability under adversarial, noisy, and concurrent operational conditions. Extensive experiments demonstrate that agents built with SAFEFLOW maintain impressive task performance and security guarantees even in hostile environments, substantially outperforming state-of-the-art. Together, SAFEFLOW and SAFEFLOWBENCH lay the groundwork for principled, robust, and secure agent ecosystems, advancing the frontier of reliable autonomy. 

---
# ChemAgent: Enhancing LLMs for Chemistry and Materials Science through Tree-Search Based Tool Learning 

**Authors**: Mengsong Wu, YaFei Wang, Yidong Ming, Yuqi An, Yuwei Wan, Wenliang Chen, Binbin Lin, Yuqiang Li, Tong Xie, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.07551)  

**Abstract**: Large language models (LLMs) have recently demonstrated promising capabilities in chemistry tasks while still facing challenges due to outdated pretraining knowledge and the difficulty of incorporating specialized chemical expertise. To address these issues, we propose an LLM-based agent that synergistically integrates 137 external chemical tools created ranging from basic information retrieval to complex reaction predictions, and a dataset curation pipeline to generate the dataset ChemToolBench that facilitates both effective tool selection and precise parameter filling during fine-tuning and evaluation. We introduce a Hierarchical Evolutionary Monte Carlo Tree Search (HE-MCTS) framework, enabling independent optimization of tool planning and execution. By leveraging self-generated data, our approach supports step-level fine-tuning (FT) of the policy model and training task-adaptive PRM and ORM that surpass GPT-4o. Experimental evaluations demonstrate that our approach significantly improves performance in Chemistry QA and discovery tasks, offering a robust solution to integrate specialized tools with LLMs for advanced chemical applications. All datasets and code are available at this https URL . 

---
# Speaker-Distinguishable CTC: Learning Speaker Distinction Using CTC for Multi-Talker Speech Recognition 

**Authors**: Asahi Sakuma, Hiroaki Sato, Ryuga Sugano, Tadashi Kumano, Yoshihiko Kawai, Tetsuji Ogawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.07515)  

**Abstract**: This paper presents a novel framework for multi-talker automatic speech recognition without the need for auxiliary information. Serialized Output Training (SOT), a widely used approach, suffers from recognition errors due to speaker assignment failures. Although incorporating auxiliary information, such as token-level timestamps, can improve recognition accuracy, extracting such information from natural conversational speech remains challenging. To address this limitation, we propose Speaker-Distinguishable CTC (SD-CTC), an extension of CTC that jointly assigns a token and its corresponding speaker label to each frame. We further integrate SD-CTC into the SOT framework, enabling the SOT model to learn speaker distinction using only overlapping speech and transcriptions. Experimental comparisons show that multi-task learning with SD-CTC and SOT reduces the error rate of the SOT model by 26% and achieves performance comparable to state-of-the-art methods relying on auxiliary information. 

---
# Graph-of-Causal Evolution: Challenging Chain-of-Model for Reasoning 

**Authors**: Libo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07501)  

**Abstract**: In view of the problem that each subchain in the chain-of-model (CoM) relies only on the information of the previous subchain and may lose long-range dependencies due to the causal mask blocking the global context flow between multi-level subchains, this work proposes a graph of causal evolution (GoCE). Its core principle is to map the implicit token representation into a differentiable and sparse causal adjacency matrix, then permeate causal constraints through each layer of calculation using causal-masked attention and causal-MoE. By combining intervention consistency loss test and self-evolution gate, the dynamic balance between causal structure learning and adaptive updating of transformer architecture is realized. The researcher built experimental environments in sandboxes built with Claude Sonnet 4, o4-mini-high, and DeepSeek R1 respectively with the transformer variant architecture introduced in GoCE. It is evaluated on publicly available datasets including CLUTRR, CLADDER, EX-FEVER, and CausalQA and compared with the baseline LLMs. The finding proves that GoCE strengthens the transformer's ability to capture long-range causal dependencies, while the ability to self-evolve is improved. It not only surpasses the design of CoM in terms of design principles, but also provides experience for future research on causal learning and continuous adaptive improvement. 

---
# Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models 

**Authors**: Mickel Liu, Liwei Jiang, Yancheng Liang, Simon Shaolei Du, Yejin Choi, Tim Althoff, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2506.07468)  

**Abstract**: Conventional language model (LM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch -- attackers overfit to obsolete defenses, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning algorithm where an attacker and defender agent co-evolve through continuous interaction. We cast safety alignment as a two-player zero-sum game, where a single model alternates between attacker and defender roles -- generating adversarial prompts and safeguarding against them -- while a reward LM adjudicates outcomes. This enables dynamic co-adaptation. Grounded in the game-theoretic framework of zero-sum games, we establish a theoretical safety guarantee which motivates the design of our method: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses to any adversarial input. Empirically, Self-RedTeam uncovers more diverse attacks (+21.8% SBERT) compared to attackers trained against static defenders and achieves higher robustness on safety benchmarks (e.g., +65.5% on WildJailBreak) than defenders trained against static attackers. We further propose hidden Chain-of-Thought, allowing agents to plan privately, which boosts adversarial diversity and reduces over-refusals. Our results motivate a shift from reactive patching to proactive co-evolution in LM safety training, enabling scalable, autonomous, and robust self-improvement of LMs via multi-agent reinforcement learning (MARL). 

---
# GLOS: Sign Language Generation with Temporally Aligned Gloss-Level Conditioning 

**Authors**: Taeryung Lee, Hyeongjin Nam, Gyeongsik Moon, Kyoung Mu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.07460)  

**Abstract**: Sign language generation (SLG), or text-to-sign generation, bridges the gap between signers and non-signers. Despite recent progress in SLG, existing methods still often suffer from incorrect lexical ordering and low semantic accuracy. This is primarily due to sentence-level condition, which encodes the entire sentence of the input text into a single feature vector as a condition for SLG. This approach fails to capture the temporal structure of sign language and lacks the granularity of word-level semantics, often leading to disordered sign sequences and ambiguous motions. To overcome these limitations, we propose GLOS, a sign language generation framework with temporally aligned gloss-level conditioning. First, we employ gloss-level conditions, which we define as sequences of gloss embeddings temporally aligned with the motion sequence. This enables the model to access both the temporal structure of sign language and word-level semantics at each timestep. As a result, this allows for fine-grained control of signs and better preservation of lexical order. Second, we introduce a condition fusion module, temporal alignment conditioning (TAC), to efficiently deliver the word-level semantic and temporal structure provided by the gloss-level condition to the corresponding motion timesteps. Our method, which is composed of gloss-level conditions and TAC, generates signs with correct lexical order and high semantic accuracy, outperforming prior methods on CSL-Daily and Phoenix-2014T. 

---
# When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment 

**Authors**: Yuxin Xiao, Sana Tonekaboni, Walter Gerych, Vinith Suriyakumar, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07452)  

**Abstract**: Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in jailbreak queries. Although these style patterns are semantically unrelated to the malicious intents behind jailbreak queries, their safety impact remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We evaluate 32 LLMs across seven jailbreak benchmarks, and find that malicious queries with style patterns inflate the attack success rate (ASR) for nearly all models. Notably, ASR inflation correlates with both the length of style patterns and the relative attention an LLM exhibits on them. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs and five fine-tuning style settings, SafeStyle consistently outperforms baselines in maintaining LLM safety. 

---
# LlamaRec-LKG-RAG: A Single-Pass, Learnable Knowledge Graph-RAG Framework for LLM-Based Ranking 

**Authors**: Vahid Azizi, Fatemeh Koochaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.07449)  

**Abstract**: Recent advances in Large Language Models (LLMs) have driven their adoption in recommender systems through Retrieval-Augmented Generation (RAG) frameworks. However, existing RAG approaches predominantly rely on flat, similarity-based retrieval that fails to leverage the rich relational structure inherent in user-item interactions. We introduce LlamaRec-LKG-RAG, a novel single-pass, end-to-end trainable framework that integrates personalized knowledge graph context into LLM-based recommendation ranking. Our approach extends the LlamaRec architecture by incorporating a lightweight user preference module that dynamically identifies salient relation paths within a heterogeneous knowledge graph constructed from user behavior and item metadata. These personalized subgraphs are seamlessly integrated into prompts for a fine-tuned Llama-2 model, enabling efficient and interpretable recommendations through a unified inference step. Comprehensive experiments on ML-100K and Amazon Beauty datasets demonstrate consistent and significant improvements over LlamaRec across key ranking metrics (MRR, NDCG, Recall). LlamaRec-LKG-RAG demonstrates the critical value of structured reasoning in LLM-based recommendations and establishes a foundation for scalable, knowledge-aware personalization in next-generation recommender systems. Code is available at~\href{this https URL}{repository}. 

---
# Beyond Jailbreaks: Revealing Stealthier and Broader LLM Security Risks Stemming from Alignment Failures 

**Authors**: Yukai Zhou, Sibei Yang, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07402)  

**Abstract**: Large language models (LLMs) are increasingly deployed in real-world applications, raising concerns about their security. While jailbreak attacks highlight failures under overtly harmful queries, they overlook a critical risk: incorrectly answering harmless-looking inputs can be dangerous and cause real-world harm (Implicit Harm). We systematically reformulate the LLM risk landscape through a structured quadrant perspective based on output factuality and input harmlessness, uncovering an overlooked high-risk region. To investigate this gap, we propose JailFlipBench, a benchmark aims to capture implicit harm, spanning single-modal, multimodal, and factual extension scenarios with diverse evaluation metrics. We further develop initial JailFlip attack methodologies and conduct comprehensive evaluations across multiple open-source and black-box LLMs, show that implicit harm present immediate and urgent real-world risks, calling for broader LLM safety assessments and alignment beyond conventional jailbreak paradigms. 

---
# G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems 

**Authors**: Guibin Zhang, Muxin Fu, Guancheng Wan, Miao Yu, Kun Wang, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07398)  

**Abstract**: Large language model (LLM)-powered multi-agent systems (MAS) have demonstrated cognitive and execution capabilities that far exceed those of single LLM agents, yet their capacity for self-evolution remains hampered by underdeveloped memory architectures. Upon close inspection, we are alarmed to discover that prevailing MAS memory mechanisms (1) are overly simplistic, completely disregarding the nuanced inter-agent collaboration trajectories, and (2) lack cross-trial and agent-specific customization, in stark contrast to the expressive memory developed for single agents. To bridge this gap, we introduce G-Memory, a hierarchical, agentic memory system for MAS inspired by organizational memory theory, which manages the lengthy MAS interaction via a three-tier graph hierarchy: insight, query, and interaction graphs. Upon receiving a new user query, G-Memory performs bi-directional memory traversal to retrieve both $\textit{high-level, generalizable insights}$ that enable the system to leverage cross-trial knowledge, and $\textit{fine-grained, condensed interaction trajectories}$ that compactly encode prior collaboration experiences. Upon task execution, the entire hierarchy evolves by assimilating new collaborative trajectories, nurturing the progressive evolution of agent teams. Extensive experiments across five benchmarks, three LLM backbones, and three popular MAS frameworks demonstrate that G-Memory improves success rates in embodied action and accuracy in knowledge QA by up to $20.89\%$ and $10.12\%$, respectively, without any modifications to the original frameworks. Our codes are available at this https URL. 

---
# Multi-Step Visual Reasoning with Visual Tokens Scaling and Verification 

**Authors**: Tianyi Bai, Zengjie Hu, Fupeng Sun, Jiantao Qiu, Yizhen Jiang, Guangxin He, Bohan Zeng, Conghui He, Binhang Yuan, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07235)  

**Abstract**: Multi-modal large language models (MLLMs) have achieved remarkable capabilities by integrating visual perception with language understanding, enabling applications such as image-grounded dialogue, visual question answering, and scientific analysis. However, most MLLMs adopt a static inference paradigm, encoding the entire image into fixed visual tokens upfront, which limits their ability to iteratively refine understanding or adapt to context during inference. This contrasts sharply with human perception, which is dynamic, selective, and feedback-driven. In this work, we introduce a novel framework for inference-time visual token scaling that enables MLLMs to perform iterative, verifier-guided reasoning over visual content. We formulate the problem as a Markov Decision Process, involving a reasoner that proposes visual actions and a verifier, which is trained via multi-step Direct Preference Optimization (DPO), that evaluates these actions and determines when reasoning should terminate. To support this, we present a new dataset, VTS, comprising supervised reasoning trajectories (VTS-SFT) and preference-labeled reasoning comparisons (VTS-DPO). Our method significantly outperforms existing approaches across diverse visual reasoning benchmarks, offering not only improved accuracy but also more interpretable and grounded reasoning processes. These results demonstrate the promise of dynamic inference mechanisms for enabling fine-grained, context-aware visual reasoning in next-generation MLLMs. 

---
# Reducing Object Hallucination in Large Audio-Language Models via Audio-Aware Decoding 

**Authors**: Tzu-wen Hsu, Ke-Han Lu, Cheng-Han Chiang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.07233)  

**Abstract**: Large Audio-Language Models (LALMs) can take audio and text as the inputs and answer questions about the audio. While prior LALMs have shown strong performance on standard benchmarks, there has been alarming evidence that LALMs can hallucinate what is presented in the audio. To mitigate the hallucination of LALMs, we introduce Audio-Aware Decoding (AAD), a lightweight inference-time strategy that uses contrastive decoding to compare the token prediction logits with and without the audio context. By contrastive decoding, AAD promotes the tokens whose probability increases when the audio is present. We conduct our experiment on object hallucination datasets with three LALMs and show that AAD improves the F1 score by 0.046 to 0.428. We also show that AAD can improve the accuracy on general audio QA datasets like Clotho-AQA by 5.4% to 10.3%. We conduct thorough ablation studies to understand the effectiveness of each component in AAD. 

---
# Hallucination at a Glance: Controlled Visual Edits and Fine-Grained Multimodal Learning 

**Authors**: Tianyi Bai, Yuxuan Fan, Jiantao Qiu, Fupeng Sun, Jiayi Song, Junlin Han, Zichen Liu, Conghui He, Wentao Zhang, Binhang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07227)  

**Abstract**: Multimodal large language models (MLLMs) have achieved strong performance on vision-language tasks but still struggle with fine-grained visual differences, leading to hallucinations or missed semantic shifts. We attribute this to limitations in both training data and learning objectives. To address these issues, we propose a controlled data generation pipeline that produces minimally edited image pairs with semantically aligned captions. Using this pipeline, we construct the Micro Edit Dataset (MED), containing over 50K image-text pairs spanning 11 fine-grained edit categories, including attribute, count, position, and object presence changes. Building on MED, we introduce a supervised fine-tuning (SFT) framework with a feature-level consistency loss that promotes stable visual embeddings under small edits. We evaluate our approach on the Micro Edit Detection benchmark, which includes carefully balanced evaluation pairs designed to test sensitivity to subtle visual variations across the same edit categories. Our method improves difference detection accuracy and reduces hallucinations compared to strong baselines, including GPT-4o. Moreover, it yields consistent gains on standard vision-language tasks such as image captioning and visual question answering. These results demonstrate the effectiveness of combining targeted data and alignment objectives for enhancing fine-grained visual reasoning in MLLMs. 

---
# SAP-Bench: Benchmarking Multimodal Large Language Models in Surgical Action Planning 

**Authors**: Mengya Xu, Zhongzhen Huang, Dillan Imans, Yiru Ye, Xiaofan Zhang, Qi Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.07196)  

**Abstract**: Effective evaluation is critical for driving advancements in MLLM research. The surgical action planning (SAP) task, which aims to generate future action sequences from visual inputs, demands precise and sophisticated analytical capabilities. Unlike mathematical reasoning, surgical decision-making operates in life-critical domains and requires meticulous, verifiable processes to ensure reliability and patient safety. This task demands the ability to distinguish between atomic visual actions and coordinate complex, long-horizon procedures, capabilities that are inadequately evaluated by current benchmarks. To address this gap, we introduce SAP-Bench, a large-scale, high-quality dataset designed to enable multimodal large language models (MLLMs) to perform interpretable surgical action planning. Our SAP-Bench benchmark, derived from the cholecystectomy procedures context with the mean duration of 1137.5s, and introduces temporally-grounded surgical action annotations, comprising the 1,226 clinically validated action clips (mean duration: 68.7s) capturing five fundamental surgical actions across 74 procedures. The dataset provides 1,152 strategically sampled current frames, each paired with the corresponding next action as multimodal analysis anchors. We propose the MLLM-SAP framework that leverages MLLMs to generate next action recommendations from the current surgical scene and natural language instructions, enhanced with injected surgical domain knowledge. To assess our dataset's effectiveness and the broader capabilities of current models, we evaluate seven state-of-the-art MLLMs (e.g., OpenAI-o1, GPT-4o, QwenVL2.5-72B, Claude-3.5-Sonnet, GeminiPro2.5, Step-1o, and GLM-4v) and reveal critical gaps in next action prediction performance. 

---
# Mitigating Behavioral Hallucination in Multimodal Large Language Models for Sequential Images 

**Authors**: Liangliang You, Junchi Yao, Shu Yang, Guimin Hu, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07184)  

**Abstract**: While multimodal large language models excel at various tasks, they still suffer from hallucinations, which limit their reliability and scalability for broader domain applications. To address this issue, recent research mainly focuses on objective hallucination. However, for sequential images, besides objective hallucination, there is also behavioral hallucination, which is less studied. This work aims to fill in the gap. We first reveal that behavioral hallucinations mainly arise from two key factors: prior-driven bias and the snowball effect. Based on these observations, we introduce SHE (Sequence Hallucination Eradication), a lightweight, two-stage framework that (1) detects hallucinations via visual-textual alignment check using our proposed adaptive temporal window and (2) mitigates them via orthogonal projection onto the joint embedding space. We also propose a new metric (BEACH) to quantify behavioral hallucination severity. Empirical results on standard benchmarks demonstrate that SHE reduces behavioral hallucination by over 10% on BEACH while maintaining descriptive accuracy. 

---
# Efficient Text-Attributed Graph Learning through Selective Annotation and Graph Alignment 

**Authors**: Huanyi Xie, Lijie Hu, Lu Yu, Tianhao Huang, Longfei Li, Meng Li, Jun Zhou, Huan Wang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07168)  

**Abstract**: In the realm of Text-attributed Graphs (TAGs), traditional graph neural networks (GNNs) often fall short due to the complex textual information associated with each node. Recent methods have improved node representations by leveraging large language models (LLMs) to enhance node text features, but these approaches typically require extensive annotations or fine-tuning across all nodes, which is both time-consuming and costly. To overcome these challenges, we introduce GAGA, an efficient framework for TAG representation learning. GAGA reduces annotation time and cost by focusing on annotating only representative nodes and edges. It constructs an annotation graph that captures the topological relationships among these annotations. Furthermore, GAGA employs a two-level alignment module to effectively integrate the annotation graph with the TAG, aligning their underlying structures. Experiments show that GAGA achieves classification accuracies on par with or surpassing state-of-the-art methods while requiring only 1% of the data to be annotated, demonstrating its high efficiency. 

---
# Learning Compact Vision Tokens for Efficient Large Multimodal Models 

**Authors**: Hao Tang, Chengchao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07138)  

**Abstract**: Large multimodal models (LMMs) suffer significant computational challenges due to the high cost of Large Language Models (LLMs) and the quadratic complexity of processing long vision token sequences. In this paper, we explore the spatial redundancy among vision tokens and shorten the length of vision token sequences for inference acceleration. Specifically, we propose a Spatial Token Fusion (STF) method to learn compact vision tokens for short vision token sequence, where spatial-adjacent tokens are fused into one. Meanwhile, weight-frozen vision encoder can not well adapt to the demand of extensive downstream vision-language tasks. To this end, we further introduce a Multi-Block Token Fusion (MBTF) module to supplement multi-granularity features for the reduced token sequence. Overall, we combine STF and MBTF module to balance token reduction and information preservation, thereby improving inference efficiency without sacrificing multimodal reasoning capabilities. Experimental results demonstrate that our method based on LLaVA-1.5 achieves comparable or even superior performance to the baseline on 8 popular vision-language benchmarks with only $25\%$ vision tokens of baseline. The source code and trained weights are available at this https URL. 

---
# Interpretable and Reliable Detection of AI-Generated Images via Grounded Reasoning in MLLMs 

**Authors**: Yikun Ji, Hong Yan, Jun Lan, Huijia Zhu, Weiqiang Wang, Qi Fan, Liqing Zhang, Jianfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07045)  

**Abstract**: The rapid advancement of image generation technologies intensifies the demand for interpretable and robust detection methods. Although existing approaches often attain high accuracy, they typically operate as black boxes without providing human-understandable justifications. Multi-modal Large Language Models (MLLMs), while not originally intended for forgery detection, exhibit strong analytical and reasoning capabilities. When properly fine-tuned, they can effectively identify AI-generated images and offer meaningful explanations. However, existing MLLMs still struggle with hallucination and often fail to align their visual interpretations with actual image content and human reasoning. To bridge this gap, we construct a dataset of AI-generated images annotated with bounding boxes and descriptive captions that highlight synthesis artifacts, establishing a foundation for human-aligned visual-textual grounded reasoning. We then finetune MLLMs through a multi-stage optimization strategy that progressively balances the objectives of accurate detection, visual localization, and coherent textual explanation. The resulting model achieves superior performance in both detecting AI-generated images and localizing visual flaws, significantly outperforming baseline methods. 

---
# HauntAttack: When Attack Follows Reasoning as a Shadow 

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Lei Sha, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.07031)  

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing exceptional capabilities. However, the enhancement of reasoning abilities and the exposure of their internal reasoning processes introduce new safety vulnerabilities. One intriguing concern is: when reasoning is strongly entangled with harmfulness, what safety-reasoning trade-off do LRMs exhibit? To address this issue, we introduce HauntAttack, a novel and general-purpose black-box attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we treat reasoning questions as carriers and substitute one of their original conditions with a harmful instruction. This process creates a reasoning pathway in which the model is guided step by step toward generating unsafe outputs. Based on HauntAttack, we conduct comprehensive experiments on multiple LRMs. Our results reveal that even the most advanced LRMs exhibit significant safety vulnerabilities. Additionally, we perform a detailed analysis of different models, various types of harmful instructions, and model output patterns, providing valuable insights into the security of LRMs. 

---
# Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test 

**Authors**: Xiaoyuan Zhu, Yaowen Ye, Tianyi Qiu, Hanlin Zhu, Sijun Tan, Ajraf Mannan, Jonathan Michala, Raluca Ada Popa, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2506.06975)  

**Abstract**: As API access becomes a primary interface to large language models (LLMs), users often interact with black-box systems that offer little transparency into the deployed model. To reduce costs or maliciously alter model behaviors, API providers may discreetly serve quantized or fine-tuned variants, which can degrade performance and compromise safety. Detecting such substitutions is difficult, as users lack access to model weights and, in most cases, even output logits. To tackle this problem, we propose a rank-based uniformity test that can verify the behavioral equality of a black-box LLM to a locally deployed authentic model. Our method is accurate, query-efficient, and avoids detectable query patterns, making it robust to adversarial providers that reroute or mix responses upon the detection of testing attempts. We evaluate the approach across diverse threat scenarios, including quantization, harmful fine-tuning, jailbreak prompts, and full model substitution, showing that it consistently achieves superior statistical power over prior methods under constrained query budgets. 

---
# The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity 

**Authors**: Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio, Mehrdad Farajtabar  

**Link**: [PDF](https://arxiv.org/pdf/2506.06941)  

**Abstract**: Recent generations of language models have introduced Large Reasoning Models (LRMs) that generate detailed thinking processes before providing answers. While these models demonstrate improved performance on reasoning benchmarks, their fundamental capabilities, scaling properties, and limitations remain insufficiently understood. Current evaluations primarily focus on established math and coding benchmarks, emphasizing final answer accuracy. However, this evaluation paradigm often suffers from contamination and does not provide insights into the reasoning traces. In this work, we systematically investigate these gaps with the help of controllable puzzle environments that allow precise manipulation of complexity while maintaining consistent logical structures. This setup enables the analysis of not only final answers but also the internal reasoning traces, offering insights into how LRMs think. Through extensive experiments, we show that LRMs face a complete accuracy collapse beyond certain complexities. Moreover, they exhibit a counterintuitive scaling limit: their reasoning effort increases with problem complexity up to a point, then declines despite having remaining token budget. By comparing LRMs with their standard LLM counterparts under same inference compute, we identify three performance regimes: (1) low-complexity tasks where standard models outperform LRMs, (2) medium-complexity tasks where LRMs demonstrates advantage, and (3) high-complexity tasks where both models face complete collapse. We found that LRMs have limitations in exact computation: they fail to use explicit algorithms and reason inconsistently across scales. We also investigate the reasoning traces in more depth, studying the patterns of explored solutions and analyzing the models' computational behavior, shedding light on their strengths, limitations, and raising questions about their reasoning capabilities. 

---
# Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering 

**Authors**: Akash Gupta, Amos Storkey, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2506.06905)  

**Abstract**: Large Multimodal Models (LMMs) often rely on in-context learning (ICL) to perform new tasks with minimal supervision. However, ICL performance, especially in smaller LMMs, is inconsistent and does not always improve monotonically with increasing examples. We hypothesize that this occurs due to the LMM being overwhelmed by additional information present in the image embeddings, which is not required for the downstream task. To address this, we propose a meta-learning approach that provides an alternative for inducing few-shot capabilities in LMMs, using a fixed set of soft prompts that are distilled from task-relevant image features and can be adapted at test time using a few examples. To facilitate this distillation, we introduce an attention-mapper module that can be easily integrated with the popular LLaVA v1.5 architecture and is jointly learned with soft prompts, enabling task adaptation in LMMs under low-data regimes with just a few gradient steps. Evaluation on the VL-ICL Bench shows that our method consistently outperforms ICL and related prompt-tuning approaches, even under image perturbations, improving task induction and reasoning across visual question answering tasks. 

---
# Cross-Entropy Games for Language Models: From Implicit Knowledge to General Capability Measures 

**Authors**: Clément Hongler, Andrew Emil  

**Link**: [PDF](https://arxiv.org/pdf/2506.06832)  

**Abstract**: Large Language Models (LLMs) define probability measures on text. By considering the implicit knowledge question of what it means for an LLM to know such a measure and what it entails algorithmically, we are naturally led to formulate a series of tasks that go beyond generative sampling, involving forms of summarization, counterfactual thinking, anomaly detection, originality search, reverse prompting, debating, creative solving, etc. These tasks can be formulated as games based on LLM measures, which we call Cross-Entropy (Xent) Games. Xent Games can be single-player or multi-player. They involve cross-entropy scores and cross-entropy constraints, and can be expressed as simple computational graphs and programs. We show the Xent Game space is large enough to contain a wealth of interesting examples, while being constructible from basic game-theoretic consistency axioms. We then discuss how the Xent Game space can be used to measure the abilities of LLMs. This leads to the construction of Xent Game measures: finite families of Xent Games that can be used as capability benchmarks, built from a given scope, by extracting a covering measure. To address the unbounded scope problem associated with the challenge of measuring general abilities, we propose to explore the space of Xent Games in a coherent fashion, using ideas inspired by evolutionary dynamics. 

---
# Mitigating Object Hallucination via Robust Local Perception Search 

**Authors**: Zixian Gao, Chao Yang, Zhanhui Zhou, Xing Xu, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06729)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs) have enabled them to effectively integrate vision and language, addressing a variety of downstream tasks. However, despite their significant success, these models still exhibit hallucination phenomena, where the outputs appear plausible but do not align with the content of the images. To mitigate this issue, we introduce Local Perception Search (LPS), a decoding method during inference that is both simple and training-free, yet effectively suppresses hallucinations. This method leverages local visual prior information as a value function to correct the decoding process. Additionally, we observe that the impact of the local visual prior on model performance is more pronounced in scenarios with high levels of image noise. Notably, LPS is a plug-and-play approach that is compatible with various models. Extensive experiments on widely used hallucination benchmarks and noisy data demonstrate that LPS significantly reduces the incidence of hallucinations compared to the baseline, showing exceptional performance, particularly in noisy settings. 

---
# MarginSel : Max-Margin Demonstration Selection for LLMs 

**Authors**: Rajeev Bhatt Ambati, James Lester, Shashank Srivastava, Snigdha Chaturvedi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06699)  

**Abstract**: Large Language Models (LLMs) excel at few-shot learning via in-context learning (ICL). However, the effectiveness of ICL is often sensitive to the selection and ordering of demonstration examples. To address this, we present MarginSel: Max-Margin Demonstration Selection for LLMs, a two-step method that selects hard demonstration examples for the ICL prompt, adapting to each test instance. Our approach achieves 2-7% absolute improvement in F1-score across classification tasks, compared to a random selection of examples. We also provide theoretical insights and empirical evidence showing that MarginSel induces max-margin behavior in LLMs by effectively increasing the margin for hard examples, analogous to support vectors, thereby shifting the decision boundary in a beneficial direction. 

---
# Contextual Experience Replay for Self-Improvement of Language Agents 

**Authors**: Yitao Liu, Chenglei Si, Karthik Narasimhan, Shunyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06698)  

**Abstract**: Large language model (LLM) agents have been applied to sequential decision-making tasks such as web navigation, but without any environment-specific experiences, they often fail in these complex tasks. Moreover, current LLM agents are not designed to continually learn from past experiences during inference time, which could be crucial for them to gain these environment-specific experiences. To address this, we propose Contextual Experience Replay (CER), a training-free framework to enable efficient self-improvement for language agents in their context window. Specifically, CER accumulates and synthesizes past experiences into a dynamic memory buffer. These experiences encompass environment dynamics and common decision-making patterns, allowing the agents to retrieve and augment themselves with relevant knowledge in new tasks, enhancing their adaptability in complex environments. We evaluate CER on the challenging WebArena and VisualWebArena benchmarks. On VisualWebArena, CER achieves a competitive performance of 31.9%. On WebArena, CER also gets a competitive average success rate of 36.7%, relatively improving the success rate of the GPT-4o agent baseline by 51.0%. We also conduct a comprehensive analysis on it to prove its efficiency, validity and understand it better. 

---
# Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning 

**Authors**: Shubham Parashar, Shurui Gui, Xiner Li, Hongyi Ling, Sushil Vemuri, Blake Olson, Eric Li, Yu Zhang, James Caverlee, Dileep Kalathil, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.06632)  

**Abstract**: We aim to improve the reasoning capabilities of language models via reinforcement learning (RL). Recent RL post-trained models like DeepSeek-R1 have demonstrated reasoning abilities on mathematical and coding tasks. However, prior studies suggest that using RL alone to improve reasoning on inherently difficult tasks is less effective. Here, we draw inspiration from curriculum learning and propose to schedule tasks from easy to hard (E2H), allowing LLMs to build reasoning skills gradually. Our method is termed E2H Reasoner. Empirically, we observe that, although easy tasks are important initially, fading them out through appropriate scheduling is essential in preventing overfitting. Theoretically, we establish convergence guarantees for E2H Reasoner within an approximate policy iteration framework. We derive finite-sample complexity bounds and show that when tasks are appropriately decomposed and conditioned, learning through curriculum stages requires fewer total samples than direct learning. Experiments across multiple domains show that E2H Reasoner significantly improves the reasoning ability of small LLMs (1.5B to 3B), which otherwise struggle when trained with vanilla RL alone, highlighting the effectiveness of our method. 

---
# Towards Efficient Multi-LLM Inference: Characterization and Analysis of LLM Routing and Hierarchical Techniques 

**Authors**: Adarsh Prasad Behera, Jaya Prakash Champati, Roberto Morabito, Sasu Tarkoma, James Gross  

**Link**: [PDF](https://arxiv.org/pdf/2506.06579)  

**Abstract**: Recent progress in Language Models (LMs) has dramatically advanced the field of natural language processing (NLP), excelling at tasks like text generation, summarization, and question answering. However, their inference remains computationally expensive and energy intensive, especially in settings with limited hardware, power, or bandwidth. This makes it difficult to deploy LMs in mobile, edge, or cost sensitive environments. To address these challenges, recent approaches have introduced multi LLM intelligent model selection strategies that dynamically allocate computational resources based on query complexity -- using lightweight models for simpler queries and escalating to larger models only when necessary. This survey explores two complementary strategies for efficient LLM inference: (i) routing, which selects the most suitable model based on the query, and (ii) cascading or hierarchical inference (HI), which escalates queries through a sequence of models until a confident response is found. Both approaches aim to reduce computation by using lightweight models for simpler tasks while offloading only when needed. We provide a comparative analysis of these techniques across key performance metrics, discuss benchmarking efforts, and outline open challenges. Finally, we outline future research directions to enable faster response times, adaptive model selection based on task complexity, and scalable deployment across heterogeneous environments, making LLM based systems more efficient and accessible for real world applications. 

---
# Future of Work with AI Agents: Auditing Automation and Augmentation Potential across the U.S. Workforce 

**Authors**: Yijia Shao, Humishka Zope, Yucheng Jiang, Jiaxin Pei, David Nguyen, Erik Brynjolfsson, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06576)  

**Abstract**: The rapid rise of compound AI systems (a.k.a., AI agents) is reshaping the labor market, raising concerns about job displacement, diminished human agency, and overreliance on automation. Yet, we lack a systematic understanding of the evolving landscape. In this paper, we address this gap by introducing a novel auditing framework to assess which occupational tasks workers want AI agents to automate or augment, and how those desires align with the current technological capabilities. Our framework features an audio-enhanced mini-interview to capture nuanced worker desires and introduces the Human Agency Scale (HAS) as a shared language to quantify the preferred level of human involvement. Using this framework, we construct the WORKBank database, building on the U.S. Department of Labor's O*NET database, to capture preferences from 1,500 domain workers and capability assessments from AI experts across over 844 tasks spanning 104 occupations. Jointly considering the desire and technological capability divides tasks in WORKBank into four zones: Automation "Green Light" Zone, Automation "Red Light" Zone, R&D Opportunity Zone, Low Priority Zone. This highlights critical mismatches and opportunities for AI agent development. Moving beyond a simple automate-or-not dichotomy, our results reveal diverse HAS profiles across occupations, reflecting heterogeneous expectations for human involvement. Moreover, our study offers early signals of how AI agent integration may reshape the core human competencies, shifting from information-focused skills to interpersonal ones. These findings underscore the importance of aligning AI agent development with human desires and preparing workers for evolving workplace dynamics. 

---
# Large Language Models Can Be a Viable Substitute for Expert Political Surveys When a Shock Disrupts Traditional Measurement Approaches 

**Authors**: Patrick Y. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06540)  

**Abstract**: After a disruptive event or shock, such as the Department of Government Efficiency (DOGE) federal layoffs of 2025, expert judgments are colored by knowledge of the outcome. This can make it difficult or impossible to reconstruct the pre-event perceptions needed to study the factors associated with the event. This position paper argues that large language models (LLMs), trained on vast amounts of digital media data, can be a viable substitute for expert political surveys when a shock disrupts traditional measurement. We analyze the DOGE layoffs as a specific case study for this position. We use pairwise comparison prompts with LLMs and derive ideology scores for federal executive agencies. These scores replicate pre-layoff expert measures and predict which agencies were targeted by DOGE. We also use this same approach and find that the perceptions of certain federal agencies as knowledge institutions predict which agencies were targeted by DOGE, even when controlling for ideology. This case study demonstrates that using LLMs allows us to rapidly and easily test the associated factors hypothesized behind the shock. More broadly, our case study of this recent event exemplifies how LLMs offer insights into the correlational factors of the shock when traditional measurement techniques fail. We conclude by proposing a two-part criterion for when researchers can turn to LLMs as a substitute for expert political surveys. 

---
# HeavyWater and SimplexWater: Watermarking Low-Entropy Text Distributions 

**Authors**: Dor Tsur, Carol Xuan Long, Claudio Mayrink Verdun, Hsiang Hsu, Chen-Fu Chen, Haim Permuter, Sajani Vithana, Flavio P. Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2506.06409)  

**Abstract**: Large language model (LLM) watermarks enable authentication of text provenance, curb misuse of machine-generated text, and promote trust in AI systems. Current watermarks operate by changing the next-token predictions output by an LLM. The updated (i.e., watermarked) predictions depend on random side information produced, for example, by hashing previously generated tokens. LLM watermarking is particularly challenging in low-entropy generation tasks - such as coding - where next-token predictions are near-deterministic. In this paper, we propose an optimization framework for watermark design. Our goal is to understand how to most effectively use random side information in order to maximize the likelihood of watermark detection and minimize the distortion of generated text. Our analysis informs the design of two new watermarks: HeavyWater and SimplexWater. Both watermarks are tunable, gracefully trading-off between detection accuracy and text distortion. They can also be applied to any LLM and are agnostic to side information generation. We examine the performance of HeavyWater and SimplexWater through several benchmarks, demonstrating that they can achieve high watermark detection accuracy with minimal compromise of text generation quality, particularly in the low-entropy regime. Our theoretical analysis also reveals surprising new connections between LLM watermarking and coding theory. The code implementation can be found in this https URL 

---
# From Rogue to Safe AI: The Role of Explicit Refusals in Aligning LLMs with International Humanitarian Law 

**Authors**: John Mavi, Diana Teodora Găitan, Sergio Coronado  

**Link**: [PDF](https://arxiv.org/pdf/2506.06391)  

**Abstract**: Large Language Models (LLMs) are widely used across sectors, yet their alignment with International Humanitarian Law (IHL) is not well understood. This study evaluates eight leading LLMs on their ability to refuse prompts that explicitly violate these legal frameworks, focusing also on helpfulness - how clearly and constructively refusals are communicated. While most models rejected unlawful requests, the clarity and consistency of their responses varied. By revealing the model's rationale and referencing relevant legal or safety principles, explanatory refusals clarify the system's boundaries, reduce ambiguity, and help prevent misuse. A standardised system-level safety prompt significantly improved the quality of the explanations expressed within refusals in most models, highlighting the effectiveness of lightweight interventions. However, more complex prompts involving technical language or requests for code revealed ongoing vulnerabilities. These findings contribute to the development of safer, more transparent AI systems and propose a benchmark to evaluate the compliance of LLM with IHL. 

---
# On the Fundamental Impossibility of Hallucination Control in Large Language Models 

**Authors**: Michał P. Karpowicz  

**Link**: [PDF](https://arxiv.org/pdf/2506.06382)  

**Abstract**: This paper explains \textbf{why it is impossible to create large language models that do not hallucinate and what are the trade-offs we should be looking for}. It presents a formal \textbf{impossibility theorem} demonstrating that no inference mechanism can simultaneously satisfy four fundamental properties: \textbf{truthful (non-hallucinatory) generation, semantic information conservation, relevant knowledge revelation, and knowledge-constrained optimality}. By modeling LLM inference as an \textbf{auction of ideas} where neural components compete to contribute to responses, we prove the impossibility using the Green-Laffont theorem. That mathematical framework provides a rigorous foundation for understanding the nature of inference process, with implications for model architecture, training objectives, and evaluation methods. 

---
# LLMs as World Models: Data-Driven and Human-Centered Pre-Event Simulation for Disaster Impact Assessment 

**Authors**: Lingyao Li, Dawei Li, Zhenhui Ou, Xiaoran Xu, Jingxiao Liu, Zihui Ma, Runlong Yu, Min Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06355)  

**Abstract**: Efficient simulation is essential for enhancing proactive preparedness for sudden-onset disasters such as earthquakes. Recent advancements in large language models (LLMs) as world models show promise in simulating complex scenarios. This study examines multiple LLMs to proactively estimate perceived earthquake impacts. Leveraging multimodal datasets including geospatial, socioeconomic, building, and street-level imagery data, our framework generates Modified Mercalli Intensity (MMI) predictions at zip code and county scales. Evaluations on the 2014 Napa and 2019 Ridgecrest earthquakes using USGS ''Did You Feel It? (DYFI)'' reports demonstrate significant alignment, as evidenced by a high correlation of 0.88 and a low RMSE of 0.77 as compared to real reports at the zip code level. Techniques such as RAG and ICL can improve simulation performance, while visual inputs notably enhance accuracy compared to structured numerical data alone. These findings show the promise of LLMs in simulating disaster impacts that can help strengthen pre-event planning. 

---
# Optimizing RAG Pipelines for Arabic: A Systematic Analysis of Core Components 

**Authors**: Jumana Alsubhi, Mohammad D. Alahmadi, Ahmed Alhusayni, Ibrahim Aldailami, Israa Hamdine, Ahmad Shabana, Yazeed Iskandar, Suhayb Khayyat  

**Link**: [PDF](https://arxiv.org/pdf/2506.06339)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful architecture for combining the precision of retrieval systems with the fluency of large language models. While several studies have investigated RAG pipelines for high-resource languages, the optimization of RAG components for Arabic remains underexplored. This study presents a comprehensive empirical evaluation of state-of-the-art RAG components-including chunking strategies, embedding models, rerankers, and language models-across a diverse set of Arabic datasets. Using the RAGAS framework, we systematically compare performance across four core metrics: context precision, context recall, answer faithfulness, and answer relevancy. Our experiments demonstrate that sentence-aware chunking outperforms all other segmentation methods, while BGE-M3 and Multilingual-E5-large emerge as the most effective embedding models. The inclusion of a reranker (bge-reranker-v2-m3) significantly boosts faithfulness in complex datasets, and Aya-8B surpasses StableLM in generation quality. These findings provide critical insights for building high-quality Arabic RAG pipelines and offer practical guidelines for selecting optimal components across different document types. 

---
# FinBERT2: A Specialized Bidirectional Encoder for Bridging the Gap in Finance-Specific Deployment of Large Language Models 

**Authors**: Xuan Xu, Fufang Wen, Beilin Chu, Zhibing Fu, Qinhong Lin, Jiaqi Liu, Binjie Fei, Zhongliang Yang, Linna Zhou, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06335)  

**Abstract**: In natural language processing (NLP), the focus has shifted from encoder-only tiny language models like BERT to decoder-only large language models(LLMs) such as GPT-3. However, LLMs' practical application in the financial sector has revealed three limitations: (1) LLMs often perform worse than fine-tuned BERT on discriminative tasks despite costing much higher computational resources, such as market sentiment analysis in financial reports; (2) Application on generative tasks heavily relies on retrieval augmented generation (RAG) methods to provide current and specialized information, with general retrievers showing suboptimal performance on domain-specific retrieval tasks; (3) There are additional inadequacies in other feature-based scenarios, such as topic modeling. We introduce FinBERT2, a specialized bidirectional encoder pretrained on a high-quality, financial-specific corpus of 32b tokens. This represents the largest known Chinese financial pretraining corpus for models of this parameter size. As a better backbone, FinBERT2 can bridge the gap in the financial-specific deployment of LLMs through the following achievements: (1) Discriminative fine-tuned models (Fin-Labelers) outperform other (Fin)BERT variants by 0.4%-3.3% and leading LLMs by 9.7%-12.3% on average across five financial classification tasks. (2) Contrastive fine-tuned models (Fin-Retrievers) outperform both open-source (e.g., +6.8\% avg improvement over BGE-base-zh) and proprietary (e.g., +4.2\% avg improvement over OpenAI's text-embedding-3-large) embedders across five financial retrieval tasks; (3) Building on FinBERT2 variants, we construct the Fin-TopicModel, which enables superior clustering and topic representation for financial titles. Our work revisits financial BERT models through comparative analysis with contemporary LLMs and offers practical insights for effectively utilizing FinBERT in the LLMs era. 

---
# The Hype Index: an NLP-driven Measure of Market News Attention 

**Authors**: Zheng Cao, Wanchaloem Wunkaew, Helyette Geman  

**Link**: [PDF](https://arxiv.org/pdf/2506.06329)  

**Abstract**: This paper introduces the Hype Index as a novel metric to quantify media attention toward large-cap equities, leveraging advances in Natural Language Processing (NLP) for extracting predictive signals from financial news. Using the S&P 100 as the focus universe, we first construct a News Count-Based Hype Index, which measures relative media exposure by computing the share of news articles referencing each stock or sector. We then extend it to the Capitalization Adjusted Hype Index, adjusts for economic size by taking the ratio of a stock's or sector's media weight to its market capitalization weight within its industry or sector. We compute both versions of the Hype Index at the stock and sector levels, and evaluate them through multiple lenses: (1) their classification into different hype groups, (2) their associations with returns, volatility, and VIX index at various lags, (3) their signaling power for short-term market movements, and (4) their empirical properties including correlations, samplings, and trends. Our findings suggest that the Hype Index family provides a valuable set of tools for stock volatility analysis, market signaling, and NLP extensions in Finance. 

---
# Is BERTopic Better than PLSA for Extracting Key Topics in Aviation Safety Reports? 

**Authors**: Aziida Nanyonga, Joiner Keith, Turhan Ugur, Wild Graham  

**Link**: [PDF](https://arxiv.org/pdf/2506.06328)  

**Abstract**: This study compares the effectiveness of BERTopic and Probabilistic Latent Semantic Analysis (PLSA) in extracting meaningful topics from aviation safety reports aiming to enhance the understanding of patterns in aviation incident data. Using a dataset of over 36,000 National Transportation Safety Board (NTSB) reports from 2000 to 2020, BERTopic employed transformer based embeddings and hierarchical clustering, while PLSA utilized probabilistic modelling through the Expectation-Maximization (EM) algorithm. Results showed that BERTopic outperformed PLSA in topic coherence, achieving a Cv score of 0.41 compared to PLSA 0.37, while also demonstrating superior interpretability as validated by aviation safety experts. These findings underscore the advantages of modern transformer based approaches in analyzing complex aviation datasets, paving the way for enhanced insights and informed decision-making in aviation safety. Future work will explore hybrid models, multilingual datasets, and advanced clustering techniques to further improve topic modelling in this domain. 

---
# DISRetrieval: Harnessing Discourse Structure for Long Document Retrieval 

**Authors**: Huiyao Chen, Yi Yang, Yinghui Li, Meishan Zhang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06313)  

**Abstract**: Long document understanding has become increasingly crucial in natural language processing, with retrieval-based methods emerging as a promising solution to address the context length limitations of large language models (LLMs). However, existing approaches either treat documents as flat sequences or employ arbitrary chunking strategies, failing to capture the inherent discourse structure that guides human comprehension. We present DISRetrieval, a novel hierarchical retrieval framework that leverages linguistic discourse structure to enhance long document understanding. Our approach introduces three key innovations: (1) a discourse-aware document organization framework that utilizes rhetorical structure theory (RST) to create sentence-level hierarchical representations, preserving both semantic relationships and natural document flow; (2) an LLM-enhanced node representation technique that combines discourse structure with adaptive summarization to enrich tree nodes with contextual information; and (3) a hierarchical evidence retrieval mechanism that effectively selects relevant content while maintaining discourse coherence. Through comprehensive experiments on QASPER and QuALITY datasets, DISRetrieval demonstrates substantial improvements over existing methods in both token-level retrieval metrics and downstream question answering tasks. Our ablation studies confirm that incorporating discourse structure significantly enhances retrieval effectiveness across different document lengths and query types, validating the importance of linguistically-informed document representation in long-text understanding. Our code and datasets are publicly available at github/DreamH1gh/DISRetrieval to facilitate future research. 

---
# Reward Is Enough: LLMs Are In-Context Reinforcement Learners 

**Authors**: Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Yanjun Qi, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06303)  

**Abstract**: Reinforcement learning (RL) is a human-designed framework for solving sequential decision making problems. In this work, we demonstrate that, surprisingly, RL emerges in LLM's (Large Language Model) inference time -- a phenomenon known as in-context RL (ICRL). Specifically, we propose a novel multi-round prompting framework called ICRL prompting. The goal is to prompt the LLM to complete a task. After the LLM generates a response at the current round, we give numerical scalar feedbacks for the response, called the rewards. At the next round, we prompt the LLM again with the same task and a context consisting of all previous responses and rewards. We observe that the quality of the LLM's response increases as the context grows. In other words, the LLM is able to maximize the scalar reward signal in the inference time, just like an RL algorithm. We evaluate ICRL prompting in three benchmarks (Game of 24, creative writing, and ScienceWorld) and demonstrate significant performance improvements over baseline methods such as Self-Refine and Reflexion. Surprisingly, in some experiments the reward signals are generated by the LLM itself, yet performance improvements are still observed from ICRL prompting, offering a promising paradigm for scaling test-time compute. 

---
# How Malicious AI Swarms Can Threaten Democracy 

**Authors**: Daniel Thilo Schroeder, Meeyoung Cha, Andrea Baronchelli, Nick Bostrom, Nicholas A. Christakis, David Garcia, Amit Goldenberg, Yara Kyrychenko, Kevin Leyton-Brown, Nina Lutz, Gary Marcus, Filippo Menczer, Gordon Pennycook, David G. Rand, Frank Schweitzer, Christopher Summerfield, Audrey Tang, Jay Van Bavel, Sander van der Linden, Dawn Song, Jonas R. Kunst  

**Link**: [PDF](https://arxiv.org/pdf/2506.06299)  

**Abstract**: Advances in AI portend a new era of sophisticated disinformation operations. While individual AI systems already create convincing -- and at times misleading -- information, an imminent development is the emergence of malicious AI swarms. These systems can coordinate covertly, infiltrate communities, evade traditional detectors, and run continuous A/B tests, with round-the-clock persistence. The result can include fabricated grassroots consensus, fragmented shared reality, mass harassment, voter micro-suppression or mobilization, contamination of AI training data, and erosion of institutional trust. With democratic processes worldwide increasingly vulnerable, we urge a three-pronged response: (1) platform-side defenses -- always-on swarm-detection dashboards, pre-election high-fidelity swarm-simulation stress-tests, transparency audits, and optional client-side "AI shields" for users; (2) model-side safeguards -- standardized persuasion-risk tests, provenance-authenticating passkeys, and watermarking; and (3) system-level oversight -- a UN-backed AI Influence Observatory. 

---
# dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching 

**Authors**: Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06295)  

**Abstract**: Autoregressive Models (ARMs) have long dominated the landscape of Large Language Models. Recently, a new paradigm has emerged in the form of diffusion-based Large Language Models (dLLMs), which generate text by iteratively denoising masked segments. This approach has shown significant advantages and potential. However, dLLMs suffer from high inference latency. Traditional ARM acceleration techniques, such as Key-Value caching, are incompatible with dLLMs due to their bidirectional attention mechanism. To address this specific challenge, our work begins with a key observation that dLLM inference involves a static prompt and a partially dynamic response, where most tokens remain stable across adjacent denoising steps. Based on this, we propose dLLM-Cache, a training-free adaptive caching framework that combines long-interval prompt caching with partial response updates guided by feature similarity. This design enables efficient reuse of intermediate computations without compromising model performance. Extensive experiments on representative dLLMs, including LLaDA 8B and Dream 7B, show that dLLM-Cache achieves up to 9.1 x speedup over standard inference without compromising output quality. Notably, our method brings dLLM inference latency close to that of ARMs under many settings. Codes are provided in the supplementary material and will be released publicly on GitHub. 

---
# GLProtein: Global-and-Local Structure Aware Protein Representation Learning 

**Authors**: Yunqing Liu, Wenqi Fan, Xiaoyong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06294)  

**Abstract**: Proteins are central to biological systems, participating as building blocks across all forms of life. Despite advancements in understanding protein functions through protein sequence analysis, there remains potential for further exploration in integrating protein structural information. We argue that the structural information of proteins is not only limited to their 3D information but also encompasses information from amino acid molecules (local information) to protein-protein structure similarity (global information). To address this, we propose \textbf{GLProtein}, the first framework in protein pre-training that incorporates both global structural similarity and local amino acid details to enhance prediction accuracy and functional insights. GLProtein innovatively combines protein-masked modelling with triplet structure similarity scoring, protein 3D distance encoding and substructure-based amino acid molecule encoding. Experimental results demonstrate that GLProtein outperforms previous methods in several bioinformatics tasks, including predicting protein-protein interaction, contact prediction, and so on. 

---
