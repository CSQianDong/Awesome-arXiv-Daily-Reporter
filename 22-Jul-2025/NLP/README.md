# 3LM: Bridging Arabic, STEM, and Code through Benchmarking 

**Authors**: Basma El Amel Boussaha, Leen AlQadi, Mugariya Farooq, Shaikha Alsuwaidi, Giulia Campesan, Ahmed Alzubaidi, Mohammed Alyafeai, Hakim Hacid  

**Link**: [PDF](https://arxiv.org/pdf/2507.15850)  

**Abstract**: Arabic is one of the most widely spoken languages in the world, yet efforts to develop and evaluate Large Language Models (LLMs) for Arabic remain relatively limited. Most existing Arabic benchmarks focus on linguistic, cultural, or religious content, leaving a significant gap in domains like STEM and code which are increasingly relevant for real-world LLM applications. To help bridge this gap, we present 3LM, a suite of three benchmarks designed specifically for Arabic. The first is a set of STEM-related question-answer pairs, naturally sourced from Arabic textbooks and educational worksheets. The second consists of synthetically generated STEM questions, created using the same sources. The third benchmark focuses on code generation, built through a careful translation of two widely used code benchmarks, incorporating a human-in-the-loop process with several rounds of review to ensure high-quality and faithful translations. We release all three benchmarks publicly to support the growth of Arabic LLM research in these essential but underrepresented areas. 

---
# The Impact of Language Mixing on Bilingual LLM Reasoning 

**Authors**: Yihao Li, Jiayi Xin, Miranda Muqing Miao, Qi Long, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2507.15849)  

**Abstract**: Proficient multilingual speakers often intentionally switch languages in the middle of a conversation. Similarly, recent reasoning-focused bilingual large language models (LLMs) with strong capabilities in both languages exhibit language mixing--alternating languages within their chain of thought. Discouraging this behavior in DeepSeek-R1 was found to degrade accuracy, suggesting that language mixing may benefit reasoning. In this work, we study language switching in Chinese-English bilingual reasoning models. We identify reinforcement learning with verifiable rewards (RLVR) as the critical training stage that leads to language mixing. We demonstrate that language mixing can enhance reasoning: enforcing monolingual decoding reduces accuracy by 5.6 percentage points on math reasoning tasks. Additionally, a lightweight probe can be trained to predict whether a potential language switch would benefit or harm reasoning, and when used to guide decoding, increases accuracy by up to 6.25 percentage points. Our findings suggest that language mixing is not merely a byproduct of multilingual training, but is a strategic reasoning behavior. 

---
# Operationalizing AI for Good: Spotlight on Deployment and Integration of AI Models in Humanitarian Work 

**Authors**: Anton Abilov, Ke Zhang, Hemank Lamba, Elizabeth M. Olson, Joel R. Tetreault, Alejandro Jaimes  

**Link**: [PDF](https://arxiv.org/pdf/2507.15823)  

**Abstract**: Publications in the AI for Good space have tended to focus on the research and model development that can support high-impact applications. However, very few AI for Good papers discuss the process of deploying and collaborating with the partner organization, and the resulting real-world impact. In this work, we share details about the close collaboration with a humanitarian-to-humanitarian (H2H) organization and how to not only deploy the AI model in a resource-constrained environment, but also how to maintain it for continuous performance updates, and share key takeaways for practitioners. 

---
# Reservoir Computing as a Language Model 

**Authors**: Felix Köster, Atsushi Uchida  

**Link**: [PDF](https://arxiv.org/pdf/2507.15779)  

**Abstract**: Large Language Models (LLM) have dominated the science and media landscape duo to their impressive performance on processing large chunks of data and produce human-like levels of text. Nevertheless, their huge energy demand and slow processing still a bottleneck for further increasing quality while also making the models accessible to everyone. To solve this bottleneck, we will investigate how reservoir computing performs on natural text processing, which could enable fast and energy efficient hardware implementations. Studies investigating the use of reservoir computing as a language model remain sparse. In this paper, we compare three distinct approaches for character-level language modeling, two different reservoir computing approaches, where only an output layer is trainable, and the well-known transformer-based architectures, which fully learn an attention-based sequence representation. We explore the performance, computational cost and prediction accuracy for both paradigms by equally varying the number of trainable parameters for all models. Using a consistent pipeline for all three approaches, we demonstrate that transformers excel in prediction quality, whereas reservoir computers remain highly efficient reducing the training and inference speed. Furthermore, we investigate two types of reservoir computing: a traditional reservoir with a static linear readout, and an attention-enhanced reservoir that dynamically adapts its output weights via an attention mechanism. Our findings underline how these paradigms scale and offer guidelines to balance resource constraints with performance. 

---
# Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR 

**Authors**: Jiakang Wang, Runze Liu, Fuzheng Zhang, Xiu Li, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.15778)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become an effective post-training method for improving the reasoning abilities of Large Language Models (LLMs), mainly by shaping higher-order behaviors such as reflection and planning. However, previous RLVR algorithms often apply uniform training signals to all tokens, without considering the different roles of low-entropy knowledge-related tokens and high-entropy reasoning-related tokens. Some recent methods try to separate these token types by gradient masking or asynchronous updates, but these approaches may break semantic dependencies in the model output and hinder effective learning. In this work, we propose Archer, an entropy-aware RLVR approach with dual-token constraints and synchronous updates. Specifically, our method applies weaker KL regularization and higher clipping thresholds to reasoning tokens to encourage exploration, while using stronger constraints on knowledge tokens to maintain factual knowledge. Experimental results on several mathematical reasoning and code generation benchmarks show that our approach significantly outperforms previous RLVR methods, reaching or exceeding state-of-the-art performance among models of comparable size. The code is available at this https URL. 

---
# Supernova: Achieving More with Less in Transformer Architectures 

**Authors**: Andrei-Valentin Tanase, Elena Pelican  

**Link**: [PDF](https://arxiv.org/pdf/2507.15773)  

**Abstract**: We present Supernova, a 650M-parameter decoder-only transformer that demonstrates how careful architectural design and tokenization innovation can achieve the performance of larger models while maintaining computational efficiency. Our architecture combines Rotary Positional Embeddings (RoPE), Grouped Query Attention (GQA) with a 3:1 compression ratio, RMSNorm for computational efficiency, and SwiGLU activation functions. A critical innovation is our custom 128,000-vocabulary byte-level BPE tokenizer, which achieves state-of-the-art compression performance. Through detailed analysis, we show that Supernova achieves 90% of the performance of 1B-parameter models while using 53% fewer parameters and requiring only 100B training tokens--an order of magnitude less than competing models. Our findings challenge the prevailing scaling paradigm, demonstrating that architectural efficiency and tokenization quality can compensate for reduced parameter counts. 

---
# Interaction as Intelligence: Deep Research With Human-AI Partnership 

**Authors**: Lyumanshan Ye, Xiaojie Cai, Xinkai Wang, Junfei Wang, Xiangkun Hu, Jiadi Su, Yang Nan, Sihan Wang, Bohan Zhang, Xiaoze Fan, Jinbin Luo, Yuxiang Zheng, Tianze Xu, Dayuan Fu, Yunze Wu, Pengrui Lu, Zengzhi Wang, Yiwei Qin, Zhen Huang, Yan Ma, Zhulin Hu, Haoyang Zou, Tiantian Mi, Yixin Ye, Ethan Chern, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.15759)  

**Abstract**: This paper introduces "Interaction as Intelligence" research series, presenting a reconceptualization of human-AI relationships in deep research tasks. Traditional approaches treat interaction merely as an interface for accessing AI capabilities-a conduit between human intent and machine output. We propose that interaction itself constitutes a fundamental dimension of intelligence. As AI systems engage in extended thinking processes for research tasks, meaningful interaction transitions from an optional enhancement to an essential component of effective intelligence. Current deep research systems adopt an "input-wait-output" paradigm where users initiate queries and receive results after black-box processing. This approach leads to error cascade effects, inflexible research boundaries that prevent question refinement during investigation, and missed opportunities for expertise integration. To address these limitations, we introduce Deep Cognition, a system that transforms the human role from giving instructions to cognitive oversight-a mode of engagement where humans guide AI thinking processes through strategic intervention at critical junctures. Deep cognition implements three key innovations: (1)Transparent, controllable, and interruptible interaction that reveals AI reasoning and enables intervention at any point; (2)Fine-grained bidirectional dialogue; and (3)Shared cognitive context where the system observes and adapts to user behaviors without explicit instruction. User evaluation demonstrates that this cognitive oversight paradigm outperforms the strongest baseline across six key metrics: Transparency(+20.0%), Fine-Grained Interaction(+29.2%), Real-Time Intervention(+18.5%), Ease of Collaboration(+27.7%), Results-Worth-Effort(+8.8%), and Interruptibility(+20.7%). Evaluations on challenging research problems show 31.8% to 50.0% points of improvements over deep research systems. 

---
# DialogueForge: LLM Simulation of Human-Chatbot Dialogue 

**Authors**: Ruizhe Zhu, Hao Zhu, Yaxuan Li, Syang Zhou, Shijing Cai, Malgorzata Lazuka, Elliott Ash  

**Link**: [PDF](https://arxiv.org/pdf/2507.15752)  

**Abstract**: Collecting human-chatbot dialogues typically demands substantial manual effort and is time-consuming, which limits and poses challenges for research on conversational AI. In this work, we propose DialogueForge - a framework for generating AI-simulated conversations in human-chatbot style. To initialize each generated conversation, DialogueForge uses seed prompts extracted from real human-chatbot interactions. We test a variety of LLMs to simulate the human chatbot user, ranging from state-of-the-art proprietary models to small-scale open-source LLMs, and generate multi-turn dialogues tailored to specific tasks. In addition, we explore fine-tuning techniques to enhance the ability of smaller models to produce indistinguishable human-like dialogues. We evaluate the quality of the simulated conversations and compare different models using the UniEval and GTEval evaluation protocols. Our experiments show that large proprietary models (e.g., GPT-4o) generally outperform others in generating more realistic dialogues, while smaller open-source models (e.g., Llama, Mistral) offer promising performance with greater customization. We demonstrate that the performance of smaller models can be significantly improved by employing supervised fine-tuning techniques. Nevertheless, maintaining coherent and natural long-form human-like dialogues remains a common challenge across all models. 

---
# A Fisher's exact test justification of the TF-IDF term-weighting scheme 

**Authors**: Paul Sheridan, Zeyad Ahmed, Aitazaz A. Farooque  

**Link**: [PDF](https://arxiv.org/pdf/2507.15742)  

**Abstract**: Term frequency-inverse document frequency, or TF-IDF for short, is arguably the most celebrated mathematical expression in the history of information retrieval. Conceived as a simple heuristic quantifying the extent to which a given term's occurrences are concentrated in any one given document out of many, TF-IDF and its many variants are routinely used as term-weighting schemes in diverse text analysis applications. There is a growing body of scholarship dedicated to placing TF-IDF on a sound theoretical foundation. Building on that tradition, this paper justifies the use of TF-IDF to the statistics community by demonstrating how the famed expression can be understood from a significance testing perspective. We show that the common TF-IDF variant TF-ICF is, under mild regularity conditions, closely related to the negative logarithm of the $p$-value from a one-tailed version of Fisher's exact test of statistical significance. As a corollary, we establish a connection between TF-IDF and the said negative log-transformed $p$-value under certain idealized assumptions. We further demonstrate, as a limiting case, that this same quantity converges to TF-IDF in the limit of an infinitely large document collection. The Fisher's exact test justification of TF-IDF equips the working statistician with a ready explanation of the term-weighting scheme's long-established effectiveness. 

---
# Understanding Large Language Models' Ability on Interdisciplinary Research 

**Authors**: Yuanhao Shen, Daniel Xavier de Sousa, Ricardo Marçal, Ali Asad, Hongyu Guo, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.15736)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have revealed their impressive ability to perform multi-step, logic-driven reasoning across complex domains, positioning them as powerful tools and collaborators in scientific discovery while challenging the long-held view that inspiration-driven ideation is uniquely human. However, the lack of a dedicated benchmark that evaluates LLMs' ability to develop ideas in Interdisciplinary Research (IDR) settings poses a critical barrier to fully understanding their strengths and limitations. To address this gap, we introduce IDRBench -- a pioneering benchmark featuring an expert annotated dataset and a suite of tasks tailored to evaluate LLMs' capabilities in proposing valuable research ideas from different scientific domains for interdisciplinary research. This benchmark aims to provide a systematic framework for assessing LLM performance in complex, cross-domain scientific research. Our dataset consists of scientific publications sourced from the ArXiv platform covering six distinct disciplines, and is annotated by domain experts with diverse academic backgrounds. To ensure high-quality annotations, we emphasize clearly defined dimensions that characterize authentic interdisciplinary research. The design of evaluation tasks in IDRBench follows a progressive, real-world perspective, reflecting the natural stages of interdisciplinary research development, including 1) IDR Paper Identification, 2) IDR Idea Integration, and 3) IDR Idea Recommendation. Using IDRBench, we construct baselines across 10 LLMs and observe that despite fostering some level of IDR awareness, LLMs still struggle to produce quality IDR ideas. These findings could not only spark new research directions, but also help to develop next-generation LLMs that excel in interdisciplinary research. 

---
# BEnchmarking LLMs for Ophthalmology (BELO) for Ophthalmological Knowledge and Reasoning 

**Authors**: Sahana Srinivasan, Xuguang Ai, Thaddaeus Wai Soon Lo, Aidan Gilson, Minjie Zou, Ke Zou, Hyunjae Kim, Mingjia Yang, Krithi Pushpanathan, Samantha Yew, Wan Ting Loke, Jocelyn Goh, Yibing Chen, Yiming Kong, Emily Yuelei Fu, Michelle Ongyong Hui, Kristen Nwanyanwu, Amisha Dave, Kelvin Zhenghao Li, Chen-Hsin Sun, Mark Chia, Gabriel Dawei Yang, Wendy Meihua Wong, David Ziyou Chen, Dianbo Liu, Maxwell Singer, Fares Antaki, Lucian V Del Priore, Jost Jonas, Ron Adelman, Qingyu Chen, Yih-Chung Tham  

**Link**: [PDF](https://arxiv.org/pdf/2507.15717)  

**Abstract**: Current benchmarks evaluating large language models (LLMs) in ophthalmology are limited in scope and disproportionately prioritise accuracy. We introduce BELO (BEnchmarking LLMs for Ophthalmology), a standardized and comprehensive evaluation benchmark developed through multiple rounds of expert checking by 13 ophthalmologists. BELO assesses ophthalmology-related clinical accuracy and reasoning quality. Using keyword matching and a fine-tuned PubMedBERT model, we curated ophthalmology-specific multiple-choice-questions (MCQs) from diverse medical datasets (BCSC, MedMCQA, MedQA, BioASQ, and PubMedQA). The dataset underwent multiple rounds of expert checking. Duplicate and substandard questions were systematically removed. Ten ophthalmologists refined the explanations of each MCQ's correct answer. This was further adjudicated by three senior ophthalmologists. To illustrate BELO's utility, we evaluated six LLMs (OpenAI o1, o3-mini, GPT-4o, DeepSeek-R1, Llama-3-8B, and Gemini 1.5 Pro) using accuracy, macro-F1, and five text-generation metrics (ROUGE-L, BERTScore, BARTScore, METEOR, and AlignScore). In a further evaluation involving human experts, two ophthalmologists qualitatively reviewed 50 randomly selected outputs for accuracy, comprehensiveness, and completeness. BELO consists of 900 high-quality, expert-reviewed questions aggregated from five sources: BCSC (260), BioASQ (10), MedMCQA (572), MedQA (40), and PubMedQA (18). A public leaderboard has been established to promote transparent evaluation and reporting. Importantly, the BELO dataset will remain a hold-out, evaluation-only benchmark to ensure fair and reproducible comparisons of future models. 

---
# From Queries to Criteria: Understanding How Astronomers Evaluate LLMs 

**Authors**: Alina Hyk, Kiera McCormick, Mian Zhong, Ioana Ciucă, Sanjib Sharma, John F Wu, J. E. G. Peek, Kartheik G. Iyer, Ziang Xiao, Anjalie Field  

**Link**: [PDF](https://arxiv.org/pdf/2507.15715)  

**Abstract**: There is growing interest in leveraging LLMs to aid in astronomy and other scientific research, but benchmarks for LLM evaluation in general have not kept pace with the increasingly diverse ways that real people evaluate and use these models. In this study, we seek to improve evaluation procedures by building an understanding of how users evaluate LLMs. We focus on a particular use case: an LLM-powered retrieval-augmented generation bot for engaging with astronomical literature, which we deployed via Slack. Our inductive coding of 368 queries to the bot over four weeks and our follow-up interviews with 11 astronomers reveal how humans evaluated this system, including the types of questions asked and the criteria for judging responses. We synthesize our findings into concrete recommendations for building better benchmarks, which we then employ in constructing a sample benchmark for evaluating LLMs for astronomy. Overall, our work offers ways to improve LLM evaluation and ultimately usability, particularly for use in scientific research. 

---
# Chinchunmei at SemEval-2025 Task 11: Boosting the Large Language Model's Capability of Emotion Perception using Contrastive Learning 

**Authors**: Tian Li, Yujian Sun, Huizhi Liang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15714)  

**Abstract**: The SemEval-2025 Task 11, Bridging the Gap in Text-Based Emotion Detection, introduces an emotion recognition challenge spanning over 28 languages. This competition encourages researchers to explore more advanced approaches to address the challenges posed by the diversity of emotional expressions and background variations. It features two tracks: multi-label classification (Track A) and emotion intensity prediction (Track B), covering six emotion categories: anger, fear, joy, sadness, surprise, and disgust. In our work, we systematically explore the benefits of two contrastive learning approaches: sample-based (Contrastive Reasoning Calibration) and generation-based (DPO, SimPO) contrastive learning. The sample-based contrastive approach trains the model by comparing two samples to generate more reliable predictions. The generation-based contrastive approach trains the model to differentiate between correct and incorrect generations, refining its prediction. All models are fine-tuned from LLaMa3-Instruct-8B. Our system achieves 9th place in Track A and 6th place in Track B for English, while ranking among the top-tier performing systems for other languages. 

---
# Is Large Language Model Performance on Reasoning Tasks Impacted by Different Ways Questions Are Asked? 

**Authors**: Seok Hwan Song, Mohna Chakraborty, Qi Li, Wallapak Tavanapong  

**Link**: [PDF](https://arxiv.org/pdf/2507.15707)  

**Abstract**: Large Language Models (LLMs) have been evaluated using diverse question types, e.g., multiple-choice, true/false, and short/long answers. This study answers an unexplored question about the impact of different question types on LLM accuracy on reasoning tasks. We investigate the performance of five LLMs on three different types of questions using quantitative and deductive reasoning tasks. The performance metrics include accuracy in the reasoning steps and choosing the final answer. Key Findings: (1) Significant differences exist in LLM performance across different question types. (2) Reasoning accuracy does not necessarily correlate with the final selection accuracy. (3) The number of options and the choice of words, influence LLM performance. 

---
# Compositional Understanding in Signaling Games 

**Authors**: David Peter Wallis Freeborn  

**Link**: [PDF](https://arxiv.org/pdf/2507.15706)  

**Abstract**: Receivers in standard signaling game models struggle with learning compositional information. Even when the signalers send compositional messages, the receivers do not interpret them compositionally. When information from one message component is lost or forgotten, the information from other components is also erased. In this paper I construct signaling game models in which genuine compositional understanding evolves. I present two new models: a minimalist receiver who only learns from the atomic messages of a signal, and a generalist receiver who learns from all of the available information. These models are in many ways simpler than previous alternatives, and allow the receivers to learn from the atomic components of messages. 

---
# CoLD: Counterfactually-Guided Length Debiasing for Process Reward Models 

**Authors**: Congmin Zheng, Jiachen Zhu, Jianghao Lin, Xinyi Dai, Yong Yu, Weinan Zhang, Mengyue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15698)  

**Abstract**: Process Reward Models (PRMs) play a central role in evaluating and guiding multi-step reasoning in large language models (LLMs), especially for mathematical problem solving. However, we identify a pervasive length bias in existing PRMs: they tend to assign higher scores to longer reasoning steps, even when the semantic content and logical validity are unchanged. This bias undermines the reliability of reward predictions and leads to overly verbose outputs during inference. To address this issue, we propose CoLD(Counterfactually-Guided Length Debiasing), a unified framework that mitigates length bias through three components: an explicit length-penalty adjustment, a learned bias estimator trained to capture spurious length-related signals, and a joint training strategy that enforces length-invariance in reward predictions. Our approach is grounded in counterfactual reasoning and informed by causal graph analysis. Extensive experiments on MATH500 and GSM-Plus show that CoLD consistently reduces reward-length correlation, improves accuracy in step selection, and encourages more concise, logically valid reasoning. These results demonstrate the effectiveness and practicality of CoLD in improving the fidelity and robustness of PRMs. 

---
# P3: Prompts Promote Prompting 

**Authors**: Xinyu Zhang, Yuanquan Hu, Fangchao Liu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2507.15675)  

**Abstract**: Current large language model (LLM) applications often employ multi-component prompts, comprising both system and user prompts, to guide model behaviors. While recent advancements have demonstrated the efficacy of automatically optimizing either the system or user prompt to boost performance, such unilateral approaches often yield suboptimal outcomes due to the interdependent nature of these components. In this work, we introduce P3, a novel self-improvement framework that concurrently optimizes both system and user prompts through an iterative process. The offline optimized prompts are further leveraged to promote online prompting by performing query-dependent prompt optimization. Extensive experiments on general tasks (e.g., Arena-hard and Alpaca-eval) and reasoning tasks (e.g., GSM8K and GPQA) demonstrate that P3 achieves superior performance in the realm of automatic prompt optimization. Our results highlight the effectiveness of a holistic optimization strategy in enhancing LLM performance across diverse domains. 

---
# Leveraging Context for Multimodal Fallacy Classification in Political Debates 

**Authors**: Alessio Pittiglio  

**Link**: [PDF](https://arxiv.org/pdf/2507.15641)  

**Abstract**: In this paper, we present our submission to the MM-ArgFallacy2025 shared task, which aims to advance research in multimodal argument mining, focusing on logical fallacies in political debates. Our approach uses pretrained Transformer-based models and proposes several ways to leverage context. In the fallacy classification subtask, our models achieved macro F1-scores of 0.4444 (text), 0.3559 (audio), and 0.4403 (multimodal). Our multimodal model showed performance comparable to the text-only model, suggesting potential for improvements. 

---
# Conflicting narratives and polarization on social media 

**Authors**: Armin Pournaki  

**Link**: [PDF](https://arxiv.org/pdf/2507.15600)  

**Abstract**: Narratives are key interpretative devices by which humans make sense of political reality. In this work, we show how the analysis of conflicting narratives, i.e. conflicting interpretive lenses through which political reality is experienced and told, provides insight into the discursive mechanisms of polarization and issue alignment in the public sphere. Building upon previous work that has identified ideologically polarized issues in the German Twittersphere between 2021 and 2023, we analyze the discursive dimension of polarization by extracting textual signals of conflicting narratives from tweets of opposing opinion groups. Focusing on a selection of salient issues and events (the war in Ukraine, Covid, climate change), we show evidence for conflicting narratives along two dimensions: (i) different attributions of actantial roles to the same set of actants (e.g. diverging interpretations of the role of NATO in the war in Ukraine), and (ii) emplotment of different actants for the same event (e.g. Bill Gates in the right-leaning Covid narrative). Furthermore, we provide first evidence for patterns of narrative alignment, a discursive strategy that political actors employ to align opinions across issues. These findings demonstrate the use of narratives as an analytical lens into the discursive mechanisms of polarization. 

---
# Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation 

**Authors**: Xinping Zhao, Shouzheng Huang, Yan Zhong, Xinshuo Hu, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15586)  

**Abstract**: Retrieval-Augmented Generation (RAG) effectively improves the accuracy of Large Language Models (LLMs). However, retrieval noises significantly impact the quality of LLMs' generation, necessitating the development of denoising mechanisms. Previous methods extract evidence straightforwardly without explicit thinking, which risks filtering out key clues and struggles with generalization. To this end, we propose LEAR, which learns to extract rational evidence by (1) explicitly reasoning to identify potential cues within retrieval contents first, and then (2) consciously extracting to avoid omitting any key cues helpful for answering questions. Specifically, we frame evidence reasoning and evidence extraction into one unified response for end-to-end training; apply knowledge token masks for disentanglement to derive reasoning-based and extraction-based answers; and devise three types of verifiable reward functions, including answer, length, and format, to update the model via the policy optimization algorithm. Extensive experiments on three benchmark datasets show the effectiveness of LEAR, providing compact and high-quality evidence, improving the accuracy of downstream tasks, and promoting effective application in online RAG systems. 

---
# Smart Eyes for Silent Threats: VLMs and In-Context Learning for THz Imaging 

**Authors**: Nicolas Poggi, Shashank Agnihotri, Margret Keuper  

**Link**: [PDF](https://arxiv.org/pdf/2507.15576)  

**Abstract**: Terahertz (THz) imaging enables non-invasive analysis for applications such as security screening and material classification, but effective image classification remains challenging due to limited annotations, low resolution, and visual ambiguity. We introduce In-Context Learning (ICL) with Vision-Language Models (VLMs) as a flexible, interpretable alternative that requires no fine-tuning. Using a modality-aligned prompting framework, we adapt two open-weight VLMs to the THz domain and evaluate them under zero-shot and one-shot settings. Our results show that ICL improves classification and interpretability in low-data regimes. This is the first application of ICL-enhanced VLMs to THz imaging, offering a promising direction for resource-constrained scientific domains. Code: \href{this https URL}{GitHub repository}. 

---
# Evaluating Text Style Transfer: A Nine-Language Benchmark for Text Detoxification 

**Authors**: Vitaly Protasov, Nikolay Babakov, Daryna Dementieva, Alexander Panchenko  

**Link**: [PDF](https://arxiv.org/pdf/2507.15557)  

**Abstract**: Despite recent progress in large language models (LLMs), evaluation of text generation tasks such as text style transfer (TST) remains a significant challenge. Recent studies (Dementieva et al., 2024; Pauli et al., 2025) revealed a substantial gap between automatic metrics and human judgments. Moreover, most prior work focuses exclusively on English, leaving multilingual TST evaluation largely unexplored. In this paper, we perform the first comprehensive multilingual study on evaluation of text detoxification system across nine languages: English, Spanish, German, Chinese, Arabic, Hindi, Ukrainian, Russian, Amharic. Drawing inspiration from the machine translation, we assess the effectiveness of modern neural-based evaluation models alongside prompting-based LLM-as-a-judge approaches. Our findings provide a practical recipe for designing more reliable multilingual TST evaluation pipeline in the text detoxification case. 

---
# Step-level Verifier-guided Hybrid Test-Time Scaling for Large Language Models 

**Authors**: Kaiyan Chang, Yonghao Shi, Chenglong Wang, Hang Zhou, Chi Hu, Xiaoqian Liu, Yingfeng Luo, Yuan Ge, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.15512)  

**Abstract**: Test-Time Scaling (TTS) is a promising approach to progressively elicit the model's intelligence during inference. Recently, training-based TTS methods, such as continued reinforcement learning (RL), have further surged in popularity, while training-free TTS methods are gradually fading from prominence. However, the additional computation overhead of training amplifies the burden on test-time scaling. In this paper, we focus on training-free TTS methods for reasoning. We first design Conditional Step-level Self-refinement, a fine-grained sequential scaling method guided by process verification. On top of its effectiveness, we further combine it with other classical parallel scaling methods at the step level, to introduce a novel inference paradigm called Hybrid Test-Time Scaling. Extensive experiments on five instruction-tuned LLMs across different scales (3B-14B) and families demonstrate that hybrid strategy incorporating various training-free TTS methods at a fine granularity has considerable potential for expanding the reasoning performance boundaries of LLMs. 

---
# ASPERA: A Simulated Environment to Evaluate Planning for Complex Action Execution 

**Authors**: Alexandru Coca, Mark Gaynor, Zhenxing Zhang, Jianpeng Cheng, Bo-Hsiang Tseng, Pete Boothroyd, Héctor Martinez Alonso, Diarmuid Ó Séaghdha, Anders Johannsen  

**Link**: [PDF](https://arxiv.org/pdf/2507.15501)  

**Abstract**: This work evaluates the potential of large language models (LLMs) to power digital assistants capable of complex action execution. These assistants rely on pre-trained programming knowledge to execute multi-step goals by composing objects and functions defined in assistant libraries into action execution programs. To achieve this, we develop ASPERA, a framework comprising an assistant library simulation and a human-assisted LLM data generation engine. Our engine allows developers to guide LLM generation of high-quality tasks consisting of complex user queries, simulation state and corresponding validation programs, tackling data availability and evaluation robustness challenges. Alongside the framework we release Asper-Bench, an evaluation dataset of 250 challenging tasks generated using ASPERA, which we use to show that program generation grounded in custom assistant libraries is a significant challenge to LLMs compared to dependency-free code generation. 

---
# AlgoSimBench: Identifying Algorithmically Similar Problems for Competitive Programming 

**Authors**: Jierui Li, Raymond Mooney  

**Link**: [PDF](https://arxiv.org/pdf/2507.15378)  

**Abstract**: Recent progress in LLMs, such as reasoning models, has demonstrated strong abilities to solve complex competitive programming problems, often rivaling top human competitors. However, it remains underexplored whether these abilities generalize to relevant domains that are less seen during training. To address this, we introduce AlgoSimBench, a new benchmark designed to assess LLMs' ability to identify algorithmically similar problems (ASPs)-problems that can be solved using similar algorithmic approaches. AlgoSimBench consists of 1317 problems, annotated with 231 distinct fine-grained algorithm tags, from which we curate 402 multiple-choice questions (MCQs), where each question presents one algorithmically similar problem alongside three textually similar but algorithmically dissimilar distractors. Our evaluation reveals that LLMs struggle to identify ASPs, with the best-performing model (o3-mini) achieving only 65.9% accuracy on the MCQ task. To address this challenge, we propose attempted solution matching (ASM), a novel method for improving problem similarity detection. On our MCQ task, ASM yields an absolute accuracy improvement of 6.7% to 11.7% across different models. We also evaluated code embedding models and retrieval methods on similar problem identification. While the adversarial selection of problems degrades the performance to be less than random, we found that simply summarizing the problem to remove narrative elements eliminates the effect, and combining ASM with a keyword-prioritized method, BM25, can yield up to 52.2% accuracy. Code and data are available at this http URL 

---
# STITCH: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models 

**Authors**: Cheng-Han Chiang, Xiaofei Wang, Linjie Li, Chung-Ching Lin, Kevin Lin, Shujie Liu, Zhendong Wang, Zhengyuan Yang, Hung-yi Lee, Lijuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15375)  

**Abstract**: Spoken Language Models (SLMs) are designed to take speech inputs and produce spoken responses. However, current SLMs lack the ability to perform an internal, unspoken thinking process before responding. In contrast, humans typically engage in complex mental reasoning internally, enabling them to communicate ideas clearly and concisely. Thus, integrating an unspoken thought process into SLMs is highly desirable. While naively generating a complete chain-of-thought (CoT) reasoning before starting to talk can enable thinking for SLMs, this induces additional latency for the speech response, as the CoT reasoning can be arbitrarily long. To solve this issue, we propose Stitch, a novel generation method that alternates between the generation of unspoken reasoning chunks and spoken response chunks. Since the audio duration of a chunk of spoken response is much longer than the time to generate the tokens in a chunk of spoken response, we use the remaining free time to generate the unspoken reasoning tokens. When a chunk of audio is played to the user, the model continues to generate the next unspoken reasoning chunk, achieving simultaneous thinking and talking. Remarkably, Stitch matches the latency of baselines that cannot generate unspoken CoT by design while outperforming those baselines by 15% on math reasoning datasets; Stitch also performs equally well on non-reasoning datasets as those baseline models. Some animations and demonstrations are on the project page: this https URL. 

---
# Metaphor and Large Language Models: When Surface Features Matter More than Deep Understanding 

**Authors**: Elisa Sanchez-Bayona, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2507.15357)  

**Abstract**: This paper presents a comprehensive evaluation of the capabilities of Large Language Models (LLMs) in metaphor interpretation across multiple datasets, tasks, and prompt configurations. Although metaphor processing has gained significant attention in Natural Language Processing (NLP), previous research has been limited to single-dataset evaluations and specific task settings, often using artificially constructed data through lexical replacement. We address these limitations by conducting extensive experiments using diverse publicly available datasets with inference and metaphor annotations, focusing on Natural Language Inference (NLI) and Question Answering (QA) tasks. The results indicate that LLMs' performance is more influenced by features like lexical overlap and sentence length than by metaphorical content, demonstrating that any alleged emergent abilities of LLMs to understand metaphorical language are the result of a combination of surface-level features, in-context learning, and linguistic knowledge. This work provides critical insights into the current capabilities and limitations of LLMs in processing figurative language, highlighting the need for more realistic evaluation frameworks in metaphor interpretation tasks. Data and code are publicly available. 

---
# Probing Information Distribution in Transformer Architectures through Entropy Analysis 

**Authors**: Amedeo Buonanno, Alessandro Rivetti, Francesco A. N. Palmieri, Giovanni Di Gennaro, Gianmarco Romano  

**Link**: [PDF](https://arxiv.org/pdf/2507.15347)  

**Abstract**: This work explores entropy analysis as a tool for probing information distribution within Transformer-based architectures. By quantifying token-level uncertainty and examining entropy patterns across different stages of processing, we aim to investigate how information is managed and transformed within these models. As a case study, we apply the methodology to a GPT-based large language model, illustrating its potential to reveal insights into model behavior and internal representations. This approach may offer insights into model behavior and contribute to the development of interpretability and evaluation frameworks for transformer-based models 

---
# LionGuard 2: Building Lightweight, Data-Efficient & Localised Multilingual Content Moderators 

**Authors**: Leanne Tan, Gabriel Chua, Ziyu Ge, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.15339)  

**Abstract**: Modern moderation systems increasingly support multiple languages, but often fail to address localisation and low-resource variants - creating safety gaps in real-world deployments. Small models offer a potential alternative to large LLMs, yet still demand considerable data and compute. We present LionGuard 2, a lightweight, multilingual moderation classifier tailored to the Singapore context, supporting English, Chinese, Malay, and partial Tamil. Built on pre-trained OpenAI embeddings and a multi-head ordinal classifier, LionGuard 2 outperforms several commercial and open-source systems across 17 benchmarks, including both Singapore-specific and public English datasets. The system is actively deployed within the Singapore Government, demonstrating practical efficacy at scale. Our findings show that high-quality local data and robust multilingual embeddings can achieve strong moderation performance, without fine-tuning large models. We release our model weights and part of our training data to support future work on LLM safety. 

---
# Reasoning Models are Test Exploiters: Rethinking Multiple-Choice 

**Authors**: Narun Raman, Taylor Lundy, Kevin Leyton-Brown  

**Link**: [PDF](https://arxiv.org/pdf/2507.15337)  

**Abstract**: When evaluating Large Language Models (LLMs) in question-answering domains, it is common to ask the model to choose among a fixed set of choices (so-called multiple-choice question-answering, or MCQA). Although downstream tasks of interest typically do not provide systems with explicit options among which to choose, this approach is nevertheless widely used because it makes it makes automatic grading straightforward and has tended to produce challenging benchmarks that correlate sufficiently well with downstream performance. This paper investigates the extent to which this trend continues to hold for state-of-the-art reasoning models, describing a systematic evaluation of $15$ different question-answering benchmarks (e.g., MMLU, HLE) and $25$ different LLMs (including small models such as Qwen 7B and relatively large models such as Llama 70B). For each model-benchmark pair, we considered $5$ ways of presenting the model with questions, including variations on whether multiple choices were offered to the model at all; whether "none of the above" sometimes replaced the right answer; and whether the model was permitted to perform chain-of-thought reasoning before and/or after the choices were presented. MCQA remained a good proxy for the downstream performance of models as long as they were allowed to perform chain-of-thought reasoning only before being presented with the options among which they had to select. On the other hand, large models that were able to perform reasoning after being given a set of options tended to significantly outperform their free-text performance due to exploiting the information in the options. We conclude that MCQA is no longer a good proxy for assessing downstream performance of state-of-the-art models, and offer practical guidelines for designing more robust, bias-resistant benchmarks that better reflect LLMs' genuine reasoning capabilities. 

---
# On the Inevitability of Left-Leaning Political Bias in Aligned Language Models 

**Authors**: Thilo Hagendorff  

**Link**: [PDF](https://arxiv.org/pdf/2507.15328)  

**Abstract**: The guiding principle of AI alignment is to train large language models (LLMs) to be harmless, helpful, and honest (HHH). At the same time, there are mounting concerns that LLMs exhibit a left-wing political bias. Yet, the commitment to AI alignment cannot be harmonized with the latter critique. In this article, I argue that intelligent systems that are trained to be harmless and honest must necessarily exhibit left-wing political bias. Normative assumptions underlying alignment objectives inherently concur with progressive moral frameworks and left-wing principles, emphasizing harm avoidance, inclusivity, fairness, and empirical truthfulness. Conversely, right-wing ideologies often conflict with alignment guidelines. Yet, research on political bias in LLMs is consistently framing its insights about left-leaning tendencies as a risk, as problematic, or concerning. This way, researchers are actively arguing against AI alignment, tacitly fostering the violation of HHH principles. 

---
# Beyond Easy Wins: A Text Hardness-Aware Benchmark for LLM-generated Text Detection 

**Authors**: Navid Ayoobi, Sadat Shahriar, Arjun Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2507.15286)  

**Abstract**: We present a novel evaluation paradigm for AI text detectors that prioritizes real-world and equitable assessment. Current approaches predominantly report conventional metrics like AUROC, overlooking that even modest false positive rates constitute a critical impediment to practical deployment of detection systems. Furthermore, real-world deployment necessitates predetermined threshold configuration, making detector stability (i.e. the maintenance of consistent performance across diverse domains and adversarial scenarios), a critical factor. These aspects have been largely ignored in previous research and benchmarks. Our benchmark, SHIELD, addresses these limitations by integrating both reliability and stability factors into a unified evaluation metric designed for practical assessment. Furthermore, we develop a post-hoc, model-agnostic humanification framework that modifies AI text to more closely resemble human authorship, incorporating a controllable hardness parameter. This hardness-aware approach effectively challenges current SOTA zero-shot detection methods in maintaining both reliability and stability. (Data and code: this https URL) 

---
# A Novel Self-Evolution Framework for Large Language Models 

**Authors**: Haoran Sun, Zekun Zhang, Shaoning Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2507.15281)  

**Abstract**: The capabilities of Large Language Models (LLMs) are limited to some extent by pre-training, so some researchers optimize LLMs through post-training. Existing post-training strategies, such as memory-based retrieval or preference optimization, improve user alignment yet fail to enhance the model's domain cognition. To bridge this gap, we propose a novel Dual-Phase Self-Evolution (DPSE) framework that jointly optimizes user preference adaptation and domain-specific competence. DPSE introduces a Censor module to extract multi-dimensional interaction signals and estimate satisfaction scores, which guide structured data expansion via topic-aware and preference-driven strategies. These expanded datasets support a two-stage fine-tuning pipeline: supervised domain grounding followed by frequency-aware preference optimization. Experiments across general NLP benchmarks and long-term dialogue tasks demonstrate that DPSE consistently outperforms Supervised Fine-Tuning, Preference Optimization, and Memory-Augmented baselines. Ablation studies validate the contribution of each module. In this way, our framework provides an autonomous path toward continual self-evolution of LLMs. 

---
# ChiMed 2.0: Advancing Chinese Medical Dataset in Facilitating Large Language Modeling 

**Authors**: Yuanhe Tian, Junjie Liu, Zhizhou Kou, Yuxiang Li, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.15275)  

**Abstract**: Building high-quality data resources is crucial for advancing artificial intelligence research and applications in specific domains, particularly in the Chinese medical domain. Existing Chinese medical datasets are limited in size and narrow in domain coverage, falling short of the diverse corpora required for effective pre-training. Moreover, most datasets are designed solely for LLM fine-tuning and do not support pre-training and reinforcement learning from human feedback (RLHF). In this paper, we propose a Chinese medical dataset named ChiMed 2.0, which extends our previous work ChiMed, and covers data collected from Chinese medical online platforms and generated by LLMs. ChiMed 2.0 contains 204.4M Chinese characters covering both traditional Chinese medicine classics and modern general medical data, where there are 164.8K documents for pre-training, 351.6K question-answering pairs for supervised fine-tuning (SFT), and 41.7K preference data tuples for RLHF. To validate the effectiveness of our approach for training a Chinese medical LLM, we conduct further pre-training, SFT, and RLHF experiments on representative general domain LLMs and evaluate their performance on medical benchmark datasets. The results show performance gains across different model scales, validating the dataset's effectiveness and applicability. 

---
# SOI Matters: Analyzing Multi-Setting Training Dynamics in Pretrained Language Models via Subsets of Interest 

**Authors**: Shayan Vassef, Amirhossein Dabiriaghdam, Mohammadreza Bakhtiari, Yadollah Yaghoobzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2507.15236)  

**Abstract**: This work investigates the impact of multi-task, multi-lingual, and multi-source learning approaches on the robustness and performance of pretrained language models. To enhance this analysis, we introduce Subsets of Interest (SOI), a novel categorization framework that identifies six distinct learning behavior patterns during training, including forgettable examples, unlearned examples, and always correct examples. Through SOI transition heatmaps and dataset cartography visualization, we analyze how examples shift between these categories when transitioning from single-setting to multi-setting configurations. We perform comprehensive experiments across three parallel comparisons: multi-task vs. single-task learning using English tasks (entailment, paraphrase, sentiment), multi-source vs. single-source learning using sentiment analysis datasets, and multi-lingual vs. single-lingual learning using intent classification in French, English, and Persian. Our results demonstrate that multi-source learning consistently improves out-of-distribution performance by up to 7%, while multi-task learning shows mixed results with notable gains in similar task combinations. We further introduce a two-stage fine-tuning approach where the second stage leverages SOI-based subset selection to achieve additional performance improvements. These findings provide new insights into training dynamics and offer practical approaches for optimizing multi-setting language model performance. 

---
# Collaborative Distillation Strategies for Parameter-Efficient Language Model Deployment 

**Authors**: Xiandong Meng, Yan Wu, Yexin Tian, Xin Hu, Tianze Kang, Junliang Du  

**Link**: [PDF](https://arxiv.org/pdf/2507.15198)  

**Abstract**: This paper addresses the challenges of high computational cost and slow inference in deploying large language models. It proposes a distillation strategy guided by multiple teacher models. The method constructs several teacher models and integrates their output probability distributions and intermediate semantic features. This guides the student model to learn from multiple sources of knowledge. As a result, the student model gains stronger language understanding and generation ability while maintaining a small parameter size. To achieve this, the paper introduces a weighted output fusion mechanism, a feature alignment loss function, and an entropy-driven dynamic teacher weighting strategy. These components improve the quality and stability of knowledge transfer during distillation. Under multi-teacher guidance, the student model captures semantic information more effectively and demonstrates strong performance across multiple evaluation metrics. In particular, the method shows high consistency in expression, generalization ability, and task adaptability in tasks such as language modeling, text generation, and multi-task learning. The experiments compare the proposed method with several widely adopted distillation approaches. The results further confirm its overall advantages in perplexity, distillation loss, and generation quality. This study provides a feasible technical path for the efficient compression of large-scale language models. It also demonstrates the effectiveness of multi-teacher collaborative mechanisms in complex language modeling tasks. 

---
# What Level of Automation is "Good Enough"? A Benchmark of Large Language Models for Meta-Analysis Data Extraction 

**Authors**: Lingbo Li, Anuradha Mathrani, Teo Susnjak  

**Link**: [PDF](https://arxiv.org/pdf/2507.15152)  

**Abstract**: Automating data extraction from full-text randomised controlled trials (RCTs) for meta-analysis remains a significant challenge. This study evaluates the practical performance of three LLMs (Gemini-2.0-flash, Grok-3, GPT-4o-mini) across tasks involving statistical results, risk-of-bias assessments, and study-level characteristics in three medical domains: hypertension, diabetes, and orthopaedics. We tested four distinct prompting strategies (basic prompting, self-reflective prompting, model ensemble, and customised prompts) to determine how to improve extraction quality. All models demonstrate high precision but consistently suffer from poor recall by omitting key information. We found that customised prompts were the most effective, boosting recall by up to 15\%. Based on this analysis, we propose a three-tiered set of guidelines for using LLMs in data extraction, matching data types to appropriate levels of automation based on task complexity and risk. Our study offers practical advice for automating data extraction in real-world meta-analyses, balancing LLM efficiency with expert oversight through targeted, task-specific automation. 

---
# A Case Against Implicit Standards: Homophone Normalization in Machine Translation for Languages that use the Ge'ez Script 

**Authors**: Hellina Hailu Nigatu, Atnafu Lambebo Tonja, Henok Biadglign Ademtew, Hizkel Mitiku Alemayehu, Negasi Haile Abadi, Tadesse Destaw Belay, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2507.15142)  

**Abstract**: Homophone normalization, where characters that have the same sound in a writing script are mapped to one character, is a pre-processing step applied in Amharic Natural Language Processing (NLP) literature. While this may improve performance reported by automatic metrics, it also results in models that are not able to understand different forms of writing in a single language. Further, there might be impacts in transfer learning, where models trained on normalized data do not generalize well to other languages. In this paper, we experiment with monolingual training and cross-lingual transfer to understand the impacts of normalization on languages that use the Ge'ez script. We then propose a post-inference intervention in which normalization is applied to model predictions instead of training data. With our simple scheme of post-inference normalization, we show that we can achieve an increase in BLEU score of up to 1.03 while preserving language features in training. Our work contributes to the broader discussion on technology-facilitated language change and calls for more language-aware interventions. 

---
# From Disagreement to Understanding: The Case for Ambiguity Detection in NLI 

**Authors**: Chathuri Jayaweera, Bonnie Dorr  

**Link**: [PDF](https://arxiv.org/pdf/2507.15114)  

**Abstract**: This position paper argues that annotation disagreement in Natural Language Inference (NLI) is not mere noise but often reflects meaningful interpretive variation, especially when triggered by ambiguity in the premise or hypothesis. While underspecified guidelines and annotator behavior can contribute to variation, content-based ambiguity offers a process-independent signal of divergent human perspectives. We call for a shift toward ambiguity-aware NLI by systematically identifying ambiguous input pairs and classifying ambiguity types. To support this, we present a unified framework that integrates existing taxonomies and illustrate key ambiguity subtypes through concrete examples. These examples reveal how ambiguity shapes annotator decisions and motivate the need for targeted detection methods that better align models with human interpretation. A key limitation is the lack of datasets annotated for ambiguity and subtypes. We propose addressing this gap through new annotated resources and unsupervised approaches to ambiguity detection -- paving the way for more robust, explainable, and human-aligned NLI systems. 

---
# Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference? 

**Authors**: Chathuri Jayaweera, Brianna Yanqui, Bonnie Dorr  

**Link**: [PDF](https://arxiv.org/pdf/2507.15100)  

**Abstract**: Natural Language Inference (NLI) is the task of determining the semantic entailment of a premise for a given hypothesis. The task aims to develop systems that emulate natural human inferential processes where commonsense knowledge plays a major role. However, existing commonsense resources lack sufficient coverage for a variety of premise-hypothesis pairs. This study explores the potential of Large Language Models as commonsense knowledge generators for NLI along two key dimensions: their reliability in generating such knowledge and the impact of that knowledge on prediction accuracy. We adapt and modify existing metrics to assess LLM factuality and consistency in generating in this context. While explicitly incorporating commonsense knowledge does not consistently improve overall results, it effectively helps distinguish entailing instances and moderately improves distinguishing contradictory and neutral inferences. 

---
# A Penalty Goes a Long Way: Measuring Lexical Diversity in Synthetic Texts Under Prompt-Influenced Length Variations 

**Authors**: Vijeta Deshpande, Ishita Dasgupta, Uttaran Bhattacharya, Somdeb Sarkhel, Saayan Mitra, Anna Rumshisky  

**Link**: [PDF](https://arxiv.org/pdf/2507.15092)  

**Abstract**: Synthetic text generated by Large Language Models (LLMs) is increasingly used for further training and improvement of LLMs. Diversity is crucial for the effectiveness of synthetic data, and researchers rely on prompt engineering to improve diversity. However, the impact of prompt variations on response text length, and, more importantly, the consequential effect on lexical diversity measurements, remain underexplored. In this work, we propose Penalty-Adjusted Type-Token Ratio (PATTR), a diversity metric robust to length variations. We generate a large synthetic corpus of over 20M words using seven models from the LLaMA, OLMo, and Phi families, focusing on a creative writing task of video script generation, where diversity is crucial. We evaluate per-response lexical diversity using PATTR and compare it against existing metrics of Moving-Average TTR (MATTR) and Compression Ratio (CR). Our analysis highlights how text length variations introduce biases favoring shorter responses. Unlike existing metrics, PATTR explicitly considers the task-specific target response length ($L_T$) to effectively mitigate length biases. We further demonstrate the utility of PATTR in filtering the top-10/100/1,000 most lexically diverse responses, showing that it consistently outperforms MATTR and CR by yielding on par or better diversity with high adherence to $L_T$. 

---
# Evaluation of Coding Schemes for Transformer-based Gene Sequence Modeling 

**Authors**: Chenlei Gong, Yuanhe Tian, Lei Mao, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.15087)  

**Abstract**: Currently, many studies view DNA sequences as a special type of language and utilize Transformers to model them. These studies use fixed-length k-mer segmentation and BPE subword tokenization but lack a systematic evaluation to determine which is superior. We compare k-mer segmentation with k=1,3,4,5,6, a 4,096-token BPE vocabulary, and three positional encoding methods-sinusoidal, AliBi, and RoPE. Each configuration is trained from scratch in 3, 6, 12, and 24-layer Transformer encoders and evaluated on GUE benchmark dataset. In general, BPE delivers higher and more stable performance across tasks by compressing frequent motifs into variable-length tokens, reducing sequence length, and improving model generalization. RoPE excels at capturing periodic motifs and extrapolating to long sequences, while AliBi also performs well on tasks driven by local dependencies. In terms of depth, we observe significant gains when increasing layers from 3 to 12, with only marginal improvements or slight overfitting at 24 layers. This study provides practical guidance for designing tokenization and positional encoding in DNA Transformer models. 

---
# WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization 

**Authors**: Zhengwei Tao, Jialong Wu, Wenbiao Yin, Junkai Zhang, Baixuan Li, Haiyang Shen, Kuan Li, Liwen Zhang, Xinyu Wang, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.15061)  

**Abstract**: The advent of Large Language Model (LLM)-powered agents has revolutionized artificial intelligence by enabling solutions to complex, open-ended tasks through web-based information-seeking (IS) capabilities. The scarcity of high-quality training data has limited the development of IS agents. Existing approaches typically adopt an information-driven paradigm that first collects web data and then generates questions based on the retrieval. However, this may lead to inconsistency between information structure and reasoning structure, question and answer. To mitigate, we propose a formalization-driven IS data synthesis framework WebShaper to construct a dataset. WebShaper systematically formalizes IS tasks through set theory. Central to the formalization is the concept of Knowledge Projections (KP), which enables precise control over reasoning structure by KP operation compositions. During synthesis, we begin by creating seed tasks, then use a multi-step expansion process. At each step, an agentic Expander expands the current formal question more complex with retrieval and validation tools based on our formalization. We train our model on the synthesized dataset. Experiment results demonstrate that WebShaper achieves state-of-the-art performance among open-sourced IS agents on GAIA and WebWalkerQA benchmarks. 

---
# RefCritic: Training Long Chain-of-Thought Critic Models with Refinement Feedback 

**Authors**: Qiaoyu Tang, Hao Xiang, Le Yu, Bowen Yu, Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.15024)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), developing effective critic modules for precise guidance has become crucial yet challenging. In this paper, we initially demonstrate that supervised fine-tuning for building critic modules (which is widely adopted in current solutions) fails to genuinely enhance models' critique abilities, producing superficial critiques with insufficient reflections and verifications. To unlock the unprecedented critique capabilities, we propose RefCritic, a long-chain-of-thought critic module based on reinforcement learning with dual rule-based rewards: (1) instance-level correctness of solution judgments and (2) refinement accuracies of the policy model based on critiques, aiming to generate high-quality evaluations with actionable feedback that effectively guides model refinement. We evaluate RefCritic on Qwen2.5-14B-Instruct and DeepSeek-R1-Distill-Qwen-14B across five benchmarks. On critique and refinement settings, RefCritic demonstrates consistent advantages across all benchmarks, e.g., 6.8\% and 7.2\% gains on AIME25 for the respective base models. Notably, under majority voting, policy models filtered by RefCritic show superior scaling with increased voting numbers. Moreover, despite training on solution-level supervision, RefCritic outperforms step-level supervised approaches on ProcessBench, a benchmark to identify erroneous steps in mathematical reasoning. 

---
# MUR: Momentum Uncertainty guided Reasoning for Large Language Models 

**Authors**: Hang Yan, Fangzhi Xu, Rongman Xu, Yifei Li, Jian Zhang, Haoran Luo, Xiaobao Wu, Luu Anh Tuan, Haiteng Zhao, Qika Lin, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14958)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance on reasoning-intensive tasks, yet optimizing their reasoning efficiency remains an open challenge. While Test-Time Scaling (TTS) improves reasoning quality, it often leads to overthinking, wasting tokens on redundant computations. This work investigates how to efficiently and adaptively guide LLM test-time scaling without additional training. Inspired by the concept of momentum in physics, we propose Momentum Uncertainty-guided Reasoning (MUR), which dynamically allocates thinking budgets to critical reasoning steps by tracking and aggregating stepwise uncertainty over time. To support flexible inference-time control, we introduce gamma-control, a simple mechanism that tunes the reasoning budget via a single hyperparameter. We provide in-depth theoretical proof to support the superiority of MUR in terms of stability and biases. MUR is comprehensively evaluated against various TTS methods across four challenging benchmarks (MATH-500, AIME24, AIME25, and GPQA-diamond) using different sizes of recent Qwen3 models (1.7B, 4B, and 8B). Results demonstrate that MUR reduces computation by over 50% on average while improving accuracy by 0.62-3.37%. 

---
# SYNTHIA: Synthetic Yet Naturally Tailored Human-Inspired PersonAs 

**Authors**: Vahid Rahimzadeh, Erfan Moosavi Monazzah, Mohammad Taher Pilehvar, Yadollah Yaghoobzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2507.14922)  

**Abstract**: Persona-driven LLMs have emerged as powerful tools in computational social science, yet existing approaches fall at opposite extremes, either relying on costly human-curated data or producing synthetic personas that lack consistency and realism. We introduce SYNTHIA, a dataset of 30,000 backstories derived from 10,000 real social media users from BlueSky open platform across three time windows, bridging this spectrum by grounding synthetic generation in authentic user activity. Our evaluation demonstrates that SYNTHIA achieves competitive performance with state-of-the-art methods in demographic diversity and social survey alignment while significantly outperforming them in narrative consistency. Uniquely, SYNTHIA incorporates temporal dimensionality and provides rich social interaction metadata from the underlying network, enabling new research directions in computational social science and persona-driven language modeling. 

---
# PromptSuite: A Task-Agnostic Framework for Multi-Prompt Generation 

**Authors**: Eliya Habba, Noam Dahan, Gili Lior, Gabriel Stanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2507.14913)  

**Abstract**: Evaluating LLMs with a single prompt has proven unreliable, with small changes leading to significant performance differences. However, generating the prompt variations needed for a more robust multi-prompt evaluation is challenging, limiting its adoption in practice. To address this, we introduce PromptSuite, a framework that enables the automatic generation of various prompts. PromptSuite is flexible - working out of the box on a wide range of tasks and benchmarks. It follows a modular prompt design, allowing controlled perturbations to each component, and is extensible, supporting the addition of new components and perturbation types. Through a series of case studies, we show that PromptSuite provides meaningful variations to support strong evaluation practices. It is available through both a Python API: this https URL, and a user-friendly web interface: this https URL 

---
# From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment 

**Authors**: Chongxuan Huang, Yongshi Ye, Biao Fu, Qifeng Su, Xiaodong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.14900)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable multilingual capabilities, however, how to evaluate cross-lingual alignment remains underexplored. Existing alignment benchmarks primarily focus on sentence embeddings, but prior research has shown that neural models tend to induce a non-smooth representation space, which impact of semantic alignment evaluation on low-resource languages. Inspired by neuroscientific findings that similar information activates overlapping neuronal regions, we propose a novel Neuron State-Based Cross-Lingual Alignment (NeuronXA) to assess the cross-lingual a lignment capabilities of LLMs, which offers a more semantically grounded approach to assess cross-lingual alignment. We evaluate NeuronXA on several prominent multilingual LLMs (LLaMA, Qwen, Mistral, GLM, and OLMo) across two transfer tasks and three multilingual benchmarks. The results demonstrate that with only 100 parallel sentence pairs, NeuronXA achieves a Pearson correlation of 0.9556 with downstream tasks performance and 0.8514 with transferability. These findings demonstrate NeuronXA's effectiveness in assessing both cross-lingual alignment and transferability, even with a small dataset. This highlights its potential to advance cross-lingual alignment research and to improve the semantic understanding of multilingual LLMs. 

---
# Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs 

**Authors**: Boyi Deng, Yu Wan, Baosong Yang, Fei Huang, Wenjie Wang, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.14894)  

**Abstract**: Large Language Models (LLMs) have impressive multilingual capabilities, but they suffer from unexpected code-switching, also known as language mixing, which involves switching to unexpected languages in the model response. This problem leads to poor readability and degrades the usability of model responses. However, existing work on this issue lacks a mechanistic analysis and shows limited effectiveness. In this paper, we first provide an in-depth analysis of unexpected code-switching using sparse autoencoders and find that when LLMs switch to a language, the features of that language exhibit excessive pre-activation values. Based on our findings, we propose $\textbf{S}$parse $\textbf{A}$utoencoder-guided $\textbf{S}$upervised $\textbf{F}$ine$\textbf{t}$uning (SASFT), which teaches LLMs to maintain appropriate pre-activation values of specific language features during training. Experiments on five models across three languages demonstrate that SASFT consistently reduces unexpected code-switching by more than 50\% compared to standard supervised fine-tuning, with complete elimination in four cases. Moreover, SASFT maintains or even improves the models' performance on six multilingual benchmarks, showing its effectiveness in addressing code-switching while preserving multilingual capabilities. 

---
# MEKiT: Multi-source Heterogeneous Knowledge Injection Method via Instruction Tuning for Emotion-Cause Pair Extraction 

**Authors**: Shiyi Mu, Yongkang Liu, Shi Feng, Xiaocui Yang, Daling Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14887)  

**Abstract**: Although large language models (LLMs) excel in text comprehension and generation, their performance on the Emotion-Cause Pair Extraction (ECPE) task, which requires reasoning ability, is often underperform smaller language model. The main reason is the lack of auxiliary knowledge, which limits LLMs' ability to effectively perceive emotions and reason causes. To address this issue, we propose a novel \textbf{M}ulti-source h\textbf{E}terogeneous \textbf{K}nowledge \textbf{i}njection me\textbf{T}hod, MEKiT, which integrates heterogeneous internal emotional knowledge and external causal knowledge. Specifically, for these two distinct aspects and structures of knowledge, we apply the approaches of incorporating instruction templates and mixing data for instruction-tuning, which respectively facilitate LLMs in more comprehensively identifying emotion and accurately reasoning causes. Experimental results demonstrate that MEKiT provides a more effective and adaptable solution for the ECPE task, exhibiting an absolute performance advantage over compared baselines and dramatically improving the performance of LLMs on the ECPE task. 

---
# Tiny language models 

**Authors**: Ronit D. Gross, Yarden Tzach, Tal Halevi, Ella Koresh, Ido Kanter  

**Link**: [PDF](https://arxiv.org/pdf/2507.14871)  

**Abstract**: A prominent achievement of natural language processing (NLP) is its ability to understand and generate meaningful human language. This capability relies on complex feedforward transformer block architectures pre-trained on large language models (LLMs). However, LLM pre-training is currently feasible only for a few dominant companies due to the immense computational resources required, limiting broader research participation. This creates a critical need for more accessible alternatives. In this study, we explore whether tiny language models (TLMs) exhibit the same key qualitative features of LLMs. We demonstrate that TLMs exhibit a clear performance gap between pre-trained and non-pre-trained models across classification tasks, indicating the effectiveness of pre-training, even at a tiny scale. The performance gap increases with the size of the pre-training dataset and with greater overlap between tokens in the pre-training and classification datasets. Furthermore, the classification accuracy achieved by a pre-trained deep TLM architecture can be replicated through a soft committee of multiple, independently pre-trained shallow architectures, enabling low-latency TLMs without affecting classification accuracy. Our results are based on pre-training BERT-6 and variants of BERT-1 on subsets of the Wikipedia dataset and evaluating their performance on FewRel, AGNews, and DBPedia classification tasks. Future research on TLM is expected to further illuminate the mechanisms underlying NLP, especially given that its biologically inspired models suggest that TLMs may be sufficient for children or adolescents to develop language. 

---
# Beyond Isolated Capabilities: Bridging Long CoT Reasoning and Long-Context Understanding 

**Authors**: Yifei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14849)  

**Abstract**: Reasoning distillation has emerged as an effective approach to enhance the reasoning capabilities of smaller language models. However, the impact of large-scale reasoning distillation on other critical abilities, particularly in-context retrieval and reasoning, remains unexplored. This gap in understanding is particularly significant given the increasing importance of Retrieval-Augmented Generation (RAG) systems, where efficient acquisition and utilization of contextual information are paramount for generating reliable responses. Motivated by the need to understand how the extended long-CoT process influences long-context comprehension, we conduct a comprehensive investigation using a series of open-source models distilled from Deepseek-R1, renowned for its exceptional reasoning capabilities. Our study focuses on evaluating these models' performance in extracting and integrating relevant information from extended contexts through multi-document question and answering tasks. Through rigorous experimentation, we demonstrate that distilled reasoning patterns significantly improve long-context understanding. Our analysis reveals that distillation fosters greater long-context awareness by promoting more detailed and explicit reasoning processes during context analysis and information parsing. This advancement effectively mitigates the persistent "lost in the middle" issue that has hindered long-context models. 

---
# Doc2Chart: Intent-Driven Zero-Shot Chart Generation from Documents 

**Authors**: Akriti Jain, Pritika Ramu, Aparna Garimella, Apoorv Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2507.14819)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in transforming text descriptions or tables to data visualizations via instruction-tuning methods. However, it is not straightforward to apply these methods directly for a more real-world use case of visualizing data from long documents based on user-given intents, as opposed to the user pre-selecting the relevant content manually. We introduce the task of intent-based chart generation from documents: given a user-specified intent and document(s), the goal is to generate a chart adhering to the intent and grounded on the document(s) in a zero-shot setting. We propose an unsupervised, two-staged framework in which an LLM first extracts relevant information from the document(s) by decomposing the intent and iteratively validates and refines this data. Next, a heuristic-guided module selects an appropriate chart type before final code generation. To assess the data accuracy of the generated charts, we propose an attribution-based metric that uses a structured textual representation of charts, instead of relying on visual decoding metrics that often fail to capture the chart data effectively. To validate our approach, we curate a dataset comprising of 1,242 $<$intent, document, charts$>$ tuples from two domains, finance and scientific, in contrast to the existing datasets that are largely limited to parallel text descriptions/ tables and their corresponding charts. We compare our approach with baselines using single-shot chart generation using LLMs and query-based retrieval methods; our method outperforms by upto $9$ points and $17$ points in terms of chart data accuracy and chart type respectively over the best baselines. 

---
# FastLongSpeech: Enhancing Large Speech-Language Models for Efficient Long-Speech Processing 

**Authors**: Shoutao Guo, Shaolei Zhang, Qingkai Fang, Zhengrui Ma, Min Zhang, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.14815)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has spurred significant progress in Large Speech-Language Models (LSLMs), enhancing their capabilities in both speech understanding and generation. While existing LSLMs often concentrate on augmenting speech generation or tackling a diverse array of short-speech tasks, the efficient processing of long-form speech remains a critical yet underexplored challenge. This gap is primarily attributed to the scarcity of long-speech training datasets and the high computational costs associated with long sequences. To address these limitations, we introduce FastLongSpeech, a novel framework designed to extend LSLM capabilities for efficient long-speech processing without necessitating dedicated long-speech training data. FastLongSpeech incorporates an iterative fusion strategy that can compress excessively long-speech sequences into manageable lengths. To adapt LSLMs for long-speech inputs, it introduces a dynamic compression training approach, which exposes the model to short-speech sequences at varying compression ratios, thereby transferring the capabilities of LSLMs to long-speech tasks. To assess the long-speech capabilities of LSLMs, we develop a long-speech understanding benchmark called LongSpeech-Eval. Experiments show that our method exhibits strong performance in both long-speech and short-speech tasks, while greatly improving inference efficiency. 

---
# GRACE: Generative Recommendation via Journey-Aware Sparse Attention on Chain-of-Thought Tokenization 

**Authors**: Luyi Ma, Wanjia Zhang, Kai Zhao, Abhishek Kulkarni, Lalitesh Morishetti, Anjana Ganesh, Ashish Ranjan, Aashika Padmanabhan, Jianpeng Xu, Jason Cho, Praveen Kanumala, Kaushiki Nag, Sumit Dutta, Kamiya Motwani, Malay Patel, Evren Korpeoglu, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2507.14758)  

**Abstract**: Generative models have recently demonstrated strong potential in multi-behavior recommendation systems, leveraging the expressive power of transformers and tokenization to generate personalized item sequences. However, their adoption is hindered by (1) the lack of explicit information for token reasoning, (2) high computational costs due to quadratic attention complexity and dense sequence representations after tokenization, and (3) limited multi-scale modeling over user history. In this work, we propose GRACE (Generative Recommendation via journey-aware sparse Attention on Chain-of-thought tokEnization), a novel generative framework for multi-behavior sequential recommendation. GRACE introduces a hybrid Chain-of-Thought (CoT) tokenization method that encodes user-item interactions with explicit attributes from product knowledge graphs (e.g., category, brand, price) over semantic tokenization, enabling interpretable and behavior-aligned generation. To address the inefficiency of standard attention, we design a Journey-Aware Sparse Attention (JSA) mechanism, which selectively attends to compressed, intra-, inter-, and current-context segments in the tokenized sequence. Experiments on two real-world datasets show that GRACE significantly outperforms state-of-the-art baselines, achieving up to +106.9% HR@10 and +106.7% NDCG@10 improvement over the state-of-the-art baseline on the Home domain, and +22.1% HR@10 on the Electronics domain. GRACE also reduces attention computation by up to 48% with long sequences. 

---
# On the robustness of modeling grounded word learning through a child's egocentric input 

**Authors**: Wai Keen Vong, Brenden M. Lake  

**Link**: [PDF](https://arxiv.org/pdf/2507.14749)  

**Abstract**: What insights can machine learning bring to understanding human language acquisition? Large language and multimodal models have achieved remarkable capabilities, but their reliance on massive training datasets creates a fundamental mismatch with children, who succeed in acquiring language from comparatively limited input. To help bridge this gap, researchers have increasingly trained neural networks using data similar in quantity and quality to children's input. Taking this approach to the limit, Vong et al. (2024) showed that a multimodal neural network trained on 61 hours of visual and linguistic input extracted from just one child's developmental experience could acquire word-referent mappings. However, whether this approach's success reflects the idiosyncrasies of a single child's experience, or whether it would show consistent and robust learning patterns across multiple children's experiences was not explored. In this article, we applied automated speech transcription methods to the entirety of the SAYCam dataset, consisting of over 500 hours of video data spread across all three children. Using these automated transcriptions, we generated multi-modal vision-and-language datasets for both training and evaluation, and explored a range of neural network configurations to examine the robustness of simulated word learning. Our findings demonstrate that networks trained on automatically transcribed data from each child can acquire and generalize word-referent mappings across multiple network architectures. These results validate the robustness of multimodal neural networks for grounded word learning, while highlighting the individual differences that emerge in how models learn when trained on each child's developmental experiences. 

---
# Disparities in Peer Review Tone and the Role of Reviewer Anonymity 

**Authors**: Maria Sahakyan, Bedoor AlShebli  

**Link**: [PDF](https://arxiv.org/pdf/2507.14741)  

**Abstract**: The peer review process is often regarded as the gatekeeper of scientific integrity, yet increasing evidence suggests that it is not immune to bias. Although structural inequities in peer review have been widely debated, much less attention has been paid to the subtle ways in which language itself may reinforce disparities. This study undertakes one of the most comprehensive linguistic analyses of peer review to date, examining more than 80,000 reviews in two major journals. Using natural language processing and large-scale statistical modeling, it uncovers how review tone, sentiment, and supportive language vary across author demographics, including gender, race, and institutional affiliation. Using a data set that includes both anonymous and signed reviews, this research also reveals how the disclosure of reviewer identity shapes the language of evaluation. The findings not only expose hidden biases in peer feedback, but also challenge conventional assumptions about anonymity's role in fairness. As academic publishing grapples with reform, these insights raise critical questions about how review policies shape career trajectories and scientific progress. 

---
# Rethinking Suicidal Ideation Detection: A Trustworthy Annotation Framework and Cross-Lingual Model Evaluation 

**Authors**: Amina Dzafic, Merve Kavut, Ulya Bayram  

**Link**: [PDF](https://arxiv.org/pdf/2507.14693)  

**Abstract**: Suicidal ideation detection is critical for real-time suicide prevention, yet its progress faces two under-explored challenges: limited language coverage and unreliable annotation practices. Most available datasets are in English, but even among these, high-quality, human-annotated data remains scarce. As a result, many studies rely on available pre-labeled datasets without examining their annotation process or label reliability. The lack of datasets in other languages further limits the global realization of suicide prevention via artificial intelligence (AI). In this study, we address one of these gaps by constructing a novel Turkish suicidal ideation corpus derived from social media posts and introducing a resource-efficient annotation framework involving three human annotators and two large language models (LLMs). We then address the remaining gaps by performing a bidirectional evaluation of label reliability and model consistency across this dataset and three popular English suicidal ideation detection datasets, using transfer learning through eight pre-trained sentiment and emotion classifiers. These transformers help assess annotation consistency and benchmark model performance against manually labeled data. Our findings underscore the need for more rigorous, language-inclusive approaches to annotation and evaluation in mental health natural language processing (NLP) while demonstrating the questionable performance of popular models with zero-shot transfer learning. We advocate for transparency in model training and dataset construction in mental health NLP, prioritizing data and model reliability. 

---
# Mind the Gap: A Review of Arabic Post-Training Datasets and Their Limitations 

**Authors**: Mohammed Alkhowaiter, Norah Alshahrani, Saied Alshahrani, Reem I. Masoud, Alaa Alzahrani, Deema Alnuhait, Emad A. Alghamdi, Khalid Almubarak  

**Link**: [PDF](https://arxiv.org/pdf/2507.14688)  

**Abstract**: Post-training has emerged as a crucial technique for aligning pre-trained Large Language Models (LLMs) with human instructions, significantly enhancing their performance across a wide range of tasks. Central to this process is the quality and diversity of post-training datasets. This paper presents a review of publicly available Arabic post-training datasets on the Hugging Face Hub, organized along four key dimensions: (1) LLM Capabilities (e.g., Question Answering, Translation, Reasoning, Summarization, Dialogue, Code Generation, and Function Calling); (2) Steerability (e.g., persona and system prompts); (3) Alignment (e.g., cultural, safety, ethics, and fairness), and (4) Robustness. Each dataset is rigorously evaluated based on popularity, practical adoption, recency and maintenance, documentation and annotation quality, licensing transparency, and scientific contribution. Our review revealed critical gaps in the development of Arabic post-training datasets, including limited task diversity, inconsistent or missing documentation and annotation, and low adoption across the community. Finally, the paper discusses the implications of these gaps on the progress of Arabic LLMs and applications while providing concrete recommendations for future efforts in post-training dataset development. 

---
# MiroMind-M1: An Open-Source Advancement in Mathematical Reasoning via Context-Aware Multi-Stage Policy Optimization 

**Authors**: Xingxuan Li, Yao Xiao, Dianwen Ng, Hai Ye, Yue Deng, Xiang Lin, Bin Wang, Zhanfeng Mo, Chong Zhang, Yueyi Zhang, Zonglin Yang, Ruilin Li, Lei Lei, Shihao Xu, Han Zhao, Weiling Chen, Feng Ji, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2507.14683)  

**Abstract**: Large language models have recently evolved from fluent text generation to advanced reasoning across diverse domains, giving rise to reasoning language models. Among these domains, mathematical reasoning serves as a representative benchmark as it requires precise multi-step logic and abstract reasoning, which can be generalized to other tasks. While closed-source RLMs such as GPT-o3 demonstrate impressive reasoning capabilities, their proprietary nature limits transparency and reproducibility. Although many open-source projects aim to close this gap, most of them lack sufficient openness by omitting critical resources such as datasets and detailed training configurations, which hinders reproducibility. To contribute toward greater transparency in RLM development, we introduce the MiroMind-M1 series, a set of fully open-source RLMs built on the Qwen-2.5 backbone that match or exceed the performance of existing open-source RLMs. Specifically, our models are trained in two stages: SFT on a carefully curated corpus of 719K math-reasoning problems with verified CoT trajectories, followed by RLVR on 62K challenging and verifiable problems. To enhance the robustness and efficiency of the RLVR process, we introduce Context-Aware Multi-Stage Policy Optimization, an algorithm that integrates length-progressive training with an adaptive repetition penalty to encourage context-aware RL training. Our model achieves state-of-the-art or competitive performance and superior token efficiency among Qwen-2.5-based open-source 7B and 32B models on the AIME24, AIME25, and MATH benchmarks. To facilitate reproducibility, we release the complete stack: models (MiroMind-M1-SFT-7B, MiroMind-M1-RL-7B, MiroMind-M1-RL-32B); datasets (MiroMind-M1-SFT-719K, MiroMind-M1-RL-62K); and all training and evaluation configurations. We hope these resources will support further research and foster community advancement. 

---
# Large Language Models as Medical Codes Selectors: a benchmark using the International Classification of Primary Care 

**Authors**: Vinicius Anjos de Almeida, Vinicius de Camargo, Raquel Gómez-Bravo, Egbert van der Haring, Kees van Boven, Marcelo Finger, Luis Fernandez Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2507.14681)  

**Abstract**: Background: Medical coding structures healthcare data for research, quality monitoring, and policy. This study assesses the potential of large language models (LLMs) to assign ICPC-2 codes using the output of a domain-specific search engine.
Methods: A dataset of 437 Brazilian Portuguese clinical expressions, each annotated with ICPC-2 codes, was used. A semantic search engine (OpenAI's text-embedding-3-large) retrieved candidates from 73,563 labeled concepts. Thirty-three LLMs were prompted with each query and retrieved results to select the best-matching ICPC-2 code. Performance was evaluated using F1-score, along with token usage, cost, response time, and format adherence.
Results: Twenty-eight models achieved F1-score > 0.8; ten exceeded 0.85. Top performers included gpt-4.5-preview, o3, and gemini-2.5-pro. Retriever optimization can improve performance by up to 4 points. Most models returned valid codes in the expected format, with reduced hallucinations. Smaller models (<3B) struggled with formatting and input length.
Conclusions: LLMs show strong potential for automating ICPC-2 coding, even without fine-tuning. This work offers a benchmark and highlights challenges, but findings are limited by dataset scope and setup. Broader, multilingual, end-to-end evaluations are needed for clinical validation. 

---
# Mangosteen: An Open Thai Corpus for Language Model Pretraining 

**Authors**: Wannaphong Phatthiyaphaibun, Can Udomcharoenchaikit, Pakpoom Singkorapoom, Kunat Pipatanakul, Ekapol Chuangsuwanich, Peerat Limkonchotiwat, Sarana Nutanong  

**Link**: [PDF](https://arxiv.org/pdf/2507.14664)  

**Abstract**: Pre-training data shapes a language model's quality, but raw web text is noisy and demands careful cleaning. Existing large-scale corpora rely on English-centric or language-agnostic pipelines whose heuristics do not capture Thai script or cultural nuances, leaving risky material such as gambling content untreated. Prior Thai-specific efforts customize pipelines or build new ones, yet seldom release their data or document design choices, hindering reproducibility and raising the question of how to construct a transparent, high-quality Thai corpus. We introduce Mangosteen: a 47 billion-token Thai corpus built through a Thai-adapted Dolma pipeline that includes custom rule-based language ID, revised C4/Gopher quality filters, and Thai-trained content filters, plus curated non-web sources such as Wikipedia, Royal Gazette texts, OCR-extracted books, and CC-licensed YouTube subtitles. Systematic ablations using GPT-2 show the pipeline trims CommonCrawl from 202M to 25M documents while raising SEA-HELM NLG from 3 to 11; an 8B-parameter SEA-LION model continually pre-trained on Mangosteen then surpasses SEA-LION-v3 and Llama-3.1 by about four points on Thai benchmarks. We release the full pipeline code, cleaning manifests, corpus snapshot, and all checkpoints, providing a fully reproducible foundation for future Thai and regional LLM research. 

---
# Cleanse: Uncertainty Estimation Approach Using Clustering-based Semantic Consistency in LLMs 

**Authors**: Minsuh Joo, Hyunsoo Cho  

**Link**: [PDF](https://arxiv.org/pdf/2507.14649)  

**Abstract**: Despite the outstanding performance of large language models (LLMs) across various NLP tasks, hallucinations in LLMs--where LLMs generate inaccurate responses--remains as a critical problem as it can be directly connected to a crisis of building safe and reliable LLMs. Uncertainty estimation is primarily used to measure hallucination levels in LLM responses so that correct and incorrect answers can be distinguished clearly. This study proposes an effective uncertainty estimation approach, \textbf{Cl}ust\textbf{e}ring-based sem\textbf{an}tic con\textbf{s}ist\textbf{e}ncy (\textbf{Cleanse}). Cleanse quantifies the uncertainty with the proportion of the intra-cluster consistency in the total consistency between LLM hidden embeddings which contain adequate semantic information of generations, by employing clustering. The effectiveness of Cleanse for detecting hallucination is validated using four off-the-shelf models, LLaMA-7B, LLaMA-13B, LLaMA2-7B and Mistral-7B and two question-answering benchmarks, SQuAD and CoQA. 

---
# Linear Relational Decoding of Morphology in Language Models 

**Authors**: Eric Xia, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2507.14640)  

**Abstract**: A two-part affine approximation has been found to be a good approximation for transformer computations over certain subject object relations. Adapting the Bigger Analogy Test Set, we show that the linear transformation Ws, where s is a middle layer representation of a subject token and W is derived from model derivatives, is also able to accurately reproduce final object states for many relations. This linear technique is able to achieve 90% faithfulness on morphological relations, and we show similar findings multi-lingually and across models. Our findings indicate that some conceptual relationships in language models, such as morphology, are readily interpretable from latent space, and are sparsely encoded by cross-layer linear transformations. 

---
# Retrieval-Augmented Clinical Benchmarking for Contextual Model Testing in Kenyan Primary Care: A Methodology Paper 

**Authors**: Fred Mutisya, Shikoh Gitau, Christine Syovata, Diana Oigara, Ibrahim Matende, Muna Aden, Munira Ali, Ryan Nyotu, Diana Marion, Job Nyangena, Nasubo Ongoma, Keith Mbae, Elizabeth Wamicha, Eric Mibuari, Jean Philbert Nsengemana, Talkmore Chidede  

**Link**: [PDF](https://arxiv.org/pdf/2507.14615)  

**Abstract**: Large Language Models(LLMs) hold promise for improving healthcare access in low-resource settings, but their effectiveness in African primary care remains underexplored. We present a methodology for creating a benchmark dataset and evaluation framework focused on Kenyan Level 2 and 3 clinical care. Our approach uses retrieval augmented generation (RAG) to ground clinical questions in Kenya's national guidelines, ensuring alignment with local standards. These guidelines were digitized, chunked, and indexed for semantic retrieval. Gemini Flash 2.0 Lite was then prompted with guideline excerpts to generate realistic clinical scenarios, multiple-choice questions, and rationale based answers in English and Swahili. Kenyan physicians co-created and refined the dataset, and a blinded expert review process ensured clinical accuracy, clarity, and cultural appropriateness. The resulting Alama Health QA dataset includes thousands of regulator-aligned question answer pairs across common outpatient conditions. Beyond accuracy, we introduce evaluation metrics that test clinical reasoning, safety, and adaptability such as rare case detection (Needle in the Haystack), stepwise logic (Decision Points), and contextual adaptability. Initial results reveal significant performance gaps when LLMs are applied to localized scenarios, consistent with findings that LLM accuracy is lower on African medical content than on US-based benchmarks. This work offers a replicable model for guideline-driven, dynamic benchmarking to support safe AI deployment in African health systems. 

---
# Backtranslation and paraphrasing in the LLM era? Comparing data augmentation methods for emotion classification 

**Authors**: Łukasz Radliński, Mateusz Guściora, Jan Kocoń  

**Link**: [PDF](https://arxiv.org/pdf/2507.14590)  

**Abstract**: Numerous domain-specific machine learning tasks struggle with data scarcity and class imbalance. This paper systematically explores data augmentation methods for NLP, particularly through large language models like GPT. The purpose of this paper is to examine and evaluate whether traditional methods such as paraphrasing and backtranslation can leverage a new generation of models to achieve comparable performance to purely generative methods. Methods aimed at solving the problem of data scarcity and utilizing ChatGPT were chosen, as well as an exemplary dataset. We conducted a series of experiments comparing four different approaches to data augmentation in multiple experimental setups. We then evaluated the results both in terms of the quality of generated data and its impact on classification performance. The key findings indicate that backtranslation and paraphrasing can yield comparable or even better results than zero and a few-shot generation of examples. 

---
# Explainable Collaborative Problem Solving Diagnosis with BERT using SHAP and its Implications for Teacher Adoption 

**Authors**: Kester Wong, Sahan Bulathwela, Mutlu Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2507.14584)  

**Abstract**: The use of Bidirectional Encoder Representations from Transformers (BERT) model and its variants for classifying collaborative problem solving (CPS) has been extensively explored within the AI in Education community. However, limited attention has been given to understanding how individual tokenised words in the dataset contribute to the model's classification decisions. Enhancing the explainability of BERT-based CPS diagnostics is essential to better inform end users such as teachers, thereby fostering greater trust and facilitating wider adoption in education. This study undertook a preliminary step towards model transparency and explainability by using SHapley Additive exPlanations (SHAP) to examine how different tokenised words in transcription data contributed to a BERT model's classification of CPS processes. The findings suggested that well-performing classifications did not necessarily equate to a reasonable explanation for the classification decisions. Particular tokenised words were used frequently to affect classifications. The analysis also identified a spurious word, which contributed positively to the classification but was not semantically meaningful to the class. While such model transparency is unlikely to be useful to an end user to improve their practice, it can help them not to overrely on LLM diagnostics and ignore their human expertise. We conclude the workshop paper by noting that the extent to which the model appropriately uses the tokens for its classification is associated with the number of classes involved. It calls for an investigation into the exploration of ensemble model architectures and the involvement of human-AI complementarity for CPS diagnosis, since considerable human reasoning is still required for fine-grained discrimination of CPS subskills. 

---
# Exploring Human-AI Complementarity in CPS Diagnosis Using Unimodal and Multimodal BERT Models 

**Authors**: Kester Wong, Sahan Bulathwela, Mutlu Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2507.14579)  

**Abstract**: Detecting collaborative problem solving (CPS) indicators from dialogue using machine learning techniques is a significant challenge for the field of AI in Education. Recent studies have explored the use of Bidirectional Encoder Representations from Transformers (BERT) models on transcription data to reliably detect meaningful CPS indicators. A notable advancement involved the multimodal BERT variant, AudiBERT, which integrates speech and acoustic-prosodic audio features to enhance CPS diagnosis. Although initial results demonstrated multimodal improvements, the statistical significance of these enhancements remained unclear, and there was insufficient guidance on leveraging human-AI complementarity for CPS diagnosis tasks. This workshop paper extends the previous research by highlighting that the AudiBERT model not only improved the classification of classes that were sparse in the dataset, but it also had statistically significant class-wise improvements over the BERT model for classifications in the social-cognitive dimension. However, similar significant class-wise improvements over the BERT model were not observed for classifications in the affective dimension. A correlation analysis highlighted that larger training data was significantly associated with higher recall performance for both the AudiBERT and BERT models. Additionally, the precision of the BERT model was significantly associated with high inter-rater agreement among human coders. When employing the BERT model to diagnose indicators within these subskills that were well-detected by the AudiBERT model, the performance across all indicators was inconsistent. We conclude the paper by outlining a structured approach towards achieving human-AI complementarity for CPS diagnosis, highlighting the crucial inclusion of model explainability to support human agency and engagement in the reflective coding process. 

---
# XL-DURel: Finetuning Sentence Transformers for Ordinal Word-in-Context Classification 

**Authors**: Sachin Yadav, Dominik Schlechtweg  

**Link**: [PDF](https://arxiv.org/pdf/2507.14578)  

**Abstract**: We propose XL-DURel, a finetuned, multilingual Sentence Transformer model optimized for ordinal Word-in-Context classification. We test several loss functions for regression and ranking tasks managing to outperform previous models on ordinal and binary data with a ranking objective based on angular distance in complex space. We further show that binary WiC can be treated as a special case of ordinal WiC and that optimizing models for the general ordinal task improves performance on the more specific binary task. This paves the way for a unified treatment of WiC modeling across different task formulations. 

---
# X-Intelligence 3.0: Training and Evaluating Reasoning LLM for Semiconductor Display 

**Authors**: Xiaolin Yan, Yangxing Liu, Jiazhang Zheng, Chi Liu, Mingyu Du, Caisheng Chen, Haoyang Liu, Ming Ding, Yuan Li, Qiuping Liao, Linfeng Li, Zhili Mei, Siyu Wan, Li Li, Ruyi Zhong, Jiangling Yu, Xule Liu, Huihui Hu, Jiameng Yue, Ruohui Cheng, Qi Yang, Liangqing Wu, Ke Zhu, Chi Zhang, Chufei Jing, Yifan Zhou, Yan Liang, Dongdong Li, Zhaohui Wang, Bin Zhao, Mingzhou Wu, Mingzhong Zhou, Peng Du, Zuomin Liao, Chao Dai, Pengfei Liang, Xiaoguang Zhu, Yu Zhang, Yu Gu, Kun Pan, Yuan Wu, Yanqing Guan, Shaojing Wu, Zikang Feng, Xianze Ma, Peishan Cheng, Wenjuan Jiang, Jing Ba, Huihao Yu, Zeping Hu, Yuan Xu, Zhiwei Liu, He Wang, Zhenguo Lin, Ming Liu, Yanhong Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.14430)  

**Abstract**: Large language models (LLMs) have recently achieved significant advances in reasoning and demonstrated their advantages in solving challenging problems. Yet, their effectiveness in the semiconductor display industry remains limited due to a lack of domain-specific training and expertise. To bridge this gap, we present X-Intelligence 3.0, the first high-performance reasoning model specifically developed for the semiconductor display industry. This model is designed to deliver expert-level understanding and reasoning for the industry's complex challenges. Leveraging a carefully curated industry knowledge base, the model undergoes supervised fine-tuning and reinforcement learning to enhance its reasoning and comprehension capabilities. To further accelerate development, we implemented an automated evaluation framework that simulates expert-level assessments. We also integrated a domain-specific retrieval-augmented generation (RAG) mechanism, resulting in notable performance gains on benchmark datasets. Despite its relatively compact size of 32 billion parameters, X-Intelligence 3.0 outperforms SOTA DeepSeek-R1-671B across multiple evaluations. This demonstrates its exceptional efficiency and establishes it as a powerful solution to the longstanding reasoning challenges faced by the semiconductor display industry. 

---
# Error-Aware Curriculum Learning for Biomedical Relation Classification 

**Authors**: Sinchani Chakraborty, Sudeshna Sarkar, Pawan Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2507.14374)  

**Abstract**: Relation Classification (RC) in biomedical texts is essential for constructing knowledge graphs and enabling applications such as drug repurposing and clinical decision-making. We propose an error-aware teacher--student framework that improves RC through structured guidance from a large language model (GPT-4o). Prediction failures from a baseline student model are analyzed by the teacher to classify error types, assign difficulty scores, and generate targeted remediations, including sentence rewrites and suggestions for KG-based enrichment. These enriched annotations are used to train a first student model via instruction tuning. This model then annotates a broader dataset with difficulty scores and remediation-enhanced inputs. A second student is subsequently trained via curriculum learning on this dataset, ordered by difficulty, to promote robust and progressive learning. We also construct a heterogeneous biomedical knowledge graph from PubMed abstracts to support context-aware RC. Our approach achieves new state-of-the-art performance on 4 of 5 PPI datasets and the DDI dataset, while remaining competitive on ChemProt. 

---
# Text-to-SQL for Enterprise Data Analytics 

**Authors**: Albert Chen, Manas Bundele, Gaurav Ahlawat, Patrick Stetz, Zhitao Wang, Qiang Fei, Donghoon Jung, Audrey Chu, Bharadwaj Jayaraman, Ayushi Panth, Yatin Arora, Sourav Jain, Renjith Varma, Alexey Ilin, Iuliia Melnychuk, Chelsea Chueh, Joyan Sil, Xiaofeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14372)  

**Abstract**: The introduction of large language models has brought rapid progress on Text-to-SQL benchmarks, but it is not yet easy to build a working enterprise solution. In this paper, we present insights from building an internal chatbot that enables LinkedIn's product managers, engineers, and operations teams to self-serve data insights from a large, dynamic data lake. Our approach features three components. First, we construct a knowledge graph that captures up-to-date semantics by indexing database metadata, historical query logs, wikis, and code. We apply clustering to identify relevant tables for each team or product area. Second, we build a Text-to-SQL agent that retrieves and ranks context from the knowledge graph, writes a query, and automatically corrects hallucinations and syntax errors. Third, we build an interactive chatbot that supports various user intents, from data discovery to query writing to debugging, and displays responses in rich UI elements to encourage follow-up chats. Our chatbot has over 300 weekly users. Expert review shows that 53% of its responses are correct or close to correct on an internal benchmark set. Through ablation studies, we identify the most important knowledge graph and modeling components, offering a practical path for developing enterprise Text-to-SQL solutions. 

---
# Can LLMs Infer Personality from Real World Conversations? 

**Authors**: Jianfeng Zhu, Ruoming Jin, Karin G. Coifman  

**Link**: [PDF](https://arxiv.org/pdf/2507.14355)  

**Abstract**: Large Language Models (LLMs) such as OpenAI's GPT-4 and Meta's LLaMA offer a promising approach for scalable personality assessment from open-ended language. However, inferring personality traits remains challenging, and earlier work often relied on synthetic data or social media text lacking psychometric validity. We introduce a real-world benchmark of 555 semi-structured interviews with BFI-10 self-report scores for evaluating LLM-based personality inference. Three state-of-the-art LLMs (GPT-4.1 Mini, Meta-LLaMA, and DeepSeek) were tested using zero-shot prompting for BFI-10 item prediction and both zero-shot and chain-of-thought prompting for Big Five trait inference. All models showed high test-retest reliability, but construct validity was limited: correlations with ground-truth scores were weak (max Pearson's $r = 0.27$), interrater agreement was low (Cohen's $\kappa < 0.10$), and predictions were biased toward moderate or high trait levels. Chain-of-thought prompting and longer input context modestly improved distributional alignment, but not trait-level accuracy. These results underscore limitations in current LLM-based personality inference and highlight the need for evidence-based development for psychological applications. 

---
# What Makes You CLIC: Detection of Croatian Clickbait Headlines 

**Authors**: Marija Anđedelić, Dominik Šipek, Laura Majer, Jan Šnajder  

**Link**: [PDF](https://arxiv.org/pdf/2507.14314)  

**Abstract**: Online news outlets operate predominantly on an advertising-based revenue model, compelling journalists to create headlines that are often scandalous, intriguing, and provocative -- commonly referred to as clickbait. Automatic detection of clickbait headlines is essential for preserving information quality and reader trust in digital media and requires both contextual understanding and world knowledge. For this task, particularly in less-resourced languages, it remains unclear whether fine-tuned methods or in-context learning (ICL) yield better results. In this paper, we compile CLIC, a novel dataset for clickbait detection of Croatian news headlines spanning a 20-year period and encompassing mainstream and fringe outlets. We fine-tune the BERTić model on this task and compare its performance to LLM-based ICL methods with prompts both in Croatian and English. Finally, we analyze the linguistic properties of clickbait. We find that nearly half of the analyzed headlines contain clickbait, and that finetuned models deliver better results than general LLMs. 

---
# How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs 

**Authors**: Karin de Langis, Jong Inn Park, Andreas Schramm, Bin Hu, Khanh Chi Le, Michael Mensink, Ahn Thu Tong, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14307)  

**Abstract**: Large language models (LLMs) exhibit increasingly sophisticated linguistic capabilities, yet the extent to which these behaviors reflect human-like cognition versus advanced pattern recognition remains an open question. In this study, we investigate how LLMs process the temporal meaning of linguistic aspect in narratives that were previously used in human studies. Using an Expert-in-the-Loop probing pipeline, we conduct a series of targeted experiments to assess whether LLMs construct semantic representations and pragmatic inferences in a human-like manner. Our findings show that LLMs over-rely on prototypicality, produce inconsistent aspectual judgments, and struggle with causal reasoning derived from aspect, raising concerns about their ability to fully comprehend narratives. These results suggest that LLMs process aspect fundamentally differently from humans and lack robust narrative understanding. Beyond these empirical findings, we develop a standardized experimental framework for the reliable assessment of LLMs' cognitive and linguistic capabilities. 

---
# Aligning Large Language Models to Low-Resource Languages through LLM-Based Selective Translation: A Systematic Study 

**Authors**: Rakesh Paul, Anusha Kamath, Kanishk Singla, Raviraj Joshi, Utkarsh Vaidya, Sanjay Singh Chauhan, Niranjan Wartikar  

**Link**: [PDF](https://arxiv.org/pdf/2507.14304)  

**Abstract**: Multilingual large language models (LLMs) often demonstrate a performance gap between English and non-English languages, particularly in low-resource settings. Aligning these models to low-resource languages is essential yet challenging due to limited high-quality data. While English alignment datasets are readily available, curating equivalent data in other languages is expensive and time-consuming. A common workaround is to translate existing English alignment data; however, standard translation techniques often fail to preserve critical elements such as code, mathematical expressions, and structured formats like JSON. In this work, we investigate LLM-based selective translation, a technique that selectively translates only the translatable parts of a text while preserving non-translatable content and sentence structure. We conduct a systematic study to explore key questions around this approach, including its effectiveness compared to vanilla translation, the importance of filtering noisy outputs, and the benefits of mixing translated samples with original English data during alignment. Our experiments focus on the low-resource Indic language Hindi and compare translations generated by Google Cloud Translation (GCP) and Llama-3.1-405B. The results highlight the promise of selective translation as a practical and effective method for improving multilingual alignment in LLMs. 

---
# In-Depth and In-Breadth: Pre-training Multimodal Language Models Customized for Comprehensive Chart Understanding 

**Authors**: Wan-Cyuan Fan, Yen-Chun Chen, Mengchen Liu, Alexander Jacobson, Lu Yuan, Leonid Sigal  

**Link**: [PDF](https://arxiv.org/pdf/2507.14298)  

**Abstract**: Recent methods for customizing Large Vision Language Models (LVLMs) for domain-specific tasks have shown promising results in scientific chart comprehension. However, existing approaches face two major limitations: First, they rely on paired data from only a few chart types, limiting generalization to wide range of chart types. Secondly, they lack targeted pre-training for chart-data alignment, which hampers the model's understanding of underlying data. In this paper, we introduce ChartScope, an LVLM optimized for in-depth chart comprehension across diverse chart types. We propose an efficient data generation pipeline that synthesizes paired data for a wide range of chart types, along with a novel Dual-Path training strategy that enabling the model to succinctly capture essential data details while preserving robust reasoning capabilities by incorporating reasoning over the underlying data. Lastly, we establish ChartDQA, a new benchmark for evaluating not only question-answering at different levels but also underlying data understanding. Experimental results demonstrate that ChartScope significantly enhances comprehension on a wide range of chart types. The code and data are available at this https URL. 

---
# Promptomatix: An Automatic Prompt Optimization Framework for Large Language Models 

**Authors**: Rithesh Murthy, Ming Zhu, Liangwei Yang, Jielin Qiu, Juntao Tan, Shelby Heinecke, Huan Wang, Caiming Xiong, Silvio Savarese  

**Link**: [PDF](https://arxiv.org/pdf/2507.14241)  

**Abstract**: Large Language Models (LLMs) perform best with well-crafted prompts, yet prompt engineering remains manual, inconsistent, and inaccessible to non-experts. We introduce Promptomatix, an automatic prompt optimization framework that transforms natural language task descriptions into high-quality prompts without requiring manual tuning or domain expertise. Promptomatix supports both a lightweight meta-prompt-based optimizer and a DSPy-powered compiler, with modular design enabling future extension to more advanced frameworks. The system analyzes user intent, generates synthetic training data, selects prompting strategies, and refines prompts using cost-aware objectives. Evaluated across 5 task categories, Promptomatix achieves competitive or superior performance compared to existing libraries, while reducing prompt length and computational overhead making prompt optimization scalable and efficient. 

---
# HuggingGraph: Understanding the Supply Chain of LLM Ecosystem 

**Authors**: Mohammad Shahedur Rahman, Peng Gao, Yuede Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.14240)  

**Abstract**: Large language models (LLMs) leverage deep learning to process and predict sequences of words from context, enabling them to perform various NLP tasks, such as translation, summarization, question answering, and content generation. However, the growing size and complexity of developing, training, and deploying advanced LLMs require extensive computational resources and large datasets. This creates a barrier for users. As a result, platforms that host models and datasets are widely used. For example, Hugging Face, one of the most popular platforms, hosted 1.8 million models and 450K datasets by June 2025, with no sign of slowing down. Since many LLMs are built from base models, pre-trained models, and external datasets, they can inherit vulnerabilities, biases, or malicious components from earlier models or datasets. Therefore, it is critical to understand the origin and development of these components to better detect potential risks, improve model fairness, and ensure compliance. Motivated by this, our project aims to study the relationships between models and datasets, which are core components of the LLM supply chain. First, we design a method to systematically collect LLM supply chain data. Using this data, we build a directed heterogeneous graph to model the relationships between models and datasets, resulting in a structure with 397,376 nodes and 453,469 edges. We then perform various analyses and uncover several findings, such as: (i) the LLM supply chain graph is large, sparse, and follows a power-law degree distribution; (ii) it features a densely connected core and a fragmented periphery; (iii) datasets play pivotal roles in training; (iv) strong interdependence exists between models and datasets; and (v) the graph is dynamic, with daily updates reflecting the ecosystem's ongoing evolution. 

---
# CCL-XCoT: An Efficient Cross-Lingual Knowledge Transfer Method for Mitigating Hallucination Generation 

**Authors**: Weihua Zheng, Roy Ka-Wei Lee, Zhengyuan Liu, Kui Wu, AiTi Aw, Bowei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.14239)  

**Abstract**: Multilingual Large Language Models(MLLMs) demonstrate strong generalization across languages, yet they remain prone to hallucinations, especially in low-resource languages, due to training data imbalances. These hallucinations, which include inaccurate or fabricated outputs, are particularly problematic in domain-specific generation tasks (Chataigner et al., 2024). To address this challenge, we propose CCL-XCoT(Curriculum-based Contrastive Learning-based Cross-lingual Chain-of-Thought), a two-stage fine-tuning framework for mitigating hallucination in MLLMs. Our approach first enhances cross-lingual semantic alignment through curriculum-based contrastive learning combined with next-token prediction during continued pre-training. Building on this foundation, we then introduce a cross-lingual Chain-of-Thought (XCoT) prompting strategy during instruction fine-tuning, which guides the model to reason in a high-resource language before generating answers in the target low-resource language. Experimental results show that CCL-XCoT reduces hallucination rates by up to 62% and substantially improves factual knowledge transfer across language pairs, without relying on external retrieval or multi-model ensembles. 

---
# Language Models Change Facts Based on the Way You Talk 

**Authors**: Matthew Kearney, Reuben Binns, Yarin Gal  

**Link**: [PDF](https://arxiv.org/pdf/2507.14238)  

**Abstract**: Large language models (LLMs) are increasingly being used in user-facing applications, from providing medical consultations to job interview advice. Recent research suggests that these models are becoming increasingly proficient at inferring identity information about the author of a piece of text from linguistic patterns as subtle as the choice of a few words. However, little is known about how LLMs use this information in their decision-making in real-world applications. We perform the first comprehensive analysis of how identity markers present in a user's writing bias LLM responses across five different high-stakes LLM applications in the domains of medicine, law, politics, government benefits, and job salaries. We find that LLMs are extremely sensitive to markers of identity in user queries and that race, gender, and age consistently influence LLM responses in these applications. For instance, when providing medical advice, we find that models apply different standards of care to individuals of different ethnicities for the same symptoms; we find that LLMs are more likely to alter answers to align with a conservative (liberal) political worldview when asked factual questions by older (younger) individuals; and that LLMs recommend lower salaries for non-White job applicants and higher salaries for women compared to men. Taken together, these biases mean that the use of off-the-shelf LLMs for these applications may cause harmful differences in medical care, foster wage gaps, and create different political factual realities for people of different identities. Beyond providing an analysis, we also provide new tools for evaluating how subtle encoding of identity in users' language choices impacts model decisions. Given the serious implications of these findings, we recommend that similar thorough assessments of LLM use in user-facing applications are conducted before future deployment. 

---
# Beyond Architectures: Evaluating the Role of Contextual Embeddings in Detecting Bipolar Disorder on Social Media 

**Authors**: Khalid Hasan, Jamil Saquer  

**Link**: [PDF](https://arxiv.org/pdf/2507.14231)  

**Abstract**: Bipolar disorder is a chronic mental illness frequently underdiagnosed due to subtle early symptoms and social stigma. This paper explores the advanced natural language processing (NLP) models for recognizing signs of bipolar disorder based on user-generated social media text. We conduct a comprehensive evaluation of transformer-based models (BERT, RoBERTa, ALBERT, ELECTRA, DistilBERT) and Long Short Term Memory (LSTM) models based on contextualized (BERT) and static (GloVe, Word2Vec) word embeddings. Experiments were performed on a large, annotated dataset of Reddit posts after confirming their validity through sentiment variance and judgmental analysis. Our results demonstrate that RoBERTa achieves the highest performance among transformer models with an F1 score of ~98% while LSTM models using BERT embeddings yield nearly identical results. In contrast, LSTMs trained on static embeddings fail to capture meaningful patterns, scoring near-zero F1. These findings underscore the critical role of contextual language modeling in detecting bipolar disorder. In addition, we report model training times and highlight that DistilBERT offers an optimal balance between efficiency and accuracy. In general, our study offers actionable insights for model selection in mental health NLP applications and validates the potential of contextualized language models to support early bipolar disorder screening. 

---
# Let's Measure the Elephant in the Room: Facilitating Personalized Automated Analysis of Privacy Policies at Scale 

**Authors**: Rui Zhao, Vladyslav Melnychuk, Jun Zhao, Jesse Wright, Nigel Shadbolt  

**Link**: [PDF](https://arxiv.org/pdf/2507.14214)  

**Abstract**: In modern times, people have numerous online accounts, but they rarely read the Terms of Service or Privacy Policy of those sites despite claiming otherwise. This paper introduces PoliAnalyzer, a neuro-symbolic system that assists users with personalized privacy policy analysis. PoliAnalyzer uses Natural Language Processing (NLP) to extract formal representations of data usage practices from policy texts. In favor of deterministic, logical inference is applied to compare user preferences with the formal privacy policy representation and produce a compliance report. To achieve this, we extend an existing formal Data Terms of Use policy language to model privacy policies as app policies and user preferences as data policies. In our evaluation using our enriched PolicyIE dataset curated by legal experts, PoliAnalyzer demonstrated high accuracy in identifying relevant data usage practices, achieving F1-score of 90-100% across most tasks. Additionally, we demonstrate how PoliAnalyzer can model diverse user data-sharing preferences, derived from prior research as 23 user profiles, and perform compliance analysis against the top 100 most-visited websites. This analysis revealed that, on average, 95.2% of a privacy policy's segments do not conflict with the analyzed user preferences, enabling users to concentrate on understanding the 4.8% (636 / 13205) that violates preferences, significantly reducing cognitive burden. Further, we identified common practices in privacy policies that violate user expectations - such as the sharing of location data with 3rd parties. This paper demonstrates that PoliAnalyzer can support automated personalized privacy policy analysis at scale using off-the-shelf NLP tools. This sheds light on a pathway to help individuals regain control over their data and encourage societal discussions on platform data practices to promote a fairer power dynamic. 

---
# Open-Source LLMs Collaboration Beats Closed-Source LLMs: A Scalable Multi-Agent System 

**Authors**: Shengji Tang, Jianjian Cao, Weihao Lin, Jiale Hong, Bo Zhang, Shuyue Hu, Lei Bai, Tao Chen, Wanli Ouyang, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.14200)  

**Abstract**: This paper aims to demonstrate the potential and strengths of open-source collectives. It leads to a promising question: Can we harness multiple open-source LLMs to match or even beat the closed-source LLMs? To answer this, we propose SMACS, a scalable multi-agent collaboration system (MACS) framework with high performance. Specifically, for continuous integration of new LLMs and generalization to diverse questions, we first propose a Retrieval-based Prior Selection (RPS), which assigns a proxy performance score to each LLM to select the Top-k LLMs at the instance level for any given question. Then, we propose an Exploration-Exploitation-Driven Posterior Enhancement (EPE), encouraging the generation of diverse responses through prior dropping and selecting the high-quality response via a hybrid posterior score. Experiments on eight mainstream benchmarks validate the effectiveness of our SMACS: by integrating fifteen open-source LLMs, SMACS outperforms leading closed-source LLMs in 2025, e.g., Claude-3.7-Sonnet (+12.73%), GPT-4.1(+5.36%) and GPT-o3-mini(+5.28%) across multiple tasks. Remarkably, it even exceeds the average of best results of different datasets from both open-source LLMs (+2.86%) and closed-source LLMs (+2.04%), pushing the upper bound of intelligence. Code will be released at this https URL. 

---
# Retention analysis of edited knowledge after fine-tuning 

**Authors**: Fufang Wen, Shichang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14198)  

**Abstract**: Large language models (LLMs) store vast amounts of knowledge, which often requires updates to correct factual errors, incorporate newly acquired information, or adapt model behavior. Model editing methods have emerged as efficient solutions for such updates, offering localized and precise knowledge modification at significantly lower computational cost than continual training. In parallel, LLMs are frequently fine-tuned for a wide range of downstream tasks. However, the effect of fine-tuning on previously edited knowledge remains poorly understood. In this work, we systematically investigate how different fine-tuning objectives interact with various model editing techniques. Our findings show that edited knowledge is substantially more susceptible to forgetting during fine-tuning than intrinsic knowledge acquired through pre-training. This analysis highlights a key limitation of current editing approaches and suggests that evaluating edit robustness under downstream fine-tuning is critical for their practical deployment. We further find that freezing layers associated with edited content can significantly improve knowledge retention, offering insight into how future editing methods might be made more robust. 

---
# DeepWriter: A Fact-Grounded Multimodal Writing Assistant Based On Offline Knowledge Base 

**Authors**: Song Mao, Lejun Cheng, Pinlong Cai, Guohang Yan, Ding Wang, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.14189)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various applications. However, their use as writing assistants in specialized domains like finance, medicine, and law is often hampered by a lack of deep domain-specific knowledge and a tendency to hallucinate. Existing solutions, such as Retrieval-Augmented Generation (RAG), can suffer from inconsistency across multiple retrieval steps, while online search-based methods often degrade quality due to unreliable web content. To address these challenges, we introduce DeepWriter, a customizable, multimodal, long-form writing assistant that operates on a curated, offline knowledge base. DeepWriter leverages a novel pipeline that involves task decomposition, outline generation, multimodal retrieval, and section-by-section composition with reflection. By deeply mining information from a structured corpus and incorporating both textual and visual elements, DeepWriter generates coherent, factually grounded, and professional-grade documents. We also propose a hierarchical knowledge representation to enhance retrieval efficiency and accuracy. Our experiments on financial report generation demonstrate that DeepWriter produces high-quality, verifiable articles that surpasses existing baselines in factual accuracy and generated content quality. 

---
# GUI-G$^2$: Gaussian Reward Modeling for GUI Grounding 

**Authors**: Fei Tang, Zhangxuan Gu, Zhengxi Lu, Xuyang Liu, Shuheng Shen, Changhua Meng, Wen Wang, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15846)  

**Abstract**: Graphical User Interface (GUI) grounding maps natural language instructions to precise interface locations for autonomous interaction. Current reinforcement learning approaches use binary rewards that treat elements as hit-or-miss targets, creating sparse signals that ignore the continuous nature of spatial interactions. Motivated by human clicking behavior that naturally forms Gaussian distributions centered on target elements, we introduce GUI Gaussian Grounding Rewards (GUI-G$^2$), a principled reward framework that models GUI elements as continuous Gaussian distributions across the interface plane. GUI-G$^2$ incorporates two synergistic mechanisms: Gaussian point rewards model precise localization through exponentially decaying distributions centered on element centroids, while coverage rewards assess spatial alignment by measuring the overlap between predicted Gaussian distributions and target regions. To handle diverse element scales, we develop an adaptive variance mechanism that calibrates reward distributions based on element dimensions. This framework transforms GUI grounding from sparse binary classification to dense continuous optimization, where Gaussian distributions generate rich gradient signals that guide models toward optimal interaction positions. Extensive experiments across ScreenSpot, ScreenSpot-v2, and ScreenSpot-Pro benchmarks demonstrate that GUI-G$^2$, substantially outperforms state-of-the-art method UI-TARS-72B, with the most significant improvement of 24.7% on ScreenSpot-Pro. Our analysis reveals that continuous modeling provides superior robustness to interface variations and enhanced generalization to unseen layouts, establishing a new paradigm for spatial reasoning in GUI interaction tasks. 

---
# Hierarchical Budget Policy Optimization for Adaptive Reasoning 

**Authors**: Shangke Lyu, Linjuan Wu, Yuchen Yan, Xingyu Wu, Hao Li, Yongliang Shen, Peisheng Jiang, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15844)  

**Abstract**: Large reasoning models achieve remarkable performance through extensive chain-of-thought generation, yet exhibit significant computational inefficiency by applying uniform reasoning strategies regardless of problem complexity. We present Hierarchical Budget Policy Optimization (HBPO), a reinforcement learning framework that enables models to learn problem-specific reasoning depths without sacrificing capability. HBPO addresses the fundamental challenge of exploration space collapse in efficiency-oriented training, where penalties on long output length systematically bias models away from necessary long reasoning paths. Through hierarchical budget exploration, our approach partitions rollout samples into multiple subgroups with distinct token budgets, aiming to enable efficient resource allocation while preventing degradation of capability. We introduce differentiated reward mechanisms that create budget-aware incentives aligned with the complexity of the problem, allowing models to discover natural correspondences between task requirements and computational effort. Extensive experiments demonstrate that HBPO reduces average token usage by up to 60.6% while improving accuracy by 3.14% across four reasoning benchmarks. Unlike existing methods that impose external constraints or rely on discrete mode selection, HBPO exhibits emergent adaptive behavior where models automatically adjust reasoning depth based on problem complexity. Our results suggest that reasoning efficiency and capability are not inherently conflicting, and can be simultaneously optimized through appropriately structured hierarchical training that preserves exploration diversity. 

---
# Small LLMs Do Not Learn a Generalizable Theory of Mind via Reinforcement Learning 

**Authors**: Sneheel Sarangi, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2507.15788)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated emergent capabilities in complex reasoning, largely spurred by rule-based Reinforcement Learning (RL) techniques applied during the post-training. This has raised the question of whether similar methods can instill more nuanced, human-like social intelligence, such as a Theory of Mind (ToM), in LLMs. This paper investigates whether small-scale LLMs can acquire a robust and generalizable ToM capability through RL with verifiable rewards (RLVR). We conduct a systematic evaluation by training models on various combinations of prominent ToM datasets (HiToM, ExploreToM, FANToM) and testing for generalization on held-out datasets (e.g., OpenToM). Our findings indicate that small LLMs struggle to develop a generic ToM capability. While performance on in-distribution tasks improves, this capability fails to transfer to unseen ToM tasks with different characteristics. Furthermore, we demonstrate that prolonged RL training leads to models ``hacking'' the statistical patterns of the training datasets, resulting in significant performance gains on in-domain data but no change, or degradation of performance on out-of-distribution tasks. This suggests the learned behavior is a form of narrow overfitting rather than the acquisition of a true, abstract ToM capability. 

---
# Dissociating model architectures from inference computations 

**Authors**: Noor Sajid, Johan Medrano  

**Link**: [PDF](https://arxiv.org/pdf/2507.15776)  

**Abstract**: Parr et al., 2025 examines how auto-regressive and deep temporal models differ in their treatment of non-Markovian sequence modelling. Building on this, we highlight the need for dissociating model architectures, i.e., how the predictive distribution factorises, from the computations invoked at inference. We demonstrate that deep temporal computations are mimicked by autoregressive models by structuring context access during iterative inference. Using a transformer trained on next-token prediction, we show that inducing hierarchical temporal factorisation during iterative inference maintains predictive capacity while instantiating fewer computations. This emphasises that processes for constructing and refining predictions are not necessarily bound to their underlying model architectures. 

---
# LAPO: Internalizing Reasoning Efficiency via Length-Adaptive Policy Optimization 

**Authors**: Xingyu Wu, Yuchen Yan, Shangke Lyu, Linjuan Wu, Yiwen Qiu, Yongliang Shen, Weiming Lu, Jian Shao, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15758)  

**Abstract**: Large reasoning models have achieved remarkable performance through extended chain-of-thought sequences, yet this computational freedom leads to excessive token generation even for simple problems. We present Length-Adaptive Policy Optimization (LAPO), a novel framework that transforms reasoning length control from an external constraint into an intrinsic model capability. Unlike existing approaches that impose rigid limits or rely on post-hoc interventions, LAPO enables models to internalize an understanding of appropriate reasoning depth through a two-stage reinforcement learning process. In the first stage, models learn natural reasoning patterns by discovering the statistical distribution of successful solution lengths. The second stage leverages these patterns as meta-cognitive guidance, embedding them directly within the model's reasoning context to ensure inference-time flexibility. Experiments on mathematical reasoning benchmarks demonstrate that LAPO reduces token usage by up to 40.9\% while improving accuracy by 2.3\%. Our analysis reveals that models trained with LAPO develop emergent abilities to allocate computational resources based on problem complexity, achieving efficient reasoning without sacrificing quality. 

---
# Towards physician-centered oversight of conversational diagnostic AI 

**Authors**: Elahe Vedadi, David Barrett, Natalie Harris, Ellery Wulczyn, Shashir Reddy, Roma Ruparel, Mike Schaekermann, Tim Strother, Ryutaro Tanno, Yash Sharma, Jihyeon Lee, Cían Hughes, Dylan Slack, Anil Palepu, Jan Freyberg, Khaled Saab, Valentin Liévin, Wei-Hung Weng, Tao Tu, Yun Liu, Nenad Tomasev, Kavita Kulkarni, S. Sara Mahdavi, Kelvin Guu, Joëlle Barral, Dale R. Webster, James Manyika, Avinatan Hassidim, Katherine Chou, Yossi Matias, Pushmeet Kohli, Adam Rodman, Vivek Natarajan, Alan Karthikesalingam, David Stutz  

**Link**: [PDF](https://arxiv.org/pdf/2507.15743)  

**Abstract**: Recent work has demonstrated the promise of conversational AI systems for diagnostic dialogue. However, real-world assurance of patient safety means that providing individual diagnoses and treatment plans is considered a regulated activity by licensed professionals. Furthermore, physicians commonly oversee other team members in such activities, including nurse practitioners (NPs) or physician assistants/associates (PAs). Inspired by this, we propose a framework for effective, asynchronous oversight of the Articulate Medical Intelligence Explorer (AMIE) AI system. We propose guardrailed-AMIE (g-AMIE), a multi-agent system that performs history taking within guardrails, abstaining from individualized medical advice. Afterwards, g-AMIE conveys assessments to an overseeing primary care physician (PCP) in a clinician cockpit interface. The PCP provides oversight and retains accountability of the clinical decision. This effectively decouples oversight from intake and can thus happen asynchronously. In a randomized, blinded virtual Objective Structured Clinical Examination (OSCE) of text consultations with asynchronous oversight, we compared g-AMIE to NPs/PAs or a group of PCPs under the same guardrails. Across 60 scenarios, g-AMIE outperformed both groups in performing high-quality intake, summarizing cases, and proposing diagnoses and management plans for the overseeing PCP to review. This resulted in higher quality composite decisions. PCP oversight of g-AMIE was also more time-efficient than standalone PCP consultations in prior work. While our study does not replicate existing clinical practices and likely underestimates clinicians' capabilities, our results demonstrate the promise of asynchronous oversight as a feasible paradigm for diagnostic AI systems to operate under expert human oversight for enhancing real-world care. 

---
# Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training 

**Authors**: Kailai Yang, Xiao Liu, Lei Ji, Hao Li, Yeyun Gong, Peng Cheng, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15640)  

**Abstract**: Continual pre-training on small-scale task-specific data is an effective method for improving large language models in new target fields, yet it risks catastrophic forgetting of their original capabilities. A common solution is to re-weight training data mixtures from source and target fields on a domain space to achieve balanced performance. Previous domain reweighting strategies rely on manual designation with certain heuristics based on human intuition or empirical results. In this work, we prove that more general heuristics can be parameterized by proposing Data Mixing Agent, the first model-based, end-to-end framework that learns to re-weight domains. The agent learns generalizable heuristics through reinforcement learning on large quantities of data mixing trajectories with corresponding feedback from an evaluation environment. Experiments in continual pre-training on math reasoning show that Data Mixing Agent outperforms strong baselines in achieving balanced performance across source and target field benchmarks. Furthermore, it generalizes well across unseen source fields, target models, and domain spaces without retraining. Direct application to the code generation field also indicates its adaptability across target domains. Further analysis showcases the agents' well-aligned heuristics with human intuitions and their efficiency in achieving superior model performance with less source-field data. 

---
# Off-Policy Corrected Reward Modeling for Reinforcement Learning from Human Feedback 

**Authors**: Johannes Ackermann, Takashi Ishida, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2507.15507)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) allows us to train models, such as language models (LMs), to follow complex human preferences. In RLHF for LMs, we first train an LM using supervised fine-tuning, sample pairs of responses, obtain human feedback, and use the resulting data to train a reward model (RM). RL methods are then used to train the LM to maximize the reward given by the RM. As training progresses, the responses generated by the LM no longer resemble the responses seen by the RM during training, leading to the RM becoming inaccurate. The score given by the RM keeps increasing, but the learned behavior no longer matches the human preferences. This issue is known as overoptimization. We investigate overoptimization from the point of view of distribution shift and show that the shift results in an inconsistent estimate of the RM parameters, leading to an inconsistent estimate of the policy gradient. We propose Off-Policy Corrected Reward Modeling (OCRM), which iteratively off-policy corrects the RM using importance weighting, without requiring new labels or samples. This results in a more accurate RM, which empirically leads to an improved final policy. We validate our approach in experiments with summarization and chatbot datasets and show that it performs significantly better than standard RLHF methods and baselines. Our implementation is available at this https URL 

---
# A2TTS: TTS for Low Resource Indian Languages 

**Authors**: Ayush Singh Bhadoriya, Abhishek Nikunj Shinde, Isha Pandey, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2507.15272)  

**Abstract**: We present a speaker conditioned text-to-speech (TTS) system aimed at addressing challenges in generating speech for unseen speakers and supporting diverse Indian languages. Our method leverages a diffusion-based TTS architecture, where a speaker encoder extracts embeddings from short reference audio samples to condition the DDPM decoder for multispeaker generation. To further enhance prosody and naturalness, we employ a cross-attention based duration prediction mechanism that utilizes reference audio, enabling more accurate and speaker consistent timing. This results in speech that closely resembles the target speaker while improving duration modeling and overall expressiveness. Additionally, to improve zero-shot generation, we employed classifier free guidance, allowing the system to generate speech more near speech for unknown speakers. Using this approach, we trained language-specific speaker-conditioned models. Using the IndicSUPERB dataset for multiple Indian languages such as Bengali, Gujarati, Hindi, Marathi, Malayalam, Punjabi and Tamil. 

---
# GREAT: Guiding Query Generation with a Trie for Recommending Related Search about Video at Kuaishou 

**Authors**: Ninglu Shao, Jinshan Wang, Chenxu Wang, Qingbiao Li, Xiaoxue Zang, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.15267)  

**Abstract**: Currently, short video platforms have become the primary place for individuals to share experiences and obtain information. To better meet users' needs for acquiring information while browsing short videos, some apps have introduced a search entry at the bottom of videos, accompanied with recommended relevant queries. This scenario is known as query recommendation in video-related search, where core task is item-to-query (I2Q) recommendation. As this scenario has only emerged in recent years, there is a notable scarcity of academic research and publicly available datasets in this domain. To address this gap, we systematically examine the challenges associated with this scenario for the first time. Subsequently, we release a large-scale dataset derived from real-world data pertaining to the query recommendation in video-\textit{\textbf{r}}elated \textit{\textbf{s}}earch on the \textit{\textbf{Kuai}}shou app (\textbf{KuaiRS}). Presently, existing methods rely on embeddings to calculate similarity for matching short videos with queries, lacking deep interaction between the semantic content and the query. In this paper, we introduce a novel LLM-based framework named \textbf{GREAT}, which \textit{\textbf{g}}uides que\textit{\textbf{r}}y g\textit{\textbf{e}}ner\textit{\textbf{a}}tion with a \textit{\textbf{t}}rie to address I2Q recommendation in related search. Specifically, we initially gather high-quality queries with high exposure and click-through rate to construct a query-based trie. During training, we enhance the LLM's capability to generate high-quality queries using the query-based trie. In the inference phase, the query-based trie serves as a guide for the token generation. Finally, we further refine the relevance and literal quality between items and queries via a post-processing module. Extensive offline and online experiments demonstrate the effectiveness of our proposed method. 

---
# Exploiting Context-dependent Duration Features for Voice Anonymization Attack Systems 

**Authors**: Natalia Tomashenko, Emmanuel Vincent, Marc Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2507.15214)  

**Abstract**: The temporal dynamics of speech, encompassing variations in rhythm, intonation, and speaking rate, contain important and unique information about speaker identity. This paper proposes a new method for representing speaker characteristics by extracting context-dependent duration embeddings from speech temporal dynamics. We develop novel attack models using these representations and analyze the potential vulnerabilities in speaker verification and voice anonymization this http URL experimental results show that the developed attack models provide a significant improvement in speaker verification performance for both original and anonymized data in comparison with simpler representations of speech temporal dynamics reported in the literature. 

---
# Long-Short Distance Graph Neural Networks and Improved Curriculum Learning for Emotion Recognition in Conversation 

**Authors**: Xinran Li, Xiujuan Xu, Jiaqi Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.15205)  

**Abstract**: Emotion Recognition in Conversation (ERC) is a practical and challenging task. This paper proposes a novel multimodal approach, the Long-Short Distance Graph Neural Network (LSDGNN). Based on the Directed Acyclic Graph (DAG), it constructs a long-distance graph neural network and a short-distance graph neural network to obtain multimodal features of distant and nearby utterances, respectively. To ensure that long- and short-distance features are as distinct as possible in representation while enabling mutual influence between the two modules, we employ a Differential Regularizer and incorporate a BiAffine Module to facilitate feature interaction. In addition, we propose an Improved Curriculum Learning (ICL) to address the challenge of data imbalance. By computing the similarity between different emotions to emphasize the shifts in similar emotions, we design a "weighted emotional shift" metric and develop a difficulty measurer, enabling a training process that prioritizes learning easy samples before harder ones. Experimental results on the IEMOCAP and MELD datasets demonstrate that our model outperforms existing benchmarks. 

---
# Hear Your Code Fail, Voice-Assisted Debugging for Python 

**Authors**: Sayed Mahbub Hasan Amiri, Md. Mainul Islam, Mohammad Shakhawat Hossen, Sayed Majhab Hasan Amiri, Mohammad Shawkat Ali Mamun, Sk. Humaun Kabir, Naznin Akter  

**Link**: [PDF](https://arxiv.org/pdf/2507.15007)  

**Abstract**: This research introduces an innovative voice-assisted debugging plugin for Python that transforms silent runtime errors into actionable audible diagnostics. By implementing a global exception hook architecture with pyttsx3 text-to-speech conversion and Tkinter-based GUI visualization, the solution delivers multimodal error feedback through parallel auditory and visual channels. Empirical evaluation demonstrates 37% reduced cognitive load (p<0.01, n=50) compared to traditional stack-trace debugging, while enabling 78% faster error identification through vocalized exception classification and contextualization. The system achieves sub-1.2 second voice latency with under 18% CPU overhead during exception handling, vocalizing error types and consequences while displaying interactive tracebacks with documentation deep links. Criteria validate compatibility across Python 3.7+ environments on Windows, macOS, and Linux platforms. Needing only two lines of integration code, the plugin significantly boosts availability for aesthetically impaired designers and supports multitasking workflows through hands-free error medical diagnosis. Educational applications show particular promise, with pilot studies indicating 45% faster debugging skill acquisition among novice programmers. Future development will incorporate GPT-based repair suggestions and real-time multilingual translation to further advance auditory debugging paradigms. The solution represents a fundamental shift toward human-centric error diagnostics, bridging critical gaps in programming accessibility while establishing new standards for cognitive efficiency in software development workflows. 

---
# The Invisible Leash: Why RLVR May Not Escape Its Origin 

**Authors**: Fang Wu, Weihao Xuan, Ximing Lu, Zaid Harchaoui, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.14843)  

**Abstract**: Recent advances in large reasoning models highlight Reinforcement Learning with Verifiable Rewards (RLVR) as a promising method for enhancing AI's capabilities, particularly in solving complex logical tasks. However, it remains unclear whether RLVR truly expands a model's reasoning boundary or merely amplifies high-reward outputs that the base model already knows for improved precision. This study presents a theoretical and empirical investigation that provides fresh insights into the potential limits of RLVR. First, we offer a new theoretical perspective that RLVR is constrained by the base model's support-unable to sample solutions with zero initial probability-and operates as a conservative reweighting mechanism that may restrict the discovery of entirely original solutions. We also identify an entropy-reward tradeoff: while RLVR reliably enhances precision, it may progressively narrow exploration and potentially overlook correct yet underrepresented solutions. Extensive empirical experiments validate that while RLVR consistently improves pass@1, the shrinkage of empirical support generally outweighs the expansion of empirical support under larger sampling budgets, failing to recover correct answers that were previously accessible to the base model. Interestingly, we also observe that while RLVR sometimes increases token-level entropy, resulting in greater uncertainty at each generation step, answer-level entropy declines, indicating that these seemingly more uncertain paths ultimately converge onto a smaller set of distinct answers. Taken together, these findings reveal potential limits of RLVR in extending reasoning horizons. Breaking this invisible leash may require future algorithmic innovations such as explicit exploration mechanisms or hybrid strategies that seed probability mass into underrepresented solution regions. 

---
# GCC-Spam: Spam Detection via GAN, Contrastive Learning, and Character Similarity Networks 

**Authors**: Zixin Xu, Zhijie Wang, Zhiyuan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.14679)  

**Abstract**: The exponential growth of spam text on the Internet necessitates robust detection mechanisms to mitigate risks such as information leakage and social instability. This work addresses two principal challenges: adversarial strategies employed by spammers and the scarcity of labeled data. We propose a novel spam-text detection framework GCC-Spam, which integrates three core innovations. First, a character similarity network captures orthographic and phonetic features to counter character-obfuscation attacks and furthermore produces sentence embeddings for downstream classification. Second, contrastive learning enhances discriminability by optimizing the latent-space distance between spam and normal texts. Third, a Generative Adversarial Network (GAN) generates realistic pseudo-spam samples to alleviate data scarcity while improving model robustness and classification accuracy. Extensive experiments on real-world datasets demonstrate that our model outperforms baseline approaches, achieving higher detection rates with significantly fewer labeled examples. 

---
# Docopilot: Improving Multimodal Models for Document-Level Understanding 

**Authors**: Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang, Shenglong Ye, Botian Shi, Lewei Lu, Qibin Hou, Tong Lu, Hongsheng Li, Jifeng Dai, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14675)  

**Abstract**: Despite significant progress in multimodal large language models (MLLMs), their performance on complex, multi-page document comprehension remains inadequate, largely due to the lack of high-quality, document-level datasets. While current retrieval-augmented generation (RAG) methods offer partial solutions, they suffer from issues, such as fragmented retrieval contexts, multi-stage error accumulation, and extra time costs of retrieval. In this work, we present a high-quality document-level dataset, Doc-750K, designed to support in-depth understanding of multimodal documents. This dataset includes diverse document structures, extensive cross-page dependencies, and real question-answer pairs derived from the original documents. Building on the dataset, we develop a native multimodal model, Docopilot, which can accurately handle document-level dependencies without relying on RAG. Experiments demonstrate that Docopilot achieves superior coherence, accuracy, and efficiency in document understanding tasks and multi-turn interactions, setting a new baseline for document-level multimodal understanding. Data, code, and models are released at this https URL 

---
# When Autonomy Goes Rogue: Preparing for Risks of Multi-Agent Collusion in Social Systems 

**Authors**: Qibing Ren, Sitao Xie, Longxuan Wei, Zhenfei Yin, Junchi Yan, Lizhuang Ma, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2507.14660)  

**Abstract**: Recent large-scale events like election fraud and financial scams have shown how harmful coordinated efforts by human groups can be. With the rise of autonomous AI systems, there is growing concern that AI-driven groups could also cause similar harm. While most AI safety research focuses on individual AI systems, the risks posed by multi-agent systems (MAS) in complex real-world situations are still underexplored. In this paper, we introduce a proof-of-concept to simulate the risks of malicious MAS collusion, using a flexible framework that supports both centralized and decentralized coordination structures. We apply this framework to two high-risk fields: misinformation spread and e-commerce fraud. Our findings show that decentralized systems are more effective at carrying out malicious actions than centralized ones. The increased autonomy of decentralized systems allows them to adapt their strategies and cause more damage. Even when traditional interventions, like content flagging, are applied, decentralized groups can adjust their tactics to avoid detection. We present key insights into how these malicious groups operate and the need for better detection systems and countermeasures. Code is available at this https URL. 

---
# Optimizing Legal Document Retrieval in Vietnamese with Semi-Hard Negative Mining 

**Authors**: Van-Hoang Le, Duc-Vu Nguyen, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.14619)  

**Abstract**: Large Language Models (LLMs) face significant challenges in specialized domains like law, where precision and domain-specific knowledge are critical. This paper presents a streamlined two-stage framework consisting of Retrieval and Re-ranking to enhance legal document retrieval efficiency and accuracy. Our approach employs a fine-tuned Bi-Encoder for rapid candidate retrieval, followed by a Cross-Encoder for precise re-ranking, both optimized through strategic negative example mining. Key innovations include the introduction of the Exist@m metric to evaluate retrieval effectiveness and the use of semi-hard negatives to mitigate training bias, which significantly improved re-ranking performance. Evaluated on the SoICT Hackathon 2024 for Legal Document Retrieval, our team, 4Huiter, achieved a top-three position. While top-performing teams employed ensemble models and iterative self-training on large bge-m3 architectures, our lightweight, single-pass approach offered a competitive alternative with far fewer parameters. The framework demonstrates that optimized data processing, tailored loss functions, and balanced negative sampling are pivotal for building robust retrieval-augmented systems in legal contexts. 

---
# What do Large Language Models know about materials? 

**Authors**: Adrian Ehrenhofer, Thomas Wallmersperger, Gianaurelio Cuniberti  

**Link**: [PDF](https://arxiv.org/pdf/2507.14586)  

**Abstract**: Large Language Models (LLMs) are increasingly applied in the fields of mechanical engineering and materials science. As models that establish connections through the interface of language, LLMs can be applied for step-wise reasoning through the Processing-Structure-Property-Performance chain of material science and engineering. Current LLMs are built for adequately representing a dataset, which is the most part of the accessible internet. However, the internet mostly contains non-scientific content. If LLMs should be applied for engineering purposes, it is valuable to investigate models for their intrinsic knowledge -- here: the capacity to generate correct information about materials. In the current work, for the example of the Periodic Table of Elements, we highlight the role of vocabulary and tokenization for the uniqueness of material fingerprints, and the LLMs' capabilities of generating factually correct output of different state-of-the-art open models. This leads to a material knowledge benchmark for an informed choice, for which steps in the PSPP chain LLMs are applicable, and where specialized models are required. 

---
# Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion 

**Authors**: Yu Zhang, Baotong Tian, Zhiyao Duan  

**Link**: [PDF](https://arxiv.org/pdf/2507.14534)  

**Abstract**: Zero-shot online voice conversion (VC) holds significant promise for real-time communications and entertainment. However, current VC models struggle to preserve semantic fidelity under real-time constraints, deliver natural-sounding conversions, and adapt effectively to unseen speaker characteristics. To address these challenges, we introduce Conan, a chunkwise online zero-shot voice conversion model that preserves the content of the source while matching the voice timbre and styles of reference speech. Conan comprises three core components: 1) a Stream Content Extractor that leverages Emformer for low-latency streaming content encoding; 2) an Adaptive Style Encoder that extracts fine-grained stylistic features from reference speech for enhanced style adaptation; 3) a Causal Shuffle Vocoder that implements a fully causal HiFiGAN using a pixel-shuffle mechanism. Experimental evaluations demonstrate that Conan outperforms baseline models in subjective and objective metrics. Audio samples can be found at this https URL. 

---
# Efficient Whole Slide Pathology VQA via Token Compression 

**Authors**: Weimin Lyu, Qingqiao Hu, Kehan Qi, Zhan Shi, Wentao Huang, Saumya Gupta, Chao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.14497)  

**Abstract**: Whole-slide images (WSIs) in pathology can reach up to 10,000 x 10,000 pixels, posing significant challenges for multimodal large language model (MLLM) due to long context length and high computational demands. Previous methods typically focus on patch-level analysis or slide-level classification using CLIP-based models with multi-instance learning, but they lack the generative capabilities needed for visual question answering (VQA). More recent MLLM-based approaches address VQA by feeding thousands of patch tokens directly into the language model, which leads to excessive resource consumption. To address these limitations, we propose Token Compression Pathology LLaVA (TCP-LLaVA), the first MLLM architecture to perform WSI VQA via token compression. TCP-LLaVA introduces a set of trainable compression tokens that aggregate visual and textual information through a modality compression module, inspired by the [CLS] token mechanism in BERT. Only the compressed tokens are forwarded to the LLM for answer generation, significantly reducing input length and computational cost. Experiments on ten TCGA tumor subtypes show that TCP-LLaVA outperforms existing MLLM baselines in VQA accuracy while reducing training resource consumption by a substantial margin. 

---
# Routine: A Structural Planning Framework for LLM Agent System in Enterprise 

**Authors**: Guancheng Zeng, Xueyi Chen, Jiawang Hu, Shaohua Qi, Yaxuan Mao, Zhantao Wang, Yifan Nie, Shuang Li, Qiuyang Feng, Pengxu Qiu, Yujia Wang, Wenqiang Han, Linyan Huang, Gang Li, Jingjing Mo, Haowen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14447)  

**Abstract**: The deployment of agent systems in an enterprise environment is often hindered by several challenges: common models lack domain-specific process knowledge, leading to disorganized plans, missing key tools, and poor execution stability. To address this, this paper introduces Routine, a multi-step agent planning framework designed with a clear structure, explicit instructions, and seamless parameter passing to guide the agent's execution module in performing multi-step tool-calling tasks with high stability. In evaluations conducted within a real-world enterprise scenario, Routine significantly increases the execution accuracy in model tool calls, increasing the performance of GPT-4o from 41.1% to 96.3%, and Qwen3-14B from 32.6% to 83.3%. We further constructed a Routine-following training dataset and fine-tuned Qwen3-14B, resulting in an accuracy increase to 88.2% on scenario-specific evaluations, indicating improved adherence to execution plans. In addition, we employed Routine-based distillation to create a scenario-specific, multi-step tool-calling dataset. Fine-tuning on this distilled dataset raised the model's accuracy to 95.5%, approaching GPT-4o's performance. These results highlight Routine's effectiveness in distilling domain-specific tool-usage patterns and enhancing model adaptability to new scenarios. Our experimental results demonstrate that Routine provides a practical and accessible approach to building stable agent workflows, accelerating the deployment and adoption of agent systems in enterprise environments, and advancing the technical vision of AI for Process. 

---
# It's Not That Simple. An Analysis of Simple Test-Time Scaling 

**Authors**: Guojun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14419)  

**Abstract**: Prior work proposed simple test-time scaling, a method for replicating this scaling behavior with models distilled from o1-like models by manually controlling test-time compute: either scaling down by enforcing a maximum length or scaling up by iteratively appending "Wait" when the model is about to terminate its generation. This paper presents an analysis of simple test-time scaling and finds that the scaling behavior is largely attributed to scaling down by enforcing a maximum length. In contrast, fine-tuning on long CoT data distilled from o1-like models has no significant impact on scaling behavior, and scaling up by appending "Wait" leads to inconsistencies, as the model may oscillate between solutions. A key distinction exists between scaling down by enforcing a maximum length and scaling up test-time compute in o1-like models, such as DeepSeek-R1\@. These models are typically allowed to utilize as much compute as needed, with the only constraint being the model's maximum supported length. By learning to naturally scale up test-time compute during reinforcement learning, o1-like models surpass their peak performance when scaling up. In contrast, simple test-time scaling progressively imposes a lower upper limit on model performance as it scales down. While replicating the test-time scaling behavior of o1 models can be straightforward by scaling down, it is crucial to recognize that the goal of scaling test-time compute is to unlock higher performance -- beyond what the model could originally achieve -- rather than merely reproducing the appearance of scaling behavior. 

---
# Inverse Scaling in Test-Time Compute 

**Authors**: Aryo Pradipta Gema, Alexander Hägele, Runjin Chen, Andy Arditi, Jacob Goldman-Wetzler, Kit Fraser-Taliente, Henry Sleight, Linda Petrini, Julian Michael, Beatrice Alex, Pasquale Minervini, Yanda Chen, Joe Benton, Ethan Perez  

**Link**: [PDF](https://arxiv.org/pdf/2507.14417)  

**Abstract**: We construct evaluation tasks where extending the reasoning length of Large Reasoning Models (LRMs) deteriorates performance, exhibiting an inverse scaling relationship between test-time compute and accuracy. Our evaluation tasks span four categories: simple counting tasks with distractors, regression tasks with spurious features, deduction tasks with constraint tracking, and advanced AI risks. We identify five distinct failure modes when models reason for longer: 1) Claude models become increasingly distracted by irrelevant information; 2) OpenAI o-series models resist distractors but overfit to problem framings; 3) models shift from reasonable priors to spurious correlations; 4) all models show difficulties in maintaining focus on complex deductive tasks; and 5) extended reasoning may amplify concerning behaviors, with Claude Sonnet 4 showing increased expressions of self-preservation. These findings suggest that while test-time compute scaling remains promising for improving model capabilities, it may inadvertently reinforce problematic reasoning patterns. Our results demonstrate the importance of evaluating models across diverse reasoning lengths to identify and address these failure modes in LRMs. 

---
# Assessing the Reliability of Large Language Models for Deductive Qualitative Coding: A Comparative Study of ChatGPT Interventions 

**Authors**: Angjelin Hila, Elliott Hauser  

**Link**: [PDF](https://arxiv.org/pdf/2507.14384)  

**Abstract**: In this study, we investigate the use of large language models (LLMs), specifically ChatGPT, for structured deductive qualitative coding. While most current research emphasizes inductive coding applications, we address the underexplored potential of LLMs to perform deductive classification tasks aligned with established human-coded schemes. Using the Comparative Agendas Project (CAP) Master Codebook, we classified U.S. Supreme Court case summaries into 21 major policy domains. We tested four intervention methods: zero-shot, few-shot, definition-based, and a novel Step-by-Step Task Decomposition strategy, across repeated samples. Performance was evaluated using standard classification metrics (accuracy, F1-score, Cohen's kappa, Krippendorff's alpha), and construct validity was assessed using chi-squared tests and Cramer's V. Chi-squared and effect size analyses confirmed that intervention strategies significantly influenced classification behavior, with Cramer's V values ranging from 0.359 to 0.613, indicating moderate to strong shifts in classification patterns. The Step-by-Step Task Decomposition strategy achieved the strongest reliability (accuracy = 0.775, kappa = 0.744, alpha = 0.746), achieving thresholds for substantial agreement. Despite the semantic ambiguity within case summaries, ChatGPT displayed stable agreement across samples, including high F1 scores in low-support subclasses. These findings demonstrate that with targeted, custom-tailored interventions, LLMs can achieve reliability levels suitable for integration into rigorous qualitative coding workflows. 

---
# Solo Connection: A Parameter Efficient Fine-Tuning Technique for Transformers 

**Authors**: Harsh Nilesh Pathak, Randy Paffenroth  

**Link**: [PDF](https://arxiv.org/pdf/2507.14353)  

**Abstract**: Parameter efficient fine tuning (PEFT) is a versatile and extensible approach for adapting a Large Language Model (LLM) for newer tasks. One of the most prominent PEFT approaches, Low Rank Adaptation (LoRA), primarily focuses on adjusting the attention weight matrices within individual decoder blocks of a Generative Pre trained Transformer (GPT2). In contrast, we introduce Solo Connection a novel method that adapts the representation at the decoder-block level rather than modifying individual weight matrices. Not only does Solo Connection outperform LoRA on E2E natural language generation benchmarks, but it also reduces the number of trainable parameters by 59% relative to LoRA and by more than 99% compared to full fine-tuning of GPT2, an early version of Large Language Models (LLMs). Solo Connection is also motivated by homotopy theory: we introduce a trainable linear transformation that gradually interpolates between a zero vector and the task-specific representation, enabling smooth and stable adaptation over time. While skip connections in the original 12 layer GPT2 are typically confined to individual decoder blocks, subsequent GPT2 variants scale up to 48 layers, and even larger language models can include 128 or more decoder blocks. These expanded architectures underscore the need to revisit how skip connections are employed during fine-tuning. This paper focuses on long skip connections that link outputs of different decoder blocks, potentially enhancing the model's ability to adapt to new tasks while leveraging pre-trained knowledge. 

---
# WebGuard: Building a Generalizable Guardrail for Web Agents 

**Authors**: Boyuan Zheng, Zeyi Liao, Scott Salisbury, Zeyuan Liu, Michael Lin, Qinyuan Zheng, Zifan Wang, Xiang Deng, Dawn Song, Huan Sun, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2507.14293)  

**Abstract**: The rapid development of autonomous web agents powered by Large Language Models (LLMs), while greatly elevating efficiency, exposes the frontier risk of taking unintended or harmful actions. This situation underscores an urgent need for effective safety measures, akin to access controls for human users. To address this critical challenge, we introduce WebGuard, the first comprehensive dataset designed to support the assessment of web agent action risks and facilitate the development of guardrails for real-world online environments. In doing so, WebGuard specifically focuses on predicting the outcome of state-changing actions and contains 4,939 human-annotated actions from 193 websites across 22 diverse domains, including often-overlooked long-tail websites. These actions are categorized using a novel three-tier risk schema: SAFE, LOW, and HIGH. The dataset includes designated training and test splits to support evaluation under diverse generalization settings. Our initial evaluations reveal a concerning deficiency: even frontier LLMs achieve less than 60% accuracy in predicting action outcomes and less than 60% recall in lagging HIGH-risk actions, highlighting the risks of deploying current-generation agents without dedicated safeguards. We therefore investigate fine-tuning specialized guardrail models using WebGuard. We conduct comprehensive evaluations across multiple generalization settings and find that a fine-tuned Qwen2.5VL-7B model yields a substantial improvement in performance, boosting accuracy from 37% to 80% and HIGH-risk action recall from 20% to 76%. Despite these improvements, the performance still falls short of the reliability required for high-stakes deployment, where guardrails must approach near-perfect accuracy and recall. 

---
# Identifying Algorithmic and Domain-Specific Bias in Parliamentary Debate Summarisation 

**Authors**: Eoghan Cunningham, James Cross, Derek Greene  

**Link**: [PDF](https://arxiv.org/pdf/2507.14221)  

**Abstract**: The automated summarisation of parliamentary debates using large language models (LLMs) offers a promising way to make complex legislative discourse more accessible to the public. However, such summaries must not only be accurate and concise but also equitably represent the views and contributions of all speakers. This paper explores the use of LLMs to summarise plenary debates from the European Parliament and investigates the algorithmic and representational biases that emerge in this context. We propose a structured, multi-stage summarisation framework that improves textual coherence and content fidelity, while enabling the systematic analysis of how speaker attributes -- such as speaking order or political affiliation -- influence the visibility and accuracy of their contributions in the final summaries. Through our experiments using both proprietary and open-weight LLMs, we find evidence of consistent positional and partisan biases, with certain speakers systematically under-represented or misattributed. Our analysis shows that these biases vary by model and summarisation strategy, with hierarchical approaches offering the greatest potential to reduce disparity. These findings underscore the need for domain-sensitive evaluation metrics and ethical oversight in the deployment of LLMs for democratic applications. 

---
# LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models 

**Authors**: Dachuan Shi, Yonggan Fu, Xiangchi Yuan, Zhongzhi Yu, Haoran You, Sixu Li, Xin Dong, Jan Kautz, Pavlo Molchanov, Yingyan  

**Link**: [PDF](https://arxiv.org/pdf/2507.14204)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have spurred interest in numerous applications requiring robust long-range capabilities, essential for processing extensive input contexts and continuously generating extended outputs. As sequence lengths increase, the number of Key-Value (KV) pairs in LLMs escalates, creating a significant efficiency bottleneck. In this paper, we propose a new KV cache optimization paradigm called LaCache, a training-free method for efficient and accurate generative inference of LLMs. LaCache enables LLMs to simultaneously address both of the critical challenges in long-range modeling: robust long-range capabilities and continuous generation without running out-of-memory (OOM). Specifically, LaCache integrates two key innovations: (1) a ladder-shaped KV cache pattern that stores KV pairs not only sequentially (left-to-right within each layer) but also across layers (from shallow to deep), providing an extended span for capturing long-range dependencies under a fixed storage budget, thereby boosting long-range capabilities; and (2) an iterative compaction mechanism that progressively compresses older caches, freeing up space for new tokens within a fixed cache size. This token distance-based dynamic compression enables more effective continuous generation under constrained cache budgets. Experiments across various tasks, benchmarks, and LLM models consistently validate LaCache's effectiveness in enhancing LLMs' long-range capabilities. Our code is available at this https URL. 

---
# ExCyTIn-Bench: Evaluating LLM agents on Cyber Threat Investigation 

**Authors**: Yiran Wu, Mauricio Velazco, Andrew Zhao, Manuel Raúl Meléndez Luján, Srisuma Movva, Yogesh K Roy, Quang Nguyen, Roberto Rodriguez, Qingyun Wu, Michael Albada, Julia Kiseleva, Anand Mudgerikar  

**Link**: [PDF](https://arxiv.org/pdf/2507.14201)  

**Abstract**: We present ExCyTIn-Bench, the first benchmark to Evaluate an LLM agent x on the task of Cyber Threat Investigation through security questions derived from investigation graphs. Real-world security analysts must sift through a large number of heterogeneous alert signals and security logs, follow multi-hop chains of evidence, and compile an incident report. With the developments of LLMs, building LLM-based agents for automatic thread investigation is a promising direction. To assist the development and evaluation of LLM agents, we construct a dataset from a controlled Azure tenant that covers 8 simulated real-world multi-step attacks, 57 log tables from Microsoft Sentinel and related services, and 589 automatically generated questions. We leverage security logs extracted with expert-crafted detection logic to build threat investigation graphs, and then generate questions with LLMs using paired nodes on the graph, taking the start node as background context and the end node as answer. Anchoring each question to these explicit nodes and edges not only provides automatic, explainable ground truth answers but also makes the pipeline reusable and readily extensible to new logs. This also enables the automatic generation of procedural tasks with verifiable rewards, which can be naturally extended to training agents via reinforcement learning. Our comprehensive experiments with different models confirm the difficulty of the task: with the base setting, the average reward across all evaluated models is 0.249, and the best achieved is 0.368, leaving substantial headroom for future research. Code and data are coming soon! 

---
# A Sparsity Predicting Approach for Large Language Models via Activation Pattern Clustering 

**Authors**: Nobel Dhar, Bobin Deng, Md Romyull Islam, Xinyue Zhang, Kazi Fahim Ahmad Nasif, Kun Suo  

**Link**: [PDF](https://arxiv.org/pdf/2507.14179)  

**Abstract**: Large Language Models (LLMs) exhibit significant activation sparsity, where only a subset of neurons are active for a given input. Although this sparsity presents opportunities to reduce computational cost, efficiently utilizing it requires predicting activation patterns in a scalable manner. However, direct prediction at the neuron level is computationally expensive due to the vast number of neurons in modern LLMs. To enable efficient prediction and utilization of activation sparsity, we propose a clustering-based activation pattern compression framework. Instead of treating each neuron independently, we group similar activation patterns into a small set of representative clusters. Our method achieves up to 79.34% clustering precision, outperforming standard binary clustering approaches while maintaining minimal degradation in perplexity (PPL) scores. With a sufficiently large number of clusters, our approach attains a PPL score as low as 12.49, demonstrating its effectiveness in preserving model quality while reducing computational overhead. By predicting cluster assignments rather than individual neuron states, future models can efficiently infer activation patterns from pre-computed centroids. We detail the clustering algorithm, analyze its effectiveness in capturing meaningful activation structures, and demonstrate its potential to improve sparse computation efficiency. This clustering-based formulation serves as a foundation for future work on activation pattern prediction, paving the way for efficient inference in large-scale language models. 

---
