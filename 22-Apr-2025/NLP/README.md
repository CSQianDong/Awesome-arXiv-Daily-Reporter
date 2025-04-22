# Evaluating Judges as Evaluators: The JETTS Benchmark of LLM-as-Judges as Test-Time Scaling Evaluators 

**Authors**: Yilun Zhou, Austin Xu, Peifeng Wang, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2504.15253)  

**Abstract**: Scaling test-time computation, or affording a generator large language model (LLM) extra compute during inference, typically employs the help of external non-generative evaluators (i.e., reward models). Concurrently, LLM-judges, models trained to generate evaluations and critiques (explanations) in natural language, are becoming increasingly popular in automatic evaluation. Despite judge empirical successes, their effectiveness as evaluators in test-time scaling settings is largely unknown. In this paper, we introduce the Judge Evaluation for Test-Time Scaling (JETTS) benchmark, which evaluates judge performance in three domains (math reasoning, code generation, and instruction following) under three task settings: response reranking, step-level beam search, and critique-based response refinement. We evaluate 10 different judge models (7B-70B parameters) for 8 different base generator models (6.7B-72B parameters). Our benchmark shows that while judges are competitive with outcome reward models in reranking, they are consistently worse than process reward models in beam search procedures. Furthermore, though unique to LLM-judges, their natural language critiques are currently ineffective in guiding the generator towards better responses. 

---
# MR. Guard: Multilingual Reasoning Guardrail using Curriculum Learning 

**Authors**: Yahan Yang, Soham Dan, Shuo Li, Dan Roth, Insup Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.15241)  

**Abstract**: Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual setting, where multilingual safety-aligned data are often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we propose an approach to build a multilingual guardrail with reasoning. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-guided Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail consistently outperforms recent baselines across both in-domain and out-of-domain languages. The multilingual reasoning capability of our guardrail enables it to generate multilingual explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation. 

---
# Values in the Wild: Discovering and Analyzing Values in Real-World Language Model Interactions 

**Authors**: Saffron Huang, Esin Durmus, Miles McCain, Kunal Handa, Alex Tamkin, Jerry Hong, Michael Stern, Arushi Somani, Xiuruo Zhang, Deep Ganguli  

**Link**: [PDF](https://arxiv.org/pdf/2504.15236)  

**Abstract**: AI assistants can impart value judgments that shape people's decisions and worldviews, yet little is known empirically about what values these systems rely on in practice. To address this, we develop a bottom-up, privacy-preserving method to extract the values (normative considerations stated or demonstrated in model responses) that Claude 3 and 3.5 models exhibit in hundreds of thousands of real-world interactions. We empirically discover and taxonomize 3,307 AI values and study how they vary by context. We find that Claude expresses many practical and epistemic values, and typically supports prosocial human values while resisting values like "moral nihilism". While some values appear consistently across contexts (e.g. "transparency"), many are more specialized and context-dependent, reflecting the diversity of human interlocutors and their varied contexts. For example, "harm prevention" emerges when Claude resists users, "historical accuracy" when responding to queries about controversial events, "healthy boundaries" when asked for relationship advice, and "human agency" in technology ethics discussions. By providing the first large-scale empirical mapping of AI values in deployment, our work creates a foundation for more grounded evaluation and design of values in AI systems. 

---
# Fully Bayesian Approaches to Topics over Time 

**Authors**: Julián Cendrero, Julio Gonzalo, Ivar Zapata  

**Link**: [PDF](https://arxiv.org/pdf/2504.15220)  

**Abstract**: The Topics over Time (ToT) model captures thematic changes in timestamped datasets by explicitly modeling publication dates jointly with word co-occurrence patterns. However, ToT was not approached in a fully Bayesian fashion, a flaw that makes it susceptible to stability problems. To address this issue, we propose a fully Bayesian Topics over Time (BToT) model via the introduction of a conjugate prior to the Beta distribution. This prior acts as a regularization that prevents the online version of the algorithm from unstable updates when a topic is poorly represented in a mini-batch. The characteristics of this prior to the Beta distribution are studied here for the first time. Still, this model suffers from a difference in scale between the single-time observations and the multiplicity of words per document. A variation of BToT, Weighted Bayesian Topics over Time (WBToT), is proposed as a solution. In WBToT, publication dates are repeated a certain number of times per document, which balances the relative influence of words and timestamps along the inference process. We have tested our models on two datasets: a collection of over 200 years of US state-of-the-union (SOTU) addresses and a large-scale COVID-19 Twitter corpus of 10 million tweets. The results show that WBToT captures events better than Latent Dirichlet Allocation and other SOTA topic models like BERTopic: the median absolute deviation of the topic presence over time is reduced by $51\%$ and $34\%$, respectively. Our experiments also demonstrate the superior coherence of WBToT over BToT, which highlights the importance of balancing the time and word modalities. Finally, we illustrate the stability of the online optimization algorithm in WBToT, which allows the application of WBToT to problems that are intractable for standard ToT. 

---
# EvalAgent: Discovering Implicit Evaluation Criteria from the Web 

**Authors**: Manya Wadhwa, Zayne Sprague, Chaitanya Malaviya, Philippe Laban, Junyi Jessy Li, Greg Durrett  

**Link**: [PDF](https://arxiv.org/pdf/2504.15219)  

**Abstract**: Evaluation of language model outputs on structured writing tasks is typically conducted with a number of desirable criteria presented to human evaluators or large language models (LLMs). For instance, on a prompt like "Help me draft an academic talk on coffee intake vs research productivity", a model response may be evaluated for criteria like accuracy and coherence. However, high-quality responses should do more than just satisfy basic task requirements. An effective response to this query should include quintessential features of an academic talk, such as a compelling opening, clear research questions, and a takeaway. To help identify these implicit criteria, we introduce EvalAgent, a novel framework designed to automatically uncover nuanced and task-specific criteria. EvalAgent first mines expert-authored online guidance. It then uses this evidence to propose diverse, long-tail evaluation criteria that are grounded in reliable external sources. Our experiments demonstrate that the grounded criteria produced by EvalAgent are often implicit (not directly stated in the user's prompt), yet specific (high degree of lexical precision). Further, EvalAgent criteria are often not satisfied by initial responses but they are actionable, such that responses can be refined to satisfy them. Finally, we show that combining LLM-generated and EvalAgent criteria uncovers more human-valued criteria than using LLMs alone. 

---
# Support Evaluation for the TREC 2024 RAG Track: Comparing Human versus LLM Judges 

**Authors**: Nandan Thakur, Ronak Pradeep, Shivani Upadhyay, Daniel Campos, Nick Craswell, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15205)  

**Abstract**: Retrieval-augmented generation (RAG) enables large language models (LLMs) to generate answers with citations from source documents containing "ground truth", thereby reducing system hallucinations. A crucial factor in RAG evaluation is "support", whether the information in the cited documents supports the answer. To this end, we conducted a large-scale comparative study of 45 participant submissions on 36 topics to the TREC 2024 RAG Track, comparing an automatic LLM judge (GPT-4o) against human judges for support assessment. We considered two conditions: (1) fully manual assessments from scratch and (2) manual assessments with post-editing of LLM predictions. Our results indicate that for 56% of the manual from-scratch assessments, human and GPT-4o predictions match perfectly (on a three-level scale), increasing to 72% in the manual with post-editing condition. Furthermore, by carefully analyzing the disagreements in an unbiased study, we found that an independent human judge correlates better with GPT-4o than a human judge, suggesting that LLM judges can be a reliable alternative for support assessment. To conclude, we provide a qualitative analysis of human and GPT-4o errors to help guide future iterations of support assessment. 

---
# On true empty category 

**Authors**: Qilin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2504.15168)  

**Abstract**: According to Chomsky (1981, 1986), empty categories consist of PRO, pro, trace, and variable. However, some empty object positions seem to be incompatible with extant empty categories. Given this, Li (2007a, 2007b, 2014) and Li & Wei (2014) raise the true empty category hypothesis, which holds that true empty category is only an empty position with category and Case features. As a last resort option, it is used mainly to meet the subcatgorization of a verb. This assumption is ingenious, and if proved to be true, it will exert a great impact on the study of UG. In this paper, we evaluate their evidence from topicalization and demonstrate that it can be accounted for without invoking true empty category. 

---
# The Synthetic Imputation Approach: Generating Optimal Synthetic Texts For Underrepresented Categories In Supervised Classification Tasks 

**Authors**: Joan C. Timoneda  

**Link**: [PDF](https://arxiv.org/pdf/2504.15160)  

**Abstract**: Encoder-decoder Large Language Models (LLMs), such as BERT and RoBERTa, require that all categories in an annotation task be sufficiently represented in the training data for optimal performance. However, it is often difficult to find sufficient examples for all categories in a task when building a high-quality training set. In this article, I describe this problem and propose a solution, the synthetic imputation approach. Leveraging a generative LLM (GPT-4o), this approach generates synthetic texts based on careful prompting and five original examples drawn randomly with replacement from the sample. This approach ensures that new synthetic texts are sufficiently different from the original texts to reduce overfitting, but retain the underlying substantive meaning of the examples to maximize out-of-sample performance. With 75 original examples or more, synthetic imputation's performance is on par with a full sample of original texts, and overfitting remains low, predictable and correctable with 50 original samples. The synthetic imputation approach provides a novel role for generative LLMs in research and allows applied researchers to balance their datasets for best performance. 

---
# EasyEdit2: An Easy-to-use Steering Framework for Editing Large Language Models 

**Authors**: Ziwen Xu, Shuxun Wang, Kewei Xu, Haoming Xu, Mengru Wang, Xinle Deng, Yunzhi Yao, Guozhou Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15133)  

**Abstract**: In this paper, we introduce EasyEdit2, a framework designed to enable plug-and-play adjustability for controlling Large Language Model (LLM) behaviors. EasyEdit2 supports a wide range of test-time interventions, including safety, sentiment, personality, reasoning patterns, factuality, and language features. Unlike its predecessor, EasyEdit2 features a new architecture specifically designed for seamless model steering. It comprises key modules such as the steering vector generator and the steering vector applier, which enable automatic generation and application of steering vectors to influence the model's behavior without modifying its parameters. One of the main advantages of EasyEdit2 is its ease of use-users do not need extensive technical knowledge. With just a single example, they can effectively guide and adjust the model's responses, making precise control both accessible and efficient. Empirically, we report model steering performance across different LLMs, demonstrating the effectiveness of these techniques. We have released the source code on GitHub at this https URL along with a demonstration notebook. In addition, we provide a demo video at this https URL for a quick introduction. 

---
# Kuwain 1.5B: An Arabic SLM via Language Injection 

**Authors**: Khalil Hennara, Sara Chrouf, Mohamed Motaism Hamed, Zeina Aldallal, Omar Hadid, Safwan AlModhayan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15120)  

**Abstract**: Enhancing existing models with new knowledge is a crucial aspect of AI development. This paper introduces a novel method for integrating a new language into a large language model (LLM). Our approach successfully incorporates a previously unseen target language into an existing LLM without compromising its prior knowledge. We trained a tiny model with 1.5 billion parameters named Kuwain by injecting the Arabic language into a small open-source model mainly trained in English. Our method demonstrates significant improvements in Arabic language performance, with an average 8% improvement across various benchmarks, while retaining the model's existing knowledge with a minimum amount of the original model's data. This offers a cost-effective alternative to training a comprehensive model in both English and Arabic. The results highlight the potential for efficient, targeted language model expansion without extensive retraining or resource-intensive processes. 

---
# Rethinking the Potential of Multimodality in Collaborative Problem Solving Diagnosis with Large Language Models 

**Authors**: K. Wong, B. Wu, S. Bulathwela, M. Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2504.15093)  

**Abstract**: Detecting collaborative and problem-solving behaviours from digital traces to interpret students' collaborative problem solving (CPS) competency is a long-term goal in the Artificial Intelligence in Education (AIEd) field. Although multimodal data and advanced models are argued to have the potential to detect complex CPS behaviours, empirical evidence on their value remains limited with some contrasting evidence. In this study, we investigated the potential of multimodal data to improve model performance in diagnosing 78 secondary school students' CPS subskills and indicators in authentic educational settings. In particular, text embeddings from verbal data and acoustic embeddings from audio data were used in a multimodal classification model for CPS diagnosis. Both unimodal and multimodal transformer-based models outperformed traditional models in detecting CPS classes. Although the inclusion of multimodality did not improve the performance of traditional unimodal models, its integration into transformer-based models demonstrated improved performance for diagnosing social-cognitive CPS classes compared to unimodal transformer-based models. Based on the results, the paper argues that multimodality and the selection of a particular modelling technique should not be taken for granted to achieve the best performance in the automated detection of every CPS subskill and indicator. Rather, their value is limited to certain types of CPS indicators, affected by the complexity of the labels, and dependent on the composition of indicators in the dataset. We conclude the paper by discussing the required nuance when considering the value of LLMs and multimodality in automated CPS diagnosis, highlighting the need for human-AI complementarity, and proposing the exploration of relevant model architectures and techniques to improve CPS diagnosis in authentic educational contexts. 

---
# Testing LLMs' Capabilities in Annotating Translations Based on an Error Typology Designed for LSP Translation: First Experiments with ChatGPT 

**Authors**: Joachim Minder, Guillaume Wisniewski, Natalie Kübler  

**Link**: [PDF](https://arxiv.org/pdf/2504.15052)  

**Abstract**: This study investigates the capabilities of large language models (LLMs), specifically ChatGPT, in annotating MT outputs based on an error typology. In contrast to previous work focusing mainly on general language, we explore ChatGPT's ability to identify and categorise errors in specialised translations. By testing two different prompts and based on a customised error typology, we compare ChatGPT annotations with human expert evaluations of translations produced by DeepL and ChatGPT itself. The results show that, for translations generated by DeepL, recall and precision are quite high. However, the degree of accuracy in error categorisation depends on the prompt's specific features and its level of detail, ChatGPT performing very well with a detailed prompt. When evaluating its own translations, ChatGPT achieves significantly poorer results, revealing limitations with self-assessment. These results highlight both the potential and the limitations of LLMs for translation evaluation, particularly in specialised domains. Our experiments pave the way for future research on open-source LLMs, which could produce annotations of comparable or even higher quality. In the future, we also aim to test the practical effectiveness of this automated evaluation in the context of translation training, particularly by optimising the process of human evaluation by teachers and by exploring the impact of annotations by LLMs on students' post-editing and translation learning. 

---
# RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search 

**Authors**: Quy-Anh Dang, Chris Ngo, Truong-Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2504.15047)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but are susceptible to adversarial prompts that exploit vulnerabilities to produce unsafe or biased outputs. Existing red-teaming methods often face scalability challenges, resource-intensive requirements, or limited diversity in attack strategies. We propose RainbowPlus, a novel red-teaming framework rooted in evolutionary computation, enhancing adversarial prompt generation through an adaptive quality-diversity (QD) search that extends classical evolutionary algorithms like MAP-Elites with innovations tailored for language models. By employing a multi-element archive to store diverse high-quality prompts and a comprehensive fitness function to evaluate multiple prompts concurrently, RainbowPlus overcomes the constraints of single-prompt archives and pairwise comparisons in prior QD methods like Rainbow Teaming. Experiments comparing RainbowPlus to QD methods across six benchmark datasets and four open-source LLMs demonstrate superior attack success rate (ASR) and diversity (Diverse-Score $\approx 0.84$), generating up to 100 times more unique prompts (e.g., 10,418 vs. 100 for Ministral-8B-Instruct-2410). Against nine state-of-the-art methods on the HarmBench dataset with twelve LLMs (ten open-source, two closed-source), RainbowPlus achieves an average ASR of 81.1%, surpassing AutoDAN-Turbo by 3.9%, and is 9 times faster (1.45 vs. 13.50 hours). Our open-source implementation fosters further advancements in LLM safety, offering a scalable tool for vulnerability assessment. Code and resources are publicly available at this https URL, supporting reproducibility and future research in LLM red-teaming. 

---
# DistilQwen2.5: Industrial Practices of Training Distilled Open Lightweight Language Models 

**Authors**: Chengyu Wang, Junbing Yan, Yuanhao Yue, Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15027)  

**Abstract**: Enhancing computational efficiency and reducing deployment costs for large language models (LLMs) have become critical challenges in various resource-constrained scenarios. In this work, we present DistilQwen2.5, a family of distilled, lightweight LLMs derived from the public Qwen2.5 models. These distilled models exhibit enhanced instruction-following capabilities compared to the original models based on a series of distillation techniques that incorporate knowledge from much larger LLMs. In our industrial practice, we first leverage powerful proprietary LLMs with varying capacities as multi-agent teachers to select, rewrite, and refine instruction-response pairs that are more suitable for student LLMs to learn. After standard fine-tuning, we further leverage a computationally efficient model fusion approach that enables student models to progressively integrate fine-grained hidden knowledge from their teachers. Experimental evaluations demonstrate that the distilled models possess significantly stronger capabilities than their original checkpoints. Additionally, we present use cases to illustrate the applications of our framework in real-world scenarios. To facilitate practical use, we have released all the DistilQwen2.5 models to the open-source community. 

---
# LLMs as Data Annotators: How Close Are We to Human Performance 

**Authors**: Muhammad Uzair Ul Haq, Davide Rigoni, Alessandro Sperduti  

**Link**: [PDF](https://arxiv.org/pdf/2504.15022)  

**Abstract**: In NLP, fine-tuning LLMs is effective for various applications but requires high-quality annotated data. However, manual annotation of data is labor-intensive, time-consuming, and costly. Therefore, LLMs are increasingly used to automate the process, often employing in-context learning (ICL) in which some examples related to the task are given in the prompt for better performance. However, manually selecting context examples can lead to inefficiencies and suboptimal model performance. This paper presents comprehensive experiments comparing several LLMs, considering different embedding models, across various datasets for the Named Entity Recognition (NER) task. The evaluation encompasses models with approximately $7$B and $70$B parameters, including both proprietary and non-proprietary models. Furthermore, leveraging the success of Retrieval-Augmented Generation (RAG), it also considers a method that addresses the limitations of ICL by automatically retrieving contextual examples, thereby enhancing performance. The results highlight the importance of selecting the appropriate LLM and embedding model, understanding the trade-offs between LLM sizes and desired performance, and the necessity to direct research efforts towards more challenging datasets. 

---
# Stay Hungry, Stay Foolish: On the Extended Reading Articles Generation with LLMs 

**Authors**: Yow-Fu Liou, Yu-Chien Tang, An-Zi Yen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15013)  

**Abstract**: The process of creating educational materials is both time-consuming and demanding for educators. This research explores the potential of Large Language Models (LLMs) to streamline this task by automating the generation of extended reading materials and relevant course suggestions. Using the TED-Ed Dig Deeper sections as an initial exploration, we investigate how supplementary articles can be enriched with contextual knowledge and connected to additional learning resources. Our method begins by generating extended articles from video transcripts, leveraging LLMs to include historical insights, cultural examples, and illustrative anecdotes. A recommendation system employing semantic similarity ranking identifies related courses, followed by an LLM-based refinement process to enhance relevance. The final articles are tailored to seamlessly integrate these recommendations, ensuring they remain cohesive and informative. Experimental evaluations demonstrate that our model produces high-quality content and accurate course suggestions, assessed through metrics such as Hit Rate, semantic similarity, and coherence. Our experimental analysis highlight the nuanced differences between the generated and existing materials, underscoring the model's capacity to offer more engaging and accessible learning experiences. This study showcases how LLMs can bridge the gap between core content and supplementary learning, providing students with additional recommended resources while also assisting teachers in designing educational materials. 

---
# Efficient Pretraining Length Scaling 

**Authors**: Bohong Wu, Shen Yan, Sijun Zhang, Jianqiao Lu, Yutao Zeng, Ya Wang, Xun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.14992)  

**Abstract**: Recent advances in large language models have demonstrated the effectiveness of length scaling during post-training, yet its potential in pre-training remains underexplored. We present the Parallel Hidden Decoding Transformer (\textit{PHD}-Transformer), a novel framework that enables efficient length scaling during pre-training while maintaining inference efficiency. \textit{PHD}-Transformer achieves this through an innovative KV cache management strategy that distinguishes between original tokens and hidden decoding tokens. By retaining only the KV cache of original tokens for long-range dependencies while immediately discarding hidden decoding tokens after use, our approach maintains the same KV cache size as the vanilla transformer while enabling effective length scaling. To further enhance performance, we introduce two optimized variants: \textit{PHD-SWA} employs sliding window attention to preserve local dependencies, while \textit{PHD-CSWA} implements chunk-wise sliding window attention to eliminate linear growth in pre-filling time. Extensive experiments demonstrate consistent improvements across multiple benchmarks. 

---
# Evaluating LLMs on Chinese Topic Constructions: A Research Proposal Inspired by Tian et al. (2024) 

**Authors**: Xiaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14969)  

**Abstract**: This paper proposes a framework for evaluating large language models (LLMs) on Chinese topic constructions, focusing on their sensitivity to island constraints. Drawing inspiration from Tian et al. (2024), we outline an experimental design for testing LLMs' grammatical knowledge of Mandarin syntax. While no experiments have been conducted yet, this proposal aims to provide a foundation for future studies and invites feedback on the methodology. 

---
# Speaker Fuzzy Fingerprints: Benchmarking Text-Based Identification in Multiparty Dialogues 

**Authors**: Rui Ribeiro, Luísa Coheur, Joao P. Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2504.14963)  

**Abstract**: Speaker identification using voice recordings leverages unique acoustic features, but this approach fails when only textual data is available. Few approaches have attempted to tackle the problem of identifying speakers solely from text, and the existing ones have primarily relied on traditional methods. In this work, we explore the use of fuzzy fingerprints from large pre-trained models to improve text-based speaker identification. We integrate speaker-specific tokens and context-aware modeling, demonstrating that conversational context significantly boosts accuracy, reaching 70.6% on the Friends dataset and 67.7% on the Big Bang Theory dataset. Additionally, we show that fuzzy fingerprints can approximate full fine-tuning performance with fewer hidden units, offering improved interpretability. Finally, we analyze ambiguous utterances and propose a mechanism to detect speaker-agnostic lines. Our findings highlight key challenges and provide insights for future improvements in text-based speaker identification. 

---
# CRAVE: A Conflicting Reasoning Approach for Explainable Claim Verification Using LLMs 

**Authors**: Yingming Zheng, Xiaoliang Liu, Peng Wu, Li Pan  

**Link**: [PDF](https://arxiv.org/pdf/2504.14905)  

**Abstract**: The rapid spread of misinformation, driven by digital media and AI-generated content, has made automatic claim verification essential. Traditional methods, which depend on expert-annotated evidence, are labor-intensive and not scalable. Although recent automated systems have improved, they still struggle with complex claims that require nuanced reasoning. To address this, we propose CRAVE, a Conflicting Reasoning Approach for explainable claim VErification, that verify the complex claims based on the conflicting rationales reasoned by large language models (LLMs). Specifically, CRAVE introduces a three-module framework. Ambiguity Elimination enchanced Evidence Retrieval module performs ambiguity elimination and entity-based search to gather relevant evidence related to claim verification from external sources like Wikipedia. Conflicting Perspective Reasoning and Preliminary Judgment module with LLMs adopts LLMs to reason rationales with conflicting stances about claim verification from retrieved evidence across four dimensions, i.e., direct evidence, semantic relationships, linguistic patterns, and logical reasoning and make a preliminary judgment. Finally, Small Language Model (SLM) based Judge module is fine-tuned to make use of preliminary judgment from LLMs to assess the confidence of the conflicting rationales and make a final authenticity judgment. This methodology allows CRAVE to capture subtle inconsistencies in complex claims, improving both the accuracy and transparency of claim verification. Extensive experiments on two public claim verification datasets demonstrate that our CRAVE model achieves much better performance than state-of-the-art methods and exhibits a superior capacity for finding relevant evidence and explaining the model predictions. The code is provided at this https URL. 

---
# Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey 

**Authors**: Aoran Gan, Hao Yu, Kai Zhang, Qi Liu, Wenyu Yan, Zhenya Huang, Shiwei Tong, Guoping Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14891)  

**Abstract**: Recent advancements in Retrieval-Augmented Generation (RAG) have revolutionized natural language processing by integrating Large Language Models (LLMs) with external information retrieval, enabling accurate, up-to-date, and verifiable text generation across diverse applications. However, evaluating RAG systems presents unique challenges due to their hybrid architecture that combines retrieval and generation components, as well as their dependence on dynamic knowledge sources in the LLM era. In response, this paper provides a comprehensive survey of RAG evaluation methods and frameworks, systematically reviewing traditional and emerging evaluation approaches, for system performance, factual accuracy, safety, and computational efficiency in the LLM era. We also compile and categorize the RAG-specific datasets and evaluation frameworks, conducting a meta-analysis of evaluation practices in high-impact RAG research. To the best of our knowledge, this work represents the most comprehensive survey for RAG evaluation, bridging traditional and LLM-driven methods, and serves as a critical resource for advancing RAG development. 

---
# Natural Fingerprints of Large Language Models 

**Authors**: Teppei Suzuki, Ryokan Ri, Sho Takase  

**Link**: [PDF](https://arxiv.org/pdf/2504.14871)  

**Abstract**: Large language models (LLMs) often exhibit biases -- systematic deviations from expected norms -- in their outputs. These range from overt issues, such as unfair responses, to subtler patterns that can reveal which model produced them. We investigate the factors that give rise to identifiable characteristics in LLMs. Since LLMs model training data distribution, it is reasonable that differences in training data naturally lead to the characteristics. However, our findings reveal that even when LLMs are trained on the exact same data, it is still possible to distinguish the source model based on its generated text. We refer to these unintended, distinctive characteristics as natural fingerprints. By systematically controlling training conditions, we show that the natural fingerprints can emerge from subtle differences in the training process, such as parameter sizes, optimization settings, and even random seeds. We believe that understanding natural fingerprints offers new insights into the origins of unintended bias and ways for improving control over LLM behavior. 

---
# Transparentize the Internal and External Knowledge Utilization in LLMs with Trustworthy Citation 

**Authors**: Jiajun Shen, Tong Zhou, Yubo Chen, Delai Qiu, Shengping Liu, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.14856)  

**Abstract**: While hallucinations of large language models could been alleviated through retrieval-augmented generation and citation generation, how the model utilizes internal knowledge is still opaque, and the trustworthiness of its generated answers remains questionable. In this work, we introduce Context-Prior Augmented Citation Generation task, requiring models to generate citations considering both external and internal knowledge while providing trustworthy references, with 5 evaluation metrics focusing on 3 aspects: answer helpfulness, citation faithfulness, and trustworthiness. We introduce RAEL, the paradigm for our task, and also design INTRALIGN, an integrated method containing customary data generation and an alignment algorithm. Our experimental results show that our method achieves a better cross-scenario performance with regard to other baselines. Our extended experiments further reveal that retrieval quality, question types, and model knowledge have considerable influence on the trustworthiness in citation generation. 

---
# On Self-improving Token Embeddings 

**Authors**: Mario M. Kubek, Shiraj Pokharel, Thomas Böhme, Emma L. McDaniel, Herwig Unger, Armin R. Mikler  

**Link**: [PDF](https://arxiv.org/pdf/2504.14808)  

**Abstract**: This article introduces a novel and fast method for refining pre-trained static word or, more generally, token embeddings. By incorporating the embeddings of neighboring tokens in text corpora, it continuously updates the representation of each token, including those without pre-assigned embeddings. This approach effectively addresses the out-of-vocabulary problem, too. Operating independently of large language models and shallow neural networks, it enables versatile applications such as corpus exploration, conceptual search, and word sense disambiguation. The method is designed to enhance token representations within topically homogeneous corpora, where the vocabulary is restricted to a specific domain, resulting in more meaningful embeddings compared to general-purpose pre-trained vectors. As an example, the methodology is applied to explore storm events and their impacts on infrastructure and communities using narratives from a subset of the NOAA Storm Events database. The article also demonstrates how the approach improves the representation of storm-related terms over time, providing valuable insights into the evolving nature of disaster narratives. 

---
# Automatic Evaluation Metrics for Document-level Translation: Overview, Challenges and Trends 

**Authors**: Jiaxin GUO, Xiaoyu Chen, Zhiqiang Rao, Jinlong Yang, Zongyao Li, Hengchao Shang, Daimeng Wei, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14804)  

**Abstract**: With the rapid development of deep learning technologies, the field of machine translation has witnessed significant progress, especially with the advent of large language models (LLMs) that have greatly propelled the advancement of document-level translation. However, accurately evaluating the quality of document-level translation remains an urgent issue. This paper first introduces the development status of document-level translation and the importance of evaluation, highlighting the crucial role of automatic evaluation metrics in reflecting translation quality and guiding the improvement of translation systems. It then provides a detailed analysis of the current state of automatic evaluation schemes and metrics, including evaluation methods with and without reference texts, as well as traditional metrics, Model-based metrics and LLM-based metrics. Subsequently, the paper explores the challenges faced by current evaluation methods, such as the lack of reference diversity, dependence on sentence-level alignment information, and the bias, inaccuracy, and lack of interpretability of the LLM-as-a-judge method. Finally, the paper looks ahead to the future trends in evaluation methods, including the development of more user-friendly document-level evaluation methods and more robust LLM-as-a-judge methods, and proposes possible research directions, such as reducing the dependency on sentence-level information, introducing multi-level and multi-granular evaluation approaches, and training models specifically for machine translation evaluation. This study aims to provide a comprehensive analysis of automatic evaluation for document-level translation and offer insights into future developments. 

---
# Knowledge Distillation and Dataset Distillation of Large Language Models: Emerging Trends, Challenges, and Future Directions 

**Authors**: Luyang Fang, Xiaowei Yu, Jiazhang Cai, Yongkai Chen, Shushan Wu, Zhengliang Liu, Zhenyuan Yang, Haoran Lu, Xilin Gong, Yufang Liu, Terry Ma, Wei Ruan, Ali Abbasi, Jing Zhang, Tao Wang, Ehsan Latif, Wei Liu, Wei Zhang, Soheil Kolouri, Xiaoming Zhai, Dajiang Zhu, Wenxuan Zhong, Tianming Liu, Ping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.14772)  

**Abstract**: The exponential growth of Large Language Models (LLMs) continues to highlight the need for efficient strategies to meet ever-expanding computational and data demands. This survey provides a comprehensive analysis of two complementary paradigms: Knowledge Distillation (KD) and Dataset Distillation (DD), both aimed at compressing LLMs while preserving their advanced reasoning capabilities and linguistic diversity. We first examine key methodologies in KD, such as task-specific alignment, rationale-based training, and multi-teacher frameworks, alongside DD techniques that synthesize compact, high-impact datasets through optimization-based gradient matching, latent space regularization, and generative synthesis. Building on these foundations, we explore how integrating KD and DD can produce more effective and scalable compression strategies. Together, these approaches address persistent challenges in model scalability, architectural heterogeneity, and the preservation of emergent LLM abilities. We further highlight applications across domains such as healthcare and education, where distillation enables efficient deployment without sacrificing performance. Despite substantial progress, open challenges remain in preserving emergent reasoning and linguistic diversity, enabling efficient adaptation to continually evolving teacher models and datasets, and establishing comprehensive evaluation protocols. By synthesizing methodological innovations, theoretical foundations, and practical insights, our survey charts a path toward sustainable, resource-efficient LLMs through the tighter integration of KD and DD principles. 

---
# Disentangling Linguistic Features with Dimension-Wise Analysis of Vector Embeddings 

**Authors**: Saniya Karwa, Navpreet Singh  

**Link**: [PDF](https://arxiv.org/pdf/2504.14766)  

**Abstract**: Understanding the inner workings of neural embeddings, particularly in models such as BERT, remains a challenge because of their high-dimensional and opaque nature. This paper proposes a framework for uncovering the specific dimensions of vector embeddings that encode distinct linguistic properties (LPs). We introduce the Linguistically Distinct Sentence Pairs (LDSP-10) dataset, which isolates ten key linguistic features such as synonymy, negation, tense, and quantity. Using this dataset, we analyze BERT embeddings with various methods, including the Wilcoxon signed-rank test, mutual information, and recursive feature elimination, to identify the most influential dimensions for each LP. We introduce a new metric, the Embedding Dimension Impact (EDI) score, which quantifies the relevance of each embedding dimension to a LP. Our findings show that certain properties, such as negation and polarity, are robustly encoded in specific dimensions, while others, like synonymy, exhibit more complex patterns. This study provides insights into the interpretability of embeddings, which can guide the development of more transparent and optimized language models, with implications for model bias mitigation and the responsible deployment of AI systems. 

---
# PROMPTEVALS: A Dataset of Assertions and Guardrails for Custom Production Large Language Model Pipelines 

**Authors**: Reya Vir, Shreya Shankar, Harrison Chase, Will Fu-Hinthorn, Aditya Parameswaran  

**Link**: [PDF](https://arxiv.org/pdf/2504.14738)  

**Abstract**: Large language models (LLMs) are increasingly deployed in specialized production data processing pipelines across diverse domains -- such as finance, marketing, and e-commerce. However, when running them in production across many inputs, they often fail to follow instructions or meet developer expectations. To improve reliability in these applications, creating assertions or guardrails for LLM outputs to run alongside the pipelines is essential. Yet, determining the right set of assertions that capture developer requirements for a task is challenging. In this paper, we introduce PROMPTEVALS, a dataset of 2087 LLM pipeline prompts with 12623 corresponding assertion criteria, sourced from developers using our open-source LLM pipeline tools. This dataset is 5x larger than previous collections. Using a hold-out test split of PROMPTEVALS as a benchmark, we evaluated closed- and open-source models in generating relevant assertions. Notably, our fine-tuned Mistral and Llama 3 models outperform GPT-4o by 20.93% on average, offering both reduced latency and improved performance. We believe our dataset can spur further research in LLM reliability, alignment, and prompt engineering. 

---
# Evaluating BERTopic on Open-Ended Data: A Case Study with Belgian Dutch Daily Narratives 

**Authors**: Ratna Kandala, Katie Hoemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.14707)  

**Abstract**: This study explores BERTopic's potential for modeling open-ended Belgian Dutch daily narratives, contrasting its performance with Latent Dirichlet Allocation (LDA) and KMeans. Although LDA scores well on certain automated metrics, human evaluations reveal semantically irrelevant co-occurrences, highlighting the limitations of purely statistic-based methods. In contrast, BERTopic's reliance on contextual embeddings yields culturally resonant themes, underscoring the importance of hybrid evaluation frameworks that account for morphologically rich languages. KMeans performed less coherently than prior research suggested, pointing to the unique challenges posed by personal narratives. Our findings emphasize the need for robust generalization in NLP models, especially in underrepresented linguistic contexts. 

---
# OmniV-Med: Scaling Medical Vision-Language Model for Universal Visual Understanding 

**Authors**: Songtao Jiang, Yuan Wang, Sibo Song, Yan Zhang, Zijie Meng, Bohan Lei, Jian Wu, Jimeng Sun, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14692)  

**Abstract**: The practical deployment of medical vision-language models (Med-VLMs) necessitates seamless integration of textual data with diverse visual modalities, including 2D/3D images and videos, yet existing models typically employ separate encoders for different modalities. To address this limitation, we present OmniV-Med, a unified framework for multimodal medical understanding. Our technical contributions are threefold: First, we construct OmniV-Med-Instruct, a comprehensive multimodal medical dataset containing 252K instructional samples spanning 14 medical image modalities and 11 clinical tasks. Second, we devise a rotary position-adaptive encoder that processes multi-resolution 2D/3D images and videos within a unified architecture, diverging from conventional modality-specific encoders. Third, we introduce a medical-aware token pruning mechanism that exploits spatial-temporal redundancy in volumetric data (e.g., consecutive CT slices) and medical videos, effectively reducing 60\% of visual tokens without performance degradation. Empirical evaluations demonstrate that OmniV-Med-7B achieves state-of-the-art performance on 7 benchmarks spanning 2D/3D medical imaging and video understanding tasks. Notably, our lightweight variant (OmniV-Med-1.5B) attains comparable performance while requiring only 8 RTX3090 GPUs for training and supporting efficient long-video inference. Data, code and model will be released. 

---
# FarsEval-PKBETS: A new diverse benchmark for evaluating Persian large language models 

**Authors**: Mehrnoush Shamsfard, Zahra Saaberi, Mostafa Karimi manesh, Seyed Mohammad Hossein Hashemi, Zahra Vatankhah, Motahareh Ramezani, Niki Pourazin, Tara Zare, Maryam Azimi, Sarina Chitsaz, Sama Khoraminejad, Morteza Mahdavi Mortazavi, Mohammad Mahdi Chizari, Sahar Maleki, Seyed Soroush Majd, Mostafa Masumi, Sayed Ali Musavi Khoeini, Amir Mohseni, Sogol Alipour  

**Link**: [PDF](https://arxiv.org/pdf/2504.14690)  

**Abstract**: Research on evaluating and analyzing large language models (LLMs) has been extensive for resource-rich languages such as English, yet their performance in languages such as Persian has received considerably less attention. This paper introduces FarsEval-PKBETS benchmark, a subset of FarsEval project for evaluating large language models in Persian. This benchmark consists of 4000 questions and answers in various formats, including multiple choice, short answer and descriptive responses. It covers a wide range of domains and tasks,including medicine, law, religion, Persian language, encyclopedic knowledge, human preferences, social knowledge, ethics and bias, text generation, and respecting others' rights. This bechmark incorporates linguistics, cultural, and local considerations relevant to the Persian language and Iran. To ensure the questions are challenging for current LLMs, three models -- Llama3-70B, PersianMind, and Dorna -- were evaluated using this benchmark. Their average accuracy was below 50%, meaning they provided fully correct answers to fewer than half of the questions. These results indicate that current language models are still far from being able to solve this benchmark 

---
# Trans-Zero: Self-Play Incentivizes Large Language Models for Multilingual Translation Without Parallel Data 

**Authors**: Wei Zou, Sen Yang, Yu Bao, Shujian Huang, Jiajun Chen, Shanbo Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14669)  

**Abstract**: The rise of Large Language Models (LLMs) has reshaped machine translation (MT), but multilingual MT still relies heavily on parallel data for supervised fine-tuning (SFT), facing challenges like data scarcity for low-resource languages and catastrophic forgetting. To address these issues, we propose TRANS-ZERO, a self-play framework that leverages only monolingual data and the intrinsic multilingual knowledge of LLM. TRANS-ZERO combines Genetic Monte-Carlo Tree Search (G-MCTS) with preference optimization, achieving strong translation performance that rivals supervised methods. Experiments demonstrate that this approach not only matches the performance of models trained on large-scale parallel data but also excels in non-English translation directions. Further analysis reveals that G-MCTS itself significantly enhances translation quality by exploring semantically consistent candidates through iterative translations, providing a robust foundation for the framework's succuss. 

---
# A Case Study Exploring the Current Landscape of Synthetic Medical Record Generation with Commercial LLMs 

**Authors**: Yihan Lin, Zhirong Bella Yu, Simon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.14657)  

**Abstract**: Synthetic Electronic Health Records (EHRs) offer a valuable opportunity to create privacy preserving and harmonized structured data, supporting numerous applications in healthcare. Key benefits of synthetic data include precise control over the data schema, improved fairness and representation of patient populations, and the ability to share datasets without concerns about compromising real individuals privacy. Consequently, the AI community has increasingly turned to Large Language Models (LLMs) to generate synthetic data across various domains. However, a significant challenge in healthcare is ensuring that synthetic health records reliably generalize across different hospitals, a long standing issue in the field. In this work, we evaluate the current state of commercial LLMs for generating synthetic data and investigate multiple aspects of the generation process to identify areas where these models excel and where they fall short. Our main finding from this work is that while LLMs can reliably generate synthetic health records for smaller subsets of features, they struggle to preserve realistic distributions and correlations as the dimensionality of the data increases, ultimately limiting their ability to generalize across diverse hospital settings. 

---
# Harnessing Generative LLMs for Enhanced Financial Event Entity Extraction Performance 

**Authors**: Soo-joon Choi, Ji-jun Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.14633)  

**Abstract**: Financial event entity extraction is a crucial task for analyzing market dynamics and building financial knowledge graphs, yet it presents significant challenges due to the specialized language and complex structures in financial texts. Traditional approaches often rely on sequence labeling models, which can struggle with long-range dependencies and the inherent complexity of extracting multiple, potentially overlapping entities. Motivated by the advanced language understanding and generative capabilities of Large Language Models (LLMs), we propose a novel method that reframes financial event entity extraction as a text-to-structured-output generation task. Our approach involves fine-tuning a pre-trained LLM using Parameter-Efficient Fine-Tuning (PEFT) to directly generate a structured representation, such as a JSON object, containing the extracted entities and their precise character spans from the input text. We evaluate our method on the challenging CCKS 2019 Financial Event Entity Extraction dataset, comparing its performance against strong sequence labeling baselines, including SEBERTNets and sebertNets. Experimental results demonstrate that our generative LLM method achieves a new state-of-the-art F1 score on this benchmark, significantly outperforming previous methods. Through detailed quantitative analysis across event types, entity types, and instance complexity, as well as human evaluation, we show that our approach is more effective at handling the nuances of financial text and extracting high-quality entities. This work validates the potential of applying generative LLMs directly to complex, domain-specific information extraction tasks requiring structured output. 

---
# Automatic Text Summarization (ATS) for Research Documents in Sorani Kurdish 

**Authors**: Rondik Hadi Abdulrahman, Hossein Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2504.14630)  

**Abstract**: Extracting concise information from scientific documents aids learners, researchers, and practitioners. Automatic Text Summarization (ATS), a key Natural Language Processing (NLP) application, automates this process. While ATS methods exist for many languages, Kurdish remains underdeveloped due to limited resources. This study develops a dataset and language model based on 231 scientific papers in Sorani Kurdish, collected from four academic departments in two universities in the Kurdistan Region of Iraq (KRI), averaging 26 pages per document. Using Sentence Weighting and Term Frequency-Inverse Document Frequency (TF-IDF) algorithms, two experiments were conducted, differing in whether the conclusions were included. The average word count was 5,492.3 in the first experiment and 5,266.96 in the second. Results were evaluated manually and automatically using ROUGE-1, ROUGE-2, and ROUGE-L metrics, with the best accuracy reaching 19.58%. Six experts conducted manual evaluations using three criteria, with results varying by document. This research provides valuable resources for Kurdish NLP researchers to advance ATS and related fields. 

---
# A Hierarchical Framework for Measuring Scientific Paper Innovation via Large Language Models 

**Authors**: Hongming Tan, Shaoxiong Zhan, Fengwei Jia, Hai-Tao Zheng, Wai Kin Chan  

**Link**: [PDF](https://arxiv.org/pdf/2504.14620)  

**Abstract**: Measuring scientific paper innovation is both important and challenging. Existing content-based methods often overlook the full-paper context, fail to capture the full scope of innovation, and lack generalization. We propose HSPIM, a hierarchical and training-free framework based on large language models (LLMs). It introduces a Paper-to-Sections-to-QAs decomposition to assess innovation. We segment the text by section titles and use zero-shot LLM prompting to implement section classification, question-answering (QA) augmentation, and weighted novelty scoring. The generated QA pair focuses on section-level innovation and serves as additional context to improve the LLM scoring. For each chunk, the LLM outputs a novelty score and a confidence score. We use confidence scores as weights to aggregate novelty scores into a paper-level innovation score. To further improve performance, we propose a two-layer question structure consisting of common and section-specific questions, and apply a genetic algorithm to optimize the question-prompt combinations. Comprehensive experiments on scientific conference paper datasets show that HSPIM outperforms baseline methods in effectiveness, generalization, and interpretability. 

---
# Translation Analytics for Freelancers: I. Introduction, Data Preparation, Baseline Evaluations 

**Authors**: Yuri Balashov, Alex Balashov, Shiho Fukuda Koski  

**Link**: [PDF](https://arxiv.org/pdf/2504.14619)  

**Abstract**: This is the first in a series of papers exploring the rapidly expanding new opportunities arising from recent progress in language technologies for individual translators and language service providers with modest resources. The advent of advanced neural machine translation systems, large language models, and their integration into workflows via computer-assisted translation tools and translation management systems have reshaped the translation landscape. These advancements enable not only translation but also quality evaluation, error spotting, glossary generation, and adaptation to domain-specific needs, creating new technical opportunities for freelancers. In this series, we aim to empower translators with actionable methods to harness these advancements. Our approach emphasizes Translation Analytics, a suite of evaluation techniques traditionally reserved for large-scale industry applications but now becoming increasingly available for smaller-scale users. This first paper introduces a practical framework for adapting automatic evaluation metrics -- such as BLEU, chrF, TER, and COMET -- to freelancers' needs. We illustrate the potential of these metrics using a trilingual corpus derived from a real-world project in the medical domain and provide statistical analysis correlating human evaluations with automatic scores. Our findings emphasize the importance of proactive engagement with emerging technologies to not only adapt but thrive in the evolving professional environment. 

---
# a1: Steep Test-time Scaling Law via Environment Augmented Generation 

**Authors**: Lingrui Mei, Shenghua Liu, Yiwei Wang, Baolong Bi, Yuyao Ge, Jun Wan, Yurong Wu, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14597)  

**Abstract**: Large Language Models (LLMs) have made remarkable breakthroughs in reasoning, yet continue to struggle with hallucinations, logical errors, and inability to self-correct during complex multi-step tasks. Current approaches like chain-of-thought prompting offer limited reasoning capabilities that fail when precise step validation is required. We propose Environment Augmented Generation (EAG), a framework that enhances LLM reasoning through: (1) real-time environmental feedback validating each reasoning step, (2) dynamic branch exploration for investigating alternative solution paths when faced with errors, and (3) experience-based learning from successful reasoning trajectories. Unlike existing methods, EAG enables deliberate backtracking and strategic replanning through tight integration of execution feedback with branching exploration. Our a1-32B model achieves state-of-the-art performance among similar-sized models across all benchmarks, matching larger models like o1 on competition mathematics while outperforming comparable models by up to 24.4 percentage points. Analysis reveals EAG's distinctive scaling pattern: initial token investment in environment interaction yields substantial long-term performance dividends, with advantages amplifying proportionally to task complexity. EAG's theoretical framework demonstrates how environment interactivity and systematic branch exploration together establish a new paradigm for reliable machine reasoning, particularly for problems requiring precise multi-step calculation and logical verification. 

---
# BookWorld: From Novels to Interactive Agent Societies for Creative Story Generation 

**Authors**: Yiting Ran, Xintao Wang, Tian Qiu, Jiaqing Liang, Yanghua Xiao, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14538)  

**Abstract**: Recent advances in large language models (LLMs) have enabled social simulation through multi-agent systems. Prior efforts focus on agent societies created from scratch, assigning agents with newly defined personas. However, simulating established fictional worlds and characters remain largely underexplored, despite its significant practical value. In this paper, we introduce BookWorld, a comprehensive system for constructing and simulating book-based multi-agent societies. BookWorld's design covers comprehensive real-world intricacies, including diverse and dynamic characters, fictional worldviews, geographical constraints and changes, e.t.c. BookWorld enables diverse applications including story generation, interactive games and social simulation, offering novel ways to extend and explore beloved fictional works. Through extensive experiments, we demonstrate that BookWorld generates creative, high-quality stories while maintaining fidelity to the source books, surpassing previous methods with a win rate of 75.36%. The code of this paper can be found at the project page: this https URL. 

---
# Causality for Natural Language Processing 

**Authors**: Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14530)  

**Abstract**: Causal reasoning is a cornerstone of human intelligence and a critical capability for artificial systems aiming to achieve advanced understanding and decision-making. This thesis delves into various dimensions of causal reasoning and understanding in large language models (LLMs). It encompasses a series of studies that explore the causal inference skills of LLMs, the mechanisms behind their performance, and the implications of causal and anticausal learning for natural language processing (NLP) tasks. Additionally, it investigates the application of causal reasoning in text-based computational social science, specifically focusing on political decision-making and the evaluation of scientific impact through citations. Through novel datasets, benchmark tasks, and methodological frameworks, this work identifies key challenges and opportunities to improve the causal capabilities of LLMs, providing a comprehensive foundation for future research in this evolving field. 

---
# Functional Abstraction of Knowledge Recall in Large Language Models 

**Authors**: Zijian Wang, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14496)  

**Abstract**: Pre-trained transformer large language models (LLMs) demonstrate strong knowledge recall capabilities. This paper investigates the knowledge recall mechanism in LLMs by abstracting it into a functional structure. We propose that during knowledge recall, the model's hidden activation space implicitly entails a function execution process where specific activation vectors align with functional components (Input argument, Function body, and Return values). Specifically, activation vectors of relation-related tokens define a mapping function from subjects to objects, with subject-related token activations serving as input arguments and object-related token activations as return values. For experimental verification, we first design a patching-based knowledge-scoring algorithm to identify knowledge-aware activation vectors as independent functional components. Then, we conduct counter-knowledge testing to examine the independent functional effects of each component on knowledge recall outcomes. From this functional perspective, we improve the contextual knowledge editing approach augmented by activation patching. By rewriting incoherent activations in context, we enable improved short-term memory retention for new knowledge prompting. 

---
# FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering 

**Authors**: Yichen Li, Zhiting Fan, Ruizhe Chen, Xiaotang Gai, Luqi Gong, Yan Zhang, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14492)  

**Abstract**: Large language models (LLMs) are prone to capturing biases from training corpus, leading to potential negative social impacts. Existing prompt-based debiasing methods exhibit instability due to their sensitivity to prompt changes, while fine-tuning-based techniques incur substantial computational overhead and catastrophic forgetting. In this paper, we propose FairSteer, a novel inference-time debiasing framework without requiring customized prompt design or model retraining. Motivated by the linear representation hypothesis, our preliminary investigation demonstrates that fairness-related features can be encoded into separable directions in the hidden activation space. FairSteer operates in three steps: biased activation detection, debiasing steering vector (DSV) computation, and dynamic activation steering. Specifically, it first trains a lightweight linear classifier to detect bias signatures in activations, and then computes DSVs as intervention directions derived from small contrastive prompt pairs. Subsequently, it performs debiasing by adjusting activations with DSVs in the inference stage. Comprehensive evaluation with six LLMs demonstrates the superiority of FairSteer across question-answering, counterfactual input evaluation and open-ended text generation tasks. Code will be released. 

---
# DialogueAgents: A Hybrid Agent-Based Speech Synthesis Framework for Multi-Party Dialogue 

**Authors**: Xiang Li, Duyi Pan, Hongru Xiao, Jiale Han, Jing Tang, Jiabao Ma, Wei Wang, Bo Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14482)  

**Abstract**: Speech synthesis is crucial for human-computer interaction, enabling natural and intuitive communication. However, existing datasets involve high construction costs due to manual annotation and suffer from limited character diversity, contextual scenarios, and emotional expressiveness. To address these issues, we propose DialogueAgents, a novel hybrid agent-based speech synthesis framework, which integrates three specialized agents -- a script writer, a speech synthesizer, and a dialogue critic -- to collaboratively generate dialogues. Grounded in a diverse character pool, the framework iteratively refines dialogue scripts and synthesizes speech based on speech review, boosting emotional expressiveness and paralinguistic features of the synthesized dialogues. Using DialogueAgent, we contribute MultiTalk, a bilingual, multi-party, multi-turn speech dialogue dataset covering diverse topics. Extensive experiments demonstrate the effectiveness of our framework and the high quality of the MultiTalk dataset. We release the dataset and code this https URL to facilitate future research on advanced speech synthesis models and customized data generation. 

---
# sEEG-based Encoding for Sentence Retrieval: A Contrastive Learning Approach to Brain-Language Alignment 

**Authors**: Yijun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14468)  

**Abstract**: Interpreting neural activity through meaningful latent representations remains a complex and evolving challenge at the intersection of neuroscience and artificial intelligence. We investigate the potential of multimodal foundation models to align invasive brain recordings with natural language. We present SSENSE, a contrastive learning framework that projects single-subject stereo-electroencephalography (sEEG) signals into the sentence embedding space of a frozen CLIP model, enabling sentence-level retrieval directly from brain activity. SSENSE trains a neural encoder on spectral representations of sEEG using InfoNCE loss, without fine-tuning the text encoder. We evaluate our method on time-aligned sEEG and spoken transcripts from a naturalistic movie-watching dataset. Despite limited data, SSENSE achieves promising results, demonstrating that general-purpose language representations can serve as effective priors for neural decoding. 

---
# CoLoTa: A Dataset for Entity-based Commonsense Reasoning over Long-Tail Knowledge 

**Authors**: Armin Toroghi, Willis Guo, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2504.14462)  

**Abstract**: The rise of Large Language Models (LLMs) has redefined the AI landscape, particularly due to their ability to encode factual and commonsense knowledge, and their outstanding performance in tasks requiring reasoning. Despite these advances, hallucinations and reasoning errors remain a significant barrier to their deployment in high-stakes settings. In this work, we observe that even the most prominent LLMs, such as OpenAI-o1, suffer from high rates of reasoning errors and hallucinations on tasks requiring commonsense reasoning over obscure, long-tail entities. To investigate this limitation, we present a new dataset for Commonsense reasoning over Long-Tail entities (CoLoTa), that consists of 3,300 queries from question answering and claim verification tasks and covers a diverse range of commonsense reasoning skills. We remark that CoLoTa can also serve as a Knowledge Graph Question Answering (KGQA) dataset since the support of knowledge required to answer its queries is present in the Wikidata knowledge graph. However, as opposed to existing KGQA benchmarks that merely focus on factoid questions, our CoLoTa queries also require commonsense reasoning. Our experiments with strong LLM-based KGQA methodologies indicate their severe inability to answer queries involving commonsense reasoning. Hence, we propose CoLoTa as a novel benchmark for assessing both (i) LLM commonsense reasoning capabilities and their robustness to hallucinations on long-tail entities and (ii) the commonsense reasoning capabilities of KGQA methods. 

---
# ParaPO: Aligning Language Models to Reduce Verbatim Reproduction of Pre-training Data 

**Authors**: Tong Chen, Faeze Brahman, Jiacheng Liu, Niloofar Mireshghallah, Weijia Shi, Pang Wei Koh, Luke Zettlemoyer, Hannaneh Hajishirzi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14452)  

**Abstract**: Language models (LMs) can memorize and reproduce segments from their pretraining data verbatim even in non-adversarial settings, raising concerns about copyright, plagiarism, privacy, and creativity. We introduce Paraphrase Preference Optimization (ParaPO), a post-training method that fine-tunes LMs to reduce unintentional regurgitation while preserving their overall utility. ParaPO trains LMs to prefer paraphrased versions of memorized segments over the original verbatim content from the pretraining data. To maintain the ability to recall famous quotations when appropriate, we develop a variant of ParaPO that uses system prompts to control regurgitation behavior. In our evaluation on Llama3.1-8B, ParaPO consistently reduces regurgitation across all tested datasets (e.g., reducing the regurgitation metric from 17.3 to 12.9 in creative writing), whereas unlearning methods used in prior work to mitigate regurgitation are less effective outside their targeted unlearned domain (from 17.3 to 16.9). When applied to the instruction-tuned Tulu3-8B model, ParaPO with system prompting successfully preserves famous quotation recall while reducing unintentional regurgitation (from 8.7 to 6.3 in creative writing) when prompted not to regurgitate. In contrast, without ParaPO tuning, prompting the model not to regurgitate produces only a marginal reduction (8.7 to 8.4). 

---
# Diverse Prompts: Illuminating the Prompt Space of Large Language Models with MAP-Elites 

**Authors**: Gabriel Machado Santos, Rita Maria da Silva Julia, Marcelo Zanchetta do Nascimento  

**Link**: [PDF](https://arxiv.org/pdf/2504.14367)  

**Abstract**: Prompt engineering is essential for optimizing large language models (LLMs), yet the link between prompt structures and task performance remains underexplored. This work introduces an evolutionary approach that combines context-free grammar (CFG) with the MAP-Elites algorithm to systematically explore the prompt space. Our method prioritizes quality and diversity, generating high-performing and structurally varied prompts while analyzing their alignment with diverse tasks by varying traits such as the number of examples (shots) and reasoning depth. By systematically mapping the phenotypic space, we reveal how structural variations influence LLM performance, offering actionable insights for task-specific and adaptable prompt design. Evaluated on seven BigBench Lite tasks across multiple LLMs, our results underscore the critical interplay of quality and diversity, advancing the effectiveness and versatility of LLMs. 

---
# Empirical Evaluation of Knowledge Distillation from Transformers to Subquadratic Language Models 

**Authors**: Patrick Haller, Jonas Golde, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2504.14366)  

**Abstract**: Knowledge distillation is a widely used technique for compressing large language models (LLMs) by training a smaller student model to mimic a larger teacher model. Typically, both the teacher and student are Transformer-based architectures, leveraging softmax attention for sequence modeling. However, the quadratic complexity of self-attention at inference time remains a significant bottleneck, motivating the exploration of subquadratic alternatives such as structured state-space models (SSMs), linear attention, and recurrent architectures. In this work, we systematically evaluate the transferability of knowledge distillation from a Transformer teacher to nine subquadratic student architectures. Our study aims to determine which subquadratic model best aligns with the teacher's learned representations and how different architectural constraints influence the distillation process. We also investigate the impact of intelligent initialization strategies, including matrix mixing and query-key-value (QKV) copying, on the adaptation process. Our empirical results on multiple NLP benchmarks provide insights into the trade-offs between efficiency and performance, highlighting key factors for successful knowledge transfer to subquadratic architectures. 

---
# Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach 

**Authors**: Xingyu Li, Chen Gong, Guohong Fu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14321)  

**Abstract**: Multimodal coreference resolution (MCR) aims to identify mentions referring to the same entity across different modalities, such as text and visuals, and is essential for understanding multimodal content. In the era of rapidly growing mutimodal content and social media, MCR is particularly crucial for interpreting user interactions and bridging text-visual references to improve communication and personalization. However, MCR research for real-world dialogues remains unexplored due to the lack of sufficient data this http URL address this gap, we introduce TikTalkCoref, the first Chinese multimodal coreference dataset for social media in real-world scenarios, derived from the popular Douyin short-video platform. This dataset pairs short videos with corresponding textual dialogues from user comments and includes manually annotated coreference clusters for both person mentions in the text and the coreferential person head regions in the corresponding video frames. We also present an effective benchmark approach for MCR, focusing on the celebrity domain, and conduct extensive experiments on our dataset, providing reliable benchmark results for this newly constructed dataset. We will release the TikTalkCoref dataset to facilitate future research on MCR for real-world social media dialogues. 

---
# Probing the Subtle Ideological Manipulation of Large Language Models 

**Authors**: Demetris Paschalides, George Pallis, Marios D. Dikaiakos  

**Link**: [PDF](https://arxiv.org/pdf/2504.14287)  

**Abstract**: Large Language Models (LLMs) have transformed natural language processing, but concerns have emerged about their susceptibility to ideological manipulation, particularly in politically sensitive areas. Prior work has focused on binary Left-Right LLM biases, using explicit prompts and fine-tuning on political QA datasets. In this work, we move beyond this binary approach to explore the extent to which LLMs can be influenced across a spectrum of political ideologies, from Progressive-Left to Conservative-Right. We introduce a novel multi-task dataset designed to reflect diverse ideological positions through tasks such as ideological QA, statement ranking, manifesto cloze completion, and Congress bill comprehension. By fine-tuning three LLMs-Phi-2, Mistral, and Llama-3-on this dataset, we evaluate their capacity to adopt and express these nuanced ideologies. Our findings indicate that fine-tuning significantly enhances nuanced ideological alignment, while explicit prompts provide only minor refinements. This highlights the models' susceptibility to subtle ideological manipulation, suggesting a need for more robust safeguards to mitigate these risks. 

---
# Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale 

**Authors**: Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle Ungar, Camillo J. Taylor, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2504.14225)  

**Abstract**: Large Language Models (LLMs) have emerged as personalized assistants for users across a wide range of tasks -- from offering writing support to delivering tailored recommendations or consultations. Over time, the interaction history between a user and an LLM can provide extensive information about an individual's traits and preferences. However, open questions remain on how well LLMs today can effectively leverage such history to (1) internalize the user's inherent traits and preferences, (2) track how the user profiling and preferences evolve over time, and (3) generate personalized responses accordingly in new scenarios.
In this work, we introduce the PERSONAMEM benchmark. PERSONAMEM features curated user profiles with over 180 simulated user-LLM interaction histories, each containing up to 60 sessions of multi-turn conversations across 15 real-world tasks that require personalization. Given an in-situ user query, i.e. query issued by the user from the first-person perspective, we evaluate LLM chatbots' ability to identify the most suitable response according to the current state of the user's profile. We observe that current LLMs still struggle to recognize the dynamic evolution in users' profiles over time through direct prompting approaches. As a consequence, LLMs often fail to deliver responses that align with users' current situations and preferences, with frontier models such as GPT-4.1, o4-mini, GPT-4.5, o1, or Gemini-2.0 achieving only around 50% overall accuracy, suggesting room for improvement. We hope that PERSONAMEM, along with the user profile and conversation simulation pipeline, can facilitate future research in the development of truly user-aware chatbots. Code and data are available at this http URL. 

---
# SimplifyMyText: An LLM-Based System for Inclusive Plain Language Text Simplification 

**Authors**: Michael Färber, Parisa Aghdam, Kyuri Im, Mario Tawfelis, Hardik Ghoshal  

**Link**: [PDF](https://arxiv.org/pdf/2504.14223)  

**Abstract**: Text simplification is essential for making complex content accessible to diverse audiences who face comprehension challenges. Yet, the limited availability of simplified materials creates significant barriers to personal and professional growth and hinders social inclusion. Although researchers have explored various methods for automatic text simplification, none fully leverage large language models (LLMs) to offer tailored customization for different target groups and varying levels of simplicity. Moreover, despite its proven benefits for both consumers and organizations, the well-established practice of plain language remains underutilized. In this paper, we this https URL, the first system designed to produce plain language content from multiple input formats, including typed text and file uploads, with flexible customization options for diverse audiences. We employ GPT-4 and Llama-3 and evaluate outputs across multiple metrics. Overall, our work contributes to research on automatic text simplification and highlights the importance of tailored communication in promoting inclusivity. 

---
# Understanding the Repeat Curse in Large Language Models from a Feature Perspective 

**Authors**: Junchi Yao, Shu Yang, Jianhua Xu, Lijie Hu, Mengdi Li, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14218)  

**Abstract**: Large language models (LLMs) have made remarkable progress in various domains, yet they often suffer from repetitive text generation, a phenomenon we refer to as the "Repeat Curse". While previous studies have proposed decoding strategies to mitigate repetition, the underlying mechanism behind this issue remains insufficiently explored. In this work, we investigate the root causes of repetition in LLMs through the lens of mechanistic interpretability. Inspired by recent advances in Sparse Autoencoders (SAEs), which enable monosemantic feature extraction, we propose a novel approach, "Duplicatus Charm", to induce and analyze the Repeat Curse. Our method systematically identifies "Repetition Features" -the key model activations responsible for generating repetitive outputs. First, we locate the layers most involved in repetition through logit analysis. Next, we extract and stimulate relevant features using SAE-based activation manipulation. To validate our approach, we construct a repetition dataset covering token and paragraph level repetitions and introduce an evaluation pipeline to quantify the influence of identified repetition features. Furthermore, by deactivating these features, we have effectively mitigated the Repeat Curse. 

---
# Bias Analysis and Mitigation through Protected Attribute Detection and Regard Classification 

**Authors**: Takuma Udagawa, Yang Zhao, Hiroshi Kanayama, Bishwaranjan Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2504.14212)  

**Abstract**: Large language models (LLMs) acquire general linguistic knowledge from massive-scale pretraining. However, pretraining data mainly comprised of web-crawled texts contain undesirable social biases which can be perpetuated or even amplified by LLMs. In this study, we propose an efficient yet effective annotation pipeline to investigate social biases in the pretraining corpora. Our pipeline consists of protected attribute detection to identify diverse demographics, followed by regard classification to analyze the language polarity towards each attribute. Through our experiments, we demonstrate the effect of our bias analysis and mitigation measures, focusing on Common Crawl as the most representative pretraining corpus. 

---
# EIoU-EMC: A Novel Loss for Domain-specific Nested Entity Recognition 

**Authors**: Jian Zhang, Tianqing Zhang, Qi Li, Hongwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14203)  

**Abstract**: In recent years, research has mainly focused on the general NER task. There still have some challenges with nested NER task in the specific domains. Specifically, the scenarios of low resource and class imbalance impede the wide application for biomedical and industrial domains. In this study, we design a novel loss EIoU-EMC, by enhancing the implement of Intersection over Union loss and Multiclass loss. Our proposed method specially leverages the information of entity boundary and entity classification, thereby enhancing the model's capacity to learn from a limited number of data samples. To validate the performance of this innovative method in enhancing NER task, we conducted experiments on three distinct biomedical NER datasets and one dataset constructed by ourselves from industrial complex equipment maintenance documents. Comparing to strong baselines, our method demonstrates the competitive performance across all datasets. During the experimental analysis, our proposed method exhibits significant advancements in entity boundary recognition and entity classification. Our code are available here. 

---
# Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models 

**Authors**: Xinlin Zhuang, Jiahui Peng, Ren Ma, Yinfan Wang, Tianyi Bai, Xingjian Wei, Jiantao Qiu, Chi Zhang, Ying Qian, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2504.14194)  

**Abstract**: The composition of pre-training datasets for large language models (LLMs) remains largely undisclosed, hindering transparency and efforts to optimize data quality, a critical driver of model performance. Current data selection methods, such as natural language quality assessments, diversity-based filters, and classifier-based approaches, are limited by single-dimensional evaluation or redundancy-focused strategies. To address these gaps, we propose PRRC to evaluate data quality across Professionalism, Readability, Reasoning, and Cleanliness. We further introduce Meta-rater, a multi-dimensional data selection method that integrates these dimensions with existing quality metrics through learned optimal weightings. Meta-rater employs proxy models to train a regression model that predicts validation loss, enabling the identification of optimal combinations of quality scores. Experiments demonstrate that Meta-rater doubles convergence speed for 1.3B parameter models and improves downstream task performance by 3.23, with scalable benefits observed in 3.3B models trained on 100B tokens. Additionally, we release the annotated SlimPajama-627B dataset, labeled across 25 quality metrics (including PRRC), to advance research in data-centric LLM development. Our work establishes that holistic, multi-dimensional quality integration significantly outperforms conventional single-dimension approaches, offering a scalable paradigm for enhancing pre-training efficiency and model capability. 

---
# Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion 

**Authors**: Yejun Yoon, Jaeyoon Jung, Seunghyun Yoon, Kunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.14175)  

**Abstract**: Query expansion methods powered by large language models (LLMs) have demonstrated effectiveness in zero-shot retrieval tasks. These methods assume that LLMs can generate hypothetical documents that, when incorporated into a query vector, enhance the retrieval of real evidence. However, we challenge this assumption by investigating whether knowledge leakage in benchmarks contributes to the observed performance gains. Using fact verification as a testbed, we analyzed whether the generated documents contained information entailed by ground truth evidence and assessed their impact on performance. Our findings indicate that performance improvements occurred consistently only for claims whose generated documents included sentences entailed by ground truth evidence. This suggests that knowledge leakage may be present in these benchmarks, inflating the perceived performance of LLM-based query expansion methods, particularly in real-world scenarios that require retrieving niche or novel knowledge. 

---
# Self-Correction Makes LLMs Better Parsers 

**Authors**: Ziyan Zhang, Yang Hou, Chen Gong, Zhenghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14165)  

**Abstract**: Large language models (LLMs) have achieved remarkable success across various natural language processing (NLP) tasks. However, recent studies suggest that they still face challenges in performing fundamental NLP tasks essential for deep language understanding, particularly syntactic parsing. In this paper, we conduct an in-depth analysis of LLM parsing capabilities, delving into the specific shortcomings of their parsing results. We find that LLMs may stem from limitations to fully leverage grammar rules in existing treebanks, which restricts their capability to generate valid syntactic structures. To help LLMs acquire knowledge without additional training, we propose a self-correction method that leverages grammar rules from existing treebanks to guide LLMs in correcting previous errors. Specifically, we automatically detect potential errors and dynamically search for relevant rules, offering hints and examples to guide LLMs in making corrections themselves. Experimental results on three datasets with various LLMs, demonstrate that our method significantly improves performance in both in-domain and cross-domain settings on the English and Chinese datasets. 

---
# SConU: Selective Conformal Uncertainty in Large Language Models 

**Authors**: Zhiyuan Wang, Qingni Wang, Yue Zhang, Tianlong Chen, Xiaofeng Zhu, Xiaoshuang Shi, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14154)  

**Abstract**: As large language models are increasingly utilized in real-world applications, guarantees of task-specific metrics are essential for their reliable deployment. Previous studies have introduced various criteria of conformal uncertainty grounded in split conformal prediction, which offer user-specified correctness coverage. However, existing frameworks often fail to identify uncertainty data outliers that violate the exchangeability assumption, leading to unbounded miscoverage rates and unactionable prediction sets. In this paper, we propose a novel approach termed Selective Conformal Uncertainty (SConU), which, for the first time, implements significance tests, by developing two conformal p-values that are instrumental in determining whether a given sample deviates from the uncertainty distribution of the calibration set at a specific manageable risk level. Our approach not only facilitates rigorous management of miscoverage rates across both single-domain and interdisciplinary contexts, but also enhances the efficiency of predictions. Furthermore, we comprehensively analyze the components of the conformal procedures, aiming to approximate conditional coverage, particularly in high-stakes question-answering tasks. 

---
# Walk the Talk? Measuring the Faithfulness of Large Language Model Explanations 

**Authors**: Katie Matton, Robert Osazuwa Ness, John Guttag, Emre Kıcıman  

**Link**: [PDF](https://arxiv.org/pdf/2504.14150)  

**Abstract**: Large language models (LLMs) are capable of generating plausible explanations of how they arrived at an answer to a question. However, these explanations can misrepresent the model's "reasoning" process, i.e., they can be unfaithful. This, in turn, can lead to over-trust and misuse. We introduce a new approach for measuring the faithfulness of LLM explanations. First, we provide a rigorous definition of faithfulness. Since LLM explanations mimic human explanations, they often reference high-level concepts in the input question that purportedly influenced the model. We define faithfulness in terms of the difference between the set of concepts that LLM explanations imply are influential and the set that truly are. Second, we present a novel method for estimating faithfulness that is based on: (1) using an auxiliary LLM to modify the values of concepts within model inputs to create realistic counterfactuals, and (2) using a Bayesian hierarchical model to quantify the causal effects of concepts at both the example- and dataset-level. Our experiments show that our method can be used to quantify and discover interpretable patterns of unfaithfulness. On a social bias task, we uncover cases where LLM explanations hide the influence of social bias. On a medical question answering task, we uncover cases where LLM explanations provide misleading claims about which pieces of evidence influenced the model's decisions. 

---
# PEFT A2Z: Parameter-Efficient Fine-Tuning Survey for Large Language and Vision Models 

**Authors**: Nusrat Jahan Prottasha, Upama Roy Chowdhury, Shetu Mohanto, Tasfia Nuzhat, Abdullah As Sami, Md Shamol Ali, Md Shohanur Islam Sobuj, Hafijur Raman, Md Kowsher, Ozlem Ozmen Garibay  

**Link**: [PDF](https://arxiv.org/pdf/2504.14117)  

**Abstract**: Large models such as Large Language Models (LLMs) and Vision Language Models (VLMs) have transformed artificial intelligence, powering applications in natural language processing, computer vision, and multimodal learning. However, fully fine-tuning these models remains expensive, requiring extensive computational resources, memory, and task-specific data. Parameter-Efficient Fine-Tuning (PEFT) has emerged as a promising solution that allows adapting large models to downstream tasks by updating only a small portion of parameters. This survey presents a comprehensive overview of PEFT techniques, focusing on their motivations, design principles, and effectiveness. We begin by analyzing the resource and accessibility challenges posed by traditional fine-tuning and highlight key issues, such as overfitting, catastrophic forgetting, and parameter inefficiency. We then introduce a structured taxonomy of PEFT methods -- grouped into additive, selective, reparameterized, hybrid, and unified frameworks -- and systematically compare their mechanisms and trade-offs. Beyond taxonomy, we explore the impact of PEFT across diverse domains, including language, vision, and generative modeling, showing how these techniques offer strong performance with lower resource costs. We also discuss important open challenges in scalability, interpretability, and robustness, and suggest future directions such as federated learning, domain adaptation, and theoretical grounding. Our goal is to provide a unified understanding of PEFT and its growing role in enabling practical, efficient, and sustainable use of large models. 

---
# LogicTree: Structured Proof Exploration for Coherent and Rigorous Logical Reasoning with Large Language Models 

**Authors**: Kang He, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2504.14089)  

**Abstract**: Large language models (LLMs) have achieved remarkable multi-step reasoning capabilities across various domains. However, LLMs still face distinct challenges in complex logical reasoning, as (1) proof-finding requires systematic exploration and the maintenance of logical coherence and (2) searching the right combination of premises at each reasoning step is inherently challenging in tasks with large premise space. To address this, we propose LogicTree, an inference-time modular framework employing algorithm-guided search to automate structured proof exploration and ensure logical coherence. Advancing beyond tree-of-thought (ToT), we incorporate caching mechanism into LogicTree to enable effective utilization of historical knowledge, preventing reasoning stagnation and minimizing redundancy. Furthermore, we address the combinatorial complexity of premise search by decomposing it into a linear process. The refined premise selection restricts subsequent inference to at most one derivation per step, enhancing reasoning granularity and enforcing strict step-by-step reasoning. Additionally, we introduce two LLM-free heuristics for premise prioritization, enabling strategic proof search. Experimental results on five datasets demonstrate that LogicTree optimally scales inference-time computation to achieve higher proof accuracy, surpassing chain-of-thought (CoT) and ToT with average gains of 23.6% and 12.5%, respectively, on GPT-4o. Moreover, within LogicTree, GPT-4o outperforms o3-mini by 7.6% on average. 

---
# A Baseline for Self-state Identification and Classification in Mental Health Data: CLPsych 2025 Task 

**Authors**: Laerdon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.14066)  

**Abstract**: We present a baseline for the CLPsych 2025 A.1 task: classifying self-states in mental health data taken from Reddit. We use few-shot learning with a 4-bit quantized Gemma 2 9B model and a data preprocessing step which first identifies relevant sentences indicating self-state evidence, and then performs a binary classification to determine whether the sentence is evidence of an adaptive or maladaptive self-state. This system outperforms our other method which relies on an LLM to highlight spans of variable length independently. We attribute the performance of our model to the benefits of this sentence chunking step for two reasons: partitioning posts into sentences 1) broadly matches the granularity at which self-states were human-annotated and 2) simplifies the task for our language model to a binary classification problem. Our system places third out of fourteen systems submitted for Task A.1, achieving a test-time recall of 0.579. 

---
# MEQA: A Meta-Evaluation Framework for Question & Answer LLM Benchmarks 

**Authors**: Jaime Raldua Veuthey, Zainab Ali Majid, Suhas Hariharan, Jacob Haimes  

**Link**: [PDF](https://arxiv.org/pdf/2504.14039)  

**Abstract**: As Large Language Models (LLMs) advance, their potential for widespread societal impact grows simultaneously. Hence, rigorous LLM evaluations are both a technical necessity and social imperative. While numerous evaluation benchmarks have been developed, there remains a critical gap in meta-evaluation: effectively assessing benchmarks' quality. We propose MEQA, a framework for the meta-evaluation of question and answer (QA) benchmarks, to provide standardized assessments, quantifiable scores, and enable meaningful intra-benchmark comparisons. We demonstrate this approach on cybersecurity benchmarks, using human and LLM evaluators, highlighting the benchmarks' strengths and weaknesses. We motivate our choice of test domain by AI models' dual nature as powerful defensive tools and security threats. 

---
# Uncovering Conspiratorial Narratives within Arabic Online Content 

**Authors**: Djamila Mohdeb, Meriem Laifa, Zineb Guemraoui, Dalila Behih  

**Link**: [PDF](https://arxiv.org/pdf/2504.14037)  

**Abstract**: This study investigates the spread of conspiracy theories in Arabic digital spaces through computational analysis of online content. By combining Named Entity Recognition and Topic Modeling techniques, specifically the Top2Vec algorithm, we analyze data from Arabic blogs and Facebook to identify and classify conspiratorial narratives. Our analysis uncovers six distinct categories: gender/feminist, geopolitical, government cover-ups, apocalyptic, Judeo-Masonic, and geoengineering. The research highlights how these narratives are deeply embedded in Arabic social media discourse, shaped by regional historical, cultural, and sociopolitical contexts. By applying advanced Natural Language Processing methods to Arabic content, this study addresses a gap in conspiracy theory research, which has traditionally focused on English-language content or offline data. The findings provide new insights into the manifestation and evolution of conspiracy theories in Arabic digital spaces, enhancing our understanding of their role in shaping public discourse in the Arab world. 

---
# Seed-Thinking-v1.5: Advancing Superb Reasoning Models with Reinforcement Learning 

**Authors**: ByteDance Seed, Yufeng Yuan, Yu Yue, Mingxuan Wang, Xiaochen Zuo, Jiaze Chen, Lin Yan, Wenyuan Xu, Chi Zhang, Xin Liu, Chengyi Wang, TianTian Fan, Lingjun Liu, Qiying Yu, Xiangpeng Wei, Zhiqi Lin, Ruofei Zhu, Qingping Yang, Chengzhi Wei, Jerry He, Guanlin Liu, Zheng Wu, Xiangyu Yu, Zhicheng Liu, Jingjing Xu, Jiangjie Chen, Haojie Pan, Shengding Hu, Zhengyin Du, Wenqi Wang, Zewei Sun, Chenwei Lou, Bole Ma, Zihan Wang, Mofan Zhang, Wang Zhang, Gaohong Liu, Kaihua Jiang, Haibin Lin, Ru Zhang, Juncai Liu, Li Han, Jinxin Chi, Wenqiang Zhang, Jiayi Xu, Jun Yuan, Zhen Xiao, Yuqiao Xian, Jingqiao Wu, Kai Hua, Na Zhou, Jianhui Duan, Heyang Lu, Changbao Wang, Jinxiang Ou, Shihang Wang, Xiaoran Jin, Xuesong Yao, Chengyin Xu, Wenchang Ma, Zhecheng An, Renming Pang, Xia Xiao, Jing Su, Yuyu Zhang, Tao Sun, Kaibo Liu, Yifan Sun, Kai Shen, Sijun Zhang, Yiyuan Ma, Xingyan Bin, Ji Li, Yao Luo, Deyi Liu, Shiyi Zhan, Yunshui Li, Yuan Yang, Defa Zhu, Ke Shen, Chenggang Li, Xun Zhou, Liang Xiang, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13914)  

**Abstract**: We introduce Seed-Thinking-v1.5, capable of reasoning through thinking before responding, resulting in improved performance on a wide range of benchmarks. Seed-Thinking-v1.5 achieves 86.7 on AIME 2024, 55.0 on Codeforces and 77.3 on GPQA, demonstrating excellent reasoning abilities in STEM and coding. Beyond reasoning tasks, the method demonstrates notable generalization across diverse domains. For instance, it surpasses DeepSeek R1 by 8% in win rate on non-reasoning tasks, indicating its broader applicability. Compared to other state-of-the-art reasoning models, Seed-Thinking-v1.5 is a Mixture-of-Experts (MoE) model with a relatively small size, featuring 20B activated and 200B total parameters. As part of our effort to assess generalized reasoning, we develop two internal benchmarks, BeyondAIME and Codeforces, both of which will be publicly released to support future research. 

---
# Seeing from Another Perspective: Evaluating Multi-View Understanding in MLLMs 

**Authors**: Chun-Hsiao Yeh, Chenyu Wang, Shengbang Tong, Ta-Ying Cheng, Rouyu Wang, Tianzhe Chu, Yuexiang Zhai, Yubei Chen, Shenghua Gao, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.15280)  

**Abstract**: Multi-view understanding, the ability to reconcile visual information across diverse viewpoints for effective navigation, manipulation, and 3D scene comprehension, is a fundamental challenge in Multi-Modal Large Language Models (MLLMs) to be used as embodied agents. While recent MLLMs have shown impressive advances in high-level reasoning and planning, they frequently fall short when confronted with multi-view geometric consistency and cross-view correspondence. To comprehensively evaluate the challenges of MLLMs in multi-view scene reasoning, we propose All-Angles Bench, a benchmark of over 2,100 human carefully annotated multi-view question-answer pairs across 90 diverse real-world scenes. Our six tasks (counting, attribute identification, relative distance, relative direction, object manipulation, and camera pose estimation) specifically test model's geometric correspondence and the capacity to align information consistently across views. Our extensive experiments, benchmark on 27 representative MLLMs including Gemini-2.0-Flash, Claude-3.7-Sonnet, and GPT-4o against human evaluators reveals a substantial performance gap, indicating that current MLLMs remain far from human-level proficiency. Through in-depth analysis, we show that MLLMs are particularly underperforming under two aspects: (1) cross-view correspondence for partially occluded views and (2) establishing the coarse camera poses. These findings highlight the necessity of domain-specific refinements or modules that embed stronger multi-view awareness. We believe that our All-Angles Bench offers valuable insights and contribute to bridging the gap between MLLMs and human-level multi-view understanding. The project and benchmark are publicly available at this https URL. 

---
# An LMM for Efficient Video Understanding via Reinforced Compression of Video Cubes 

**Authors**: Ji Qi, Yuan Yao, Yushi Bai, Bin Xu, Juanzi Li, Zhiyuan Liu, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.15270)  

**Abstract**: Large Multimodal Models (LMMs) uniformly perceive video frames, creating computational inefficiency for videos with inherently varying temporal information density. This paper present \textbf{Quicksviewer}, an LMM with new perceiving paradigm that partitions a video of nonuniform density into varying cubes using Gumbel Softmax, followed by a unified resampling for each cube to achieve efficient video understanding. This simple and intuitive approach dynamically compress video online based on its temporal density, significantly reducing spatiotemporal redundancy (overall 45$\times$ compression rate), while enabling efficient training with large receptive field. We train the model from a language backbone through three progressive stages, each incorporating lengthy videos on average of 420s/1fps thanks to the perceiving efficiency. With only 0.8M total video-text samples for training, our model outperforms the direct baseline employing a fixed partitioning strategy by a maximum of 8.72 in accuracy, demonstrating the effectiveness in performance. On Video-MME, Quicksviewer achieves SOTA under modest sequence lengths using just up to 5\% of tokens per frame required by baselines. With this paradigm, scaling up the number of input frames reveals a clear power law of the model capabilities. It is also empirically verified that the segments generated by the cubing network can help for analyzing continuous events in videos. 

---
# Roll the dice & look before you leap: Going beyond the creative limits of next-token prediction 

**Authors**: Vaishnavh Nagarajan, Chen Henry Wu, Charles Ding, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15266)  

**Abstract**: We design a suite of minimal algorithmic tasks that are a loose abstraction of open-ended real-world tasks. This allows us to cleanly and controllably quantify the creative limits of the present-day language model. Much like real-world tasks that require a creative, far-sighted leap of thought, our tasks require an implicit, open-ended stochastic planning step that either (a) discovers new connections in an abstract knowledge graph (like in wordplay, drawing analogies, or research) or (b) constructs new patterns (like in designing math problems or new proteins). In these tasks, we empirically and conceptually argue how next-token learning is myopic and memorizes excessively; comparatively, multi-token approaches, namely teacherless training and diffusion models, excel in producing diverse and original output. Secondly, in our tasks, we find that to elicit randomness from the Transformer without hurting coherence, it is better to inject noise right at the input layer (via a method we dub hash-conditioning) rather than defer to temperature sampling from the output layer. Thus, our work offers a principled, minimal test-bed for analyzing open-ended creative skills, and offers new arguments for going beyond next-token learning and softmax-based sampling. We make part of the code available under this https URL 

---
# CRUST-Bench: A Comprehensive Benchmark for C-to-safe-Rust Transpilation 

**Authors**: Anirudh Khatry, Robert Zhang, Jia Pan, Ziteng Wang, Qiaochu Chen, Greg Durrett, Isil Dillig  

**Link**: [PDF](https://arxiv.org/pdf/2504.15254)  

**Abstract**: C-to-Rust transpilation is essential for modernizing legacy C code while enhancing safety and interoperability with modern Rust ecosystems. However, no dataset currently exists for evaluating whether a system can transpile C into safe Rust that passes a set of test cases. We introduce CRUST-Bench, a dataset of 100 C repositories, each paired with manually-written interfaces in safe Rust as well as test cases that can be used to validate correctness of the transpilation. By considering entire repositories rather than isolated functions, CRUST-Bench captures the challenges of translating complex projects with dependencies across multiple files. The provided Rust interfaces provide explicit specifications that ensure adherence to idiomatic, memory-safe Rust patterns, while the accompanying test cases enforce functional correctness. We evaluate state-of-the-art large language models (LLMs) on this task and find that safe and idiomatic Rust generation is still a challenging problem for various state-of-the-art methods and techniques. We also provide insights into the errors LLMs usually make in transpiling code from C to safe Rust. The best performing model, OpenAI o1, is able to solve only 15 tasks in a single-shot setting. Improvements on CRUST-Bench would lead to improved transpilation systems that can reason about complex scenarios and help in migrating legacy codebases from C into languages like Rust that ensure memory safety. You can find the dataset and code at this https URL. 

---
# KGMEL: Knowledge Graph-Enhanced Multimodal Entity Linking 

**Authors**: Juyeon Kim, Geon Lee, Taeuk Kim, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15135)  

**Abstract**: Entity linking (EL) aligns textual mentions with their corresponding entities in a knowledge base, facilitating various applications such as semantic search and question answering. Recent advances in multimodal entity linking (MEL) have shown that combining text and images can reduce ambiguity and improve alignment accuracy. However, most existing MEL methods overlook the rich structural information available in the form of knowledge-graph (KG) triples. In this paper, we propose KGMEL, a novel framework that leverages KG triples to enhance MEL. Specifically, it operates in three stages: (1) Generation: Produces high-quality triples for each mention by employing vision-language models based on its text and images. (2) Retrieval: Learns joint mention-entity representations, via contrastive learning, that integrate text, images, and (generated or KG) triples to retrieve candidate entities for each mention. (3) Reranking: Refines the KG triples of the candidate entities and employs large language models to identify the best-matching entity for the mention. Extensive experiments on benchmark datasets demonstrate that KGMEL outperforms existing methods. Our code and datasets are available at: this https URL. 

---
# Rhythm of Opinion: A Hawkes-Graph Framework for Dynamic Propagation Analysis 

**Authors**: Yulong Li, Zhixiang Lu, Feilong Tang, Simin Lai, Ming Hu, Yuxuan Zhang, Haochen Xue, Zhaodong Wu, Imran Razzak, Qingxia Li, Jionglong Su  

**Link**: [PDF](https://arxiv.org/pdf/2504.15072)  

**Abstract**: The rapid development of social media has significantly reshaped the dynamics of public opinion, resulting in complex interactions that traditional models fail to effectively capture. To address this challenge, we propose an innovative approach that integrates multi-dimensional Hawkes processes with Graph Neural Network, modeling opinion propagation dynamics among nodes in a social network while considering the intricate hierarchical relationships between comments. The extended multi-dimensional Hawkes process captures the hierarchical structure, multi-dimensional interactions, and mutual influences across different topics, forming a complex propagation network. Moreover, recognizing the lack of high-quality datasets capable of comprehensively capturing the evolution of public opinion dynamics, we introduce a new dataset, VISTA. It includes 159 trending topics, corresponding to 47,207 posts, 327,015 second-level comments, and 29,578 third-level comments, covering diverse domains such as politics, entertainment, sports, health, and medicine. The dataset is annotated with detailed sentiment labels across 11 categories and clearly defined hierarchical relationships. When combined with our method, it offers strong interpretability by linking sentiment propagation to the comment hierarchy and temporal evolution. Our approach provides a robust baseline for future research. 

---
# The Great Nugget Recall: Automating Fact Extraction and RAG Evaluation with Large Language Models 

**Authors**: Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick Craswell, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15068)  

**Abstract**: Large Language Models (LLMs) have significantly enhanced the capabilities of information access systems, especially with retrieval-augmented generation (RAG). Nevertheless, the evaluation of RAG systems remains a barrier to continued progress, a challenge we tackle in this work by proposing an automatic evaluation framework that is validated against human annotations. We believe that the nugget evaluation methodology provides a solid foundation for evaluating RAG systems. This approach, originally developed for the TREC Question Answering (QA) Track in 2003, evaluates systems based on atomic facts that should be present in good answers. Our efforts focus on "refactoring" this methodology, where we describe the AutoNuggetizer framework that specifically applies LLMs to both automatically create nuggets and automatically assign nuggets to system answers. In the context of the TREC 2024 RAG Track, we calibrate a fully automatic approach against strategies where nuggets are created manually or semi-manually by human assessors and then assigned manually to system answers. Based on results from a community-wide evaluation, we observe strong agreement at the run level between scores derived from fully automatic nugget evaluation and human-based variants. The agreement is stronger when individual framework components such as nugget assignment are automated independently. This suggests that our evaluation framework provides tradeoffs between effort and quality that can be used to guide the development of future RAG systems. However, further research is necessary to refine our approach, particularly in establishing robust per-topic agreement to diagnose system failures effectively. 

---
# Learning to Reason under Off-Policy Guidance 

**Authors**: Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14945)  

**Abstract**: Recent advances in large reasoning models (LRMs) demonstrate that sophisticated behaviors such as multi-step reasoning and self-reflection can emerge via reinforcement learning (RL) with simple rule-based rewards. However, existing zero-RL approaches are inherently ``on-policy'', limiting learning to a model's own outputs and failing to acquire reasoning abilities beyond its initial capabilities. We introduce LUFFY (Learning to reason Under oFF-policY guidance), a framework that augments zero-RL with off-policy reasoning traces. LUFFY dynamically balances imitation and exploration by combining off-policy demonstrations with on-policy rollouts during training. Notably, we propose policy shaping via regularized importance sampling to avoid superficial and rigid imitation during mixed-policy training. Remarkably, LUFFY achieves an over +7.0 average gain across six math benchmarks and an advantage of over +6.2 points in out-of-distribution tasks. It also substantially surpasses imitation-based supervised fine-tuning (SFT), particularly in generalization. Analysis shows LUFFY not only imitates effectively but also explores beyond demonstrations, offering a scalable path to train generalizable reasoning models with off-policy guidance. 

---
# EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework 

**Authors**: Yao Shi, Rongkeng Liang, Yong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14928)  

**Abstract**: Large language models (LLMs) increasingly serve as educational tools, yet evaluating their teaching capabilities remains challenging due to the resource-intensive, context-dependent, and methodologically complex nature of teacher-student interactions. We introduce EducationQ, a multi-agent dialogue framework that efficiently assesses teaching capabilities through simulated dynamic educational scenarios, featuring specialized agents for teaching, learning, and evaluation. Testing 14 LLMs across major AI Organizations (OpenAI, Meta, Google, Anthropic, and others) on 1,498 questions spanning 13 disciplines and 10 difficulty levels reveals that teaching effectiveness does not correlate linearly with model scale or general reasoning capabilities - with some smaller open-source models outperforming larger commercial counterparts in teaching contexts. This finding highlights a critical gap in current evaluations that prioritize knowledge recall over interactive pedagogy. Our mixed-methods evaluation, combining quantitative metrics with qualitative analysis and expert case studies, identifies distinct pedagogical strengths employed by top-performing models (e.g., sophisticated questioning strategies, adaptive feedback mechanisms). Human expert evaluations show 78% agreement with our automated qualitative analysis of effective teaching behaviors, validating our methodology. EducationQ demonstrates that LLMs-as-teachers require specialized optimization beyond simple scaling, suggesting next-generation educational AI prioritize targeted enhancement of specific pedagogical effectiveness. 

---
# VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform 

**Authors**: Xingyu Lu, Tianke Zhang, Chang Meng, Xiaobei Wang, Jinpeng Wang, YiFan Zhang, Shisong Tang, Changyi Liu, Haojie Ding, Kaiyu Jiang, Kaiyu Tang, Bin Wen, Hai-Tao Zheng, Fan Yang, Tingting Gao, Di Zhang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2504.14904)  

**Abstract**: Exponentially growing short video platforms (SVPs) face significant challenges in moderating content detrimental to users' mental health, particularly for minors. The dissemination of such content on SVPs can lead to catastrophic societal consequences. Although substantial efforts have been dedicated to moderating such content, existing methods suffer from critical limitations: (1) Manual review is prone to human bias and incurs high operational costs. (2) Automated methods, though efficient, lack nuanced content understanding, resulting in lower accuracy. (3) Industrial moderation regulations struggle to adapt to rapidly evolving trends due to long update cycles. In this paper, we annotate the first SVP content moderation benchmark with authentic user/reviewer feedback to fill the absence of benchmark in this field. Then we evaluate various methods on the benchmark to verify the existence of the aforementioned limitations. We further propose our common-law content moderation framework named KuaiMod to address these challenges. KuaiMod consists of three components: training data construction, offline adaptation, and online deployment & refinement. Leveraging large vision language model (VLM) and Chain-of-Thought (CoT) reasoning, KuaiMod adequately models video toxicity based on sparse user feedback and fosters dynamic moderation policy with rapid update speed and high accuracy. Offline experiments and large-scale online A/B test demonstrates the superiority of KuaiMod: KuaiMod achieves the best moderation performance on our benchmark. The deployment of KuaiMod reduces the user reporting rate by 20% and its application in video recommendation increases both Daily Active User (DAU) and APP Usage Time (AUT) on several Kuaishou scenarios. We have open-sourced our benchmark at this https URL. 

---
# OTC: Optimal Tool Calls via Reinforcement Learning 

**Authors**: Hongru Wang, Cheng Qian, Wanjun Zhong, Xiusi Chen, Jiahao Qiu, Shijue Huang, Bowen Jin, Mengdi Wang, Kam-Fai Wong, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.14870)  

**Abstract**: Tool-integrated reasoning (TIR) augments large language models (LLMs) with the ability to invoke external tools, such as search engines and code interpreters, to solve tasks beyond the capabilities of language-only reasoning. While reinforcement learning (RL) has shown promise in improving TIR by optimizing final answer correctness, existing approaches often overlook the efficiency and cost associated with tool usage. This can lead to suboptimal behavior, including excessive tool calls that increase computational and financial overhead, or insufficient tool use that compromises answer quality. In this work, we propose Optimal Tool Call-controlled Policy Optimization (OTC-PO), a simple yet effective RL-based framework that encourages models to produce accurate answers with minimal tool calls. Our method introduces a tool-integrated reward that jointly considers correctness and tool efficiency, promoting high tool productivity. We instantiate this framework within both Proximal Policy Optimization (PPO) and Group Relative Preference Optimization (GRPO), resulting in OTC-PPO and OTC-GRPO. Experiments with Qwen-2.5 and Qwen-Math across multiple QA benchmarks show that our approach reduces tool calls by up to 73.1\% and improves tool productivity by up to 229.4\%, while maintaining comparable answer accuracy. To the best of our knowledge, this is the first RL-based framework that explicitly optimizes tool-use efficiency in TIR. 

---
# AlignRAG: An Adaptable Framework for Resolving Misalignments in Retrieval-Aware Reasoning of RAG 

**Authors**: Jiaqi Wei, Hao Zhou, Xiang Zhang, Di Zhang, Zijie Qiu, Wei Wei, Jinzhe Li, Wanli Ouyang, Siqi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.14858)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a foundational paradigm for knowledge-grounded text generation. However, existing RAG pipelines often fail to ensure that the reasoning trajectories align with the evidential constraints imposed by retrieved content. In this paper, we reframe RAG as a problem of retrieval-aware reasoning and identify a core challenge: reasoning misalignment-the mismatch between a model's reasoning trajectory and the retrieved evidence. To address this challenge, we propose AlignRAG, a novel test-time framework that mitigates reasoning misalignment through iterative Critique-Driven Alignment (CDA) steps. In contrast to prior approaches that rely on static training or post-hoc selection, AlignRAG actively refines reasoning trajectories during inference by enforcing fine-grained alignment with evidence. Our framework introduces a new paradigm for retrieval-aware reasoning by: (1) constructing context-rich training corpora; (2) generating contrastive critiques from preference-aware reasoning trajectories; (3) training a dedicated \textit{Critic Language Model (CLM)} to identify reasoning misalignments; and (4) applying CDA steps to optimize reasoning trajectories iteratively. Empirical results demonstrate that AlignRAG consistently outperforms all baselines and could integrate as a plug-and-play module into existing RAG pipelines without further changes. By reconceptualizing RAG as a structured reasoning trajectory and establishing the test-time framework for correcting reasoning misalignments in RAG, AlignRAG provides practical advancements for retrieval-aware generation. 

---
# Completing A Systematic Review in Hours instead of Months with Interactive AI Agents 

**Authors**: Rui Qiu, Shijie Chen, Yu Su, Po-Yin Yen, Han-Wei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14822)  

**Abstract**: Systematic reviews (SRs) are vital for evidence-based practice in high stakes disciplines, such as healthcare, but are often impeded by intensive labors and lengthy processes that can take months to complete. Due to the high demand for domain expertise, existing automatic summarization methods fail to accurately identify relevant studies and generate high-quality summaries. To that end, we introduce InsightAgent, a human-centered interactive AI agent powered by large language models that revolutionize this workflow. InsightAgent partitions a large literature corpus based on semantics and employs a multi-agent design for more focused processing of literature, leading to significant improvement in the quality of generated SRs. InsightAgent also provides intuitive visualizations of the corpus and agent trajectories, allowing users to effortlessly monitor the actions of the agent and provide real-time feedback based on their expertise. Our user studies with 9 medical professionals demonstrate that the visualization and interaction mechanisms can effectively improve the quality of synthesized SRs by 27.2%, reaching 79.7% of human-written quality. At the same time, user satisfaction is improved by 34.4%. With InsightAgent, it only takes a clinician about 1.5 hours, rather than months, to complete a high-quality systematic review. 

---
# PLANET: A Collection of Benchmarks for Evaluating LLMs' Planning Capabilities 

**Authors**: Haoming Li, Zhaoliang Chen, Jonathan Zhang, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14773)  

**Abstract**: Planning is central to agents and agentic AI. The ability to plan, e.g., creating travel itineraries within a budget, holds immense potential in both scientific and commercial contexts. Moreover, optimal plans tend to require fewer resources compared to ad-hoc methods. To date, a comprehensive understanding of existing planning benchmarks appears to be lacking. Without it, comparing planning algorithms' performance across domains or selecting suitable algorithms for new scenarios remains challenging. In this paper, we examine a range of planning benchmarks to identify commonly used testbeds for algorithm development and highlight potential gaps. These benchmarks are categorized into embodied environments, web navigation, scheduling, games and puzzles, and everyday task automation. Our study recommends the most appropriate benchmarks for various algorithms and offers insights to guide future benchmark development. 

---
# LeetCodeDataset: A Temporal Dataset for Robust Evaluation and Efficient Training of Code LLMs 

**Authors**: Yunhui Xia, Wei Shen, Yan Wang, Jason Klein Liu, Huifeng Sun, Siyue Wu, Jian Hu, Xiaolong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14655)  

**Abstract**: We introduce LeetCodeDataset, a high-quality benchmark for evaluating and training code-generation models, addressing two key challenges in LLM research: the lack of reasoning-focused coding benchmarks and self-contained training testbeds. By curating LeetCode Python problems with rich metadata, broad coverage, 100+ test cases per problem, and temporal splits (pre/post July 2024), our dataset enables contamination-free evaluation and efficient supervised fine-tuning (SFT). Experiments show reasoning models significantly outperform non-reasoning counterparts, while SFT with only 2.6K model-generated solutions achieves performance comparable to 110K-sample counterparts. The dataset and evaluation framework are available on Hugging Face and Github. 

---
# Risk Assessment Framework for Code LLMs via Leveraging Internal States 

**Authors**: Yuheng Huang, Lei Ma, Keizaburo Nishikino, Takumi Akazaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.14640)  

**Abstract**: The pre-training paradigm plays a key role in the success of Large Language Models (LLMs), which have been recognized as one of the most significant advancements of AI recently. Building on these breakthroughs, code LLMs with advanced coding capabilities bring huge impacts on software engineering, showing the tendency to become an essential part of developers' daily routines. However, the current code LLMs still face serious challenges related to trustworthiness, as they can generate incorrect, insecure, or unreliable code. Recent exploratory studies find that it can be promising to detect such risky outputs by analyzing LLMs' internal states, akin to how the human brain unconsciously recognizes its own mistakes. Yet, most of these approaches are limited to narrow sub-domains of LLM operations and fall short of achieving industry-level scalability and practicability. To address these challenges, in this paper, we propose PtTrust, a two-stage risk assessment framework for code LLM based on internal state pre-training, designed to integrate seamlessly with the existing infrastructure of software companies. The core idea is that the risk assessment framework could also undergo a pre-training process similar to LLMs. Specifically, PtTrust first performs unsupervised pre-training on large-scale unlabeled source code to learn general representations of LLM states. Then, it uses a small, labeled dataset to train a risk predictor. We demonstrate the effectiveness of PtTrust through fine-grained, code line-level risk assessment and demonstrate that it generalizes across tasks and different programming languages. Further experiments also reveal that PtTrust provides highly intuitive and interpretable features, fostering greater user trust. We believe PtTrust makes a promising step toward scalable and trustworthy assurance for code LLMs. 

---
# HealthGenie: Empowering Users with Healthy Dietary Guidance through Knowledge Graph and Large Language Models 

**Authors**: Fan Gao, Xinjie Zhao, Ding Xia, Zhongyi Zhou, Rui Yang, Jinghui Lu, Hang Jiang, Chanjun Park, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14594)  

**Abstract**: Seeking dietary guidance often requires navigating complex professional knowledge while accommodating individual health conditions. Knowledge Graphs (KGs) offer structured and interpretable nutritional information, whereas Large Language Models (LLMs) naturally facilitate conversational recommendation delivery. In this paper, we present HealthGenie, an interactive system that combines the strengths of LLMs and KGs to provide personalized dietary recommendations along with hierarchical information visualization for a quick and intuitive overview. Upon receiving a user query, HealthGenie performs query refinement and retrieves relevant information from a pre-built KG. The system then visualizes and highlights pertinent information, organized by defined categories, while offering detailed, explainable recommendation rationales. Users can further tailor these recommendations by adjusting preferences interactively. Our evaluation, comprising a within-subject comparative experiment and an open-ended discussion, demonstrates that HealthGenie effectively supports users in obtaining personalized dietary guidance based on their health conditions while reducing interaction effort and cognitive load. These findings highlight the potential of LLM-KG integration in supporting decision-making through explainable and visualized information. We examine the system's usefulness and effectiveness with an N=12 within-subject study and provide design considerations for future systems that integrate conversational LLM and KG. 

---
# Are Vision LLMs Road-Ready? A Comprehensive Benchmark for Safety-Critical Driving Video Understanding 

**Authors**: Tong Zeng, Longfeng Wu, Liang Shi, Dawei Zhou, Feng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14526)  

**Abstract**: Vision Large Language Models (VLLMs) have demonstrated impressive capabilities in general visual tasks such as image captioning and visual question answering. However, their effectiveness in specialized, safety-critical domains like autonomous driving remains largely unexplored. Autonomous driving systems require sophisticated scene understanding in complex environments, yet existing multimodal benchmarks primarily focus on normal driving conditions, failing to adequately assess VLLMs' performance in safety-critical scenarios. To address this, we introduce DVBench, a pioneering benchmark designed to evaluate the performance of VLLMs in understanding safety-critical driving videos. Built around a hierarchical ability taxonomy that aligns with widely adopted frameworks for describing driving scenarios used in assessing highly automated driving systems, DVBench features 10,000 multiple-choice questions with human-annotated ground-truth answers, enabling a comprehensive evaluation of VLLMs' capabilities in perception and reasoning. Experiments on 14 SOTA VLLMs, ranging from 0.5B to 72B parameters, reveal significant performance gaps, with no model achieving over 40% accuracy, highlighting critical limitations in understanding complex driving scenarios. To probe adaptability, we fine-tuned selected models using domain-specific data from DVBench, achieving accuracy gains ranging from 5.24 to 10.94 percentage points, with relative improvements of up to 43.59%. This improvement underscores the necessity of targeted adaptation to bridge the gap between general-purpose VLLMs and mission-critical driving applications. DVBench establishes an essential evaluation framework and research roadmap for developing VLLMs that meet the safety and robustness requirements for real-world autonomous systems. We released the benchmark toolbox and the fine-tuned model at: this https URL. 

---
# Meta-Thinking in LLMs via Multi-Agent Reinforcement Learning: A Survey 

**Authors**: Ahsan Bilal, Muhammad Ahmed Mohsin, Muhammad Umer, Muhammad Awais Khan Bangash, Muhammad Ali Jamshed  

**Link**: [PDF](https://arxiv.org/pdf/2504.14520)  

**Abstract**: This survey explores the development of meta-thinking capabilities in Large Language Models (LLMs) from a Multi-Agent Reinforcement Learning (MARL) perspective. Meta-thinking self-reflection, assessment, and control of thinking processes is an important next step in enhancing LLM reliability, flexibility, and performance, particularly for complex or high-stakes tasks. The survey begins by analyzing current LLM limitations, such as hallucinations and the lack of internal self-assessment mechanisms. It then talks about newer methods, including RL from human feedback (RLHF), self-distillation, and chain-of-thought prompting, and each of their limitations. The crux of the survey is to talk about how multi-agent architectures, namely supervisor-agent hierarchies, agent debates, and theory of mind frameworks, can emulate human-like introspective behavior and enhance LLM robustness. By exploring reward mechanisms, self-play, and continuous learning methods in MARL, this survey gives a comprehensive roadmap to building introspective, adaptive, and trustworthy LLMs. Evaluation metrics, datasets, and future research avenues, including neuroscience-inspired architectures and hybrid symbolic reasoning, are also discussed. 

---
# LoRe: Personalizing LLMs via Low-Rank Reward Modeling 

**Authors**: Avinandan Bose, Zhihan Xiong, Yuejie Chi, Simon Shaolei Du, Lin Xiao, Maryam Fazel  

**Link**: [PDF](https://arxiv.org/pdf/2504.14439)  

**Abstract**: Personalizing large language models (LLMs) to accommodate diverse user preferences is essential for enhancing alignment and user satisfaction. Traditional reinforcement learning from human feedback (RLHF) approaches often rely on monolithic value representations, limiting their ability to adapt to individual preferences. We introduce a novel framework that leverages low-rank preference modeling to efficiently learn and generalize user-specific reward functions. By representing reward functions in a low-dimensional subspace and modeling individual preferences as weighted combinations of shared basis functions, our approach avoids rigid user categorization while enabling scalability and few-shot adaptation. We validate our method on multiple preference datasets, demonstrating superior generalization to unseen users and improved accuracy in preference prediction tasks. 

---
# Density Measures for Language Generation 

**Authors**: Jon Kleinberg, Fan Wei  

**Link**: [PDF](https://arxiv.org/pdf/2504.14370)  

**Abstract**: The recent successes of large language models (LLMs) have led to a surge of theoretical research into language generation. A recent line of work proposes an abstract view, called language generation in the limit, where generation is seen as a game between an adversary and an algorithm: the adversary generates strings from an unknown language $K$, chosen from a countable collection of candidate languages, and after seeing a finite set of these strings, the algorithm must generate new strings from $K$ that it has not seen before. This formalism highlights a key tension: the trade-off between validity (the algorithm should only produce strings from the language) and breadth (it should be able to produce many strings from the language). This trade-off is central in applied language generation as well, where it appears as a balance between hallucination (generating invalid utterances) and mode collapse (generating only a restricted set of outputs). Despite its importance, this trade-off has been challenging to study quantitatively. We develop ways to quantify this trade-off by formalizing breadth using measures of density. Existing algorithms for language generation in the limit produce output sets that can have zero density in the true language, and this important failure of breadth might seem unavoidable. We show, however, that such a failure is not necessary: we provide an algorithm for language generation in the limit whose outputs have strictly positive density in $K$. We also study the internal representations built by these algorithms, specifically the sequence of hypothesized candidate languages they consider, and show that achieving the strongest form of breadth may require oscillating indefinitely between high- and low-density representations. Our analysis introduces a novel topology on language families, with notions of convergence and limit points playing a key role. 

---
# Improving RL Exploration for LLM Reasoning through Retrospective Replay 

**Authors**: Shihan Dou, Muling Wu, Jingwen Xu, Rui Zheng, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14363)  

**Abstract**: Reinforcement learning (RL) has increasingly become a pivotal technique in the post-training of large language models (LLMs). The effective exploration of the output space is essential for the success of RL. We observe that for complex problems, during the early stages of training, the model exhibits strong exploratory capabilities and can identify promising solution ideas. However, its limited capability at this stage prevents it from successfully solving these problems. The early suppression of these potentially valuable solution ideas by the policy gradient hinders the model's ability to revisit and re-explore these ideas later. Consequently, although the LLM's capabilities improve in the later stages of training, it still struggles to effectively address these complex problems. To address this exploration issue, we propose a novel algorithm named Retrospective Replay-based Reinforcement Learning (RRL), which introduces a dynamic replay mechanism throughout the training process. RRL enables the model to revisit promising states identified in the early stages, thereby improving its efficiency and effectiveness in exploration. To evaluate the effectiveness of RRL, we conduct extensive experiments on complex reasoning tasks, including mathematical reasoning and code generation, and general dialogue tasks. The results indicate that RRL maintains high exploration efficiency throughout the training period, significantly enhancing the effectiveness of RL in optimizing LLMs for complicated reasoning tasks. Moreover, it also improves the performance of RLHF, making the model both safer and more helpful. 

---
# Integrating Single-Cell Foundation Models with Graph Neural Networks for Drug Response Prediction 

**Authors**: Till Rossner, Ziteng Li, Jonas Balke, Nikoo Salehfard, Tom Seifert, Ming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14361)  

**Abstract**: In this study, we propose an innovative methodology for predicting Cancer Drug Response (CDR) through the integration of the scGPT foundation model within the DeepCDR model. Our approach utilizes scGPT to generate embeddings from gene expression data, which are then used as gene expression input data for DeepCDR. The experimental findings demonstrate the efficacy of this scGPT-based method in outperforming previous related works, including the original DeepCDR model and the scFoundation-based model. This study highlights the potential of scGPT embeddings to enhance the accuracy of CDR predictions and offers a promising alternative to existing approaches. 

---
# A Multimodal Recaptioning Framework to Account for Perceptual Diversity in Multilingual Vision-Language Modeling 

**Authors**: Kyle Buettner, Jacob Emmerson, Adriana Kovashka  

**Link**: [PDF](https://arxiv.org/pdf/2504.14359)  

**Abstract**: There are many ways to describe, name, and group objects when captioning an image. Differences are evident when speakers come from diverse cultures due to the unique experiences that shape perception. Machine translation of captions has pushed multilingual capabilities in vision-language models (VLMs), but data comes mainly from English speakers, indicating a perceptual bias and lack of model flexibility. In this work, we address this challenge and outline a data-efficient framework to instill multilingual VLMs with greater understanding of perceptual diversity. We specifically propose an LLM-based, multimodal recaptioning strategy that alters the object descriptions of English captions before translation. The greatest benefits are demonstrated in a targeted multimodal mechanism guided by native speaker data. By adding produced rewrites as augmentations in training, we improve on German and Japanese text-image retrieval cases studies (up to +3.5 mean recall overall, +4.7 on non-native error cases). We further propose a mechanism to analyze the specific object description differences across datasets, and we offer insights into cross-dataset and cross-language generalization. 

---
# Cross-attention for State-based model RWKV-7 

**Authors**: Liu Xiao, Li Zhiyuan, Lin Yueyu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14260)  

**Abstract**: We introduce CrossWKV, a novel cross-attention mechanism for the state-based RWKV-7 model, designed to enhance the expressive power of text-to-image generation. Leveraging RWKV-7's linear-complexity Weighted Key-Value (WKV) architecture, CrossWKV integrates text and image modalities in a single pass, utilizing a generalized delta rule with vector-valued gating and low-rank adaptations (LoRA) to achieve superior cross-modal alignment. Unlike Transformer-based models, CrossWKV's non-diagonal, input-dependent transition matrix enables it to represent complex functions beyond the $\mathrm{TC}^0$ complexity class, including all regular languages, as demonstrated by its ability to perform state-tracking tasks like $S_5$ permutation modeling. Evaluated within the Diffusion in RWKV-7 (DIR-7) on datasets such as LAION-5B and ImageNet, CrossWKV achieves a Frechet Inception Distance (FID) of 2.88 and a CLIP score of 0.33 on ImageNet 256x256, matching state-of-the-art performance while offering robust generalization across diverse prompts. The model's enhanced expressivity, combined with constant memory usage and linear scaling, positions it as a powerful solution for advanced cross-modal tasks, with potential applications in high-resolution generation and dynamic state this http URL at this https URL 

---
# Towards Explainable Fake Image Detection with Multi-Modal Large Language Models 

**Authors**: Yikun Ji, Yan Hong, Jiahui Zhan, Haoxing Chen, jun lan, Huijia Zhu, Weiqiang Wang, Liqing Zhang, Jianfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14245)  

**Abstract**: Progress in image generation raises significant public security concerns. We argue that fake image detection should not operate as a "black box". Instead, an ideal approach must ensure both strong generalization and transparency. Recent progress in Multi-modal Large Language Models (MLLMs) offers new opportunities for reasoning-based AI-generated image detection. In this work, we evaluate the capabilities of MLLMs in comparison to traditional detection methods and human evaluators, highlighting their strengths and limitations. Furthermore, we design six distinct prompts and propose a framework that integrates these prompts to develop a more robust, explainable, and reasoning-driven detection system. The code is available at this https URL. 

---
# InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners 

**Authors**: Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xiaotian Han, Shengyu Zhang, Hongxia Yang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14239)  

**Abstract**: Multimodal Large Language Models (MLLMs) have powered Graphical User Interface (GUI) Agents, showing promise in automating tasks on computing devices. Recent works have begun exploring reasoning in GUI tasks with encouraging results. However, many current approaches rely on manually designed reasoning templates, which may result in reasoning that is not sufficiently robust and adaptive for complex GUI environments. Meanwhile, some existing agents continue to operate as Reactive Actors, relying primarily on implicit reasoning that may lack sufficient depth for GUI tasks demanding planning and error recovery. We argue that advancing these agents requires a shift from reactive acting towards acting based on deliberate reasoning. To facilitate this transformation, we introduce InfiGUI-R1, an MLLM-based GUI agent developed through our Actor2Reasoner framework, a reasoning-centric, two-stage training approach designed to progressively evolve agents from Reactive Actors to Deliberative Reasoners. The first stage, Reasoning Injection, focuses on establishing a basic reasoner. We employ Spatial Reasoning Distillation to transfer cross-modal spatial reasoning capabilities from teacher models to MLLMs through trajectories with explicit reasoning steps, enabling models to integrate GUI visual-spatial information with logical reasoning before action generation. The second stage, Deliberation Enhancement, refines the basic reasoner into a deliberative one using Reinforcement Learning. This stage introduces two approaches: Sub-goal Guidance, which rewards models for generating accurate intermediate sub-goals, and Error Recovery Scenario Construction, which creates failure-and-recovery training scenarios from identified prone-to-error steps. Experimental results show InfiGUI-R1 achieves strong performance in GUI grounding and trajectory tasks. Resources at this https URL. 

---
# Assessing AI-Generated Questions' Alignment with Cognitive Frameworks in Educational Assessment 

**Authors**: Antoun Yaacoub, Jérôme Da-Rugna, Zainab Assaghir  

**Link**: [PDF](https://arxiv.org/pdf/2504.14232)  

**Abstract**: This study evaluates the integration of Bloom's Taxonomy into OneClickQuiz, an Artificial Intelligence (AI) driven plugin for automating Multiple-Choice Question (MCQ) generation in Moodle. Bloom's Taxonomy provides a structured framework for categorizing educational objectives into hierarchical cognitive levels. Our research investigates whether incorporating this taxonomy can improve the alignment of AI-generated questions with specific cognitive objectives. We developed a dataset of 3691 questions categorized according to Bloom's levels and employed various classification models-Multinomial Logistic Regression, Naive Bayes, Linear Support Vector Classification (SVC), and a Transformer-based model (DistilBERT)-to evaluate their effectiveness in categorizing questions. Our results indicate that higher Bloom's levels generally correlate with increased question length, Flesch-Kincaid Grade Level (FKGL), and Lexical Density (LD), reflecting the increased complexity of higher cognitive demands. Multinomial Logistic Regression showed varying accuracy across Bloom's levels, performing best for "Knowledge" and less accurately for higher-order levels. Merging higher-level categories improved accuracy for complex cognitive tasks. Naive Bayes and Linear SVC also demonstrated effective classification for lower levels but struggled with higher-order tasks. DistilBERT achieved the highest performance, significantly improving classification of both lower and higher-order cognitive levels, achieving an overall validation accuracy of 91%. This study highlights the potential of integrating Bloom's Taxonomy into AI-driven assessment tools and underscores the advantages of advanced models like DistilBERT for enhancing educational content generation. 

---
# AI Idea Bench 2025: AI Research Idea Generation Benchmark 

**Authors**: Yansheng Qiu, Haoquan Zhang, Zhaopan Xu, Ming Li, Diping Song, Zheng Wang, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14191)  

**Abstract**: Large-scale Language Models (LLMs) have revolutionized human-AI interaction and achieved significant success in the generation of novel ideas. However, current assessments of idea generation overlook crucial factors such as knowledge leakage in LLMs, the absence of open-ended benchmarks with grounded truth, and the limited scope of feasibility analysis constrained by prompt design. These limitations hinder the potential of uncovering groundbreaking research ideas. In this paper, we present AI Idea Bench 2025, a framework designed to quantitatively evaluate and compare the ideas generated by LLMs within the domain of AI research from diverse perspectives. The framework comprises a comprehensive dataset of 3,495 AI papers and their associated inspired works, along with a robust evaluation methodology. This evaluation system gauges idea quality in two dimensions: alignment with the ground-truth content of the original papers and judgment based on general reference material. AI Idea Bench 2025's benchmarking system stands to be an invaluable resource for assessing and comparing idea-generation techniques, thereby facilitating the automation of scientific discovery. 

---
# The First VoicePrivacy Attacker Challenge 

**Authors**: Natalia Tomashenko, Xiaoxiao Miao, Emmanuel Vincent, Junichi Yamagishi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14183)  

**Abstract**: The First VoicePrivacy Attacker Challenge is an ICASSP 2025 SP Grand Challenge which focuses on evaluating attacker systems against a set of voice anonymization systems submitted to the VoicePrivacy 2024 Challenge. Training, development, and evaluation datasets were provided along with a baseline attacker. Participants developed their attacker systems in the form of automatic speaker verification systems and submitted their scores on the development and evaluation data. The best attacker systems reduced the equal error rate (EER) by 25-44% relative w.r.t. the baseline. 

---
# Direct Advantage Regression: Aligning LLMs with Online AI Reward 

**Authors**: Li He, He Zhao, Stephen Wan, Dadong Wang, Lina Yao, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14177)  

**Abstract**: Online AI Feedback (OAIF) presents a promising alternative to Reinforcement Learning from Human Feedback (RLHF) by utilizing online AI preference in aligning language models (LLMs). However, the straightforward replacement of humans with AI deprives LLMs from learning more fine-grained AI supervision beyond binary signals. In this paper, we propose Direct Advantage Regression (DAR), a simple alignment algorithm using online AI reward to optimize policy improvement through weighted supervised fine-tuning. As an RL-free approach, DAR maintains theoretical consistency with online RLHF pipelines while significantly reducing implementation complexity and improving learning efficiency. Our empirical results underscore that AI reward is a better form of AI supervision consistently achieving higher human-AI agreement as opposed to AI preference. Additionally, evaluations using GPT-4-Turbo and MT-bench show that DAR outperforms both OAIF and online RLHF baselines. 

---
# HF4Rec: Human-Like Feedback-Driven Optimization Framework for Explainable Recommendation 

**Authors**: Jiakai Tang, Jingsen Zhang, Zihang Tian, Xueyang Feng, Lei Wang, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14147)  

**Abstract**: Recent advancements in explainable recommendation have greatly bolstered user experience by elucidating the decision-making rationale. However, the existing methods actually fail to provide effective feedback signals for potentially better or worse generated explanations due to their reliance on traditional supervised learning paradigms in sparse interaction data. To address these issues, we propose a novel human-like feedback-driven optimization framework. This framework employs a dynamic interactive optimization mechanism for achieving human-centered explainable requirements without incurring high labor costs. Specifically, we propose to utilize large language models (LLMs) as human simulators to predict human-like feedback for guiding the learning process. To enable the LLMs to deeply understand the task essence and meet user's diverse personalized requirements, we introduce a human-induced customized reward scoring method, which helps stimulate the language understanding and logical reasoning capabilities of LLMs. Furthermore, considering the potential conflicts between different perspectives of explanation quality, we introduce a principled Pareto optimization that transforms the multi-perspective quality enhancement task into a multi-objective optimization problem for improving explanation performance. At last, to achieve efficient model training, we design an off-policy optimization pipeline. By incorporating a replay buffer and addressing the data distribution biases, we can effectively improve data utilization and enhance model generality. Extensive experiments on four datasets demonstrate the superiority of our approach. 

---
# TALES: Text Adventure Learning Environment Suite 

**Authors**: Christopher Zhang Cui, Xingdi Yuan, Zhang Xiao, Prithviraj Ammanabrolu, Marc-Alexandre Côté  

**Link**: [PDF](https://arxiv.org/pdf/2504.14128)  

**Abstract**: Reasoning is an essential skill to enable Large Language Models (LLMs) to interact with the world. As tasks become more complex, they demand increasingly sophisticated and diverse reasoning capabilities for sequential decision-making, requiring structured reasoning over the context history to determine the next best action. We introduce TALES, a diverse collection of synthetic and human-written text-adventure games designed to challenge and evaluate diverse reasoning capabilities. We present results over a range of LLMs, open- and closed-weights, performing a qualitative analysis on the top performing models. Despite an impressive showing on synthetic games, even the top LLM-driven agents fail to achieve 15% on games designed for human enjoyment. Code and visualization of the experiments can be found at this https URL. 

---
# Large Language Model Enhanced Particle Swarm Optimization for Hyperparameter Tuning for Deep Learning Models 

**Authors**: Saad Hameed, Basheer Qolomany, Samir Brahim Belhaouari, Mohamed Abdallah, Junaid Qadir, Ala Al-Fuqaha  

**Link**: [PDF](https://arxiv.org/pdf/2504.14126)  

**Abstract**: Determining the ideal architecture for deep learning models, such as the number of layers and neurons, is a difficult and resource-intensive process that frequently relies on human tuning or computationally costly optimization approaches. While Particle Swarm Optimization (PSO) and Large Language Models (LLMs) have been individually applied in optimization and deep learning, their combined use for enhancing convergence in numerical optimization tasks remains underexplored. Our work addresses this gap by integrating LLMs into PSO to reduce model evaluations and improve convergence for deep learning hyperparameter tuning. The proposed LLM-enhanced PSO method addresses the difficulties of efficiency and convergence by using LLMs (particularly ChatGPT-3.5 and Llama3) to improve PSO performance, allowing for faster achievement of target objectives. Our method speeds up search space exploration by substituting underperforming particle placements with best suggestions offered by LLMs. Comprehensive experiments across three scenarios -- (1) optimizing the Rastrigin function, (2) using Long Short-Term Memory (LSTM) networks for time series regression, and (3) using Convolutional Neural Networks (CNNs) for material classification -- show that the method significantly improves convergence rates and lowers computational costs. Depending on the application, computational complexity is lowered by 20% to 60% compared to traditional PSO methods. Llama3 achieved a 20% to 40% reduction in model calls for regression tasks, whereas ChatGPT-3.5 reduced model calls by 60% for both regression and classification tasks, all while preserving accuracy and error rates. This groundbreaking methodology offers a very efficient and effective solution for optimizing deep learning models, leading to substantial computational performance improvements across a wide range of applications. 

---
# Bayesian Principles Improve Prompt Learning In Vision-Language Models 

**Authors**: Mingyu Kim, Jongwoo Ko, Mijung Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.14123)  

**Abstract**: Prompt learning is a popular fine-tuning method for vision-language models due to its efficiency. It requires a small number of additional learnable parameters while significantly enhancing performance on target tasks. However, most existing methods suffer from overfitting to fine-tuning data, yielding poor generalizability. To address this, we propose a new training objective function based on a Bayesian learning principle to balance adaptability and generalizability. We derive a prior over the logits, where the mean function is parameterized by the pre-trained model, while the posterior corresponds to the fine-tuned model. This objective establishes a balance by allowing the fine-tuned model to adapt to downstream tasks while remaining close to the pre-trained model. 

---
# System of Agentic AI for the Discovery of Metal-Organic Frameworks 

**Authors**: Theo Jaffrelot Inizan, Sherry Yang, Aaron Kaplan, Yen-hsu Lin, Jian Yin, Saber Mirzaei, Mona Abdelgaid, Ali H. Alawadhi, KwangHwan Cho, Zhiling Zheng, Ekin Dogus Cubuk, Christian Borgs, Jennifer T. Chayes, Kristin A. Persson, Omar M. Yaghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14110)  

**Abstract**: Generative models and machine learning promise accelerated material discovery in MOFs for CO2 capture and water harvesting but face significant challenges navigating vast chemical spaces while ensuring synthetizability. Here, we present MOFGen, a system of Agentic AI comprising interconnected agents: a large language model that proposes novel MOF compositions, a diffusion model that generates crystal structures, quantum mechanical agents that optimize and filter candidates, and synthetic-feasibility agents guided by expert rules and machine learning. Trained on all experimentally reported MOFs and computational databases, MOFGen generated hundreds of thousands of novel MOF structures and synthesizable organic linkers. Our methodology was validated through high-throughput experiments and the successful synthesis of five "AI-dreamt" MOFs, representing a major step toward automated synthesizable material discovery. 

---
# Linking forward-pass dynamics in Transformers and real-time human processing 

**Authors**: Jennifer Hu, Michael A. Lepori, Michael Franke  

**Link**: [PDF](https://arxiv.org/pdf/2504.14107)  

**Abstract**: Modern AI models are increasingly being used as theoretical tools to study human cognition. One dominant approach is to evaluate whether human-derived measures (such as offline judgments or real-time processing) are predicted by a model's output: that is, the end-product of forward pass(es) through the network. At the same time, recent advances in mechanistic interpretability have begun to reveal the internal processes that give rise to model outputs, raising the question of whether models and humans might arrive at outputs using similar "processing strategies". Here, we investigate the link between real-time processing in humans and "layer-time" dynamics in Transformer models. Across five studies spanning domains and modalities, we test whether the dynamics of computation in a single forward pass of pre-trained Transformers predict signatures of processing in humans, above and beyond properties of the model's output probability distribution. We consistently find that layer-time dynamics provide additional predictive power on top of output measures. Our results suggest that Transformer processing and human processing may be facilitated or impeded by similar properties of an input stimulus, and this similarity has emerged through general-purpose objectives such as next-token prediction or image recognition. Our work suggests a new way of using AI models to study human cognition: not just as a black box mapping stimuli to responses, but potentially also as explicit processing models. 

---
# Sentiment Analysis of Airbnb Reviews: Exploring Their Impact on Acceptance Rates and Pricing Across Multiple U.S. Regions 

**Authors**: Ali Safari  

**Link**: [PDF](https://arxiv.org/pdf/2504.14053)  

**Abstract**: This research examines whether Airbnb guests' positive and negative comments influence acceptance rates and rental prices across six U.S. regions: Rhode Island, Broward County, Chicago, Dallas, San Diego, and Boston. Thousands of reviews were collected and analyzed using Natural Language Processing (NLP) to classify sentiments as positive or negative, followed by statistical testing (t-tests and basic correlations) on the average scores. The findings reveal that over 90 percent of reviews in each region are positive, indicating that having additional reviews does not significantly enhance prices. However, listings with predominantly positive feedback exhibit slightly higher acceptance rates, suggesting that sentiment polarity, rather than the sheer volume of reviews, is a more critical factor for host success. Additionally, budget listings often gather extensive reviews while maintaining competitive pricing, whereas premium listings sustain higher prices with fewer but highly positive reviews. These results underscore the importance of sentiment quality over quantity in shaping guest behavior and pricing strategies in an overwhelmingly positive review environment. 

---
# Gradual Binary Search and Dimension Expansion : A general method for activation quantization in LLMs 

**Authors**: Lucas Maisonnave, Cyril Moineau, Olivier Bichler, Fabrice Rastello  

**Link**: [PDF](https://arxiv.org/pdf/2504.13989)  

**Abstract**: Large language models (LLMs) have become pivotal in artificial intelligence, demonstrating strong capabilities in reasoning, understanding, and generating data. However, their deployment on edge devices is hindered by their substantial size, often reaching several billion parameters. Quantization is a widely used method to reduce memory usage and inference time, however LLMs present unique challenges due to the prevalence of outliers in their activations. In this work, we leverage the theoretical advantages of Hadamard matrices over random rotation matrices to push the boundaries of quantization in LLMs. We demonstrate that Hadamard matrices are more effective in reducing outliers, which are a significant obstacle in achieving low-bit quantization. Our method based on a gradual binary search enables 3-bit quantization for weights, activations, and key-value (KV) caches, resulting in a 40\% increase in accuracy on common benchmarks compared to SoTA methods. We extend the use of rotation matrices to support non-power-of-2 embedding dimensions, similar to the Qwen architecture, by employing the Paley algorithm. We theoretically demonstrates the superiority of Hadamard matrices in reducing this http URL achieved 3-bit quantization for weights, activations, and KV cache, significantly enhancing model performance. Our experimental results on multiple models family like Mistral, LLaMA, and Qwen demonstrate the effectiveness of our approach, outperforming existing methods and enabling practical 3-bit quantization. 

---
# One Jump Is All You Need: Short-Cutting Transformers for Early Exit Prediction with One Jump to Fit All Exit Levels 

**Authors**: Amrit Diggavi Seshadri  

**Link**: [PDF](https://arxiv.org/pdf/2504.13984)  

**Abstract**: To reduce the time and computational costs of inference of large language models, there has been interest in parameter-efficient low-rank early-exit casting of transformer hidden-representations to final-representations. Such low-rank short-cutting has been shown to outperform identity shortcuts at early model stages while offering parameter-efficiency in shortcut jumps. However, current low-rank methods maintain a separate early-exit shortcut jump to final-representations for each transformer intermediate block-level during inference. In this work, we propose selection of a single One-Jump-Fits-All (OJFA) low-rank shortcut that offers over a 30x reduction in shortcut parameter costs during inference. We show that despite this extreme reduction, our OJFA choice largely matches the performance of maintaining multiple shortcut jumps during inference and offers stable precision from all transformer block-levels for GPT2-XL, Phi3-Mini and Llama2-7B transformer models. 

---
# AI Safety Should Prioritize the Future of Work 

**Authors**: Sanchaita Hazra, Bodhisattwa Prasad Majumder, Tuhin Chakrabarty  

**Link**: [PDF](https://arxiv.org/pdf/2504.13959)  

**Abstract**: Current efforts in AI safety prioritize filtering harmful content, preventing manipulation of human behavior, and eliminating existential risks in cybersecurity or biosecurity. While pressing, this narrow focus overlooks critical human-centric considerations that shape the long-term trajectory of a society. In this position paper, we identify the risks of overlooking the impact of AI on the future of work and recommend comprehensive transition support towards the evolution of meaningful labor with human agency. Through the lens of economic theories, we highlight the intertemporal impacts of AI on human livelihood and the structural changes in labor markets that exacerbate income inequality. Additionally, the closed-source approach of major stakeholders in AI development resembles rent-seeking behavior through exploiting resources, breeding mediocrity in creative labor, and monopolizing innovation. To address this, we argue in favor of a robust international copyright anatomy supported by implementing collective licensing that ensures fair compensation mechanisms for using data to train AI models. We strongly recommend a pro-worker framework of global AI governance to enhance shared prosperity and economic justice while reducing technical debt. 

---
# ToolRL: Reward is All Tool Learning Needs 

**Authors**: Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.13958)  

**Abstract**: Current Large Language Models (LLMs) often undergo supervised fine-tuning (SFT) to acquire tool use capabilities. However, SFT struggles to generalize to unfamiliar or complex tool use scenarios. Recent advancements in reinforcement learning (RL), particularly with R1-like models, have demonstrated promising reasoning and generalization abilities. Yet, reward design for tool use presents unique challenges: multiple tools may be invoked with diverse parameters, and coarse-grained reward signals, such as answer matching, fail to offer the finegrained feedback required for effective learning. In this work, we present the first comprehensive study on reward design for tool selection and application tasks within the RL paradigm. We systematically explore a wide range of reward strategies, analyzing their types, scales, granularity, and temporal dynamics. Building on these insights, we propose a principled reward design tailored for tool use tasks and apply it to train LLMs using Group Relative Policy Optimization (GRPO). Empirical evaluations across diverse benchmarks demonstrate that our approach yields robust, scalable, and stable training, achieving a 17% improvement over base models and a 15% gain over SFT models. These results highlight the critical role of thoughtful reward design in enhancing the tool use capabilities and generalization performance of LLMs. All the codes are released to facilitate future research. 

---
# Thousand Voices of Trauma: A Large-Scale Synthetic Dataset for Modeling Prolonged Exposure Therapy Conversations 

**Authors**: Suhas BN, Dominik Mattioli, Saeed Abdullah, Rosa I. Arriaga, Chris W. Wiese, Andrew M. Sherrill  

**Link**: [PDF](https://arxiv.org/pdf/2504.13955)  

**Abstract**: The advancement of AI systems for mental health support is hindered by limited access to therapeutic conversation data, particularly for trauma treatment. We present Thousand Voices of Trauma, a synthetic benchmark dataset of 3,000 therapy conversations based on Prolonged Exposure therapy protocols for Post-traumatic Stress Disorder (PTSD). The dataset comprises 500 unique cases, each explored through six conversational perspectives that mirror the progression of therapy from initial anxiety to peak distress to emotional processing. We incorporated diverse demographic profiles (ages 18-80, M=49.3, 49.4% male, 44.4% female, 6.2% non-binary), 20 trauma types, and 10 trauma-related behaviors using deterministic and probabilistic generation methods. Analysis reveals realistic distributions of trauma types (witnessing violence 10.6%, bullying 10.2%) and symptoms (nightmares 23.4%, substance abuse 20.8%). Clinical experts validated the dataset's therapeutic fidelity, highlighting its emotional depth while suggesting refinements for greater authenticity. We also developed an emotional trajectory benchmark with standardized metrics for evaluating model responses. This privacy-preserving dataset addresses critical gaps in trauma-focused mental health data, offering a valuable resource for advancing both patient-facing applications and clinician training tools. 

---
# Enhancing Ultra-Low-Bit Quantization of Large Language Models Through Saliency-Aware Partial Retraining 

**Authors**: Deyu Cao, Samin Aref  

**Link**: [PDF](https://arxiv.org/pdf/2504.13932)  

**Abstract**: Large language models offer remarkable capabilities, but their size and computational demands pose practical challenges. Quantization methods compress their size through replacing their high-precision parameters by quantized values of lower precision. Post-training quantization reduces model size efficiently at the cost of decreased accuracy, while quantization-aware training better preserves accuracy but is resource-intensive. Among existing post-training quantization algorithms, the ApiQ method achieves superior accuracy preservation at minimal memory and time overhead. We investigate two ideas to extend performance in ultra-low-bit quantization beyond ApiQ's level. First, we look into combining existing quantization-aware training techniques with ApiQ's partial training. We show that this does not outperform the baseline ApiQ method with limited training data and frozen weights. This leads to two key insights: (1) The substantial representational capacity that is gained through full retraining may not be feasible through partial training. (2) This gain seems to depend on using a large and diverse dataset in quantization-aware training. Second, through a novel approach informed by the two insights, we propose an ultra-low-bit quantization method that builds upon ApiQ and extends its performance without the need for full retraining. It relies on a saliency-aware regularization term that prioritizes preserving the most impactful parameters during quantization. Our experiments on benchmark language models from the LLaMA family show that our proposed approach boosts accuracy and tightens the gap between the quantized model and the full-precision model, with minimal overhead. Our method will be made publicly available to facilitate future developments in ultra-low-bit quantization of large language models. 

---
# Evaluation and Incident Prevention in an Enterprise AI Assistant 

**Authors**: Akash V. Maharaj, David Arbour, Daniel Lee, Uttaran Bhattacharya, Anup Rao, Austin Zane, Avi Feller, Kun Qian, Yunyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.13924)  

**Abstract**: Enterprise AI Assistants are increasingly deployed in domains where accuracy is paramount, making each erroneous output a potentially significant incident. This paper presents a comprehensive framework for monitoring, benchmarking, and continuously improving such complex, multi-component systems under active development by multiple teams. Our approach encompasses three key elements: (1) a hierarchical ``severity'' framework for incident detection that identifies and categorizes errors while attributing component-specific error rates, facilitating targeted improvements; (2) a scalable and principled methodology for benchmark construction, evaluation, and deployment, designed to accommodate multiple development teams, mitigate overfitting risks, and assess the downstream impact of system modifications; and (3) a continual improvement strategy leveraging multidimensional evaluation, enabling the identification and implementation of diverse enhancement opportunities. By adopting this holistic framework, organizations can systematically enhance the reliability and performance of their AI Assistants, ensuring their efficacy in critical enterprise environments. We conclude by discussing how this multifaceted evaluation approach opens avenues for various classes of enhancements, paving the way for more robust and trustworthy AI systems. 

---
# Generative Framework for Personalized Persuasion: Inferring Causal, Counterfactual, and Latent Knowledge 

**Authors**: Donghuo Zeng, Roberto Legaspi, Yuewen Sun, Xinshuai Dong, Kazushi Ikeda, Peter Spirtes, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13904)  

**Abstract**: We hypothesize that optimal system responses emerge from adaptive strategies grounded in causal and counterfactual knowledge. Counterfactual inference allows us to create hypothetical scenarios to examine the effects of alternative system responses. We enhance this process through causal discovery, which identifies the strategies informed by the underlying causal structure that govern system behaviors. Moreover, we consider the psychological constructs and unobservable noises that might be influencing user-system interactions as latent factors. We show that these factors can be effectively estimated. We employ causal discovery to identify strategy-level causal relationships among user and system utterances, guiding the generation of personalized counterfactual dialogues. We model the user utterance strategies as causal factors, enabling system strategies to be treated as counterfactual actions. Furthermore, we optimize policies for selecting system responses based on counterfactual data. Our results using a real-world dataset on social good demonstrate significant improvements in persuasive system outcomes, with increased cumulative rewards validating the efficacy of causal discovery in guiding personalized counterfactual inference and optimizing dialogue policies for a persuasive dialogue system. 

---
# TALLMesh: a simple application for performing Thematic Analysis with Large Language Models 

**Authors**: Stefano De Paoli, Alex Fawzi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13892)  

**Abstract**: Thematic analysis (TA) is a widely used qualitative research method for identifying and interpreting patterns within textual data, such as qualitative interviews. Recent research has shown that it is possible to satisfactorily perform TA using Large Language Models (LLMs). This paper presents a novel application using LLMs to assist researchers in conducting TA. The application enables users to upload textual data, generate initial codes and themes. All of this is possible through a simple Graphical User Interface, (GUI) based on the streamlit framework, working with python scripts for the analysis, and using Application Program Interfaces of LLMs. Having a GUI is particularly important for researchers in fields where coding skills may not be prevalent, such as social sciences or humanities. With the app, users can iteratively refine codes and themes adopting a human-in-the-loop process, without the need to work with programming and scripting. The paper describes the application key features, highlighting its potential for qualitative research while preserving methodological rigor. The paper discusses the design and interface of the app and outlines future directions for this work. 

---
# Measuring Mental Health Variables in Computational Research: Toward Validated, Dimensional, and Transdiagnostic Approaches 

**Authors**: Chen Shani, Elizabeth C. Stade  

**Link**: [PDF](https://arxiv.org/pdf/2504.13890)  

**Abstract**: Computational mental health research develops models to predict and understand psychological phenomena, but often relies on inappropriate measures of psychopathology constructs, undermining validity. We identify three key issues: (1) reliance on unvalidated measures (e.g., self-declared diagnosis) over validated ones (e.g., diagnosis by clinician); (2) treating mental health constructs as categorical rather than dimensional; and (3) focusing on disorder-specific constructs instead of transdiagnostic ones. We outline the benefits of using validated, dimensional, and transdiagnostic measures and offer practical recommendations for practitioners. Using valid measures that reflect the nature and structure of psychopathology is essential for computational mental health research. 

---
# Kanji Workbook: A Writing-Based Intelligent Tutoring System for Learning Proper Japanese Kanji Writing Technique with Instructor-Emulated Assessment 

**Authors**: Paul Taele, Jung In Koh, Tracy Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2504.13888)  

**Abstract**: Kanji script writing is a skill that is often introduced to novice Japanese foreign language students for achieving Japanese writing mastery, but often poses difficulties to students with primarily English fluency due to their its vast differences with written English. Instructors often introduce various pedagogical methods -- such as visual structure and written techniques -- to assist students in kanji study, but may lack availability providing direct feedback on students' writing outside of class. Current educational applications are also limited due to lacking richer instructor-emulated feedback. We introduce Kanji Workbook, a writing-based intelligent tutoring system for students to receive intelligent assessment that emulates human instructor feedback. Our interface not only leverages students' computing devices for allowing them to learn, practice, and review the writing of prompted characters from their course's kanji script lessons, but also provides a diverse set of writing assessment metrics -- derived from instructor interviews and classroom observation insights -- through intelligent scoring and visual animations. We deployed our interface onto novice- and intermediate-level university courses over an entire academic year, and observed that interface users on average achieved higher course grades than their peers and also reacted positively to our interface's various features. 

---
# AI as a deliberative partner fosters intercultural empathy for Americans but fails for Latin American participants 

**Authors**: Isabel Villanueva, Tara Bobinac, Binwei Yao, Junjie Hu, Kaiping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.13887)  

**Abstract**: Despite the growing integration of AI chatbots as conversational agents in public discourse, empirical evidence regarding their capacity to foster intercultural empathy remains limited. Using a randomized dialogue experiment, we examined how different types of AI chatbot interaction, i.e., deliberative versus non-deliberative and culturally aligned versus non-aligned, affect intercultural empathy across cultural groups. Results show that deliberative conversations increased intercultural empathy among American participants but not Latin American participants, who perceived AI responses as culturally inaccurate and failing to represent their cultural contexts and perspectives authentically. Real-time interaction analyses reveal that these differences stem from cultural knowledge gaps inherent in Large Language Models. Despite explicit prompting and instruction to represent cultural perspectives in participants' native languages, AI systems still exhibit significant disparities in cultural representation. This highlights the importance of designing AI systems capable of culturally authentic engagement in deliberative conversations. Our study contributes to deliberation theory and AI alignment research by underscoring AI's role in intercultural dialogue and the persistent challenge of representational asymmetry in democratic discourse. 

---
# Toward Automated Qualitative Analysis: Leveraging Large Language Models for Tutoring Dialogue Evaluation 

**Authors**: Megan Gu, Chloe Qianhui Zhao, Claire Liu, Nikhil Patel, Jahnvi Shah, Jionghao Lin, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2504.13882)  

**Abstract**: Our study introduces an automated system leveraging large language models (LLMs) to assess the effectiveness of five key tutoring strategies: 1. giving effective praise, 2. reacting to errors, 3. determining what students know, 4. helping students manage inequity, and 5. responding to negative self-talk. Using a public dataset from the Teacher-Student Chatroom Corpus, our system classifies each tutoring strategy as either being employed as desired or undesired. Our study utilizes GPT-3.5 with few-shot prompting to assess the use of these strategies and analyze tutoring dialogues. The results show that for the five tutoring strategies, True Negative Rates (TNR) range from 0.655 to 0.738, and Recall ranges from 0.327 to 0.432, indicating that the model is effective at excluding incorrect classifications but struggles to consistently identify the correct strategy. The strategy \textit{helping students manage inequity} showed the highest performance with a TNR of 0.738 and Recall of 0.432. The study highlights the potential of LLMs in tutoring strategy analysis and outlines directions for future improvements, including incorporating more advanced models for more nuanced feedback. 

---
# A Survey on (M)LLM-Based GUI Agents 

**Authors**: Fei Tang, Haolei Xu, Hang Zhang, Siqi Chen, Xingyu Wu, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Zeqi Tan, Yuchen Yan, Kaitao Song, Jian Shao, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13865)  

**Abstract**: Graphical User Interface (GUI) Agents have emerged as a transformative paradigm in human-computer interaction, evolving from rule-based automation scripts to sophisticated AI-driven systems capable of understanding and executing complex interface operations. This survey provides a comprehensive examination of the rapidly advancing field of LLM-based GUI Agents, systematically analyzing their architectural foundations, technical components, and evaluation methodologies. We identify and analyze four fundamental components that constitute modern GUI Agents: (1) perception systems that integrate text-based parsing with multimodal understanding for comprehensive interface comprehension; (2) exploration mechanisms that construct and maintain knowledge bases through internal modeling, historical experience, and external information retrieval; (3) planning frameworks that leverage advanced reasoning methodologies for task decomposition and execution; and (4) interaction systems that manage action generation with robust safety controls. Through rigorous analysis of these components, we reveal how recent advances in large language models and multimodal learning have revolutionized GUI automation across desktop, mobile, and web platforms. We critically examine current evaluation frameworks, highlighting methodological limitations in existing benchmarks while proposing directions for standardization. This survey also identifies key technical challenges, including accurate element localization, effective knowledge retrieval, long-horizon planning, and safety-aware execution control, while outlining promising research directions for enhancing GUI Agents' capabilities. Our systematic review provides researchers and practitioners with a thorough understanding of the field's current state and offers insights into future developments in intelligent interface automation. 

---
# 3MDBench: Medical Multimodal Multi-agent Dialogue Benchmark 

**Authors**: Ivan Sviridov, Amina Miftakhova, Artemiy Tereshchenko, Galina Zubkova, Pavel Blinov, Andrey Savchenko  

**Link**: [PDF](https://arxiv.org/pdf/2504.13861)  

**Abstract**: Large Vision-Language Models (LVLMs) are increasingly being explored for applications in telemedicine, yet their ability to engage with diverse patient behaviors remains underexplored. We introduce 3MDBench (Medical Multimodal Multi-agent Dialogue Benchmark), an open-source evaluation framework designed to assess LLM-driven medical consultations. Unlike existing benchmarks, 3MDBench simulates real-world patient variability by incorporating four temperament-driven Patient Agents and an Assessor Agent that evaluates diagnostic accuracy and dialogue quality. The benchmark integrates textual and image-based patient data across 34 common diagnoses, mirroring real-world telemedicine interactions. Under different diagnostic strategies, we evaluate state-of-the-art LVLMs. Our findings demonstrate that incorporating dialogue improves the F1 score from 50.4 to 54.2 compared to non-dialogue settings, underscoring the value of context-driven, information-seeking questioning. Additionally, we demonstrate that multimodal inputs enhance diagnostic efficiency. Image-supported models outperform text-only counterparts by raising the diagnostic F1 score from 52.8 to 54.2 in a similar dialogue setting. Finally, we suggest an approach that improves the diagnostic F1-score to 70.3 by training the CNN model on the diagnosis prediction task and incorporating its top-3 predictions into the LVLM context. 3MDBench provides a reproducible and extendable evaluation framework for AI-driven medical assistants. It offers insights into how patient temperament, dialogue strategies, and multimodal reasoning influence diagnosis quality. By addressing real-world complexities in telemedicine, our benchmark paves the way for more empathetic, reliable, and context-aware AI-driven healthcare solutions. The source code of our benchmark is publicly available: this https URL 

---
# Interview AI-ssistant: Designing for Real-Time Human-AI Collaboration in Interview Preparation and Execution 

**Authors**: Zhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13847)  

**Abstract**: Recent advances in large language models (LLMs) offer unprecedented opportunities to enhance human-AI collaboration in qualitative research methods, including interviews. While interviews are highly valued for gathering deep, contextualized insights, interviewers often face significant cognitive challenges, such as real-time information processing, question adaptation, and rapport maintenance. My doctoral research introduces Interview AI-ssistant, a system designed for real-time interviewer-AI collaboration during both the preparation and execution phases. Through four interconnected studies, this research investigates the design of effective human-AI collaboration in interviewing contexts, beginning with a formative study of interviewers' needs, followed by a prototype development study focused on AI-assisted interview preparation, an experimental evaluation of real-time AI assistance during interviews, and a field study deploying the system in a real-world research setting. Beyond informing practical implementations of intelligent interview support systems, this work contributes to the Intelligent User Interfaces (IUI) community by advancing the understanding of human-AI collaborative interfaces in complex social tasks and establishing design guidelines for AI-enhanced qualitative research tools. 

---
