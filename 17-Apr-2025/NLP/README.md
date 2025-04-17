# BitNet b1.58 2B4T Technical Report 

**Authors**: Shuming Ma, Hongyu Wang, Shaohan Huang, Xingxing Zhang, Ying Hu, Ting Song, Yan Xia, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2504.12285)  

**Abstract**: We introduce BitNet b1.58 2B4T, the first open-source, native 1-bit Large Language Model (LLM) at the 2-billion parameter scale. Trained on a corpus of 4 trillion tokens, the model has been rigorously evaluated across benchmarks covering language understanding, mathematical reasoning, coding proficiency, and conversational ability. Our results demonstrate that BitNet b1.58 2B4T achieves performance on par with leading open-weight, full-precision LLMs of similar size, while offering significant advantages in computational efficiency, including substantially reduced memory footprint, energy consumption, and decoding latency. To facilitate further research and adoption, the model weights are released via Hugging Face along with open-source inference implementations for both GPU and CPU architectures. 

---
# d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning 

**Authors**: Siyan Zhao, Devaansh Gupta, Qinqing Zheng, Aditya Grover  

**Link**: [PDF](https://arxiv.org/pdf/2504.12216)  

**Abstract**: Recent large language models (LLMs) have demonstrated strong reasoning capabilities that benefits from online reinforcement learning (RL). These capabilities have primarily been demonstrated within the left-to-right autoregressive (AR) generation paradigm. In contrast, non-autoregressive paradigms based on diffusion generate text in a coarse-to-fine manner. Although recent diffusion-based large language models (dLLMs) have achieved competitive language modeling performance compared to their AR counterparts, it remains unclear if dLLMs can also leverage recent advances in LLM reasoning. To this end, we propose d1, a framework to adapt pre-trained masked dLLMs into reasoning models via a combination of supervised finetuning (SFT) and RL. Specifically, we develop and extend techniques to improve reasoning in pretrained dLLMs: (a) we utilize a masked SFT technique to distill knowledge and instill self-improvement behavior directly from existing datasets, and (b) we introduce a novel critic-free, policy-gradient based RL algorithm called diffu-GRPO. Through empirical studies, we investigate the performance of different post-training recipes on multiple mathematical and logical reasoning benchmarks. We find that d1 yields the best performance and significantly improves performance of a state-of-the-art dLLM. 

---
# What Do Large Language Models Know? Tacit Knowledge as a Potential Causal-Explanatory Structure 

**Authors**: Céline Budding  

**Link**: [PDF](https://arxiv.org/pdf/2504.12187)  

**Abstract**: It is sometimes assumed that Large Language Models (LLMs) know language, or for example that they know that Paris is the capital of France. But what -- if anything -- do LLMs actually know? In this paper, I argue that LLMs can acquire tacit knowledge as defined by Martin Davies (1990). Whereas Davies himself denies that neural networks can acquire tacit knowledge, I demonstrate that certain architectural features of LLMs satisfy the constraints of semantic description, syntactic structure, and causal systematicity. Thus, tacit knowledge may serve as a conceptual framework for describing, explaining, and intervening on LLMs and their behavior. 

---
# SALAD: Improving Robustness and Generalization through Contrastive Learning with Structure-Aware and LLM-Driven Augmented Data 

**Authors**: Suyoung Bae, Hyojun Kim, YunSeok Choi, Jee-Hyong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.12185)  

**Abstract**: In various natural language processing (NLP) tasks, fine-tuning Pre-trained Language Models (PLMs) often leads to the issue of spurious correlations, which negatively impacts performance, particularly when dealing with out-of-distribution data. To address this problem, we propose SALAD}(Structure Aware and LLM-driven Augmented Data), a novel approach designed to enhance model robustness and generalization by generating structure-aware and counterfactually augmented data for contrastive learning. Our method leverages a tagging-based approach to generate structure-aware positive samples and utilizes large language models (LLMs) to generate counterfactual negative samples with diverse sentence patterns. By applying contrastive learning, SALAD enables the model to focus on learning the structural relationships between key sentence components while minimizing reliance on spurious correlations. We validate our approach through experiments on three tasks: Sentiment Classification, Sexism Detection, and Natural Language Inference. The results demonstrate that SALAD not only improves model robustness and performance across different environments but also enhances generalization to out-of-distribution datasets and cross-domain scenarios. 

---
# Trusting CHATGPT: how minor tweaks in the prompts lead to major differences in sentiment classification 

**Authors**: Jaime E. Cuellar, Oscar Moreno-Martinez, Paula Sofia Torres-Rodriguez, Jaime Andres Pavlich-Mariscal, Andres Felipe Mican-Castiblanco, Juan Guillermo Torres-Hurtado  

**Link**: [PDF](https://arxiv.org/pdf/2504.12180)  

**Abstract**: One fundamental question for the social sciences today is: how much can we trust highly complex predictive models like ChatGPT? This study tests the hypothesis that subtle changes in the structure of prompts do not produce significant variations in the classification results of sentiment polarity analysis generated by the Large Language Model GPT-4o mini. Using a dataset of 100.000 comments in Spanish on four Latin American presidents, the model classified the comments as positive, negative, or neutral on 10 occasions, varying the prompts slightly each time. The experimental methodology included exploratory and confirmatory analyses to identify significant discrepancies among classifications.
The results reveal that even minor modifications to prompts such as lexical, syntactic, or modal changes, or even their lack of structure impact the classifications. In certain cases, the model produced inconsistent responses, such as mixing categories, providing unsolicited explanations, or using languages other than Spanish. Statistical analysis using Chi-square tests confirmed significant differences in most comparisons between prompts, except in one case where linguistic structures were highly similar.
These findings challenge the robustness and trust of Large Language Models for classification tasks, highlighting their vulnerability to variations in instructions. Moreover, it was evident that the lack of structured grammar in prompts increases the frequency of hallucinations. The discussion underscores that trust in Large Language Models is based not only on technical performance but also on the social and institutional relationships underpinning their use. 

---
# Mapping Controversies Using Artificial Intelligence: An Analysis of the Hamas-Israel Conflict on YouTube 

**Authors**: Victor Manuel Hernandez Lopez, Jaime E. Cuellar  

**Link**: [PDF](https://arxiv.org/pdf/2504.12177)  

**Abstract**: This article analyzes the Hamas-Israel controversy through 253,925 Spanish-language YouTube comments posted between October 2023 and January 2024, following the October 7 attack that escalated the conflict. Adopting an interdisciplinary approach, the study combines the analysis of controversies from Science and Technology Studies (STS) with advanced computational methodologies, specifically Natural Language Processing (NLP) using the BERT (Bidirectional Encoder Representations from Transformers) model. Using this approach, the comments were automatically classified into seven categories, reflecting pro-Palestinian, pro-Israeli, anti- Palestinian, anti-Israeli positions, among others. The results show a predominance of pro- Palestinian comments, although pro-Israeli and anti-Palestinian comments received more "likes." This study also applies the agenda-setting theory to demonstrate how media coverage significantly influences public perception, observing a notable shift in public opinion, transitioning from a pro- Palestinian stance to a more critical position towards Israel. This work highlights the importance of combining social science perspectives with technological tools in the analysis of controversies, presenting a methodological innovation by integrating computational analysis with critical social theories to address complex public opinion phenomena and media narratives. 

---
# Poem Meter Classification of Recited Arabic Poetry: Integrating High-Resource Systems for a Low-Resource Task 

**Authors**: Maged S. Al-Shaibani, Zaid Alyafeai, Irfan Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2504.12172)  

**Abstract**: Arabic poetry is an essential and integral part of Arabic language and culture. It has been used by the Arabs to spot lights on their major events such as depicting brutal battles and conflicts. They also used it, as in many other languages, for various purposes such as romance, pride, lamentation, etc. Arabic poetry has received major attention from linguistics over the decades. One of the main characteristics of Arabic poetry is its special rhythmic structure as opposed to prose. This structure is referred to as a meter. Meters, along with other poetic characteristics, are intensively studied in an Arabic linguistic field called "\textit{Aroud}". Identifying these meters for a verse is a lengthy and complicated process. It also requires technical knowledge in \textit{Aruod}. For recited poetry, it adds an extra layer of processing. Developing systems for automatic identification of poem meters for recited poems need large amounts of labelled data. In this study, we propose a state-of-the-art framework to identify the poem meters of recited Arabic poetry, where we integrate two separate high-resource systems to perform the low-resource task. To ensure generalization of our proposed architecture, we publish a benchmark for this task for future research. 

---
# Multilingual Contextualization of Large Language Models for Document-Level Machine Translation 

**Authors**: Miguel Moura Ramos, Patrick Fernandes, Sweta Agrawal, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2504.12140)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance in sentence-level machine translation, but scaling to document-level translation remains challenging, particularly in modeling long-range dependencies and discourse phenomena across sentences and paragraphs. In this work, we propose a method to improve LLM-based long-document translation through targeted fine-tuning on high-quality document-level data, which we curate and introduce as DocBlocks. Our approach supports multiple translation paradigms, including direct document-to-document and chunk-level translation, by integrating instructions both with and without surrounding context. This enables models to better capture cross-sentence dependencies while maintaining strong sentence-level translation performance. Experimental results show that incorporating multiple translation paradigms improves document-level translation quality and inference speed compared to prompting and agent-based methods. 

---
# Entropy-Guided Watermarking for LLMs: A Test-Time Framework for Robust and Traceable Text Generation 

**Authors**: Shizhan Cai, Liang Ding, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12108)  

**Abstract**: The rapid development of Large Language Models (LLMs) has intensified concerns about content traceability and potential misuse. Existing watermarking schemes for sampled text often face trade-offs between maintaining text quality and ensuring robust detection against various attacks. To address these issues, we propose a novel watermarking scheme that improves both detectability and text quality by introducing a cumulative watermark entropy threshold. Our approach is compatible with and generalizes existing sampling functions, enhancing adaptability. Experimental results across multiple LLMs show that our scheme significantly outperforms existing methods, achieving over 80\% improvements on widely-used datasets, e.g., MATH and GSM8K, while maintaining high detection accuracy. 

---
# Gauging Overprecision in LLMs: An Empirical Study 

**Authors**: Adil Bahaj, Hamed Rahimi, Mohamed Chetouani, Mounir Ghogho  

**Link**: [PDF](https://arxiv.org/pdf/2504.12098)  

**Abstract**: Recently, overconfidence in large language models (LLMs) has garnered considerable attention due to its fundamental importance in quantifying the trustworthiness of LLM generation. However, existing approaches prompt the \textit{black box LLMs} to produce their confidence (\textit{verbalized confidence}), which can be subject to many biases and hallucinations. Inspired by a different aspect of overconfidence in cognitive science called \textit{overprecision}, we designed a framework for its study in black box LLMs. This framework contains three main phases: 1) generation, 2) refinement and 3) evaluation. In the generation phase we prompt the LLM to generate answers to numerical questions in the form of intervals with a certain level of confidence. This confidence level is imposed in the prompt and not required for the LLM to generate as in previous approaches. We use various prompting techniques and use the same prompt multiple times to gauge the effects of randomness in the generation process. In the refinement phase, answers from the previous phase are refined to generate better answers. The LLM answers are evaluated and studied in the evaluation phase to understand its internal workings. This study allowed us to gain various insights into LLM overprecision: 1) LLMs are highly uncalibrated for numerical tasks 2) {\color{blue}there is no correlation between the length of the interval and the imposed confidence level, which can be symptomatic of a a) lack of understanding of the concept of confidence or b) inability to adjust self-confidence by following instructions}, {\color{blue}3)} LLM numerical precision differs depending on the task, scale of answer and prompting technique {\color{blue}4) Refinement of answers doesn't improve precision in most cases}. We believe this study offers new perspectives on LLM overconfidence and serves as a strong baseline for overprecision in LLMs. 

---
# Selective Demonstration Retrieval for Improved Implicit Hate Speech Detection 

**Authors**: Yumin Kim, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.12082)  

**Abstract**: Hate speech detection is a crucial area of research in natural language processing, essential for ensuring online community safety. However, detecting implicit hate speech, where harmful intent is conveyed in subtle or indirect ways, remains a major challenge. Unlike explicit hate speech, implicit expressions often depend on context, cultural subtleties, and hidden biases, making them more challenging to identify consistently. Additionally, the interpretation of such speech is influenced by external knowledge and demographic biases, resulting in varied detection results across different language models. Furthermore, Large Language Models often show heightened sensitivity to toxic language and references to vulnerable groups, which can lead to misclassifications. This over-sensitivity results in false positives (incorrectly identifying harmless statements as hateful) and false negatives (failing to detect genuinely harmful content). Addressing these issues requires methods that not only improve detection precision but also reduce model biases and enhance robustness. To address these challenges, we propose a novel method, which utilizes in-context learning without requiring model fine-tuning. By adaptively retrieving demonstrations that focus on similar groups or those with the highest similarity scores, our approach enhances contextual comprehension. Experimental results show that our method outperforms current state-of-the-art techniques. Implementation details and code are available at TBD. 

---
# Bayesian dynamic borrowing considering semantic similarity between outcomes for disproportionality analysis in FAERS 

**Authors**: François Haguinet, Jeffery L Painter, Gregory E Powell, Andrea Callegaro, Andrew Bate  

**Link**: [PDF](https://arxiv.org/pdf/2504.12052)  

**Abstract**: We present a Bayesian dynamic borrowing (BDB) approach to enhance the quantitative identification of adverse events (AEs) in spontaneous reporting systems (SRSs). The method embeds a robust meta-analytic predictive (MAP) prior within a Bayesian hierarchical model and incorporates semantic similarity measures (SSMs) to enable weighted information sharing from MedDRA Preferred Terms (PTs) that are clinical similar to the target PT. This continuous similarity-based borrowing addresses limitation of rigid hierarchical grouping in current disproportionality analysis (DPA).
Using data from the FDA Adverse Event Reporting System (FAERS) between 2015 and 2019, we evalute this approach - termed IC SSM - against standard Information Component (IC) analysis and IC with borrowing at the MedDRA high-level group term (HLGT) level. A novel references set (PVLens), derived from FDA product label updates, enabled prospective evaluation of method performance in identifying AEs prior to official labeling.
The IC SSM approach demonstrated improved sensitivity compared to both traditional IC and HLGT-based borrowing, with minor trade-offs in F1 scores and Youden's index. IC SSM consistently identified more true positives and detected signals over 5 months sooner than traditional IC. Despite a marginally lower aggregate Youden's index, IC SSM showed higher performance in the early post-marketing period, providing more stable and relevant estimates than HLGT-based borrowing and traditional IC.
These findings support the use of SSM-informed Bayesian borrowing as a scalable and context-aware enhancement to traditional DPA methods. Future research should validate this approach across other datasets and explore additional similarity metrics and Bayesian inference strategies using case-level data. 

---
# Language Models as Quasi-Crystalline Thought: Structure, Constraint, and Emergence in Generative Systems 

**Authors**: Jose Manuel Guevara-Vela  

**Link**: [PDF](https://arxiv.org/pdf/2504.11986)  

**Abstract**: This essay proposes an analogy between large language models (LLMs) and quasicrystals: systems that exhibit global coherence without periodic repetition and that are generated through local constraints. While LLMs are often evaluated in terms of predictive accuracy, factuality, or alignment, this structural perspective suggests that their most characteristic behavior is the production of internally resonant linguistic patterns. Just as quasicrystals forced a redefinition of order in physical systems, viewing LLMs as generators of quasi-structured language opens new paths for evaluation and design: privileging propagation of constraint over token-level accuracy, and coherence of form over fixed meaning. LLM outputs should be read not only for what they say, but for the patterns of constraint and coherence that organize them. This shift reframes generative language as a space of emergent patterning: LLMs are neither fully random nor strictly rule-based, but defined by a logic of constraint, resonance, and structural depth. 

---
# SemEval-2025 Task 3: Mu-SHROOM, the Multilingual Shared Task on Hallucinations and Related Observable Overgeneration Mistakes 

**Authors**: Raúl Vázquez, Timothee Mickus, Elaine Zosa, Teemu Vahtola, Jörg Tiedemann, Aman Sinha, Vincent Segonne, Fernando Sánchez-Vega, Alessandro Raganato, Jindřich Libovický, Jussi Karlgren, Shaoxiong Ji, Jindřich Helcl, Liane Guillou, Ona de Gibert, Jaione Bengoetxea, Joseph Attieh, Marianna Apidianaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.11975)  

**Abstract**: We present the Mu-SHROOM shared task which is focused on detecting hallucinations and other overgeneration mistakes in the output of instruction-tuned large language models (LLMs). Mu-SHROOM addresses general-purpose LLMs in 14 languages, and frames the hallucination detection problem as a span-labeling task. We received 2,618 submissions from 43 participating teams employing diverse methodologies. The large number of submissions underscores the interest of the community in hallucination detection. We present the results of the participating systems and conduct an empirical analysis to identify key factors contributing to strong performance in this task. We also emphasize relevant current challenges, notably the varying degree of hallucinations across languages and the high annotator disagreement when labeling hallucination spans. 

---
# LLM-as-a-Judge: Reassessing the Performance of LLMs in Extractive QA 

**Authors**: Xanh Ho, Jiahao Huang, Florian Boudin, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2504.11972)  

**Abstract**: Extractive reading comprehension question answering (QA) datasets are typically evaluated using Exact Match (EM) and F1-score, but these metrics often fail to fully capture model performance. With the success of large language models (LLMs), they have been employed in various tasks, including serving as judges (LLM-as-a-judge). In this paper, we reassess the performance of QA models using LLM-as-a-judge across four reading comprehension QA datasets. We examine different families of LLMs and various answer types to evaluate the effectiveness of LLM-as-a-judge in these tasks. Our results show that LLM-as-a-judge is highly correlated with human judgments and can replace traditional EM/F1 metrics. By using LLM-as-a-judge, the correlation with human judgments improves significantly, from 0.17 (EM) and 0.36 (F1-score) to 0.85. These findings confirm that EM and F1 metrics underestimate the true performance of the QA models. While LLM-as-a-judge is not perfect for more difficult answer types (e.g., job), it still outperforms EM/F1, and we observe no bias issues, such as self-preference, when the same model is used for both the QA and judgment tasks. 

---
# Robust and Fine-Grained Detection of AI Generated Texts 

**Authors**: Ram Mohan Rao Kadiyala, Siddartha Pullakhandam, Kanwal Mehreen, Drishti Sharma, Siddhant Gupta, Jebish Purbey, Ashay Srivastava, Subhasya TippaReddy, Arvind Reddy Bobbili, Suraj Telugara Chandrashekhar, Modabbir Adeeb, Srinadh Vura, Hamza Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2504.11952)  

**Abstract**: An ideal detection system for machine generated content is supposed to work well on any generator as many more advanced LLMs come into existence day by day. Existing systems often struggle with accurately identifying AI-generated content over shorter texts. Further, not all texts might be entirely authored by a human or LLM, hence we focused more over partial cases i.e human-LLM co-authored texts. Our paper introduces a set of models built for the task of token classification which are trained on an extensive collection of human-machine co-authored texts, which performed well over texts of unseen domains, unseen generators, texts by non-native speakers and those with adversarial inputs. We also introduce a new dataset of over 2.4M such texts mostly co-authored by several popular proprietary LLMs over 23 languages. We also present findings of our models' performance over each texts of each domain and generator. Additional findings include comparison of performance against each adversarial method, length of input texts and characteristics of generated texts compared to the original human authored texts. 

---
# An LLM-as-a-judge Approach for Scalable Gender-Neutral Translation Evaluation 

**Authors**: Andrea Piergentili, Beatrice Savoldi, Matteo Negri, Luisa Bentivogli  

**Link**: [PDF](https://arxiv.org/pdf/2504.11934)  

**Abstract**: Gender-neutral translation (GNT) aims to avoid expressing the gender of human referents when the source text lacks explicit cues about the gender of those referents. Evaluating GNT automatically is particularly challenging, with current solutions being limited to monolingual classifiers. Such solutions are not ideal because they do not factor in the source sentence and require dedicated data and fine-tuning to scale to new languages. In this work, we address such limitations by investigating the use of large language models (LLMs) as evaluators of GNT. Specifically, we explore two prompting approaches: one in which LLMs generate sentence-level assessments only, and another, akin to a chain-of-thought approach, where they first produce detailed phrase-level annotations before a sentence-level judgment. Through extensive experiments on multiple languages with five models, both open and proprietary, we show that LLMs can serve as evaluators of GNT. Moreover, we find that prompting for phrase-level annotations before sentence-level assessments consistently improves the accuracy of all models, providing a better and more scalable alternative to current solutions. 

---
# Finding Flawed Fictions: Evaluating Complex Reasoning in Language Models via Plot Hole Detection 

**Authors**: Kabir Ahuja, Melanie Sclar, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2504.11900)  

**Abstract**: Stories are a fundamental aspect of human experience. Engaging deeply with stories and spotting plot holes -- inconsistencies in a storyline that break the internal logic or rules of a story's world -- requires nuanced reasoning skills, including tracking entities and events and their interplay, abstract thinking, pragmatic narrative understanding, commonsense and social reasoning, and theory of mind. As Large Language Models (LLMs) increasingly generate, interpret, and modify text, rigorously assessing their narrative consistency and deeper language understanding becomes critical. However, existing benchmarks focus mainly on surface-level comprehension. In this work, we propose plot hole detection in stories as a proxy to evaluate language understanding and reasoning in LLMs. We introduce FlawedFictionsMaker, a novel algorithm to controllably and carefully synthesize plot holes in human-written stories. Using this algorithm, we construct a benchmark to evaluate LLMs' plot hole detection abilities in stories -- FlawedFictions -- , which is robust to contamination, with human filtering ensuring high quality. We find that state-of-the-art LLMs struggle in accurately solving FlawedFictions regardless of the reasoning effort allowed, with performance significantly degrading as story length increases. Finally, we show that LLM-based story summarization and story generation are prone to introducing plot holes, with more than 50% and 100% increases in plot hole detection rates with respect to human-written originals. 

---
# FiSMiness: A Finite State Machine Based Paradigm for Emotional Support Conversations 

**Authors**: Yue Zhao, Qingqing Gu, Xiaoyu Wang, Teng Chen, Zhonglin Jiang, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.11837)  

**Abstract**: Emotional support conversation (ESC) aims to alleviate the emotional distress of individuals through effective conversations. Although large language models (LLMs) have obtained remarkable progress on ESC, most of these studies might not define the diagram from the state model perspective, therefore providing a suboptimal solution for long-term satisfaction. To address such an issue, we leverage the Finite State Machine (FSM) on LLMs, and propose a framework called FiSMiness. Our framework allows a single LLM to bootstrap the planning during ESC, and self-reason the seeker's emotion, support strategy and the final response upon each conversational turn. Substantial experiments on ESC datasets suggest that FiSMiness outperforms many baselines, including direct inference, self-refine, chain of thought, finetuning, and external-assisted methods, even those with many more parameters. 

---
# Could Thinking Multilingually Empower LLM Reasoning? 

**Authors**: Changjiang Gao, Xu Huang, Wenhao Zhu, Shujian Huang, Lei Li, Fei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11833)  

**Abstract**: Previous work indicates that large language models exhibit a significant "English bias", i.e. they often perform better when tasks are presented in English. Interestingly, we have observed that using certain other languages in reasoning tasks can yield better performance than English. However, this phenomenon remains under-explored. In this paper, we explore the upper bound of harnessing multilingualism in reasoning tasks, suggesting that multilingual reasoning promises significantly (by nearly 10 Acc@$k$ points) and robustly (tolerance for variations in translation quality and language choice) higher upper bounds than English-only reasoning. Besides analyzing the reason behind the upper bound and challenges in reaching it, we also find that common answer selection methods cannot achieve this upper bound, due to their limitations and biases. These insights could pave the way for future research aimed at fully harnessing the potential of multilingual reasoning in LLMs. 

---
# Déjà Vu: Multilingual LLM Evaluation through the Lens of Machine Translation Evaluation 

**Authors**: Julia Kreutzer, Eleftheria Briakou, Sweta Agrawal, Marzieh Fadaee, Kocmi Tom  

**Link**: [PDF](https://arxiv.org/pdf/2504.11829)  

**Abstract**: Generation capabilities and language coverage of multilingual large language models (mLLMs) are advancing rapidly. However, evaluation practices for generative abilities of mLLMs are still lacking comprehensiveness, scientific rigor, and consistent adoption across research labs, which undermines their potential to meaningfully guide mLLM development. We draw parallels with machine translation (MT) evaluation, a field that faced similar challenges and has, over decades, developed transparent reporting standards and reliable evaluations for multilingual generative models. Through targeted experiments across key stages of the generative evaluation pipeline, we demonstrate how best practices from MT evaluation can deepen the understanding of quality differences between models. Additionally, we identify essential components for robust meta-evaluation of mLLMs, ensuring the evaluation methods themselves are rigorously assessed. We distill these insights into a checklist of actionable recommendations for mLLM research and development. 

---
# ARWI: Arabic Write and Improve 

**Authors**: Kirill Chirkunov, Bashar Alhafni, Chatrine Qwaider, Nizar Habash, Ted Briscoe  

**Link**: [PDF](https://arxiv.org/pdf/2504.11814)  

**Abstract**: Although Arabic is spoken by over 400 million people, advanced Arabic writing assistance tools remain limited. To address this gap, we present ARWI, a new writing assistant that helps learners improve essay writing in Modern Standard Arabic. ARWI is the first publicly available Arabic writing assistant to include a prompt database for different proficiency levels, an Arabic text editor, state-of-the-art grammatical error detection and correction, and automated essay scoring aligned with the Common European Framework of Reference standards for language attainment. Moreover, ARWI can be used to gather a growing auto-annotated corpus, facilitating further research on Arabic grammar correction and essay scoring, as well as profiling patterns of errors made by native speakers and non-native learners. A preliminary user study shows that ARWI provides actionable feedback, helping learners identify grammatical gaps, assess language proficiency, and guide improvement. 

---
# Efficient and Adaptive Simultaneous Speech Translation with Fully Unidirectional Architecture 

**Authors**: Biao Fu, Donglei Yu, Minpeng Liao, Chengxi Li, Yidong Chen, Kai Fan, Xiaodong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.11809)  

**Abstract**: Simultaneous speech translation (SimulST) produces translations incrementally while processing partial speech input. Although large language models (LLMs) have showcased strong capabilities in offline translation tasks, applying them to SimulST poses notable challenges. Existing LLM-based SimulST approaches either incur significant computational overhead due to repeated encoding of bidirectional speech encoder, or they depend on a fixed read/write policy, limiting the efficiency and performance. In this work, we introduce Efficient and Adaptive Simultaneous Speech Translation (EASiST) with fully unidirectional architecture, including both speech encoder and LLM. EASiST includes a multi-latency data curation strategy to generate semantically aligned SimulST training samples and redefines SimulST as an interleaved generation task with explicit read/write tokens. To facilitate adaptive inference, we incorporate a lightweight policy head that dynamically predicts read/write actions. Additionally, we employ a multi-stage training strategy to align speech-text modalities and optimize both translation and policy behavior. Experiments on the MuST-C En$\rightarrow$De and En$\rightarrow$Es datasets demonstrate that EASiST offers superior latency-quality trade-offs compared to several strong baselines. 

---
# Selective Attention Federated Learning: Improving Privacy and Efficiency for Clinical Text Classification 

**Authors**: Yue Li, Lihong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11793)  

**Abstract**: Federated Learning (FL) faces major challenges regarding communication overhead and model privacy when training large language models (LLMs), especially in healthcare applications. To address these, we introduce Selective Attention Federated Learning (SAFL), a novel approach that dynamically fine-tunes only those transformer layers identified as attention-critical. By employing attention patterns to determine layer importance, SAFL significantly reduces communication bandwidth and enhances differential privacy resilience. Evaluations on clinical NLP benchmarks (i2b2 Clinical Concept Extraction and MIMIC-III discharge summaries) demonstrate that SAFL achieves competitive performance with centralized models while substantially improving communication efficiency and privacy preservation. 

---
# Enhancing Web Agents with Explicit Rollback Mechanisms 

**Authors**: Zhisong Zhang, Tianqing Fang, Kaixin Ma, Wenhao Yu, Hongming Zhang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11788)  

**Abstract**: With recent advancements in large language models, web agents have been greatly improved. However, dealing with complex and dynamic web environments requires more advanced planning and search abilities. Previous studies usually adopt a greedy one-way search strategy, which may struggle to recover from erroneous states. In this work, we enhance web agents with an explicit rollback mechanism, enabling the agent to revert back to a previous state in its navigation trajectory. This mechanism gives the model the flexibility to directly control the search process, leading to an effective and efficient web navigation method. We conduct experiments on two live web navigation benchmarks with zero-shot and fine-tuning settings. The results demonstrate the effectiveness of our proposed approach. 

---
# Unsupervised Classification of English Words Based on Phonological Information: Discovery of Germanic and Latinate Clusters 

**Authors**: Takashi Morita, Timothy J. O'Donnell  

**Link**: [PDF](https://arxiv.org/pdf/2504.11770)  

**Abstract**: Cross-linguistically, native words and loanwords follow different phonological rules. In English, for example, words of Germanic and Latinate origin exhibit different stress patterns, and a certain syntactic structure is exclusive to Germanic verbs. When seeing them as a cognitive model, however, such etymology-based generalizations face challenges in terms of learnability, since the historical origins of words are presumably inaccessible information for general language learners. In this study, we present computational evidence indicating that the Germanic-Latinate distinction in the English lexicon is learnable from the phonotactic information of individual words. Specifically, we performed an unsupervised clustering on corpus-extracted words, and the resulting word clusters largely aligned with the etymological distinction. The model-discovered clusters also recovered various linguistic generalizations documented in the previous literature regarding the corresponding etymological classes. Moreover, our findings also uncovered previously unrecognized features of the quasi-etymological clusters, offering novel hypotheses for future experimental studies. 

---
# Higher-Order Binding of Language Model Virtual Personas: a Study on Approximating Political Partisan Misperceptions 

**Authors**: Minwoo Kang, Suhong Moon, Seung Hyeong Lee, Ayush Raj, Joseph Suh, David M. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11673)  

**Abstract**: Large language models (LLMs) are increasingly capable of simulating human behavior, offering cost-effective ways to estimate user responses during the early phases of survey design. While previous studies have examined whether models can reflect individual opinions or attitudes, we argue that a \emph{higher-order} binding of virtual personas requires successfully approximating not only the opinions of a user as an identified member of a group, but also the nuanced ways in which that user perceives and evaluates those outside the group. In particular, faithfully simulating how humans perceive different social groups is critical for applying LLMs to various political science studies, including timely topics on polarization dynamics, inter-group conflict, and democratic backsliding. To this end, we propose a novel methodology for constructing virtual personas with synthetic user ``backstories" generated as extended, multi-turn interview transcripts. Our generated backstories are longer, rich in detail, and consistent in authentically describing a singular individual, compared to previous methods. We show that virtual personas conditioned on our backstories closely replicate human response distributions (up to an 87\% improvement as measured by Wasserstein Distance) and produce effect sizes that closely match those observed in the original studies. Altogether, our work extends the applicability of LLMs beyond estimating individual self-opinions, enabling their use in a broader range of human studies. 

---
# Improving Instruct Models for Free: A Study on Partial Adaptation 

**Authors**: Ozan İrsoy, Pengxiang Cheng, Jennifer L. Chen, Daniel Preoţiuc-Pietro, Shiyue Zhang, Duccio Pappadopulo  

**Link**: [PDF](https://arxiv.org/pdf/2504.11626)  

**Abstract**: Instruct models, obtained from various instruction tuning or post-training steps, are commonly deemed superior and more usable than their base counterpart. While the model gains instruction following ability, instruction tuning may lead to forgetting the knowledge from pre-training or it may encourage the model being overly conversational or verbose. This, in turn, can lead to degradation of in-context few-shot learning performance. In this work, we study the performance trajectory between base and instruct models by scaling down the strength of instruction-tuning via the partial adaption method. We show that, across several model families and model sizes, reducing the strength of instruction-tuning results in material improvement on a few-shot in-context learning benchmark covering a variety of classic natural language tasks. This comes at the cost of losing some degree of instruction following ability as measured by AlpacaEval. Our study shines light on the potential trade-off between in-context learning and instruction following abilities that is worth considering in practice. 

---
# AskQE: Question Answering as Automatic Evaluation for Machine Translation 

**Authors**: Dayeon Ki, Kevin Duh, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2504.11582)  

**Abstract**: How can a monolingual English speaker determine whether an automatic translation in French is good enough to be shared? Existing MT error detection and quality estimation (QE) techniques do not address this practical scenario. We introduce AskQE, a question generation and answering framework designed to detect critical MT errors and provide actionable feedback, helping users decide whether to accept or reject MT outputs even without the knowledge of the target language. Using ContraTICO, a dataset of contrastive synthetic MT errors in the COVID-19 domain, we explore design choices for AskQE and develop an optimized version relying on LLaMA-3 70B and entailed facts to guide question generation. We evaluate the resulting system on the BioMQM dataset of naturally occurring MT errors, where AskQE has higher Kendall's Tau correlation and decision accuracy with human ratings compared to other QE metrics. 

---
# ReTool: Reinforcement Learning for Strategic Tool Use in LLMs 

**Authors**: Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, Wanjun Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2504.11536)  

**Abstract**: While reasoning models (e.g., DeepSeek R1) trained with reinforcement learning (RL), excel in textual reasoning, they struggle in scenarios requiring structured problem-solving, such as geometric reasoning, concise computation, or complex equation solving-areas where computational tools like code interpreters (CI) demonstrate distinct advantages. To bridge this gap, we propose ReTool, which enhances long-form reasoning with tool-integrated learning, including two key features: (1) dynamic interleaving of real-time code execution within natural language reasoning processes, and (2) an automated RL paradigm that allows policy rollouts with multi-turn real-time code execution and teaches the model in learning when and how to invoke tools based on outcome feedback. ReTool employs a systematic training framework, beginning with synthetic cold-start data generation to produce code-augmented long-form reasoning traces for fine-tuning base models. Subsequent RL training leverages task outcomes as rewards to iteratively refine the model's tool use strategy, enabling autonomous discovery of optimal tool invocation patterns without human priors. Experiments on the challenging MATH Olympiad benchmark AIME demonstrate ReTool's superiority: Our 32B model achieves 67% accuracy with 400 training steps, outperforming text-based RL baseline (40% accuracy, 1080 steps) in efficiency and performance. Remarkably, ReTool-32B attains 72.5% accuracy in extended settings, surpassing OpenAI's o1-preview by 27.9%. Further analysis reveals emergent behaviors such as code self-correction, signaling an ''aha moment'' in which the model autonomously masters adaptive tool use. These findings highlight the promise of outcome-driven tool integration for advancing complex mathematical reasoning and offer new insights into hybrid neuro-symbolic systems. 

---
# SFT or RL? An Early Investigation into Training R1-Like Reasoning Large Vision-Language Models 

**Authors**: Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, Cihang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.11468)  

**Abstract**: This work revisits the dominant supervised fine-tuning (SFT) then reinforcement learning (RL) paradigm for training Large Vision-Language Models (LVLMs), and reveals a key finding: SFT can significantly undermine subsequent RL by inducing ``pseudo reasoning paths'' imitated from expert models. While these paths may resemble the native reasoning paths of RL models, they often involve prolonged, hesitant, less informative steps, and incorrect reasoning. To systematically study this effect, we introduce VLAA-Thinking, a new multimodal dataset designed to support reasoning in LVLMs. Constructed via a six-step pipeline involving captioning, reasoning distillation, answer rewrite and verification, VLAA-Thinking comprises high-quality, step-by-step visual reasoning traces for SFT, along with a more challenging RL split from the same data source. Using this dataset, we conduct extensive experiments comparing SFT, RL and their combinations. Results show that while SFT helps models learn reasoning formats, it often locks aligned models into imitative, rigid reasoning modes that impede further learning. In contrast, building on the Group Relative Policy Optimization (GRPO) with a novel mixed reward module integrating both perception and cognition signals, our RL approach fosters more genuine, adaptive reasoning behavior. Notably, our model VLAA-Thinker, based on Qwen2.5VL 3B, achieves top-1 performance on Open LMM Reasoning Leaderboard (this https URL) among 4B scale LVLMs, surpassing the previous state-of-the-art by 1.8%. We hope our findings provide valuable insights in developing reasoning-capable LVLMs and can inform future research in this area. 

---
# Dysarthria Normalization via Local Lie Group Transformations for Robust ASR 

**Authors**: Mikhail Osipov  

**Link**: [PDF](https://arxiv.org/pdf/2504.12279)  

**Abstract**: We present a geometry-driven method for normalizing dysarthric speech using local Lie group transformations of spectrograms. Time, frequency, and amplitude distortions are modeled as smooth, invertible deformations, parameterized by scalar fields and applied via exponential maps. A neural network is trained to infer these fields from synthetic distortions of typical speech-without using any pathological data. At test time, the model applies an approximate inverse to real dysarthric inputs. Despite zero-shot generalization, we observe substantial ASR gains, including up to 16 percentage points WER reduction on challenging TORGO samples, with no degradation on clean speech. This work introduces a principled, interpretable approach for robust speech recognition under motor speech disorders 

---
# Advancing Arabic Speech Recognition Through Large-Scale Weakly Supervised Learning 

**Authors**: Mahmoud Salhab, Marwan Elghitany, Shameed Sait, Syed Sibghat Ullah, Mohammad Abusheikh, Hasan Abusheikh  

**Link**: [PDF](https://arxiv.org/pdf/2504.12254)  

**Abstract**: Automatic speech recognition (ASR) is crucial for human-machine interaction in diverse applications like conversational agents, industrial robotics, call center automation, and automated subtitling. However, developing high-performance ASR models remains challenging, particularly for low-resource languages like Arabic, due to the scarcity of large, labeled speech datasets, which are costly and labor-intensive to produce. In this work, we employ weakly supervised learning to train an Arabic ASR model using the Conformer architecture. Our model is trained from scratch on 15,000 hours of weakly annotated speech data covering both Modern Standard Arabic (MSA) and Dialectal Arabic (DA), eliminating the need for costly manual transcriptions. Despite the absence of human-verified labels, our approach attains state-of-the-art (SOTA) performance, exceeding all previous efforts in the field of Arabic ASR on the standard benchmarks. By demonstrating the effectiveness of weak supervision as a scalable, cost-efficient alternative to traditional supervised approaches, paving the way for improved ASR systems in low resource settings. 

---
# Watermarking Needs Input Repetition Masking 

**Authors**: David Khachaturov, Robert Mullins, Ilia Shumailov, Sumanth Dathathri  

**Link**: [PDF](https://arxiv.org/pdf/2504.12229)  

**Abstract**: Recent advancements in Large Language Models (LLMs) raised concerns over potential misuse, such as for spreading misinformation. In response two counter measures emerged: machine learning-based detectors that predict if text is synthetic, and LLM watermarking, which subtly marks generated text for identification and attribution. Meanwhile, humans are known to adjust language to their conversational partners both syntactically and lexically. By implication, it is possible that humans or unwatermarked LLMs could unintentionally mimic properties of LLM generated text, making counter measures unreliable. In this work we investigate the extent to which such conversational adaptation happens. We call the concept $\textit{mimicry}$ and demonstrate that both humans and LLMs end up mimicking, including the watermarking signal even in seemingly improbable settings. This challenges current academic assumptions and suggests that for long-term watermarking to be reliable, the likelihood of false positives needs to be significantly lower, while longer word sequences should be used for seeding watermarking mechanisms. 

---
# Efficient Contrastive Decoding with Probabilistic Hallucination Detection - Mitigating Hallucinations in Large Vision Language Models - 

**Authors**: Laura Fieback, Nishilkumar Balar, Jakob Spiegelberg, Hanno Gottschalk  

**Link**: [PDF](https://arxiv.org/pdf/2504.12137)  

**Abstract**: Despite recent advances in Large Vision Language Models (LVLMs), these models still suffer from generating hallucinatory responses that do not align with the visual input provided. To mitigate such hallucinations, we introduce Efficient Contrastive Decoding (ECD), a simple method that leverages probabilistic hallucination detection to shift the output distribution towards contextually accurate answers at inference time. By contrasting token probabilities and hallucination scores, ECD subtracts hallucinated concepts from the original distribution, effectively suppressing hallucinations. Notably, our proposed method can be applied to any open-source LVLM and does not require additional LVLM training. We evaluate our method on several benchmark datasets and across different LVLMs. Our experiments show that ECD effectively mitigates hallucinations, outperforming state-of-the-art methods with respect to performance on LVLM benchmarks and computation time. 

---
# ADAT: Time-Series-Aware Adaptive Transformer Architecture for Sign Language Translation 

**Authors**: Nada Shahin, Leila Ismail  

**Link**: [PDF](https://arxiv.org/pdf/2504.11942)  

**Abstract**: Current sign language machine translation systems rely on recognizing hand movements, facial expressions and body postures, and natural language processing, to convert signs into text. Recent approaches use Transformer architectures to model long-range dependencies via positional encoding. However, they lack accuracy in recognizing fine-grained, short-range temporal dependencies between gestures captured at high frame rates. Moreover, their high computational complexity leads to inefficient training. To mitigate these issues, we propose an Adaptive Transformer (ADAT), which incorporates components for enhanced feature extraction and adaptive feature weighting through a gating mechanism to emphasize contextually relevant features while reducing training overhead and maintaining translation accuracy. To evaluate ADAT, we introduce MedASL, the first public medical American Sign Language dataset. In sign-to-gloss-to-text experiments, ADAT outperforms the encoder-decoder transformer, improving BLEU-4 accuracy by 0.1% while reducing training time by 14.33% on PHOENIX14T and 3.24% on MedASL. In sign-to-text experiments, it improves accuracy by 8.7% and reduces training time by 2.8% on PHOENIX14T and achieves 4.7% higher accuracy and 7.17% faster training on MedASL. Compared to encoder-only and decoder-only baselines in sign-to-text, ADAT is at least 6.8% more accurate despite being up to 12.1% slower due to its dual-stream structure. 

---
# Rethinking LLM-Based Recommendations: A Query Generation-Based, Training-Free Approach 

**Authors**: Donghee Han, Hwanjun Song, Mun Yong Yi  

**Link**: [PDF](https://arxiv.org/pdf/2504.11889)  

**Abstract**: Existing large language model LLM-based recommendation methods face several challenges, including inefficiency in handling large candidate pools, sensitivity to item order within prompts ("lost in the middle" phenomenon) poor scalability, and unrealistic evaluation due to random negative sampling. To address these issues, we propose a Query-to-Recommendation approach that leverages LLMs to generate personalized queries for retrieving relevant items from the entire candidate pool, eliminating the need for candidate pre-selection. This method can be integrated into an ID-based recommendation system without additional training, enhances recommendation performance and diversity through LLMs' world knowledge, and performs well even for less popular item groups. Experiments on three datasets show up to 57 percent improvement, with an average gain of 31 percent, demonstrating strong zero-shot performance and further gains when ensembled with existing models. 

---
# Evaluating the Goal-Directedness of Large Language Models 

**Authors**: Tom Everitt, Cristina Garbacea, Alexis Bellot, Jonathan Richens, Henry Papadatos, Siméon Campos, Rohin Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.11844)  

**Abstract**: To what extent do LLMs use their capabilities towards their given goal? We take this as a measure of their goal-directedness. We evaluate goal-directedness on tasks that require information gathering, cognitive effort, and plan execution, where we use subtasks to infer each model's relevant capabilities. Our evaluations of LLMs from Google DeepMind, OpenAI, and Anthropic show that goal-directedness is relatively consistent across tasks, differs from task performance, and is only moderately sensitive to motivational prompts. Notably, most models are not fully goal-directed. We hope our goal-directedness evaluations will enable better monitoring of LLM progress, and enable more deliberate design choices of agentic properties in LLMs. 

---
# Climbing the Ladder of Reasoning: What LLMs Can-and Still Can't-Solve after SFT? 

**Authors**: Yiyou Sun, Georgia Zhou, Hao Wang, Dacheng Li, Nouha Dziri, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.11741)  

**Abstract**: Recent supervised fine-tuning (SFT) approaches have significantly improved language models' performance on mathematical reasoning tasks, even when models are trained at a small scale. However, the specific capabilities enhanced through such fine-tuning remain poorly understood. In this paper, we conduct a detailed analysis of model performance on the AIME24 dataset to understand how reasoning capabilities evolve. We discover a ladder-like structure in problem difficulty, categorize questions into four tiers (Easy, Medium, Hard, and Extremely Hard (Exh)), and identify the specific requirements for advancing between tiers. We find that progression from Easy to Medium tier requires adopting an R1 reasoning style with minimal SFT (500-1K instances), while Hard-level questions suffer from frequent model's errors at each step of the reasoning chain, with accuracy plateauing at around 65% despite logarithmic scaling. Exh-level questions present a fundamentally different challenge; they require unconventional problem-solving skills that current models uniformly struggle with. Additional findings reveal that carefully curated small-scale datasets offer limited advantage-scaling dataset size proves far more effective. Our analysis provides a clearer roadmap for advancing language model capabilities in mathematical reasoning. 

---
# The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation 

**Authors**: Bingjie Gao, Xinyu Gao, Xiaoxue Wu, Yujie Zhou, Yu Qiao, Li Niu, Xinyuan Chen, Yaohui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11739)  

**Abstract**: The evolution of Text-to-video (T2V) generative models, trained on large-scale datasets, has been marked by significant progress. However, the sensitivity of T2V generative models to input prompts highlights the critical role of prompt design in influencing generative outcomes. Prior research has predominantly relied on Large Language Models (LLMs) to align user-provided prompts with the distribution of training prompts, albeit without tailored guidance encompassing prompt vocabulary and sentence structure nuances. To this end, we introduce \textbf{RAPO}, a novel \textbf{R}etrieval-\textbf{A}ugmented \textbf{P}rompt \textbf{O}ptimization framework. In order to address potential inaccuracies and ambiguous details generated by LLM-generated prompts. RAPO refines the naive prompts through dual optimization branches, selecting the superior prompt for T2V generation. The first branch augments user prompts with diverse modifiers extracted from a learned relational graph, refining them to align with the format of training prompts via a fine-tuned LLM. Conversely, the second branch rewrites the naive prompt using a pre-trained LLM following a well-defined instruction set. Extensive experiments demonstrate that RAPO can effectively enhance both the static and dynamic dimensions of generated videos, demonstrating the significance of prompt optimization for user-provided prompts. Project website: \href{this https URL}{GitHub}. 

---
# GraphicBench: A Planning Benchmark for Graphic Design with Language Agents 

**Authors**: Dayeon Ki, Tianyi Zhou, Marine Carpuat, Gang Wu, Puneet Mathur, Viswanathan Swaminathan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11571)  

**Abstract**: Large Language Model (LLM)-powered agents have unlocked new possibilities for automating human tasks. While prior work has focused on well-defined tasks with specified goals, the capabilities of agents in creative design tasks with open-ended goals remain underexplored. We introduce GraphicBench, a new planning benchmark for graphic design that covers 1,079 user queries and input images across four design types. We further present GraphicTown, an LLM agent framework with three design experts and 46 actions (tools) to choose from for executing each step of the planned workflows in web environments. Experiments with six LLMs demonstrate their ability to generate workflows that integrate both explicit design constraints from user queries and implicit commonsense constraints. However, these workflows often do not lead to successful execution outcomes, primarily due to challenges in: (1) reasoning about spatial relationships, (2) coordinating global dependencies across experts, and (3) retrieving the most appropriate action per step. We envision GraphicBench as a challenging yet valuable testbed for advancing LLM-agent planning and execution in creative design tasks. 

---
# HypoBench: Towards Systematic and Principled Benchmarking for Hypothesis Generation 

**Authors**: Haokun Liu, Sicong Huang, Jingyu Hu, Yangqiaoyu Zhou, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11524)  

**Abstract**: There is growing interest in hypothesis generation with large language models (LLMs). However, fundamental questions remain: what makes a good hypothesis, and how can we systematically evaluate methods for hypothesis generation? To address this, we introduce HypoBench, a novel benchmark designed to evaluate LLMs and hypothesis generation methods across multiple aspects, including practical utility, generalizability, and hypothesis discovery rate. HypoBench includes 7 real-world tasks and 5 synthetic tasks with 194 distinct datasets. We evaluate four state-of-the-art LLMs combined with six existing hypothesis-generation methods. Overall, our results suggest that existing methods are capable of discovering valid and novel patterns in the data. However, the results from synthetic datasets indicate that there is still significant room for improvement, as current hypothesis generation methods do not fully uncover all relevant or meaningful patterns. Specifically, in synthetic settings, as task difficulty increases, performance significantly drops, with best models and methods only recovering 38.8% of the ground-truth hypotheses. These findings highlight challenges in hypothesis generation and demonstrate that HypoBench serves as a valuable resource for improving AI systems designed to assist scientific discovery. 

---
# Graph-Driven Multimodal Feature Learning Framework for Apparent Personality Assessment 

**Authors**: Kangsheng Wang, Chengwei Ye, Huanzhen Zhang, Linuo Xu, Shuyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11515)  

**Abstract**: Predicting personality traits automatically has become a challenging problem in computer vision. This paper introduces an innovative multimodal feature learning framework for personality analysis in short video clips. For visual processing, we construct a facial graph and design a Geo-based two-stream network incorporating an attention mechanism, leveraging both Graph Convolutional Networks (GCN) and Convolutional Neural Networks (CNN) to capture static facial expressions. Additionally, ResNet18 and VGGFace networks are employed to extract global scene and facial appearance features at the frame level. To capture dynamic temporal information, we integrate a BiGRU with a temporal attention module for extracting salient frame representations. To enhance the model's robustness, we incorporate the VGGish CNN for audio-based features and XLM-Roberta for text-based features. Finally, a multimodal channel attention mechanism is introduced to integrate different modalities, and a Multi-Layer Perceptron (MLP) regression model is used to predict personality traits. Experimental results confirm that our proposed framework surpasses existing state-of-the-art approaches in performance. 

---
# Language and Knowledge Representation: A Stratified Approach 

**Authors**: Mayukh Bagchi  

**Link**: [PDF](https://arxiv.org/pdf/2504.11492)  

**Abstract**: The thesis proposes the problem of representation heterogeneity to emphasize the fact that heterogeneity is an intrinsic property of any representation, wherein, different observers encode different representations of the same target reality in a stratified manner using different concepts, language and knowledge (as well as data). The thesis then advances a top-down solution approach to the above stratified problem of representation heterogeneity in terms of several solution components, namely: (i) a representation formalism stratified into concept level, language level, knowledge level and data level to accommodate representation heterogeneity, (ii) a top-down language representation using Universal Knowledge Core (UKC), UKC namespaces and domain languages to tackle the conceptual and language level heterogeneity, (iii) a top-down knowledge representation using the notions of language teleontology and knowledge teleontology to tackle the knowledge level heterogeneity, (iv) the usage and further development of the existing LiveKnowledge catalog for enforcing iterative reuse and sharing of language and knowledge representations, and, (v) the kTelos methodology integrating the solution components above to iteratively generate the language and knowledge representations absolving representation heterogeneity. The thesis also includes proof-of-concepts of the language and knowledge representations developed for two international research projects - DataScientia (data catalogs) and JIDEP (materials modelling). Finally, the thesis concludes with future lines of research. 

---
# Semantic Matters: Multimodal Features for Affective Analysis 

**Authors**: Tobias Hallmen, Robin-Nico Kampa, Fabian Deuser, Norbert Oswald, Elisabeth André  

**Link**: [PDF](https://arxiv.org/pdf/2504.11460)  

**Abstract**: In this study, we present our methodology for two tasks: the Behavioural Ambivalence/Hesitancy (BAH) Recognition Challenge and the Emotional Mimicry Intensity (EMI) Estimation Challenge, both conducted as part of the 8th Workshop and Competition on Affective & Behavior Analysis in-the-wild. Building on previous work, we utilize a Wav2Vec 2.0 model pre-trained on a large podcast dataset to extract various audio features, capturing both linguistic and paralinguistic information. Our approach incorporates a valence-arousal-dominance (VAD) module derived from Wav2Vec 2.0, a BERT-like encoder, and a vision transformer (ViT) with predictions subsequently processed through a long short-term memory (LSTM) architecture for temporal modeling. In this iteration, we integrate the textual and visual modality into our analysis, recognizing that semantic content provides valuable contextual cues and underscoring that the meaning of speech often conveys more critical insights than its acoustic counterpart alone. Fusing in the vision modality helps in some cases to interpret the textual modality more precisely. This combined approach yields significant performance improvements over baseline methods. 

---
# From Conceptual Data Models to Multimodal Representation 

**Authors**: Peter Stockinger  

**Link**: [PDF](https://arxiv.org/pdf/2504.11459)  

**Abstract**: 1) Introduction and Conceptual Framework: This document explores the concept of information design by dividing it into two major practices: defining the meaning of a corpus of textual data and its visual or multimodal representation. It draws on expertise in enriching textual corpora, particularly audiovisual ones, and transforming them into multiple narrative formats. The text highlights a crucial distinction between the semantic content of a domain and the modalities of its graphic expression, illustrating this approach with concepts rooted in structural semiotics and linguistics traditions.
2) Modeling and Conceptual Design:  The article emphasizes the importance of semantic modeling, often achieved through conceptual networks or graphs. These tools enable the structuring of knowledge within a domain by accounting for relationships between concepts, contexts of use, and specific objectives. Stockinger also highlights the constraints and challenges involved in creating dynamic and adaptable models, integrating elements such as thesauri or interoperable ontologies to facilitate the analysis and publication of complex corpora.
3) Applications and Multimodal Visualization:  The text concludes by examining the practical application of these models in work environments like OKAPI, developed to analyze, publish, and reuse audiovisual data. It also discusses innovative approaches such as visual storytelling and document reengineering, which involve transforming existing content into new resources tailored to various contexts. These methods emphasize interoperability, flexibility, and the intelligence of communication systems, paving the way for richer and more collaborative use of digital data. The content of this document was presented during the "Semiotics of Information Design" Day organized by Anne Beyaert-Geslin of the University of Bordeaux Montaigne (MICA laboratory) on June 21, 2018, in Bordeaux. 

---
