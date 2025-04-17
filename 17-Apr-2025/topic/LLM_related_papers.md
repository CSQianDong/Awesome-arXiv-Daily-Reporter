# Clarifying Ambiguities: on the Role of Ambiguity Types in Prompting Methods for Clarification Generation 

**Authors**: Anfu Tang, Laure Soulier, Vincent Guigue  

**Link**: [PDF](https://arxiv.org/pdf/2504.12113)  

**Abstract**: In information retrieval (IR), providing appropriate clarifications to better understand users' information needs is crucial for building a proactive search-oriented dialogue system. Due to the strong in-context learning ability of large language models (LLMs), recent studies investigate prompting methods to generate clarifications using few-shot or Chain of Thought (CoT) prompts. However, vanilla CoT prompting does not distinguish the characteristics of different information needs, making it difficult to understand how LLMs resolve ambiguities in user queries. In this work, we focus on the concept of ambiguity for clarification, seeking to model and integrate ambiguities in the clarification process. To this end, we comprehensively study the impact of prompting schemes based on reasoning and ambiguity for clarification. The idea is to enhance the reasoning abilities of LLMs by limiting CoT to predict first ambiguity types that can be interpreted as instructions to clarify, then correspondingly generate clarifications. We name this new prompting scheme Ambiguity Type-Chain of Thought (AT-CoT). Experiments are conducted on various datasets containing human-annotated clarifying questions to compare AT-CoT with multiple baselines. We also perform user simulations to implicitly measure the quality of generated clarifications under various IR scenarios. 

---
# Improving LLM Interpretability and Performance via Guided Embedding Refinement for Sequential Recommendation 

**Authors**: Nanshan Jia, Chenfei Yuan, Yuhang Wu, Zeyu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.11658)  

**Abstract**: The fast development of Large Language Models (LLMs) offers growing opportunities to further improve sequential recommendation systems. Yet for some practitioners, integrating LLMs to their existing base recommendation systems raises questions about model interpretability, transparency and related safety. To partly alleviate challenges from these questions, we propose guided embedding refinement, a method that carries out a guided and interpretable usage of LLM to enhance the embeddings associated with the base recommendation system. Instead of directly using LLMs as the backbone of sequential recommendation systems, we utilize them as auxiliary tools to emulate the sales logic of recommendation and generate guided embeddings that capture domain-relevant semantic information on interpretable attributes. Benefiting from the strong generalization capabilities of the guided embedding, we construct refined embedding by using the guided embedding and reduced-dimension version of the base embedding. We then integrate the refined embedding into the recommendation module for training and inference. A range of numerical experiments demonstrate that guided embedding is adaptable to various given existing base embedding models, and generalizes well across different recommendation tasks. The numerical results show that the refined embedding not only improves recommendation performance, achieving approximately $10\%$ to $50\%$ gains in Mean Reciprocal Rank (MRR), Recall rate, and Normalized Discounted Cumulative Gain (NDCG), but also enhances interpretability, as evidenced by case studies. 

---
# Generative Recommendation with Continuous-Token Diffusion 

**Authors**: Haohao Qu, Wenqi Fan, Shanru Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.12007)  

**Abstract**: In recent years, there has been a significant trend toward using large language model (LLM)-based recommender systems (RecSys). Current research primarily focuses on representing complex user-item interactions within a discrete space to align with the inherent discrete nature of language models. However, this approach faces limitations due to its discrete nature: (i) information is often compressed during discretization; (ii) the tokenization and generation for the vast number of users and items in real-world scenarios are constrained by a limited vocabulary. Embracing continuous data presents a promising alternative to enhance expressive capabilities, though this approach is still in its early stages. To address this gap, we propose a novel framework, DeftRec, which incorporates \textbf{de}noising di\textbf{f}fusion models to enable LLM-based RecSys to seamlessly support continuous \textbf{t}oken as input and target. First, we introduce a robust tokenizer with a masking operation and an additive K-way architecture to index users and items, capturing their complex collaborative relationships into continuous tokens. Crucially, we develop a denoising diffusion model to process user preferences within continuous domains by conditioning on reasoning content from pre-trained large language model. During the denoising process, we reformulate the objective to include negative interactions, building a comprehensive understanding of user preferences for effective and accurate recommendation generation. Finally, given a continuous token as output, recommendations can be easily generated through score-based retrieval. Extensive experiments demonstrate the effectiveness of the proposed methods, showing that DeftRec surpasses competitive benchmarks, including both traditional and emerging LLM-based RecSys. 

---
# A New Paradigm of User-Centric Wireless Communication Driven by Large Language Models 

**Authors**: Kuiyuan Ding, Caili Guo, Yang Yang, Wuxia Hu, Yonina C. Eldar  

**Link**: [PDF](https://arxiv.org/pdf/2504.11696)  

**Abstract**: The next generation of wireless communications seeks to deeply integrate artificial intelligence (AI) with user-centric communication networks, with the goal of developing AI-native networks that more accurately address user requirements. The rapid development of large language models (LLMs) offers significant potential in realizing these goals. However, existing efforts that leverage LLMs for wireless communication often overlook the considerable gap between human natural language and the intricacies of real-world communication systems, thus failing to fully exploit the capabilities of LLMs. To address this gap, we propose a novel LLM-driven paradigm for wireless communication that innovatively incorporates the nature language to structured query language (NL2SQL) tool. Specifically, in this paradigm, user personal requirements is the primary focus. Upon receiving a user request, LLMs first analyze the user intent in terms of relevant communication metrics and system parameters. Subsequently, a structured query language (SQL) statement is generated to retrieve the specific parameter values from a high-performance real-time database. We further utilize LLMs to formulate and solve an optimization problem based on the user request and the retrieved parameters. The solution to this optimization problem then drives adjustments in the communication system to fulfill the user's requirements. To validate the feasibility of the proposed paradigm, we present a prototype system. In this prototype, we consider user-request centric semantic communication (URC-SC) system in which a dynamic semantic representation network at the physical layer adapts its encoding depth to meet user requirements. Additionally, two LLMs are employed to analyze user requests and generate SQL statements, respectively. Simulation results demonstrate the effectiveness. 

---
# Optimizing Compound Retrieval Systems 

**Authors**: Harrie Oosterhuis, Rolf Jagerman, Zhen Qin, Xuanhui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12063)  

**Abstract**: Modern retrieval systems do not rely on a single ranking model to construct their rankings. Instead, they generally take a cascading approach where a sequence of ranking models are applied in multiple re-ranking stages. Thereby, they balance the quality of the top-K ranking with computational costs by limiting the number of documents each model re-ranks. However, the cascading approach is not the only way models can interact to form a retrieval system.
We propose the concept of compound retrieval systems as a broader class of retrieval systems that apply multiple prediction models. This encapsulates cascading models but also allows other types of interactions than top-K re-ranking. In particular, we enable interactions with large language models (LLMs) which can provide relative relevance comparisons. We focus on the optimization of compound retrieval system design which uniquely involves learning where to apply the component models and how to aggregate their predictions into a final ranking. This work shows how our compound approach can combine the classic BM25 retrieval model with state-of-the-art (pairwise) LLM relevance predictions, while optimizing a given ranking metric and efficiency target. Our experimental results show optimized compound retrieval systems provide better trade-offs between effectiveness and efficiency than cascading approaches, even when applied in a self-supervised manner.
With the introduction of compound retrieval systems, we hope to inspire the information retrieval field to more out-of-the-box thinking on how prediction models can interact to form rankings. 

---
# Rethinking LLM-Based Recommendations: A Query Generation-Based, Training-Free Approach 

**Authors**: Donghee Han, Hwanjun Song, Mun Yong Yi  

**Link**: [PDF](https://arxiv.org/pdf/2504.11889)  

**Abstract**: Existing large language model LLM-based recommendation methods face several challenges, including inefficiency in handling large candidate pools, sensitivity to item order within prompts ("lost in the middle" phenomenon) poor scalability, and unrealistic evaluation due to random negative sampling. To address these issues, we propose a Query-to-Recommendation approach that leverages LLMs to generate personalized queries for retrieving relevant items from the entire candidate pool, eliminating the need for candidate pre-selection. This method can be integrated into an ID-based recommendation system without additional training, enhances recommendation performance and diversity through LLMs' world knowledge, and performs well even for less popular item groups. Experiments on three datasets show up to 57 percent improvement, with an average gain of 31 percent, demonstrating strong zero-shot performance and further gains when ensembled with existing models. 

---
# Gauging Overprecision in LLMs: An Empirical Study 

**Authors**: Adil Bahaj, Hamed Rahimi, Mohamed Chetouani, Mounir Ghogho  

**Link**: [PDF](https://arxiv.org/pdf/2504.12098)  

**Abstract**: Recently, overconfidence in large language models (LLMs) has garnered considerable attention due to its fundamental importance in quantifying the trustworthiness of LLM generation. However, existing approaches prompt the \textit{black box LLMs} to produce their confidence (\textit{verbalized confidence}), which can be subject to many biases and hallucinations. Inspired by a different aspect of overconfidence in cognitive science called \textit{overprecision}, we designed a framework for its study in black box LLMs. This framework contains three main phases: 1) generation, 2) refinement and 3) evaluation. In the generation phase we prompt the LLM to generate answers to numerical questions in the form of intervals with a certain level of confidence. This confidence level is imposed in the prompt and not required for the LLM to generate as in previous approaches. We use various prompting techniques and use the same prompt multiple times to gauge the effects of randomness in the generation process. In the refinement phase, answers from the previous phase are refined to generate better answers. The LLM answers are evaluated and studied in the evaluation phase to understand its internal workings. This study allowed us to gain various insights into LLM overprecision: 1) LLMs are highly uncalibrated for numerical tasks 2) {\color{blue}there is no correlation between the length of the interval and the imposed confidence level, which can be symptomatic of a a) lack of understanding of the concept of confidence or b) inability to adjust self-confidence by following instructions}, {\color{blue}3)} LLM numerical precision differs depending on the task, scale of answer and prompting technique {\color{blue}4) Refinement of answers doesn't improve precision in most cases}. We believe this study offers new perspectives on LLM overconfidence and serves as a strong baseline for overprecision in LLMs. 

---
# What Do Large Language Models Know? Tacit Knowledge as a Potential Causal-Explanatory Structure 

**Authors**: Céline Budding  

**Link**: [PDF](https://arxiv.org/pdf/2504.12187)  

**Abstract**: It is sometimes assumed that Large Language Models (LLMs) know language, or for example that they know that Paris is the capital of France. But what -- if anything -- do LLMs actually know? In this paper, I argue that LLMs can acquire tacit knowledge as defined by Martin Davies (1990). Whereas Davies himself denies that neural networks can acquire tacit knowledge, I demonstrate that certain architectural features of LLMs satisfy the constraints of semantic description, syntactic structure, and causal systematicity. Thus, tacit knowledge may serve as a conceptual framework for describing, explaining, and intervening on LLMs and their behavior. 

---
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
# Multilingual Contextualization of Large Language Models for Document-Level Machine Translation 

**Authors**: Miguel Moura Ramos, Patrick Fernandes, Sweta Agrawal, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2504.12140)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance in sentence-level machine translation, but scaling to document-level translation remains challenging, particularly in modeling long-range dependencies and discourse phenomena across sentences and paragraphs. In this work, we propose a method to improve LLM-based long-document translation through targeted fine-tuning on high-quality document-level data, which we curate and introduce as DocBlocks. Our approach supports multiple translation paradigms, including direct document-to-document and chunk-level translation, by integrating instructions both with and without surrounding context. This enables models to better capture cross-sentence dependencies while maintaining strong sentence-level translation performance. Experimental results show that incorporating multiple translation paradigms improves document-level translation quality and inference speed compared to prompting and agent-based methods. 

---
# LLM-as-a-Judge: Reassessing the Performance of LLMs in Extractive QA 

**Authors**: Xanh Ho, Jiahao Huang, Florian Boudin, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2504.11972)  

**Abstract**: Extractive reading comprehension question answering (QA) datasets are typically evaluated using Exact Match (EM) and F1-score, but these metrics often fail to fully capture model performance. With the success of large language models (LLMs), they have been employed in various tasks, including serving as judges (LLM-as-a-judge). In this paper, we reassess the performance of QA models using LLM-as-a-judge across four reading comprehension QA datasets. We examine different families of LLMs and various answer types to evaluate the effectiveness of LLM-as-a-judge in these tasks. Our results show that LLM-as-a-judge is highly correlated with human judgments and can replace traditional EM/F1 metrics. By using LLM-as-a-judge, the correlation with human judgments improves significantly, from 0.17 (EM) and 0.36 (F1-score) to 0.85. These findings confirm that EM and F1 metrics underestimate the true performance of the QA models. While LLM-as-a-judge is not perfect for more difficult answer types (e.g., job), it still outperforms EM/F1, and we observe no bias issues, such as self-preference, when the same model is used for both the QA and judgment tasks. 

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
# FiSMiness: A Finite State Machine Based Paradigm for Emotional Support Conversations 

**Authors**: Yue Zhao, Qingqing Gu, Xiaoyu Wang, Teng Chen, Zhonglin Jiang, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.11837)  

**Abstract**: Emotional support conversation (ESC) aims to alleviate the emotional distress of individuals through effective conversations. Although large language models (LLMs) have obtained remarkable progress on ESC, most of these studies might not define the diagram from the state model perspective, therefore providing a suboptimal solution for long-term satisfaction. To address such an issue, we leverage the Finite State Machine (FSM) on LLMs, and propose a framework called FiSMiness. Our framework allows a single LLM to bootstrap the planning during ESC, and self-reason the seeker's emotion, support strategy and the final response upon each conversational turn. Substantial experiments on ESC datasets suggest that FiSMiness outperforms many baselines, including direct inference, self-refine, chain of thought, finetuning, and external-assisted methods, even those with many more parameters. 

---
# Déjà Vu: Multilingual LLM Evaluation through the Lens of Machine Translation Evaluation 

**Authors**: Julia Kreutzer, Eleftheria Briakou, Sweta Agrawal, Marzieh Fadaee, Kocmi Tom  

**Link**: [PDF](https://arxiv.org/pdf/2504.11829)  

**Abstract**: Generation capabilities and language coverage of multilingual large language models (mLLMs) are advancing rapidly. However, evaluation practices for generative abilities of mLLMs are still lacking comprehensiveness, scientific rigor, and consistent adoption across research labs, which undermines their potential to meaningfully guide mLLM development. We draw parallels with machine translation (MT) evaluation, a field that faced similar challenges and has, over decades, developed transparent reporting standards and reliable evaluations for multilingual generative models. Through targeted experiments across key stages of the generative evaluation pipeline, we demonstrate how best practices from MT evaluation can deepen the understanding of quality differences between models. Additionally, we identify essential components for robust meta-evaluation of mLLMs, ensuring the evaluation methods themselves are rigorously assessed. We distill these insights into a checklist of actionable recommendations for mLLM research and development. 

---
# Could Thinking Multilingually Empower LLM Reasoning? 

**Authors**: Changjiang Gao, Xu Huang, Wenhao Zhu, Shujian Huang, Lei Li, Fei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11833)  

**Abstract**: Previous work indicates that large language models exhibit a significant "English bias", i.e. they often perform better when tasks are presented in English. Interestingly, we have observed that using certain other languages in reasoning tasks can yield better performance than English. However, this phenomenon remains under-explored. In this paper, we explore the upper bound of harnessing multilingualism in reasoning tasks, suggesting that multilingual reasoning promises significantly (by nearly 10 Acc@$k$ points) and robustly (tolerance for variations in translation quality and language choice) higher upper bounds than English-only reasoning. Besides analyzing the reason behind the upper bound and challenges in reaching it, we also find that common answer selection methods cannot achieve this upper bound, due to their limitations and biases. These insights could pave the way for future research aimed at fully harnessing the potential of multilingual reasoning in LLMs. 

---
# Finding Flawed Fictions: Evaluating Complex Reasoning in Language Models via Plot Hole Detection 

**Authors**: Kabir Ahuja, Melanie Sclar, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2504.11900)  

**Abstract**: Stories are a fundamental aspect of human experience. Engaging deeply with stories and spotting plot holes -- inconsistencies in a storyline that break the internal logic or rules of a story's world -- requires nuanced reasoning skills, including tracking entities and events and their interplay, abstract thinking, pragmatic narrative understanding, commonsense and social reasoning, and theory of mind. As Large Language Models (LLMs) increasingly generate, interpret, and modify text, rigorously assessing their narrative consistency and deeper language understanding becomes critical. However, existing benchmarks focus mainly on surface-level comprehension. In this work, we propose plot hole detection in stories as a proxy to evaluate language understanding and reasoning in LLMs. We introduce FlawedFictionsMaker, a novel algorithm to controllably and carefully synthesize plot holes in human-written stories. Using this algorithm, we construct a benchmark to evaluate LLMs' plot hole detection abilities in stories -- FlawedFictions -- , which is robust to contamination, with human filtering ensuring high quality. We find that state-of-the-art LLMs struggle in accurately solving FlawedFictions regardless of the reasoning effort allowed, with performance significantly degrading as story length increases. Finally, we show that LLM-based story summarization and story generation are prone to introducing plot holes, with more than 50% and 100% increases in plot hole detection rates with respect to human-written originals. 

---
# An LLM-as-a-judge Approach for Scalable Gender-Neutral Translation Evaluation 

**Authors**: Andrea Piergentili, Beatrice Savoldi, Matteo Negri, Luisa Bentivogli  

**Link**: [PDF](https://arxiv.org/pdf/2504.11934)  

**Abstract**: Gender-neutral translation (GNT) aims to avoid expressing the gender of human referents when the source text lacks explicit cues about the gender of those referents. Evaluating GNT automatically is particularly challenging, with current solutions being limited to monolingual classifiers. Such solutions are not ideal because they do not factor in the source sentence and require dedicated data and fine-tuning to scale to new languages. In this work, we address such limitations by investigating the use of large language models (LLMs) as evaluators of GNT. Specifically, we explore two prompting approaches: one in which LLMs generate sentence-level assessments only, and another, akin to a chain-of-thought approach, where they first produce detailed phrase-level annotations before a sentence-level judgment. Through extensive experiments on multiple languages with five models, both open and proprietary, we show that LLMs can serve as evaluators of GNT. Moreover, we find that prompting for phrase-level annotations before sentence-level assessments consistently improves the accuracy of all models, providing a better and more scalable alternative to current solutions. 

---
# Higher-Order Binding of Language Model Virtual Personas: a Study on Approximating Political Partisan Misperceptions 

**Authors**: Minwoo Kang, Suhong Moon, Seung Hyeong Lee, Ayush Raj, Joseph Suh, David M. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11673)  

**Abstract**: Large language models (LLMs) are increasingly capable of simulating human behavior, offering cost-effective ways to estimate user responses during the early phases of survey design. While previous studies have examined whether models can reflect individual opinions or attitudes, we argue that a \emph{higher-order} binding of virtual personas requires successfully approximating not only the opinions of a user as an identified member of a group, but also the nuanced ways in which that user perceives and evaluates those outside the group. In particular, faithfully simulating how humans perceive different social groups is critical for applying LLMs to various political science studies, including timely topics on polarization dynamics, inter-group conflict, and democratic backsliding. To this end, we propose a novel methodology for constructing virtual personas with synthetic user ``backstories" generated as extended, multi-turn interview transcripts. Our generated backstories are longer, rich in detail, and consistent in authentically describing a singular individual, compared to previous methods. We show that virtual personas conditioned on our backstories closely replicate human response distributions (up to an 87\% improvement as measured by Wasserstein Distance) and produce effect sizes that closely match those observed in the original studies. Altogether, our work extends the applicability of LLMs beyond estimating individual self-opinions, enabling their use in a broader range of human studies. 

---
# SFT or RL? An Early Investigation into Training R1-Like Reasoning Large Vision-Language Models 

**Authors**: Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, Cihang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.11468)  

**Abstract**: This work revisits the dominant supervised fine-tuning (SFT) then reinforcement learning (RL) paradigm for training Large Vision-Language Models (LVLMs), and reveals a key finding: SFT can significantly undermine subsequent RL by inducing ``pseudo reasoning paths'' imitated from expert models. While these paths may resemble the native reasoning paths of RL models, they often involve prolonged, hesitant, less informative steps, and incorrect reasoning. To systematically study this effect, we introduce VLAA-Thinking, a new multimodal dataset designed to support reasoning in LVLMs. Constructed via a six-step pipeline involving captioning, reasoning distillation, answer rewrite and verification, VLAA-Thinking comprises high-quality, step-by-step visual reasoning traces for SFT, along with a more challenging RL split from the same data source. Using this dataset, we conduct extensive experiments comparing SFT, RL and their combinations. Results show that while SFT helps models learn reasoning formats, it often locks aligned models into imitative, rigid reasoning modes that impede further learning. In contrast, building on the Group Relative Policy Optimization (GRPO) with a novel mixed reward module integrating both perception and cognition signals, our RL approach fosters more genuine, adaptive reasoning behavior. Notably, our model VLAA-Thinker, based on Qwen2.5VL 3B, achieves top-1 performance on Open LMM Reasoning Leaderboard (this https URL) among 4B scale LVLMs, surpassing the previous state-of-the-art by 1.8%. We hope our findings provide valuable insights in developing reasoning-capable LVLMs and can inform future research in this area. 

---
# Efficient Contrastive Decoding with Probabilistic Hallucination Detection - Mitigating Hallucinations in Large Vision Language Models - 

**Authors**: Laura Fieback, Nishilkumar Balar, Jakob Spiegelberg, Hanno Gottschalk  

**Link**: [PDF](https://arxiv.org/pdf/2504.12137)  

**Abstract**: Despite recent advances in Large Vision Language Models (LVLMs), these models still suffer from generating hallucinatory responses that do not align with the visual input provided. To mitigate such hallucinations, we introduce Efficient Contrastive Decoding (ECD), a simple method that leverages probabilistic hallucination detection to shift the output distribution towards contextually accurate answers at inference time. By contrasting token probabilities and hallucination scores, ECD subtracts hallucinated concepts from the original distribution, effectively suppressing hallucinations. Notably, our proposed method can be applied to any open-source LVLM and does not require additional LVLM training. We evaluate our method on several benchmark datasets and across different LVLMs. Our experiments show that ECD effectively mitigates hallucinations, outperforming state-of-the-art methods with respect to performance on LVLM benchmarks and computation time. 

---
# Evaluating the Goal-Directedness of Large Language Models 

**Authors**: Tom Everitt, Cristina Garbacea, Alexis Bellot, Jonathan Richens, Henry Papadatos, Siméon Campos, Rohin Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.11844)  

**Abstract**: To what extent do LLMs use their capabilities towards their given goal? We take this as a measure of their goal-directedness. We evaluate goal-directedness on tasks that require information gathering, cognitive effort, and plan execution, where we use subtasks to infer each model's relevant capabilities. Our evaluations of LLMs from Google DeepMind, OpenAI, and Anthropic show that goal-directedness is relatively consistent across tasks, differs from task performance, and is only moderately sensitive to motivational prompts. Notably, most models are not fully goal-directed. We hope our goal-directedness evaluations will enable better monitoring of LLM progress, and enable more deliberate design choices of agentic properties in LLMs. 

---
# GraphicBench: A Planning Benchmark for Graphic Design with Language Agents 

**Authors**: Dayeon Ki, Tianyi Zhou, Marine Carpuat, Gang Wu, Puneet Mathur, Viswanathan Swaminathan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11571)  

**Abstract**: Large Language Model (LLM)-powered agents have unlocked new possibilities for automating human tasks. While prior work has focused on well-defined tasks with specified goals, the capabilities of agents in creative design tasks with open-ended goals remain underexplored. We introduce GraphicBench, a new planning benchmark for graphic design that covers 1,079 user queries and input images across four design types. We further present GraphicTown, an LLM agent framework with three design experts and 46 actions (tools) to choose from for executing each step of the planned workflows in web environments. Experiments with six LLMs demonstrate their ability to generate workflows that integrate both explicit design constraints from user queries and implicit commonsense constraints. However, these workflows often do not lead to successful execution outcomes, primarily due to challenges in: (1) reasoning about spatial relationships, (2) coordinating global dependencies across experts, and (3) retrieving the most appropriate action per step. We envision GraphicBench as a challenging yet valuable testbed for advancing LLM-agent planning and execution in creative design tasks. 

---
# ReTool: Reinforcement Learning for Strategic Tool Use in LLMs 

**Authors**: Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, Wanjun Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2504.11536)  

**Abstract**: While reasoning models (e.g., DeepSeek R1) trained with reinforcement learning (RL), excel in textual reasoning, they struggle in scenarios requiring structured problem-solving, such as geometric reasoning, concise computation, or complex equation solving-areas where computational tools like code interpreters (CI) demonstrate distinct advantages. To bridge this gap, we propose ReTool, which enhances long-form reasoning with tool-integrated learning, including two key features: (1) dynamic interleaving of real-time code execution within natural language reasoning processes, and (2) an automated RL paradigm that allows policy rollouts with multi-turn real-time code execution and teaches the model in learning when and how to invoke tools based on outcome feedback. ReTool employs a systematic training framework, beginning with synthetic cold-start data generation to produce code-augmented long-form reasoning traces for fine-tuning base models. Subsequent RL training leverages task outcomes as rewards to iteratively refine the model's tool use strategy, enabling autonomous discovery of optimal tool invocation patterns without human priors. Experiments on the challenging MATH Olympiad benchmark AIME demonstrate ReTool's superiority: Our 32B model achieves 67% accuracy with 400 training steps, outperforming text-based RL baseline (40% accuracy, 1080 steps) in efficiency and performance. Remarkably, ReTool-32B attains 72.5% accuracy in extended settings, surpassing OpenAI's o1-preview by 27.9%. Further analysis reveals emergent behaviors such as code self-correction, signaling an ''aha moment'' in which the model autonomously masters adaptive tool use. These findings highlight the promise of outcome-driven tool integration for advancing complex mathematical reasoning and offer new insights into hybrid neuro-symbolic systems. 

---
# HypoBench: Towards Systematic and Principled Benchmarking for Hypothesis Generation 

**Authors**: Haokun Liu, Sicong Huang, Jingyu Hu, Yangqiaoyu Zhou, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11524)  

**Abstract**: There is growing interest in hypothesis generation with large language models (LLMs). However, fundamental questions remain: what makes a good hypothesis, and how can we systematically evaluate methods for hypothesis generation? To address this, we introduce HypoBench, a novel benchmark designed to evaluate LLMs and hypothesis generation methods across multiple aspects, including practical utility, generalizability, and hypothesis discovery rate. HypoBench includes 7 real-world tasks and 5 synthetic tasks with 194 distinct datasets. We evaluate four state-of-the-art LLMs combined with six existing hypothesis-generation methods. Overall, our results suggest that existing methods are capable of discovering valid and novel patterns in the data. However, the results from synthetic datasets indicate that there is still significant room for improvement, as current hypothesis generation methods do not fully uncover all relevant or meaningful patterns. Specifically, in synthetic settings, as task difficulty increases, performance significantly drops, with best models and methods only recovering 38.8% of the ground-truth hypotheses. These findings highlight challenges in hypothesis generation and demonstrate that HypoBench serves as a valuable resource for improving AI systems designed to assist scientific discovery. 

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
# Reasoning-Based AI for Startup Evaluation (R.A.I.S.E.): A Memory-Augmented, Multi-Step Decision Framework 

**Authors**: Jack Preuveneers, Joseph Ternasky, Fuat Alican, Yigit Ihlamur  

**Link**: [PDF](https://arxiv.org/pdf/2504.12090)  

**Abstract**: We present a novel framework that bridges the gap between the interpretability of decision trees and the advanced reasoning capabilities of large language models (LLMs) to predict startup success. Our approach leverages chain-of-thought prompting to generate detailed reasoning logs, which are subsequently distilled into structured, human-understandable logical rules. The pipeline integrates multiple enhancements - efficient data ingestion, a two-step refinement process, ensemble candidate sampling, simulated reinforcement learning scoring, and persistent memory - to ensure both stable decision-making and transparent output. Experimental evaluations on curated startup datasets demonstrate that our combined pipeline improves precision by 54% from 0.225 to 0.346 and accuracy by 50% from 0.46 to 0.70 compared to a standalone OpenAI o3 model. Notably, our model achieves over 2x the precision of a random classifier (16%). By combining state-of-the-art AI reasoning with explicit rule-based explanations, our method not only augments traditional decision-making processes but also facilitates expert intervention and continuous policy refinement. This work lays the foundation for the implementation of interpretable LLM-powered decision frameworks in high-stakes investment environments and other domains that require transparent and data-driven insights. 

---
# Rethinking the Generation of High-Quality CoT Data from the Perspective of LLM-Adaptive Question Difficulty Grading 

**Authors**: Qianjin Yu, Keyu Wu, Zihan Chen, Chushu Zhang, Manlin Mei, Lingjun Huang, Fang Tan, Yongsheng Du, Kunlin Liu, Yurui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11919)  

**Abstract**: Recently, DeepSeek-R1 (671B) (DeepSeek-AIet al., 2025) has demonstrated its excellent reasoning ability in complex tasks and has publiclyshared its methodology. This provides potentially high-quality chain-of-thought (CoT) data for stimulating the reasoning abilities of small-sized large language models (LLMs). To generate high-quality CoT data for different LLMs, we seek an efficient method for generating high-quality CoT data with LLM-Adaptive questiondifficulty levels. First, we grade the difficulty of the questions according to the reasoning ability of the LLMs themselves and construct a LLM-Adaptive question database. Second, we sample the problem database based on a distribution of difficulty levels of the questions and then use DeepSeek-R1 (671B) (DeepSeek-AI et al., 2025) to generate the corresponding high-quality CoT data with correct answers. Thanks to the construction of CoT data with LLM-Adaptive difficulty levels, we have significantly reduced the cost of data generation and enhanced the efficiency of model supervised fine-tuning (SFT). Finally, we have validated the effectiveness and generalizability of the proposed method in the fields of complex mathematical competitions and code generation tasks. Notably, with only 2k high-quality mathematical CoT data, our ZMath-32B surpasses DeepSeek-Distill-32B in math reasoning task. Similarly, with only 2k high-quality code CoT data, our ZCode-32B surpasses DeepSeek-Distill-32B in code reasoning tasks. 

---
# Purposefully Induced Psychosis (PIP): Embracing Hallucination as Imagination in Large Language Models 

**Authors**: Kris Pilcher, Esen K. Tütüncü  

**Link**: [PDF](https://arxiv.org/pdf/2504.12012)  

**Abstract**: Hallucinations in Large Language Models (LLMs) are widely regarded as errors - outputs that deviate from factual accuracy. However, in creative or exploratory contexts, these "mistakes" may represent unexpected avenues for innovation. We introduce Purposefully Induced Psychosis (PIP), a novel approach that amplifies LLM hallucinations for imaginative tasks such as speculative fiction, interactive storytelling, and mixed-reality simulations. Drawing on Herman Melville's Moby-Dick, where Pip's "madness" reveals profound insight, we reframe hallucinations as a source of computational imagination rather than a flaw. Our method fine-tunes LLMs to encourage speculative, metaphorical, and surreal outputs - hallucinations that are useful when factual accuracy is not the chief objective. Inspired by the consensual illusions of theater and stage magic, PIP situates these creative missteps in contexts where users willingly suspend disbelief, thereby transforming "errors" into catalysts for new ways of thinking. We discuss potential applications, design principles for ensuring user consent, preliminary observations, and implications for broader AI ethics and human-AI collaboration. 

---
# Large Language Models for Drug Overdose Prediction from Longitudinal Medical Records 

**Authors**: Md Sultan Al Nahian, Chris Delcher, Daniel Harris, Peter Akpunonu, Ramakanth Kavuluru  

**Link**: [PDF](https://arxiv.org/pdf/2504.11792)  

**Abstract**: The ability to predict drug overdose risk from a patient's medical records is crucial for timely intervention and prevention. Traditional machine learning models have shown promise in analyzing longitudinal medical records for this task. However, recent advancements in large language models (LLMs) offer an opportunity to enhance prediction performance by leveraging their ability to process long textual data and their inherent prior knowledge across diverse tasks. In this study, we assess the effectiveness of Open AI's GPT-4o LLM in predicting drug overdose events using patients' longitudinal insurance claims records. We evaluate its performance in both fine-tuned and zero-shot settings, comparing them to strong traditional machine learning methods as baselines. Our results show that LLMs not only outperform traditional models in certain settings but can also predict overdose risk in a zero-shot setting without task-specific training. These findings highlight the potential of LLMs in clinical decision support, particularly for drug overdose risk prediction. 

---
# A Library of LLM Intrinsics for Retrieval-Augmented Generation 

**Authors**: Marina Danilevsky, Kristjan Greenewald, Chulaka Gunasekara, Maeda Hanafi, Lihong He, Yannis Katsis, Krishnateja Killamsetty, Yatin Nandwani, Lucian Popa, Dinesh Raghu, Frederick Reiss, Vraj Shah, Khoi-Nguyen Tran, Huaiyu Zhu, Luis Lastras  

**Link**: [PDF](https://arxiv.org/pdf/2504.11704)  

**Abstract**: In the developer community for large language models (LLMs), there is not yet a clean pattern analogous to a software library, to support very large scale collaboration. Even for the commonplace use case of Retrieval-Augmented Generation (RAG), it is not currently possible to write a RAG application against a well-defined set of APIs that are agreed upon by different LLM providers. Inspired by the idea of compiler intrinsics, we propose some elements of such a concept through introducing a library of LLM Intrinsics for RAG. An LLM intrinsic is defined as a capability that can be invoked through a well-defined API that is reasonably stable and independent of how the LLM intrinsic itself is implemented. The intrinsics in our library are released as LoRA adapters on HuggingFace, and through a software interface with clear structured input/output characteristics on top of vLLM as an inference platform, accompanied in both places with documentation and code. This article describes the intended usage, training details, and evaluations for each intrinsic, as well as compositions of multiple intrinsics. 

---
# HLS-Eval: A Benchmark and Framework for Evaluating LLMs on High-Level Synthesis Design Tasks 

**Authors**: Stefan Abi-Karam, Cong Hao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12268)  

**Abstract**: The rapid scaling of large language model (LLM) training and inference has driven their adoption in semiconductor design across academia and industry. While most prior work evaluates LLMs on hardware description language (HDL) tasks, particularly Verilog, designers are increasingly using high-level synthesis (HLS) to build domain-specific accelerators and complex hardware systems. However, benchmarks and tooling to comprehensively evaluate LLMs for HLS design tasks remain scarce.
To address this, we introduce HLS-Eval, the first complete benchmark and evaluation framework for LLM-driven HLS design. HLS-Eval targets two core tasks: (1) generating HLS code from natural language descriptions, and (2) performing HLS-specific code edits to optimize performance and hardware efficiency. The benchmark includes 94 unique designs drawn from standard HLS benchmarks and novel sources. Each case is prepared via a semi-automated flow that produces a natural language description and a paired testbench for C-simulation and synthesis validation, ensuring each task is "LLM-ready."
Beyond the benchmark, HLS-Eval offers a modular Python framework for automated, parallel evaluation of both local and hosted LLMs. It includes a parallel evaluation engine, direct HLS tool integration, and abstractions for to support different LLM interaction paradigms, enabling rapid prototyping of new benchmarks, tasks, and LLM methods.
We demonstrate HLS-Eval through baseline evaluations of open-source LLMs on Vitis HLS, measuring outputs across four key metrics - parseability, compilability, runnability, and synthesizability - reflecting the iterative HLS design cycle. We also report pass@k metrics, establishing clear baselines and reusable infrastructure for the broader LLM-for-hardware community.
All benchmarks, framework code, and results are open-sourced at this https URL. 

---
# FLIP Reasoning Challenge 

**Authors**: Andreas Plesner, Turlan Kuzhagaliyev, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2504.12256)  

**Abstract**: Over the past years, advances in artificial intelligence (AI) have demonstrated how AI can solve many perception and generation tasks, such as image classification and text writing, yet reasoning remains a challenge. This paper introduces the FLIP dataset, a benchmark for evaluating AI reasoning capabilities based on human verification tasks on the Idena blockchain. FLIP challenges present users with two orderings of 4 images, requiring them to identify the logically coherent one. By emphasizing sequential reasoning, visual storytelling, and common sense, FLIP provides a unique testbed for multimodal AI systems. Our experiments evaluate state-of-the-art models, leveraging both vision-language models (VLMs) and large language models (LLMs). Results reveal that even the best open-sourced and closed-sourced models achieve maximum accuracies of 75.5% and 77.9%, respectively, in zero-shot settings, compared to human performance of 95.3%. Captioning models aid reasoning models by providing text descriptions of images, yielding better results than when using the raw images directly, 69.6% vs. 75.2% for Gemini 1.5 Pro. Combining the predictions from 15 models in an ensemble increases the accuracy to 85.2%. These findings highlight the limitations of existing reasoning models and the need for robust multimodal benchmarks like FLIP. The full codebase and dataset will be available at this https URL. 

---
# Steering Prosocial AI Agents: Computational Basis of LLM's Decision Making in Social Simulation 

**Authors**: Ji Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.11671)  

**Abstract**: Large language models (LLMs) increasingly serve as human-like decision-making agents in social science and applied settings. These LLM-agents are typically assigned human-like characters and placed in real-life contexts. However, how these characters and contexts shape an LLM's behavior remains underexplored. This study proposes and tests methods for probing, quantifying, and modifying an LLM's internal representations in a Dictator Game -- a classic behavioral experiment on fairness and prosocial behavior. We extract ``vectors of variable variations'' (e.g., ``male'' to ``female'') from the LLM's internal state. Manipulating these vectors during the model's inference can substantially alter how those variables relate to the model's decision-making. This approach offers a principled way to study and regulate how social concepts can be encoded and engineered within transformer-based models, with implications for alignment, debiasing, and designing AI agents for social simulations in both academic and commercial applications. 

---
# Enhancing Autonomous Driving Systems with On-Board Deployed Large Language Models 

**Authors**: Nicolas Baumann, Cheng Hu, Paviththiren Sivasothilingam, Haotong Qin, Lei Xie, Michele Magno, Luca Benini  

**Link**: [PDF](https://arxiv.org/pdf/2504.11514)  

**Abstract**: Neural Networks (NNs) trained through supervised learning struggle with managing edge-case scenarios common in real-world driving due to the intractability of exhaustive datasets covering all edge-cases, making knowledge-driven approaches, akin to how humans intuitively detect unexpected driving behavior, a suitable complement to data-driven methods. This work proposes a hybrid architecture combining low-level Model Predictive Controller (MPC) with locally deployed Large Language Models (LLMs) to enhance decision-making and Human Machine Interaction (HMI). The DecisionxLLM module evaluates robotic state information against natural language instructions to ensure adherence to desired driving behavior. The MPCxLLM module then adjusts MPC parameters based on LLM-generated insights, achieving control adaptability while preserving the safety and constraint guarantees of traditional MPC systems. Further, to enable efficient on-board deployment and to eliminate dependency on cloud connectivity, we shift processing to the on-board computing platform: We propose an approach that exploits Retrieval Augmented Generation (RAG), Low Rank Adaptation (LoRA) fine-tuning, and quantization. Experimental results demonstrate that these enhancements yield significant improvements in reasoning accuracy by up to 10.45%, control adaptability by as much as 52.2%, and up to 10.5x increase in computational efficiency (tokens/s), validating the proposed framework's practicality for real-time deployment even on down-scaled robotic platforms. This work bridges high-level decision-making with low-level control adaptability, offering a synergistic framework for knowledge-driven and adaptive Autonomous Driving Systems (ADS). 

---
# The Hitchhiker's Guide to Program Analysis, Part II: Deep Thoughts by LLMs 

**Authors**: Haonan Li, Hang Zhang, Kexin Pei, Zhiyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.11711)  

**Abstract**: Static analysis is a cornerstone for software vulnerability detection, yet it often struggles with the classic precision-scalability trade-off. In practice, such tools often produce high false positive rates, particularly in large codebases like the Linux kernel. This imprecision can arise from simplified vulnerability modeling and over-approximation of path and data constraints. While large language models (LLMs) show promise in code understanding, their naive application to program analysis yields unreliable results due to inherent reasoning limitations. We introduce BugLens, a post-refinement framework that significantly improves static analysis precision. BugLens guides an LLM to follow traditional analysis steps by assessing buggy code patterns for security impact and validating the constraints associated with static warnings. Evaluated on real-world Linux kernel bugs, BugLens raises precision from 0.10 (raw) and 0.50 (semi-automated refinement) to 0.72, substantially reducing false positives and revealing four previously unreported vulnerabilities. Our results suggest that a structured LLM-based workflow can meaningfully enhance the effectiveness of static analysis tools. 

---
# Can GPT tell us why these images are synthesized? Empowering Multimodal Large Language Models for Forensics 

**Authors**: Yiran He, Yun Cao, Bowen Yang, Zeyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11686)  

**Abstract**: The rapid development of generative AI facilitates content creation and makes image manipulation easier and more difficult to detect. While multimodal Large Language Models (LLMs) have encoded rich world knowledge, they are not inherently tailored for combating AI-generated Content (AIGC) and struggle to comprehend local forgery details. In this work, we investigate the application of multimodal LLMs in forgery detection. We propose a framework capable of evaluating image authenticity, localizing tampered regions, providing evidence, and tracing generation methods based on semantic tampering clues. Our method demonstrates that the potential of LLMs in forgery analysis can be effectively unlocked through meticulous prompt engineering and the application of few-shot learning techniques. We conduct qualitative and quantitative experiments and show that GPT4V can achieve an accuracy of 92.1% in Autosplice and 86.3% in LaMa, which is competitive with state-of-the-art AIGC detection methods. We further discuss the limitations of multimodal LLMs in such tasks and propose potential improvements. 

---
# SDIGLM: Leveraging Large Language Models and Multi-Modal Chain of Thought for Structural Damage Identification 

**Authors**: Yunkai Zhang, Shiyin Wei, Yong Huang, Yawu Su, Shanshan Lu, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.11477)  

**Abstract**: Existing computer vision(CV)-based structural damage identification models demonstrate notable accuracy in categorizing and localizing damage. However, these models present several critical limitations that hinder their practical application in civil engineering(CE). Primarily, their ability to recognize damage types remains constrained, preventing comprehensive analysis of the highly varied and complex conditions encountered in real-world CE structures. Second, these models lack linguistic capabilities, rendering them unable to articulate structural damage characteristics through natural language descriptions. With the continuous advancement of artificial intelligence(AI), large multi-modal models(LMMs) have emerged as a transformative solution, enabling the unified encoding and alignment of textual and visual data. These models can autonomously generate detailed descriptive narratives of structural damage while demonstrating robust generalization across diverse scenarios and tasks. This study introduces SDIGLM, an innovative LMM for structural damage identification, developed based on the open-source VisualGLM-6B architecture. To address the challenge of adapting LMMs to the intricate and varied operating conditions in CE, this work integrates a U-Net-based semantic segmentation module to generate defect segmentation maps as visual Chain of Thought(CoT). Additionally, a multi-round dialogue fine-tuning dataset is constructed to enhance logical reasoning, complemented by a language CoT formed through prompt engineering. By leveraging this multi-modal CoT, SDIGLM surpasses general-purpose LMMs in structural damage identification, achieving an accuracy of 95.24% across various infrastructure types. Moreover, the model effectively describes damage characteristics such as hole size, crack direction, and corrosion severity. 

---
