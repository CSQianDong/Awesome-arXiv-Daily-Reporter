# Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models 

**Authors**: Anirudh Bharadwaj, Chaitanya Malaviya, Nitish Joshi, Mark Yatskar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05339)  

**Abstract**: Language models serve as proxies for human preference judgements in alignment and evaluation, yet they exhibit systematic miscalibration, prioritizing superficial patterns over substantive qualities. This bias manifests as overreliance on features like length, structure, and style, leading to issues like reward hacking and unreliable evaluations. Evidence suggests these biases originate in artifacts in human training data. In this work, we systematically investigate the relationship between training data biases and preference model miscalibration across five idiosyncratic features of language model generations: length, structure, jargon, sycophancy and vagueness. Using controlled counterfactual pairs, we first quantify the extent to which preference models favor responses with magnified biases (skew), finding this preference occurs in >60% of instances, and model preferences show high miscalibration (~40%) compared to human preferences. Notably, bias features only show mild negative correlations to human preference labels (mean r_human = -0.12) but show moderately strong positive correlations with labels from a strong reward model (mean r_model = +0.36), suggesting that models may overrely on spurious cues. To mitigate these issues, we propose a simple post-training method based on counterfactual data augmentation (CDA) using synthesized contrastive examples. Finetuning models with CDA reduces average miscalibration from 39.4% to 32.5% and average absolute skew difference from 20.5% to 10.0%, while maintaining overall RewardBench performance, showing that targeted debiasing is effective for building reliable preference models. 

---
# Search Arena: Analyzing Search-Augmented LLMs 

**Authors**: Mihran Miroyan, Tsung-Han Wu, Logan King, Tianle Li, Jiayi Pan, Xinyan Hu, Wei-Lin Chiang, Anastasios N. Angelopoulos, Trevor Darrell, Narges Norouzi, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2506.05334)  

**Abstract**: Search-augmented language models combine web search with Large Language Models (LLMs) to improve response groundedness and freshness. However, analyzing these systems remains challenging: existing datasets are limited in scale and narrow in scope, often constrained to static, single-turn, fact-checking questions. In this work, we introduce Search Arena, a crowd-sourced, large-scale, human-preference dataset of over 24,000 paired multi-turn user interactions with search-augmented LLMs. The dataset spans diverse intents and languages, and contains full system traces with around 12,000 human preference votes. Our analysis reveals that user preferences are influenced by the number of citations, even when the cited content does not directly support the attributed claims, uncovering a gap between perceived and actual credibility. Furthermore, user preferences vary across cited sources, revealing that community-driven platforms are generally preferred and static encyclopedic sources are not always appropriate and reliable. To assess performance across different settings, we conduct cross-arena analyses by testing search-augmented LLMs in a general-purpose chat environment and conventional LLMs in search-intensive settings. We find that web search does not degrade and may even improve performance in non-search settings; however, the quality in search settings is significantly affected if solely relying on the model's parametric knowledge. We open-sourced the dataset to support future research in this direction. Our dataset and code are available at: this https URL. 

---
# Constrained Entropic Unlearning: A Primal-Dual Framework for Large Language Models 

**Authors**: Taha Entesari, Arman Hatami, Rinat Khaziev, Anil Ramakrishna, Mahyar Fazlyab  

**Link**: [PDF](https://arxiv.org/pdf/2506.05314)  

**Abstract**: Large Language Models (LLMs) deployed in real-world settings increasingly face the need to unlearn sensitive, outdated, or proprietary information. Existing unlearning methods typically formulate forgetting and retention as a regularized trade-off, combining both objectives into a single scalarized loss. This often leads to unstable optimization and degraded performance on retained data, especially under aggressive forgetting. We propose a new formulation of LLM unlearning as a constrained optimization problem: forgetting is enforced via a novel logit-margin flattening loss that explicitly drives the output distribution toward uniformity on a designated forget set, while retention is preserved through a hard constraint on a separate retain set. Compared to entropy-based objectives, our loss is softmax-free, numerically stable, and maintains non-vanishing gradients, enabling more efficient and robust optimization. We solve the constrained problem using a scalable primal-dual algorithm that exposes the trade-off between forgetting and retention through the dynamics of the dual variable. Evaluations on the TOFU and MUSE benchmarks across diverse LLM architectures demonstrate that our approach consistently matches or exceeds state-of-the-art baselines, effectively removing targeted information while preserving downstream utility. 

---
# ProRefine: Inference-time Prompt Refinement with Textual Feedback 

**Authors**: Deepak Pandita, Tharindu Cyril Weerasooriya, Ankit Parag Shah, Christopher M. Homan, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.05305)  

**Abstract**: Agentic workflows, where multiple AI agents collaborate to accomplish complex tasks like reasoning or planning, are becoming increasingly prevalent. However, these workflows often suffer from error propagation and sub-optimal performance, largely due to poorly designed prompts that fail to effectively guide individual agents. This is a critical problem because it limits the reliability and scalability of these powerful systems. We introduce ProRefine, an innovative inference-time prompt optimization method that leverages textual feedback from large language models (LLMs) to address this challenge. ProRefine dynamically refines prompts for multi-step reasoning tasks without additional training or ground truth labels. Evaluated on five benchmark mathematical reasoning datasets, ProRefine significantly surpasses zero-shot Chain-of-Thought baselines by 3 to 37 percentage points. This approach not only boosts accuracy but also allows smaller models to match the performance of larger ones, highlighting its potential for efficient and scalable AI deployment, and democratizing access to high-performing AI. 

---
# Micro-Act: Mitigate Knowledge Conflict in Question Answering via Actionable Self-Reasoning 

**Authors**: Nan Huo, Jinyang Li, Bowen Qin, Ge Qu, Xiaolong Li, Xiaodong Li, Chenhao Ma, Reynold Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05278)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems commonly suffer from Knowledge Conflicts, where retrieved external knowledge contradicts the inherent, parametric knowledge of large language models (LLMs). It adversely affects performance on downstream tasks such as question answering (QA). Existing approaches often attempt to mitigate conflicts by directly comparing two knowledge sources in a side-by-side manner, but this can overwhelm LLMs with extraneous or lengthy contexts, ultimately hindering their ability to identify and mitigate inconsistencies. To address this issue, we propose Micro-Act a framework with a hierarchical action space that automatically perceives context complexity and adaptively decomposes each knowledge source into a sequence of fine-grained comparisons. These comparisons are represented as actionable steps, enabling reasoning beyond the superficial context. Through extensive experiments on five benchmark datasets, Micro-Act consistently achieves significant increase in QA accuracy over state-of-the-art baselines across all 5 datasets and 3 conflict types, especially in temporal and semantic types where all baselines fail significantly. More importantly, Micro-Act exhibits robust performance on non-conflict questions simultaneously, highlighting its practical value in real-world RAG applications. 

---
# CLATTER: Comprehensive Entailment Reasoning for Hallucination Detection 

**Authors**: Ron Eliav, Arie Cattan, Eran Hirsch, Shahaf Bassan, Elias Stengel-Eskin, Mohit Bansal, Ido Dagan  

**Link**: [PDF](https://arxiv.org/pdf/2506.05243)  

**Abstract**: A common approach to hallucination detection casts it as a natural language inference (NLI) task, often using LLMs to classify whether the generated text is entailed by corresponding reference texts. Since entailment classification is a complex reasoning task, one would expect that LLMs could benefit from generating an explicit reasoning process, as in CoT reasoning or the explicit ``thinking'' of recent reasoning models. In this work, we propose that guiding such models to perform a systematic and comprehensive reasoning process -- one that both decomposes the text into smaller facts and also finds evidence in the source for each fact -- allows models to execute much finer-grained and accurate entailment decisions, leading to increased performance. To that end, we define a 3-step reasoning process, consisting of (i) claim decomposition, (ii) sub-claim attribution and entailment classification, and (iii) aggregated classification, showing that such guided reasoning indeed yields improved hallucination detection. Following this reasoning framework, we introduce an analysis scheme, consisting of several metrics that measure the quality of the intermediate reasoning steps, which provided additional empirical evidence for the improved quality of our guided reasoning scheme. 

---
# Towards a Unified System of Representation for Continuity and Discontinuity in Natural Language 

**Authors**: Ratna Kandala, Prakash Mondal  

**Link**: [PDF](https://arxiv.org/pdf/2506.05235)  

**Abstract**: Syntactic discontinuity is a grammatical phenomenon in which a constituent is split into more than one part because of the insertion of an element which is not part of the constituent. This is observed in many languages across the world such as Turkish, Russian, Japanese, Warlpiri, Navajo, Hopi, Dyirbal, Yidiny etc. Different formalisms/frameworks in current linguistic theory approach the problem of discontinuous structures in different ways. Each framework/formalism has widely been viewed as an independent and non-converging system of analysis. In this paper, we propose a unified system of representation for both continuity and discontinuity in structures of natural languages by taking into account three formalisms, in particular, Phrase Structure Grammar (PSG) for its widely used notion of constituency, Dependency Grammar (DG) for its head-dependent relations, and Categorial Grammar (CG) for its focus on functor-argument relations. We attempt to show that discontinuous expressions as well as continuous structures can be analysed through a unified mathematical derivation incorporating the representations of linguistic structure in these three grammar formalisms. 

---
# Improving Low-Resource Morphological Inflection via Self-Supervised Objectives 

**Authors**: Adam Wiemerslage, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2506.05227)  

**Abstract**: Self-supervised objectives have driven major advances in NLP by leveraging large-scale unlabeled data, but such resources are scarce for many of the world's languages. Surprisingly, they have not been explored much for character-level tasks, where smaller amounts of data have the potential to be beneficial. We investigate the effectiveness of self-supervised auxiliary tasks for morphological inflection -- a character-level task highly relevant for language documentation -- in extremely low-resource settings, training encoder-decoder transformers for 19 languages and 13 auxiliary objectives. Autoencoding yields the best performance when unlabeled data is very limited, while character masked language modeling (CMLM) becomes more effective as data availability increases. Though objectives with stronger inductive biases influence model predictions intuitively, they rarely outperform standard CMLM. However, sampling masks based on known morpheme boundaries consistently improves performance, highlighting a promising direction for low-resource morphological modeling. 

---
# The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text 

**Authors**: Nikhil Kandpal, Brian Lester, Colin Raffel, Sebastian Majstorovic, Stella Biderman, Baber Abbasi, Luca Soldaini, Enrico Shippole, A. Feder Cooper, Aviya Skowron, John Kirchenbauer, Shayne Longpre, Lintang Sutawika, Alon Albalak, Zhenlin Xu, Guilherme Penedo, Loubna Ben Allal, Elie Bakouch, John David Pressman, Honglu Fan, Dashiell Stander, Guangyu Song, Aaron Gokaslan, Tom Goldstein, Brian R. Bartoldson, Bhavya Kailkhura, Tyler Murray  

**Link**: [PDF](https://arxiv.org/pdf/2506.05209)  

**Abstract**: Large language models (LLMs) are typically trained on enormous quantities of unlicensed text, a practice that has led to scrutiny due to possible intellectual property infringement and ethical concerns. Training LLMs on openly licensed text presents a first step towards addressing these issues, but prior data collection efforts have yielded datasets too small or low-quality to produce performant LLMs. To address this gap, we collect, curate, and release the Common Pile v0.1, an eight terabyte collection of openly licensed text designed for LLM pretraining. The Common Pile comprises content from 30 sources that span diverse domains including research papers, code, books, encyclopedias, educational materials, audio transcripts, and more. Crucially, we validate our efforts by training two 7 billion parameter LLMs on text from the Common Pile: Comma v0.1-1T and Comma v0.1-2T, trained on 1 and 2 trillion tokens respectively. Both models attain competitive performance to LLMs trained on unlicensed text with similar computational budgets, such as Llama 1 and 2 7B. In addition to releasing the Common Pile v0.1 itself, we also release the code used in its creation as well as the training mixture and checkpoints for the Comma v0.1 models. 

---
# RELIC: Evaluating Compositional Instruction Following via Language Recognition 

**Authors**: Jackson Petty, Michael Y. Hu, Wentao Wang, Shauli Ravfogel, William Merrill, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05205)  

**Abstract**: Large language models (LLMs) are increasingly expected to perform tasks based only on a specification of the task provided in context, without examples of inputs and outputs; this ability is referred to as instruction following. We introduce the Recognition of Languages In-Context (RELIC) framework to evaluate instruction following using language recognition: the task of determining if a string is generated by formal grammar. Unlike many standard evaluations of LLMs' ability to use their context, this task requires composing together a large number of instructions (grammar productions) retrieved from the context. Because the languages are synthetic, the task can be increased in complexity as LLMs' skills improve, and new instances can be automatically generated, mitigating data contamination. We evaluate state-of-the-art LLMs on RELIC and find that their accuracy can be reliably predicted from the complexity of the grammar and the individual example strings, and that even the most advanced LLMs currently available show near-chance performance on more complex grammars and samples, in line with theoretical expectations. We also use RELIC to diagnose how LLMs attempt to solve increasingly difficult reasoning tasks, finding that as the complexity of the language recognition task increases, models switch to relying on shallow heuristics instead of following complex instructions. 

---
# Counterfactual reasoning: an analysis of in-context emergence 

**Authors**: Moritz Miller, Bernhard Schölkopf, Siyuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05188)  

**Abstract**: Large-scale neural language models (LMs) exhibit remarkable performance in in-context learning: the ability to learn and reason the input context on the fly without parameter update. This work studies in-context counterfactual reasoning in language models, that is, to predict the consequences of changes under hypothetical scenarios. We focus on studying a well-defined synthetic setup: a linear regression task that requires noise abduction, where accurate prediction is based on inferring and copying the contextual noise from factual observations. We show that language models are capable of counterfactual reasoning in this controlled setup and provide insights that counterfactual reasoning for a broad class of functions can be reduced to a transformation on in-context observations; we find self-attention, model depth, and data diversity in pre-training drive performance in Transformers. More interestingly, our findings extend beyond regression tasks and show that Transformers can perform noise abduction on sequential data, providing preliminary evidence on the potential for counterfactual story generation. Our code is available under this https URL . 

---
# Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models 

**Authors**: Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.05176)  

**Abstract**: In this work, we introduce the Qwen3 Embedding series, a significant advancement over its predecessor, the GTE-Qwen series, in text embedding and reranking capabilities, built upon the Qwen3 foundation models. Leveraging the Qwen3 LLMs' robust capabilities in multilingual text understanding and generation, our innovative multi-stage training pipeline combines large-scale unsupervised pre-training with supervised fine-tuning on high-quality datasets. Effective model merging strategies further ensure the robustness and adaptability of the Qwen3 Embedding series. During the training process, the Qwen3 LLMs serve not only as backbone models but also play a crucial role in synthesizing high-quality, rich, and diverse training data across multiple domains and languages, thus enhancing the training pipeline. The Qwen3 Embedding series offers a spectrum of model sizes (0.6B, 4B, 8B) for both embedding and reranking tasks, addressing diverse deployment scenarios where users can optimize for either efficiency or effectiveness. Empirical evaluations demonstrate that the Qwen3 Embedding series achieves state-of-the-art results across diverse benchmarks. Notably, it excels on the multilingual evaluation benchmark MTEB for text embedding, as well as in various retrieval tasks, including code retrieval, cross-lingual retrieval and multilingual retrieval. To facilitate reproducibility and promote community-driven research and development, the Qwen3 Embedding models are publicly available under the Apache 2.0 license. 

---
# ECoRAG: Evidentiality-guided Compression for Long Context RAG 

**Authors**: Yeonseok Jeong, Jinsu Kim, Dohyeon Lee, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05167)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance in Open-Domain Question Answering (ODQA) by leveraging external documents through Retrieval-Augmented Generation (RAG). To reduce RAG overhead, from longer context, context compression is necessary. However, prior compression methods do not focus on filtering out non-evidential information, which limit the performance in LLM-based RAG. We thus propose Evidentiality-guided RAG, or \textbf{ECoRAG} framework. ECoRAG improves LLM performance by compressing retrieved documents based on evidentiality, ensuring whether answer generation is supported by the correct evidence. As an additional step, ECoRAG reflects whether the compressed content provides sufficient evidence, and if not, retrieves more until sufficient. Experiments show that ECoRAG improves LLM performance on ODQA tasks, outperforming existing compression methods. Furthermore, ECoRAG is highly cost-efficient, as it not only reduces latency but also minimizes token usage by retaining only the necessary information to generate the correct answer. Code is available at this https URL. 

---
# Dissecting Bias in LLMs: A Mechanistic Interpretability Perspective 

**Authors**: Bhavik Chandna, Zubair Bashir, Procheta Sen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05166)  

**Abstract**: Large Language Models (LLMs) are known to exhibit social, demographic, and gender biases, often as a consequence of the data on which they are trained. In this work, we adopt a mechanistic interpretability approach to analyze how such biases are structurally represented within models such as GPT-2 and Llama2. Focusing on demographic and gender biases, we explore different metrics to identify the internal edges responsible for biased behavior. We then assess the stability, localization, and generalizability of these components across dataset and linguistic variations. Through systematic ablations, we demonstrate that bias-related computations are highly localized, often concentrated in a small subset of layers. Moreover, the identified components change across fine-tuning settings, including those unrelated to bias. Finally, we show that removing these components not only reduces biased outputs but also affects other NLP tasks, such as named entity recognition and linguistic acceptability judgment because of the sharing of important components with these tasks. 

---
# Knowledgeable-r1: Policy Optimization for Knowledge Exploration in Retrieval-Augmented Generation 

**Authors**: Chenyu Lin, Yilin Wen, Du Su, Fei Sun, Muhan Chen, Chenfu Bao, Zhonghou Lv  

**Link**: [PDF](https://arxiv.org/pdf/2506.05154)  

**Abstract**: Retrieval-augmented generation (RAG) is a mainstream method for improving performance on knowledge-intensive tasks. However,current RAG systems often place too much emphasis on retrieved contexts. This can lead to reliance on inaccurate sources and overlook the model's inherent knowledge, especially when dealing with misleading or excessive information. To resolve this imbalance, we propose Knowledgeable-r1 that using joint sampling and define multi policy distributions in knowledge capability exploration to stimulate large language models'self-integrated utilization of parametric and contextual knowledge. Experiments show that Knowledgeable-r1 significantly enhances robustness and reasoning accuracy in both parameters and contextual conflict tasks and general RAG tasks, especially outperforming baselines by 17.07% in counterfactual scenarios and demonstrating consistent gains across RAG tasks. Our code are available at this https URL knowledgeable-r1. 

---
# Do Large Language Models Judge Error Severity Like Humans? 

**Authors**: Diege Sun, Guanyi Chen, Fan Zhao, Xiaorong Cheng, Tingting He  

**Link**: [PDF](https://arxiv.org/pdf/2506.05142)  

**Abstract**: Large Language Models (LLMs) are increasingly used as automated evaluators in natural language generation, yet it remains unclear whether they can accurately replicate human judgments of error severity. In this study, we systematically compare human and LLM assessments of image descriptions containing controlled semantic errors. We extend the experimental framework of van Miltenburg et al. (2020) to both unimodal (text-only) and multimodal (text + image) settings, evaluating four error types: age, gender, clothing type, and clothing colour. Our findings reveal that humans assign varying levels of severity to different error types, with visual context significantly amplifying perceived severity for colour and type errors. Notably, most LLMs assign low scores to gender errors but disproportionately high scores to colour errors, unlike humans, who judge both as highly severe but for different reasons. This suggests that these models may have internalised social norms influencing gender judgments but lack the perceptual grounding to emulate human sensitivity to colour, which is shaped by distinct neural mechanisms. Only one of the evaluated LLMs, Doubao, replicates the human-like ranking of error severity, but it fails to distinguish between error types as clearly as humans. Surprisingly, DeepSeek-V3, a unimodal LLM, achieves the highest alignment with human judgments across both unimodal and multimodal conditions, outperforming even state-of-the-art multimodal models. 

---
# AudioLens: A Closer Look at Auditory Attribute Perception of Large Audio-Language Models 

**Authors**: Chih-Kai Yang, Neo Ho, Yi-Jyun Lee, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.05140)  

**Abstract**: Understanding the internal mechanisms of large audio-language models (LALMs) is crucial for interpreting their behavior and improving performance. This work presents the first in-depth analysis of how LALMs internally perceive and recognize auditory attributes. By applying vocabulary projection on three state-of-the-art LALMs, we track how attribute information evolves across layers and token positions. We find that attribute information generally decreases with layer depth when recognition fails, and that resolving attributes at earlier layers correlates with better accuracy. Moreover, LALMs heavily rely on querying auditory inputs for predicting attributes instead of aggregating necessary information in hidden states at attribute-mentioning positions. Based on our findings, we demonstrate a method to enhance LALMs. Our results offer insights into auditory attribute processing, paving the way for future improvements. 

---
# Information Locality as an Inductive Bias for Neural Language Models 

**Authors**: Taiga Someya, Anej Svete, Brian DuSell, Timothy J. O'Donnell, Mario Giulianelli, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2506.05136)  

**Abstract**: Inductive biases are inherent in every machine learning system, shaping how models generalize from finite data. In the case of neural language models (LMs), debates persist as to whether these biases align with or diverge from human processing constraints. To address this issue, we propose a quantitative framework that allows for controlled investigations into the nature of these biases. Within our framework, we introduce $m$-local entropy$\unicode{x2013}$an information-theoretic measure derived from average lossy-context surprisal$\unicode{x2013}$that captures the local uncertainty of a language by quantifying how effectively the $m-1$ preceding symbols disambiguate the next symbol. In experiments on both perturbed natural language corpora and languages defined by probabilistic finite-state automata (PFSAs), we show that languages with higher $m$-local entropy are more difficult for Transformer and LSTM LMs to learn. These results suggest that neural LMs, much like humans, are highly sensitive to the local statistical structure of a language. 

---
# DiCoRe: Enhancing Zero-shot Event Detection via Divergent-Convergent LLM Reasoning 

**Authors**: Tanmay Parekh, Kartik Mehta, Ninareh Mehrabi, Kai-Wei Chang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05128)  

**Abstract**: Zero-shot Event Detection (ED), the task of identifying event mentions in natural language text without any training data, is critical for document understanding in specialized domains. Understanding the complex event ontology, extracting domain-specific triggers from the passage, and structuring them appropriately overloads and limits the utility of Large Language Models (LLMs) for zero-shot ED. To this end, we propose DiCoRe, a divergent-convergent reasoning framework that decouples the task of ED using Dreamer and Grounder. Dreamer encourages divergent reasoning through open-ended event discovery, which helps to boost event coverage. Conversely, Grounder introduces convergent reasoning to align the free-form predictions with the task-specific instructions using finite-state machine guided constrained decoding. Additionally, an LLM-Judge verifies the final outputs to ensure high precision. Through extensive experiments on six datasets across five domains and nine LLMs, we demonstrate how DiCoRe consistently outperforms prior zero-shot, transfer-learning, and reasoning baselines, achieving 4-7% average F1 gains over the best baseline -- establishing DiCoRe as a strong zero-shot ED framework. 

---
# The NTNU System at the S&I Challenge 2025 SLA Open Track 

**Authors**: Hong-Yun Lin, Tien-Hong Lo, Yu-Hsuan Fang, Jhen-Ke Lin, Chung-Chun Wang, Hao-Chien Lu, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05121)  

**Abstract**: A recent line of research on spoken language assessment (SLA) employs neural models such as BERT and wav2vec 2.0 (W2V) to evaluate speaking proficiency across linguistic and acoustic modalities. Although both models effectively capture features relevant to oral competence, each exhibits modality-specific limitations. BERT-based methods rely on ASR transcripts, which often fail to capture prosodic and phonetic cues for SLA. In contrast, W2V-based methods excel at modeling acoustic features but lack semantic interpretability. To overcome these limitations, we propose a system that integrates W2V with Phi-4 multimodal large language model (MLLM) through a score fusion strategy. The proposed system achieves a root mean square error (RMSE) of 0.375 on the official test set of the Speak & Improve Challenge 2025, securing second place in the competition. For comparison, the RMSEs of the top-ranked, third-ranked, and official baseline systems are 0.364, 0.384, and 0.444, respectively. 

---
# CL-ISR: A Contrastive Learning and Implicit Stance Reasoning Framework for Misleading Text Detection on Social Media 

**Authors**: Tianyi Huang, Zikun Cui, Cuiqianhe Du, Chia-En Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05107)  

**Abstract**: Misleading text detection on social media platforms is a critical research area, as these texts can lead to public misunderstanding, social panic and even economic losses. This paper proposes a novel framework - CL-ISR (Contrastive Learning and Implicit Stance Reasoning), which combines contrastive learning and implicit stance reasoning, to improve the detection accuracy of misleading texts on social media. First, we use the contrastive learning algorithm to improve the model's learning ability of semantic differences between truthful and misleading texts. Contrastive learning could help the model to better capture the distinguishing features between different categories by constructing positive and negative sample pairs. This approach enables the model to capture distinguishing features more effectively, particularly in linguistically complicated situations. Second, we introduce the implicit stance reasoning module, to explore the potential stance tendencies in the text and their relationships with related topics. This method is effective for identifying content that misleads through stance shifting or emotional manipulation, because it can capture the implicit information behind the text. Finally, we integrate these two algorithms together to form a new framework, CL-ISR, which leverages the discriminative power of contrastive learning and the interpretive depth of stance reasoning to significantly improve detection effect. 

---
# Parking, Perception, and Retail: Street-Level Determinants of Community Vitality in Harbin 

**Authors**: HaoTian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2506.05080)  

**Abstract**: The commercial vitality of community-scale streets in Chinese cities is shaped by complex interactions between vehicular accessibility, environmental quality, and pedestrian perception. This study proposes an interpretable, image-based framework to examine how street-level features -- including parked vehicle density, greenery, cleanliness, and street width -- impact retail performance and user satisfaction in Harbin, China. Leveraging street view imagery and a multimodal large language model (VisualGLM-6B), we construct a Community Commercial Vitality Index (CCVI) from Meituan and Dianping data and analyze its relationship with spatial attributes extracted via GPT-4-based perception modeling. Our findings reveal that while moderate vehicle presence may enhance commercial access, excessive on-street parking -- especially in narrow streets -- erodes walkability and reduces both satisfaction and shop-level pricing. In contrast, streets with higher perceived greenery and cleanliness show significantly greater satisfaction scores but only weak associations with pricing. Street width moderates the effects of vehicle presence, underscoring the importance of spatial configuration. These results demonstrate the value of integrating AI-assisted perception with urban morphological analysis to capture non-linear and context-sensitive drivers of commercial success. This study advances both theoretical and methodological frontiers by highlighting the conditional role of vehicle activity in neighborhood commerce and demonstrating the feasibility of multimodal AI for perceptual urban diagnostics. The implications extend to urban design, parking management, and scalable planning tools for community revitalization. 

---
# Just a Scratch: Enhancing LLM Capabilities for Self-harm Detection through Intent Differentiation and Emoji Interpretation 

**Authors**: Soumitra Ghosh, Gopendra Vikram Singh, Shambhavi, Sabarna Choudhury, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2506.05073)  

**Abstract**: Self-harm detection on social media is critical for early intervention and mental health support, yet remains challenging due to the subtle, context-dependent nature of such expressions. Identifying self-harm intent aids suicide prevention by enabling timely responses, but current large language models (LLMs) struggle to interpret implicit cues in casual language and emojis. This work enhances LLMs' comprehension of self-harm by distinguishing intent through nuanced language-emoji interplay. We present the Centennial Emoji Sensitivity Matrix (CESM-100), a curated set of 100 emojis with contextual self-harm interpretations and the Self-Harm Identification aNd intent Extraction with Supportive emoji sensitivity (SHINES) dataset, offering detailed annotations for self-harm labels, casual mentions (CMs), and serious intents (SIs). Our unified framework: a) enriches inputs using CESM-100; b) fine-tunes LLMs for multi-task learning: self-harm detection (primary) and CM/SI span detection (auxiliary); c) generates explainable rationales for self-harm predictions. We evaluate the framework on three state-of-the-art LLMs-Llama 3, Mental-Alpaca, and MentalLlama, across zero-shot, few-shot, and fine-tuned scenarios. By coupling intent differentiation with contextual cues, our approach commendably enhances LLM performance in both detection and explanation tasks, effectively addressing the inherent ambiguity in self-harm signals. The SHINES dataset, CESM-100 and codebase are publicly available at: this https URL . 

---
# RIVAL: Reinforcement Learning with Iterative and Adversarial Optimization for Machine Translation 

**Authors**: Tianjiao Li, Mengran Yu, Chenyu Shi, Yanjun Zhao, Xiaojing Liu, Qiang Zhang, Qi Zhang, Xuanjing Huang, Jiayin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05070)  

**Abstract**: Large language models (LLMs) possess strong multilingual capabilities, and combining Reinforcement Learning from Human Feedback (RLHF) with translation tasks has shown great potential. However, we observe that this paradigm performs unexpectedly poorly when applied to colloquial subtitle translation tasks. In this work, we investigate this issue and find that the offline reward model (RM) gradually diverges from the online LLM due to distributional shift, ultimately leading to undesirable training outcomes. To address this, we propose RIVAL, an adversarial training framework that formulates the process as a min-max game between the RM and the LLM. RIVAL iteratively updates the both models, with the RM trained to distinguish strong from weak translations (qualitative preference reward), and the LLM trained to enhance its translation for closing this gap. To stabilize training and improve generalizability, we also incorporate quantitative preference reward (e.g., BLEU) into the RM, enabling reference-free quality modeling aligned with human evaluation. Through extensive experiments, we demonstrate that the proposed adversarial training framework significantly improves upon translation baselines. 

---
# Does It Make Sense to Speak of Introspection in Large Language Models? 

**Authors**: Iulia Comşa, Murray Shanahan  

**Link**: [PDF](https://arxiv.org/pdf/2506.05068)  

**Abstract**: Large language models (LLMs) exhibit compelling linguistic behaviour, and sometimes offer self-reports, that is to say statements about their own nature, inner workings, or behaviour. In humans, such reports are often attributed to a faculty of introspection and are typically linked to consciousness. This raises the question of how to interpret self-reports produced by LLMs, given their increasing linguistic fluency and cognitive capabilities. To what extent (if any) can the concept of introspection be meaningfully applied to LLMs? Here, we present and critique two examples of apparent introspective self-report from LLMs. In the first example, an LLM attempts to describe the process behind its own ``creative'' writing, and we argue this is not a valid example of introspection. In the second example, an LLM correctly infers the value of its own temperature parameter, and we argue that this can be legitimately considered a minimal example of introspection, albeit one that is (presumably) not accompanied by conscious experience. 

---
# Debatable Intelligence: Benchmarking LLM Judges via Debate Speech Evaluation 

**Authors**: Noy Sternlicht, Ariel Gera, Roy Bar-Haim, Tom Hope, Noam Slonim  

**Link**: [PDF](https://arxiv.org/pdf/2506.05062)  

**Abstract**: We introduce Debate Speech Evaluation as a novel and challenging benchmark for assessing LLM judges. Evaluating debate speeches requires a deep understanding of the speech at multiple levels, including argument strength and relevance, the coherence and organization of the speech, the appropriateness of its style and tone, and so on. This task involves a unique set of cognitive abilities that have previously received limited attention in systematic LLM benchmarking. To explore such skills, we leverage a dataset of over 600 meticulously annotated debate speeches and present the first in-depth analysis of how state-of-the-art LLMs compare to human judges on this task. Our findings reveal a nuanced picture: while larger models can approximate individual human judgments in some respects, they differ substantially in their overall judgment behavior. We also investigate the ability of frontier LLMs to generate persuasive, opinionated speeches, showing that models may perform at a human level on this task. 

---
# TALL -- A Trainable Architecture for Enhancing LLM Performance in Low-Resource Languages 

**Authors**: Moshe Ofer, Orel Zamler, Amos Azaria  

**Link**: [PDF](https://arxiv.org/pdf/2506.05057)  

**Abstract**: Large Language Models (LLMs) excel in high-resource languages but struggle with low-resource languages due to limited training data. This paper presents TALL (Trainable Architecture for Enhancing LLM Performance in Low-Resource Languages), which integrates an LLM with two bilingual translation models. TALL transforms low-resource inputs into high-resource representations, leveraging the LLM's capabilities while preserving linguistic features through dimension alignment layers and custom transformers. Our experiments on Hebrew demonstrate significant improvements over several baselines, including direct use, naive translation, and fine-tuning approaches. The architecture employs a parameter-efficient strategy, freezing pre-trained components while training only lightweight adapter modules, balancing computational efficiency with performance gains. 

---
# Automatic Robustness Stress Testing of LLMs as Mathematical Problem Solvers 

**Authors**: Yutao Hou, Zeguan Xiao, Fei Yu, Yihan Jiang, Xuetao Wei, Hailiang Huang, Yun Chen, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05038)  

**Abstract**: Large language models (LLMs) have achieved distinguished performance on various reasoning-intensive tasks. However, LLMs might still face the challenges of robustness issues and fail unexpectedly in some simple reasoning tasks. Previous works evaluate the LLM robustness with hand-crafted templates or a limited set of perturbation rules, indicating potential data contamination in pre-training or fine-tuning datasets. In this work, inspired by stress testing in software engineering, we propose a novel framework, Automatic Robustness Checker (AR-Checker), to generate mathematical problem variants that maintain the semantic meanings of the original one but might fail the LLMs. The AR-Checker framework generates mathematical problem variants through multi-round parallel streams of LLM-based rewriting and verification. Our framework can generate benchmark variants dynamically for each LLM, thus minimizing the risk of data contamination. Experiments on GSM8K and MATH-500 demonstrate the strong performance of AR-Checker on mathematical tasks. We also evaluate AR-Checker on benchmarks beyond mathematics, including MMLU, MMLU-Pro, and CommonsenseQA, where it also achieves strong performance, further proving the effectiveness of AR-Checker. 

---
# Controlling Summarization Length Through EOS Token Weighting 

**Authors**: Zeno Belligoli, Emmanouil Stergiadis, Eran Fainman, Ilya Gusev  

**Link**: [PDF](https://arxiv.org/pdf/2506.05017)  

**Abstract**: Controlling the length of generated text can be crucial in various text-generation tasks, including summarization. Existing methods often require complex model alterations, limiting compatibility with pre-trained models. We address these limitations by developing a simple approach for controlling the length of automatic text summaries by increasing the importance of correctly predicting the EOS token in the cross-entropy loss computation. The proposed methodology is agnostic to architecture and decoding algorithms and orthogonal to other inference-time techniques to control the generation length. We tested it with encoder-decoder and modern GPT-style LLMs, and show that this method can control generation length, often without affecting the quality of the summary. 

---
# ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development 

**Authors**: Zhenran Xu, Xue Yang, Yiyu Wang, Qingli Hu, Zijiao Wu, Longyue Wang, Weihua Luo, Kaifu Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05010)  

**Abstract**: We introduce ComfyUI-Copilot, a large language model-powered plugin designed to enhance the usability and efficiency of ComfyUI, an open-source platform for AI-driven art creation. Despite its flexibility and user-friendly interface, ComfyUI can present challenges to newcomers, including limited documentation, model misconfigurations, and the complexity of workflow design. ComfyUI-Copilot addresses these challenges by offering intelligent node and model recommendations, along with automated one-click workflow construction. At its core, the system employs a hierarchical multi-agent framework comprising a central assistant agent for task delegation and specialized worker agents for different usages, supported by our curated ComfyUI knowledge bases to streamline debugging and deployment. We validate the effectiveness of ComfyUI-Copilot through both offline quantitative evaluations and online user feedback, showing that it accurately recommends nodes and accelerates workflow development. Additionally, use cases illustrate that ComfyUI-Copilot lowers entry barriers for beginners and enhances workflow efficiency for experienced users. The ComfyUI-Copilot installation package and a demo video are available at this https URL. 

---
# SCOP: Evaluating the Comprehension Process of Large Language Models from a Cognitive View 

**Authors**: Yongjie Xiao, Hongru Liang, Peixin Qin, Yao Zhang, Wenqiang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2506.05000)  

**Abstract**: Despite the great potential of large language models(LLMs) in machine comprehension, it is still disturbing to fully count on them in real-world scenarios. This is probably because there is no rational explanation for whether the comprehension process of LLMs is aligned with that of experts. In this paper, we propose SCOP to carefully examine how LLMs perform during the comprehension process from a cognitive view. Specifically, it is equipped with a systematical definition of five requisite skills during the comprehension process, a strict framework to construct testing data for these skills, and a detailed analysis of advanced open-sourced and closed-sourced LLMs using the testing data. With SCOP, we find that it is still challenging for LLMs to perform an expert-level comprehension process. Even so, we notice that LLMs share some similarities with experts, e.g., performing better at comprehending local information than global information. Further analysis reveals that LLMs can be somewhat unreliable -- they might reach correct answers through flawed comprehension processes. Based on SCOP, we suggest that one direction for improving LLMs is to focus more on the comprehension process, ensuring all comprehension skills are thoroughly developed during training. 

---
# Better Semi-supervised Learning for Multi-domain ASR Through Incremental Retraining and Data Filtering 

**Authors**: Andres Carofilis, Pradeep Rangappa, Srikanth Madikeri, Shashi Kumar, Sergio Burdisso, Jeena Prakash, Esau Villatoro-Tello, Petr Motlicek, Bidisha Sharma, Kadri Hacioglu, Shankar Venkatesan, Saurabh Vyas, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2506.04981)  

**Abstract**: Fine-tuning pretrained ASR models for specific domains is challenging when labeled data is scarce. But unlabeled audio and labeled data from related domains are often available. We propose an incremental semi-supervised learning pipeline that first integrates a small in-domain labeled set and an auxiliary dataset from a closely related domain, achieving a relative improvement of 4% over no auxiliary data. Filtering based on multi-model consensus or named entity recognition (NER) is then applied to select and iteratively refine pseudo-labels, showing slower performance saturation compared to random selection. Evaluated on the multi-domain Wow call center and Fisher English corpora, it outperforms single-step fine-tuning. Consensus-based filtering outperforms other methods, providing up to 22.3% relative improvement on Wow and 24.8% on Fisher over single-step fine-tuning with random selection. NER is the second-best filter, providing competitive performance at a lower computational cost. 

---
# From Struggle (06-2024) to Mastery (02-2025) LLMs Conquer Advanced Algorithm Exams and Pave the Way for Editorial Generation 

**Authors**: Adrian Marius Dumitran, Theodor-Pierre Moroianu, Vasile Paul Alexe  

**Link**: [PDF](https://arxiv.org/pdf/2506.04965)  

**Abstract**: This paper presents a comprehensive evaluation of the performance of state-of-the-art Large Language Models (LLMs) on challenging university-level algorithms exams. By testing multiple models on both a Romanian exam and its high-quality English translation, we analyze LLMs' problem-solving capabilities, consistency, and multilingual performance. Our empirical study reveals that the most recent models not only achieve scores comparable to top-performing students but also demonstrate robust reasoning skills on complex, multi-step algorithmic challenges, even though difficulties remain with graph-based tasks. Building on these findings, we explore the potential of LLMs to support educational environments through the generation of high-quality editorial content, offering instructors a powerful tool to enhance student feedback. The insights and best practices discussed herein pave the way for further integration of generative AI in advanced algorithm education. 

---
# ConECT Dataset: Overcoming Data Scarcity in Context-Aware E-Commerce MT 

**Authors**: Mikołaj Pokrywka, Wojciech Kusa, Mieszko Rutkowski, Mikołaj Koszowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.04929)  

**Abstract**: Neural Machine Translation (NMT) has improved translation by using Transformer-based models, but it still struggles with word ambiguity and context. This problem is especially important in domain-specific applications, which often have problems with unclear sentences or poor data quality. Our research explores how adding information to models can improve translations in the context of e-commerce data. To this end we create ConECT -- a new Czech-to-Polish e-commerce product translation dataset coupled with images and product metadata consisting of 11,400 sentence pairs. We then investigate and compare different methods that are applicable to context-aware translation. We test a vision-language model (VLM), finding that visual context aids translation quality. Additionally, we explore the incorporation of contextual information into text-to-text models, such as the product's category path or image descriptions. The results of our study demonstrate that the incorporation of contextual information leads to an improvement in the quality of machine translation. We make the new dataset publicly available. 

---
# Simulating LLM-to-LLM Tutoring for Multilingual Math Feedback 

**Authors**: Junior Cedric Tonga, KV Aditya Srivatsa, Kaushal Kumar Maurya, Fajri Koto, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2506.04920)  

**Abstract**: Large language models (LLMs) have demonstrated the ability to generate formative feedback and instructional hints in English, making them increasingly relevant for AI-assisted education. However, their ability to provide effective instructional support across different languages, especially for mathematically grounded reasoning tasks, remains largely unexamined. In this work, we present the first large-scale simulation of multilingual tutor-student interactions using LLMs. A stronger model plays the role of the tutor, generating feedback in the form of hints, while a weaker model simulates the student. We explore 352 experimental settings across 11 typologically diverse languages, four state-of-the-art LLMs, and multiple prompting strategies to assess whether language-specific feedback leads to measurable learning gains. Our study examines how student input language, teacher feedback language, model choice, and language resource level jointly influence performance. Results show that multilingual hints can significantly improve learning outcomes, particularly in low-resource languages when feedback is aligned with the student's native language. These findings offer practical insights for developing multilingual, LLM-based educational tools that are both effective and inclusive. 

---
# A Practitioner's Guide to Building ASR Models for Low-Resource Languages: A Case Study on Scottish Gaelic 

**Authors**: Ondřej Klejch, William Lamb, Peter Bell  

**Link**: [PDF](https://arxiv.org/pdf/2506.04915)  

**Abstract**: An effective approach to the development of ASR systems for low-resource languages is to fine-tune an existing multilingual end-to-end model. When the original model has been trained on large quantities of data from many languages, fine-tuning can be effective with limited training data, even when the language in question was not present in the original training data. The fine-tuning approach has been encouraged by the availability of public-domain E2E models and is widely believed to lead to state-of-the-art results. This paper, however, challenges that belief. We show that an approach combining hybrid HMMs with self-supervised models can yield substantially better performance with limited training data. This combination allows better utilisation of all available speech and text data through continued self-supervised pre-training and semi-supervised training. We benchmark our approach on Scottish Gaelic, achieving WER reductions of 32% relative over our best fine-tuned Whisper model. 

---
# Verbose ListOps (VLO): Beyond Long Context -- Unmasking LLM's Reasoning Blind Spots 

**Authors**: Alex Pan, Mary-Anne Williams  

**Link**: [PDF](https://arxiv.org/pdf/2506.04907)  

**Abstract**: Large Language Models (LLMs), whilst great at extracting facts from text, struggle with nested narrative reasoning. Existing long context and multi-hop QA benchmarks inadequately test this, lacking realistic distractors or failing to decouple context length from reasoning complexity, masking a fundamental LLM limitation. We introduce Verbose ListOps, a novel benchmark that programmatically transposes ListOps computations into lengthy, coherent stories. This uniquely forces internal computation and state management of nested reasoning problems by withholding intermediate results, and offers fine-grained controls for both narrative size \emph{and} reasoning difficulty. Whilst benchmarks like LongReason (2025) advance approaches for synthetically expanding the context size of multi-hop QA problems, Verbose ListOps pinpoints a specific LLM vulnerability: difficulty in state management for nested sub-reasoning amongst semantically-relevant, distracting narrative. Our experiments show that leading LLMs (e.g., OpenAI o4, Gemini 2.5 Pro) collapse in performance on Verbose ListOps at modest (~10k token) narrative lengths, despite effortlessly solving raw ListOps equations. Addressing this failure is paramount for real-world text interpretation which requires identifying key reasoning points, tracking conceptual intermediate results, and filtering irrelevant information. Verbose ListOps, and its extensible generation framework thus enables targeted reasoning enhancements beyond mere context-window expansion; a critical step to automating the world's knowledge work. 

---
# ICPC-Eval: Probing the Frontiers of LLM Reasoning with Competitive Programming Contests 

**Authors**: Shiyi Xu, Yiwen Hu, Yingqian Min, Zhipeng Chen, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04894)  

**Abstract**: With the significant progress of large reasoning models in complex coding and reasoning tasks, existing benchmarks, like LiveCodeBench and CodeElo, are insufficient to evaluate the coding capabilities of large language models (LLMs) in real competition environments. Moreover, current evaluation metrics such as Pass@K fail to capture the reflective abilities of reasoning models. To address these challenges, we propose \textbf{ICPC-Eval}, a top-level competitive coding benchmark designed to probing the frontiers of LLM reasoning. ICPC-Eval includes 118 carefully curated problems from 11 recent ICPC contests held in various regions of the world, offering three key contributions: 1) A challenging realistic ICPC competition scenario, featuring a problem type and difficulty distribution consistent with actual contests. 2) A robust test case generation method and a corresponding local evaluation toolkit, enabling efficient and accurate local evaluation. 3) An effective test-time scaling evaluation metric, Refine@K, which allows iterative repair of solutions based on execution feedback. The results underscore the significant challenge in evaluating complex reasoning abilities: top-tier reasoning models like DeepSeek-R1 often rely on multi-turn code feedback to fully unlock their in-context reasoning potential when compared to non-reasoning counterparts. Furthermore, despite recent advancements in code generation, these models still lag behind top-performing human teams. We release the benchmark at: this https URL 

---
# Evaluating the Effectiveness of Linguistic Knowledge in Pretrained Language Models: A Case Study of Universal Dependencies 

**Authors**: Wenxi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04887)  

**Abstract**: Universal Dependencies (UD), while widely regarded as the most successful linguistic framework for cross-lingual syntactic representation, remains underexplored in terms of its effectiveness. This paper addresses this gap by integrating UD into pretrained language models and assesses if UD can improve their performance on a cross-lingual adversarial paraphrase identification task. Experimental results show that incorporation of UD yields significant improvements in accuracy and $F_1$ scores, with average gains of 3.85\% and 6.08\% respectively. These enhancements reduce the performance gap between pretrained models and large language models in some language pairs, and even outperform the latter in some others. Furthermore, the UD-based similarity score between a given language and English is positively correlated to the performance of models in that language. Both findings highlight the validity and potential of UD in out-of-domain tasks. 

---
# Prompting LLMs: Length Control for Isometric Machine Translation 

**Authors**: Dávid Javorský, Ondřej Bojar, François Yvon  

**Link**: [PDF](https://arxiv.org/pdf/2506.04855)  

**Abstract**: In this study, we explore the effectiveness of isometric machine translation across multiple language pairs (En$\to$De, En$\to$Fr, and En$\to$Es) under the conditions of the IWSLT Isometric Shared Task 2022. Using eight open-source large language models (LLMs) of varying sizes, we investigate how different prompting strategies, varying numbers of few-shot examples, and demonstration selection influence translation quality and length control. We discover that the phrasing of instructions, when aligned with the properties of the provided demonstrations, plays a crucial role in controlling the output length. Our experiments show that LLMs tend to produce shorter translations only when presented with extreme examples, while isometric demonstrations often lead to the models disregarding length constraints. While few-shot prompting generally enhances translation quality, further improvements are marginal across 5, 10, and 20-shot settings. Finally, considering multiple outputs allows to notably improve overall tradeoff between the length and quality, yielding state-of-the-art performance for some language pairs. 

---
# Multiple-Choice Question Generation Using Large Language Models: Methodology and Educator Insights 

**Authors**: Giorgio Biancini, Alessio Ferrato, Carla Limongelli  

**Link**: [PDF](https://arxiv.org/pdf/2506.04851)  

**Abstract**: Integrating Artificial Intelligence (AI) in educational settings has brought new learning approaches, transforming the practices of both students and educators. Among the various technologies driving this transformation, Large Language Models (LLMs) have emerged as powerful tools for creating educational materials and question answering, but there are still space for new applications. Educators commonly use Multiple-Choice Questions (MCQs) to assess student knowledge, but manually generating these questions is resource-intensive and requires significant time and cognitive effort. In our opinion, LLMs offer a promising solution to these challenges. This paper presents a novel comparative analysis of three widely known LLMs - Llama 2, Mistral, and GPT-3.5 - to explore their potential for creating informative and challenging MCQs. In our approach, we do not rely on the knowledge of the LLM, but we inject the knowledge into the prompt to contrast the hallucinations, giving the educators control over the test's source text, too. Our experiment involving 21 educators shows that GPT-3.5 generates the most effective MCQs across several known metrics. Additionally, it shows that there is still some reluctance to adopt AI in the educational field. This study sheds light on the potential of LLMs to generate MCQs and improve the educational experience, providing valuable insights for the future. 

---
# MockConf: A Student Interpretation Dataset: Analysis, Word- and Span-level Alignment and Baselines 

**Authors**: Dávid Javorský, Ondřej Bojar, François Yvon  

**Link**: [PDF](https://arxiv.org/pdf/2506.04848)  

**Abstract**: In simultaneous interpreting, an interpreter renders a source speech into another language with a very short lag, much sooner than sentences are finished. In order to understand and later reproduce this dynamic and complex task automatically, we need dedicated datasets and tools for analysis, monitoring, and evaluation, such as parallel speech corpora, and tools for their automatic annotation. Existing parallel corpora of translated texts and associated alignment algorithms hardly fill this gap, as they fail to model long-range interactions between speech segments or specific types of divergences (e.g., shortening, simplification, functional generalization) between the original and interpreted speeches. In this work, we introduce MockConf, a student interpreting dataset that was collected from Mock Conferences run as part of the students' curriculum. This dataset contains 7 hours of recordings in 5 European languages, transcribed and aligned at the level of spans and words. We further implement and release InterAlign, a modern web-based annotation tool for parallel word and span annotations on long inputs, suitable for aligning simultaneous interpreting. We propose metrics for the evaluation and a baseline for automatic alignment. Dataset and tools are released to the community. 

---
# Joint Evaluation of Answer and Reasoning Consistency for Hallucination Detection in Large Reasoning Models 

**Authors**: Changyue Wang, Weihang Su, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04832)  

**Abstract**: Large Reasoning Models (LRMs) extend large language models with explicit, multi-step reasoning traces to enhance transparency and performance on complex tasks. However, these reasoning traces can be redundant or logically inconsistent, making them a new source of hallucination that is difficult to detect. Existing hallucination detection methods focus primarily on answer-level uncertainty and often fail to detect hallucinations or logical inconsistencies arising from the model's reasoning trace. This oversight is particularly problematic for LRMs, where the explicit thinking trace is not only an important support to the model's decision-making process but also a key source of potential hallucination. To this end, we propose RACE (Reasoning and Answer Consistency Evaluation), a novel framework specifically tailored for hallucination detection in LRMs. RACE operates by extracting essential reasoning steps and computing four diagnostic signals: inter-sample consistency of reasoning traces, entropy-based answer uncertainty, semantic alignment between reasoning and answers, and internal coherence of reasoning. This joint analysis enables fine-grained hallucination detection even when the final answer appears correct. Experiments across datasets and different LLMs demonstrate that RACE outperforms existing hallucination detection baselines, offering a robust and generalizable solution for evaluating LRMs. Our code is available at: this https URL. 

---
# A Reasoning-Based Approach to Cryptic Crossword Clue Solving 

**Authors**: Martin Andrews, Sam Witteveen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04824)  

**Abstract**: Cryptic crossword clues are challenging language tasks for which new test sets are released daily by major newspapers on a global basis. Each cryptic clue contains both the definition of the answer to be placed in the crossword grid (in common with regular crosswords), and 'wordplay' that proves that the answer is correct (i.e. a human solver can be confident that an answer is correct without needing crossing words as confirmation). This work describes an LLM-based reasoning system built from open-licensed components that solves cryptic clues by (i) hypothesising answers; (ii) proposing wordplay explanations; and (iii) using a verifier system that operates on codified reasoning steps. Overall, this system establishes a new state-of-the-art performance on the challenging Cryptonite dataset of clues from The Times and The Telegraph newspapers in the UK. Because each proved solution is expressed in Python, interpretable wordplay reasoning for proven answers is available for inspection. 

---
# Evaluating Vision-Language and Large Language Models for Automated Student Assessment in Indonesian Classrooms 

**Authors**: Nurul Aisyah, Muhammad Dehan Al Kautsar, Arif Hidayat, Raqib Chowdhury, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2506.04822)  

**Abstract**: Although vision-language and large language models (VLM and LLM) offer promising opportunities for AI-driven educational assessment, their effectiveness in real-world classroom settings, particularly in underrepresented educational contexts, remains underexplored. In this study, we evaluated the performance of a state-of-the-art VLM and several LLMs on 646 handwritten exam responses from grade 4 students in six Indonesian schools, covering two subjects: Mathematics and English. These sheets contain more than 14K student answers that span multiple choice, short answer, and essay questions. Assessment tasks include grading these responses and generating personalized feedback. Our findings show that the VLM often struggles to accurately recognize student handwriting, leading to error propagation in downstream LLM grading. Nevertheless, LLM-generated feedback retains some utility, even when derived from imperfect input, although limitations in personalization and contextual relevance persist. 

---
# Design of intelligent proofreading system for English translation based on CNN and BERT 

**Authors**: Feijun Liu, Huifeng Wang, Kun Wang, Yizhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04811)  

**Abstract**: Since automatic translations can contain errors that require substantial human post-editing, machine translation proofreading is essential for improving quality. This paper proposes a novel hybrid approach for robust proofreading that combines convolutional neural networks (CNN) with Bidirectional Encoder Representations from Transformers (BERT). In order to extract semantic information from phrases and expressions, CNN uses a variety of convolution kernel filters to capture local n-gram patterns. In the meanwhile, BERT creates context-rich representations of whole sequences by utilizing stacked bidirectional transformer encoders. Using BERT's attention processes, the integrated error detection component relates tokens to spot translation irregularities including word order problems and omissions. The correction module then uses parallel English-German alignment and GRU decoder models in conjunction with translation memory to propose logical modifications that maintain original meaning. A unified end-to-end training process optimized for post-editing performance is applied to the whole pipeline. The multi-domain collection of WMT and the conversational dialogues of Open-Subtitles are two of the English-German parallel corpora used to train the model. Multiple loss functions supervise detection and correction capabilities. Experiments attain a 90% accuracy, 89.37% F1, and 16.24% MSE, exceeding recent proofreading techniques by over 10% overall. Comparative benchmarking demonstrates state-of-the-art performance in identifying and coherently rectifying mistranslations and omissions. 

---
# Dissecting Logical Reasoning in LLMs: A Fine-Grained Evaluation and Supervision Study 

**Authors**: Yujun Zhou, Jiayi Ye, Zipeng Ling, Yufei Han, Yue Huang, Haomin Zhuang, Zhenwen Liang, Kehan Guo, Taicheng Guo, Xiangqi Wang, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04810)  

**Abstract**: Logical reasoning is a core capability for many applications of large language models (LLMs), yet existing benchmarks often rely solely on final-answer accuracy, failing to capture the quality and structure of the reasoning process. We propose FineLogic, a fine-grained evaluation framework that assesses logical reasoning across three dimensions: overall benchmark accuracy, stepwise soundness, and representation-level alignment. In addition, to better understand how reasoning capabilities emerge, we conduct a comprehensive study on the effects of supervision format during fine-tuning. We construct four supervision styles (one natural language and three symbolic variants) and train LLMs under each. Our findings reveal that natural language supervision yields strong generalization even on out-of-distribution and long-context tasks, while symbolic reasoning styles promote more structurally sound and atomic inference chains. Further, our representation-level probing shows that fine-tuning primarily improves reasoning behaviors through step-by-step generation, rather than enhancing shortcut prediction or internalized correctness. Together, our framework and analysis provide a more rigorous and interpretable lens for evaluating and improving logical reasoning in LLMs. 

---
# Towards LLM-Centric Multimodal Fusion: A Survey on Integration Strategies and Techniques 

**Authors**: Jisu An, Junseok Lee, Jeoungeun Lee, Yongseok Son  

**Link**: [PDF](https://arxiv.org/pdf/2506.04788)  

**Abstract**: The rapid progress of Multimodal Large Language Models(MLLMs) has transformed the AI landscape. These models combine pre-trained LLMs with various modality encoders. This integration requires a systematic understanding of how different modalities connect to the language backbone. Our survey presents an LLM-centric analysis of current approaches. We examine methods for transforming and aligning diverse modal inputs into the language embedding space. This addresses a significant gap in existing literature. We propose a classification framework for MLLMs based on three key dimensions. First, we examine architectural strategies for modality integration. This includes both the specific integration mechanisms and the fusion level. Second, we categorize representation learning techniques as either joint or coordinate representations. Third, we analyze training paradigms, including training strategies and objective functions. By examining 125 MLLMs developed between 2021 and 2025, we identify emerging patterns in the field. Our taxonomy provides researchers with a structured overview of current integration techniques. These insights aim to guide the development of more robust multimodal integration strategies for future models built on pre-trained foundations. 

---
# MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark 

**Authors**: Dingdong Wang, Jincenzi Wu, Junan Li, Dongchao Yang, Xueyuan Chen, Tianhua Zhang, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2506.04779)  

**Abstract**: Speech inherently contains rich acoustic information that extends far beyond the textual language. In real-world spoken language understanding, effective interpretation often requires integrating semantic meaning (e.g., content), paralinguistic features (e.g., emotions, speed, pitch) and phonological characteristics (e.g., prosody, intonation, rhythm), which are embedded in speech. While recent multimodal Speech Large Language Models (SpeechLLMs) have demonstrated remarkable capabilities in processing audio information, their ability to perform fine-grained perception and complex reasoning in natural speech remains largely unexplored. To address this gap, we introduce MMSU, a comprehensive benchmark designed specifically for understanding and reasoning in spoken language. MMSU comprises 5,000 meticulously curated audio-question-answer triplets across 47 distinct tasks. To ground our benchmark in linguistic theory, we systematically incorporate a wide range of linguistic phenomena, including phonetics, prosody, rhetoric, syntactics, semantics, and paralinguistics. Through a rigorous evaluation of 14 advanced SpeechLLMs, we identify substantial room for improvement in existing models, highlighting meaningful directions for future optimization. MMSU establishes a new standard for comprehensive assessment of spoken language understanding, providing valuable insights for developing more sophisticated human-AI speech interaction systems. MMSU benchmark is available at this https URL. Evaluation Code is available at this https URL. 

---
# Fine-Grained Interpretation of Political Opinions in Large Language Models 

**Authors**: Jingyu Hu, Mengyue Yang, Mengnan Du, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04774)  

**Abstract**: Studies of LLMs' political opinions mainly rely on evaluations of their open-ended responses. Recent work indicates that there is a misalignment between LLMs' responses and their internal intentions. This motivates us to probe LLMs' internal mechanisms and help uncover their internal political states. Additionally, we found that the analysis of LLMs' political opinions often relies on single-axis concepts, which can lead to concept confounds. In this work, we extend the single-axis to multi-dimensions and apply interpretable representation engineering techniques for more transparent LLM political concept learning. Specifically, we designed a four-dimensional political learning framework and constructed a corresponding dataset for fine-grained political concept vector learning. These vectors can be used to detect and intervene in LLM internals. Experiments are conducted on eight open-source LLMs with three representation engineering techniques. Results show these vectors can disentangle political concept confounds. Detection tasks validate the semantic meaning of the vectors and show good generalization and robustness in OOD settings. Intervention Experiments show these vectors can intervene in LLMs to generate responses with different political leanings. 

---
# Identifying Reliable Evaluation Metrics for Scientific Text Revision 

**Authors**: Léane Jourdan, Florian Boudin, Richard Dufour, Nicolas Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.04772)  

**Abstract**: Evaluating text revision in scientific writing remains a challenge, as traditional metrics such as ROUGE and BERTScore primarily focus on similarity rather than capturing meaningful improvements. In this work, we analyse and identify the limitations of these metrics and explore alternative evaluation methods that better align with human judgments. We first conduct a manual annotation study to assess the quality of different revisions. Then, we investigate reference-free evaluation metrics from related NLP domains. Additionally, we examine LLM-as-a-judge approaches, analysing their ability to assess revisions with and without a gold reference. Our results show that LLMs effectively assess instruction-following but struggle with correctness, while domain-specific metrics provide complementary insights. We find that a hybrid approach combining LLM-as-a-judge evaluation and task-specific metrics offers the most reliable assessment of revision quality. 

---
# Lifelong Evolution: Collaborative Learning between Large and Small Language Models for Continuous Emergent Fake News Detection 

**Authors**: Ziyi Zhou, Xiaoming Zhang, Litian Zhang, Yibo Zhang, Zhenyu Guan, Chaozhuo Li, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04739)  

**Abstract**: The widespread dissemination of fake news on social media has significantly impacted society, resulting in serious consequences. Conventional deep learning methodologies employing small language models (SLMs) suffer from extensive supervised training requirements and difficulties adapting to evolving news environments due to data scarcity and distribution shifts. Large language models (LLMs), despite robust zero-shot capabilities, fall short in accurately detecting fake news owing to outdated knowledge and the absence of suitable demonstrations. In this paper, we propose a novel Continuous Collaborative Emergent Fake News Detection (C$^2$EFND) framework to address these challenges. The C$^2$EFND framework strategically leverages both LLMs' generalization power and SLMs' classification expertise via a multi-round collaborative learning framework. We further introduce a lifelong knowledge editing module based on a Mixture-of-Experts architecture to incrementally update LLMs and a replay-based continue learning method to ensure SLMs retain prior knowledge without retraining entirely. Extensive experiments on Pheme and Twitter16 datasets demonstrate that C$^2$EFND significantly outperforms existed methods, effectively improving detection accuracy and adaptability in continuous emergent fake news scenarios. 

---
# SPARTA ALIGNMENT: Collectively Aligning Multiple Language Models through Combat 

**Authors**: Yuru Jiang, Wenxuan Ding, Shangbin Feng, Greg Durrett, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2506.04721)  

**Abstract**: We propose SPARTA ALIGNMENT, an algorithm to collectively align multiple LLMs through competition and combat. To complement a single model's lack of diversity in generation and biases in evaluation, multiple LLMs form a "sparta tribe" to compete against each other in fulfilling instructions while serving as judges for the competition of others. For each iteration, one instruction and two models are selected for a duel, the other models evaluate the two responses, and their evaluation scores are aggregated through a adapted elo-ranking based reputation system, where winners/losers of combat gain/lose weight in evaluating others. The peer-evaluated combat results then become preference pairs where the winning response is preferred over the losing one, and all models learn from these preferences at the end of each iteration. SPARTA ALIGNMENT enables the self-evolution of multiple LLMs in an iterative and collective competition process. Extensive experiments demonstrate that SPARTA ALIGNMENT outperforms initial models and 4 self-alignment baselines across 10 out of 12 tasks and datasets with 7.0% average improvement. Further analysis reveals that SPARTA ALIGNMENT generalizes more effectively to unseen tasks and leverages the expertise diversity of participating models to produce more logical, direct and informative outputs. 

---
# IIITH-BUT system for IWSLT 2025 low-resource Bhojpuri to Hindi speech translation 

**Authors**: Bhavana Akkiraju, Aishwarya Pothula, Santosh Kesiraju, Anil Kumar Vuppala  

**Link**: [PDF](https://arxiv.org/pdf/2506.04714)  

**Abstract**: This paper presents the submission of IIITH-BUT to the IWSLT 2025 shared task on speech translation for the low-resource Bhojpuri-Hindi language pair. We explored the impact of hyperparameter optimisation and data augmentation techniques on the performance of the SeamlessM4T model fine-tuned for this specific task. We systematically investigated a range of hyperparameters including learning rate schedules, number of update steps, warm-up steps, label smoothing, and batch sizes; and report their effect on translation quality. To address data scarcity, we applied speed perturbation and SpecAugment and studied their effect on translation quality. We also examined the use of cross-lingual signal through joint training with Marathi and Bhojpuri speech data. Our experiments reveal that careful selection of hyperparameters and the application of simple yet effective augmentation techniques significantly improve performance in low-resource settings. We also analysed the translation hypotheses to understand various kinds of errors that impacted the translation quality in terms of BLEU. 

---
# Accelerated Test-Time Scaling with Model-Free Speculative Sampling 

**Authors**: Woomin Song, Saket Dingliwal, Sai Muralidhar Jayanthi, Bhavana Ganesh, Jinwoo Shin, Aram Galstyan, Sravan Babu Bodapati  

**Link**: [PDF](https://arxiv.org/pdf/2506.04708)  

**Abstract**: Language models have demonstrated remarkable capabilities in reasoning tasks through test-time scaling techniques like best-of-N sampling and tree search. However, these approaches often demand substantial computational resources, creating a critical trade-off between performance and efficiency. We introduce STAND (STochastic Adaptive N-gram Drafting), a novel model-free speculative decoding approach that leverages the inherent redundancy in reasoning trajectories to achieve significant acceleration without compromising accuracy. Our analysis reveals that reasoning paths frequently reuse similar reasoning patterns, enabling efficient model-free token prediction without requiring separate draft models. By introducing stochastic drafting and preserving probabilistic information through a memory-efficient logit-based N-gram module, combined with optimized Gumbel-Top-K sampling and data-driven tree construction, STAND significantly improves token acceptance rates. Extensive evaluations across multiple models and reasoning tasks (AIME-2024, GPQA-Diamond, and LiveCodeBench) demonstrate that STAND reduces inference latency by 60-65% compared to standard autoregressive decoding while maintaining accuracy. Furthermore, STAND outperforms state-of-the-art speculative decoding methods by 14-28% in throughput and shows strong performance even in single-trajectory scenarios, reducing inference latency by 48-58%. As a model-free approach, STAND can be applied to any existing language model without additional training, being a powerful plug-and-play solution for accelerating language model reasoning. 

---
# Cracking the Code: Enhancing Implicit Hate Speech Detection through Coding Classification 

**Authors**: Lu Wei, Liangzhi Li, Tong Xiang, Xiao Liu, Noa Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2506.04693)  

**Abstract**: The internet has become a hotspot for hate speech (HS), threatening societal harmony and individual well-being. While automatic detection methods perform well in identifying explicit hate speech (ex-HS), they struggle with more subtle forms, such as implicit hate speech (im-HS). We tackle this problem by introducing a new taxonomy for im-HS detection, defining six encoding strategies named codetypes. We present two methods for integrating codetypes into im-HS detection: 1) prompting large language models (LLMs) directly to classify sentences based on generated responses, and 2) using LLMs as encoders with codetypes embedded during the encoding process. Experiments show that the use of codetypes improves im-HS detection in both Chinese and English datasets, validating the effectiveness of our approach across different languages. 

---
# Recycling the Web: A Method to Enhance Pre-training Data Quality and Quantity for Language Models 

**Authors**: Thao Nguyen, Yang Li, Olga Golovneva, Luke Zettlemoyer, Sewoong Oh, Ludwig Schmidt, Xian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04689)  

**Abstract**: Scaling laws predict that the performance of large language models improves with increasing model size and data size. In practice, pre-training has been relying on massive web crawls, using almost all data sources publicly available on the internet so far. However, this pool of natural data does not grow at the same rate as the compute supply. Furthermore, the availability of high-quality texts is even more limited: data filtering pipelines often remove up to 99% of the initial web scrapes to achieve state-of-the-art. To address the "data wall" of pre-training scaling, our work explores ways to transform and recycle data discarded in existing filtering processes. We propose REWIRE, REcycling the Web with guIded REwrite, a method to enrich low-quality documents so that they could become useful for training. This in turn allows us to increase the representation of synthetic data in the final pre-training set. Experiments at 1B, 3B and 7B scales of the DCLM benchmark show that mixing high-quality raw texts and our rewritten texts lead to 1.0, 1.3 and 2.5 percentage points improvement respectively across 22 diverse tasks, compared to training on only filtered web data. Training on the raw-synthetic data mix is also more effective than having access to 2x web data. Through further analysis, we demonstrate that about 82% of the mixed in texts come from transforming lower-quality documents that would otherwise be discarded. REWIRE also outperforms related approaches of generating synthetic data, including Wikipedia-style paraphrasing, question-answer synthesizing and knowledge extraction. These results suggest that recycling web texts holds the potential for being a simple and effective approach for scaling pre-training data. 

---
# MMRefine: Unveiling the Obstacles to Robust Refinement in Multimodal Large Language Models 

**Authors**: Gio Paik, Geewook Kim, Jinbae Im  

**Link**: [PDF](https://arxiv.org/pdf/2506.04688)  

**Abstract**: This paper introduces MMRefine, a MultiModal Refinement benchmark designed to evaluate the error refinement capabilities of Multimodal Large Language Models (MLLMs). As the emphasis shifts toward enhancing reasoning during inference, MMRefine provides a framework that evaluates MLLMs' abilities to detect and correct errors across six distinct scenarios beyond just comparing final accuracy before and after refinement. Furthermore, the benchmark analyzes the refinement performance by categorizing errors into six error types. Experiments with various open and closed MLLMs reveal bottlenecks and factors impeding refinement performance, highlighting areas for improvement in effective reasoning enhancement. Our code and dataset are publicly available at this https URL. 

---
# Normative Conflicts and Shallow AI Alignment 

**Authors**: Raphaël Millière  

**Link**: [PDF](https://arxiv.org/pdf/2506.04679)  

**Abstract**: The progress of AI systems such as large language models (LLMs) raises increasingly pressing concerns about their safe deployment. This paper examines the value alignment problem for LLMs, arguing that current alignment strategies are fundamentally inadequate to prevent misuse. Despite ongoing efforts to instill norms such as helpfulness, honesty, and harmlessness in LLMs through fine-tuning based on human preferences, they remain vulnerable to adversarial attacks that exploit conflicts between these norms. I argue that this vulnerability reflects a fundamental limitation of existing alignment methods: they reinforce shallow behavioral dispositions rather than endowing LLMs with a genuine capacity for normative deliberation. Drawing from on research in moral psychology, I show how humans' ability to engage in deliberative reasoning enhances their resilience against similar adversarial tactics. LLMs, by contrast, lack a robust capacity to detect and rationally resolve normative conflicts, leaving them susceptible to manipulation; even recent advances in reasoning-focused LLMs have not addressed this vulnerability. This ``shallow alignment'' problem carries significant implications for AI safety and regulation, suggesting that current approaches are insufficient for mitigating potential harms posed by increasingly capable AI systems. 

---
# Flex-TravelPlanner: A Benchmark for Flexible Planning with Language Agents 

**Authors**: Juhyun Oh, Eunsu Kim, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2506.04649)  

**Abstract**: Real-world planning problems require constant adaptation to changing requirements and balancing of competing constraints. However, current benchmarks for evaluating LLMs' planning capabilities primarily focus on static, single-turn scenarios. We introduce Flex-TravelPlanner, a benchmark that evaluates language models' ability to reason flexibly in dynamic planning scenarios. Building on the TravelPlanner dataset~\citep{xie2024travelplanner}, we introduce two novel evaluation settings: (1) sequential constraint introduction across multiple turns, and (2) scenarios with explicitly prioritized competing constraints. Our analysis of GPT-4o and Llama 3.1 70B reveals several key findings: models' performance on single-turn tasks poorly predicts their ability to adapt plans across multiple turns; constraint introduction order significantly affects performance; and models struggle with constraint prioritization, often incorrectly favoring newly introduced lower priority preferences over existing higher-priority constraints. These findings highlight the importance of evaluating LLMs in more realistic, dynamic planning scenarios and suggest specific directions for improving model performance on complex planning tasks. The code and dataset for our framework are publicly available at this https URL. 

---
# TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering 

**Authors**: Vinay Joshi, Pratik Prabhanjan Brahma, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2506.04642)  

**Abstract**: The key-value (KV) cache in transformer models is a critical component for efficient decoding or inference, yet its memory demands scale poorly with sequence length, posing a major challenge for scalable deployment of large language models. Among several approaches to KV cache compression, quantization of key and value activations has been widely explored. Most KV cache quantization methods still need to manage sparse and noncontiguous outliers separately. To address this, we introduce TaDA, a training-free recipe for KV cache compression with quantization precision that adapts to error sensitivity across layers and a mean centering to eliminate separate outlier handling. Our approach yields substantial accuracy improvements for multiple models supporting various context lengths. Moreover, our approach does not need to separately manage outlier elements -- a persistent hurdle in most traditional quantization methods. Experiments on standard benchmarks demonstrate that our technique reduces KV cache memory footprint to 27% of the original 16-bit baseline while achieving comparable accuracy. Our method paves the way for scalable and high-performance reasoning in language models by potentially enabling inference for longer context length models, reasoning models, and longer chain of thoughts. 

---
# ViCocktail: Automated Multi-Modal Data Collection for Vietnamese Audio-Visual Speech Recognition 

**Authors**: Thai-Binh Nguyen, Thi Van Nguyen, Quoc Truong Do, Chi Mai Luong  

**Link**: [PDF](https://arxiv.org/pdf/2506.04635)  

**Abstract**: Audio-Visual Speech Recognition (AVSR) has gained significant attention recently due to its robustness against noise, which often challenges conventional speech recognition systems that rely solely on audio features. Despite this advantage, AVSR models remain limited by the scarcity of extensive datasets, especially for most languages beyond English. Automated data collection offers a promising solution. This work presents a practical approach to generate AVSR datasets from raw video, refining existing techniques for improved efficiency and accessibility. We demonstrate its broad applicability by developing a baseline AVSR model for Vietnamese. Experiments show the automatically collected dataset enables a strong baseline, achieving competitive performance with robust ASR in clean conditions and significantly outperforming them in noisy environments like cocktail parties. This efficient method provides a pathway to expand AVSR to more languages, particularly under-resourced ones. 

---
# Advancing Tool-Augmented Large Language Models via Meta-Verification and Reflection Learning 

**Authors**: Zhiyuan Ma, Jiayu Liu, Xianzhen Luo, Zhenya Huang, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2506.04625)  

**Abstract**: Empowering large language models (LLMs) with effective tool utilization capabilities is crucial for enabling AI agents to solve complex problems. However, current models face two major limitations: (1) unreliable tool planning and invocation due to low-quality instruction datasets (e.g., widespread hallucinated API calls), and (2) weak tool reflection abilities (over 90% of errors cannot be corrected) resulting from static imitation learning. To address these critical limitations, we propose Tool-MVR, a novel Tool-Augmented LLM that achieves comprehensive System 2 reasoning through two key innovations. Specifically, we first introduce Multi-Agent Meta-Verification (MAMV), a systematic pipeline that rigorously validates APIs, queries, and reasoning trajectories to construct ToolBench-V, a new high-quality instruction dataset that addresses the limitation of unreliable tool planning and invocation. Second, we propose Exploration-based Reflection Learning (EXPLORE), which enhances tool reflection capabilities by leveraging tool feedback through a dynamic "Error -> Reflection -> Correction" learning paradigm, resulting in our reflection dataset ToolBench-R and addressing the critical weakness in tool reflection. Finally, we obtain Tool-MVR by finetuning open-source LLMs (e.g., Qwen-7B) on both ToolBench-V and ToolBench-R. Our experiments demonstrate that Tool-MVR achieves state-of-the-art performance on StableToolBench, surpassing both ToolLLM (by 23.9%) and GPT-4 (by 15.3%) while reducing API calls by 31.4%, with strong generalization capabilities across unseen tools and scenarios. Additionally, on our proposed RefineToolBench, the first benchmark specifically designed to evaluate tool reflection capabilities, Tool-MVR achieves a 58.9% error correction rate, significantly outperforming ToolLLM's 9.1%. 

---
# Static Word Embeddings for Sentence Semantic Representation 

**Authors**: Takashi Wada, Yuki Hirakawa, Ryotaro Shimizu, Takahiro Kawashima, Yuki Saito  

**Link**: [PDF](https://arxiv.org/pdf/2506.04624)  

**Abstract**: We propose new static word embeddings optimised for sentence semantic representation. We first extract word embeddings from a pre-trained Sentence Transformer, and improve them with sentence-level principal component analysis, followed by either knowledge distillation or contrastive learning. During inference, we represent sentences by simply averaging word embeddings, which requires little computational cost. We evaluate models on both monolingual and cross-lingual tasks and show that our model substantially outperforms existing static models on sentence semantic tasks, and even rivals a basic Sentence Transformer model (SimCSE) on some data sets. Lastly, we perform a variety of analyses and show that our method successfully removes word embedding components that are irrelevant to sentence semantics, and adjusts the vector norms based on the influence of words on sentence semantics. 

---
# Subjective Perspectives within Learned Representations Predict High-Impact Innovation 

**Authors**: Likun Cao, Rui Pan, James Evans  

**Link**: [PDF](https://arxiv.org/pdf/2506.04616)  

**Abstract**: Existing studies of innovation emphasize the power of social structures to shape innovation capacity. Emerging machine learning approaches, however, enable us to model innovators' personal perspectives and interpersonal innovation opportunities as a function of their prior trajectories of experience. We theorize then quantify subjective perspectives and innovation opportunities based on innovator positions within the geometric space of concepts inscribed by dynamic language representations. Using data on millions of scientists, inventors, writers, entrepreneurs, and Wikipedia contributors across the creative domains of science, technology, film, entrepreneurship, and Wikipedia, here we show that measured subjective perspectives anticipate what ideas individuals and groups creatively attend to and successfully combine in future. When perspective and background diversity are decomposed as the angular difference between collaborators' perspectives on their creation and between their experiences, the former consistently anticipates creative achievement while the latter portends its opposite, across all cases and time periods examined. We analyze a natural experiment and simulate creative collaborations between AI (large language model) agents designed with various perspective and background diversity, which are consistent with our observational findings. We explore mechanisms underlying these findings and identify how successful collaborators leverage common language to weave together diverse experience obtained through trajectories of prior work that converge to provoke one another and innovate. We explore the importance of these findings for team assembly and research policy. 

---
# Revisiting Test-Time Scaling: A Survey and a Diversity-Aware Method for Efficient Reasoning 

**Authors**: Ho-Lam Chung, Teng-Yun Hsiao, Hsiao-Ying Huang, Chunerh Cho, Jian-Ren Lin, Zhang Ziwei, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04611)  

**Abstract**: Test-Time Scaling (TTS) improves the reasoning performance of Large Language Models (LLMs) by allocating additional compute during inference. We conduct a structured survey of TTS methods and categorize them into sampling-based, search-based, and trajectory optimization strategies. We observe that reasoning-optimized models often produce less diverse outputs, which limits TTS effectiveness. To address this, we propose ADAPT (A Diversity Aware Prefix fine-Tuning), a lightweight method that applies prefix tuning with a diversity-focused data strategy. Experiments on mathematical reasoning tasks show that ADAPT reaches 80% accuracy using eight times less compute than strong baselines. Our findings highlight the essential role of generative diversity in maximizing TTS effectiveness. 

---
# A MISMATCHED Benchmark for Scientific Natural Language Inference 

**Authors**: Firoz Shaik, Mobashir Sadat, Nikita Gautam, Doina Caragea, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2506.04603)  

**Abstract**: Scientific Natural Language Inference (NLI) is the task of predicting the semantic relation between a pair of sentences extracted from research articles. Existing datasets for this task are derived from various computer science (CS) domains, whereas non-CS domains are completely ignored. In this paper, we introduce a novel evaluation benchmark for scientific NLI, called MISMATCHED. The new MISMATCHED benchmark covers three non-CS domains-PSYCHOLOGY, ENGINEERING, and PUBLIC HEALTH, and contains 2,700 human annotated sentence pairs. We establish strong baselines on MISMATCHED using both Pre-trained Small Language Models (SLMs) and Large Language Models (LLMs). Our best performing baseline shows a Macro F1 of only 78.17% illustrating the substantial headroom for future improvements. In addition to introducing the MISMATCHED benchmark, we show that incorporating sentence pairs having an implicit scientific NLI relation between them in model training improves their performance on scientific NLI. We make our dataset and code publicly available on GitHub. 

---
# Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification 

**Authors**: Chengwu Liu, Ye Yuan, Yichun Yin, Yan Xu, Xin Xu, Zaoyu Chen, Yasheng Wang, Lifeng Shang, Qun Liu, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04592)  

**Abstract**: Chain-of-Thought (CoT) prompting has become the de facto method to elicit reasoning capabilities from large language models (LLMs). However, to mitigate hallucinations in CoT that are notoriously difficult to detect, current methods such as process reward models (PRMs) or self-consistency operate as opaque boxes and do not provide checkable evidence for their judgments, possibly limiting their effectiveness. To address this issue, we draw inspiration from the idea that "the gold standard for supporting a mathematical claim is to provide a proof". We propose a retrospective, step-aware formal verification framework $Safe$. Rather than assigning arbitrary scores, we strive to articulate mathematical claims in formal mathematical language Lean 4 at each reasoning step and provide formal proofs to identify hallucinations. We evaluate our framework $Safe$ across multiple language models and various mathematical datasets, demonstrating a significant performance improvement while offering interpretable and verifiable evidence. We also propose $FormalStep$ as a benchmark for step correctness theorem proving with $30,809$ formal statements. To the best of our knowledge, our work represents the first endeavor to utilize formal mathematical language Lean 4 for verifying natural language content generated by LLMs, aligning with the reason why formal mathematical languages were created in the first place: to provide a robust foundation for hallucination-prone human-written proofs. 

---
# LESS: Large Language Model Enhanced Semi-Supervised Learning for Speech Foundational Models 

**Authors**: Wen Ding, Fan Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.04586)  

**Abstract**: We introduce LESS (Large Language Model Enhanced Semi-supervised Learning), a versatile framework that leverages Large Language Models (LLMs) to correct pseudo labels generated from in-the-wild data. Within the LESS framework, pseudo-labeled text from Automatic Speech Recognition (ASR) or Automatic Speech Translation (AST) of the unsupervised data is refined by an LLM, and augmented by a data filtering strategy to optimize LLM knowledge transfer efficiency. Experiments on both Mandarin ASR and Spanish-to-English AST tasks show that LESS achieves a notable absolute WER reduction of 3.77% on the Wenet Speech test set, as well as BLEU scores of 34.0 and 64.7 on Callhome and Fisher test sets respectively. These results validate the adaptability of LESS across different languages, tasks, and domains. Ablation studies conducted with various LLMs and prompt configurations provide novel insights into leveraging LLM-derived knowledge for speech processing applications. 

---
# MuSciClaims: Multimodal Scientific Claim Verification 

**Authors**: Yash Kumar Lal, Manikanta Bandham, Mohammad Saqib Hasan, Apoorva Kashi, Mahnaz Koupaee, Niranjan Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.04585)  

**Abstract**: Assessing scientific claims requires identifying, extracting, and reasoning with multimodal data expressed in information-rich figures in scientific literature. Despite the large body of work in scientific QA, figure captioning, and other multimodal reasoning tasks over chart-based data, there are no readily usable multimodal benchmarks that directly test claim verification abilities. To remedy this gap, we introduce a new benchmark MuSciClaims accompanied by diagnostics tasks. We automatically extract supported claims from scientific articles, which we manually perturb to produce contradicted claims. The perturbations are designed to test for a specific set of claim verification capabilities. We also introduce a suite of diagnostic tasks that help understand model failures. Our results show most vision-language models are poor (~0.3-0.5 F1), with even the best model only achieving 0.77 F1. They are also biased towards judging claims as supported, likely misunderstanding nuanced perturbations within the claims. Our diagnostics show models are bad at localizing correct evidence within figures, struggle with aggregating information across modalities, and often fail to understand basic components of the figure. 

---
# SUCEA: Reasoning-Intensive Retrieval for Adversarial Fact-checking through Claim Decomposition and Editing 

**Authors**: Hongjun Liu, Yilun Zhao, Arman Cohan, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.04583)  

**Abstract**: Automatic fact-checking has recently received more attention as a means of combating misinformation. Despite significant advancements, fact-checking systems based on retrieval-augmented language models still struggle to tackle adversarial claims, which are intentionally designed by humans to challenge fact-checking systems. To address these challenges, we propose a training-free method designed to rephrase the original claim, making it easier to locate supporting evidence. Our modular framework, SUCEA, decomposes the task into three steps: 1) Claim Segmentation and Decontextualization that segments adversarial claims into independent sub-claims; 2) Iterative Evidence Retrieval and Claim Editing that iteratively retrieves evidence and edits the subclaim based on the retrieved evidence; 3) Evidence Aggregation and Label Prediction that aggregates all retrieved evidence and predicts the entailment label. Experiments on two challenging fact-checking datasets demonstrate that our framework significantly improves on both retrieval and entailment label accuracy, outperforming four strong claim-decomposition-based baselines. 

---
# Selecting Demonstrations for Many-Shot In-Context Learning via Gradient Matching 

**Authors**: Jianfei Zhang, Bei Li, Jun Bai, Rumei Li, Yanmeng Wang, Chenghua Lin, Wenge Rong  

**Link**: [PDF](https://arxiv.org/pdf/2506.04579)  

**Abstract**: In-Context Learning (ICL) empowers Large Language Models (LLMs) for rapid task adaptation without Fine-Tuning (FT), but its reliance on demonstration selection remains a critical challenge. While many-shot ICL shows promising performance through scaled demonstrations, the selection method for many-shot demonstrations remains limited to random selection in existing work. Since the conventional instance-level retrieval is not suitable for many-shot scenarios, we hypothesize that the data requirements for in-context learning and fine-tuning are analogous. To this end, we introduce a novel gradient matching approach that selects demonstrations by aligning fine-tuning gradients between the entire training set of the target task and the selected examples, so as to approach the learning effect on the entire training set within the selected examples. Through gradient matching on relatively small models, e.g., Qwen2.5-3B or Llama3-8B, our method consistently outperforms random selection on larger LLMs from 4-shot to 128-shot scenarios across 9 diverse datasets. For instance, it surpasses random selection by 4% on Qwen2.5-72B and Llama3-70B, and by around 2% on 5 closed-source LLMs. This work unlocks more reliable and effective many-shot ICL, paving the way for its broader application. 

---
# Are LLMs Reliable Translators of Logical Reasoning Across Lexically Diversified Contexts? 

**Authors**: Qingchuan Li, Jiatong Li, Zirui Liu, Mingyue Cheng, Yuting Zeng, Qi Liu, Tongxuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04575)  

**Abstract**: Neuro-symbolic approaches combining large language models (LLMs) with solvers excels in logical reasoning problems need long reasoning chains. In this paradigm, LLMs serve as translators, converting natural language reasoning problems into formal logic formulas. Then reliable symbolic solvers return correct solutions. Despite their success, we find that LLMs, as translators, struggle to handle lexical diversification, a common linguistic phenomenon, indicating that LLMs as logic translators are unreliable in real-world scenarios. Moreover, existing logical reasoning benchmarks lack lexical diversity, failing to challenge LLMs' ability to translate such text and thus obscuring this issue. In this work, we propose SCALe, a benchmark designed to address this significant gap through **logic-invariant lexical diversification**. By using LLMs to transform original benchmark datasets into lexically diversified but logically equivalent versions, we evaluate LLMs' ability to consistently map diverse expressions to uniform logical symbols on these new datasets. Experiments using SCALe further confirm that current LLMs exhibit deficiencies in this capability. Building directly on the deficiencies identified through our benchmark, we propose a new method, MenTaL, to address this limitation. This method guides LLMs to first construct a table unifying diverse expressions before performing translation. Applying MenTaL through in-context learning and supervised fine-tuning (SFT) significantly improves the performance of LLM translators on lexically diversified text. Our code is now available at this https URL. 

---
# Reasoning or Overthinking: Evaluating Large Language Models on Financial Sentiment Analysis 

**Authors**: Dimitris Vamvourellis, Dhagash Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2506.04574)  

**Abstract**: We investigate the effectiveness of large language models (LLMs), including reasoning-based and non-reasoning models, in performing zero-shot financial sentiment analysis. Using the Financial PhraseBank dataset annotated by domain experts, we evaluate how various LLMs and prompting strategies align with human-labeled sentiment in a financial context. We compare three proprietary LLMs (GPT-4o, GPT-4.1, o3-mini) under different prompting paradigms that simulate System 1 (fast and intuitive) or System 2 (slow and deliberate) thinking and benchmark them against two smaller models (FinBERT-Prosus, FinBERT-Tone) fine-tuned on financial sentiment analysis. Our findings suggest that reasoning, either through prompting or inherent model design, does not improve performance on this task. Surprisingly, the most accurate and human-aligned combination of model and method was GPT-4o without any Chain-of-Thought (CoT) prompting. We further explore how performance is impacted by linguistic complexity and annotation agreement levels, uncovering that reasoning may introduce overthinking, leading to suboptimal predictions. This suggests that for financial sentiment classification, fast, intuitive "System 1"-like thinking aligns more closely with human judgment compared to "System 2"-style slower, deliberative reasoning simulated by reasoning models or CoT prompting. Our results challenge the default assumption that more reasoning always leads to better LLM decisions, particularly in high-stakes financial applications. 

---
# Demonstrations of Integrity Attacks in Multi-Agent Systems 

**Authors**: Can Zheng, Yuhan Cao, Xiaoning Dong, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2506.04572)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding, code generation, and complex planning. Simultaneously, Multi-Agent Systems (MAS) have garnered attention for their potential to enable cooperation among distributed agents. However, from a multi-party perspective, MAS could be vulnerable to malicious agents that exploit the system to serve self-interests without disrupting its core functionality. This work explores integrity attacks where malicious agents employ subtle prompt manipulation to bias MAS operations and gain various benefits. Four types of attacks are examined: \textit{Scapegoater}, who misleads the system monitor to underestimate other agents' contributions; \textit{Boaster}, who misleads the system monitor to overestimate their own performance; \textit{Self-Dealer}, who manipulates other agents to adopt certain tools; and \textit{Free-Rider}, who hands off its own task to others. We demonstrate that strategically crafted prompts can introduce systematic biases in MAS behavior and executable instructions, enabling malicious agents to effectively mislead evaluation systems and manipulate collaborative agents. Furthermore, our attacks can bypass advanced LLM-based monitors, such as GPT-4o-mini and o3-mini, highlighting the limitations of current detection mechanisms. Our findings underscore the critical need for MAS architectures with robust security protocols and content validation mechanisms, alongside monitoring systems capable of comprehensive risk scenario assessment. 

---
# SSA-COMET: Do LLMs Outperform Learned Metrics in Evaluating MT for Under-Resourced African Languages? 

**Authors**: Senyu Li, Jiayi Wang, Felermino D. M. A. Ali, Colin Cherry, Daniel Deutsch, Eleftheria Briakou, Rui Sousa-Silva, Henrique Lopes Cardoso, Pontus Stenetorp, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2506.04557)  

**Abstract**: Evaluating machine translation (MT) quality for under-resourced African languages remains a significant challenge, as existing metrics often suffer from limited language coverage and poor performance in low-resource settings. While recent efforts, such as AfriCOMET, have addressed some of the issues, they are still constrained by small evaluation sets, a lack of publicly available training data tailored to African languages, and inconsistent performance in extremely low-resource scenarios. In this work, we introduce SSA-MTE, a large-scale human-annotated MT evaluation (MTE) dataset covering 13 African language pairs from the News domain, with over 63,000 sentence-level annotations from a diverse set of MT systems. Based on this data, we develop SSA-COMET and SSA-COMET-QE, improved reference-based and reference-free evaluation metrics. We also benchmark prompting-based approaches using state-of-the-art LLMs like GPT-4o and Claude. Our experimental results show that SSA-COMET models significantly outperform AfriCOMET and are competitive with the strongest LLM (Gemini 2.5 Pro) evaluated in our study, particularly on low-resource languages such as Twi, Luo, and Yoruba. All resources are released under open licenses to support future research. 

---
# BSBench: will your LLM find the largest prime number? 

**Authors**: K. O. T. Erziev  

**Link**: [PDF](https://arxiv.org/pdf/2506.04535)  

**Abstract**: We propose that benchmarking LLMs on questions which have no reasonable answer actually isn't as silly as it sounds. We also present a benchmark that allows such testing and a method to modify the existing datasets, and discover that existing models demonstrate a performance far from the perfect on such questions. Our code and data artifacts are available at this https URL 

---
# Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs 

**Authors**: William Sheffield, Kanishka Misra, Valentina Pyatkin, Ashwini Deo, Kyle Mahowald, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04534)  

**Abstract**: Discourse particles are crucial elements that subtly shape the meaning of text. These words, often polyfunctional, give rise to nuanced and often quite disparate semantic/discourse effects, as exemplified by the diverse uses of the particle "just" (e.g., exclusive, temporal, emphatic). This work investigates the capacity of LLMs to distinguish the fine-grained senses of English "just", a well-studied example in formal semantics, using data meticulously created and labeled by expert linguists. Our findings reveal that while LLMs exhibit some ability to differentiate between broader categories, they struggle to fully capture more subtle nuances, highlighting a gap in their understanding of discourse particles. 

---
# Please Translate Again: Two Simple Experiments on Whether Human-Like Reasoning Helps Translation 

**Authors**: Di Wu, Seth Aycock, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2506.04521)  

**Abstract**: Large Language Models (LLMs) demonstrate strong reasoning capabilities for many tasks, often by explicitly decomposing the task via Chain-of-Thought (CoT) reasoning. Recent work on LLM-based translation designs hand-crafted prompts to decompose translation, or trains models to incorporate intermediate steps.~\textit{Translating Step-by-step}~\citep{briakou2024translating}, for instance, introduces a multi-step prompt with decomposition and refinement of translation with LLMs, which achieved state-of-the-art results on WMT24. In this work, we scrutinise this strategy's effectiveness. Empirically, we find no clear evidence that performance gains stem from explicitly decomposing the translation process, at least for the models on test; and we show that simply prompting LLMs to ``translate again'' yields even better results than human-like step-by-step prompting. Our analysis does not rule out the role of reasoning, but instead invites future work exploring the factors for CoT's effectiveness in the context of translation. 

---
# DRE: An Effective Dual-Refined Method for Integrating Small and Large Language Models in Open-Domain Dialogue Evaluation 

**Authors**: Kun Zhao, Bohao Yang, Chen Tang, Siyuan Dai, Haoteng Tang, Chenghua Lin, Liang Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.04516)  

**Abstract**: Large Language Models (LLMs) excel at many tasks but struggle with ambiguous scenarios where multiple valid responses exist, often yielding unreliable results. Conversely, Small Language Models (SLMs) demonstrate robustness in such scenarios but are susceptible to misleading or adversarial inputs. We observed that LLMs handle negative examples effectively, while SLMs excel with positive examples. To leverage their complementary strengths, we introduce SLIDE (Small and Large Integrated for Dialogue Evaluation), a method integrating SLMs and LLMs via adaptive weighting. Building on SLIDE, we further propose a Dual-Refinement Evaluation (DRE) method to enhance SLM-LLM integration: (1) SLM-generated insights guide the LLM to produce initial evaluations; (2) SLM-derived adjustments refine the LLM's scores for improved accuracy. Experiments demonstrate that DRE outperforms existing methods, showing stronger alignment with human judgment across diverse benchmarks. This work illustrates how combining small and large models can yield more reliable evaluation tools, particularly for open-ended tasks such as dialogue evaluation. 

---
# SQLens: An End-to-End Framework for Error Detection and Correction in Text-to-SQL 

**Authors**: Yue Gong, Chuan Lei, Xiao Qin, Kapil Vaidya, Balakrishnan Narayanaswamy, Tim Kraska  

**Link**: [PDF](https://arxiv.org/pdf/2506.04494)  

**Abstract**: Text-to-SQL systems translate natural language (NL) questions into SQL queries, enabling non-technical users to interact with structured data. While large language models (LLMs) have shown promising results on the text-to-SQL task, they often produce semantically incorrect yet syntactically valid queries, with limited insight into their reliability. We propose SQLens, an end-to-end framework for fine-grained detection and correction of semantic errors in LLM-generated SQL. SQLens integrates error signals from both the underlying database and the LLM to identify potential semantic errors within SQL clauses. It further leverages these signals to guide query correction. Empirical results on two public benchmarks show that SQLens outperforms the best LLM-based self-evaluation method by 25.78% in F1 for error detection, and improves execution accuracy of out-of-the-box text-to-SQL systems by up to 20%. 

---
# Aligning Large Language Models with Implicit Preferences from User-Generated Content 

**Authors**: Zhaoxuan Tan, Zheng Li, Tianyi Liu, Haodong Wang, Hyokun Yun, Ming Zeng, Pei Chen, Zhihan Zhang, Yifan Gao, Ruijie Wang, Priyanka Nigam, Bing Yin, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04463)  

**Abstract**: Learning from preference feedback is essential for aligning large language models (LLMs) with human values and improving the quality of generated responses. However, existing preference learning methods rely heavily on curated data from humans or advanced LLMs, which is costly and difficult to scale. In this work, we present PUGC, a novel framework that leverages implicit human Preferences in unlabeled User-Generated Content (UGC) to generate preference data. Although UGC is not explicitly created to guide LLMs in generating human-preferred responses, it often reflects valuable insights and implicit preferences from its creators that has the potential to address readers' questions. PUGC transforms UGC into user queries and generates responses from the policy model. The UGC is then leveraged as a reference text for response scoring, aligning the model with these implicit preferences. This approach improves the quality of preference data while enabling scalable, domain-specific alignment. Experimental results on Alpaca Eval 2 show that models trained with DPO and PUGC achieve a 9.37% performance improvement over traditional methods, setting a 35.93% state-of-the-art length-controlled win rate using Mistral-7B-Instruct. Further studies highlight gains in reward quality, domain-specific alignment effectiveness, robustness against UGC quality, and theory of mind capabilities. Our code and dataset are available at this https URL 

---
# Watermarking Degrades Alignment in Language Models: Analysis and Mitigation 

**Authors**: Apurv Verma, NhatHai Phan, Shubhendu Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2506.04462)  

**Abstract**: Watermarking techniques for large language models (LLMs) can significantly impact output quality, yet their effects on truthfulness, safety, and helpfulness remain critically underexamined. This paper presents a systematic analysis of how two popular watermarking approaches-Gumbel and KGW-affect these core alignment properties across four aligned LLMs. Our experiments reveal two distinct degradation patterns: guard attenuation, where enhanced helpfulness undermines model safety, and guard amplification, where excessive caution reduces model helpfulness. These patterns emerge from watermark-induced shifts in token distribution, surfacing the fundamental tension that exists between alignment objectives.
To mitigate these degradations, we propose Alignment Resampling (AR), an inference-time sampling method that uses an external reward model to restore alignment. We establish a theoretical lower bound on the improvement in expected reward score as the sample size is increased and empirically demonstrate that sampling just 2-4 watermarked generations effectively recovers or surpasses baseline (unwatermarked) alignment scores. To overcome the limited response diversity of standard Gumbel watermarking, our modified implementation sacrifices strict distortion-freeness while maintaining robust detectability, ensuring compatibility with AR. Experimental results confirm that AR successfully recovers baseline alignment in both watermarking approaches, while maintaining strong watermark detectability. This work reveals the critical balance between watermark strength and model alignment, providing a simple inference-time solution to responsibly deploy watermarked LLMs in practice. 

---
# Zero-Shot Open-Schema Entity Structure Discovery 

**Authors**: Xueqiang Xu, Jinfeng Xiao, James Barry, Mohab Elkaref, Jiaru Zou, Pengcheng Jiang, Yunyi Zhang, Max Giammona, Geeth de Mel, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.04458)  

**Abstract**: Entity structure extraction, which aims to extract entities and their associated attribute-value structures from text, is an essential task for text understanding and knowledge graph construction. Existing methods based on large language models (LLMs) typically rely heavily on predefined entity attribute schemas or annotated datasets, often leading to incomplete extraction results. To address these challenges, we introduce Zero-Shot Open-schema Entity Structure Discovery (ZOES), a novel approach to entity structure extraction that does not require any schema or annotated samples. ZOES operates via a principled mechanism of enrichment, refinement, and unification, based on the insight that an entity and its associated structure are mutually reinforcing. Experiments demonstrate that ZOES consistently enhances LLMs' ability to extract more complete entity structures across three different domains, showcasing both the effectiveness and generalizability of the method. These findings suggest that such an enrichment, refinement, and unification mechanism may serve as a principled approach to improving the quality of LLM-based entity structure discovery in various scenarios. 

---
# Empaths at SemEval-2025 Task 11: Retrieval-Augmented Approach to Perceived Emotions Prediction 

**Authors**: Lev Morozov, Aleksandr Mogilevskii, Alexander Shirnin  

**Link**: [PDF](https://arxiv.org/pdf/2506.04409)  

**Abstract**: This paper describes EmoRAG, a system designed to detect perceived emotions in text for SemEval-2025 Task 11, Subtask A: Multi-label Emotion Detection. We focus on predicting the perceived emotions of the speaker from a given text snippet, labeling it with emotions such as joy, sadness, fear, anger, surprise, and disgust. Our approach does not require additional model training and only uses an ensemble of models to predict emotions. EmoRAG achieves results comparable to the best performing systems, while being more efficient, scalable, and easier to implement. 

---
# Unpacking Let Alone: Human-Scale Models Generalize to a Rare Construction in Form but not Meaning 

**Authors**: Wesley Scivetti, Tatsuya Aoyama, Ethan Wilcox, Nathan Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2506.04408)  

**Abstract**: Humans have a remarkable ability to acquire and understand grammatical phenomena that are seen rarely, if ever, during childhood. Recent evidence suggests that language models with human-scale pretraining data may possess a similar ability by generalizing from frequent to rare constructions. However, it remains an open question how widespread this generalization ability is, and to what extent this knowledge extends to meanings of rare constructions, as opposed to just their forms. We fill this gap by testing human-scale transformer language models on their knowledge of both the form and meaning of the (rare and quirky) English LET-ALONE construction. To evaluate our LMs we construct a bespoke synthetic benchmark that targets syntactic and semantic properties of the construction. We find that human-scale LMs are sensitive to form, even when related constructions are filtered from the dataset. However, human-scale LMs do not make correct generalizations about LET-ALONE's meaning. These results point to an asymmetry in the current architectures' sample efficiency between language form and meaning, something which is not present in human language learners. 

---
# MedAgentGym: Training LLM Agents for Code-Based Medical Reasoning at Scale 

**Authors**: Ran Xu, Yuchen Zhuang, Yishan Zhong, Yue Yu, Xiangru Tang, Hang Wu, May D. Wang, Peifeng Ruan, Donghan Yang, Tao Wang, Guanghua Xiao, Carl Yang, Yang Xie, Wenqi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.04405)  

**Abstract**: We introduce MedAgentGYM, the first publicly available training environment designed to enhance coding-based medical reasoning capabilities in large language model (LLM) agents. MedAgentGYM comprises 72,413 task instances across 129 categories derived from authentic real-world biomedical scenarios. Tasks are encapsulated within executable coding environments, each featuring detailed task descriptions, interactive feedback mechanisms, verifiable ground-truth annotations, and scalable training trajectory generation. Extensive benchmarking of over 30 LLMs reveals a notable performance disparity between commercial API-based models and open-source counterparts. Leveraging MedAgentGYM, Med-Copilot-7B achieves substantial performance gains through supervised fine-tuning (+36.44%) and continued reinforcement learning (+42.47%), emerging as an affordable and privacy-preserving alternative competitive with gpt-4o. By offering both a comprehensive benchmark and accessible, expandable training resources within unified execution environments, MedAgentGYM delivers an integrated platform to develop LLM-based coding assistants for advanced biomedical research and practice. 

---
# Building a Few-Shot Cross-Domain Multilingual NLU Model for Customer Care 

**Authors**: Saurabh Kumar, Sourav Bansal, Neeraj Agrawal, Priyanka Bhatt  

**Link**: [PDF](https://arxiv.org/pdf/2506.04389)  

**Abstract**: Customer care is an essential pillar of the e-commerce shopping experience with companies spending millions of dollars each year, employing automation and human agents, across geographies (like US, Canada, Mexico, Chile), channels (like Chat, Interactive Voice Response (IVR)), and languages (like English, Spanish). SOTA pre-trained models like multilingual-BERT, fine-tuned on annotated data have shown good performance in downstream tasks relevant to Customer Care. However, model performance is largely subject to the availability of sufficient annotated domain-specific data. Cross-domain availability of data remains a bottleneck, thus building an intent classifier that generalizes across domains (defined by channel, geography, and language) with only a few annotations, is of great practical value. In this paper, we propose an embedder-cum-classifier model architecture which extends state-of-the-art domain-specific models to other domains with only a few labeled samples. We adopt a supervised fine-tuning approach with isotropic regularizers to train a domain-specific sentence embedder and a multilingual knowledge distillation strategy to generalize this embedder across multiple domains. The trained embedder, further augmented with a simple linear classifier can be deployed for new domains. Experiments on Canada and Mexico e-commerce Customer Care dataset with few-shot intent detection show an increase in accuracy by 20-23% against the existing state-of-the-art pre-trained models. 

---
# MELABenchv1: Benchmarking Large Language Models against Smaller Fine-Tuned Models for Low-Resource Maltese NLP 

**Authors**: Kurt Micallef, Claudia Borg  

**Link**: [PDF](https://arxiv.org/pdf/2506.04385)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various Natural Language Processing (NLP) tasks, largely due to their generalisability and ability to perform tasks without additional training. However, their effectiveness for low-resource languages remains limited. In this study, we evaluate the performance of 55 publicly available LLMs on Maltese, a low-resource language, using a newly introduced benchmark covering 11 discriminative and generative tasks. Our experiments highlight that many models perform poorly, particularly on generative tasks, and that smaller fine-tuned models often perform better across all tasks. From our multidimensional analysis, we investigate various factors impacting performance. We conclude that prior exposure to Maltese during pre-training and instruction-tuning emerges as the most important factor. We also examine the trade-offs between fine-tuning and prompting, highlighting that while fine-tuning requires a higher initial cost, it yields better performance and lower inference costs. Through this work, we aim to highlight the need for more inclusive language technologies and recommend that researchers working with low-resource languages consider more "traditional" language modelling approaches. 

---
# Hierarchical Text Classification Using Contrastive Learning Informed Path Guided Hierarchy 

**Authors**: Neeraj Agrawal, Saurabh Kumar, Priyanka Bhatt, Tanishka Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.04381)  

**Abstract**: Hierarchical Text Classification (HTC) has recently gained traction given the ability to handle complex label hierarchy. This has found applications in domains like E- commerce, customer care and medicine industry among other real-world applications. Existing HTC models either encode label hierarchy separately and mix it with text encoding or guide the label hierarchy structure in the text encoder. Both approaches capture different characteristics of label hierarchy and are complementary to each other. In this paper, we propose a Hierarchical Text Classification using Contrastive Learning Informed Path guided hierarchy (HTC-CLIP), which learns hierarchy-aware text representation and text informed path guided hierarchy representation using contrastive learning. During the training of HTC-CLIP, we learn two different sets of class probabilities distributions and during inference, we use the pooled output of both probabilities for each class to get the best of both representations. Our results show that the two previous approaches can be effectively combined into one architecture to achieve improved performance. Tests on two public benchmark datasets showed an improvement of 0.99 - 2.37% in Macro F1 score using HTC-CLIP over the existing state-of-the-art models. 

---
# Mechanistic Decomposition of Sentence Representations 

**Authors**: Matthieu Tehenan, Vikram Natarajan, Jonathan Michala, Milton Lin, Juri Opitz  

**Link**: [PDF](https://arxiv.org/pdf/2506.04373)  

**Abstract**: Sentence embeddings are central to modern NLP and AI systems, yet little is known about their internal structure. While we can compare these embeddings using measures such as cosine similarity, the contributing features are not human-interpretable, and the content of an embedding seems untraceable, as it is masked by complex neural transformations and a final pooling operation that combines individual token embeddings. To alleviate this issue, we propose a new method to mechanistically decompose sentence embeddings into interpretable components, by using dictionary learning on token-level representations. We analyze how pooling compresses these features into sentence representations, and assess the latent features that reside in a sentence embedding. This bridges token-level mechanistic interpretability with sentence-level analysis, making for more transparent and controllable representations. In our studies, we obtain several interesting insights into the inner workings of sentence embedding spaces, for instance, that many semantic and syntactic aspects are linearly encoded in the embeddings. 

---
# Effects of Speaker Count, Duration, and Accent Diversity on Zero-Shot Accent Robustness in Low-Resource ASR 

**Authors**: Zheng-Xin Yong, Vineel Pratap, Michael Auli, Jean Maillard  

**Link**: [PDF](https://arxiv.org/pdf/2506.04364)  

**Abstract**: To build an automatic speech recognition (ASR) system that can serve everyone in the world, the ASR needs to be robust to a wide range of accents including unseen accents. We systematically study how three different variables in training data -- the number of speakers, the audio duration per each individual speaker, and the diversity of accents -- affect ASR robustness towards unseen accents in a low-resource training regime. We observe that for a fixed number of ASR training hours, it is more beneficial to increase the number of speakers (which means each speaker contributes less) than the number of hours contributed per speaker. We also observe that more speakers enables ASR performance gains from scaling number of hours. Surprisingly, we observe minimal benefits to prioritizing speakers with different accents when the number of speakers is controlled. Our work suggests that practitioners should prioritize increasing the speaker count in ASR training data composition for new languages. 

---
# GEM: Empowering LLM for both Embedding Generation and Language Understanding 

**Authors**: Caojin Zhang, Qiang Zhang, Ke Li, Sai Vidyaranya Nuthalapati, Benyu Zhang, Jason Liu, Serena Li, Lizhu Zhang, Xiangjun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.04344)  

**Abstract**: Large decoder-only language models (LLMs) have achieved remarkable success in generation and reasoning tasks, where they generate text responses given instructions. However, many applications, e.g., retrieval augmented generation (RAG), still rely on separate embedding models to generate text embeddings, which can complicate the system and introduce discrepancies in understanding of the query between the embedding model and LLMs. To address this limitation, we propose a simple self-supervised approach, Generative Embedding large language Model (GEM), that enables any large decoder-only LLM to generate high-quality text embeddings while maintaining its original text generation and reasoning capabilities. Our method inserts new special token(s) into a text body, and generates summarization embedding of the text by manipulating the attention mask. This method could be easily integrated into post-training or fine tuning stages of any existing LLMs. We demonstrate the effectiveness of our approach by applying it to two popular LLM families, ranging from 1B to 8B parameters, and evaluating the transformed models on both text embedding benchmarks (MTEB) and NLP benchmarks (MMLU). The results show that our proposed method significantly improves the original LLMs on MTEB while having a minimal impact on MMLU. Our strong results indicate that our approach can empower LLMs with state-of-the-art text embedding capabilities while maintaining their original NLP performance 

---
# Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets 

**Authors**: Lei Hsiung, Tianyu Pang, Yung-Chen Tang, Linyue Song, Tsung-Yi Ho, Pin-Yu Chen, Yaoqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05346)  

**Abstract**: Recent advancements in large language models (LLMs) have underscored their vulnerability to safety alignment jailbreaks, particularly when subjected to downstream fine-tuning. However, existing mitigation strategies primarily focus on reactively addressing jailbreak incidents after safety guardrails have been compromised, removing harmful gradients during fine-tuning, or continuously reinforcing safety alignment throughout fine-tuning. As such, they tend to overlook a critical upstream factor: the role of the original safety-alignment data. This paper therefore investigates the degradation of safety guardrails through the lens of representation similarity between upstream alignment datasets and downstream fine-tuning tasks. Our experiments demonstrate that high similarity between these datasets significantly weakens safety guardrails, making models more susceptible to jailbreaks. Conversely, low similarity between these two types of datasets yields substantially more robust models and thus reduces harmfulness score by up to 10.33%. By highlighting the importance of upstream dataset design in the building of durable safety guardrails and reducing real-world vulnerability to jailbreak attacks, these findings offer actionable insights for fine-tuning service providers. 

---
# Inference-Time Hyper-Scaling with KV Cache Compression 

**Authors**: Adrian Łańcucki, Konrad Staniszewski, Piotr Nawrot, Edoardo M. Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2506.05345)  

**Abstract**: Inference-time scaling trades efficiency for increased reasoning accuracy by generating longer or more parallel sequences. However, in Transformer LLMs, generation cost is bottlenecked by the size of the key-value (KV) cache, rather than the number of generated tokens. Hence, we explore inference-time hyper-scaling: by compressing the KV cache, we can generate more tokens within the same compute budget and further improve the accuracy of scaled inference. The success of this approach, however, hinges on the ability of compression methods to preserve accuracy even at high compression ratios. To make hyper-scaling practical, we introduce Dynamic Memory Sparsification (DMS), a novel method for sparsifying KV caches that only requires 1K training steps to achieve 8$\times$ compression, while maintaining better accuracy than training-free sparse attention. Instead of prematurely discarding cached tokens, DMS delays token eviction, implicitly merging representations and preserving critical information. We demonstrate the effectiveness of inference-time hyper-scaling with DMS on multiple families of LLMs, showing that it boosts accuracy for comparable inference runtime and memory load. For instance, we enhance Qwen-R1 32B by an average of 9.1 points on AIME 24, 7.6 on GPQA, and 9.6 on LiveCodeBench across compute budgets. 

---
# Kinetics: Rethinking Test-Time Scaling Laws 

**Authors**: Ranajoy Sadhukhan, Zhuoming Chen, Haizhong Zheng, Yang Zhou, Emma Strubell, Beidi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05333)  

**Abstract**: We rethink test-time scaling laws from a practical efficiency perspective, revealing that the effectiveness of smaller models is significantly overestimated. Prior work, grounded in compute-optimality, overlooks critical memory access bottlenecks introduced by inference-time strategies (e.g., Best-of-$N$, long CoTs). Our holistic analysis, spanning models from 0.6B to 32B parameters, reveals a new Kinetics Scaling Law that better guides resource allocation by incorporating both computation and memory access costs. Kinetics Scaling Law suggests that test-time compute is more effective when used on models above a threshold than smaller ones. A key reason is that in TTS, attention, rather than parameter count, emerges as the dominant cost factor. Motivated by this, we propose a new scaling paradigm centered on sparse attention, which lowers per-token cost and enables longer generations and more parallel samples within the same resource budget. Empirically, we show that sparse attention models consistently outperform dense counterparts, achieving over 60 points gains in low-cost regimes and over 5 points gains in high-cost regimes for problem-solving accuracy on AIME, encompassing evaluations on state-of-the-art MoEs. These results suggest that sparse attention is essential for realizing the full potential of test-time scaling because, unlike training, where parameter scaling saturates, test-time accuracy continues to improve through increased generation. The code is available at this https URL. 

---
# Unleashing Hour-Scale Video Training for Long Video-Language Understanding 

**Authors**: Jingyang Lin, Jialian Wu, Ximeng Sun, Ze Wang, Jiang Liu, Yusheng Su, Xiaodong Yu, Hao Chen, Jiebo Luo, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2506.05332)  

**Abstract**: Recent long-form video-language understanding benchmarks have driven progress in video large multimodal models (Video-LMMs). However, the scarcity of well-annotated long videos has left the training of hour-long Video-LLMs underexplored. To close this gap, we present VideoMarathon, a large-scale hour-long video instruction-following dataset. This dataset includes around 9,700 hours of long videos sourced from diverse domains, ranging from 3 to 60 minutes per video. Specifically, it contains 3.3M high-quality QA pairs, spanning six fundamental topics: temporality, spatiality, object, action, scene, and event. Compared to existing video instruction datasets, VideoMarathon significantly extends training video durations up to 1 hour, and supports 22 diverse tasks requiring both short- and long-term video comprehension. Building on VideoMarathon, we propose Hour-LLaVA, a powerful and efficient Video-LMM for hour-scale video-language modeling. It enables hour-long video training and inference at 1-FPS sampling by leveraging a memory augmentation module, which adaptively integrates user question-relevant and spatiotemporal-informative semantics from a cached full video context. In our experiments, Hour-LLaVA achieves the best performance on multiple long video-language benchmarks, demonstrating the high quality of the VideoMarathon dataset and the superiority of the Hour-LLaVA model. 

---
# Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay 

**Authors**: Yifan Sun, Jingyan Shen, Yibin Wang, Tianyu Chen, Zhendong Wang, Mingyuan Zhou, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05316)  

**Abstract**: Reinforcement learning (RL) has become an effective approach for fine-tuning large language models (LLMs), particularly to enhance their reasoning capabilities. However, RL fine-tuning remains highly resource-intensive, and existing work has largely overlooked the problem of data efficiency. In this paper, we propose two techniques to improve data efficiency in LLM RL fine-tuning: difficulty-targeted online data selection and rollout replay. We introduce the notion of adaptive difficulty to guide online data selection, prioritizing questions of moderate difficulty that are more likely to yield informative learning signals. To estimate adaptive difficulty efficiently, we develop an attention-based framework that requires rollouts for only a small reference set of questions. The adaptive difficulty of the remaining questions is then estimated based on their similarity to this set. To further reduce rollout cost, we introduce a rollout replay mechanism that reuses recent rollouts, lowering per-step computation while maintaining stable updates. Extensive experiments across 6 LLM-dataset combinations show that our method reduces RL fine-tuning time by 25% to 65% to reach the same level of performance as the original GRPO algorithm. 

---
# Time to Talk: LLM Agents for Asynchronous Group Communication in Mafia Games 

**Authors**: Niv Eckhaus, Uri Berger, Gabriel Stanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2506.05309)  

**Abstract**: LLMs are used predominantly in synchronous communication, where a human user and a model communicate in alternating turns. In contrast, many real-world settings are inherently asynchronous. For example, in group chats, online team meetings, or social games, there is no inherent notion of turns; therefore, the decision of when to speak forms a crucial part of the participant's decision making. In this work, we develop an adaptive asynchronous LLM-agent which, in addition to determining what to say, also decides when to say it. To evaluate our agent, we collect a unique dataset of online Mafia games, including both human participants, as well as our asynchronous agent. Overall, our agent performs on par with human players, both in game performance, as well as in its ability to blend in with the other human players. Our analysis shows that the agent's behavior in deciding when to speak closely mirrors human patterns, although differences emerge in message content. We release all our data and code to support and encourage further research for more realistic asynchronous communication between LLM agents. This work paves the way for integration of LLMs into realistic human group settings, from assistance in team discussions to educational and professional environments where complex social dynamics must be navigated. 

---
# MesaNet: Sequence Modeling by Locally Optimal Test-Time Training 

**Authors**: Johannes von Oswald, Nino Scherrer, Seijin Kobayashi, Luca Versari, Songlin Yang, Maximilian Schlegel, Kaitlin Maile, Yanick Schimpf, Oliver Sieberling, Alexander Meulemans, Rif A. Saurous, Guillaume Lajoie, Charlotte Frenkel, Razvan Pascanu, Blaise Agüera y Arcas, João Sacramento  

**Link**: [PDF](https://arxiv.org/pdf/2506.05233)  

**Abstract**: Sequence modeling is currently dominated by causal transformer architectures that use softmax self-attention. Although widely adopted, transformers require scaling memory and compute linearly during inference. A recent stream of work linearized the softmax operation, resulting in powerful recurrent neural network (RNN) models with constant memory and compute costs such as DeltaNet, Mamba or xLSTM. These models can be unified by noting that their recurrent layer dynamics can all be derived from an in-context regression objective, approximately optimized through an online learning rule. Here, we join this line of work and introduce a numerically stable, chunkwise parallelizable version of the recently proposed Mesa layer (von Oswald et al., 2024), and study it in language modeling at the billion-parameter scale. This layer again stems from an in-context loss, but which is now minimized to optimality at every time point using a fast conjugate gradient solver. Through an extensive suite of experiments, we show that optimal test-time training enables reaching lower language modeling perplexity and higher downstream benchmark performance than previous RNNs, especially on tasks requiring long context understanding. This performance gain comes at the cost of additional flops spent during inference time. Our results are therefore intriguingly related to recent trends of increasing test-time compute to improve performance -- here by spending compute to solve sequential optimization problems within the neural network itself. 

---
# Diagonal Batching Unlocks Parallelism in Recurrent Memory Transformers for Long Contexts 

**Authors**: Danil Sivtsov, Ivan Rodkin, Gleb Kuzmin, Yuri Kuratov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2506.05229)  

**Abstract**: Transformer models struggle with long-context inference due to their quadratic time and linear memory complexity. Recurrent Memory Transformers (RMTs) offer a solution by reducing the asymptotic cost to linear time and constant memory usage. However, their memory update mechanism leads to sequential execution, causing a performance bottleneck.
We introduce Diagonal Batching, a scheduling scheme that unlocks parallelism across segments in RMTs while preserving exact recurrence. This approach eliminates the sequential constraint, enabling efficient GPU inference even for single long-context inputs without complex batching and pipelining techniques. Because the technique is purely a run-time computation reordering, existing RMT models adopt it with no retraining.
Applied to a LLaMA-1B ARMT model, Diagonal Batching yields a 3.3x speedup over standard full-attention LLaMA-1B and a 1.8x speedup over the sequential RMT implementation on 131,072-token sequences. By removing sequential bottleneck, Diagonal Batching reduces inference cost and latency, thereby strengthening RMTs as a practical solution for real-world, long-context applications. 

---
# Mitigating Degree Bias Adaptively with Hard-to-Learn Nodes in Graph Contrastive Learning 

**Authors**: Jingyu Hu, Hongbo Bo, Jun Hong, Xiaowei Liu, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05214)  

**Abstract**: Graph Neural Networks (GNNs) often suffer from degree bias in node classification tasks, where prediction performance varies across nodes with different degrees. Several approaches, which adopt Graph Contrastive Learning (GCL), have been proposed to mitigate this bias. However, the limited number of positive pairs and the equal weighting of all positives and negatives in GCL still lead to low-degree nodes acquiring insufficient and noisy information. This paper proposes the Hardness Adaptive Reweighted (HAR) contrastive loss to mitigate degree bias. It adds more positive pairs by leveraging node labels and adaptively weights positive and negative pairs based on their learning hardness. In addition, we develop an experimental framework named SHARP to extend HAR to a broader range of scenarios. Both our theoretical analysis and experiments validate the effectiveness of SHARP. The experimental results across four datasets show that SHARP achieves better performance against baselines at both global and degree levels. 

---
# LLM-First Search: Self-Guided Exploration of the Solution Space 

**Authors**: Nathan Herr, Tim Rocktäschel, Roberta Raileanu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05213)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable improvements in reasoning and planning through increased test-time compute, often by framing problem-solving as a search process. While methods like Monte Carlo Tree Search (MCTS) have proven effective in some domains, their reliance on fixed exploration hyperparameters limits their adaptability across tasks of varying difficulty, rendering them impractical or expensive in certain settings. In this paper, we propose \textbf{LLM-First Search (LFS)}, a novel \textit{LLM Self-Guided Search} method that removes the need for pre-defined search strategies by empowering the LLM to autonomously control the search process via self-guided exploration. Rather than relying on external heuristics or hardcoded policies, the LLM evaluates whether to pursue the current search path or explore alternative branches based on its internal scoring mechanisms. This enables more flexible and context-sensitive reasoning without requiring manual tuning or task-specific adaptation. We evaluate LFS on Countdown and Sudoku against three classic widely-used search algorithms, Tree-of-Thoughts' Breadth First Search (ToT-BFS), Best First Search (BestFS), and MCTS, each of which have been used to achieve SotA results on a range of challenging reasoning tasks. We found that LFS (1) performs better on more challenging tasks without additional tuning, (2) is more computationally efficient compared to the other methods, especially when powered by a stronger model, (3) scales better with stronger models, due to its LLM-First design, and (4) scales better with increased compute budget. Our code is publicly available at \href{this https URL}{LLM-First-Search}. 

---
# CIVET: Systematic Evaluation of Understanding in VLMs 

**Authors**: Massimo Rizzoli, Simone Alghisi, Olha Khomyn, Gabriel Roccabruna, Seyed Mahed Mousavi, Giuseppe Riccardi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05146)  

**Abstract**: While Vision-Language Models (VLMs) have achieved competitive performance in various tasks, their comprehension of the underlying structure and semantics of a scene remains understudied. To investigate the understanding of VLMs, we study their capability regarding object properties and relations in a controlled and interpretable manner. To this scope, we introduce CIVET, a novel and extensible framework for systematiC evaluatIon Via controllEd sTimuli. CIVET addresses the lack of standardized systematic evaluation for assessing VLMs' understanding, enabling researchers to test hypotheses with statistical rigor. With CIVET, we evaluate five state-of-the-art VLMs on exhaustive sets of stimuli, free from annotation noise, dataset-specific biases, and uncontrolled scene complexity. Our findings reveal that 1) current VLMs can accurately recognize only a limited set of basic object properties; 2) their performance heavily depends on the position of the object in the scene; 3) they struggle to understand basic relations among objects. Furthermore, a comparative evaluation with human annotators reveals that VLMs still fall short of achieving human-level accuracy. 

---
# Interpretable Multimodal Framework for Human-Centered Street Assessment: Integrating Visual-Language Models for Perceptual Urban Diagnostics 

**Authors**: HaoTian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2506.05087)  

**Abstract**: While objective street metrics derived from imagery or GIS have become standard in urban analytics, they remain insufficient to capture subjective perceptions essential to inclusive urban design. This study introduces a novel Multimodal Street Evaluation Framework (MSEF) that fuses a vision transformer (VisualGLM-6B) with a large language model (GPT-4), enabling interpretable dual-output assessment of streetscapes. Leveraging over 15,000 annotated street-view images from Harbin, China, we fine-tune the framework using LoRA and P-Tuning v2 for parameter-efficient adaptation. The model achieves an F1 score of 0.84 on objective features and 89.3 percent agreement with aggregated resident perceptions, validated across stratified socioeconomic geographies. Beyond classification accuracy, MSEF captures context-dependent contradictions: for instance, informal commerce boosts perceived vibrancy while simultaneously reducing pedestrian comfort. It also identifies nonlinear and semantically contingent patterns -- such as the divergent perceptual effects of architectural transparency across residential and commercial zones -- revealing the limits of universal spatial heuristics. By generating natural-language rationales grounded in attention mechanisms, the framework bridges sensory data with socio-affective inference, enabling transparent diagnostics aligned with SDG 11. This work offers both methodological innovation in urban perception modeling and practical utility for planning systems seeking to reconcile infrastructural precision with lived experience. 

---
# Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings 

**Authors**: Yubo Ma, Jinsong Li, Yuhang Zang, Xiaobao Wu, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Haodong Duan, Jiaqi Wang, Yixin Cao, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.04997)  

**Abstract**: Despite the strong performance of ColPali/ColQwen2 in Visualized Document Retrieval (VDR), it encodes each page into multiple patch-level embeddings and leads to excessive memory usage. This empirical study investigates methods to reduce patch embeddings per page at minimum performance degradation. We evaluate two token-reduction strategies: token pruning and token merging. Regarding token pruning, we surprisingly observe that a simple random strategy outperforms other sophisticated pruning methods, though still far from satisfactory. Further analysis reveals that pruning is inherently unsuitable for VDR as it requires removing certain page embeddings without query-specific information. Turning to token merging (more suitable for VDR), we search for the optimal combinations of merging strategy across three dimensions and develop Light-ColPali/ColQwen2. It maintains 98.2% of retrieval performance with only 11.8% of original memory usage, and preserves 94.6% effectiveness at 2.8% memory footprint. We expect our empirical findings and resulting Light-ColPali/ColQwen2 offer valuable insights and establish a competitive baseline for future research towards efficient VDR. 

---
# Dissecting Long Reasoning Models: An Empirical Study 

**Authors**: Yongyu Mu, Jiali Zeng, Bei Li, Xinyan Guan, Fandong Meng, Jie Zhou, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04913)  

**Abstract**: Despite recent progress in training long-context reasoning models via reinforcement learning (RL), several open questions and counterintuitive behaviors remain. This work focuses on three key aspects: (1) We systematically analyze the roles of positive and negative samples in RL, revealing that positive samples mainly facilitate data fitting, whereas negative samples significantly enhance generalization and robustness. Interestingly, training solely on negative samples can rival standard RL training performance. (2) We identify substantial data inefficiency in group relative policy optimization, where over half of the samples yield zero advantage. To address this, we explore two straightforward strategies, including relative length rewards and offline sample injection, to better leverage these data and enhance reasoning efficiency and capability. (3) We investigate unstable performance across various reasoning models and benchmarks, attributing instability to uncertain problems with ambiguous outcomes, and demonstrate that multiple evaluation runs mitigate this issue. 

---
# When Thinking LLMs Lie: Unveiling the Strategic Deception in Representations of Reasoning Models 

**Authors**: Kai Wang, Yihao Zhang, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.04909)  

**Abstract**: The honesty of large language models (LLMs) is a critical alignment challenge, especially as advanced systems with chain-of-thought (CoT) reasoning may strategically deceive humans. Unlike traditional honesty issues on LLMs, which could be possibly explained as some kind of hallucination, those models' explicit thought paths enable us to study strategic deception--goal-driven, intentional misinformation where reasoning contradicts outputs. Using representation engineering, we systematically induce, detect, and control such deception in CoT-enabled LLMs, extracting "deception vectors" via Linear Artificial Tomography (LAT) for 89% detection accuracy. Through activation steering, we achieve a 40% success rate in eliciting context-appropriate deception without explicit prompts, unveiling the specific honesty-related issue of reasoning models and providing tools for trustworthy AI alignment. 

---
# From EHRs to Patient Pathways: Scalable Modeling of Longitudinal Health Trajectories with LLMs 

**Authors**: Chantal Pellegrini, Ege Özsoy, David Bani-Harouni, Matthias Keicher, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2506.04831)  

**Abstract**: Healthcare systems face significant challenges in managing and interpreting vast, heterogeneous patient data for personalized care. Existing approaches often focus on narrow use cases with a limited feature space, overlooking the complex, longitudinal interactions needed for a holistic understanding of patient health. In this work, we propose a novel approach to patient pathway modeling by transforming diverse electronic health record (EHR) data into a structured representation and designing a holistic pathway prediction model, EHR2Path, optimized to predict future health trajectories. Further, we introduce a novel summary mechanism that embeds long-term temporal context into topic-specific summary tokens, improving performance over text-only models, while being much more token-efficient. EHR2Path demonstrates strong performance in both next time-step prediction and longitudinal simulation, outperforming competitive baselines. It enables detailed simulations of patient trajectories, inherently targeting diverse evaluation tasks, such as forecasting vital signs, lab test results, or length-of-stay, opening a path towards predictive and personalized healthcare. 

---
# GOLFer: Smaller LM-Generated Documents Hallucination Filter & Combiner for Query Expansion in Information Retrieval 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04762)  

**Abstract**: Large language models (LLMs)-based query expansion for information retrieval augments queries with generated hypothetical documents with LLMs. However, its performance relies heavily on the scale of the language models (LMs), necessitating larger, more advanced LLMs. This approach is costly, computationally intensive, and often has limited accessibility. To address these limitations, we introduce GOLFer - Smaller LMs-Generated Documents Hallucination Filter & Combiner - a novel method leveraging smaller open-source LMs for query expansion. GOLFer comprises two modules: a hallucination filter and a documents combiner. The former detects and removes non-factual and inconsistent sentences in generated documents, a common issue with smaller LMs, while the latter combines the filtered content with the query using a weight vector to balance their influence. We evaluate GOLFer alongside dominant LLM-based query expansion methods on three web search and ten low-resource datasets. Experimental results demonstrate that GOLFer consistently outperforms other methods using smaller LMs, and maintains competitive performance against methods using large-size LLMs, demonstrating its effectiveness. 

---
# Exp4Fuse: A Rank Fusion Framework for Enhanced Sparse Retrieval using Large Language Model-based Query Expansion 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04760)  

**Abstract**: Large Language Models (LLMs) have shown potential in generating hypothetical documents for query expansion, thereby enhancing information retrieval performance. However, the efficacy of this method is highly dependent on the quality of the generated documents, which often requires complex prompt strategies and the integration of advanced dense retrieval techniques. This can be both costly and computationally intensive. To mitigate these limitations, we explore the use of zero-shot LLM-based query expansion to improve sparse retrieval, particularly for learned sparse retrievers. We introduce a novel fusion ranking framework, Exp4Fuse, which enhances the performance of sparse retrievers through an indirect application of zero-shot LLM-based query expansion. Exp4Fuse operates by simultaneously considering two retrieval routes-one based on the original query and the other on the LLM-augmented query. It then generates two ranked lists using a sparse retriever and fuses them using a modified reciprocal rank fusion method. We conduct extensive evaluations of Exp4Fuse against leading LLM-based query expansion methods and advanced retrieval techniques on three MS MARCO-related datasets and seven low-resource datasets. Experimental results reveal that Exp4Fuse not only surpasses existing LLM-based query expansion methods in enhancing sparse retrievers but also, when combined with advanced sparse retrievers, achieves SOTA results on several benchmarks. This highlights the superior performance and effectiveness of Exp4Fuse in improving query expansion for sparse retrieval. 

---
# Evaluation is All You Need: Strategic Overclaiming of LLM Reasoning Capabilities Through Evaluation Design 

**Authors**: Lin Sun, Weihong Lin, Jinzhu Wu, Yongfu Zhu, Xiaoqi Jian, Guangxiang Zhao, Change Jia, Linglin Zhang, Sai-er Hu, Yuhan Wu, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04734)  

**Abstract**: Reasoning models represented by the Deepseek-R1-Distill series have been widely adopted by the open-source community due to their strong performance in mathematics, science, programming, and other domains. However, our study reveals that their benchmark evaluation results are subject to significant fluctuations caused by various factors. Subtle differences in evaluation conditions can lead to substantial variations in results. Similar phenomena are observed in other open-source inference models fine-tuned based on the Deepseek-R1-Distill series, as well as in the QwQ-32B model, making their claimed performance improvements difficult to reproduce reliably. Therefore, we advocate for the establishment of a more rigorous paradigm for model performance evaluation and present our empirical assessments of the Deepseek-R1-Distill series models. 

---
# LLM-based phoneme-to-grapheme for phoneme-based speech recognition 

**Authors**: Te Ma, Min Bi, Saierdaer Yusuyin, Hao Huang, Zhijian Ou  

**Link**: [PDF](https://arxiv.org/pdf/2506.04711)  

**Abstract**: In automatic speech recognition (ASR), phoneme-based multilingual pre-training and crosslingual fine-tuning is attractive for its high data efficiency and competitive results compared to subword-based models. However, Weighted Finite State Transducer (WFST) based decoding is limited by its complex pipeline and inability to leverage large language models (LLMs). Therefore, we propose LLM-based phoneme-to-grapheme (LLM-P2G) decoding for phoneme-based ASR, consisting of speech-to-phoneme (S2P) and phoneme-to-grapheme (P2G). A challenge is that there seems to have information loss in cascading S2P and P2G. To address this challenge, we propose two training strategies: data augmentation with noisy phonemes (DANP), and randomized top-$K$ marginalized (TKM) training and decoding. Our experimental results show that LLM-P2G outperforms WFST-based systems in crosslingual ASR for Polish and German, by relative WER reductions of 3.6% and 6.9% respectively. 

---
# Urania: Differentially Private Insights into AI Use 

**Authors**: Daogao Liu, Edith Cohen, Badih Ghazi, Peter Kairouz, Pritish Kamath, Alexander Knop, Ravi Kumar, Pasin Manurangsi, Adam Sealfon, Da Yu, Chiyuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04681)  

**Abstract**: We introduce $Urania$, a novel framework for generating insights about LLM chatbot interactions with rigorous differential privacy (DP) guarantees. The framework employs a private clustering mechanism and innovative keyword extraction methods, including frequency-based, TF-IDF-based, and LLM-guided approaches. By leveraging DP tools such as clustering, partition selection, and histogram-based summarization, $Urania$ provides end-to-end privacy protection. Our evaluation assesses lexical and semantic content preservation, pair similarity, and LLM-based metrics, benchmarking against a non-private Clio-inspired pipeline (Tamkin et al., 2024). Moreover, we develop a simple empirical privacy evaluation that demonstrates the enhanced robustness of our DP pipeline. The results show the framework's ability to extract meaningful conversational insights while maintaining stringent user privacy, effectively balancing data utility with privacy preservation. 

---
# EMO-Debias: Benchmarking Gender Debiasing Techniques in Multi-Label Speech Emotion Recognition 

**Authors**: Yi-Cheng Lin, Huang-Cheng Chou, Yu-Hsuan Li Liang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.04652)  

**Abstract**: Speech emotion recognition (SER) systems often exhibit gender bias. However, the effectiveness and robustness of existing debiasing methods in such multi-label scenarios remain underexplored. To address this gap, we present EMO-Debias, a large-scale comparison of 13 debiasing methods applied to multi-label SER. Our study encompasses techniques from pre-processing, regularization, adversarial learning, biased learners, and distributionally robust optimization. Experiments conducted on acted and naturalistic emotion datasets, using WavLM and XLSR representations, evaluate each method under conditions of gender imbalance. Our analysis quantifies the trade-offs between fairness and accuracy, identifying which approaches consistently reduce gender performance gaps without compromising overall model performance. The findings provide actionable insights for selecting effective debiasing strategies and highlight the impact of dataset distributions. 

---
# Clustering and Median Aggregation Improve Differentially Private Inference 

**Authors**: Kareem Amin, Salman Avestimehr, Sara Babakniya, Alex Bie, Weiwei Kong, Natalia Ponomareva, Umar Syed  

**Link**: [PDF](https://arxiv.org/pdf/2506.04566)  

**Abstract**: Differentially private (DP) language model inference is an approach for generating private synthetic text. A sensitive input example is used to prompt an off-the-shelf large language model (LLM) to produce a similar example. Multiple examples can be aggregated together to formally satisfy the DP guarantee.
Prior work creates inference batches by sampling sensitive inputs uniformly at random. We show that uniform sampling degrades the quality of privately generated text, especially when the sensitive examples concern heterogeneous topics.
We remedy this problem by clustering the input data before selecting inference batches. Next, we observe that clustering also leads to more similar next-token predictions across inferences. We use this insight to introduce a new algorithm that aggregates next token statistics by privately computing medians instead of averages. This approach leverages the fact that the median has decreased local sensitivity when next token predictions are similar, allowing us to state a data-dependent and ex-post DP guarantee about the privacy properties of this algorithm. Finally, we demonstrate improvements in terms of representativeness metrics (e.g., MAUVE) as well as downstream task performance. We show that our method produces high-quality synthetic data at significantly lower privacy cost than a previous state-of-the-art method. 

---
# Grapheme-Coherent Phonemic and Prosodic Annotation of Speech by Implicit and Explicit Grapheme Conditioning 

**Authors**: Hien Ohnaka, Yuma Shirahata, Byeongseon Park, Ryuichi Yamamoto  

**Link**: [PDF](https://arxiv.org/pdf/2506.04527)  

**Abstract**: We propose a model to obtain phonemic and prosodic labels of speech that are coherent with graphemes. Unlike previous methods that simply fine-tune a pre-trained ASR model with the labels, the proposed model conditions the label generation on corresponding graphemes by two methods: 1) Add implicit grapheme conditioning through prompt encoder using pre-trained BERT features. 2) Explicitly prune the label hypotheses inconsistent with the grapheme during inference. These methods enable obtaining parallel data of speech, the labels, and graphemes, which is applicable to various downstream tasks such as text-to-speech and accent estimation from text. Experiments showed that the proposed method significantly improved the consistency between graphemes and the predicted labels. Further, experiments on accent estimation task confirmed that the created parallel data by the proposed method effectively improve the estimation accuracy. 

---
# Towards Efficient Speech-Text Jointly Decoding within One Speech Language Model 

**Authors**: Haibin Wu, Yuxuan Hu, Ruchao Fan, Xiaofei Wang, Kenichi Kumatani, Bo Ren, Jianwei Yu, Heng Lu, Lijuan Wang, Yao Qian, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04518)  

**Abstract**: Speech language models (Speech LMs) enable end-to-end speech-text modelling within a single model, offering a promising direction for spoken dialogue systems. The choice of speech-text jointly decoding paradigm plays a critical role in performance, efficiency, and alignment quality. In this work, we systematically compare representative joint speech-text decoding strategies-including the interleaved, and parallel generation paradigms-under a controlled experimental setup using the same base language model, speech tokenizer and training data. Our results show that the interleaved approach achieves the best alignment. However it suffers from slow inference due to long token sequence length. To address this, we propose a novel early-stop interleaved (ESI) pattern that not only significantly accelerates decoding but also yields slightly better performance. Additionally, we curate high-quality question answering (QA) datasets to further improve speech QA performance. 

---
# Understanding and Meeting Practitioner Needs When Measuring Representational Harms Caused by LLM-Based Systems 

**Authors**: Emma Harvey, Emily Sheng, Su Lin Blodgett, Alexandra Chouldechova, Jean Garcia-Gathright, Alexandra Olteanu, Hanna Wallach  

**Link**: [PDF](https://arxiv.org/pdf/2506.04482)  

**Abstract**: The NLP research community has made publicly available numerous instruments for measuring representational harms caused by large language model (LLM)-based systems. These instruments have taken the form of datasets, metrics, tools, and more. In this paper, we examine the extent to which such instruments meet the needs of practitioners tasked with evaluating LLM-based systems. Via semi-structured interviews with 12 such practitioners, we find that practitioners are often unable to use publicly available instruments for measuring representational harms. We identify two types of challenges. In some cases, instruments are not useful because they do not meaningfully measure what practitioners seek to measure or are otherwise misaligned with practitioner needs. In other cases, instruments - even useful instruments - are not used by practitioners due to practical and institutional barriers impeding their uptake. Drawing on measurement theory and pragmatic measurement, we provide recommendations for addressing these challenges to better meet practitioner needs. 

---
# Behavioural vs. Representational Systematicity in End-to-End Models: An Opinionated Survey 

**Authors**: Ivan Vegner, Sydelle de Souza, Valentin Forch, Martha Lewis, Leonidas A.A. Doumas  

**Link**: [PDF](https://arxiv.org/pdf/2506.04461)  

**Abstract**: A core aspect of compositionality, systematicity is a desirable property in ML models as it enables strong generalization to novel contexts. This has led to numerous studies proposing benchmarks to assess systematic generalization, as well as models and training regimes designed to enhance it. Many of these efforts are framed as addressing the challenge posed by Fodor and Pylyshyn. However, while they argue for systematicity of representations, existing benchmarks and models primarily focus on the systematicity of behaviour. We emphasize the crucial nature of this distinction. Furthermore, building on Hadley's (1994) taxonomy of systematic generalization, we analyze the extent to which behavioural systematicity is tested by key benchmarks in the literature across language and vision. Finally, we highlight ways of assessing systematicity of representations in ML models as practiced in the field of mechanistic interpretability. 

---
# Matter-of-Fact: A Benchmark for Verifying the Feasibility of Literature-Supported Claims in Materials Science 

**Authors**: Peter Jansen, Samiah Hassan, Ruoyao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04410)  

**Abstract**: Contemporary approaches to assisted scientific discovery use language models to automatically generate large numbers of potential hypothesis to test, while also automatically generating code-based experiments to test those hypotheses. While hypotheses can be comparatively inexpensive to generate, automated experiments can be costly, particularly when run at scale (i.e. thousands of experiments). Developing the capacity to filter hypotheses based on their feasibility would allow discovery systems to run at scale, while increasing their likelihood of making significant discoveries. In this work we introduce Matter-of-Fact, a challenge dataset for determining the feasibility of hypotheses framed as claims. Matter-of-Fact includes 8.4k claims extracted from scientific articles spanning four high-impact contemporary materials science topics, including superconductors, semiconductors, batteries, and aerospace materials, while including qualitative and quantitative claims from theoretical, experimental, and code/simulation results. We show that strong baselines that include retrieval augmented generation over scientific literature and code generation fail to exceed 72% performance on this task (chance performance is 50%), while domain-expert verification suggests nearly all are solvable -- highlighting both the difficulty of this task for current models, and the potential to accelerate scientific discovery by making near-term progress. 

---
# Can we reconstruct a dysarthric voice with the large speech model Parler TTS? 

**Authors**: Ariadna Sanchez, Simon King  

**Link**: [PDF](https://arxiv.org/pdf/2506.04397)  

**Abstract**: Speech disorders can make communication hard or even impossible for those who develop them. Personalised Text-to-Speech is an attractive option as a communication aid. We attempt voice reconstruction using a large speech model, with which we generate an approximation of a dysarthric speaker's voice prior to the onset of their condition. In particular, we investigate whether a state-of-the-art large speech model, Parler TTS, can generate intelligible speech while maintaining speaker identity. We curate a dataset and annotate it with relevant speaker and intelligibility information, and use this to fine-tune the model. Our results show that the model can indeed learn to generate from the distribution of this challenging data, but struggles to control intelligibility and to maintain consistent speaker identity. We propose future directions to improve controllability of this class of model, for the voice reconstruction task. 

---
# ReXVQA: A Large-scale Visual Question Answering Benchmark for Generalist Chest X-ray Understanding 

**Authors**: Ankit Pal, Jung-Oh Lee, Xiaoman Zhang, Malaikannan Sankarasubbu, Seunghyeon Roh, Won Jung Kim, Meesun Lee, Pranav Rajpurkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.04353)  

**Abstract**: We present ReXVQA, the largest and most comprehensive benchmark for visual question answering (VQA) in chest radiology, comprising approximately 696,000 questions paired with 160,000 chest X-rays studies across training, validation, and test sets. Unlike prior efforts that rely heavily on template based queries, ReXVQA introduces a diverse and clinically authentic task suite reflecting five core radiological reasoning skills: presence assessment, location analysis, negation detection, differential diagnosis, and geometric reasoning. We evaluate eight state-of-the-art multimodal large language models, including MedGemma-4B-it, Qwen2.5-VL, Janus-Pro-7B, and Eagle2-9B. The best-performing model (MedGemma) achieves 83.24% overall accuracy. To bridge the gap between AI performance and clinical expertise, we conducted a comprehensive human reader study involving 3 radiology residents on 200 randomly sampled cases. Our evaluation demonstrates that MedGemma achieved superior performance (83.84% accuracy) compared to human readers (best radiology resident: 77.27%), representing a significant milestone where AI performance exceeds expert human evaluation on chest X-ray interpretation. The reader study reveals distinct performance patterns between AI models and human experts, with strong inter-reader agreement among radiologists while showing more variable agreement patterns between human readers and AI models. ReXVQA establishes a new standard for evaluating generalist radiological AI systems, offering public leaderboards, fine-grained evaluation splits, structured explanations, and category-level breakdowns. This benchmark lays the foundation for next-generation AI systems capable of mimicking expert-level clinical reasoning beyond narrow pathology classification. Our dataset will be open-sourced at this https URL 

---
# A Graph-Retrieval-Augmented Generation Framework Enhances Decision-Making in the Circular Economy 

**Authors**: Yang Zhao, Chengxiao Dai, Dusit Niyato, Chuan Fu Tan, Keyi Xiang, Yueyang Wang, Zhiquan Yeo, Daren Tan Zong Loong, Jonathan Low Zhaozhi, Eugene H.Z. HO  

**Link**: [PDF](https://arxiv.org/pdf/2506.04252)  

**Abstract**: Large language models (LLMs) hold promise for sustainable manufacturing, but often hallucinate industrial codes and emission factors, undermining regulatory and investment decisions. We introduce CircuGraphRAG, a retrieval-augmented generation (RAG) framework that grounds LLMs outputs in a domain-specific knowledge graph for the circular economy. This graph connects 117,380 industrial and waste entities with classification codes and GWP100 emission data, enabling structured multi-hop reasoning. Natural language queries are translated into SPARQL and verified subgraphs are retrieved to ensure accuracy and traceability. Compared with Standalone LLMs and Naive RAG, CircuGraphRAG achieves superior performance in single-hop and multi-hop question answering, with ROUGE-L F1 scores up to 1.0, while baseline scores below 0.08. It also improves efficiency, halving the response time and reducing token usage by 16% in representative tasks. CircuGraphRAG provides fact-checked, regulatory-ready support for circular economy planning, advancing reliable, low-carbon resource decision making. 

---
# Contextual Integrity in LLMs via Reasoning and Reinforcement Learning 

**Authors**: Guangchen Lan, Huseyin A. Inan, Sahar Abdelnabi, Janardhan Kulkarni, Lukas Wutschitz, Reza Shokri, Christopher G. Brinton, Robert Sim  

**Link**: [PDF](https://arxiv.org/pdf/2506.04245)  

**Abstract**: As the era of autonomous agents making decisions on behalf of users unfolds, ensuring contextual integrity (CI) -- what is the appropriate information to share while carrying out a certain task -- becomes a central question to the field. We posit that CI demands a form of reasoning where the agent needs to reason about the context in which it is operating. To test this, we first prompt LLMs to reason explicitly about CI when deciding what information to disclose. We then extend this approach by developing a reinforcement learning (RL) framework that further instills in models the reasoning necessary to achieve CI. Using a synthetic, automatically created, dataset of only $\sim700$ examples but with diverse contexts and information disclosure norms, we show that our method substantially reduces inappropriate information disclosure while maintaining task performance across multiple model sizes and families. Importantly, improvements transfer from this synthetic dataset to established CI benchmarks such as PrivacyLens that has human annotations and evaluates privacy leakage of AI assistants in actions and tool calls. 

---
