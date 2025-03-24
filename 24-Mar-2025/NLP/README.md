# Dancing with Critiques: Enhancing LLM Reasoning with Stepwise Natural Language Self-Critique 

**Authors**: Yansi Li, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Qiuzhi Liu, Rui Wang, Zhuosheng Zhang, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17363)  

**Abstract**: Enhancing the reasoning capabilities of large language models (LLMs), particularly for complex tasks requiring multi-step logical deductions, remains a significant challenge. Traditional inference time scaling methods utilize scalar reward signals from process reward models to evaluate candidate reasoning steps, but these scalar rewards lack the nuanced qualitative information essential for understanding and justifying each step. In this paper, we propose a novel inference-time scaling approach -- stepwise natural language self-critique (PANEL), which employs self-generated natural language critiques as feedback to guide the step-level search process. By generating rich, human-readable critiques for each candidate reasoning step, PANEL retains essential qualitative information, facilitating better-informed decision-making during inference. This approach bypasses the need for task-specific verifiers and the associated training overhead, making it broadly applicable across diverse tasks. Experimental results on challenging reasoning benchmarks, including AIME and GPQA, demonstrate that PANEL significantly enhances reasoning performance, outperforming traditional scalar reward-based methods. Our code is available at this https URL to support and encourage future research in this promising field. 

---
# Efficient Intent-Based Filtering for Multi-Party Conversations Using Knowledge Distillation from LLMs 

**Authors**: Reem Gody, Mohamed Abdelghaffar, Mohammed Jabreel, Ahmed Tawfik  

**Link**: [PDF](https://arxiv.org/pdf/2503.17336)  

**Abstract**: Large language models (LLMs) have showcased remarkable capabilities in conversational AI, enabling open-domain responses in chat-bots, as well as advanced processing of conversations like summarization, intent classification, and insights generation. However, these models are resource-intensive, demanding substantial memory and computational power. To address this, we propose a cost-effective solution that filters conversational snippets of interest for LLM processing, tailored to the target downstream application, rather than processing every snippet. In this work, we introduce an innovative approach that leverages knowledge distillation from LLMs to develop an intent-based filter for multi-party conversations, optimized for compute power constrained environments. Our method combines different strategies to create a diverse multi-party conversational dataset, that is annotated with the target intents and is then used to fine-tune the MobileBERT model for multi-label intent classification. This model achieves a balance between efficiency and performance, effectively filtering conversation snippets based on their intents. By passing only the relevant snippets to the LLM for further processing, our approach significantly reduces overall operational costs depending on the intents and the data distribution as demonstrated in our experiments. 

---
# FastCuRL: Curriculum Reinforcement Learning with Progressive Context Extension for Efficient Training R1-like Reasoning Models 

**Authors**: Mingyang Song, Mao Zheng, Zheng Li, Wenjie Yang, Xuan Luo, Yue Pan, Feng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17287)  

**Abstract**: In this paper, we propose \textbf{\textsc{FastCuRL}}, a simple yet efficient \textbf{Cu}rriculum \textbf{R}einforcement \textbf{L}earning approach with context window extending strategy to accelerate the reinforcement learning training efficiency for R1-like reasoning models while enhancing their performance in tackling complex reasoning tasks with long chain-of-thought rationales, particularly with a 1.5B parameter language model. \textbf{\textsc{FastCuRL}} consists of two main procedures: length-aware training data segmentation and context window extension training. Specifically, the former first splits the original training data into three different levels by the input prompt length, and then the latter leverages segmented training datasets with a progressively increasing context window length to train the reasoning model. Experimental results demonstrate that \textbf{\textsc{FastCuRL}}-1.5B-Preview surpasses DeepScaleR-1.5B-Preview across all five datasets (including MATH 500, AIME 2024, AMC 2023, Minerva Math, and OlympiadBench) while only utilizing 50\% of training steps. Furthermore, all training stages for FastCuRL-1.5B-Preview are completed using just a single node with 8 GPUs. 

---
# CASE -- Condition-Aware Sentence Embeddings for Conditional Semantic Textual Similarity Measurement 

**Authors**: Gaifan Zhang, Yi Zhou, Danushka Bollegala  

**Link**: [PDF](https://arxiv.org/pdf/2503.17279)  

**Abstract**: The meaning conveyed by a sentence often depends on the context in which it appears. Despite the progress of sentence embedding methods, it remains unclear how to best modify a sentence embedding conditioned on its context. To address this problem, we propose Condition-Aware Sentence Embeddings (CASE), an efficient and accurate method to create an embedding for a sentence under a given condition. First, CASE creates an embedding for the condition using a Large Language Model (LLM), where the sentence influences the attention scores computed for the tokens in the condition during pooling. Next, a supervised nonlinear projection is learned to reduce the dimensionality of the LLM-based text embeddings. We show that CASE significantly outperforms previously proposed Conditional Semantic Textual Similarity (C-STS) methods on an existing standard benchmark dataset. We find that subtracting the condition embedding consistently improves the C-STS performance of LLM-based text embeddings. Moreover, we propose a supervised dimensionality reduction method that not only reduces the dimensionality of LLM-based embeddings but also significantly improves their performance. 

---
# KL3M Tokenizers: A Family of Domain-Specific and Character-Level Tokenizers for Legal, Financial, and Preprocessing Applications 

**Authors**: Michael J Bommarito, Daniel Martin Katz, Jillian Bommarito  

**Link**: [PDF](https://arxiv.org/pdf/2503.17247)  

**Abstract**: We present the KL3M tokenizers, a family of specialized tokenizers for legal, financial, and governmental text. Despite established work on tokenization, specialized tokenizers for professional domains remain understudied. Our paper offers two main contributions to this area.
First, we introduce domain-specific BPE tokenizers for legal, financial, and governmental text. Our kl3m-004-128k-cased tokenizer uses 9-17% fewer tokens than GPT-4o and Llama3 for domain-specific documents, despite having a smaller vocabulary. For specialized terminology, our cased tokenizer is even more efficient, using up to 83% fewer tokens for legal terms and 39% fewer tokens for financial terms.
Second, we develop character-level BPE tokenizers (4K, 8K, and 16K vocabulary sizes) for text correction tasks like OCR post-processing. These tokenizers keep consistent token boundaries between error-containing and correct text, making it easier for models to learn correction patterns.
These tokenizers help professional applications by fitting more text in context windows, reducing computational needs, and preserving the meaning of domain-specific terms. Our analysis shows these efficiency gains directly benefit the processing of long legal and financial documents. We release all tokenizers and code through GitHub and Hugging Face to support further research in specialized tokenization. 

---
# SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging 

**Authors**: Aladin Djuhera, Swanand Ravindra Kadhe, Farhan Ahmed, Syed Zawad, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2503.17239)  

**Abstract**: Fine-tuning large language models (LLMs) on downstream tasks can inadvertently erode their safety alignment, even for benign fine-tuning datasets. We address this challenge by proposing SafeMERGE, a post-fine-tuning framework that preserves safety while maintaining task utility. It achieves this by selectively merging fine-tuned and safety-aligned model layers only when those deviate from safe behavior, measured by a cosine similarity criterion. We evaluate SafeMERGE against other fine-tuning- and post-fine-tuning-stage approaches for Llama-2-7B-Chat and Qwen-2-7B-Instruct models on GSM8K and PubMedQA tasks while exploring different merging strategies. We find that SafeMERGE consistently reduces harmful outputs compared to other baselines without significantly sacrificing performance, sometimes even enhancing it. The results suggest that our selective, subspace-guided, and per-layer merging method provides an effective safeguard against the inadvertent loss of safety in fine-tuned LLMs while outperforming simpler post-fine-tuning-stage defenses. 

---
# Automating Adjudication of Cardiovascular Events Using Large Language Models 

**Authors**: Sonish Sivarajkumar, Kimia Ameri, Chuqin Li, Yanshan Wang, Min Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17222)  

**Abstract**: Cardiovascular events, such as heart attacks and strokes, remain a leading cause of mortality globally, necessitating meticulous monitoring and adjudication in clinical trials. This process, traditionally performed manually by clinical experts, is time-consuming, resource-intensive, and prone to inter-reviewer variability, potentially introducing bias and hindering trial progress. This study addresses these critical limitations by presenting a novel framework for automating the adjudication of cardiovascular events in clinical trials using Large Language Models (LLMs). We developed a two-stage approach: first, employing an LLM-based pipeline for event information extraction from unstructured clinical data and second, using an LLM-based adjudication process guided by a Tree of Thoughts approach and clinical endpoint committee (CEC) guidelines. Using cardiovascular event-specific clinical trial data, the framework achieved an F1-score of 0.82 for event extraction and an accuracy of 0.68 for adjudication. Furthermore, we introduce the CLEART score, a novel, automated metric specifically designed for evaluating the quality of AI-generated clinical reasoning in adjudicating cardiovascular events. This approach demonstrates significant potential for substantially reducing adjudication time and costs while maintaining high-quality, consistent, and auditable outcomes in clinical trials. The reduced variability and enhanced standardization also allow for faster identification and mitigation of risks associated with cardiovascular therapies. 

---
# A Language Anchor-Guided Method for Robust Noisy Domain Generalization 

**Authors**: Zilin Dai, Lehong Wang, Fangzhou Lin, Yidong Wang, Zhigang Li, Kazunori D Yamada, Ziming Zhang, Wang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17211)  

**Abstract**: Real-world machine learning applications often struggle with two major challenges: distribution shift and label noise. Models tend to overfit by focusing on redundant and uninformative features in the training data, which makes it hard for them to generalize to the target domain. Noisy data worsens this problem by causing further overfitting to the noise, meaning that existing methods often fail to tell the difference between true, invariant features and misleading, spurious ones. To tackle these issues, we introduce Anchor Alignment and Adaptive Weighting (A3W). This new algorithm uses sample reweighting guided by natural language processing (NLP) anchors to extract more representative features. In simple terms, A3W leverages semantic representations from natural language models as a source of domain-invariant prior knowledge. Additionally, it employs a weighted loss function that adjusts each sample's contribution based on its similarity to the corresponding NLP anchor. This adjustment makes the model more robust to noisy labels. Extensive experiments on standard benchmark datasets show that A3W consistently outperforms state-of-the-art domain generalization methods, offering significant improvements in both accuracy and robustness across different datasets and noise levels. 

---
# CoKe: Customizable Fine-Grained Story Evaluation via Chain-of-Keyword Rationalization 

**Authors**: Brihi Joshi, Sriram Venkatapathy, Mohit Bansal, Nanyun Peng, Haw-Shiuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17136)  

**Abstract**: Evaluating creative text such as human-written stories using language models has always been a challenging task -- owing to the subjectivity of multi-annotator ratings. To mimic the thinking process of humans, chain of thought (CoT) generates free-text explanations that help guide a model's predictions and Self-Consistency (SC) marginalizes predictions over multiple generated explanations. In this study, we discover that the widely-used self-consistency reasoning methods cause suboptimal results due to an objective mismatch between generating 'fluent-looking' explanations vs. actually leading to a good rating prediction for an aspect of a story. To overcome this challenge, we propose $\textbf{C}$hain-$\textbf{o}$f-$\textbf{Ke}$ywords (CoKe), that generates a sequence of keywords $\textit{before}$ generating a free-text rationale, that guide the rating prediction of our evaluation language model. Then, we generate a diverse set of such keywords, and aggregate the scores corresponding to these generations. On the StoryER dataset, CoKe based on our small fine-tuned evaluation models not only reach human-level performance and significantly outperform GPT-4 with a 2x boost in correlation with human annotators, but also requires drastically less number of parameters. 

---
# Modifying Large Language Model Post-Training for Diverse Creative Writing 

**Authors**: John Joon Young Chung, Vishakh Padmakumar, Melissa Roemmele, Yuqian Sun, Max Kreminski  

**Link**: [PDF](https://arxiv.org/pdf/2503.17126)  

**Abstract**: As creative writing tasks do not have singular correct answers, large language models (LLMs) trained to perform these tasks should be able to generate diverse valid outputs. However, LLM post-training often focuses on improving generation quality but neglects to facilitate output diversity. Hence, in creative writing generation, we investigate post-training approaches to promote both output diversity and quality. Our core idea is to include deviation -- the degree of difference between a training sample and all other samples with the same prompt -- in the training objective to facilitate learning from rare high-quality instances. By adopting our approach to direct preference optimization (DPO) and odds ratio preference optimization (ORPO), we demonstrate that we can promote the output diversity of trained models while minimally decreasing quality. Our best model with 8B parameters could achieve on-par diversity as a human-created dataset while having output quality similar to the best instruction-tuned models we examined, GPT-4o and DeepSeek-R1. We further validate our approaches with a human evaluation, an ablation, and a comparison to an existing diversification approach, DivPO. 

---
# A Study into Investigating Temporal Robustness of LLMs 

**Authors**: Jonas Wallat, Abdelrahman Abdallah, Adam Jatowt, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2503.17073)  

**Abstract**: Large Language Models (LLMs) encapsulate a surprising amount of factual world knowledge. However, their performance on temporal questions and historical knowledge is limited because they often cannot understand temporal scope and orientation or neglect the temporal aspect altogether. In this study, we aim to measure precisely how robust LLMs are for question answering based on their ability to process temporal information and perform tasks requiring temporal reasoning and temporal factual knowledge. Specifically, we design eight time-sensitive robustness tests for factual information to check the sensitivity of six popular LLMs in the zero-shot setting. Overall, we find LLMs lacking temporal robustness, especially to temporal reformulations and the use of different granularities of temporal references. We show how a selection of these eight tests can be used automatically to judge a model's temporal robustness for user questions on the fly. Finally, we apply the findings of this study to improve the temporal QA performance by up to 55 percent. 

---
# Summarization Metrics for Spanish and Basque: Do Automatic Scores and LLM-Judges Correlate with Humans? 

**Authors**: Jeremy Barnes, Naiara Perez, Alba Bonet-Jover, Begoña Altuna  

**Link**: [PDF](https://arxiv.org/pdf/2503.17039)  

**Abstract**: Studies on evaluation metrics and LLM-as-a-Judge models for automatic text summarization have largely been focused on English, limiting our understanding of their effectiveness in other languages. Through our new dataset BASSE (BAsque and Spanish Summarization Evaluation), we address this situation by collecting human judgments on 2,040 abstractive summaries in Basque and Spanish, generated either manually or by five LLMs with four different prompts. For each summary, annotators evaluated five criteria on a 5-point Likert scale: coherence, consistency, fluency, relevance, and 5W1H. We use these data to reevaluate traditional automatic metrics used for evaluating summaries, as well as several LLM-as-a-Judge models that show strong performance on this task in English. Our results show that currently proprietary judge LLMs have the highest correlation with human judgments, followed by criteria-specific automatic metrics, while open-sourced judge LLMs perform poorly. We release BASSE and our code publicly, along with the first large-scale Basque summarization dataset containing 22,525 news articles with their subheads. 

---
# A Survey on Personalized Alignment -- The Missing Piece for Large Language Models in Real-World Applications 

**Authors**: Jian Guan, Junfei Wu, Jia-Nan Li, Chuanqi Cheng, Wei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17003)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their transition to real-world applications reveals a critical limitation: the inability to adapt to individual preferences while maintaining alignment with universal human values. Current alignment techniques adopt a one-size-fits-all approach that fails to accommodate users' diverse backgrounds and needs. This paper presents the first comprehensive survey of personalized alignment-a paradigm that enables LLMs to adapt their behavior within ethical boundaries based on individual preferences. We propose a unified framework comprising preference memory management, personalized generation, and feedback-based alignment, systematically analyzing implementation approaches and evaluating their effectiveness across various scenarios. By examining current techniques, potential risks, and future challenges, this survey provides a structured foundation for developing more adaptable and ethically-aligned LLMs. 

---
# When Words Outperform Vision: VLMs Can Self-Improve Via Text-Only Training For Human-Centered Decision Making 

**Authors**: Zhe Hu, Jing Li, Yu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2503.16965)  

**Abstract**: Embodied decision-making is fundamental for AI agents operating in real-world environments. While Visual Language Models (VLMs) have advanced this capability, they still struggle with complex decisions, particularly in human-centered situations that require deep reasoning about human needs and values. In this study, we systematically evaluate open-sourced VLMs on multimodal human-centered decision-making tasks. We find that LLMs receiving only textual descriptions unexpectedly outperform their VLM counterparts of similar scale that process actual images, suggesting that visual alignment may hinder VLM abilities. To address this challenge, we propose a novel text-only training approach with synthesized textual data. This method strengthens VLMs' language components and transfers the learned abilities to multimodal inference, eliminating the need for expensive image-text paired data. Furthermore, we show that VLMs can achieve substantial performance gains through self-improvement, using training data generated by their LLM counterparts rather than relying on larger teacher models like GPT-4. Our findings establish a more efficient and scalable approach to enhancing VLMs' human-centered decision-making capabilities, opening new avenues for optimizing VLMs through self-improvement mechanisms. 

---
# Assessing the Reliability and Validity of GPT-4 in Annotating Emotion Appraisal Ratings 

**Authors**: Deniss Ruder, Andero Uusberg, Kairit Sirts  

**Link**: [PDF](https://arxiv.org/pdf/2503.16883)  

**Abstract**: Appraisal theories suggest that emotions arise from subjective evaluations of events, referred to as appraisals. The taxonomy of appraisals is quite diverse, and they are usually given ratings on a Likert scale to be annotated in an experiencer-annotator or reader-annotator paradigm. This paper studies GPT-4 as a reader-annotator of 21 specific appraisal ratings in different prompt settings, aiming to evaluate and improve its performance compared to human annotators. We found that GPT-4 is an effective reader-annotator that performs close to or even slightly better than human annotators, and its results can be significantly improved by using a majority voting of five completions. GPT-4 also effectively predicts appraisal ratings and emotion labels using a single prompt, but adding instruction complexity results in poorer performance. We also found that longer event descriptions lead to more accurate annotations for both model and human annotator ratings. This work contributes to the growing usage of LLMs in psychology and the strategies for improving GPT-4 performance in annotating appraisals. 

---
# Joint Extraction Matters: Prompt-Based Visual Question Answering for Multi-Field Document Information Extraction 

**Authors**: Mengsay Loem, Taiju Hosaka  

**Link**: [PDF](https://arxiv.org/pdf/2503.16868)  

**Abstract**: Visual question answering (VQA) has emerged as a flexible approach for extracting specific pieces of information from document images. However, existing work typically queries each field in isolation, overlooking potential dependencies across multiple items. This paper investigates the merits of extracting multiple fields jointly versus separately. Through experiments on multiple large vision language models and datasets, we show that jointly extracting fields often improves accuracy, especially when the fields share strong numeric or contextual dependencies. We further analyze how performance scales with the number of requested items and use a regression based metric to quantify inter field relationships. Our results suggest that multi field prompts can mitigate confusion arising from similar surface forms and related numeric values, providing practical methods for designing robust VQA systems in document information extraction tasks. 

---
# MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering 

**Authors**: Jialin Chen, Aosong Feng, Ziyu Zhao, Juan Garza, Gaukhar Nurbek, Cheng Qin, Ali Maatouk, Leandros Tassiulas, Yifeng Gao, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2503.16858)  

**Abstract**: Understanding the relationship between textual news and time-series evolution is a critical yet under-explored challenge in applied data science. While multimodal learning has gained traction, existing multimodal time-series datasets fall short in evaluating cross-modal reasoning and complex question answering, which are essential for capturing complex interactions between narrative information and temporal patterns. To bridge this gap, we introduce Multimodal Time Series Benchmark (MTBench), a large-scale benchmark designed to evaluate large language models (LLMs) on time series and text understanding across financial and weather domains. MTbench comprises paired time series and textual data, including financial news with corresponding stock price movements and weather reports aligned with historical temperature records. Unlike existing benchmarks that focus on isolated modalities, MTbench provides a comprehensive testbed for models to jointly reason over structured numerical trends and unstructured textual narratives. The richness of MTbench enables formulation of diverse tasks that require a deep understanding of both text and time-series data, including time-series forecasting, semantic and technical trend analysis, and news-driven question answering (QA). These tasks target the model's ability to capture temporal dependencies, extract key insights from textual context, and integrate cross-modal information. We evaluate state-of-the-art LLMs on MTbench, analyzing their effectiveness in modeling the complex relationships between news narratives and temporal patterns. Our findings reveal significant challenges in current models, including difficulties in capturing long-term dependencies, interpreting causality in financial and weather trends, and effectively fusing multimodal information. 

---
# MMCR: Benchmarking Cross-Source Reasoning in Scientific Papers 

**Authors**: Yang Tian, Zheng Lu, Mingqi Gao, Zheng Liu, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16856)  

**Abstract**: Fully comprehending scientific papers by machines reflects a high level of Artificial General Intelligence, requiring the ability to reason across fragmented and heterogeneous sources of information, presenting a complex and practically significant challenge. While Vision-Language Models (VLMs) have made remarkable strides in various tasks, particularly those involving reasoning with evidence source from single image or text page, their ability to use cross-source information for reasoning remains an open problem. This work presents MMCR, a high-difficulty benchmark designed to evaluate VLMs' capacity for reasoning with cross-source information from scientific papers. The benchmark comprises 276 high-quality questions, meticulously annotated by humans across 7 subjects and 10 task types. Experiments with 18 VLMs demonstrate that cross-source reasoning presents a substantial challenge for existing models. Notably, even the top-performing model, GPT-4o, achieved only 48.55% overall accuracy, with only 20% accuracy in multi-table comprehension tasks, while the second-best model, Qwen2.5-VL-72B, reached 39.86% overall accuracy. Furthermore, we investigated the impact of the Chain-of-Thought (CoT) technique on cross-source reasoning and observed a detrimental effect on small models, whereas larger models demonstrated substantially enhanced performance. These results highlight the pressing need to develop VLMs capable of effectively utilizing cross-source information for reasoning. 

---
# Imagine to Hear: Auditory Knowledge Generation can be an Effective Assistant for Language Models 

**Authors**: Suho Yoo, Hyunjong Ok, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.16853)  

**Abstract**: Language models pretrained on text-only corpora often struggle with tasks that require auditory commonsense knowledge. Previous work addresses this problem by augmenting the language model to retrieve knowledge from external audio databases. This approach has several limitations, such as the potential lack of relevant audio in databases and the high costs associated with constructing and querying the databases. To address these issues, we propose Imagine to Hear, a novel approach that dynamically generates auditory knowledge using generative models. Our framework detects multiple audio-related textual spans from the given prompt and generates corresponding auditory knowledge. We develop several mechanisms to efficiently process multiple auditory knowledge, including a CLAP-based rejection sampler and a language-audio fusion module. Our experiments show that our method achieves state-of-the-art performance on AuditoryBench without relying on external databases, highlighting the effectiveness of our generation-based approach. 

---
# When Tom Eats Kimchi: Evaluating Cultural Bias of Multimodal Large Language Models in Cultural Mixture Contexts 

**Authors**: Jun Seong Kim, Kyaw Ye Thu, Javad Ismayilzada, Junyeong Park, Eunsu Kim, Huzama Ahmad, Na Min An, James Thorne, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2503.16826)  

**Abstract**: In a highly globalized world, it is important for multi-modal large language models (MLLMs) to recognize and respond correctly to mixed-cultural inputs. For example, a model should correctly identify kimchi (Korean food) in an image both when an Asian woman is eating it, as well as an African man is eating it. However, current MLLMs show an over-reliance on the visual features of the person, leading to misclassification of the entities. To examine the robustness of MLLMs to different ethnicity, we introduce MixCuBe, a cross-cultural bias benchmark, and study elements from five countries and four ethnicities. Our findings reveal that MLLMs achieve both higher accuracy and lower sensitivity to such perturbation for high-resource cultures, but not for low-resource cultures. GPT-4o, the best-performing model overall, shows up to 58% difference in accuracy between the original and perturbed cultural settings in low-resource cultures. Our dataset is publicly available at: this https URL. 

---
# Conversational User-AI Intervention: A Study on Prompt Rewriting for Improved LLM Response Generation 

**Authors**: Rupak Sarkar, Bahareh Sarrafzadeh, Nirupama Chandrasekaran, Nagu Rangan, Philip Resnik, Longqi Yang, Sujay Kumar Jauhar  

**Link**: [PDF](https://arxiv.org/pdf/2503.16789)  

**Abstract**: Human-LLM conversations are increasingly becoming more pervasive in peoples' professional and personal lives, yet many users still struggle to elicit helpful responses from LLM Chatbots. One of the reasons for this issue is users' lack of understanding in crafting effective prompts that accurately convey their information needs. Meanwhile, the existence of real-world conversational datasets on the one hand, and the text understanding faculties of LLMs on the other, present a unique opportunity to study this problem, and its potential solutions at scale. Thus, in this paper we present the first LLM-centric study of real human-AI chatbot conversations, focused on investigating aspects in which user queries fall short of expressing information needs, and the potential of using LLMs to rewrite suboptimal user prompts. Our findings demonstrate that rephrasing ineffective prompts can elicit better responses from a conversational system, while preserving the user's original intent. Notably, the performance of rewrites improves in longer conversations, where contextual inferences about user needs can be made more accurately. Additionally, we observe that LLMs often need to -- and inherently do -- make \emph{plausible} assumptions about a user's intentions and goals when interpreting prompts. Our findings largely hold true across conversational domains, user intents, and LLMs of varying sizes and families, indicating the promise of using prompt rewriting as a solution for better human-AI interactions. 

---
# Chain-of-Tools: Utilizing Massive Unseen Tools in the CoT Reasoning of Frozen Language Models 

**Authors**: Mengsong Wu, Tong Zhu, Han Han, Xiang Zhang, Wenbiao Shao, Wenliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16779)  

**Abstract**: Tool learning can further broaden the usage scenarios of large language models (LLMs). However most of the existing methods either need to finetune that the model can only use tools seen in the training data, or add tool demonstrations into the prompt with lower efficiency. In this paper, we present a new Tool Learning method Chain-of-Tools. It makes full use of the powerful semantic representation capability of frozen LLMs to finish tool calling in CoT reasoning with a huge and flexible tool pool which may contain unseen tools. Especially, to validate the effectiveness of our approach in the massive unseen tool scenario, we construct a new dataset SimpleToolQuestions. We conduct experiments on two numerical reasoning benchmarks (GSM8K-XL and FuncQA) and two knowledge-based question answering benchmarks (KAMEL and SimpleToolQuestions). Experimental results show that our approach performs better than the baseline. We also identify dimensions of the model output that are critical in tool selection, enhancing the model interpretability. Our code and data are available at: this https URL . 

---
# SPACER: A Parallel Dataset of Speech Production And Comprehension of Error Repairs 

**Authors**: Shiva Upadhye, Jiaxuan Li, Richard Futrell  

**Link**: [PDF](https://arxiv.org/pdf/2503.16745)  

**Abstract**: Speech errors are a natural part of communication, yet they rarely lead to complete communicative failure because both speakers and comprehenders can detect and correct errors. Although prior research has examined error monitoring and correction in production and comprehension separately, integrated investigation of both systems has been impeded by the scarcity of parallel data. In this study, we present SPACER, a parallel dataset that captures how naturalistic speech errors are corrected by both speakers and comprehenders. We focus on single-word substitution errors extracted from the Switchboard corpus, accompanied by speaker's self-repairs and comprehenders' responses from an offline text-editing experiment. Our exploratory analysis suggests asymmetries in error correction strategies: speakers are more likely to repair errors that introduce greater semantic and phonemic deviations, whereas comprehenders tend to correct errors that are phonemically similar to more plausible alternatives or do not fit into prior contexts. Our dataset enables future research on integrated approaches toward studying language production and comprehension. 

---
# Natural Language Generation 

**Authors**: Emiel van Miltenburg, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.16728)  

**Abstract**: This article provides a brief overview of the field of Natural Language Generation. The term Natural Language Generation (NLG), in its broadest definition, refers to the study of systems that verbalize some form of information through natural language. That information could be stored in a large database or knowledge graph (in data-to-text applications), but NLG researchers may also study summarisation (text-to-text) or image captioning (image-to-text), for example. As a subfield of Natural Language Processing, NLG is closely related to other sub-disciplines such as Machine Translation (MT) and Dialog Systems. Some NLG researchers exclude MT from their definition of the field, since there is no content selection involved where the system has to determine what to say. Conversely, dialog systems do not typically fall under the header of Natural Language Generation since NLG is just one component of dialog systems (the others being Natural Language Understanding and Dialog Management). However, with the rise of Large Language Models (LLMs), different subfields of Natural Language Processing have converged on similar methodologies for the production of natural language and the evaluation of automatically generated text. 

---
# Through the LLM Looking Glass: A Socratic Self-Assessment of Donkeys, Elephants, and Markets 

**Authors**: Molly Kennedy, Ayyoob Imani, Timo Spinde, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2503.16674)  

**Abstract**: While detecting and avoiding bias in LLM-generated text is becoming increasingly important, media bias often remains subtle and subjective, making it particularly difficult to identify and mitigate. In this study, we assess media bias in LLM-generated content and LLMs' ability to detect subtle ideological bias. We conduct this evaluation using two datasets, PoliGen and EconoLex, covering political and economic discourse, respectively. We evaluate eight widely used LLMs by prompting them to generate articles and analyze their ideological preferences via self-assessment. By using self-assessment, the study aims to directly measure the models' biases rather than relying on external interpretations, thereby minimizing subjective judgments about media bias. Our results reveal a consistent preference of Democratic over Republican positions across all models. Conversely, in economic topics, biases vary among Western LLMs, while those developed in China lean more strongly toward socialism. 

---
# Accelerating Antibiotic Discovery with Large Language Models and Knowledge Graphs 

**Authors**: Maxime Delmas, Magdalena Wysocka, Danilo Gusicuma, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2503.16655)  

**Abstract**: The discovery of novel antibiotics is critical to address the growing antimicrobial resistance (AMR). However, pharmaceutical industries face high costs (over $1 billion), long timelines, and a high failure rate, worsened by the rediscovery of known compounds. We propose an LLM-based pipeline that acts as an alarm system, detecting prior evidence of antibiotic activity to prevent costly rediscoveries. The system integrates organism and chemical literature into a Knowledge Graph (KG), ensuring taxonomic resolution, synonym handling, and multi-level evidence classification. We tested the pipeline on a private list of 73 potential antibiotic-producing organisms, disclosing 12 negative hits for evaluation. The results highlight the effectiveness of the pipeline for evidence reviewing, reducing false negatives, and accelerating decision-making. The KG for negative hits and the user interface for interactive exploration will be made publicly available. 

---
# Leveraging Large Language Models for Explainable Activity Recognition in Smart Homes: A Critical Evaluation 

**Authors**: Michele Fiori, Gabriele Civitarese, Priyankar Choudhary, Claudio Bettini  

**Link**: [PDF](https://arxiv.org/pdf/2503.16622)  

**Abstract**: Explainable Artificial Intelligence (XAI) aims to uncover the inner reasoning of machine learning models. In IoT systems, XAI improves the transparency of models processing sensor data from multiple heterogeneous devices, ensuring end-users understand and trust their outputs. Among the many applications, XAI has also been applied to sensor-based Activities of Daily Living (ADLs) recognition in smart homes. Existing approaches highlight which sensor events are most important for each predicted activity, using simple rules to convert these events into natural language explanations for non-expert users. However, these methods produce rigid explanations lacking natural language flexibility and are not scalable. With the recent rise of Large Language Models (LLMs), it is worth exploring whether they can enhance explanation generation, considering their proven knowledge of human activities. This paper investigates potential approaches to combine XAI and LLMs for sensor-based ADL recognition. We evaluate if LLMs can be used: a) as explainable zero-shot ADL recognition models, avoiding costly labeled data collection, and b) to automate the generation of explanations for existing data-driven XAI approaches when training data is available and the goal is higher recognition rates. Our critical evaluation provides insights into the benefits and challenges of using LLMs for explainable ADL recognition. 

---
# Classification of User Reports for Detection of Faulty Computer Components using NLP Models: A Case Study 

**Authors**: Maria de Lourdes M. Silva, André L. C. Mendonça, Eduardo R. D. Neto, Iago C. Chaves, Felipe T. Brito, Victor A. E. Farias, Javam C. Machado  

**Link**: [PDF](https://arxiv.org/pdf/2503.16614)  

**Abstract**: Computer manufacturers typically offer platforms for users to report faults. However, there remains a significant gap in these platforms' ability to effectively utilize textual reports, which impedes users from describing their issues in their own words. In this context, Natural Language Processing (NLP) offers a promising solution, by enabling the analysis of user-generated text. This paper presents an innovative approach that employs NLP models to classify user reports for detecting faulty computer components, such as CPU, memory, motherboard, video card, and more. In this work, we build a dataset of 341 user reports obtained from many sources. Additionally, through extensive experimental evaluation, our approach achieved an accuracy of 79% with our dataset. 

---
# Distributed LLMs and Multimodal Large Language Models: A Survey on Advances, Challenges, and Future Directions 

**Authors**: Hadi Amini, Md Jueal Mia, Yasaman Saadati, Ahmed Imteaj, Seyedsina Nabavirazavi, Urmish Thakker, Md Zarif Hossain, Awal Ahmed Fime, S.S. Iyengar  

**Link**: [PDF](https://arxiv.org/pdf/2503.16585)  

**Abstract**: Language models (LMs) are machine learning models designed to predict linguistic patterns by estimating the probability of word sequences based on large-scale datasets, such as text. LMs have a wide range of applications in natural language processing (NLP) tasks, including autocomplete and machine translation. Although larger datasets typically enhance LM performance, scalability remains a challenge due to constraints in computational power and resources. Distributed computing strategies offer essential solutions for improving scalability and managing the growing computational demand. Further, the use of sensitive datasets in training and deployment raises significant privacy concerns. Recent research has focused on developing decentralized techniques to enable distributed training and inference while utilizing diverse computational resources and enabling edge AI. This paper presents a survey on distributed solutions for various LMs, including large language models (LLMs), vision language models (VLMs), multimodal LLMs (MLLMs), and small language models (SLMs). While LLMs focus on processing and generating text, MLLMs are designed to handle multiple modalities of data (e.g., text, images, and audio) and to integrate them for broader applications. To this end, this paper reviews key advancements across the MLLM pipeline, including distributed training, inference, fine-tuning, and deployment, while also identifying the contributions, limitations, and future areas of improvement. Further, it categorizes the literature based on six primary focus areas of decentralization. Our analysis describes gaps in current methodologies for enabling distributed solutions for LMs and outline future research directions, emphasizing the need for novel solutions to enhance the robustness and applicability of distributed LMs. 

---
# Investigating Retrieval-Augmented Generation in Quranic Studies: A Study of 13 Open-Source Large Language Models 

**Authors**: Zahra Khalila, Arbi Haza Nasution, Winda Monika, Aytug Onan, Yohei Murakami, Yasir Bin Ismail Radi, Noor Mohammad Osmani  

**Link**: [PDF](https://arxiv.org/pdf/2503.16581)  

**Abstract**: Accurate and contextually faithful responses are critical when applying large language models (LLMs) to sensitive and domain-specific tasks, such as answering queries related to quranic studies. General-purpose LLMs often struggle with hallucinations, where generated responses deviate from authoritative sources, raising concerns about their reliability in religious contexts. This challenge highlights the need for systems that can integrate domain-specific knowledge while maintaining response accuracy, relevance, and faithfulness. In this study, we investigate 13 open-source LLMs categorized into large (e.g., Llama3:70b, Gemma2:27b, QwQ:32b), medium (e.g., Gemma2:9b, Llama3:8b), and small (e.g., Llama3.2:3b, Phi3:3.8b). A Retrieval-Augmented Generation (RAG) is used to make up for the problems that come with using separate models. This research utilizes a descriptive dataset of Quranic surahs including the meanings, historical context, and qualities of the 114 surahs, allowing the model to gather relevant knowledge before responding. The models are evaluated using three key metrics set by human evaluators: context relevance, answer faithfulness, and answer relevance. The findings reveal that large models consistently outperform smaller models in capturing query semantics and producing accurate, contextually grounded responses. The Llama3.2:3b model, even though it is considered small, does very well on faithfulness (4.619) and relevance (4.857), showing the promise of smaller architectures that have been well optimized. This article examines the trade-offs between model size, computational efficiency, and response quality while using LLMs in domain-specific applications. 

---
# SeniorTalk: A Chinese Conversation Dataset with Rich Annotations for Super-Aged Seniors 

**Authors**: Yang Chen, Hui Wang, Shiyao Wang, Junyang Chen, Jiabei He, Jiaming Zhou, Xi Yang, Yequan Wang, Yonghua Lin, Yong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.16578)  

**Abstract**: While voice technologies increasingly serve aging populations, current systems exhibit significant performance gaps due to inadequate training data capturing elderly-specific vocal characteristics like presbyphonia and dialectal variations. The limited data available on super-aged individuals in existing elderly speech datasets, coupled with overly simple recording styles and annotation dimensions, exacerbates this issue. To address the critical scarcity of speech data from individuals aged 75 and above, we introduce SeniorTalk, a carefully annotated Chinese spoken dialogue dataset. This dataset contains 55.53 hours of speech from 101 natural conversations involving 202 participants, ensuring a strategic balance across gender, region, and age. Through detailed annotation across multiple dimensions, it can support a wide range of speech tasks. We perform extensive experiments on speaker verification, speaker diarization, speech recognition, and speech editing tasks, offering crucial insights for the development of speech technologies targeting this age group. 

---
# Extract, Match, and Score: An Evaluation Paradigm for Long Question-context-answer Triplets in Financial Analysis 

**Authors**: Bo Hu, Han Yuan, Vlad Pandelea, Wuqiong Luo, Yingzhu Zhao, Zheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.16575)  

**Abstract**: The rapid advancement of large language models (LLMs) has sparked widespread adoption across diverse applications, making robust evaluation frameworks crucial for assessing their performance. While conventional evaluation metrics remain applicable for shorter texts, their efficacy diminishes when evaluating the quality of long-form answers. This limitation is particularly critical in real-world scenarios involving extended questions, extensive context, and long-form answers, such as financial analysis or regulatory compliance. In this paper, we use a practical financial use case to illustrate applications that handle "long question-context-answer triplets". We construct a real-world financial dataset comprising long triplets and demonstrate the inadequacies of traditional metrics. To address this, we propose an effective Extract, Match, and Score (EMS) evaluation approach tailored to the complexities of long-form LLMs' outputs, providing practitioners with a reliable methodology for assessing LLMs' performance in complex real-world scenarios. 

---
# FutureGen: LLM-RAG Approach to Generate the Future Work of Scientific Article 

**Authors**: Ibrahim Al Azher, Miftahul Jannat Mokarrama, Zhishuai Guo, Sagnik Ray Choudhury, Hamed Alhoori  

**Link**: [PDF](https://arxiv.org/pdf/2503.16561)  

**Abstract**: The future work section of a scientific article outlines potential research directions by identifying gaps and limitations of a current study. This section serves as a valuable resource for early-career researchers seeking unexplored areas and experienced researchers looking for new projects or collaborations. In this study, we generate future work suggestions from key sections of a scientific article alongside related papers and analyze how the trends have evolved. We experimented with various Large Language Models (LLMs) and integrated Retrieval-Augmented Generation (RAG) to enhance the generation process. We incorporate a LLM feedback mechanism to improve the quality of the generated content and propose an LLM-as-a-judge approach for evaluation. Our results demonstrated that the RAG-based approach with LLM feedback outperforms other methods evaluated through qualitative and quantitative metrics. Moreover, we conduct a human evaluation to assess the LLM as an extractor and judge. The code and dataset for this project are here, code: HuggingFace 

---
# Explainable AI Components for Narrative Map Extraction 

**Authors**: Brian Keith, Fausto German, Eric Krokos, Sarah Joseph, Chris North  

**Link**: [PDF](https://arxiv.org/pdf/2503.16554)  

**Abstract**: As narrative extraction systems grow in complexity, establishing user trust through interpretable and explainable outputs becomes increasingly critical. This paper presents an evaluation of an Explainable Artificial Intelligence (XAI) system for narrative map extraction that provides meaningful explanations across multiple levels of abstraction. Our system integrates explanations based on topical clusters for low-level document relationships, connection explanations for event relationships, and high-level structure explanations for overall narrative patterns. In particular, we evaluate the XAI system through a user study involving 10 participants that examined narratives from the 2021 Cuban protests. The analysis of results demonstrates that participants using the explanations made the users trust in the system's decisions, with connection explanations and important event detection proving particularly effective at building user confidence. Survey responses indicate that the multi-level explanation approach helped users develop appropriate trust in the system's narrative extraction capabilities. This work advances the state-of-the-art in explainable narrative extraction while providing practical insights for developing reliable narrative extraction systems that support effective human-AI collaboration. 

---
# A Foundational individual Mobility Prediction Model based on Open-Source Large Language Models 

**Authors**: Zhenlin Qin, Leizhen Wang, Francisco Camara Pereira, Zhenlinag Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.16553)  

**Abstract**: Large Language Models (LLMs) are widely applied to domain-specific tasks due to their massive general knowledge and remarkable inference capacities. Current studies on LLMs have shown immense potential in applying LLMs to model individual mobility prediction problems. However, most LLM-based mobility prediction models only train on specific datasets or use single well-designed prompts, leading to difficulty in adapting to different cities and users with diverse contexts. To fill these gaps, this paper proposes a unified fine-tuning framework to train a foundational open source LLM-based mobility prediction model. We conducted extensive experiments on six real-world mobility datasets to validate the proposed model. The results showed that the proposed model achieved the best performance in prediction accuracy and transferability over state-of-the-art models based on deep learning and LLMs. 

---
# Unified Enhancement of the Generalization and Robustness of Language Models via Bi-Stage Optimization 

**Authors**: Yudao Sun, Juan Yin, Juan Zhao, Fan Zhang, Yongheng Liu, Hongji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16550)  

**Abstract**: Neural network language models (LMs) are confronted with significant challenges in generalization and robustness. Currently, many studies focus on improving either generalization or robustness in isolation, without methods addressing both aspects simultaneously, which presents a significant challenge in developing LMs that are both robust and generalized. In this paper, we propose a bi-stage optimization framework to uniformly enhance both the generalization and robustness of LMs, termed UEGR. Specifically, during the forward propagation stage, we enrich the output probability distributions of adversarial samples by adaptive dropout to generate diverse sub models, and incorporate JS divergence and adversarial losses of these output distributions to reinforce output stability. During backward propagation stage, we compute parameter saliency scores and selectively update only the most critical parameters to minimize unnecessary deviations and consolidate the model's resilience. Theoretical analysis shows that our framework includes gradient regularization to limit the model's sensitivity to input perturbations and selective parameter updates to flatten the loss landscape, thus improving both generalization and robustness. The experimental results show that our method significantly improves the generalization and robustness of LMs compared to other existing methods across 13 publicly available language datasets, achieving state-of-the-art (SOTA) performance. 

---
# Causal Discovery and Counterfactual Reasoning to Optimize Persuasive Dialogue Policies 

**Authors**: Donghuo Zeng, Roberto Legaspi, Yuewen Sun, Xinshuai Dong, Kazushi Ikeda, Peter Spirtes, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16544)  

**Abstract**: Tailoring persuasive conversations to users leads to more effective persuasion. However, existing dialogue systems often struggle to adapt to dynamically evolving user states. This paper presents a novel method that leverages causal discovery and counterfactual reasoning for optimizing system persuasion capability and outcomes. We employ the Greedy Relaxation of the Sparsest Permutation (GRaSP) algorithm to identify causal relationships between user and system utterance strategies, treating user strategies as states and system strategies as actions. GRaSP identifies user strategies as causal factors influencing system responses, which inform Bidirectional Conditional Generative Adversarial Networks (BiCoGAN) in generating counterfactual utterances for the system. Subsequently, we use the Dueling Double Deep Q-Network (D3QN) model to utilize counterfactual data to determine the best policy for selecting system utterances. Our experiments with the PersuasionForGood dataset show measurable improvements in persuasion outcomes using our approach over baseline methods. The observed increase in cumulative rewards and Q-values highlights the effectiveness of causal discovery in enhancing counterfactual reasoning and optimizing reinforcement learning policies for online dialogue systems. 

---
# Poly-FEVER: A Multilingual Fact Verification Benchmark for Hallucination Detection in Large Language Models 

**Authors**: Hanzhi Zhang, Sumera Anjum, Heng Fan, Weijian Zheng, Yan Huang, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16541)  

**Abstract**: Hallucinations in generative AI, particularly in Large Language Models (LLMs), pose a significant challenge to the reliability of multilingual applications. Existing benchmarks for hallucination detection focus primarily on English and a few widely spoken languages, lacking the breadth to assess inconsistencies in model performance across diverse linguistic contexts. To address this gap, we introduce Poly-FEVER, a large-scale multilingual fact verification benchmark specifically designed for evaluating hallucination detection in LLMs. Poly-FEVER comprises 77,973 labeled factual claims spanning 11 languages, sourced from FEVER, Climate-FEVER, and SciFact. It provides the first large-scale dataset tailored for analyzing hallucination patterns across languages, enabling systematic evaluation of LLMs such as ChatGPT and the LLaMA series. Our analysis reveals how topic distribution and web resource availability influence hallucination frequency, uncovering language-specific biases that impact model accuracy. By offering a multilingual benchmark for fact verification, Poly-FEVER facilitates cross-linguistic comparisons of hallucination detection and contributes to the development of more reliable, language-inclusive AI systems. The dataset is publicly available to advance research in responsible AI, fact-checking methodologies, and multilingual NLP, promoting greater transparency and robustness in LLM performance. The proposed Poly-FEVER is available at: this https URL. 

---
# Do Multimodal Large Language Models Understand Welding? 

**Authors**: Grigorii Khvatskii, Yong Suk Lee, Corey Angst, Maria Gibbs, Robert Landers, Nitesh V. Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2503.16537)  

**Abstract**: This paper examines the performance of Multimodal LLMs (MLLMs) in skilled production work, with a focus on welding. Using a novel data set of real-world and online weld images, annotated by a domain expert, we evaluate the performance of two state-of-the-art MLLMs in assessing weld acceptability across three contexts: RV \& Marine, Aeronautical, and Farming. While both models perform better on online images, likely due to prior exposure or memorization, they also perform relatively well on unseen, real-world weld images. Additionally, we introduce WeldPrompt, a prompting strategy that combines Chain-of-Thought generation with in-context learning to mitigate hallucinations and improve reasoning. WeldPrompt improves model recall in certain contexts but exhibits inconsistent performance across others. These results underscore the limitations and potentials of MLLMs in high-stakes technical domains and highlight the importance of fine-tuning, domain-specific data, and more sophisticated prompting strategies to improve model reliability. The study opens avenues for further research into multimodal learning in industry applications. 

---
# Word2Minecraft: Generating 3D Game Levels through Large Language Models 

**Authors**: Shuo Huang, Muhammad Umair Nasir, Steven James, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2503.16536)  

**Abstract**: We present Word2Minecraft, a system that leverages large language models to generate playable game levels in Minecraft based on structured stories. The system transforms narrative elements-such as protagonist goals, antagonist challenges, and environmental settings-into game levels with both spatial and gameplay constraints. We introduce a flexible framework that allows for the customization of story complexity, enabling dynamic level generation. The system employs a scaling algorithm to maintain spatial consistency while adapting key game elements. We evaluate Word2Minecraft using both metric-based and human-based methods. Our results show that GPT-4-Turbo outperforms GPT-4o-Mini in most areas, including story coherence and objective enjoyment, while the latter excels in aesthetic appeal. We also demonstrate the system' s ability to generate levels with high map enjoyment, offering a promising step forward in the intersection of story generation and game design. We open-source the code at this https URL 

---
# Gender and content bias in Large Language Models: a case study on Google Gemini 2.0 Flash Experimental 

**Authors**: Roberto Balestri  

**Link**: [PDF](https://arxiv.org/pdf/2503.16534)  

**Abstract**: This study evaluates the biases in Gemini 2.0 Flash Experimental, a state-of-the-art large language model (LLM) developed by Google, focusing on content moderation and gender disparities. By comparing its performance to ChatGPT-4o, examined in a previous work of the author, the analysis highlights some differences in ethical moderation practices. Gemini 2.0 demonstrates reduced gender bias, notably with female-specific prompts achieving a substantial rise in acceptance rates compared to results obtained by ChatGPT-4o. It adopts a more permissive stance toward sexual content and maintains relatively high acceptance rates for violent prompts, including gender-specific cases. Despite these changes, whether they constitute an improvement is debatable. While gender bias has been reduced, this reduction comes at the cost of permitting more violent content toward both males and females, potentially normalizing violence rather than mitigating harm. Male-specific prompts still generally receive higher acceptance rates than female-specific ones. These findings underscore the complexities of aligning AI systems with ethical standards, highlighting progress in reducing certain biases while raising concerns about the broader implications of the model's permissiveness. Ongoing refinements are essential to achieve moderation practices that ensure transparency, fairness, and inclusivity without amplifying harmful content. 

---
# From Patient Consultations to Graphs: Leveraging LLMs for Patient Journey Knowledge Graph Construction 

**Authors**: Hassan S. Al Khatib, Sudip Mittal, Shahram Rahimi, Nina Marhamati, Sean Bozorgzad  

**Link**: [PDF](https://arxiv.org/pdf/2503.16533)  

**Abstract**: The transition towards patient-centric healthcare necessitates a comprehensive understanding of patient journeys, which encompass all healthcare experiences and interactions across the care spectrum. Existing healthcare data systems are often fragmented and lack a holistic representation of patient trajectories, creating challenges for coordinated care and personalized interventions. Patient Journey Knowledge Graphs (PJKGs) represent a novel approach to addressing the challenge of fragmented healthcare data by integrating diverse patient information into a unified, structured representation. This paper presents a methodology for constructing PJKGs using Large Language Models (LLMs) to process and structure both formal clinical documentation and unstructured patient-provider conversations. These graphs encapsulate temporal and causal relationships among clinical encounters, diagnoses, treatments, and outcomes, enabling advanced temporal reasoning and personalized care insights. The research evaluates four different LLMs, such as Claude 3.5, Mistral, Llama 3.1, and Chatgpt4o, in their ability to generate accurate and computationally efficient knowledge graphs. Results demonstrate that while all models achieved perfect structural compliance, they exhibited variations in medical entity processing and computational efficiency. The paper concludes by identifying key challenges and future research directions. This work contributes to advancing patient-centric healthcare through the development of comprehensive, actionable knowledge graphs that support improved care coordination and outcome prediction. 

---
# EEG-CLIP : Learning EEG representations from natural language descriptions 

**Authors**: Tidiane Camaret N'dir, Robin Tibor Schirrmeister  

**Link**: [PDF](https://arxiv.org/pdf/2503.16531)  

**Abstract**: Deep networks for electroencephalogram (EEG) decoding are currently often trained to only solve a specific task like pathology or gender decoding. A more general approach leveraging the medical reports of clinical EEG recordings is to learn mappings between medical reports and EEG recordings. This approach was pioneered in the computer vision domain matching images and their text captions and subsequently allowed to do successful zero-shot decoding using textual class prompts. In this work, we follow this approach and develop a contrastive learning framework EEG-CLIP that aligns EEG time series and their corresponding clinical text descriptions in a shared embedding space. We investigate its potential for versatile EEG decoding, assessing performance on a range of few-shot and zero-shot settings. Overall, results show that EEG-CLIP manages to nontrivially align text and EEG representations. Our work presents a promising approach to learn general EEG representations, which could enable easier analyses of diverse decoding questions through zero shot decoding or training task-specific models from fewer training examples. The code for reproducing our results is available at this https URL. 

---
# Enhancing LLM Generation with Knowledge Hypergraph for Evidence-Based Medicine 

**Authors**: Chengfeng Dou, Ying Zhang, Zhi Jin, Wenpin Jiao, Haiyan Zhao, Yongqiang Zhao, Zhengwei Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16530)  

**Abstract**: Evidence-based medicine (EBM) plays a crucial role in the application of large language models (LLMs) in healthcare, as it provides reliable support for medical decision-making processes. Although it benefits from current retrieval-augmented generation~(RAG) technologies, it still faces two significant challenges: the collection of dispersed evidence and the efficient organization of this evidence to support the complex queries necessary for EBM. To tackle these issues, we propose using LLMs to gather scattered evidence from multiple sources and present a knowledge hypergraph-based evidence management model to integrate these evidence while capturing intricate relationships. Furthermore, to better support complex queries, we have developed an Importance-Driven Evidence Prioritization (IDEP) algorithm that utilizes the LLM to generate multiple evidence features, each with an associated importance score, which are then used to rank the evidence and produce the final retrieval results. Experimental results from six datasets demonstrate that our approach outperforms existing RAG techniques in application domains of interest to EBM, such as medical quizzing, hallucination detection, and decision support. Testsets and the constructed knowledge graph can be accessed at \href{this https URL}{this https URL}. 

---
# Safety Evaluation and Enhancement of DeepSeek Models in Chinese Contexts 

**Authors**: Wenjing Zhang, Xuejiao Lei, Zhaoxiang Liu, Limin Han, Jiaojiao Zhao, Beibei Huang, Zhenhong Long, Junting Guo, Meijuan An, Rongjia Du, Ning Wang, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2503.16529)  

**Abstract**: DeepSeek-R1, renowned for its exceptional reasoning capabilities and open-source strategy, is significantly influencing the global artificial intelligence landscape. However, it exhibits notable safety shortcomings. Recent research conducted by Robust Intelligence, a subsidiary of Cisco, in collaboration with the University of Pennsylvania, revealed that DeepSeek-R1 achieves a 100\% attack success rate when processing harmful prompts. Furthermore, multiple security firms and research institutions have identified critical security vulnerabilities within the model. Although China Unicom has uncovered safety vulnerabilities of R1 in Chinese contexts, the safety capabilities of the remaining distilled models in the R1 series have not yet been comprehensively evaluated. To address this gap, this study utilizes the comprehensive Chinese safety benchmark CHiSafetyBench to conduct an in-depth safety evaluation of the DeepSeek-R1 series distilled models. The objective is to assess the safety capabilities of these models in Chinese contexts both before and after distillation, and to further elucidate the adverse effects of distillation on model safety. Building on these findings, we implement targeted safety enhancements for six distilled models. Evaluation results indicate that the enhanced models achieve significant improvements in safety while maintaining reasoning capabilities without notable degradation. We open-source the safety-enhanced models at this https URL to serve as a valuable resource for future research and optimization of DeepSeek models. 

---
# HDLCoRe: A Training-Free Framework for Mitigating Hallucinations in LLM-Generated HDL 

**Authors**: Heng Ping, Shixuan Li, Peiyu Zhang, Anzhe Cheng, Shukai Duan, Nikos Kanakaris, Xiongye Xiao, Wei Yang, Shahin Nazarian, Andrei Irimia, Paul Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16528)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable capabilities in code generation tasks. However, when applied to hardware description languages (HDL), these models exhibit significant limitations due to data scarcity, resulting in hallucinations and incorrect code generation. To address these challenges, we propose HDLCoRe, a training-free framework that enhances LLMs' HDL generation capabilities through prompt engineering techniques and retrieval-augmented generation (RAG). Our approach consists of two main components: (1) an HDL-aware Chain-of-Thought (CoT) prompting technique with self-verification that classifies tasks by complexity and type, incorporates domain-specific knowledge, and guides LLMs through step-by-step self-simulation for error correction; and (2) a two-stage heterogeneous RAG system that addresses formatting inconsistencies through key component extraction and efficiently retrieves relevant HDL examples through sequential filtering and re-ranking. HDLCoRe eliminates the need for model fine-tuning while substantially improving LLMs' HDL generation capabilities. Experimental results demonstrate that our framework achieves superior performance on the RTLLM2.0 benchmark, significantly reducing hallucinations and improving both syntactic and functional correctness. 

---
# LLM Generated Persona is a Promise with a Catch 

**Authors**: Ang Li, Haozhe Chen, Hongseok Namkoong, Tianyi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16527)  

**Abstract**: The use of large language models (LLMs) to simulate human behavior has gained significant attention, particularly through personas that approximate individual characteristics. Persona-based simulations hold promise for transforming disciplines that rely on population-level feedback, including social science, economic analysis, marketing research, and business operations. Traditional methods to collect realistic persona data face significant challenges. They are prohibitively expensive and logistically challenging due to privacy constraints, and often fail to capture multi-dimensional attributes, particularly subjective qualities. Consequently, synthetic persona generation with LLMs offers a scalable, cost-effective alternative. However, current approaches rely on ad hoc and heuristic generation techniques that do not guarantee methodological rigor or simulation precision, resulting in systematic biases in downstream tasks. Through extensive large-scale experiments including presidential election forecasts and general opinion surveys of the U.S. population, we reveal that these biases can lead to significant deviations from real-world outcomes. Our findings underscore the need to develop a rigorous science of persona generation and outline the methodological innovations, organizational and institutional support, and empirical foundations required to enhance the reliability and scalability of LLM-driven persona simulations. To support further research and development in this area, we have open-sourced approximately one million generated personas, available for public access and analysis at this https URL. 

---
# KVShare: Semantic-Aware Key-Value Cache Sharing for Efficient Large Language Model Inference 

**Authors**: Huan Yang, Renji Zhang, Deyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16525)  

**Abstract**: This paper presents KVShare, a multi-user Key-Value (KV) Cache sharing technology based on semantic similarity, designed to enhance the inference efficiency of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Addressing the limitations of existing prefix caching (strict text prefix matching) and semantic caching (loss of response diversity), KVShare achieves fine-grained KV cache reuse through semantic alignment algorithms and differential editing operations. Experiments on real-world user conversation datasets demonstrate that KVShare improves KV cache hit rates by over 60%, while maintaining output quality comparable to full computation (no significant degradation in BLEU and Rouge-L metrics). This approach effectively reduces GPU resource consumption and is applicable to scenarios with repetitive queries, such as healthcare and education. 

---
# Mind2: Mind-to-Mind Emotional Support System with Bidirectional Cognitive Discourse Analysis 

**Authors**: Shi Yin Hong, Uttamasha Oyshi, Quan Mai, Gibson Nkhata, Susan Gauch  

**Link**: [PDF](https://arxiv.org/pdf/2503.16523)  

**Abstract**: Emotional support (ES) systems alleviate users' mental distress by generating strategic supportive dialogues based on diverse user situations. However, ES systems are limited in their ability to generate effective ES dialogues that include timely context and interpretability, hindering them from earning public trust. Driven by cognitive models, we propose Mind-to-Mind (Mind2), an ES framework that approaches interpretable ES context modeling for the ES dialogue generation task from a discourse analysis perspective. Specifically, we perform cognitive discourse analysis on ES dialogues according to our dynamic discourse context propagation window, which accommodates evolving context as the conversation between the ES system and user progresses. To enhance interpretability, Mind2 prioritizes details that reflect each speaker's belief about the other speaker with bidirectionality, integrating Theory-of-Mind, physiological expected utility, and cognitive rationality to extract cognitive knowledge from ES conversations. Experimental results support that Mind2 achieves competitive performance versus state-of-the-art ES systems while trained with only 10\% of the available training data. 

---
# Not All Personas Are Worth It: Culture-Reflective Persona Data Augmentation 

**Authors**: Ji-Eun Han, Yoonseok Heo  

**Link**: [PDF](https://arxiv.org/pdf/2503.16520)  

**Abstract**: Incorporating personas into conversational AI models is crucial for achieving authentic and engaging interactions. However, the cultural diversity and adaptability of existing persona datasets is often overlooked, reducing their efficacy in building culturally aware AI systems. To address this issue, we propose a two-step pipeline for generating culture-specific personas and introduce KoPersona, a dataset comprising 200,000 personas designed to capture Korean cultural values, behaviors, and social nuances. A comprehensive evaluation through various metrics validates the quality of KoPersona and its relevance to Korean culture. This work not only contributes to persona-based research, but also establishes a scalable approach for creating culturally relevant personas adaptable to various languages and cultural contexts. 

---
# Using LLMs for Automated Privacy Policy Analysis: Prompt Engineering, Fine-Tuning and Explainability 

**Authors**: Yuxin Chen, Peng Tang, Weidong Qiu, Shujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16516)  

**Abstract**: Privacy policies are widely used by digital services and often required for legal purposes. Many machine learning based classifiers have been developed to automate detection of different concepts in a given privacy policy, which can help facilitate other automated tasks such as producing a more reader-friendly summary and detecting legal compliance issues. Despite the successful applications of large language models (LLMs) to many NLP tasks in various domains, there is very little work studying the use of LLMs for automated privacy policy analysis, therefore, if and how LLMs can help automate privacy policy analysis remains under-explored. To fill this research gap, we conducted a comprehensive evaluation of LLM-based privacy policy concept classifiers, employing both prompt engineering and LoRA (low-rank adaptation) fine-tuning, on four state-of-the-art (SOTA) privacy policy corpora and taxonomies. Our experimental results demonstrated that combining prompt engineering and fine-tuning can make LLM-based classifiers outperform other SOTA methods, \emph{significantly} and \emph{consistently} across privacy policy corpora/taxonomies and concepts. Furthermore, we evaluated the explainability of the LLM-based classifiers using three metrics: completeness, logicality, and comprehensibility. For all three metrics, a score exceeding 91.1\% was observed in our evaluation, indicating that LLMs are not only useful to improve the classification performance, but also to enhance the explainability of detection results. 

---
# Highlighting Case Studies in LLM Literature Review of Interdisciplinary System Science 

**Authors**: Lachlan McGinness, Peter Baumgartner  

**Link**: [PDF](https://arxiv.org/pdf/2503.16515)  

**Abstract**: Large Language Models (LLMs) were used to assist four Commonwealth Scientific and Industrial Research Organisation (CSIRO) researchers to perform systematic literature reviews (SLR). We evaluate the performance of LLMs for SLR tasks in these case studies. In each, we explore the impact of changing parameters on the accuracy of LLM responses. The LLM was tasked with extracting evidence from chosen academic papers to answer specific research questions. We evaluate the models' performance in faithfully reproducing quotes from the literature and subject experts were asked to assess the model performance in answering the research questions. We developed a semantic text highlighting tool to facilitate expert review of LLM responses.
We found that state of the art LLMs were able to reproduce quotes from texts with greater than 95% accuracy and answer research questions with an accuracy of approximately 83%. We use two methods to determine the correctness of LLM responses; expert review and the cosine similarity of transformer embeddings of LLM and expert answers. The correlation between these methods ranged from 0.48 to 0.77, providing evidence that the latter is a valid metric for measuring semantic similarity. 

---
# Medifact at PerAnsSumm 2025: Leveraging Lightweight Models for Perspective-Specific Summarization of Clinical Q&A Forums 

**Authors**: Nadia Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2503.16513)  

**Abstract**: The PerAnsSumm 2025 challenge focuses on perspective-aware healthcare answer summarization (Agarwal et al., 2025). This work proposes a few-shot learning framework using a Snorkel-BART-SVM pipeline for classifying and summarizing open-ended healthcare community question-answering (CQA). An SVM model is trained with weak supervision via Snorkel, enhancing zero-shot learning. Extractive classification identifies perspective-relevant sentences, which are then summarized using a pretrained BART-CNN model. The approach achieved 12th place among 100 teams in the shared task, demonstrating computational efficiency and contextual accuracy. By leveraging pretrained summarization models, this work advances medical CQA research and contributes to clinical decision support systems. 

---
# Token-Level Uncertainty-Aware Objective for Language Model Post-Training 

**Authors**: Tingkai Liu, Ari S. Benjamin, Anthony M. Zador  

**Link**: [PDF](https://arxiv.org/pdf/2503.16511)  

**Abstract**: In the current work, we connect token-level uncertainty in causal language modeling to two types of training objectives: 1) masked maximum likelihood (MLE), 2) self-distillation. We show that masked MLE is effective in reducing epistemic uncertainty, and serve as an effective token-level automatic curriculum learning technique. However, masked MLE is prone to overfitting and requires self-distillation regularization to improve or maintain performance on out-of-distribution tasks. We demonstrate significant performance gain via the proposed training objective - combined masked MLE and self-distillation - across multiple architectures (Gemma, LLaMA, Phi) and datasets (Alpaca, ShareGPT, GSM8K), mitigating overfitting while maintaining adaptability during post-training. Our findings suggest that uncertainty-aware training provides an effective mechanism for enhancing language model training. 

---
# OpenVLThinker: An Early Exploration to Complex Vision-Language Reasoning via Iterative Self-Improvement 

**Authors**: Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17352)  

**Abstract**: Recent advancements demonstrated by DeepSeek-R1 have shown that complex reasoning abilities in large language models (LLMs), including sophisticated behaviors such as self-verification and self-correction, can be achieved by RL with verifiable rewards and significantly improves model performance on challenging tasks such as AIME. Motivated by these findings, our study investigates whether similar reasoning capabilities can be successfully integrated into large vision-language models (LVLMs) and assesses their impact on challenging multimodal reasoning tasks. We consider an approach that iteratively leverages supervised fine-tuning (SFT) on lightweight training data and Reinforcement Learning (RL) to further improve model generalization. Initially, reasoning capabilities were distilled from pure-text R1 models by generating reasoning steps using high-quality captions of the images sourced from diverse visual datasets. Subsequently, iterative RL training further enhance reasoning skills, with each iteration's RL-improved model generating refined SFT datasets for the next round. This iterative process yielded OpenVLThinker, a LVLM exhibiting consistently improved reasoning performance on challenging benchmarks such as MathVista, MathVerse, and MathVision, demonstrating the potential of our strategy for robust vision-language reasoning. The code, model and data are held at this https URL. 

---
# An Iterative Feedback Mechanism for Improving Natural Language Class Descriptions in Open-Vocabulary Object Detection 

**Authors**: Louis Y. Kim, Michelle Karker, Victoria Valledor, Seiyoung C. Lee, Karl F. Brzoska, Margaret Duff, Anthony Palladino  

**Link**: [PDF](https://arxiv.org/pdf/2503.17285)  

**Abstract**: Recent advances in open-vocabulary object detection models will enable Automatic Target Recognition systems to be sustainable and repurposed by non-technical end-users for a variety of applications or missions. New, and potentially nuanced, classes can be defined with natural language text descriptions in the field, immediately before runtime, without needing to retrain the model. We present an approach for improving non-technical users' natural language text descriptions of their desired targets of interest, using a combination of analysis techniques on the text embeddings, and proper combinations of embeddings for contrastive examples. We quantify the improvement that our feedback mechanism provides by demonstrating performance with multiple publicly-available open-vocabulary object detection models. 

---
# FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs 

**Authors**: Albert Sawczyn, Jakub Binkowski, Denis Janiak, Bogdan Gabrys, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2503.17229)  

**Abstract**: Large Language Models (LLMs) frequently generate hallucinated content, posing significant challenges for applications where factuality is crucial. While existing hallucination detection methods typically operate at the sentence level or passage level, we propose FactSelfCheck, a novel black-box sampling-based method that enables fine-grained fact-level detection. Our approach represents text as knowledge graphs consisting of facts in the form of triples. Through analyzing factual consistency across multiple LLM responses, we compute fine-grained hallucination scores without requiring external resources or training data. Our evaluation demonstrates that FactSelfCheck performs competitively with leading sampling-based methods while providing more detailed insights. Most notably, our fact-level approach significantly improves hallucination correction, achieving a 35% increase in factual content compared to the baseline, while sentence-level SelfCheckGPT yields only an 8% improvement. The granular nature of our detection enables more precise identification and correction of hallucinated content. 

---
# Text2Model: Generating dynamic chemical reactor models using large language models (LLMs) 

**Authors**: Sophia Rupprecht, Yassine Hounat, Monisha Kumar, Giacomo Lastrucci, Artur M. Schweidtmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.17004)  

**Abstract**: As large language models have shown remarkable capabilities in conversing via natural language, the question arises as to how LLMs could potentially assist chemical engineers in research and industry with domain-specific tasks. We generate dynamic chemical reactor models in Modelica code format from textual descriptions as user input. We fine-tune Llama 3.1 8B Instruct on synthetically generated Modelica code for different reactor scenarios. We compare the performance of our fine-tuned model to the baseline Llama 3.1 8B Instruct model and GPT4o. We manually assess the models' predictions regarding the syntactic and semantic accuracy of the generated dynamic models. We find that considerable improvements are achieved by the fine-tuned model with respect to both the semantic and the syntactic accuracy of the Modelica models. However, the fine-tuned model lacks a satisfactory ability to generalize to unseen scenarios compared to GPT4o. 

---
# Token Dynamics: Towards Efficient and Dynamic Video Token Representation for Video Large Language Models 

**Authors**: Haichao Zhang, Zhuowei Li, Dimitris Metaxas, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16980)  

**Abstract**: Token-based video representation has emerged as a promising approach for enabling large language models to interpret video content. However, existing token reduction techniques, such as token pruning and token merging, often disrupt essential spatial-temporal positional embeddings, failing to adequately balance computational efficiency with fewer tokens. Consequently, these methods result in relatively lengthy token sequences, limiting their applicability in scenarios requiring extreme token compression, such as video large language models. In this paper, we introduce the novel task of extreme short token reduction, aiming to represent extensive video sequences with a minimal number of tokens. To address this challenge, we propose Token Dynamics, a new video representation framework that dynamically reduces token count while preserving spatial-temporal coherence. Specifically, we disentangle video representations by separating visual embeddings from grid-level motion information, structuring them into: 1. a concise token base, created by clustering tokens that describe object-level content; 2. a token dynamics map, capturing detailed spatial-temporal motion patterns across grids. Furthermore, we introduce a cross-dynamics attention mechanism that integrates motion features into the token base without increasing token length, thereby maintaining compactness and spatial-temporal integrity. The experiments demonstrate a reduction of token count to merely 0.07% of the original tokens, with only a minor performance drop of 1.13%. Additionally, we propose two novel subtasks within extreme token reduction (fixed-length and adaptive-length compression), both effectively representing long token sequences for video-language tasks. Our method offers significantly lower theoretical complexity, fewer tokens, and enhanced throughput, thus providing an efficient solution for video LLMs. 

---
# Assessing Consistency and Reproducibility in the Outputs of Large Language Models: Evidence Across Diverse Finance and Accounting Tasks 

**Authors**: Julian Junyan Wang, Victor Xiaoqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16974)  

**Abstract**: This study provides the first comprehensive assessment of consistency and reproducibility in Large Language Model (LLM) outputs in finance and accounting research. We evaluate how consistently LLMs produce outputs given identical inputs through extensive experimentation with 50 independent runs across five common tasks: classification, sentiment analysis, summarization, text generation, and prediction. Using three OpenAI models (GPT-3.5-turbo, GPT-4o-mini, and GPT-4o), we generate over 3.4 million outputs from diverse financial source texts and data, covering MD&As, FOMC statements, finance news articles, earnings call transcripts, and financial statements. Our findings reveal substantial but task-dependent consistency, with binary classification and sentiment analysis achieving near-perfect reproducibility, while complex tasks show greater variability. More advanced models do not consistently demonstrate better consistency and reproducibility, with task-specific patterns emerging. LLMs significantly outperform expert human annotators in consistency and maintain high agreement even where human experts significantly disagree. We further find that simple aggregation strategies across 3-5 runs dramatically improve consistency. Simulation analysis reveals that despite measurable inconsistency in LLM outputs, downstream statistical inferences remain remarkably robust. These findings address concerns about what we term "G-hacking," the selective reporting of favorable outcomes from multiple Generative AI runs, by demonstrating that such risks are relatively low for finance and accounting tasks. 

---
# Federated Cross-Domain Click-Through Rate Prediction With Large Language Model Augmentation 

**Authors**: Jiangcheng Qin, Xueyuan Zhang, Baisong Liu, Jiangbo Qian, Yangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16875)  

**Abstract**: Accurately predicting click-through rates (CTR) under stringent privacy constraints poses profound challenges, particularly when user-item interactions are sparse and fragmented across domains. Conventional cross-domain CTR (CCTR) methods frequently assume homogeneous feature spaces and rely on centralized data sharing, neglecting complex inter-domain discrepancies and the subtle trade-offs imposed by privacy-preserving protocols. Here, we present Federated Cross-Domain CTR Prediction with Large Language Model Augmentation (FedCCTR-LM), a federated framework engineered to address these limitations by synchronizing data augmentation, representation disentanglement, and adaptive privacy protection. Our approach integrates three core innovations. First, the Privacy-Preserving Augmentation Network (PrivAugNet) employs large language models to enrich user and item representations and expand interaction sequences, mitigating data sparsity and feature incompleteness. Second, the Independent Domain-Specific Transformer with Contrastive Learning (IDST-CL) module disentangles domain-specific and shared user preferences, employing intra-domain representation alignment (IDRA) and crossdomain representation disentanglement (CDRD) to refine the learned embeddings and enhance knowledge transfer across domains. Finally, the Adaptive Local Differential Privacy (AdaLDP) mechanism dynamically calibrates noise injection to achieve an optimal balance between rigorous privacy guarantees and predictive accuracy. Empirical evaluations on four real-world datasets demonstrate that FedCCTR-LM substantially outperforms existing baselines, offering robust, privacy-preserving, and generalizable cross-domain CTR prediction in heterogeneous, federated environments. 

---
# Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs 

**Authors**: Anshumann, Mohd Abbas Zaidi, Akhil Kedia, Jinwoo Ahn, Taehwak Kwon, Kangwook Lee, Haejun Lee, Joohyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.16870)  

**Abstract**: Knowledge distillation can be a cost-effective technique to distill knowledge in Large Language Models, if the teacher output logits can be pre-computed and cached. However, successfully applying this to pre-training remains largely unexplored. In this work, we prove that naive approaches for sparse knowledge distillation such as caching Top-K probabilities, while intuitive, provide biased estimates of teacher probability distribution to the student, resulting in suboptimal performance and calibration. We propose an importance-sampling-based method `Random Sampling Knowledge Distillation', which provides unbiased estimates, preserves the gradient in expectation, and requires storing significantly sparser logits. Our method enables faster training of student models with marginal overhead (<10%) compared to cross-entropy based training, while maintaining competitive performance compared to full distillation, across a range of model sizes from 300M to 3B. 

---
# Towards LLM Guardrails via Sparse Representation Steering 

**Authors**: Zeqing He, Zhibo Wang, Huiyu Xu, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.16851)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance in natural language generation tasks, yet their uncontrolled outputs pose significant ethical and safety risks. Recently, representation engineering methods have shown promising results in steering model behavior by modifying the rich semantic information encoded in activation vectors. However, due to the difficulty of precisely disentangling semantic directions within high-dimensional representation space, existing approaches suffer from three major limitations: lack of fine-grained control, quality degradation of generated content, and poor interpretability. To address these challenges, we propose a sparse encoding-based representation engineering method, named SRE, which decomposes polysemantic activations into a structured, monosemantic feature space. By leveraging sparse autoencoding, our approach isolates and adjusts only task-specific sparse feature dimensions, enabling precise and interpretable steering of model behavior while preserving content quality. We validate our method on three critical domains, i.e., safety, fairness, and truthfulness using the open-source LLM Gemma-2-2B-it. Experimental results show that SRE achieves superior controllability while maintaining the overall quality of generated content (i.e., controllability and quality), demonstrating its effectiveness as a fine-grained and interpretable activation steering framework. 

---
# The Deployment of End-to-End Audio Language Models Should Take into Account the Principle of Least Privilege 

**Authors**: Luxi He, Xiangyu Qi, Michel Liao, Inyoung Cheong, Prateek Mittal, Danqi Chen, Peter Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2503.16833)  

**Abstract**: We are at a turning point for language models that accept audio input. The latest end-to-end audio language models (Audio LMs) process speech directly instead of relying on a separate transcription step. This shift preserves detailed information, such as intonation or the presence of multiple speakers, that would otherwise be lost in transcription. However, it also introduces new safety risks, including the potential misuse of speaker identity cues and other sensitive vocal attributes, which could have legal implications. In this position paper, we urge a closer examination of how these models are built and deployed. We argue that the principle of least privilege should guide decisions on whether to deploy cascaded or end-to-end models. Specifically, evaluations should assess (1) whether end-to-end modeling is necessary for a given application; and (2), the appropriate scope of information access. Finally, We highlight related gaps in current audio LM benchmarks and identify key open research questions, both technical and policy-related, that must be addressed to enable the responsible deployment of end-to-end Audio LMs. 

---
# Design and Implementation of an FPGA-Based Tiled Matrix Multiplication Accelerator for Transformer Self-Attention on the Xilinx KV260 SoM 

**Authors**: Zhaoqin "Richie" Li, Sicheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16731)  

**Abstract**: Transformer-based LLMs spend most of their compute in large matrix multiplications for attention and feed-forward layers. Recognizing that the Q, K, and V linear projections within the Multi-Head Self-Attention (MHA) module represent a critical computational bottleneck, we strategically focused our efforts on accelerating these operations. We present a tiled matrix multiplication accelerator optimized for such workloads on a Xilinx KV260 on-board FPGA. Key innovations include persistent on-chip storage for one matrix operand, two-level tiling for data reuse, and a systolic-like unrolled compute engine. Implemented via high-level synthesis (HLS) and integrated with DistilBERT for Q, K, V projections, our accelerator achieves significant speedup and energy efficiency gains over CPU baselines. Standalone GEMM benchmarks show up to a 7x speedup over an ARM CPU (PyTorch) and ~200x over naive numpy, with a throughput of up to 3.1 GFLOPs on 768x3072 matrices. Although the overall end-to-end DistilBERT acceleration is more modest, our results validate the potential of FPGA-based acceleration for critical components of Transformer models. 

---
# CAARMA: Class Augmentation with Adversarial Mixup Regularization 

**Authors**: Massa Baali, Xiang Li, Hao Chen, Rita Singh, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2503.16718)  

**Abstract**: Speaker verification is a typical zero-shot learning task, where inference of unseen classes is performed by comparing embeddings of test instances to known examples. The models performing inference must hence naturally generate embeddings that cluster same-class instances compactly, while maintaining separation across classes. In order to learn to do so, they are typically trained on a large number of classes (speakers), often using specialized losses. However real-world speaker datasets often lack the class diversity needed to effectively learn this in a generalizable manner. We introduce CAARMA, a class augmentation framework that addresses this problem by generating synthetic classes through data mixing in the embedding space, expanding the number of training classes. To ensure the authenticity of the synthetic classes we adopt a novel adversarial refinement mechanism that minimizes categorical distinctions between synthetic and real classes. We evaluate CAARMA on multiple speaker verification tasks, as well as other representative zero-shot comparison-based speech analysis tasks and obtain consistent improvements: our framework demonstrates a significant improvement of 8\% over all baseline models. Code for CAARMA will be released. 

---
# WaveFM: A High-Fidelity and Efficient Vocoder Based on Flow Matching 

**Authors**: Tianze Luo, Xingchen Miao, Wenbo Duan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16689)  

**Abstract**: Flow matching offers a robust and stable approach to training diffusion models. However, directly applying flow matching to neural vocoders can result in subpar audio quality. In this work, we present WaveFM, a reparameterized flow matching model for mel-spectrogram conditioned speech synthesis, designed to enhance both sample quality and generation speed for diffusion vocoders. Since mel-spectrograms represent the energy distribution of waveforms, WaveFM adopts a mel-conditioned prior distribution instead of a standard Gaussian prior to minimize unnecessary transportation costs during synthesis. Moreover, while most diffusion vocoders rely on a single loss function, we argue that incorporating auxiliary losses, including a refined multi-resolution STFT loss, can further improve audio quality. To speed up inference without degrading sample quality significantly, we introduce a tailored consistency distillation method for WaveFM. Experiment results demonstrate that our model achieves superior performance in both quality and efficiency compared to previous diffusion vocoders, while enabling waveform generation in a single inference step. 

---
# Gene42: Long-Range Genomic Foundation Model With Dense Attention 

**Authors**: Kirill Vishniakov, Boulbaba Ben Amor, Engin Tekin, Nancy A. ElNaker, Karthik Viswanathan, Aleksandr Medvedev, Aahan Singh, Maryam Nadeem, Mohammad Amaan Sayeed, Praveenkumar Kanithi, Tiago Magalhaes, Natalia Vassilieva, Dwarikanath Mahapatra, Marco Pimentel, and Shadab Khan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16565)  

**Abstract**: We introduce Gene42, a novel family of Genomic Foundation Models (GFMs) designed to manage context lengths of up to 192,000 base pairs (bp) at a single-nucleotide resolution. Gene42 models utilize a decoder-only (LLaMA-style) architecture with a dense self-attention mechanism. Initially trained on fixed-length sequences of 4,096 bp, our models underwent continuous pretraining to extend the context length to 192,000 bp. This iterative extension allowed for the comprehensive processing of large-scale genomic data and the capture of intricate patterns and dependencies within the human genome. Gene42 is the first dense attention model capable of handling such extensive long context lengths in genomics, challenging state-space models that often rely on convolutional operators among other mechanisms. Our pretrained models exhibit notably low perplexity values and high reconstruction accuracy, highlighting their strong ability to model genomic data. Extensive experiments on various genomic benchmarks have demonstrated state-of-the-art performance across multiple tasks, including biotype classification, regulatory region identification, chromatin profiling prediction, variant pathogenicity prediction, and species classification. The models are publicly available at this http URL. 

---
# Chem42: a Family of chemical Language Models for Target-aware Ligand Generation 

**Authors**: Aahan Singh, Engin Tekin, Maryam Nadeem, Nancy A. ElNaker, Mohammad Amaan Sayeed, Natalia Vassilieva, Boulbaba Ben Amor  

**Link**: [PDF](https://arxiv.org/pdf/2503.16563)  

**Abstract**: Revolutionizing drug discovery demands more than just understanding molecular interactions - it requires generative models that can design novel ligands tailored to specific biological targets. While chemical Language Models (cLMs) have made strides in learning molecular properties, most fail to incorporate target-specific insights, restricting their ability to drive de-novo ligand generation. Chem42, a cutting-edge family of generative chemical Language Models, is designed to bridge this gap. By integrating atomic-level interactions with multimodal inputs from Prot42, a complementary protein Language Model, Chem42 achieves a sophisticated cross-modal representation of molecular structures, interactions, and binding patterns. This innovative framework enables the creation of structurally valid, synthetically accessible ligands with enhanced target specificity. Evaluations across diverse protein targets confirm that Chem42 surpasses existing approaches in chemical validity, target-aware design, and predicted binding affinity. By reducing the search space of viable drug candidates, Chem42 could accelerate the drug discovery pipeline, offering a powerful generative AI tool for precision medicine. Our Chem42 models set a new benchmark in molecule property prediction, conditional molecule generation, and target-aware ligand design. The models are publicly available at this http URL. 

---
# EmpathyAgent: Can Embodied Agents Conduct Empathetic Actions? 

**Authors**: Xinyan Chen, Jiaxin Ge, Hongming Dai, Qiang Zhou, Qiuxuan Feng, Jingtong Hu, Yizhou Wang, Jiaming Liu, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16545)  

**Abstract**: Empathy is fundamental to human interactions, yet it remains unclear whether embodied agents can provide human-like empathetic support. Existing works have studied agents' tasks solving and social interactions abilities, but whether agents can understand empathetic needs and conduct empathetic behaviors remains overlooked. To address this, we introduce EmpathyAgent, the first benchmark to evaluate and enhance agents' empathetic actions across diverse scenarios. EmpathyAgent contains 10,000 multimodal samples with corresponding empathetic task plans and three different challenges. To systematically evaluate the agents' empathetic actions, we propose an empathy-specific evaluation suite that evaluates the agents' empathy process. We benchmark current models and found that exhibiting empathetic actions remains a significant challenge. Meanwhile, we train Llama3-8B using EmpathyAgent and find it can potentially enhance empathetic behavior. By establishing a standard benchmark for evaluating empathetic actions, we hope to advance research in empathetic embodied agents. Our code and data are publicly available at this https URL. 

---
# Earthquake Response Analysis with AI 

**Authors**: Deep Patel, Panthadeep Bhattacharjee, Amit Reza, Priodyuti Pradhan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16509)  

**Abstract**: A timely and effective response is crucial to minimize damage and save lives during natural disasters like earthquakes. Microblogging platforms, particularly Twitter, have emerged as valuable real-time information sources for such events. This work explores the potential of leveraging Twitter data for earthquake response analysis. We develop a machine learning (ML) framework by incorporating natural language processing (NLP) techniques to extract and analyze relevant information from tweets posted during earthquake events. The approach primarily focuses on extracting location data from tweets to identify affected areas, generating severity maps, and utilizing WebGIS to display valuable information. The insights gained from this analysis can aid emergency responders, government agencies, humanitarian organizations, and NGOs in enhancing their disaster response strategies and facilitating more efficient resource allocation during earthquake events. 

---
# Scalable Evaluation of Online Moderation Strategies via Synthetic Simulations 

**Authors**: Dimitris Tsirmpas, Ion Androutsopoulos, John Pavlopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.16505)  

**Abstract**: Despite the ever-growing importance of online moderation, there has been no large-scale study evaluating the effectiveness of alternative moderation strategies. This is largely due to the lack of appropriate datasets, and the difficulty of getting human discussants, moderators, and evaluators involved in multiple experiments. In this paper, we propose a methodology for leveraging synthetic experiments performed exclusively by Large Language Models (LLMs) to initially bypass the need for human participation in experiments involving online moderation. We evaluate six LLM moderation configurations; two currently used real-life moderation strategies (guidelines issued for human moderators for online moderation and real-life facilitation), two baseline strategies (guidelines elicited for LLM alignment work, and LLM moderation with minimal prompting) a baseline with no moderator at all, as well as our own proposed strategy inspired by a Reinforcement Learning (RL) formulation of the problem. We find that our own moderation strategy significantly outperforms established moderation guidelines, as well as out-of-the-box LLM moderation. We also find that smaller LLMs, with less intensive instruction-tuning, can create more varied discussions than larger models. In order to run these experiments, we create and release an efficient, purpose-built, open-source Python framework, dubbed "SynDisco" to easily simulate hundreds of discussions using LLM user-agents and moderators. Additionally, we release the Virtual Moderation Dataset (VMD), a large dataset of LLM-generated and LLM-annotated discussions, generated by three families of open-source LLMs accompanied by an exploratory analysis of the dataset. 

---
# Llms, Virtual Users, and Bias: Predicting Any Survey Question Without Human Data 

**Authors**: Enzo Sinacola, Arnault Pachot, Thierry Petit  

**Link**: [PDF](https://arxiv.org/pdf/2503.16498)  

**Abstract**: Large Language Models (LLMs) offer a promising alternative to traditional survey methods, potentially enhancing efficiency and reducing costs. In this study, we use LLMs to create virtual populations that answer survey questions, enabling us to predict outcomes comparable to human responses. We evaluate several LLMs-including GPT-4o, GPT-3.5, Claude 3.5-Sonnet, and versions of the Llama and Mistral models-comparing their performance to that of a traditional Random Forests algorithm using demographic data from the World Values Survey (WVS). LLMs demonstrate competitive performance overall, with the significant advantage of requiring no additional training data. However, they exhibit biases when predicting responses for certain religious and population groups, underperforming in these areas. On the other hand, Random Forests demonstrate stronger performance than LLMs when trained with sufficient data. We observe that removing censorship mechanisms from LLMs significantly improves predictive accuracy, particularly for underrepresented demographic segments where censored models struggle. These findings highlight the importance of addressing biases and reconsidering censorship approaches in LLMs to enhance their reliability and fairness in public opinion research. 

---
# Human Preferences for Constructive Interactions in Language Model Alignment 

**Authors**: Yara Kyrychenko, Jon Roozenbeek, Brandon Davidson, Sander van der Linden, Ramit Debnath  

**Link**: [PDF](https://arxiv.org/pdf/2503.16480)  

**Abstract**: As large language models (LLMs) enter the mainstream, aligning them to foster constructive dialogue rather than exacerbate societal divisions is critical. Using an individualized and multicultural alignment dataset of over 7,500 conversations of individuals from 74 countries engaging with 21 LLMs, we examined how linguistic attributes linked to constructive interactions are reflected in human preference data used for training AI. We found that users consistently preferred well-reasoned and nuanced responses while rejecting those high in personal storytelling. However, users who believed that AI should reflect their values tended to place less preference on reasoning in LLM responses and more on curiosity. Encouragingly, we observed that users could set the tone for how constructive their conversation would be, as LLMs mirrored linguistic attributes, including toxicity, in user queries. 

---
# Human-Centered AI in Multidisciplinary Medical Discussions: Evaluating the Feasibility of a Chat-Based Approach to Case Assessment 

**Authors**: Shinnosuke Sawano, Satoshi Kodera  

**Link**: [PDF](https://arxiv.org/pdf/2503.16464)  

**Abstract**: In this study, we investigate the feasibility of using a human-centered artificial intelligence (AI) chat platform where medical specialists collaboratively assess complex cases. As the target population for this platform, we focus on patients with cardiovascular diseases who are in a state of multimorbidity, that is, suffering from multiple chronic conditions. We evaluate simulated cases with multiple diseases using a chat application by collaborating with physicians to assess feasibility, efficiency gains through AI utilization, and the quantification of discussion content. We constructed simulated cases based on past case reports, medical errors reports and complex cases of cardiovascular diseases experienced by the physicians. The analysis of discussions across five simulated cases demonstrated a significant reduction in the time required for summarization using AI, with an average reduction of 79.98\%. Additionally, we examined hallucination rates in AI-generated summaries used in multidisciplinary medical discussions. The overall hallucination rate ranged from 1.01\% to 5.73\%, with an average of 3.62\%, whereas the harmful hallucination rate varied from 0.00\% to 2.09\%, with an average of 0.49\%. Furthermore, morphological analysis demonstrated that multidisciplinary assessments enabled a more complex and detailed representation of medical knowledge compared with single physician assessments. We examined structural differences between multidisciplinary and single physician assessments using centrality metrics derived from the knowledge graph. In this study, we demonstrated that AI-assisted summarization significantly reduced the time required for medical discussions while maintaining structured knowledge representation. These findings can support the feasibility of AI-assisted chat-based discussions as a human-centered approach to multidisciplinary medical decision-making. 

---
# Improving Interactive Diagnostic Ability of a Large Language Model Agent Through Clinical Experience Learning 

**Authors**: Zhoujian Sun, Ziyi Liu, Cheng Luo, Jiebin Chu, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16463)  

**Abstract**: Recent advances in large language models (LLMs) have shown promising results in medical diagnosis, with some studies indicating superior performance compared to human physicians in specific scenarios. However, the diagnostic capabilities of LLMs are often overestimated, as their performance significantly deteriorates in interactive diagnostic settings that require active information gathering. This study investigates the underlying mechanisms behind the performance degradation phenomenon and proposes a solution. We identified that the primary deficiency of LLMs lies in the initial diagnosis phase, particularly in information-gathering efficiency and initial diagnosis formation, rather than in the subsequent differential diagnosis phase. To address this limitation, we developed a plug-and-play method enhanced (PPME) LLM agent, leveraging over 3.5 million electronic medical records from Chinese and American healthcare facilities. Our approach integrates specialized models for initial disease diagnosis and inquiry into the history of the present illness, trained through supervised and reinforcement learning techniques. The experimental results indicate that the PPME LLM achieved over 30% improvement compared to baselines. The final diagnostic accuracy of the PPME LLM in interactive diagnostic scenarios approached levels comparable to those achieved using complete clinical data. These findings suggest a promising potential for developing autonomous diagnostic systems, although further validation studies are needed. 

---
# Integrating Personality into Digital Humans: A Review of LLM-Driven Approaches for Virtual Reality 

**Authors**: Iago Alves Brito, Julia Soares Dollis, Fernanda Bufon Färber, Pedro Schindler Freire Brasil Ribeiro, Rafael Teixeira Sousa, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2503.16457)  

**Abstract**: The integration of large language models (LLMs) into virtual reality (VR) environments has opened new pathways for creating more immersive and interactive digital humans. By leveraging the generative capabilities of LLMs alongside multimodal outputs such as facial expressions and gestures, virtual agents can simulate human-like personalities and emotions, fostering richer and more engaging user experiences. This paper provides a comprehensive review of methods for enabling digital humans to adopt nuanced personality traits, exploring approaches such as zero-shot, few-shot, and fine-tuning. Additionally, it highlights the challenges of integrating LLM-driven personality traits into VR, including computational demands, latency issues, and the lack of standardized evaluation frameworks for multimodal interactions. By addressing these gaps, this work lays a foundation for advancing applications in education, therapy, and gaming, while fostering interdisciplinary collaboration to redefine human-computer interaction in VR. 

---
# The Application of MATEC (Multi-AI Agent Team Care) Framework in Sepsis Care 

**Authors**: Andrew Cho, Jason M. Woo, Brian Shi, Aishwaryaa Udeshi, Jonathan S. H. Woo  

**Link**: [PDF](https://arxiv.org/pdf/2503.16433)  

**Abstract**: Under-resourced or rural hospitals have limited access to medical specialists and healthcare professionals, which can negatively impact patient outcomes in sepsis. To address this gap, we developed the MATEC (Multi-AI Agent Team Care) framework, which integrates a team of specialized AI agents for sepsis care. The sepsis AI agent team includes five doctor agents, four health professional agents, and a risk prediction model agent, with an additional 33 doctor agents available for consultations. Ten attending physicians at a teaching hospital evaluated this framework, spending approximately 40 minutes on the web-based MATEC application and participating in the 5-point Likert scale survey (rated from 1-unfavorable to 5-favorable). The physicians found the MATEC framework very useful (Median=4, P=0.01), and very accurate (Median=4, P<0.01). This pilot study demonstrates that a Multi-AI Agent Team Care framework (MATEC) can potentially be useful in assisting medical professionals, particularly in under-resourced hospital settings. 

---
# Multimodal Transformer Models for Turn-taking Prediction: Effects on Conversational Dynamics of Human-Agent Interaction during Cooperative Gameplay 

**Authors**: Young-Ho Bae, Casey C. Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2503.16432)  

**Abstract**: This study investigates multimodal turn-taking prediction within human-agent interactions (HAI), particularly focusing on cooperative gaming environments. It comprises both model development and subsequent user study, aiming to refine our understanding and improve conversational dynamics in spoken dialogue systems (SDSs). For the modeling phase, we introduce a novel transformer-based deep learning (DL) model that simultaneously integrates multiple modalities - text, vision, audio, and contextual in-game data to predict turn-taking events in real-time. Our model employs a Crossmodal Transformer architecture to effectively fuse information from these diverse modalities, enabling more comprehensive turn-taking predictions. The model demonstrates superior performance compared to baseline models, achieving 87.3% accuracy and 83.0% macro F1 score. A human user study was then conducted to empirically evaluate the turn-taking DL model in an interactive scenario with a virtual avatar while playing the game "Dont Starve Together", comparing a control condition without turn-taking prediction (n=20) to an experimental condition with our model deployed (n=40). Both conditions included a mix of English and Korean speakers, since turn-taking cues are known to vary by culture. We then analyzed the interaction quality, examining aspects such as utterance counts, interruption frequency, and participant perceptions of the avatar. Results from the user study suggest that our multimodal turn-taking model not only enhances the fluidity and naturalness of human-agent conversations, but also maintains a balanced conversational dynamic without significantly altering dialogue frequency. The study provides in-depth insights into the influence of turn-taking abilities on user perceptions and interaction quality, underscoring the potential for more contextually adaptive and responsive conversational agents. 

---
