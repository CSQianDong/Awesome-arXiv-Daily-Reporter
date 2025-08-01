# Checklists Are Better Than Reward Models For Aligning Language Models 

**Authors**: Vijay Viswanathan, Yanchao Sun, Shuang Ma, Xiang Kong, Meng Cao, Graham Neubig, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18624)  

**Abstract**: Language models must be adapted to understand and follow user instructions. Reinforcement learning is widely used to facilitate this -- typically using fixed criteria such as "helpfulness" and "harmfulness". In our work, we instead propose using flexible, instruction-specific criteria as a means of broadening the impact that reinforcement learning can have in eliciting instruction following. We propose "Reinforcement Learning from Checklist Feedback" (RLCF). From instructions, we extract checklists and evaluate how well responses satisfy each item - using both AI judges and specialized verifier programs - then combine these scores to compute rewards for RL. We compare RLCF with other alignment methods applied to a strong instruction following model (Qwen2.5-7B-Instruct) on five widely-studied benchmarks -- RLCF is the only method to improve performance on every benchmark, including a 4-point boost in hard satisfaction rate on FollowBench, a 6-point increase on InFoBench, and a 3-point rise in win rate on Arena-Hard. These results establish checklist feedback as a key tool for improving language models' support of queries that express a multitude of needs. 

---
# TRPrompt: Bootstrapping Query-Aware Prompt Optimization from Textual Rewards 

**Authors**: Andreea Nica, Ivan Zakazov, Nicolas Mario Baldwin, Saibo Geng, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2507.18618)  

**Abstract**: Prompt optimization improves the reasoning abilities of large language models (LLMs) without requiring parameter updates to the target model. Following heuristic-based "Think step by step" approaches, the field has evolved in two main directions: while one group of methods uses textual feedback to elicit improved prompts from general-purpose LLMs in a training-free way, a concurrent line of research relies on numerical rewards to train a special prompt model, tailored for providing optimal prompts to the target model. In this paper, we introduce the Textual Reward Prompt framework (TRPrompt), which unifies these approaches by directly incorporating textual feedback into training of the prompt model. Our framework does not require prior dataset collection and is being iteratively improved with the feedback on the generated prompts. When coupled with the capacity of an LLM to internalize the notion of what a "good" prompt is, the high-resolution signal provided by the textual rewards allows us to train a prompt model yielding state-of-the-art query-specific prompts for the problems from the challenging math datasets GSMHard and MATH. 

---
# AQuilt: Weaving Logic and Self-Inspection into Low-Cost, High-Relevance Data Synthesis for Specialist LLMs 

**Authors**: Xiaopeng Ke, Hexuan Deng, Xuebo Liu, Jun Rao, Zhenxi Song, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18584)  

**Abstract**: Despite the impressive performance of large language models (LLMs) in general domains, they often underperform in specialized domains. Existing approaches typically rely on data synthesis methods and yield promising results by using unlabeled data to capture domain-specific features. However, these methods either incur high computational costs or suffer from performance limitations, while also demonstrating insufficient generalization across different tasks. To address these challenges, we propose AQuilt, a framework for constructing instruction-tuning data for any specialized domains from corresponding unlabeled data, including Answer, Question, Unlabeled data, Inspection, Logic, and Task type. By incorporating logic and inspection, we encourage reasoning processes and self-inspection to enhance model performance. Moreover, customizable task instructions enable high-quality data generation for any task. As a result, we construct a dataset of 703k examples to train a powerful data synthesis model. Experiments show that AQuilt is comparable to DeepSeek-V3 while utilizing just 17% of the production cost. Further analysis demonstrates that our generated data exhibits higher relevance to downstream tasks. Source code, models, and scripts are available at this https URL. 

---
# System Report for CCL25-Eval Task 10: SRAG-MAV for Fine-Grained Chinese Hate Speech Recognition 

**Authors**: Jiahao Wang, Ramen Liu, Longhui Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.18580)  

**Abstract**: This paper presents our system for CCL25-Eval Task 10, addressing Fine-Grained Chinese Hate Speech Recognition (FGCHSR). We propose a novel SRAG-MAV framework that synergistically integrates task reformulation(TR), Self-Retrieval-Augmented Generation (SRAG), and Multi-Round Accumulative Voting (MAV). Our method reformulates the quadruplet extraction task into triplet extraction, uses dynamic retrieval from the training set to create contextual prompts, and applies multi-round inference with voting to improve output stability and performance. Our system, based on the Qwen2.5-7B model, achieves a Hard Score of 26.66, a Soft Score of 48.35, and an Average Score of 37.505 on the STATE ToxiCN dataset, significantly outperforming baselines such as GPT-4o (Average Score 15.63) and fine-tuned Qwen2.5-7B (Average Score 35.365). The code is available at this https URL. 

---
# Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs 

**Authors**: Feng Hong, Geng Yu, Yushi Ye, Haicheng Huang, Huangjie Zheng, Ya Zhang, Yanfeng Wang, Jiangchao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2507.18578)  

**Abstract**: Diffusion Large Language Models (DLLMs) have emerged as a compelling alternative to Autoregressive models, designed for fast parallel generation. However, existing DLLMs are plagued by a severe quality-speed trade-off, where faster parallel decoding leads to significant performance degradation. We attribute this to the irreversibility of standard decoding in DLLMs, which is easily polarized into the wrong decoding direction along with early error context accumulation. To resolve this, we introduce Wide-In, Narrow-Out (WINO), a training-free decoding algorithm that enables revokable decoding in DLLMs. WINO employs a parallel draft-and-verify mechanism, aggressively drafting multiple tokens while simultaneously using the model's bidirectional context to verify and re-mask suspicious ones for refinement. Verified in open-source DLLMs like LLaDA and MMaDA, WINO is shown to decisively improve the quality-speed trade-off. For instance, on the GSM8K math benchmark, it accelerates inference by 6$\times$ while improving accuracy by 2.58%; on Flickr30K captioning, it achieves a 10$\times$ speedup with higher performance. More comprehensive experiments are conducted to demonstrate the superiority and provide an in-depth understanding of WINO. 

---
# Hybrid Tokenization Strategy for DNA Language Model using Byte Pair Encoding and K-MER Methods 

**Authors**: Ganesh Sapkota, Md Hasibur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2507.18570)  

**Abstract**: This paper presents a novel hybrid tokenization strategy that enhances the performance of DNA Language Models (DLMs) by combining 6-mer tokenization with Byte Pair Encoding (BPE-600). Traditional k-mer tokenization is effective at capturing local DNA sequence structures but often faces challenges, including uneven token distribution and a limited understanding of global sequence context. To address these limitations, we propose merging unique 6mer tokens with optimally selected BPE tokens generated through 600 BPE cycles. This hybrid approach ensures a balanced and context-aware vocabulary, enabling the model to capture both short and long patterns within DNA sequences simultaneously. A foundational DLM trained on this hybrid vocabulary was evaluated using next-k-mer prediction as a fine-tuning task, demonstrating significantly improved performance. The model achieved prediction accuracies of 10.78% for 3-mers, 10.1% for 4-mers, and 4.12% for 5-mers, outperforming state-of-the-art models such as NT, DNABERT2, and GROVER. These results highlight the ability of the hybrid tokenization strategy to preserve both the local sequence structure and global contextual information in DNA modeling. This work underscores the importance of advanced tokenization methods in genomic language modeling and lays a robust foundation for future applications in downstream DNA sequence analysis and biological research. 

---
# GIIFT: Graph-guided Inductive Image-free Multimodal Machine Translation 

**Authors**: Jiafeng Xiong, Yuting Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.18562)  

**Abstract**: Multimodal Machine Translation (MMT) has demonstrated the significant help of visual information in machine translation. However, existing MMT methods face challenges in leveraging the modality gap by enforcing rigid visual-linguistic alignment whilst being confined to inference within their trained multimodal domains. In this work, we construct novel multimodal scene graphs to preserve and integrate modality-specific information and introduce GIIFT, a two-stage Graph-guided Inductive Image-Free MMT framework that uses a cross-modal Graph Attention Network adapter to learn multimodal knowledge in a unified fused space and inductively generalize it to broader image-free translation domains. Experimental results on the Multi30K dataset of English-to-French and English-to-German tasks demonstrate that our GIIFT surpasses existing approaches and achieves the state-of-the-art, even without images during inference. Results on the WMT benchmark show significant improvements over the image-free translation baselines, demonstrating the strength of GIIFT towards inductive image-free inference. 

---
# GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface 

**Authors**: Urchade Zaratiana, Gil Pasternak, Oliver Boyd, George Hurn-Maloney, Ash Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2507.18546)  

**Abstract**: Information extraction (IE) is fundamental to numerous NLP applications, yet existing solutions often require specialized models for different tasks or rely on computationally expensive large language models. We present GLiNER2, a unified framework that enhances the original GLiNER architecture to support named entity recognition, text classification, and hierarchical structured data extraction within a single efficient model. Built pretrained transformer encoder architecture, GLiNER2 maintains CPU efficiency and compact size while introducing multi-task composition through an intuitive schema-based interface. Our experiments demonstrate competitive performance across extraction and classification tasks with substantial improvements in deployment accessibility compared to LLM-based alternatives. We release GLiNER2 as an open-source pip-installable library with pre-trained models and documentation at this https URL. 

---
# Effective Multi-Task Learning for Biomedical Named Entity Recognition 

**Authors**: João Ruano, Gonçalo M. Correia, Leonor Barreiros, Afonso Mendes  

**Link**: [PDF](https://arxiv.org/pdf/2507.18542)  

**Abstract**: Biomedical Named Entity Recognition presents significant challenges due to the complexity of biomedical terminology and inconsistencies in annotation across datasets. This paper introduces SRU-NER (Slot-based Recurrent Unit NER), a novel approach designed to handle nested named entities while integrating multiple datasets through an effective multi-task learning strategy. SRU-NER mitigates annotation gaps by dynamically adjusting loss computation to avoid penalizing predictions of entity types absent in a given dataset. Through extensive experiments, including a cross-corpus evaluation and human assessment of the model's predictions, SRU-NER achieves competitive performance in biomedical and general-domain NER tasks, while improving cross-domain generalization. 

---
# The Moral Gap of Large Language Models 

**Authors**: Maciej Skorski, Alina Landowska  

**Link**: [PDF](https://arxiv.org/pdf/2507.18523)  

**Abstract**: Moral foundation detection is crucial for analyzing social discourse and developing ethically-aligned AI systems. While large language models excel across diverse tasks, their performance on specialized moral reasoning remains unclear.
This study provides the first comprehensive comparison between state-of-the-art LLMs and fine-tuned transformers across Twitter and Reddit datasets using ROC, PR, and DET curve analysis.
Results reveal substantial performance gaps, with LLMs exhibiting high false negative rates and systematic under-detection of moral content despite prompt engineering efforts. These findings demonstrate that task-specific fine-tuning remains superior to prompting for moral reasoning applications. 

---
# Not All Features Deserve Attention: Graph-Guided Dependency Learning for Tabular Data Generation with Language Models 

**Authors**: Zheyu Zhang, Shuo Yang, Bardh Prenkaj, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2507.18504)  

**Abstract**: Large Language Models (LLMs) have shown strong potential for tabular data generation by modeling textualized feature-value pairs. However, tabular data inherently exhibits sparse feature-level dependencies, where many feature interactions are structurally insignificant. This creates a fundamental mismatch as LLMs' self-attention mechanism inevitably distributes focus across all pairs, diluting attention on critical relationships, particularly in datasets with complex dependencies or semantically ambiguous features. To address this limitation, we propose GraDe (Graph-Guided Dependency Learning), a novel method that explicitly integrates sparse dependency graphs into LLMs' attention mechanism. GraDe employs a lightweight dynamic graph learning module guided by externally extracted functional dependencies, prioritizing key feature interactions while suppressing irrelevant ones. Our experiments across diverse real-world datasets demonstrate that GraDe outperforms existing LLM-based approaches by up to 12% on complex datasets while achieving competitive results with state-of-the-art approaches in synthetic data quality. Our method is minimally intrusive yet effective, offering a practical solution for structure-aware tabular data modeling with LLMs. 

---
# Generation of Synthetic Clinical Text: A Systematic Review 

**Authors**: Basel Alshaikhdeeb, Ahmed Abdelmonem Hemedan, Soumyabrata Ghosh, Irina Balaur, Venkata Satagopam  

**Link**: [PDF](https://arxiv.org/pdf/2507.18451)  

**Abstract**: Generating clinical synthetic text represents an effective solution for common clinical NLP issues like sparsity and privacy. This paper aims to conduct a systematic review on generating synthetic medical free-text by formulating quantitative analysis to three research questions concerning (i) the purpose of generation, (ii) the techniques, and (iii) the evaluation methods. We searched PubMed, ScienceDirect, Web of Science, Scopus, IEEE, Google Scholar, and arXiv databases for publications associated with generating synthetic medical unstructured free-text. We have identified 94 relevant articles out of 1,398 collected ones. A great deal of attention has been given to the generation of synthetic medical text from 2018 onwards, where the main purpose of such a generation is towards text augmentation, assistive writing, corpus building, privacy-preserving, annotation, and usefulness. Transformer architectures were the main predominant technique used to generate the text, especially the GPTs. On the other hand, there were four main aspects of evaluation, including similarity, privacy, structure, and utility, where utility was the most frequent method used to assess the generated synthetic medical text. Although the generated synthetic medical text demonstrated a moderate possibility to act as real medical documents in different downstream NLP tasks, it has proven to be a great asset as augmented, complementary to the real documents, towards improving the accuracy and overcoming sparsity/undersampling issues. Yet, privacy is still a major issue behind generating synthetic medical text, where more human assessments are needed to check for the existence of any sensitive information. Despite that, advances in generating synthetic medical text will considerably accelerate the adoption of workflows and pipeline development, discarding the time-consuming legalities of data transfer. 

---
# Restoring Rhythm: Punctuation Restoration Using Transformer Models for Bangla, a Low-Resource Language 

**Authors**: Md Obyedullahil Mamun, Md Adyelullahil Mamun, Arif Ahmad, Md. Imran Hossain Emu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18448)  

**Abstract**: Punctuation restoration enhances the readability of text and is critical for post-processing tasks in Automatic Speech Recognition (ASR), especially for low-resource languages like Bangla. In this study, we explore the application of transformer-based models, specifically XLM-RoBERTa-large, to automatically restore punctuation in unpunctuated Bangla text. We focus on predicting four punctuation marks: period, comma, question mark, and exclamation mark across diverse text domains. To address the scarcity of annotated resources, we constructed a large, varied training corpus and applied data augmentation techniques. Our best-performing model, trained with an augmentation factor of alpha = 0.20%, achieves an accuracy of 97.1% on the News test set, 91.2% on the Reference set, and 90.2% on the ASR set.
Results show strong generalization to reference and ASR transcripts, demonstrating the model's effectiveness in real-world, noisy scenarios. This work establishes a strong baseline for Bangla punctuation restoration and contributes publicly available datasets and code to support future research in low-resource NLP. 

---
# AraTable: Benchmarking LLMs' Reasoning and Understanding of Arabic Tabular Data 

**Authors**: Rana Alshaikh, Israa Alghanmi, Shelan Jeawak  

**Link**: [PDF](https://arxiv.org/pdf/2507.18442)  

**Abstract**: The cognitive and reasoning abilities of large language models (LLMs) have enabled remarkable progress in natural language processing. However, their performance in interpreting structured data, especially in tabular formats, remains limited. Although benchmarks for English tabular data are widely available, Arabic is still underrepresented because of the limited availability of public resources and its unique language features. To address this gap, we present AraTable, a novel and comprehensive benchmark designed to evaluate the reasoning and understanding capabilities of LLMs when applied to Arabic tabular data. AraTable consists of various evaluation tasks, such as direct question answering, fact verification, and complex reasoning, involving a wide range of Arabic tabular sources. Our methodology follows a hybrid pipeline, where initial content is generated by LLMs and subsequently filtered and verified by human experts to ensure high dataset quality. Initial analyses using AraTable show that, while LLMs perform adequately on simpler tabular tasks such as direct question answering, they continue to face significant cognitive challenges when tasks require deeper reasoning and fact verification. This indicates that there are substantial opportunities for future work to improve performance on complex tabular reasoning tasks. We also propose a fully automated evaluation framework that uses a self-deliberation mechanism and achieves performance nearly identical to that of human judges. This research provides a valuable, publicly available resource and evaluation framework that can help accelerate the development of foundational models for processing and analysing Arabic structured data. 

---
# FinDPO: Financial Sentiment Analysis for Algorithmic Trading through Preference Optimization of LLMs 

**Authors**: Giorgos Iacovides, Wuyang Zhou, Danilo Mandic  

**Link**: [PDF](https://arxiv.org/pdf/2507.18417)  

**Abstract**: Opinions expressed in online finance-related textual data are having an increasingly profound impact on trading decisions and market movements. This trend highlights the vital role of sentiment analysis as a tool for quantifying the nature and strength of such opinions. With the rapid development of Generative AI (GenAI), supervised fine-tuned (SFT) large language models (LLMs) have become the de facto standard for financial sentiment analysis. However, the SFT paradigm can lead to memorization of the training data and often fails to generalize to unseen samples. This is a critical limitation in financial domains, where models must adapt to previously unobserved events and the nuanced, domain-specific language of finance. To this end, we introduce FinDPO, the first finance-specific LLM framework based on post-training human preference alignment via Direct Preference Optimization (DPO). The proposed FinDPO achieves state-of-the-art performance on standard sentiment classification benchmarks, outperforming existing supervised fine-tuned models by 11% on the average. Uniquely, the FinDPO framework enables the integration of a fine-tuned causal LLM into realistic portfolio strategies through a novel 'logit-to-score' conversion, which transforms discrete sentiment predictions into continuous, rankable sentiment scores (probabilities). In this way, simulations demonstrate that FinDPO is the first sentiment-based approach to maintain substantial positive returns of 67% annually and strong risk-adjusted performance, as indicated by a Sharpe ratio of 2.0, even under realistic transaction costs of 5 basis points (bps). 

---
# Factual Inconsistencies in Multilingual Wikipedia Tables 

**Authors**: Silvia Cappa, Lingxiao Kong, Pille-Riin Peet, Fanfu Wei, Yuchen Zhou, Jan-Christoph Kalo  

**Link**: [PDF](https://arxiv.org/pdf/2507.18406)  

**Abstract**: Wikipedia serves as a globally accessible knowledge source with content in over 300 languages. Despite covering the same topics, the different versions of Wikipedia are written and updated independently. This leads to factual inconsistencies that can impact the neutrality and reliability of the encyclopedia and AI systems, which often rely on Wikipedia as a main training source. This study investigates cross-lingual inconsistencies in Wikipedia's structured content, with a focus on tabular data. We developed a methodology to collect, align, and analyze tables from Wikipedia multilingual articles, defining categories of inconsistency. We apply various quantitative and qualitative metrics to assess multilingual alignment using a sample dataset. These insights have implications for factual verification, multilingual knowledge interaction, and design for reliable AI systems leveraging Wikipedia content. 

---
# CLEAR: Error Analysis via LLM-as-a-Judge Made Easy 

**Authors**: Asaf Yehudai, Lilach Eden, Yotam Perlitz, Roy Bar-Haim, Michal Shmueli-Scheuer  

**Link**: [PDF](https://arxiv.org/pdf/2507.18392)  

**Abstract**: The evaluation of Large Language Models (LLMs) increasingly relies on other LLMs acting as judges. However, current evaluation paradigms typically yield a single score or ranking, answering which model is better but not why. While essential for benchmarking, these top-level scores obscure the specific, actionable reasons behind a model's performance. To bridge this gap, we introduce CLEAR, an interactive, open-source package for LLM-based error analysis. CLEAR first generates per-instance textual feedback, then it creates a set of system-level error issues, and quantifies the prevalence of each identified issue. Our package also provides users with an interactive dashboard that allows for a comprehensive error analysis through aggregate visualizations, applies interactive filters to isolate specific issues or score ranges, and drills down to the individual instances that exemplify a particular behavioral pattern. We demonstrate CLEAR analysis for RAG and Math benchmarks, and showcase its utility through a user case study. 

---
# Hybrid Annotation for Propaganda Detection: Integrating LLM Pre-Annotations with Human Intelligence 

**Authors**: Ariana Sahitaj, Premtim Sahitaj, Veronika Solopova, Jiaao Li, Sebastian Möller, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2507.18343)  

**Abstract**: Propaganda detection on social media remains challenging due to task complexity and limited high-quality labeled data. This paper introduces a novel framework that combines human expertise with Large Language Model (LLM) assistance to improve both annotation consistency and scalability. We propose a hierarchical taxonomy that organizes 14 fine-grained propaganda techniques into three broader categories, conduct a human annotation study on the HQP dataset that reveals low inter-annotator agreement for fine-grained labels, and implement an LLM-assisted pre-annotation pipeline that extracts propagandistic spans, generates concise explanations, and assigns local labels as well as a global label. A secondary human verification study shows significant improvements in both agreement and time-efficiency. Building on this, we fine-tune smaller language models (SLMs) to perform structured annotation. Instead of fine-tuning on human annotations, we train on high-quality LLM-generated data, allowing a large model to produce these annotations and a smaller model to learn to generate them via knowledge distillation. Our work contributes towards the development of scalable and robust propaganda detection systems, supporting the idea of transparent and accountable media ecosystems in line with SDG 16. The code is publicly available at our GitHub repository. 

---
# TDR: Task-Decoupled Retrieval with Fine-Grained LLM Feedback for In-Context Learning 

**Authors**: Yifu Chen, Bingchen Huang, Zhiling Wang, Yuanchao Du, Junfeng Luo, Lei Shen, Zhineng chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18340)  

**Abstract**: In-context learning (ICL) has become a classic approach for enabling LLMs to handle various tasks based on a few input-output examples. The effectiveness of ICL heavily relies on the quality of these examples, and previous works which focused on enhancing example retrieval capabilities have achieved impressive performances. However, two challenges remain in retrieving high-quality examples: (1) Difficulty in distinguishing cross-task data distributions, (2) Difficulty in making the fine-grained connection between retriever output and feedback from LLMs. In this paper, we propose a novel framework called TDR. TDR decouples the ICL examples from different tasks, which enables the retrieval module to retrieve examples specific to the target task within a multi-task dataset. Furthermore, TDR models fine-grained feedback from LLMs to supervise and guide the training of the retrieval module, which helps to retrieve high-quality examples. We conducted extensive experiments on a suite of 30 NLP tasks, the results demonstrate that TDR consistently improved results across all datasets and achieves state-of-the-art performance. Meanwhile, our approach is a plug-and-play method, which can be easily combined with various LLMs to improve example retrieval abilities for ICL. The code is available at this https URL. 

---
# Uncertainty Quantification for Evaluating Machine Translation Bias 

**Authors**: Ieva Raminta Staliūnaitė, Julius Cheng, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2507.18338)  

**Abstract**: In machine translation (MT), when the source sentence includes a lexeme whose gender is not overtly marked, but whose target-language equivalent requires gender specification, the model must infer the appropriate gender from the context and/or external knowledge. Studies have shown that MT models exhibit biased behaviour, relying on stereotypes even when they clash with contextual information. We posit that apart from confidently translating using the correct gender when it is evident from the input, models should also maintain uncertainty about the gender when it is ambiguous. Using recently proposed metrics of semantic uncertainty, we find that models with high translation and gender accuracy on unambiguous instances do not necessarily exhibit the expected level of uncertainty in ambiguous ones. Similarly, debiasing has independent effects on ambiguous and unambiguous translation instances. 

---
# BadReasoner: Planting Tunable Overthinking Backdoors into Large Reasoning Models for Fun or Profit 

**Authors**: Biao Yi, Zekun Fei, Jianing Geng, Tong Li, Lihai Nie, Zheli Liu, Yiming Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.18305)  

**Abstract**: Large reasoning models (LRMs) have emerged as a significant advancement in artificial intelligence, representing a specialized class of large language models (LLMs) designed to tackle complex reasoning tasks. The defining characteristic of LRMs lies in their extensive chain-of-thought (CoT) reasoning capabilities. In this paper, we identify a previously unexplored attack vector against LRMs, which we term "overthinking backdoors". We advance this concept by proposing a novel tunable backdoor, which moves beyond simple on/off attacks to one where an attacker can precisely control the extent of the model's reasoning verbosity. Our attack is implemented through a novel data poisoning methodology. It pairs a tunable trigger-where the number of repetitions signals the desired intensity-with a correspondingly verbose CoT response. These responses are programmatically generated by instructing a teacher LLM to inject a controlled number of redundant refinement steps into a correct reasoning process. The approach preserves output correctness, which ensures stealth and establishes the attack as a pure resource-consumption vector. Extensive empirical results on various LRMs demonstrate that our method can reliably trigger a controllable, multi-fold increase in the length of the reasoning process, without degrading the final answer's correctness. Our source code is available at this https URL. 

---
# StyleAdaptedLM: Enhancing Instruction Following Models with Efficient Stylistic Transfer 

**Authors**: Pritika Ramu, Apoorv Saxena, Meghanath M Y, Varsha Sankar, Debraj Basu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18294)  

**Abstract**: Adapting LLMs to specific stylistic characteristics, like brand voice or authorial tones, is crucial for enterprise communication but challenging to achieve from corpora which lacks instruction-response formatting without compromising instruction adherence. We introduce StyleAdaptedLM, a framework that efficiently transfers stylistic traits to instruction-following models using Low-Rank Adaptation (LoRA). LoRA adapters are first trained on a base model with diverse unstructured stylistic corpora, then merged with a separate instruction-following model. This enables robust stylistic customization without paired data or sacrificing task performance. Experiments across multiple datasets and models demonstrate improved stylistic consistency while preserving instruction adherence, with human evaluations confirming brand-specific convention uptake. StyleAdaptedLM offers an efficient path for stylistic personalization in LLMs. 

---
# Zero-shot OCR Accuracy of Low-Resourced Languages: A Comparative Analysis on Sinhala and Tamil 

**Authors**: Nevidu Jayatilleke, Nisansa de Silva  

**Link**: [PDF](https://arxiv.org/pdf/2507.18264)  

**Abstract**: Solving the problem of Optical Character Recognition (OCR) on printed text for Latin and its derivative scripts can now be considered settled due to the volumes of research done on English and other High-Resourced Languages (HRL). However, for Low-Resourced Languages (LRL) that use unique scripts, it remains an open problem. This study presents a comparative analysis of the zero-shot performance of six distinct OCR engines on two LRLs: Sinhala and Tamil. The selected engines include both commercial and open-source systems, aiming to evaluate the strengths of each category. The Cloud Vision API, Surya, Document AI, and Tesseract were evaluated for both Sinhala and Tamil, while Subasa OCR and EasyOCR were examined for only one language due to their limitations. The performance of these systems was rigorously analysed using five measurement techniques to assess accuracy at both the character and word levels. According to the findings, Surya delivered the best performance for Sinhala across all metrics, with a WER of 2.61%. Conversely, Document AI excelled across all metrics for Tamil, highlighted by a very low CER of 0.78%. In addition to the above analysis, we also introduce a novel synthetic Tamil OCR benchmarking dataset. 

---
# Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models 

**Authors**: Suhang Wu, Jialong Tang, Chengyi Yang, Pei Zhang, Baosong Yang, Junhui Li, Junfeng Yao, Min Zhang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2507.18263)  

**Abstract**: Direct speech translation (ST) has garnered increasing attention nowadays, yet the accurate translation of terminology within utterances remains a great challenge. In this regard, current studies mainly concentrate on leveraging various translation knowledge into ST models. However, these methods often struggle with interference from irrelevant noise and can not fully utilize the translation knowledge. To address these issues, in this paper, we propose a novel Locate-and-Focus method for terminology translation. It first effectively locates the speech clips containing terminologies within the utterance to construct translation knowledge, minimizing irrelevant information for the ST model. Subsequently, it associates the translation knowledge with the utterance and hypothesis from both audio and textual modalities, allowing the ST model to better focus on translation knowledge during translation. Experimental results across various datasets demonstrate that our method effectively locates terminologies within utterances and enhances the success rate of terminology translation, while maintaining robust general translation performance. 

---
# Prune&Comp: Free Lunch for Layer-Pruned LLMs via Iterative Pruning with Magnitude Compensation 

**Authors**: Xinrui Chen, Hongxing Zhang, Fanyi Zeng, Yongxian Wei, Yizhi Wang, Xitong Ling, Guanghao Li, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.18212)  

**Abstract**: Layer pruning has emerged as a promising technique for compressing large language models (LLMs) while achieving acceleration proportional to the pruning ratio. In this work, we identify that removing any layer induces a significant magnitude gap in hidden states, resulting in substantial performance degradation. To address this issue, we propose Prune&Comp, a novel plug-and-play layer pruning scheme that leverages magnitude compensation to mitigate such gaps in a training-free manner. Specifically, we first estimate the magnitude gap caused by layer removal and then eliminate this gap by rescaling the remaining weights offline, with zero runtime overhead incurred. We further demonstrate the advantages of Prune&Comp through an iterative pruning strategy. When integrated with an iterative prune-and-compensate loop, Prune&Comp consistently enhances existing layer pruning metrics. For instance, when 5 layers of LLaMA-3-8B are pruned using the prevalent block influence metric, Prune&Comp nearly halves the perplexity and retains 93.19\% of the original model's question-answering performance, outperforming the baseline by 4.01%. 

---
# Exploring the Impact of Instruction-Tuning on LLM's Susceptibility to Misinformation 

**Authors**: Kyubeen Han, Junseo Jang, Hongjin Kim, Geunyeong Jeong, Harksoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.18203)  

**Abstract**: Instruction-tuning enhances the ability of large language models (LLMs) to follow user instructions more accurately, improving usability while reducing harmful outputs. However, this process may increase the model's dependence on user input, potentially leading to the unfiltered acceptance of misinformation and the generation of hallucinations. Existing studies primarily highlight that LLMs are receptive to external information that contradict their parametric knowledge, but little research has been conducted on the direct impact of instruction-tuning on this phenomenon. In our study, we investigate the impact of instruction-tuning on LLM's susceptibility to misinformation. Our analysis reveals that instruction-tuned LLMs are significantly more likely to accept misinformation when it is presented by the user. A comparison with base models shows that instruction-tuning increases reliance on user-provided information, shifting susceptibility from the assistant role to the user role. Furthermore, we explore additional factors influencing misinformation susceptibility, such as the role of the user in prompt structure, misinformation length, and the presence of warnings in the system prompt. Our findings underscore the need for systematic approaches to mitigate unintended consequences of instruction-tuning and enhance the reliability of LLMs in real-world applications. 

---
# Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection 

**Authors**: San Kim, Jonghwi Kim, Yejin Jeon, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.18202)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by providing external knowledge for accurate and up-to-date responses. However, this reliance on external sources exposes a security risk, attackers can inject poisoned documents into the knowledge base to steer the generation process toward harmful or misleading outputs. In this paper, we propose Gradient-based Masked Token Probability (GMTP), a novel defense method to detect and filter out adversarially crafted documents. Specifically, GMTP identifies high-impact tokens by examining gradients of the retriever's similarity function. These key tokens are then masked, and their probabilities are checked via a Masked Language Model (MLM). Since injected tokens typically exhibit markedly low masked-token probabilities, this enables GMTP to easily detect malicious documents and achieve high-precision filtering. Experiments demonstrate that GMTP is able to eliminate over 90% of poisoned content while retaining relevant documents, thus maintaining robust retrieval and generation performance across diverse datasets and adversarial settings. 

---
# Integrating an ISO30401-compliant Knowledge management system with existing business processes of an organization 

**Authors**: Aline Belloni, Patrick Prieur  

**Link**: [PDF](https://arxiv.org/pdf/2507.18197)  

**Abstract**: Business process modeling is used by most organizations as an essential framework for ensuring efficiency and effectiveness of the work and workflow performed by its employees and for ensuring the alignment of such work with its strategic goals. For organizations that are compliant or near-compliant with ISO 9001, this approach involves the detailed mapping of processes, sub-processes, activities, and tasks. ISO30401 is a Management System Standard, introduced in 2018, establishing universal requirements for the set up of a Knowledge Management System in an organization. As ``ISO30401 implementers'' we regularly face the challenge of explaining our clients how the knowledge development, transformation and conveyances activities depicted in ISO30401 do integrate with existing operational processes. This article recaps process modelling principles in the context of ISO9001 and explores, based on our experience, how an ISO30401-compliant Knowledge Management System (KMS) entwines with all other processes of an Integrated Management System and in particular how it can be implemented by deploying the mechanisms of the SECI model through the steps of PDCA cycles. 

---
# TN-AutoRCA: Benchmark Construction and Agentic Framework for Self-Improving Alarm-Based Root Cause Analysis in Telecommunication Networks 

**Authors**: Keyu Wu, Qianjin Yu, Manlin Mei, Ruiting Liu, Jun Wang, Kailai Zhang, Yelun Bao  

**Link**: [PDF](https://arxiv.org/pdf/2507.18190)  

**Abstract**: Root Cause Analysis (RCA) in telecommunication networks is a critical task, yet it presents a formidable challenge for Artificial Intelligence (AI) due to its complex, graph-based reasoning requirements and the scarcity of realistic benchmarks. 

---
# SCOPE: Stochastic and Counterbiased Option Placement for Evaluating Large Language Models 

**Authors**: Wonjun Jeong, Dongseok Kim, Taegkeun Whangbo  

**Link**: [PDF](https://arxiv.org/pdf/2507.18182)  

**Abstract**: Large Language Models (LLMs) can achieve inflated scores on multiple-choice tasks by exploiting inherent biases in option positions or labels, rather than demonstrating genuine understanding. This study introduces SCOPE, an evaluation framework designed to measure and mitigate such selection bias in a dataset-independent manner. By repeatedly invoking a null prompt that lacks semantic content, SCOPE estimates each model's unique position-bias distribution. It then redistributes the answer slot according to the inverse-bias distribution, thereby equalizing the lucky-rate, the probability of selecting the correct answer by chance. Furthermore, it prevents semantically similar distractors from being placed adjacent to the answer, thereby blocking near-miss guesses based on superficial proximity cues. Across multiple benchmark experiments, SCOPE consistently outperformed existing debiasing methods in terms of stable performance improvements and showed clearer confidence distributions over correct options. This framework thus offers a new standard for enhancing the fairness and reliability of LLM evaluations. 

---
# Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models 

**Authors**: Kexin Chen, Dongxia Wang, Yi Liu, Haonan Zhang, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18171)  

**Abstract**: Despite the widespread use of Transformer-based text embedding models in NLP tasks, surprising 'sticky tokens' can undermine the reliability of embeddings. These tokens, when repeatedly inserted into sentences, pull sentence similarity toward a certain value, disrupting the normal distribution of embedding distances and degrading downstream performance. In this paper, we systematically investigate such anomalous tokens, formally defining them and introducing an efficient detection method, Sticky Token Detector (STD), based on sentence and token filtering. Applying STD to 40 checkpoints across 14 model families, we discover a total of 868 sticky tokens. Our analysis reveals that these tokens often originate from special or unused entries in the vocabulary, as well as fragmented subwords from multilingual corpora. Notably, their presence does not strictly correlate with model size or vocabulary size. We further evaluate how sticky tokens affect downstream tasks like clustering and retrieval, observing significant performance drops of up to 50%. Through attention-layer analysis, we show that sticky tokens disproportionately dominate the model's internal representations, raising concerns about tokenization robustness. Our findings show the need for better tokenization strategies and model design to mitigate the impact of sticky tokens in future text embedding applications. 

---
# HIVMedQA: Benchmarking large language models for HIV medical decision support 

**Authors**: Gonzalo Cardenal Antolin, Jacques Fellay, Bashkim Jaha, Roger Kouyos, Niko Beerenwinkel, Diane Duroux  

**Link**: [PDF](https://arxiv.org/pdf/2507.18143)  

**Abstract**: Large language models (LLMs) are emerging as valuable tools to support clinicians in routine decision-making. HIV management is a compelling use case due to its complexity, including diverse treatment options, comorbidities, and adherence challenges. However, integrating LLMs into clinical practice raises concerns about accuracy, potential harm, and clinician acceptance. Despite their promise, AI applications in HIV care remain underexplored, and LLM benchmarking studies are scarce. This study evaluates the current capabilities of LLMs in HIV management, highlighting their strengths and limitations. We introduce HIVMedQA, a benchmark designed to assess open-ended medical question answering in HIV care. The dataset consists of curated, clinically relevant questions developed with input from an infectious disease physician. We evaluated seven general-purpose and three medically specialized LLMs, applying prompt engineering to enhance performance. Our evaluation framework incorporates both lexical similarity and an LLM-as-a-judge approach, extended to better reflect clinical relevance. We assessed performance across key dimensions: question comprehension, reasoning, knowledge recall, bias, potential harm, and factual accuracy. Results show that Gemini 2.5 Pro consistently outperformed other models across most dimensions. Notably, two of the top three models were proprietary. Performance declined as question complexity increased. Medically fine-tuned models did not always outperform general-purpose ones, and larger model size was not a reliable predictor of performance. Reasoning and comprehension were more challenging than factual recall, and cognitive biases such as recency and status quo were observed. These findings underscore the need for targeted development and evaluation to ensure safe, effective LLM integration in clinical care. 

---
# MathOPEval: A Fine-grained Evaluation Benchmark for Visual Operations of MLLMs in Mathematical Reasoning 

**Authors**: Xiaoyuan Li, Moxin Li, Wenjie Wang, Rui Men, Yichang Zhang, Fuli Feng, Dayiheng Liu, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.18140)  

**Abstract**: Recent progress in Multi-modal Large Language Models (MLLMs) has enabled step-by-step multi-modal mathematical reasoning by performing visual operations based on the textual instructions. A promising approach uses code as an intermediate representation to precisely express and manipulate the images in the reasoning steps. However, existing evaluations focus mainly on text-only reasoning outputs, leaving the MLLM's ability to perform accurate visual operations via code largely unexplored. This work takes a first step toward addressing that gap by evaluating MLLM's code-based capabilities in multi-modal mathematical this http URL, our framework focuses on two key evaluation aspects: (1) Multi-modal Code Generation (MCG) evaluates the model's ability to accurately understand and construct visualizations from scratch. (2) Multi-modal Code Editing (MCE) assesses the model's capacity for fine-grained operations, which include three types: Deletion, Modification and Annotation. To evaluate the above tasks, we incorporate a dataset that covers the five most popular types of mathematical figures, including geometric diagrams, function plots, and three types of statistical charts, to provide a comprehensive and effective measurement of existing MLLMs. Our experimental evaluation involves nine mainstream MLLMs, and the results reveal that existing models still lag significantly behind human performance in performing fine-grained visual operations. 

---
# GOAT-SLM: A Spoken Language Model with Paralinguistic and Speaker Characteristic Awareness 

**Authors**: Hongjie Chen, Zehan Li, Yaodong Song, Wenming Deng, Yitong Yao, Yuxin Zhang, Hang Lv, Xuechao Zhu, Jian Kang, Jie Lian, Jie Li, Chao Wang, Shuangyong Song, Yongxiang Li, Zhongjiang He  

**Link**: [PDF](https://arxiv.org/pdf/2507.18119)  

**Abstract**: Recent advances in end-to-end spoken language models (SLMs) have significantly improved the ability of AI systems to engage in natural spoken interactions. However, most existing models treat speech merely as a vehicle for linguistic content, often overlooking the rich paralinguistic and speaker characteristic cues embedded in human speech, such as dialect, age, emotion, and non-speech vocalizations. In this work, we introduce GOAT-SLM, a novel spoken language model with paralinguistic and speaker characteristic awareness, designed to extend spoken language modeling beyond text semantics. GOAT-SLM adopts a dual-modality head architecture that decouples linguistic modeling from acoustic realization, enabling robust language understanding while supporting expressive and adaptive speech generation. To enhance model efficiency and versatility, we propose a modular, staged training strategy that progressively aligns linguistic, paralinguistic, and speaker characteristic information using large-scale speech-text corpora. Experimental results on TELEVAL, a multi-dimensional evaluation benchmark, demonstrate that GOAT-SLM achieves well-balanced performance across both semantic and non-semantic tasks, and outperforms existing open-source models in handling emotion, dialectal variation, and age-sensitive interactions. This work highlights the importance of modeling beyond linguistic content and advances the development of more natural, adaptive, and socially aware spoken language systems. 

---
# A New Pair of GloVes 

**Authors**: Riley Carlson, John Bauer, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2507.18103)  

**Abstract**: This report documents, describes, and evaluates new 2024 English GloVe (Global Vectors for Word Representation) models. While the original GloVe models built in 2014 have been widely used and found useful, languages and the world continue to evolve and we thought that current usage could benefit from updated models. Moreover, the 2014 models were not carefully documented as to the exact data versions and preprocessing that were used, and we rectify this by documenting these new models. We trained two sets of word embeddings using Wikipedia, Gigaword, and a subset of Dolma. Evaluation through vocabulary comparison, direct testing, and NER tasks shows that the 2024 vectors incorporate new culturally and linguistically relevant words, perform comparably on structural tasks like analogy and similarity, and demonstrate improved performance on recent, temporally dependent NER datasets such as non-Western newswire data. 

---
# Hybrid and Unitary Fine-Tuning of Large Language Models: Methods and Benchmarking under Resource Constraints 

**Authors**: Haomin Qi, Zihan Dai, Chengbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18076)  

**Abstract**: Fine-tuning large language models (LLMs) remains a computational bottleneck due to their scale and memory demands. This paper presents a comprehensive evaluation of parameter-efficient fine-tuning (PEFT) techniques, including LoRA, BOFT, LoRA-GA, and uRNN, and introduces a novel hybrid strategy that dynamically integrates BOFT's orthogonal stability with LoRA-GA's gradient-aligned rapid convergence. By computing per-layer adaptive updates guided by gradient norms, the hybrid method achieves superior convergence efficiency and generalization across diverse tasks. We also explore, for the first time, the adaptation of unitary RNN (uRNN) principles to transformer-based LLMs, enhancing gradient stability through structured unitary constraints. Empirical evaluations on four benchmarks -- GLUE, GSM8K, MT-Bench, and HumanEval -- using models ranging from 7B to 405B parameters demonstrate that our hybrid method consistently outperforms individual PEFT baselines, approaching full fine-tuning accuracy while reducing resource consumption by up to 2.1 times in training time and 50 percent in memory usage. These findings establish the hybrid approach as a practical and scalable fine-tuning solution for real-world deployment of LLMs under resource constraints. 

---
# TELEVAL: A Dynamic Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios 

**Authors**: Zehan Li, Hongjie Chen, Yuxin Zhang, Jing Zhou, Xuening Wang, Hang Lv, Mengjie Du, Yaodong Song, Jie Lian, Jian Kang, Jie Li, Yongxiang Li, Zhongjiang He, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.18061)  

**Abstract**: Spoken language models (SLMs) have seen rapid progress in recent years, along with the development of numerous benchmarks for evaluating their performance. However, most existing benchmarks primarily focus on evaluating whether SLMs can perform complex tasks comparable to those tackled by large language models (LLMs), often failing to align with how users naturally interact in real-world conversational scenarios. In this paper, we propose TELEVAL, a dynamic benchmark specifically designed to evaluate SLMs' effectiveness as conversational agents in realistic Chinese interactive settings. TELEVAL defines three evaluation dimensions: Explicit Semantics, Paralinguistic and Implicit Semantics, and System Abilities. It adopts a dialogue format consistent with real-world usage and evaluates text and audio outputs separately. TELEVAL particularly focuses on the model's ability to extract implicit cues from user speech and respond appropriately without additional instructions. Our experiments demonstrate that despite recent progress, existing SLMs still have considerable room for improvement in natural conversational tasks. We hope that TELEVAL can serve as a user-centered evaluation framework that directly reflects the user experience and contributes to the development of more capable dialogue-oriented SLMs. 

---
# Privacy-Preserving Synthetic Review Generation with Diverse Writing Styles Using LLMs 

**Authors**: Tevin Atwal, Chan Nam Tieu, Yefeng Yuan, Zhan Shi, Yuhong Liu, Liang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.18055)  

**Abstract**: The increasing use of synthetic data generated by Large Language Models (LLMs) presents both opportunities and challenges in data-driven applications. While synthetic data provides a cost-effective, scalable alternative to real-world data to facilitate model training, its diversity and privacy risks remain underexplored. Focusing on text-based synthetic data, we propose a comprehensive set of metrics to quantitatively assess the diversity (i.e., linguistic expression, sentiment, and user perspective), and privacy (i.e., re-identification risk and stylistic outliers) of synthetic datasets generated by several state-of-the-art LLMs. Experiment results reveal significant limitations in LLMs' capabilities in generating diverse and privacy-preserving synthetic data. Guided by the evaluation results, a prompt-based approach is proposed to enhance the diversity of synthetic reviews while preserving reviewer privacy. 

---
# Synthetic Data Generation for Phrase Break Prediction with Large Language Model 

**Authors**: Hoyeon Lee, Sejung Son, Ye-Eun Kang, Jong-Hwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.18044)  

**Abstract**: Current approaches to phrase break prediction address crucial prosodic aspects of text-to-speech systems but heavily rely on vast human annotations from audio or text, incurring significant manual effort and cost. Inherent variability in the speech domain, driven by phonetic factors, further complicates acquiring consistent, high-quality data. Recently, large language models (LLMs) have shown success in addressing data challenges in NLP by generating tailored synthetic data while reducing manual annotation needs. Motivated by this, we explore leveraging LLM to generate synthetic phrase break annotations, addressing the challenges of both manual annotation and speech-related tasks by comparing with traditional annotations and assessing effectiveness across multiple languages. Our findings suggest that LLM-based synthetic data generation effectively mitigates data challenges in phrase break prediction and highlights the potential of LLMs as a viable solution for the speech domain. 

---
# GrAInS: Gradient-based Attribution for Inference-Time Steering of LLMs and VLMs 

**Authors**: Duy Nguyen, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2507.18043)  

**Abstract**: Inference-time steering methods offer a lightweight alternative to fine-tuning large language models (LLMs) and vision-language models (VLMs) by modifying internal activations at test time without updating model weights. However, most existing approaches rely on fixed, global intervention vectors, overlook the causal influence of individual input tokens, and fail to leverage informative gradients from the model's logits, particularly in multimodal settings where visual and textual inputs contribute unevenly. To address these limitations, we introduce GrAInS, an inference-time steering approach that operates across both language-only and vision-language models and tasks. GrAInS uses contrastive, gradient-based attribution via Integrated Gradients to identify the top-k most influential tokens, both positively and negatively attributed based on their contribution to preferred versus dispreferred outputs. These tokens are then used to construct directional steering vectors that capture semantic shifts from undesirable to desirable behavior. During inference, GrAInS adjusts hidden activations at transformer layers guided by token-level attribution signals, and normalizes activations to preserve representational scale. This enables fine-grained, interpretable, and modular control over model behavior, without retraining or auxiliary supervision. Empirically, GrAInS consistently outperforms both fine-tuning and existing steering baselines: it achieves a 13.22% accuracy gain on TruthfulQA using Llama-3.1-8B, reduces hallucination rates on MMHal-Bench from 0.624 to 0.514 with LLaVA-1.6-7B, and improves alignment win rates on SPA-VL by 8.11%, all while preserving the model's fluency and general capabilities. 

---
# NeuralDB: Scaling Knowledge Editing in LLMs to 100,000 Facts with Neural KV Database 

**Authors**: Weizhi Fei, Hao Shi, Jing Xu, Jingchen Peng, Jiazheng Li, Jingzhao Zhang, Bo Bai, Wei Han, Zhenyuan Chen, Xueyan Niu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18028)  

**Abstract**: Efficiently editing knowledge stored in large language models (LLMs) enables model updates without large-scale training. One possible solution is Locate-and-Edit (L\&E), allowing simultaneous modifications of a massive number of facts. However, such editing may compromise the general abilities of LLMs and even result in forgetting edited facts when scaling up to thousands of edits. In this paper, we model existing linear L\&E methods as querying a Key-Value (KV) database. From this perspective, we then propose NeuralDB, an editing framework that explicitly represents the edited facts as a neural KV database equipped with a non-linear gated retrieval module, % In particular, our gated module only operates when inference involves the edited facts, effectively preserving the general abilities of LLMs. Comprehensive experiments involving the editing of 10,000 facts were conducted on the ZsRE and CounterFacts datasets, using GPT2-XL, GPT-J (6B) and Llama-3 (8B). The results demonstrate that NeuralDB not only excels in editing efficacy, generalization, specificity, fluency, and consistency, but also preserves overall performance across six representative text understanding and generation tasks. Further experiments indicate that NeuralDB maintains its effectiveness even when scaled to 100,000 facts (\textbf{50x} more than in prior work). 

---
# Technical Report of TeleChat2, TeleChat2.5 and T1 

**Authors**: Zihan Wang, Xinzhang Liu, Yitong Yao, Chao Wang, Yu Zhao, Zhihao Yang, Wenmin Deng, Kaipeng Jia, Jiaxin Peng, Yuyao Huang, Sishi Xiong, Zhuo Jiang, Kaidong Yu, Xiaohui Hu, Fubei Yao, Ruiyu Fang, Zhuoru Jiang, Ruiting Song, Qiyi Xie, Rui Xue, Xuewei He, Yanlei Xue, Zhu Yuan, Zhaoxi Zhang, Zilu Huang, Shiquan Wang, Xin Wang, Hanming Wu, Mingyuan Wang, Xufeng Zhan, Yuhan Sun, Zhaohu Xing, Yuhao Jiang, Bingkai Yang, Shuangyong Song, Yongxiang Li, Zhongjiang He, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.18013)  

**Abstract**: We introduce the latest series of TeleChat models: \textbf{TeleChat2}, \textbf{TeleChat2.5}, and \textbf{T1}, offering a significant upgrade over their predecessor, TeleChat. Despite minimal changes to the model architecture, the new series achieves substantial performance gains through enhanced training strategies in both pre-training and post-training stages. The series begins with \textbf{TeleChat2}, which undergoes pretraining on 10 trillion high-quality and diverse tokens. This is followed by Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to further enhance its capabilities. \textbf{TeleChat2.5} and \textbf{T1} expand the pipeline by incorporating a continual pretraining phase with domain-specific datasets, combined with reinforcement learning (RL) to improve performance in code generation and mathematical reasoning tasks. The \textbf{T1} variant is designed for complex reasoning, supporting long Chain-of-Thought (CoT) reasoning and demonstrating substantial improvements in mathematics and coding. In contrast, \textbf{TeleChat2.5} prioritizes speed, delivering rapid inference. Both flagship models of \textbf{T1} and \textbf{TeleChat2.5} are dense Transformer-based architectures with 115B parameters, showcasing significant advancements in reasoning and general task performance compared to the original TeleChat. Notably, \textbf{T1-115B} outperform proprietary models such as OpenAI's o1-mini and GPT-4o. We publicly release \textbf{TeleChat2}, \textbf{TeleChat2.5} and \textbf{T1}, including post-trained versions with 35B and 115B parameters, to empower developers and researchers with state-of-the-art language models tailored for diverse applications. 

---
# Natural Language Processing for Tigrinya: Current State and Future Directions 

**Authors**: Fitsum Gaim, Jong C. Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.17974)  

**Abstract**: Despite being spoken by millions of people, Tigrinya remains severely underrepresented in Natural Language Processing (NLP) research. This work presents a comprehensive survey of NLP research for Tigrinya, analyzing over 40 studies spanning more than a decade of work from 2011 to 2025. We systematically review the current state of computational resources, models, and applications across ten distinct downstream tasks, including morphological processing, machine translation, speech recognition, and question-answering. Our analysis reveals a clear trajectory from foundational, rule-based systems to modern neural architectures, with progress consistently unlocked by resource creation milestones. We identify key challenges rooted in Tigrinya's morphological complexity and resource scarcity, while highlighting promising research directions, including morphology-aware modeling, cross-lingual transfer, and community-centered resource development. This work serves as both a comprehensive reference for researchers and a roadmap for advancing Tigrinya NLP. A curated metadata of the surveyed studies and resources is made publicly available.\footnote{Tigrinya NLP Anthology: this https URL. 

---
# Are LLM Belief Updates Consistent with Bayes' Theorem? 

**Authors**: Sohaib Imran, Ihor Kendiukhov, Matthew Broerman, Aditya Thomas, Riccardo Campanella, Rob Lamb, Peter M. Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2507.17951)  

**Abstract**: Do larger and more capable language models learn to update their "beliefs" about propositions more consistently with Bayes' theorem when presented with evidence in-context? To test this, we formulate a Bayesian Coherence Coefficient (BCC) metric and generate a dataset with which to measure the BCC. We measure BCC for multiple pre-trained-only language models across five model families, comparing against the number of model parameters, the amount of training data, and model scores on common benchmarks. Our results provide evidence for our hypothesis that larger and more capable pre-trained language models assign credences that are more coherent with Bayes' theorem. These results have important implications for our understanding and governance of LLMs. 

---
# Evaluating the Performance of AI Text Detectors, Few-Shot and Chain-of-Thought Prompting Using DeepSeek Generated Text 

**Authors**: Hulayyil Alshammari, Praveen Rao  

**Link**: [PDF](https://arxiv.org/pdf/2507.17944)  

**Abstract**: Large language models (LLMs) have rapidly transformed the creation of written materials. LLMs have led to questions about writing integrity, thereby driving the creation of artificial intelligence (AI) detection technologies. Adversarial attacks, such as standard and humanized paraphrasing, inhibit detectors' ability to detect machine-generated text. Previous studies have mainly focused on ChatGPT and other well-known LLMs and have shown varying accuracy across detectors. However, there is a clear gap in the literature about DeepSeek, a recently published LLM. Therefore, in this work, we investigate whether six generally accessible AI detection tools -- AI Text Classifier, Content Detector AI, Copyleaks, QuillBot, GPT-2, and GPTZero -- can consistently recognize text generated by DeepSeek. The detectors were exposed to the aforementioned adversarial attacks. We also considered DeepSeek as a detector by performing few-shot prompting and chain-of-thought reasoning (CoT) for classifying AI and human-written text. We collected 49 human-authored question-answer pairs from before the LLM era and generated matching responses using DeepSeek-v3, producing 49 AI-generated samples. Then, we applied adversarial techniques such as paraphrasing and humanizing to add 196 more samples. These were used to challenge detector robustness and assess accuracy impact. While QuillBot and Copyleaks showed near-perfect performance on original and paraphrased DeepSeek text, others -- particularly AI Text Classifier and GPT-2 -- showed inconsistent results. The most effective attack was humanization, reducing accuracy to 71% for Copyleaks, 58% for QuillBot, and 52% for GPTZero. Few-shot and CoT prompting showed high accuracy, with the best five-shot result misclassifying only one of 49 samples (AI recall 96%, human recall 100%). 

---
# One Whisper to Grade Them All 

**Authors**: Nhan Phan, Anusha Porwal, Yaroslav Getman, Ekaterina Voskoboinik, Tamás Grósz, Mikko Kurimo  

**Link**: [PDF](https://arxiv.org/pdf/2507.17918)  

**Abstract**: We present an efficient end-to-end approach for holistic Automatic Speaking Assessment (ASA) of multi-part second-language tests, developed for the 2025 Speak & Improve Challenge. Our system's main novelty is the ability to process all four spoken responses with a single Whisper-small encoder, combine all information via a lightweight aggregator, and predict the final score. This architecture removes the need for transcription and per-part models, cuts inference time, and makes ASA practical for large-scale Computer-Assisted Language Learning systems.
Our system achieved a Root Mean Squared Error (RMSE) of 0.384, outperforming the text-based baseline (0.44) while using at most 168M parameters (about 70% of Whisper-small). Furthermore, we propose a data sampling strategy, allowing the model to train on only 44.8% of the speakers in the corpus and still reach 0.383 RMSE, demonstrating improved performance on imbalanced classes and strong data efficiency. 

---
# VeriMinder: Mitigating Analytical Vulnerabilities in NL2SQL 

**Authors**: Shubham Mohole, Sainyam Galhotra  

**Link**: [PDF](https://arxiv.org/pdf/2507.17896)  

**Abstract**: Application systems using natural language interfaces to databases (NLIDBs) have democratized data analysis. This positive development has also brought forth an urgent challenge to help users who might use these systems without a background in statistical analysis to formulate bias-free analytical questions. Although significant research has focused on text-to-SQL generation accuracy, addressing cognitive biases in analytical questions remains underexplored. We present VeriMinder, this https URL, an interactive system for detecting and mitigating such analytical vulnerabilities. Our approach introduces three key innovations: (1) a contextual semantic mapping framework for biases relevant to specific analysis contexts (2) an analytical framework that operationalizes the Hard-to-Vary principle and guides users in systematic data analysis (3) an optimized LLM-powered system that generates high-quality, task-specific prompts using a structured process involving multiple candidates, critic feedback, and self-reflection.
User testing confirms the merits of our approach. In direct user experience evaluation, 82.5% participants reported positively impacting the quality of the analysis. In comparative evaluation, VeriMinder scored significantly higher than alternative approaches, at least 20% better when considered for metrics of the analysis's concreteness, comprehensiveness, and accuracy. Our system, implemented as a web application, is set to help users avoid "wrong question" vulnerability during data analysis. VeriMinder code base with prompts, this https URL, is available as an MIT-licensed open-source software to facilitate further research and adoption within the community. 

---
# Dynamic and Generalizable Process Reward Modeling 

**Authors**: Zhangyue Yin, Qiushi Sun, Zhiyuan Zeng, Qinyuan Cheng, Xipeng Qiu, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17849)  

**Abstract**: Process Reward Models (PRMs) are crucial for guiding Large Language Models (LLMs) in complex scenarios by providing dense reward signals. However, existing PRMs primarily rely on heuristic approaches, which struggle with cross-domain generalization. While LLM-as-judge has been proposed to provide generalized rewards, current research has focused mainly on feedback results, overlooking the meaningful guidance embedded within the text. Additionally, static and coarse-grained evaluation criteria struggle to adapt to complex process supervision. To tackle these challenges, we propose Dynamic and Generalizable Process Reward Modeling (DG-PRM), which features a reward tree to capture and store fine-grained, multi-dimensional reward criteria. DG-PRM dynamically selects reward signals for step-wise reward scoring. To handle multifaceted reward signals, we pioneeringly adopt Pareto dominance estimation to identify discriminative positive and negative pairs. Experimental results show that DG-PRM achieves stunning performance on prevailing benchmarks, significantly boosting model performance across tasks with dense rewards. Further analysis reveals that DG-PRM adapts well to out-of-distribution scenarios, demonstrating exceptional generalizability. 

---
# Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning 

**Authors**: Yimeng Zhang, Tian Wang, Jiri Gesi, Ziyi Wang, Yuxuan Lu, Jiacheng Lin, Sinong Zhan, Vianne Gao, Ruochen Jiao, Junze Liu, Kun Qian, Yuxin Tang, Ran Xue, Houyu Zhang, Qingjun Cui, Yufan Guo, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17842)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated strong potential in generating 'believable human-like' behavior in web environments. Prior work has explored augmenting training data with LLM-synthesized rationales and applying supervised fine-tuning (SFT) to enhance reasoning ability, which in turn can improve downstream action prediction. However, the performance of such approaches remains inherently bounded by the reasoning capabilities of the model used to generate the rationales. In this paper, we introduce Shop-R1, a novel reinforcement learning (RL) framework aimed at enhancing the reasoning ability of LLMs for simulation of real human behavior in online shopping environments Specifically, Shop-R1 decomposes the human behavior simulation task into two stages: rationale generation and action prediction, each guided by distinct reward signals. For rationale generation, we leverage internal model signals (e.g., logit distributions) to guide the reasoning process in a self-supervised manner. For action prediction, we propose a hierarchical reward structure with difficulty-aware scaling to prevent reward hacking and enable fine-grained reward assignment. This design evaluates both high-level action types and the correctness of fine-grained sub-action details (attributes and values), rewarding outputs proportionally to their difficulty. Experimental results show that our method achieves a relative improvement of over 65% compared to the baseline. 

---
# SynC: Synthetic Image Caption Dataset Refinement with One-to-many Mapping for Zero-shot Image Captioning 

**Authors**: Si-Woo Kim, MinJu Jeon, Ye-Chan Kim, Soeun Lee, Taewhan Kim, Dong-Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.18616)  

**Abstract**: Zero-shot Image Captioning (ZIC) increasingly utilizes synthetic datasets generated by text-to-image (T2I) models to mitigate the need for costly manual annotation. However, these T2I models often produce images that exhibit semantic misalignments with their corresponding input captions (e.g., missing objects, incorrect attributes), resulting in noisy synthetic image-caption pairs that can hinder model training. Existing dataset pruning techniques are largely designed for removing noisy text in web-crawled data. However, these methods are ill-suited for the distinct challenges of synthetic data, where captions are typically well-formed, but images may be inaccurate representations. To address this gap, we introduce SynC, a novel framework specifically designed to refine synthetic image-caption datasets for ZIC. Instead of conventional filtering or regeneration, SynC focuses on reassigning captions to the most semantically aligned images already present within the synthetic image pool. Our approach employs a one-to-many mapping strategy by initially retrieving multiple relevant candidate images for each caption. We then apply a cycle-consistency-inspired alignment scorer that selects the best image by verifying its ability to retrieve the original caption via image-to-text retrieval. Extensive evaluations demonstrate that SynC consistently and significantly improves performance across various ZIC models on standard benchmarks (MS-COCO, Flickr30k, NoCaps), achieving state-of-the-art results in several scenarios. SynC offers an effective strategy for curating refined synthetic data to enhance ZIC. 

---
# DR.EHR: Dense Retrieval for Electronic Health Record with Knowledge Injection and Synthetic Data 

**Authors**: Zhengyun Zhao, Huaiyuan Ying, Yue Zhong, Sheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18583)  

**Abstract**: Electronic Health Records (EHRs) are pivotal in clinical practices, yet their retrieval remains a challenge mainly due to semantic gap issues. Recent advancements in dense retrieval offer promising solutions but existing models, both general-domain and biomedical-domain, fall short due to insufficient medical knowledge or mismatched training corpora. This paper introduces \texttt{this http URL}, a series of dense retrieval models specifically tailored for EHR retrieval. We propose a two-stage training pipeline utilizing MIMIC-IV discharge summaries to address the need for extensive medical knowledge and large-scale training data. The first stage involves medical entity extraction and knowledge injection from a biomedical knowledge graph, while the second stage employs large language models to generate diverse training data. We train two variants of \texttt{this http URL}, with 110M and 7B parameters, respectively. Evaluated on the CliniQ benchmark, our models significantly outperforms all existing dense retrievers, achieving state-of-the-art results. Detailed analyses confirm our models' superiority across various match and query types, particularly in challenging semantic matches like implication and abbreviation. Ablation studies validate the effectiveness of each pipeline component, and supplementary experiments on EHR QA datasets demonstrate the models' generalizability on natural language questions, including complex ones with multiple entities. This work significantly advances EHR retrieval, offering a robust solution for clinical applications. 

---
# SafeWork-R1: Coevolving Safety and Intelligence under the AI-45$^{\circ}$ Law 

**Authors**: Shanghai AI Lab, Yicheng Bao, Guanxu Chen, Mingkang Chen, Yunhao Chen, Chiyu Chen, Lingjie Chen, Sirui Chen, Xinquan Chen, Jie Cheng, Yu Cheng, Dengke Deng, Yizhuo Ding, Dan Ding, Xiaoshan Ding, Yi Ding, Zhichen Dong, Lingxiao Du, Yuyu Fan, Xinshun Feng, Yanwei Fu, Yuxuan Gao, Ruijun Ge, Tianle Gu, Lujun Gui, Jiaxuan Guo, Qianxi He, Yuenan Hou, Xuhao Hu, Hong Huang, Kaichen Huang, Shiyang Huang, Yuxian Jiang, Shanzhe Lei, Jie Li, Lijun Li, Hao Li, Juncheng Li, Xiangtian Li, Yafu Li, Lingyu Li, Xueyan Li, Haotian Liang, Dongrui Liu, Qihua Liu, Zhixuan Liu, Bangwei Liu, Huacan Liu, Yuexiao Liu, Zongkai Liu, Chaochao Lu, Yudong Lu, Xiaoya Lu, Zhenghao Lu, Qitan Lv, Caoyuan Ma, Jiachen Ma, Xiaoya Ma, Zhongtian Ma, Lingyu Meng, Ziqi Miao, Yazhe Niu, Yuezhang Peng, Yuan Pu, Han Qi, Chen Qian, Xingge Qiao, Jingjing Qu, Jiashu Qu, Wanying Qu, Wenwen Qu, Xiaoye Qu, Qihan Ren, Qingnan Ren, Qingyu Ren, Jing Shao, Wenqi Shao, Shuai Shao, Dongxing Shi, Xin Song, Xinhao Song, Yan Teng, Xuan Tong, Yingchun Wang, Xuhong Wang, Shujie Wang, Xin Wang, Yige Wang, Yixu Wang, Yuanfu Wang, Futing Wang, Ruofan Wang, Wenjie Wang, Yajie Wang, Muhao Wei, Xiaoyu Wen, Fenghua Weng, Yuqi Wu, Yingtong Xiong, Xingcheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18576)  

**Abstract**: We introduce SafeWork-R1, a cutting-edge multimodal reasoning model that demonstrates the coevolution of capabilities and safety. It is developed by our proposed SafeLadder framework, which incorporates large-scale, progressive, safety-oriented reinforcement learning post-training, supported by a suite of multi-principled verifiers. Unlike previous alignment methods such as RLHF that simply learn human preferences, SafeLadder enables SafeWork-R1 to develop intrinsic safety reasoning and self-reflection abilities, giving rise to safety `aha' moments. Notably, SafeWork-R1 achieves an average improvement of $46.54\%$ over its base model Qwen2.5-VL-72B on safety-related benchmarks without compromising general capabilities, and delivers state-of-the-art safety performance compared to leading proprietary models such as GPT-4.1 and Claude Opus 4. To further bolster its reliability, we implement two distinct inference-time intervention methods and a deliberative search mechanism, enforcing step-level verification. Finally, we further develop SafeWork-R1-InternVL3-78B, SafeWork-R1-DeepSeek-70B, and SafeWork-R1-Qwen2.5VL-7B. All resulting models demonstrate that safety and capability can co-evolve synergistically, highlighting the generalizability of our framework in building robust, reliable, and trustworthy general-purpose AI. 

---
# PosterMate: Audience-driven Collaborative Persona Agents for Poster Design 

**Authors**: Donghoon Shin, Daniel Lee, Gary Hsieh, Gromit Yeuk-Yin Chan  

**Link**: [PDF](https://arxiv.org/pdf/2507.18572)  

**Abstract**: Poster designing can benefit from synchronous feedback from target audiences. However, gathering audiences with diverse perspectives and reconciling them on design edits can be challenging. Recent generative AI models present opportunities to simulate human-like interactions, but it is unclear how they may be used for feedback processes in design. We introduce PosterMate, a poster design assistant that facilitates collaboration by creating audience-driven persona agents constructed from marketing documents. PosterMate gathers feedback from each persona agent regarding poster components, and stimulates discussion with the help of a moderator to reach a conclusion. These agreed-upon edits can then be directly integrated into the poster design. Through our user study (N=12), we identified the potential of PosterMate to capture overlooked viewpoints, while serving as an effective prototyping tool. Additionally, our controlled online evaluation (N=100) revealed that the feedback from an individual persona agent is appropriate given its persona identity, and the discussion effectively synthesizes the different persona agents' perspectives. 

---
# LLM-based Embedders for Prior Case Retrieval 

**Authors**: Damith Premasiri, Tharindu Ranasinghe, Ruslan Mitkov  

**Link**: [PDF](https://arxiv.org/pdf/2507.18455)  

**Abstract**: In common law systems, legal professionals such as lawyers and judges rely on precedents to build their arguments. As the volume of cases has grown massively over time, effectively retrieving prior cases has become essential. Prior case retrieval (PCR) is an information retrieval (IR) task that aims to automatically identify the most relevant court cases for a specific query from a large pool of potential candidates. While IR methods have seen several paradigm shifts over the last few years, the vast majority of PCR methods continue to rely on traditional IR methods, such as BM25. The state-of-the-art deep learning IR methods have not been successful in PCR due to two key challenges: i. Lengthy legal text limitation; when using the powerful BERT-based transformer models, there is a limit of input text lengths, which inevitably requires to shorten the input via truncation or division with a loss of legal context information. ii. Lack of legal training data; due to data privacy concerns, available PCR datasets are often limited in size, making it difficult to train deep learning-based models effectively. In this research, we address these challenges by leveraging LLM-based text embedders in PCR. LLM-based embedders support longer input lengths, and since we use them in an unsupervised manner, they do not require training data, addressing both challenges simultaneously. In this paper, we evaluate state-of-the-art LLM-based text embedders in four PCR benchmark datasets and show that they outperform BM25 and supervised transformer-based models. 

---
# LoRA-Leak: Membership Inference Attacks Against LoRA Fine-tuned Language Models 

**Authors**: Delong Ran, Xinlei He, Tianshuo Cong, Anyu Wang, Qi Li, Xiaoyun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18302)  

**Abstract**: Language Models (LMs) typically adhere to a "pre-training and fine-tuning" paradigm, where a universal pre-trained model can be fine-tuned to cater to various specialized domains. Low-Rank Adaptation (LoRA) has gained the most widespread use in LM fine-tuning due to its lightweight computational cost and remarkable performance. Because the proportion of parameters tuned by LoRA is relatively small, there might be a misleading impression that the LoRA fine-tuning data is invulnerable to Membership Inference Attacks (MIAs). However, we identify that utilizing the pre-trained model can induce more information leakage, which is neglected by existing MIAs. Therefore, we introduce LoRA-Leak, a holistic evaluation framework for MIAs against the fine-tuning datasets of LMs. LoRA-Leak incorporates fifteen membership inference attacks, including ten existing MIAs, and five improved MIAs that leverage the pre-trained model as a reference. In experiments, we apply LoRA-Leak to three advanced LMs across three popular natural language processing tasks, demonstrating that LoRA-based fine-tuned LMs are still vulnerable to MIAs (e.g., 0.775 AUC under conservative fine-tuning settings). We also applied LoRA-Leak to different fine-tuning settings to understand the resulting privacy risks. We further explore four defenses and find that only dropout and excluding specific LM layers during fine-tuning effectively mitigate MIA risks while maintaining utility. We highlight that under the "pre-training and fine-tuning" paradigm, the existence of the pre-trained model makes MIA a more severe risk for LoRA-based LMs. We hope that our findings can provide guidance on data privacy protection for specialized LM providers. 

---
# Recent Trends in Distant Conversational Speech Recognition: A Review of CHiME-7 and 8 DASR Challenges 

**Authors**: Samuele Cornell, Christoph Boeddeker, Taejin Park, He Huang, Desh Raj, Matthew Wiesner, Yoshiki Masuyama, Xuankai Chang, Zhong-Qiu Wang, Stefano Squartini, Paola Garcia, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2507.18161)  

**Abstract**: The CHiME-7 and 8 distant speech recognition (DASR) challenges focus on multi-channel, generalizable, joint automatic speech recognition (ASR) and diarization of conversational speech. With participation from 9 teams submitting 32 diverse systems, these challenges have contributed to state-of-the-art research in the field. This paper outlines the challenges' design, evaluation metrics, datasets, and baseline systems while analyzing key trends from participant submissions. From this analysis it emerges that: 1) Most participants use end-to-end (e2e) ASR systems, whereas hybrid systems were prevalent in previous CHiME challenges. This transition is mainly due to the availability of robust large-scale pre-trained models, which lowers the data burden for e2e-ASR. 2) Despite recent advances in neural speech separation and enhancement (SSE), all teams still heavily rely on guided source separation, suggesting that current neural SSE techniques are still unable to reliably deal with complex scenarios and different recording setups. 3) All best systems employ diarization refinement via target-speaker diarization techniques. Accurate speaker counting in the first diarization pass is thus crucial to avoid compounding errors and CHiME-8 DASR participants especially focused on this part. 4) Downstream evaluation via meeting summarization can correlate weakly with transcription quality due to the remarkable effectiveness of large-language models in handling errors. On the NOTSOFAR-1 scenario, even systems with over 50\% time-constrained minimum permutation WER can perform roughly on par with the most effective ones (around 11\%). 5) Despite recent progress, accurately transcribing spontaneous speech in challenging acoustic environments remains difficult, even when using computationally intensive system ensembles. 

---
# Agentic AI framework for End-to-End Medical Data Inference 

**Authors**: Soorya Ram Shimgekar, Shayan Vassef, Abhay Goyal, Navin Kumar, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2507.18115)  

**Abstract**: Building and deploying machine learning solutions in healthcare remains expensive and labor-intensive due to fragmented preprocessing workflows, model compatibility issues, and stringent data privacy constraints. In this work, we introduce an Agentic AI framework that automates the entire clinical data pipeline, from ingestion to inference, through a system of modular, task-specific agents. These agents handle both structured and unstructured data, enabling automatic feature selection, model selection, and preprocessing recommendation without manual intervention. We evaluate the system on publicly available datasets from geriatrics, palliative care, and colonoscopy imaging. For example, in the case of structured data (anxiety data) and unstructured data (colonoscopy polyps data), the pipeline begins with file-type detection by the Ingestion Identifier Agent, followed by the Data Anonymizer Agent ensuring privacy compliance, where we first identify the data type and then anonymize it. The Feature Extraction Agent identifies features using an embedding-based approach for tabular data, extracting all column names, and a multi-stage MedGemma-based approach for image data, which infers modality and disease name. These features guide the Model-Data Feature Matcher Agent in selecting the best-fit model from a curated repository. The Preprocessing Recommender Agent and Preprocessing Implementor Agent then apply tailored preprocessing based on data type and model requirements. Finally, the ``Model Inference Agent" runs the selected model on the uploaded data and generates interpretable outputs using tools like SHAP, LIME, and DETR attention maps. By automating these high-friction stages of the ML lifecycle, the proposed framework reduces the need for repeated expert intervention, offering a scalable, cost-efficient pathway for operationalizing AI in clinical environments. 

---
# Group Sequence Policy Optimization 

**Authors**: Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.18071)  

**Abstract**: This paper introduces Group Sequence Policy Optimization (GSPO), our stable, efficient, and performant reinforcement learning algorithm for training large language models. Unlike previous algorithms that adopt token-level importance ratios, GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization. We demonstrate that GSPO achieves superior training efficiency and performance compared to the GRPO algorithm, notably stabilizes Mixture-of-Experts (MoE) RL training, and has the potential for simplifying the design of RL infrastructure. These merits of GSPO have contributed to the remarkable improvements in the latest Qwen3 models. 

---
# RECALLED: An Unbounded Resource Consumption Attack on Large Vision-Language Models 

**Authors**: Haoran Gao, Yuanhe Zhang, Zhenhong Zhou, Lei Jiang, Fanyu Meng, Yujia Xiao, Kun Wang, Yang Liu, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.18053)  

**Abstract**: Resource Consumption Attacks (RCAs) have emerged as a significant threat to the deployment of Large Language Models (LLMs). With the integration of vision modalities, additional attack vectors exacerbate the risk of RCAs in large vision-language models (LVLMs). However, existing red-teaming studies have largely overlooked visual inputs as a potential attack surface, resulting in insufficient mitigation strategies against RCAs in LVLMs. To address this gap, we propose RECALLED (\textbf{RE}source \textbf{C}onsumption \textbf{A}ttack on \textbf{L}arge Vision-\textbf{L}anguag\textbf{E} Mo\textbf{D}els), the first approach for exploiting visual modalities to trigger unbounded RCAs red-teaming. First, we present \textit{Vision Guided Optimization}, a fine-grained pixel-level optimization, to obtain \textit{Output Recall} adversarial perturbations, which can induce repeating output. Then, we inject the perturbations into visual inputs, triggering unbounded generations to achieve the goal of RCAs. Additionally, we introduce \textit{Multi-Objective Parallel Losses} to generate universal attack templates and resolve optimization conflicts when intending to implement parallel attacks. Empirical results demonstrate that RECALLED increases service response latency by over 26 $\uparrow$, resulting in an additional 20\% increase in GPU utilization and memory consumption. Our study exposes security vulnerabilities in LVLMs and establishes a red-teaming framework that can facilitate future defense development against RCAs. 

---
# GRR-CoCa: Leveraging LLM Mechanisms in Multimodal Model Architectures 

**Authors**: Jake R. Patock, Nicole Catherine Lewis, Kevin McCoy, Christina Gomez, Canling Chen, Lorenzo Luzi  

**Link**: [PDF](https://arxiv.org/pdf/2507.18009)  

**Abstract**: State-of-the-art (SOTA) image and text generation models are multimodal models that have many similarities to large language models (LLMs). Despite achieving strong performances, leading foundational multimodal model architectures frequently lag behind the architectural sophistication of contemporary LLMs. We propose GRR-CoCa, an improved SOTA Contrastive Captioner (CoCa) model that incorporates Gaussian error gated linear units, root mean squared normalization, and rotary positional embedding into the textual decoders and the vision transformer (ViT) encoder. Each architectural modification has been shown to improve model performance in LLMs, but has yet to be adopted in CoCa. We benchmarked GRR-CoCa against Baseline CoCa, a model with the same modified textual decoders but with CoCa's original ViT encoder. We used standard pretraining and fine-tuning workflows to benchmark the models on contrastive and generative tasks. Our GRR-CoCa significantly outperformed Baseline CoCa on the pretraining dataset and three diverse fine-tuning datasets. Pretraining improvements were 27.25% in contrastive loss, 3.71% in perplexity, and 7.15% in CoCa loss. The average fine-tuning improvements were 13.66% in contrastive loss, 5.18% in perplexity, and 5.55% in CoCa loss. We show that GRR-CoCa's modified architecture improves performance and generalization across vision-language domains. 

---
# Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation 

**Authors**: Jaechul Roh, Zachary Novack, Yuefeng Peng, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Amir Houmansadr  

**Link**: [PDF](https://arxiv.org/pdf/2507.17937)  

**Abstract**: Lyrics-to-Song (LS2) generation models promise end-to-end music synthesis from text, yet their vulnerability to training data memorization remains underexplored. We introduce Adversarial PhoneTic Prompting (APT), a novel attack where lyrics are semantically altered while preserving their acoustic structure through homophonic substitutions (e.g., Eminem's famous "mom's spaghetti" $\rightarrow$ "Bob's confetti"). Despite these distortions, we uncover a powerful form of sub-lexical memorization: models like SUNO and YuE regenerate outputs strikingly similar to known training content, achieving high similarity across audio-domain metrics, including CLAP, AudioJudge, and CoverID. This vulnerability persists across multiple languages and genres. More surprisingly, we discover that phoneme-altered lyrics alone can trigger visual memorization in text-to-video models. When prompted with phonetically modified lyrics from Lose Yourself, Veo 3 reconstructs visual elements from the original music video -- including character appearance and scene composition -- despite no visual cues in the prompt. We term this phenomenon phonetic-to-visual regurgitation. Together, these findings expose a critical vulnerability in transcript-conditioned multimodal generation: phonetic prompting alone can unlock memorized audiovisual content, raising urgent questions about copyright, safety, and content provenance in modern generative systems. Example generations are available on our demo page (this http URL). 

---
# GenSelect: A Generative Approach to Best-of-N 

**Authors**: Shubham Toshniwal, Ivan Sorokin, Aleksander Ficek, Ivan Moshkov, Igor Gitman  

**Link**: [PDF](https://arxiv.org/pdf/2507.17797)  

**Abstract**: Generative reward models with parallel sampling have enabled effective test-time scaling for reasoning tasks. Current approaches employ pointwise scoring of individual solutions or pairwise comparisons. However, pointwise methods underutilize LLMs' comparative abilities, while pairwise methods scale inefficiently with larger sampling budgets. We introduce GenSelect, where the LLM uses long reasoning to select the best solution among N candidates. This leverages LLMs' comparative strengths while scaling efficiently across parallel sampling budgets. For math reasoning, we demonstrate that reasoning models, such as QwQ and DeepSeek-R1-0528, excel at GenSelect, outperforming existing scoring approaches with simple prompting. 

---
# Exploring Communication Strategies for Collaborative LLM Agents in Mathematical Problem-Solving 

**Authors**: Liang Zhang, Xiaoming Zhai, Jionghao Lin, Jionghao Lin, Jennifer Kleiman, Diego Zapata-Rivera, Carol Forsyth, Yang Jiang, Xiangen Hu, Arthur C. Graesser  

**Link**: [PDF](https://arxiv.org/pdf/2507.17753)  

**Abstract**: Large Language Model (LLM) agents are increasingly utilized in AI-aided education to support tutoring and learning. Effective communication strategies among LLM agents improve collaborative problem-solving efficiency and facilitate cost-effective adoption in education. However, little research has systematically evaluated the impact of different communication strategies on agents' problem-solving. Our study examines four communication modes, \textit{teacher-student interaction}, \textit{peer-to-peer collaboration}, \textit{reciprocal peer teaching}, and \textit{critical debate}, in a dual-agent, chat-based mathematical problem-solving environment using the OpenAI GPT-4o model. Evaluated on the MATH dataset, our results show that dual-agent setups outperform single agents, with \textit{peer-to-peer collaboration} achieving the highest accuracy. Dialogue acts like statements, acknowledgment, and hints play a key role in collaborative problem-solving. While multi-agent frameworks enhance computational tasks, effective communication strategies are essential for tackling complex problems in AI education. 

---
