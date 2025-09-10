# Parallel-R1: Towards Parallel Thinking via Reinforcement Learning 

**Authors**: Tong Zheng, Hongming Zhang, Wenhao Yu, Xiaoyang Wang, Xinyu Yang, Runpeng Dai, Rui Liu, Huiwen Bao, Chengsong Huang, Heng Huang, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07980)  

**Abstract**: Parallel thinking has emerged as a novel approach for enhancing the reasoning capabilities of large language models (LLMs) by exploring multiple reasoning paths concurrently. However, activating such capabilities through training remains challenging, as existing methods predominantly rely on supervised fine-tuning (SFT) over synthetic data, which encourages teacher-forced imitation rather than exploration and generalization. Different from them, we propose \textbf{Parallel-R1}, the first reinforcement learning (RL) framework that enables parallel thinking behaviors for complex real-world reasoning tasks. Our framework employs a progressive curriculum that explicitly addresses the cold-start problem in training parallel thinking with RL. We first use SFT on prompt-generated trajectories from easier tasks to instill the parallel thinking ability, then transition to RL to explore and generalize this skill on harder problems. Experiments on various math benchmarks, including MATH, AMC23, and AIME, show that Parallel-R1 successfully instills parallel thinking, leading to 8.4% accuracy improvements over the sequential thinking model trained directly on challenging tasks with RL. Further analysis reveals a clear shift in the model's thinking behavior: at an early stage, it uses parallel thinking as an exploration strategy, while in a later stage, it uses the same capability for multi-perspective verification. Most significantly, we validate parallel thinking as a \textbf{mid-training exploration scaffold}, where this temporary exploratory phase unlocks a higher performance ceiling after RL, yielding a 42.9% improvement over the baseline on AIME25. Our model, data, and code will be open-source at this https URL. 

---
# SimpleQA Verified: A Reliable Factuality Benchmark to Measure Parametric Knowledge 

**Authors**: Lukas Haas, Gal Yona, Giovanni D'Antonio, Sasha Goldshtein, Dipanjan Das  

**Link**: [PDF](https://arxiv.org/pdf/2509.07968)  

**Abstract**: We introduce SimpleQA Verified, a 1,000-prompt benchmark for evaluating Large Language Model (LLM) short-form factuality based on OpenAI's SimpleQA. It addresses critical limitations in OpenAI's benchmark, including noisy and incorrect labels, topical biases, and question redundancy. SimpleQA Verified was created through a rigorous multi-stage filtering process involving de-duplication, topic balancing, and source reconciliation to produce a more reliable and challenging evaluation set, alongside improvements in the autorater prompt. On this new benchmark, Gemini 2.5 Pro achieves a state-of-the-art F1-score of 55.6, outperforming other frontier models, including GPT-5. This work provides the research community with a higher-fidelity tool to track genuine progress in parametric model factuality and to mitigate hallucinations. The benchmark dataset, evaluation code, and leaderboard are available at: this https URL. 

---
# GENUINE: Graph Enhanced Multi-level Uncertainty Estimation for Large Language Models 

**Authors**: Tuo Wang, Adithya Kulkarni, Tyler Cody, Peter A. Beling, Yujun Yan, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.07925)  

**Abstract**: Uncertainty estimation is essential for enhancing the reliability of Large Language Models (LLMs), particularly in high-stakes applications. Existing methods often overlook semantic dependencies, relying on token-level probability measures that fail to capture structural relationships within the generated text. We propose GENUINE: Graph ENhanced mUlti-level uncertaINty Estimation for Large Language Models, a structure-aware framework that leverages dependency parse trees and hierarchical graph pooling to refine uncertainty quantification. By incorporating supervised learning, GENUINE effectively models semantic and structural relationships, improving confidence assessments. Extensive experiments across NLP tasks show that GENUINE achieves up to 29% higher AUROC than semantic entropy-based approaches and reduces calibration errors by over 15%, demonstrating the effectiveness of graph-based uncertainty modeling. The code is available at this https URL. 

---
# Biased Tales: Cultural and Topic Bias in Generating Children's Stories 

**Authors**: Donya Rooein, Vilém Zouhar, Debora Nozza, Dirk Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2509.07908)  

**Abstract**: Stories play a pivotal role in human communication, shaping beliefs and morals, particularly in children. As parents increasingly rely on large language models (LLMs) to craft bedtime stories, the presence of cultural and gender stereotypes in these narratives raises significant concerns. To address this issue, we present Biased Tales, a comprehensive dataset designed to analyze how biases influence protagonists' attributes and story elements in LLM-generated stories. Our analysis uncovers striking disparities. When the protagonist is described as a girl (as compared to a boy), appearance-related attributes increase by 55.26%. Stories featuring non-Western children disproportionately emphasize cultural heritage, tradition, and family themes far more than those for Western children. Our findings highlight the role of sociocultural bias in making creative AI use more equitable and diverse. 

---
# From Detection to Mitigation: Addressing Gender Bias in Chinese Texts via Efficient Tuning and Voting-Based Rebalancing 

**Authors**: Chengyan Wu, Yiqiang Cai, Yufei Cheng, Yun Xue  

**Link**: [PDF](https://arxiv.org/pdf/2509.07889)  

**Abstract**: This paper presents our team's solution to Shared Task 7 of NLPCC-2025, which focuses on sentence-level gender bias detection and mitigation in Chinese. The task aims to promote fairness and controllability in natural language generation by automatically detecting, classifying, and mitigating gender bias. To address this challenge, we adopt a fine-tuning approach based on large language models (LLMs), efficiently adapt to the bias detection task via Low-Rank Adaptation (LoRA). In terms of data processing, we construct a more balanced training set to alleviate class imbalance and introduce heterogeneous samples from multiple sources to enhance model generalization. For the detection and classification sub-tasks, we employ a majority voting strategy that integrates outputs from multiple expert models to boost performance. Additionally, to improve bias generation detection and mitigation, we design a multi-temperature sampling mechanism to capture potential variations in bias expression styles. Experimental results demonstrate the effectiveness of our approach in bias detection, classification, and mitigation. Our method ultimately achieves an average score of 47.90%, ranking fourth in the shared task. 

---
# Are Humans as Brittle as Large Language Models? 

**Authors**: Jiahui Li, Sean Papay, Roman Klinger  

**Link**: [PDF](https://arxiv.org/pdf/2509.07869)  

**Abstract**: The output of large language models (LLM) is unstable, due to both non-determinism of the decoding process as well as to prompt brittleness. While the intrinsic non-determinism of LLM generation may mimic existing uncertainty in human annotations through distributional shifts in outputs, it is largely assumed, yet unexplored, that the prompt brittleness effect is unique to LLMs. This raises the question: do human annotators show similar sensitivity to instruction changes? If so, should prompt brittleness in LLMs be considered problematic? One may alternatively hypothesize that prompt brittleness correctly reflects human annotation variances. To fill this research gap, we systematically compare the effects of prompt modifications on LLMs and identical instruction modifications for human annotators, focusing on the question of whether humans are similarly sensitive to prompt perturbations. To study this, we prompt both humans and LLMs for a set of text classification tasks conditioned on prompt variations. Our findings indicate that both humans and LLMs exhibit increased brittleness in response to specific types of prompt modifications, particularly those involving the substitution of alternative label sets or label formats. However, the distribution of human judgments is less affected by typographical errors and reversed label order than that of LLMs. 

---
# Small Open Models Achieve Near Parity with Large Models in Low Resource Literary Translation at a Fraction of the Cost 

**Authors**: Mihai Nadas, Laura Diosan, Andreea Tomescu, Andrei Piscoran  

**Link**: [PDF](https://arxiv.org/pdf/2509.07829)  

**Abstract**: Literary translation has recently gained attention as a distinct and complex task in machine translation research. However, the translation by small open models remains an open problem. We contribute to this ongoing research by introducing TINYFABULIST TRANSLATION FRAMEWORK (TF2), a unified framework for dataset creation, fine tuning, and evaluation in English-Romanian literary translations, centred on the creation and open release of both a compact, fine tuned language model (TF2-12B) and large scale synthetic parallel datasets (DS-TF2-EN-RO-3M and DS-TF2-EN-RO-15K). Building on DS-TF1-EN-3M (TF1), the largest collection of synthetic English fables to date, we address the need for rich, high quality literary datasets in low resource languages such as Romanian. Our pipeline first generates 15k high quality Romanian references from the TF1 pool using a high performing LLM. We then apply a two stage fine tuning process to a 12B parameter open weight model: (i) instruction tuning to capture genre specific narrative style, and (ii) adapter compression for efficient deployment. Evaluation combines corpus level BLEU and a five dimension LLM based rubric (accuracy, fluency, coherence, style, cultural adaptation) to provide a nuanced assessment of translation quality. Results show that our fine tuned model achieves fluency and adequacy competitive with top performing large proprietary models, while being open, accessible, and significantly more cost effective. Alongside the fine tuned model and both datasets, we publicly release all scripts and evaluation prompts. TF2 thus provides an end-to-end, reproducible pipeline for research on cost efficient translation, cross lingual narrative generation, and the broad adoption of open models for culturally significant literary content in low resource settings. 

---
# Dual Knowledge-Enhanced Two-Stage Reasoner for Multimodal Dialog Systems 

**Authors**: Xiaolin Chen, Xuemeng Song, Haokun Wen, Weili Guan, Xiangyu Zhao, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2509.07817)  

**Abstract**: Textual response generation is pivotal for multimodal \mbox{task-oriented} dialog systems, which aims to generate proper textual responses based on the multimodal context. While existing efforts have demonstrated remarkable progress, there still exist the following limitations: 1) \textit{neglect of unstructured review knowledge} and 2) \textit{underutilization of large language models (LLMs)}. Inspired by this, we aim to fully utilize dual knowledge (\textit{i.e., } structured attribute and unstructured review knowledge) with LLMs to promote textual response generation in multimodal task-oriented dialog systems. However, this task is non-trivial due to two key challenges: 1) \textit{dynamic knowledge type selection} and 2) \textit{intention-response decoupling}. To address these challenges, we propose a novel dual knowledge-enhanced two-stage reasoner by adapting LLMs for multimodal dialog systems (named DK2R). To be specific, DK2R first extracts both structured attribute and unstructured review knowledge from external knowledge base given the dialog context. Thereafter, DK2R uses an LLM to evaluate each knowledge type's utility by analyzing LLM-generated provisional probe responses. Moreover, DK2R separately summarizes the intention-oriented key clues via dedicated reasoning, which are further used as auxiliary signals to enhance LLM-based textual response generation. Extensive experiments conducted on a public dataset verify the superiority of DK2R. We have released the codes and parameters. 

---
# SciNLP: A Domain-Specific Benchmark for Full-Text Scientific Entity and Relation Extraction in NLP 

**Authors**: Decheng Duan, Yingyi Zhang, Jitong Peng, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07801)  

**Abstract**: Structured information extraction from scientific literature is crucial for capturing core concepts and emerging trends in specialized fields. While existing datasets aid model development, most focus on specific publication sections due to domain complexity and the high cost of annotating scientific texts. To address this limitation, we introduce SciNLP - a specialized benchmark for full-text entity and relation extraction in the Natural Language Processing (NLP) domain. The dataset comprises 60 manually annotated full-text NLP publications, covering 7,072 entities and 1,826 relations. Compared to existing research, SciNLP is the first dataset providing full-text annotations of entities and their relationships in the NLP domain. To validate the effectiveness of SciNLP, we conducted comparative experiments with similar datasets and evaluated the performance of state-of-the-art supervised models on this dataset. Results reveal varying extraction capabilities of existing models across academic texts of different lengths. Cross-comparisons with existing datasets show that SciNLP achieves significant performance improvements on certain baseline models. Using models trained on SciNLP, we implemented automatic construction of a fine-grained knowledge graph for the NLP domain. Our KG has an average node degree of 3.2 per entity, indicating rich semantic topological information that enhances downstream applications. The dataset is publicly available at this https URL. 

---
# Are LLMs Enough for Hyperpartisan, Fake, Polarized and Harmful Content Detection? Evaluating In-Context Learning vs. Fine-Tuning 

**Authors**: Michele Joshua Maggini, Dhia Merzougui, Rabiraj Bandyopadhyay, Gaël Dias, Fabrice Maurel, Pablo Gamallo  

**Link**: [PDF](https://arxiv.org/pdf/2509.07768)  

**Abstract**: The spread of fake news, polarizing, politically biased, and harmful content on online platforms has been a serious concern. With large language models becoming a promising approach, however, no study has properly benchmarked their performance across different models, usage methods, and languages. This study presents a comprehensive overview of different Large Language Models adaptation paradigms for the detection of hyperpartisan and fake news, harmful tweets, and political bias. Our experiments spanned 10 datasets and 5 different languages (English, Spanish, Portuguese, Arabic and Bulgarian), covering both binary and multiclass classification scenarios. We tested different strategies ranging from parameter efficient Fine-Tuning of language models to a variety of different In-Context Learning strategies and prompts. These included zero-shot prompts, codebooks, few-shot (with both randomly-selected and diversely-selected examples using Determinantal Point Processes), and Chain-of-Thought. We discovered that In-Context Learning often underperforms when compared to Fine-Tuning a model. This main finding highlights the importance of Fine-Tuning even smaller models on task-specific settings even when compared to the largest models evaluated in an In-Context Learning setup - in our case LlaMA3.1-8b-Instruct, Mistral-Nemo-Instruct-2407 and Qwen2.5-7B-Instruct. 

---
# Factuality Beyond Coherence: Evaluating LLM Watermarking Methods for Medical Texts 

**Authors**: Rochana Prih Hastuti, Rian Adam Rajagede, Mansour Al Ghanim, Mengxin Zheng, Qian Lou  

**Link**: [PDF](https://arxiv.org/pdf/2509.07755)  

**Abstract**: As large language models (LLMs) adapted to sensitive domains such as medicine, their fluency raises safety risks, particularly regarding provenance and accountability. Watermarking embeds detectable patterns to mitigate these risks, yet its reliability in medical contexts remains untested. Existing benchmarks focus on detection-quality tradeoffs, overlooking factual risks under low-entropy settings often exploited by watermarking's reweighting strategy. We propose a medical-focused evaluation workflow that jointly assesses factual accuracy and coherence. Using GPT-Judger and further human validation, we introduce the Factuality-Weighted Score (FWS), a composite metric prioritizing factual accuracy beyond coherence to guide watermarking deployment in medical domains. Our evaluation shows current watermarking methods substantially compromise medical factuality, with entropy shifts degrading medical entity representation. These findings underscore the need for domain-aware watermarking approaches that preserve the integrity of medical content. 

---
# M-BRe: Discovering Training Samples for Relation Extraction from Unlabeled Texts with Large Language Models 

**Authors**: Zexuan Li, Hongliang Dai, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.07730)  

**Abstract**: For Relation Extraction (RE), the manual annotation of training data may be prohibitively expensive, since the sentences that contain the target relations in texts can be very scarce and difficult to find. It is therefore beneficial to develop an efficient method that can automatically extract training instances from unlabeled texts for training RE models. Recently, large language models (LLMs) have been adopted in various natural language processing tasks, with RE also benefiting from their advances. However, when leveraging LLMs for RE with predefined relation categories, two key challenges arise. First, in a multi-class classification setting, LLMs often struggle to comprehensively capture the semantics of every relation, leading to suboptimal results. Second, although employing binary classification for each relation individually can mitigate this issue, it introduces significant computational overhead, resulting in impractical time complexity for real-world applications. Therefore, this paper proposes a framework called M-BRe to extract training instances from unlabeled texts for RE. It utilizes three modules to combine the advantages of both of the above classification approaches: Relation Grouping, Relation Extraction, and Label Decision. Extensive experiments confirm its superior capability in discovering high-quality training samples from unlabeled texts for RE. 

---
# MoLoRAG: Bootstrapping Document Understanding via Multi-modal Logic-aware Retrieval 

**Authors**: Xixi Wu, Yanchao Tan, Nan Hou, Ruiyang Zhang, Hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.07666)  

**Abstract**: Document Understanding is a foundational AI capability with broad applications, and Document Question Answering (DocQA) is a key evaluation task. Traditional methods convert the document into text for processing by Large Language Models (LLMs), but this process strips away critical multi-modal information like figures. While Large Vision-Language Models (LVLMs) address this limitation, their constrained input size makes multi-page document comprehension infeasible. Retrieval-augmented generation (RAG) methods mitigate this by selecting relevant pages, but they rely solely on semantic relevance, ignoring logical connections between pages and the query, which is essential for reasoning.
To this end, we propose MoLoRAG, a logic-aware retrieval framework for multi-modal, multi-page document understanding. By constructing a page graph that captures contextual relationships between pages, a lightweight VLM performs graph traversal to retrieve relevant pages, including those with logical connections often overlooked. This approach combines semantic and logical relevance to deliver more accurate retrieval. After retrieval, the top-$K$ pages are fed into arbitrary LVLMs for question answering. To enhance flexibility, MoLoRAG offers two variants: a training-free solution for easy deployment and a fine-tuned version to improve logical relevance checking. Experiments on four DocQA datasets demonstrate average improvements of 9.68% in accuracy over LVLM direct inference and 7.44% in retrieval precision over baselines. Codes and datasets are released at this https URL. 

---
# MaLei at MultiClinSUM: Summarisation of Clinical Documents using Perspective-Aware Iterative Self-Prompting with LLMs 

**Authors**: Libo Ren, Yee Man Ng, Lifeng Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.07622)  

**Abstract**: Efficient communication between patients and clinicians plays an important role in shared decision-making. However, clinical reports are often lengthy and filled with clinical jargon, making it difficult for domain experts to identify important aspects in the document efficiently. This paper presents the methodology we applied in the MultiClinSUM shared task for summarising clinical case documents. We used an Iterative Self-Prompting technique on large language models (LLMs) by asking LLMs to generate task-specific prompts and refine them via example-based few-shot learning. Furthermore, we used lexical and embedding space metrics, ROUGE and BERT-score, to guide the model fine-tuning with epochs. Our submission using perspective-aware ISP on GPT-4 and GPT-4o achieved ROUGE scores (46.53, 24.68, 30.77) and BERTscores (87.84, 83.25, 85.46) for (P, R, F1) from the official evaluation on 3,396 clinical case reports from various specialties extracted from open journals. The high BERTscore indicates that the model produced semantically equivalent output summaries compared to the references, even though the overlap at the exact lexicon level is lower, as reflected in the lower ROUGE scores. This work sheds some light on how perspective-aware ISP (PA-ISP) can be deployed for clinical report summarisation and support better communication between patients and clinicians. 

---
# BALI: Enhancing Biomedical Language Representations through Knowledge Graph and Language Model Alignment 

**Authors**: Andrey Sakhovskiy, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2509.07588)  

**Abstract**: In recent years, there has been substantial progress in using pretrained Language Models (LMs) on a range of tasks aimed at improving the understanding of biomedical texts. Nonetheless, existing biomedical LLMs show limited comprehension of complex, domain-specific concept structures and the factual information encoded in biomedical Knowledge Graphs (KGs). In this work, we propose BALI (Biomedical Knowledge Graph and Language Model Alignment), a novel joint LM and KG pre-training method that augments an LM with external knowledge by the simultaneous learning of a dedicated KG encoder and aligning the representations of both the LM and the graph. For a given textual sequence, we link biomedical concept mentions to the Unified Medical Language System (UMLS) KG and utilize local KG subgraphs as cross-modal positive samples for these mentions. Our empirical findings indicate that implementing our method on several leading biomedical LMs, such as PubMedBERT and BioLinkBERT, improves their performance on a range of language understanding tasks and the quality of entity representations, even with minimal pre-training on a small alignment dataset sourced from PubMed scientific abstracts. 

---
# Avoiding Knowledge Edit Skipping in Multi-hop Question Answering with Guided Decomposition 

**Authors**: Yi Liu, Xiangrong Zhu, Xiangyu Liu, Wei Wei, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07555)  

**Abstract**: In a rapidly evolving world where information updates swiftly, knowledge in large language models (LLMs) becomes outdated quickly. Retraining LLMs is not a cost-effective option, making knowledge editing (KE) without modifying parameters particularly necessary. We find that although existing retrieval-augmented generation (RAG)-based KE methods excel at editing simple knowledge, they struggle with KE in multi-hop question answering due to the issue of "edit skipping", which refers to skipping the relevant edited fact in inference. In addition to the diversity of natural language expressions of knowledge, edit skipping also arises from the mismatch between the granularity of LLMs in problem-solving and the facts in the edited memory. To address this issue, we propose a novel Iterative Retrieval-Augmented Knowledge Editing method with guided decomposition (IRAKE) through the guidance from single edited facts and entire edited cases. Experimental results demonstrate that IRAKE mitigates the failure of editing caused by edit skipping and outperforms state-of-the-art methods for KE in multi-hop question answering. 

---
# VeriOS: Query-Driven Proactive Human-Agent-GUI Interaction for Trustworthy OS Agents 

**Authors**: Zheng Wu, Heyuan Huang, Xingyu Lou, Xiangmou Qu, Pengzhou Cheng, Zongru Wu, Weiwen Liu, Weinan Zhang, Jun Wang, Zhaoxiang Wang, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07553)  

**Abstract**: With the rapid progress of multimodal large language models, operating system (OS) agents become increasingly capable of automating tasks through on-device graphical user interfaces (GUIs). However, most existing OS agents are designed for idealized settings, whereas real-world environments often present untrustworthy conditions. To mitigate risks of over-execution in such scenarios, we propose a query-driven human-agent-GUI interaction framework that enables OS agents to decide when to query humans for more reliable task completion. Built upon this framework, we introduce VeriOS-Agent, a trustworthy OS agent trained with a two-stage learning paradigm that falicitate the decoupling and utilization of meta-knowledge. Concretely, VeriOS-Agent autonomously executes actions in normal conditions while proactively querying humans in untrustworthy scenarios. Experiments show that VeriOS-Agent improves the average step-wise success rate by 20.64\% in untrustworthy scenarios over the state-of-the-art, without compromising normal performance. Analysis highlights VeriOS-Agent's rationality, generalizability, and scalability. The codes, datasets and models are available at this https URL. 

---
# ALLabel: Three-stage Active Learning for LLM-based Entity Recognition using Demonstration Retrieval 

**Authors**: Zihan Chen, Lei Shi, Weize Wu, Qiji Zhou, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07512)  

**Abstract**: Many contemporary data-driven research efforts in the natural sciences, such as chemistry and materials science, require large-scale, high-performance entity recognition from scientific datasets. Large language models (LLMs) have increasingly been adopted to solve the entity recognition task, with the same trend being observed on all-spectrum NLP tasks. The prevailing entity recognition LLMs rely on fine-tuned technology, yet the fine-tuning process often incurs significant cost. To achieve a best performance-cost trade-off, we propose ALLabel, a three-stage framework designed to select the most informative and representative samples in preparing the demonstrations for LLM modeling. The annotated examples are used to construct a ground-truth retrieval corpus for LLM in-context learning. By sequentially employing three distinct active learning strategies, ALLabel consistently outperforms all baselines under the same annotation budget across three specialized domain datasets. Experimental results also demonstrate that selectively annotating only 5\%-10\% of the dataset with ALLabel can achieve performance comparable to the method annotating the entire dataset. Further analyses and ablation studies verify the effectiveness and generalizability of our proposal. 

---
# HALT-RAG: A Task-Adaptable Framework for Hallucination Detection with Calibrated NLI Ensembles and Abstention 

**Authors**: Saumya Goswami, Siddharth Kurra  

**Link**: [PDF](https://arxiv.org/pdf/2509.07475)  

**Abstract**: Detecting content that contradicts or is unsupported by a given source text is a critical challenge for the safe deployment of generative language models. We introduce HALT-RAG, a post-hoc verification system designed to identify hallucinations in the outputs of Retrieval-Augmented Generation (RAG) pipelines. Our flexible and task-adaptable framework uses a universal feature set derived from an ensemble of two frozen, off-the-shelf Natural Language Inference (NLI) models and lightweight lexical signals. These features are used to train a simple, calibrated, and task-adapted meta-classifier. Using a rigorous 5-fold out-of-fold (OOF) training protocol to prevent data leakage and produce unbiased estimates, we evaluate our system on the HaluEval benchmark. By pairing our universal feature set with a lightweight, task-adapted classifier and a precision-constrained decision policy, HALT-RAG achieves strong OOF F1-scores of 0.7756, 0.9786, and 0.7391 on the summarization, QA, and dialogue tasks, respectively. The system's well-calibrated probabilities enable a practical abstention mechanism, providing a reliable tool for balancing model performance with safety requirements. 

---
# From Scarcity to Efficiency: Investigating the Effects of Data Augmentation on African Machine Translation 

**Authors**: Mardiyyah Oduwole, Oluwatosin Olajide, Jamiu Suleiman, Faith Hunja, Busayo Awobade, Fatimo Adebanjo, Comfort Akanni, Chinonyelum Igwe, Peace Ododo, Promise Omoigui, Steven Kolawole, Abraham Owodunni  

**Link**: [PDF](https://arxiv.org/pdf/2509.07471)  

**Abstract**: The linguistic diversity across the African continent presents different challenges and opportunities for machine translation. This study explores the effects of data augmentation techniques in improving translation systems in low-resource African languages. We focus on two data augmentation techniques: sentence concatenation with back translation and switch-out, applying them across six African languages. Our experiments show significant improvements in machine translation performance, with a minimum increase of 25\% in BLEU score across all six this http URL provide a comprehensive analysis and highlight the potential of these techniques to improve machine translation systems for low-resource languages, contributing to the development of more robust translation systems for under-resourced languages. 

---
# Understanding Stigmatizing Language Lexicons: A Comparative Analysis in Clinical Contexts 

**Authors**: Yiliang Zhou, Di Hu, Tianchu Lyu, Jasmine Dhillon, Alexandra L. Beck, Gelareh Sadigh, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.07462)  

**Abstract**: Stigmatizing language results in healthcare inequities, yet there is no universally accepted or standardized lexicon defining which words, terms, or phrases constitute stigmatizing language in healthcare. We conducted a systematic search of the literature to identify existing stigmatizing language lexicons and then analyzed them comparatively to examine: 1) similarities and discrepancies between these lexicons, and 2) the distribution of positive, negative, or neutral terms based on an established sentiment dataset. Our search identified four lexicons. The analysis results revealed moderate semantic similarity among them, and that most stigmatizing terms are related to judgmental expressions by clinicians to describe perceived negative behaviors. Sentiment analysis showed a predominant proportion of negatively classified terms, though variations exist across lexicons. Our findings underscore the need for a standardized lexicon and highlight challenges in defining stigmatizing language in clinical texts. 

---
# AIxcellent Vibes at GermEval 2025 Shared Task on Candy Speech Detection: Improving Model Performance by Span-Level Training 

**Authors**: Christian Rene Thelen, Patrick Gustav Blaneck, Tobias Bornheim, Niklas Grieger, Stephan Bialonski  

**Link**: [PDF](https://arxiv.org/pdf/2509.07459)  

**Abstract**: Positive, supportive online communication in social media (candy speech) has the potential to foster civility, yet automated detection of such language remains underexplored, limiting systematic analysis of its impact. We investigate how candy speech can be reliably detected in a 46k-comment German YouTube corpus by monolingual and multilingual language models, including GBERT, Qwen3 Embedding, and XLM-RoBERTa. We find that a multilingual XLM-RoBERTa-Large model trained to detect candy speech at the span level outperforms other approaches, ranking first in both binary positive F1: 0.8906) and categorized span-based detection (strict F1: 0.6307) subtasks at the GermEval 2025 Shared Task on Candy Speech Detection. We speculate that span-based training, multilingual capabilities, and emoji-aware tokenizers improved detection performance. Our results demonstrate the effectiveness of multilingual models in identifying positive, supportive language. 

---
# LongEmotion: Measuring Emotional Intelligence of Large Language Models in Long-Context Interaction 

**Authors**: Weichu Liu, Jing Xiong, Yuxuan Hu, Zixuan Li, Minghuan Tan, Ningning Mao, Chenyang Zhao, Zhongwei Wan, Chaofan Tao, Wendong Xu, Hui Shen, Chengming Li, Lingpeng Kong, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2509.07403)  

**Abstract**: Large language models (LLMs) make significant progress in Emotional Intelligence (EI) and long-context understanding. However, existing benchmarks tend to overlook certain aspects of EI in long-context scenarios, especially under realistic, practical settings where interactions are lengthy, diverse, and often noisy. To move towards such realistic settings, we present LongEmotion, a benchmark specifically designed for long-context EI tasks. It covers a diverse set of tasks, including Emotion Classification, Emotion Detection, Emotion QA, Emotion Conversation, Emotion Summary, and Emotion Expression. On average, the input length for these tasks reaches 8,777 tokens, with long-form generation required for Emotion Expression. To enhance performance under realistic constraints, we incorporate Retrieval-Augmented Generation (RAG) and Collaborative Emotional Modeling (CoEM), and compare them with standard prompt-based methods. Unlike conventional approaches, our RAG method leverages both the conversation context and the large language model itself as retrieval sources, avoiding reliance on external knowledge bases. The CoEM method further improves performance by decomposing the task into five stages, integrating both retrieval augmentation and limited knowledge injection. Experimental results show that both RAG and CoEM consistently enhance EI-related performance across most long-context tasks, advancing LLMs toward more practical and real-world EI applications. Furthermore, we conducted a comparative case study experiment on the GPT series to demonstrate the differences among various models in terms of EI. Code is available on GitHub at this https URL, and the project page can be found at this https URL. 

---
# The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering 

**Authors**: Yi-Jie Cheng, Oscar Chew, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.07399)  

**Abstract**: Integrating knowledge graphs (KGs) into the reasoning processes of large language models (LLMs) has emerged as a promising approach to mitigate hallucination. However, existing work in this area often relies on proprietary or extremely large models, limiting accessibility and scalability. In this study, we investigate the capabilities of existing integration methods for small language models (SLMs) in KG-based question answering and observe that their performance is often constrained by their limited ability to traverse and reason over knowledge graphs. To address this limitation, we propose leveraging simple and efficient exploration modules to handle knowledge graph traversal in place of the language model itself. Experiment results demonstrate that these lightweight modules effectively improve the performance of small language models on knowledge graph question answering tasks. Source code: this https URL. 

---
# Talking with Oompa Loompas: A novel framework for evaluating linguistic acquisition of LLM agents 

**Authors**: Sankalp Tattwadarshi Swain, Anshika Krishnatray, Dhruv Kumar, Jagat Sesh Challa  

**Link**: [PDF](https://arxiv.org/pdf/2509.07389)  

**Abstract**: Existing evaluation studies on linguistic competence of large language models (LLM agents) have focused primarily on vocabulary learning, morphological rule induction, syntactic generalization, pragmatic inference, and cross-linguistic transfer. However, none assess whether LLM agents can acquire a language through pattern recognition and interactive feedback, a central feature of human language acquisition. We propose a novel experimental framework in which an LLM agent is evaluated on its ability to acquire and use a newly constructed language (Tinkatongue) in conversation with a bot that understands only Tinkatongue. Our findings show that LLM agents fail to establish a conversation within 100 responses, yet they adopt distinct strategies that mirror human approaches to language learning. The results suggest a new direction for evaluation benchmarks and open pathways to model designs that learn more effectively from interactive feedback. 

---
# PersonaFuse: A Personality Activation-Driven Framework for Enhancing Human-LLM Interactions 

**Authors**: Yixuan Tang, Yi Yang, Ahmed Abbasi  

**Link**: [PDF](https://arxiv.org/pdf/2509.07370)  

**Abstract**: Recent advancements in Large Language Models (LLMs) demonstrate remarkable capabilities across various fields. These developments have led to more direct communication between humans and LLMs in various situations, such as social companionship and psychological support. However, LLMs often exhibit limitations in emotional perception and social competence during real-world conversations. These limitations partly originate from their inability to adapt their communication style and emotional expression to different social and task contexts. In this work, we introduce PersonaFuse, a novel LLM post-training framework that enables LLMs to adapt and express different personalities for varying situations. Inspired by Trait Activation Theory and the Big Five personality model, PersonaFuse employs a Mixture-of-Expert architecture that combines persona adapters with a dynamic routing network, enabling contextual trait expression. Experimental results show that PersonaFuse substantially outperforms baseline models across multiple dimensions of social-emotional intelligence. Importantly, these gains are achieved without sacrificing general reasoning ability or model safety, which remain common limitations of direct prompting and supervised fine-tuning approaches. PersonaFuse also delivers consistent improvements in downstream human-centered applications, such as mental health counseling and review-based customer service. Finally, human preference evaluations against leading LLMs, including GPT-4o and DeepSeek, demonstrate that PersonaFuse achieves competitive response quality despite its comparatively smaller model size. These findings demonstrate that PersonaFuse~offers a theoretically grounded and practical approach for developing social-emotional enhanced LLMs, marking a significant advancement toward more human-centric AI systems. 

---
# Mitigating Attention Localization in Small Scale: Self-Attention Refinement via One-step Belief Propagation 

**Authors**: Nakyung Lee, Yeongoon Kim, Minhae Oh, Suhwan Kim, Jin Woo Koo, Hyewon Jo, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.07324)  

**Abstract**: Transformer-based self-attention mechanism serves as the core of modern language models, yet it often suffers from localization, where attentions collapse onto a limited subset of tokens and fail to capture long-range dependencies. To address this issue, we propose Self-Attention One-step Belief Propagation (SAOBP), a refinement framework that injects multi-hop relationships through a belief propagation process. To interpret and quantify these interactions, we introduce Global Token Dependency (GTD) that captures the relative contribution of multihop connections within the attention graph. Empirical results indicate that SAOBP helps prevent entropy collapse in deeper layers and adaptively maintains GTD at task-appropriate levels, thereby supporting improvements in model performance. Importantly, we observe competitive gains in small-scale models, highlighting its potential for improving inference quality in resource-constrained scenarios. 

---
# Does This Look Familiar to You? Knowledge Analysis via Model Internal Representations 

**Authors**: Sihyun Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.07311)  

**Abstract**: Recent advances in large language models (LLMs) have been driven by pretraining, supervised fine tuning (SFT), and alignment tuning. Among these, SFT plays a crucial role in transforming a model 's general knowledge into structured responses tailored to specific tasks. However, there is no clearly established methodology for effective training data selection. Simply increasing the volume of data does not guarantee performance improvements, while preprocessing, sampling, and validation require substantial time and cost.
To address this issue, a variety of data selection methods have been proposed. Among them, knowledge based selection approaches identify suitable training data by analyzing the model 's responses. Nevertheless, these methods typically rely on prompt engineering, making them sensitive to variations and incurring additional costs for prompt design.
In this study, we propose Knowledge Analysis via Model Internal Representations (KAMIR), a novel approach that overcomes these limitations by analyzing data based on the model 's internal representations. KAMIR computes similarities between the hidden states of each layer (block) and the final hidden states for a given input to assess the data. Unlike prior methods that were largely limited to multiple choice tasks, KAMIR can be applied to a wide range of tasks such as machine reading comprehension and summarization. Moreover, it selects data useful for training based on the model 's familiarity with the input, even with a small dataset and a simple classifier architecture. Experiments across diverse task datasets demonstrate that training with less familiar data leads to better generalization performance. 

---
# Instance-level Performance Prediction for Long-form Generation Tasks 

**Authors**: Chi-Yang Hsu, Alexander Braylan, Yiheng Su, Omar Alonso, Matthew Lease  

**Link**: [PDF](https://arxiv.org/pdf/2509.07309)  

**Abstract**: We motivate and share a new benchmark for instance-level performance prediction of long-form generation tasks having multi-faceted, fine-grained quality metrics. Our task-, model- and metric-agnostic formulation predicts continuous evaluation metric scores given only black-box model inputs and outputs. Beyond predicting point estimates of metric scores, the benchmark also requires inferring prediction intervals to quantify uncertainty around point estimates. Evaluation spans 11 long-form datasets/tasks with multiple LLMs, baselines, and metrics per task. We show that scores can be effectively predicted across long-form generation tasks using as few as 16 training examples. Overall, we introduce a novel and useful task, a valuable benchmark to drive progress, and baselines ready for practical adoption today. 

---
# Basis Vector Metric: A Method for Robust Open-Ended State Change Detection 

**Authors**: David Oprea, Sam Powers  

**Link**: [PDF](https://arxiv.org/pdf/2509.07308)  

**Abstract**: We test a new method, which we will abbreviate using the acronym BVM (Basis Vectors Method), in its ability to judge the state changes in images through using language embeddings. We used the MIT-States dataset, containing about 53,000 images, to gather all of our data, which has 225 nouns and 115 adjectives, with each noun having about 9 different adjectives, forming approximately 1000 noun-adjective pairs. For our first experiment, we test our method's ability to determine the state of each noun class separately against other metrics for comparison. These metrics are cosine similarity, dot product, product quantization, binary index, Naive Bayes, and a custom neural network. Among these metrics, we found that our proposed BVM performs the best in classifying the states for each noun. We then perform a second experiment where we try using BVM to determine if it can differentiate adjectives from one another for each adjective separately. We compared the abilities of BVM to differentiate adjectives against the proposed method the MIT-States paper suggests: using a logistic regression model. In the end, we did not find conclusive evidence that our BVM metric could perform better than the logistic regression model at discerning adjectives. Yet, we were able to find evidence for possible improvements to our method; this leads to the chance of increasing our method's accuracy through certain changes in our methodologies. 

---
# Causal Attention with Lookahead Keys 

**Authors**: Zhuoqing Song, Peng Sun, Huizhuo Yuan, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07301)  

**Abstract**: In standard causal attention, each token's query, key, and value (QKV) are static and encode only preceding context. We introduce CAuSal aTtention with Lookahead kEys (CASTLE), an attention mechanism that continually updates each token's keys as the context unfolds. We term these updated keys lookahead keys because they belong to earlier positions yet integrate information from tokens that appear later relative to those positions, while strictly preserving the autoregressive property. Although the mechanism appears sequential, we derive a mathematical equivalence that avoids explicitly materializing lookahead keys at each position and enables efficient parallel training. On language modeling benchmarks, CASTLE consistently outperforms standard causal attention across model scales, reducing validation perplexity and improving performance on a range of downstream tasks. 

---
# LLM Analysis of 150+ years of German Parliamentary Debates on Migration Reveals Shift from Post-War Solidarity to Anti-Solidarity in the Last Decade 

**Authors**: Aida Kostikova, Ole Pütz, Steffen Eger, Olga Sabelfeld, Benjamin Paassen  

**Link**: [PDF](https://arxiv.org/pdf/2509.07274)  

**Abstract**: Migration has been a core topic in German political debate, from millions of expellees post World War II over labor migration to refugee movements in the recent past. Studying political speech regarding such wide-ranging phenomena in depth traditionally required extensive manual annotations, limiting the scope of analysis to small subsets of the data. Large language models (LLMs) have the potential to partially automate even complex annotation tasks. We provide an extensive evaluation of a multiple LLMs in annotating (anti-)solidarity subtypes in German parliamentary debates compared to a large set of thousands of human reference annotations (gathered over a year). We evaluate the influence of model size, prompting differences, fine-tuning, historical versus contemporary data; and we investigate systematic errors. Beyond methodological evaluation, we also interpret the resulting annotations from a social science lense, gaining deeper insight into (anti-)solidarity trends towards migrants in the German post-World War II period and recent past. Our data reveals a high degree of migrant-directed solidarity in the postwar period, as well as a strong trend towards anti-solidarity in the German parliament since 2015, motivating further research. These findings highlight the promise of LLMs for political text analysis and the importance of migration debates in Germany, where demographic decline and labor shortages coexist with rising polarization. 

---
# Rule-Based Moral Principles for Explaining Uncertainty in Natural Language Generation 

**Authors**: Zahra Atf, Peter R Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2509.07190)  

**Abstract**: Large language models (LLMs) are increasingly used in high-stakes settings, where explaining uncertainty is both technical and ethical. Probabilistic methods are often opaque and misaligned with expectations of transparency. We propose a framework based on rule-based moral principles for handling uncertainty in LLM-generated text. Using insights from moral psychology and virtue ethics, we define rules such as precaution, deference, and responsibility to guide responses under epistemic or aleatoric uncertainty. These rules are encoded in a lightweight Prolog engine, where uncertainty levels (low, medium, high) trigger aligned system actions with plain-language rationales. Scenario-based simulations benchmark rule coverage, fairness, and trust calibration. Use cases in clinical and legal domains illustrate how moral reasoning can improve trust and interpretability. Our approach offers a transparent, lightweight alternative to probabilistic models for socially responsible natural language generation. 

---
# DischargeSim: A Simulation Benchmark for Educational Doctor-Patient Communication at Discharge 

**Authors**: Zonghai Yao, Michael Sun, Won Seok Jang, Sunjae Kwon, Soie Kwon, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07188)  

**Abstract**: Discharge communication is a critical yet underexplored component of patient care, where the goal shifts from diagnosis to education. While recent large language model (LLM) benchmarks emphasize in-visit diagnostic reasoning, they fail to evaluate models' ability to support patients after the visit. We introduce DischargeSim, a novel benchmark that evaluates LLMs on their ability to act as personalized discharge educators. DischargeSim simulates post-visit, multi-turn conversations between LLM-driven DoctorAgents and PatientAgents with diverse psychosocial profiles (e.g., health literacy, education, emotion). Interactions are structured across six clinically grounded discharge topics and assessed along three axes: (1) dialogue quality via automatic and LLM-as-judge evaluation, (2) personalized document generation including free-text summaries and structured AHRQ checklists, and (3) patient comprehension through a downstream multiple-choice exam. Experiments across 18 LLMs reveal significant gaps in discharge education capability, with performance varying widely across patient profiles. Notably, model size does not always yield better education outcomes, highlighting trade-offs in strategy use and content prioritization. DischargeSim offers a first step toward benchmarking LLMs in post-visit clinical education and promoting equitable, personalized patient support. 

---
# Towards EnergyGPT: A Large Language Model Specialized for the Energy Sector 

**Authors**: Amal Chebbi, Babajide Kolade  

**Link**: [PDF](https://arxiv.org/pdf/2509.07177)  

**Abstract**: Large Language Models have demonstrated impressive capabilities across various domains. However, their general-purpose nature often limits their effectiveness in specialized fields such as energy, where deep technical expertise and precise domain knowledge are essential. In this paper, we introduce EnergyGPT, a domain-specialized language model tailored for the energy sector, developed by fine-tuning LLaMA 3.1-8B model using Supervised Fine-Tuning on a high-quality, curated corpus of energy-related texts. We present a complete development pipeline, including data collection and curation, model fine-tuning, benchmark design and LLM-judge choice, evaluation and deployment. Through this work, we demonstrate that our training strategy enables improvements in domain relevance and performance without the need for large-scale infrastructure. By evaluating the performance of the model using domain-specific question-answering benchmarks, our results demonstrate that EnergyGPT outperforms the base model in most of the energy-related language understanding and generation tasks. 

---
# Toward Purpose-oriented Topic Model Evaluation enabled by Large Language Models 

**Authors**: Zhiyin Tan, Jennifer D'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2509.07142)  

**Abstract**: This study presents a framework for automated evaluation of dynamically evolving topic models using Large Language Models (LLMs). Topic modeling is essential for organizing and retrieving scholarly content in digital library systems, helping users navigate complex and evolving knowledge domains. However, widely used automated metrics, such as coherence and diversity, often capture only narrow statistical patterns and fail to explain semantic failures in practice. We introduce a purpose-oriented evaluation framework that employs nine LLM-based metrics spanning four key dimensions of topic quality: lexical validity, intra-topic semantic soundness, inter-topic structural soundness, and document-topic alignment soundness. The framework is validated through adversarial and sampling-based protocols, and is applied across datasets spanning news articles, scholarly publications, and social media posts, as well as multiple topic modeling methods and open-source LLMs. Our analysis shows that LLM-based metrics provide interpretable, robust, and task-relevant assessments, uncovering critical weaknesses in topic models such as redundancy and semantic drift, which are often missed by traditional metrics. These results support the development of scalable, fine-grained evaluation tools for maintaining topic relevance in dynamic datasets. All code and data supporting this work are accessible at this https URL. 

---
# The ML-SUPERB 2.0 Challenge: Towards Inclusive ASR Benchmarking for All Language Varieties 

**Authors**: William Chen, Chutong Meng, Jiatong Shi, Martijn Bartelds, Shih-Heng Wang, Hsiu-Hsuan Wang, Rafael Mosquera, Sara Hincapie, Dan Jurafsky, Antonis Anastasopoulos, Hung-yi Lee, Karen Livescu, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2509.07139)  

**Abstract**: Recent improvements in multilingual ASR have not been equally distributed across languages and language varieties. To advance state-of-the-art (SOTA) ASR models, we present the Interspeech 2025 ML-SUPERB 2.0 Challenge. We construct a new test suite that consists of data from 200+ languages, accents, and dialects to evaluate SOTA multilingual speech models. The challenge also introduces an online evaluation server based on DynaBench, allowing for flexibility in model design and architecture for participants. The challenge received 5 submissions from 3 teams, all of which outperformed our baselines. The best-performing submission achieved an absolute improvement in LID accuracy of 23% and a reduction in CER of 18% when compared to the best baseline on a general multilingual test set. On accented and dialectal data, the best submission obtained 30.2% lower CER and 15.7% higher LID accuracy, showing the importance of community challenges in making speech technologies more inclusive. 

---
# MedBench-IT: A Comprehensive Benchmark for Evaluating Large Language Models on Italian Medical Entrance Examinations 

**Authors**: Ruggero Marino Lazzaroni, Alessandro Angioi, Michelangelo Puliga, Davide Sanna, Roberto Marras  

**Link**: [PDF](https://arxiv.org/pdf/2509.07135)  

**Abstract**: Large language models (LLMs) show increasing potential in education, yet benchmarks for non-English languages in specialized domains remain scarce. We introduce MedBench-IT, the first comprehensive benchmark for evaluating LLMs on Italian medical university entrance examinations. Sourced from Edizioni Simone, a leading preparatory materials publisher, MedBench-IT comprises 17,410 expert-written multiple-choice questions across six subjects (Biology, Chemistry, Logic, General Culture, Mathematics, Physics) and three difficulty levels. We evaluated diverse models including proprietary LLMs (GPT-4o, Claude series) and resource-efficient open-source alternatives (<30B parameters) focusing on practical deployability.
Beyond accuracy, we conducted rigorous reproducibility tests (88.86% response consistency, varying by subject), ordering bias analysis (minimal impact), and reasoning prompt evaluation. We also examined correlations between question readability and model performance, finding a statistically significant but small inverse relationship. MedBench-IT provides a crucial resource for Italian NLP community, EdTech developers, and practitioners, offering insights into current capabilities and standardized evaluation methodology for this critical domain. 

---
# Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search 

**Authors**: Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, Hengshuang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.07969)  

**Abstract**: Recent advances in large multimodal models have leveraged image-based tools with reinforcement learning to tackle visual problems. However, existing open-source approaches often exhibit monotonous reasoning patterns and allow only a limited number of interaction turns, making them inadequate for difficult tasks that require trial-and-error exploration. In this work, we address this limitation by scaling up tool-based interactions and introduce Mini-o3, a system that executes deep, multi-turn reasoning -- spanning tens of steps -- and achieves state-of-the-art performance on challenging visual search tasks. Our recipe for reproducing OpenAI o3-style behaviors comprises three key components. First, we construct the Visual Probe Dataset, a collection of thousands of challenging visual search problems designed for exploratory reasoning. Second, we develop an iterative data collection pipeline to obtain cold-start trajectories that exhibit diverse reasoning patterns, including depth-first search, trial-and-error, and goal maintenance. Third, we propose an over-turn masking strategy that prevents penalization of over-turn responses (those that hit the maximum number of turns) during reinforcement learning, thereby balancing training-time efficiency with test-time scalability. Despite training with an upper bound of only six interaction turns, our model generates trajectories that naturally scale to tens of turns at inference time, with accuracy improving as the number of turns increases. Extensive experiments demonstrate that Mini-o3 produces rich reasoning patterns and deep thinking paths, effectively solving challenging visual search problems. 

---
# Visual-TableQA: Open-Domain Benchmark for Reasoning over Table Images 

**Authors**: Boammani Aser Lompo, Marc Haraoui  

**Link**: [PDF](https://arxiv.org/pdf/2509.07966)  

**Abstract**: Visual reasoning over structured data such as tables is a critical capability for modern vision-language models (VLMs), yet current benchmarks remain limited in scale, diversity, or reasoning depth, especially when it comes to rendered table images. Addressing this gap, we introduce Visual-TableQA, a large-scale, open-domain multimodal dataset specifically designed to evaluate and enhance visual reasoning over complex tabular data. Our generation pipeline is modular, scalable, and fully autonomous, involving multiple reasoning LLMs collaborating across distinct roles: generation, validation, and inspiration. Visual-TableQA comprises 2.5k richly structured LaTeX-rendered tables and 6k reasoning-intensive QA pairs, all produced at a cost of under USD 100. To promote diversity and creativity, our pipeline performs multi-model collaborative data generation via cross-model prompting ('inspiration') and LLM-jury filtering. Stronger models seed layouts and topics that weaker models elaborate, collectively distilling diverse reasoning patterns and visual structures into the dataset. Empirical results show that models fine-tuned on Visual-TableQA generalize robustly to external benchmarks, outperforming several proprietary models despite the dataset's synthetic nature. The full pipeline and resources are publicly available at this https URL. 

---
# Uncovering Scaling Laws for Large Language Models via Inverse Problems 

**Authors**: Arun Verma, Zhaoxuan Wu, Zijian Zhou, Xiaoqiang Lin, Zhiliang Chen, Rachael Hwee Ling Sim, Rui Qiao, Jingtan Wang, Nhung Bui, Xinyuan Niu, Wenyang Hu, Gregory Kang Ruey Lau, Zi-Yu Khoo, Zitong Zhao, Xinyi Xu, Apivich Hemachandra, See-Kiong Ng, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2509.07909)  

**Abstract**: Large Language Models (LLMs) are large-scale pretrained models that have achieved remarkable success across diverse domains. These successes have been driven by unprecedented complexity and scale in both data and computations. However, due to the high costs of training such models, brute-force trial-and-error approaches to improve LLMs are not feasible. Inspired by the success of inverse problems in uncovering fundamental scientific laws, this position paper advocates that inverse problems can also efficiently uncover scaling laws that guide the building of LLMs to achieve the desirable performance with significantly better cost-effectiveness. 

---
# Competitive Audio-Language Models with Data-Efficient Single-Stage Training on Public Data 

**Authors**: Gokul Karthik Kumar, Rishabh Saraf, Ludovick Lepauloux, Abdul Muneer, Billel Mokeddem, Hakim Hacid  

**Link**: [PDF](https://arxiv.org/pdf/2509.07526)  

**Abstract**: Large language models (LLMs) have transformed NLP, yet their integration with audio remains underexplored -- despite audio's centrality to human communication. We introduce Falcon3-Audio, a family of Audio-Language Models (ALMs) built on instruction-tuned LLMs and Whisper encoders. Using a remarkably small amount of public audio data -- less than 30K hours (5K unique) -- Falcon3-Audio-7B matches the best reported performance among open-weight models on the MMAU benchmark, with a score of 64.14, matching R1-AQA, while distinguishing itself through superior data and parameter efficiency, single-stage training, and transparency. Notably, our smallest 1B model remains competitive with larger open models ranging from 2B to 13B parameters. Through extensive ablations, we find that common complexities -- such as curriculum learning, multiple audio encoders, and intricate cross-attention connectors -- are not required for strong performance, even compared to models trained on over 500K hours of data. 

---
# Astra: A Multi-Agent System for GPU Kernel Performance Optimization 

**Authors**: Anjiang Wei, Tianran Sun, Yogesh Seenichamy, Hang Song, Anne Ouyang, Azalia Mirhoseini, Ke Wang, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2509.07506)  

**Abstract**: GPU kernel optimization has long been a central challenge at the intersection of high-performance computing and machine learning. Efficient kernels are crucial for accelerating large language model (LLM) training and serving, yet attaining high performance typically requires extensive manual tuning. Compiler-based systems reduce some of this burden, but still demand substantial manual design and engineering effort. Recently, researchers have explored using LLMs for GPU kernel generation, though prior work has largely focused on translating high-level PyTorch modules into CUDA code. In this work, we introduce Astra, the first LLM-based multi-agent system for GPU kernel optimization. Unlike previous approaches, Astra starts from existing CUDA implementations extracted from SGLang, a widely deployed framework for serving LLMs, rather than treating PyTorch modules as the specification. Within Astra, specialized LLM agents collaborate through iterative code generation, testing, profiling, and planning to produce kernels that are both correct and high-performance. On kernels from SGLang, Astra achieves an average speedup of 1.32x using zero-shot prompting with OpenAI o4-mini. A detailed case study further demonstrates that LLMs can autonomously apply loop transformations, optimize memory access patterns, exploit CUDA intrinsics, and leverage fast math operations to yield substantial performance gains. Our work highlights multi-agent LLM systems as a promising new paradigm for GPU kernel optimization. 

---
# GLEAM: Learning to Match and Explain in Cross-View Geo-Localization 

**Authors**: Xudong Lu, Zhi Zheng, Yi Wan, Yongxiang Yao, Annan Wang, Renrui Zhang, Panwang Xia, Qiong Wu, Qingyun Li, Weifeng Lin, Xiangyu Zhao, Xue Yang, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.07450)  

**Abstract**: Cross-View Geo-Localization (CVGL) focuses on identifying correspondences between images captured from distinct perspectives of the same geographical location. However, existing CVGL approaches are typically restricted to a single view or modality, and their direct visual matching strategy lacks interpretability: they merely predict whether two images correspond, without explaining the rationale behind the match. In this paper, we present GLEAM-C, a foundational CVGL model that unifies multiple views and modalities-including UAV imagery, street maps, panoramic views, and ground photographs-by aligning them exclusively with satellite imagery. Our framework enhances training efficiency through optimized implementation while achieving accuracy comparable to prior modality-specific CVGL models through a two-phase training strategy. Moreover, to address the lack of interpretability in traditional CVGL methods, we leverage the reasoning capabilities of multimodal large language models (MLLMs) to propose a new task, GLEAM-X, which combines cross-view correspondence prediction with explainable reasoning. To support this task, we construct a bilingual benchmark using GPT-4o and Doubao-1.5-Thinking-Vision-Pro to generate training and testing data. The test set is further refined through detailed human revision, enabling systematic evaluation of explainable cross-view reasoning and advancing transparency and scalability in geo-localization. Together, GLEAM-C and GLEAM-X form a comprehensive CVGL pipeline that integrates multi-modal, multi-view alignment with interpretable correspondence analysis, unifying accurate cross-view matching with explainable reasoning and advancing Geo-Localization by enabling models to better Explain And Match. Code and datasets used in this work will be made publicly accessible at this https URL. 

---
# Language Self-Play For Data-Free Training 

**Authors**: Jakub Grudzien Kuba, Mengting Gu, Qi Ma, Yuandong Tian, Vijai Mohan  

**Link**: [PDF](https://arxiv.org/pdf/2509.07414)  

**Abstract**: Large language models (LLMs) have advanced rapidly in recent years, driven by scale, abundant high-quality training data, and reinforcement learning. Yet this progress faces a fundamental bottleneck: the need for ever more data from which models can continue to learn. In this work, we propose a reinforcement learning approach that removes this dependency by enabling models to improve without additional data. Our method leverages a game-theoretic framework of self-play, where a model's capabilities are cast as performance in a competitive game and stronger policies emerge by having the model play against itself - a process we call Language Self-Play (LSP). Experiments with Llama-3.2-3B-Instruct on instruction-following benchmarks show that pretrained models can not only enhance their performance on challenging tasks through self-play alone, but can also do so more effectively than data-driven baselines. 

---
# ALICE: An Interpretable Neural Architecture for Generalization in Substitution Ciphers 

**Authors**: Jeff Shen, Lindsay Smith  

**Link**: [PDF](https://arxiv.org/pdf/2509.07282)  

**Abstract**: We present cryptogram solving as an ideal testbed for studying neural network generalization in combinatorially complex domains. In this task, models must decrypt text encoded with substitution ciphers, choosing from 26! possible mappings without explicit access to the cipher. We develop ALICE (an Architecture for Learning Interpretable Cryptogram dEcipherment): a simple encoder-only Transformer that sets a new state-of-the-art for both accuracy and speed on this decryption problem. Surprisingly, ALICE generalizes to unseen ciphers after training on only ${\sim}1500$ unique ciphers, a minute fraction ($3.7 \times 10^{-24}$) of the possible cipher space. To enhance interpretability, we introduce a novel bijective decoding head that explicitly models permutations via the Gumbel-Sinkhorn method, enabling direct extraction of learned cipher mappings. Through early exit analysis, we reveal how ALICE progressively refines its predictions in a way that appears to mirror common human strategies for this task: early layers employ frequency-based heuristics, middle layers form word structures, and final layers correct individual characters. Our architectural innovations and analysis methods extend beyond cryptograms to any domain with bijective mappings and combinatorial structure, offering new insights into neural network generalization and interpretability. 

---
# Benchmarking Information Retrieval Models on Complex Retrieval Tasks 

**Authors**: Julian Killingback, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2509.07253)  

**Abstract**: Large language models (LLMs) are incredible and versatile tools for text-based tasks that have enabled countless, previously unimaginable, applications. Retrieval models, in contrast, have not yet seen such capable general-purpose models emerge. To achieve this goal, retrieval models must be able to perform complex retrieval tasks, where queries contain multiple parts, constraints, or requirements in natural language. These tasks represent a natural progression from the simple, single-aspect queries that are used in the vast majority of existing, commonly used evaluation sets. Complex queries naturally arise as people expect search systems to handle more specific and often ambitious information requests, as is demonstrated by how people use LLM-based information systems. Despite the growing desire for retrieval models to expand their capabilities in complex retrieval tasks, there exist limited resources to assess the ability of retrieval models on a comprehensive set of diverse complex tasks. The few resources that do exist feature a limited scope and often lack realistic settings making it hard to know the true capabilities of retrieval models on complex real-world retrieval tasks. To address this shortcoming and spur innovation in next-generation retrieval models, we construct a diverse and realistic set of complex retrieval tasks and benchmark a representative set of state-of-the-art retrieval models. Additionally, we explore the impact of LLM-based query expansion and rewriting on retrieval quality. Our results show that even the best models struggle to produce high-quality retrieval results with the highest average nDCG@10 of only 0.346 and R@100 of only 0.587 across all tasks. Although LLM augmentation can help weaker models, the strongest model has decreased performance across all metrics with all rewriting techniques. 

---
# Neurocognitive Modeling for Text Generation: Deep Learning Architecture for EEG Data 

**Authors**: Khushiyant  

**Link**: [PDF](https://arxiv.org/pdf/2509.07202)  

**Abstract**: Text generating capabilities have undergone a substantial transformation with the introduction of large language models (LLMs). Electroencephalography (EEG)-based text production is still difficult, though, because it requires a lot of data and processing power. This paper introduces a new method that combines the use of the Gemma 2B LLM with a classifier-LLM architecture to incorporate a Recurrent Neural Network (RNN) encoder. Our approach drastically lowers the amount of data and compute power needed while achieving performance close to that of cutting-edge methods. Notably, compared to current methodologies, our methodology delivers an overall performance improvement of 10%. The suggested architecture demonstrates the possibility of effective transfer learning for EEG-based text production, remaining strong and functional even in the face of data limits. This work highlights the potential of integrating LLMs with EEG decoding to improve assistive technologies and improve independence and communication for those with severe motor limitations. Our method pushes the limits of present capabilities and opens new paths for research and application in brain-computer interfaces by efficiently using the strengths of pre-trained language models. This makes EEG-based text production more accessible and efficient. 

---
# That's So FETCH: Fashioning Ensemble Techniques for LLM Classification in Civil Legal Intake and Referral 

**Authors**: Quinten Steenhuis  

**Link**: [PDF](https://arxiv.org/pdf/2509.07170)  

**Abstract**: Each year millions of people seek help for their legal problems by calling a legal aid program hotline, walking into a legal aid office, or using a lawyer referral service. The first step to match them to the right help is to identify the legal problem the applicant is experiencing. Misdirection has consequences. Applicants may miss a deadline, experience physical abuse, lose housing or lose custody of children while waiting to connect to the right legal help. We introduce and evaluate the FETCH classifier for legal issue classification and describe two methods for improving accuracy: a hybrid LLM/ML ensemble classification method, and the automatic generation of follow-up questions to enrich the initial problem narrative. We employ a novel data set of 419 real-world queries to a nonprofit lawyer referral service. Ultimately, we show classification accuracy (hits@2) of 97.37\% using a mix of inexpensive models, exceeding the performance of the current state-of-the-art GPT-5 model. Our approach shows promise in significantly reducing the cost of guiding users of the legal system to the right resource for their problem while achieving high accuracy. 

---
# Beyond Sequential Reranking: Reranker-Guided Search Improves Reasoning Intensive Retrieval 

**Authors**: Haike Xu, Tong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.07163)  

**Abstract**: The widely used retrieve-and-rerank pipeline faces two critical limitations: they are constrained by the initial retrieval quality of the top-k documents, and the growing computational demands of LLM-based rerankers restrict the number of documents that can be effectively processed. We introduce Reranker-Guided-Search (RGS), a novel approach that bypasses these limitations by directly retrieving documents according to reranker preferences rather than following the traditional sequential reranking method. Our method uses a greedy search on proximity graphs generated by approximate nearest neighbor algorithms, strategically prioritizing promising documents for reranking based on document similarity. Experimental results demonstrate substantial performance improvements across multiple benchmarks: 3.5 points on BRIGHT, 2.9 on FollowIR, and 5.1 on M-BEIR, all within a constrained reranker budget of 100 documents. Our analysis suggests that, given a fixed pair of embedding and reranker models, strategically selecting documents to rerank can significantly improve retrieval accuracy under limited reranker budget. 

---
# Measuring Uncertainty in Transformer Circuits with Effective Information Consistency 

**Authors**: Anatoly A. Krasnovsky  

**Link**: [PDF](https://arxiv.org/pdf/2509.07149)  

**Abstract**: Mechanistic interpretability has identified functional subgraphs within large language models (LLMs), known as Transformer Circuits (TCs), that appear to implement specific algorithms. Yet we lack a formal, single-pass way to quantify when an active circuit is behaving coherently and thus likely trustworthy. Building on prior systems-theoretic proposals, we specialize a sheaf/cohomology and causal emergence perspective to TCs and introduce the Effective-Information Consistency Score (EICS). EICS combines (i) a normalized sheaf inconsistency computed from local Jacobians and activations, with (ii) a Gaussian EI proxy for circuit-level causal emergence derived from the same forward state. The construction is white-box, single-pass, and makes units explicit so that the score is dimensionless. We further provide practical guidance on score interpretation, computational overhead (with fast and exact modes), and a toy sanity-check analysis. Empirical validation on LLM tasks is deferred. 

---
# Neuro-Symbolic Frameworks: Conceptual Characterization and Empirical Comparative Analysis 

**Authors**: Sania Sinha, Tanawan Premsri, Danial Kamali, Parisa Kordjamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2509.07122)  

**Abstract**: Neurosymbolic (NeSy) frameworks combine neural representations and learning with symbolic representations and reasoning. Combining the reasoning capacities, explainability, and interpretability of symbolic processing with the flexibility and power of neural computing allows us to solve complex problems with more reliability while being data-efficient. However, this recently growing topic poses a challenge to developers with its learning curve, lack of user-friendly tools, libraries, and unifying frameworks. In this paper, we characterize the technical facets of existing NeSy frameworks, such as the symbolic representation language, integration with neural models, and the underlying algorithms. A majority of the NeSy research focuses on algorithms instead of providing generic frameworks for declarative problem specification to leverage problem solving. To highlight the key aspects of Neurosymbolic modeling, we showcase three generic NeSy frameworks - \textit{DeepProbLog}, \textit{Scallop}, and \textit{DomiKnowS}. We identify the challenges within each facet that lay the foundation for identifying the expressivity of each framework in solving a variety of problems. Building on this foundation, we aim to spark transformative action and encourage the community to rethink this problem in novel ways. 

---
# Instruction Agent: Enhancing Agent with Expert Demonstration 

**Authors**: Yinheng Li, Hailey Hultquist, Justin Wagle, Kazuhito Koishida  

**Link**: [PDF](https://arxiv.org/pdf/2509.07098)  

**Abstract**: Graphical user interface (GUI) agents have advanced rapidly but still struggle with complex tasks involving novel UI elements, long-horizon actions, and personalized trajectories. In this work, we introduce Instruction Agent, a GUI agent that leverages expert demonstrations to solve such tasks, enabling completion of otherwise difficult workflows. Given a single demonstration, the agent extracts step-by-step instructions and executes them by strictly following the trajectory intended by the user, which avoids making mistakes during execution. The agent leverages the verifier and backtracker modules further to improve robustness. Both modules are critical to understand the current outcome from each action and handle unexpected interruptions(such as pop-up windows) during execution. Our experiments show that Instruction Agent achieves a 60% success rate on a set of tasks in OSWorld that all top-ranked agents failed to complete. The Instruction Agent offers a practical and extensible framework, bridging the gap between current GUI agents and reliable real-world GUI task automation. 

---
# From Eigenmodes to Proofs: Integrating Graph Spectral Operators with Symbolic Interpretable Reasoning 

**Authors**: Andrew Kiruluta, Priscilla Burity  

**Link**: [PDF](https://arxiv.org/pdf/2509.07017)  

**Abstract**: We introduce Spectral NSR, a fully spectral neuro-symbolic reasoning framework that embeds logical rules as spectral templates and performs inference directly in the graph spectral domain. By leveraging graph signal processing (GSP) and frequency-selective filters grounded in the Laplacian eigenstructure of knowledge graphs, the architecture unifies the interpretability of symbolic reasoning with the scalability and adaptability of spectral learning. Beyond the core formulation, we incorporate a comprehensive set of extensions, including dynamic graph and basis learning, rational and diffusion filters for sharper spectral selectivity, mixture-of-spectral-experts for modular specialization, proof-guided training with spectral curricula, and uncertainty quantification for calibrated confidence. Additional enhancements such as large language model coupling, co-spectral transfer alignment, adversarial robustness, efficient GPU kernels, generalized Laplacians, and causal interventions further expand the versatility of the framework.
Empirical evaluation on state-of-the-art reasoning benchmarks such as ProofWriter and CLUTRR demonstrates that Spectral NSR achieves superior accuracy, faster inference, improved robustness to adversarial perturbations, and higher interpretability compared to leading baselines including transformers, message-passing neural networks, and neuro-symbolic logic programming systems. Spectral attribution and proof-band agreement analyses confirm that model decisions align closely with symbolic proof structures, while transfer experiments validate effective domain adaptation through co-spectral alignment. These results establish Spectral NSR as a scalable and principled foundation for the next generation of reasoning systems, offering transparency, robustness, and generalization beyond conventional approaches. 

---
# ArGen: Auto-Regulation of Generative AI via GRPO and Policy-as-Code 

**Authors**: Kapil Madan  

**Link**: [PDF](https://arxiv.org/pdf/2509.07006)  

**Abstract**: This paper introduces ArGen (Auto-Regulation of Generative AI systems), a framework for aligning Large Language Models (LLMs) with complex sets of configurable, machine-readable rules spanning ethical principles, operational safety protocols, and regulatory compliance standards. Moving beyond just preference-based alignment, ArGen is designed to ensure LLMs adhere to these multifaceted policies through a novel synthesis of principle-based automated reward scoring, Group Relative Policy Optimisation (GRPO), and an Open Policy Agent (OPA) inspired governance layer. This approach provides the technical foundation for achieving and demonstrating compliance with diverse and nuanced governance requirements. To showcase the framework's capability to operationalize a deeply nuanced and culturally-specific value system, we present an in-depth case study: the development of a medical AI assistant guided by principles from Dharmic ethics (such as Ahimsa and Dharma), as derived from texts like the Bhagavad Gita. This challenging application demonstrates ArGen's adaptability, achieving a 70.9% improvement in domain-scope adherence over the baseline. Through our open-source repository, we show that ArGen's methodology offers a path to 'Governable Al' systems that are technically proficient, ethically robust, and verifiably compliant for safe deployment in diverse global contexts. 

---
# VLMs-in-the-Wild: Bridging the Gap Between Academic Benchmarks and Enterprise Reality 

**Authors**: Srihari Bandraupalli, Anupam Purwar  

**Link**: [PDF](https://arxiv.org/pdf/2509.06994)  

**Abstract**: Open-source Vision-Language Models show immense promise for enterprise applications, yet a critical disconnect exists between academic evaluation and enterprise deployment requirements. Current benchmarks rely heavily on multiple-choice questions and synthetic data, failing to capture the complexity of real-world business applications like social media content analysis. This paper introduces VLM-in-the-Wild (ViLD), a comprehensive framework to bridge this gap by evaluating VLMs on operational enterprise requirements. We define ten business-critical tasks: logo detection, OCR, object detection, human presence and demographic analysis, human activity and appearance analysis, scene detection, camera perspective and media quality assessment, dominant colors, comprehensive description, and NSFW detection. To this framework, we bring an innovative BlockWeaver Algorithm that solves the challenging problem of comparing unordered, variably-grouped OCR outputs from VLMs without relying on embeddings or LLMs, achieving remarkable speed and reliability. To demonstrate efficacy of ViLD, we constructed a new benchmark dataset of 7,500 diverse samples, carefully stratified from a corpus of one million real-world images and videos. ViLD provides actionable insights by combining semantic matching (both embedding-based and LLM-as-a-judge approaches), traditional metrics, and novel methods to measure the completeness and faithfulness of descriptive outputs. By benchmarking leading open-source VLMs (Qwen, MIMO, and InternVL) against a powerful proprietary baseline as per ViLD framework, we provide one of the first industry-grounded, task-driven assessment of VLMs capabilities, offering actionable insights for their deployment in enterprise environments. 

---
# CARE: Decoding Time Safety Alignment via Rollback and Introspection Intervention 

**Authors**: Xiaomeng Hu, Fei Huang, Chenhan Yuan, Junyang Lin, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2509.06982)  

**Abstract**: As large language models (LLMs) are increasingly deployed in real-world applications, ensuring the safety of their outputs during decoding has become a critical challenge. However, existing decoding-time interventions, such as Contrastive Decoding, often force a severe trade-off between safety and response quality. In this work, we propose CARE, a novel framework for decoding-time safety alignment that integrates three key components: (1) a guard model for real-time safety monitoring, enabling detection of potentially unsafe content; (2) a rollback mechanism with a token buffer to correct unsafe outputs efficiently at an earlier stage without disrupting the user experience; and (3) a novel introspection-based intervention strategy, where the model generates self-reflective critiques of its previous outputs and incorporates these reflections into the context to guide subsequent decoding steps. The framework achieves a superior safety-quality trade-off by using its guard model for precise interventions, its rollback mechanism for timely corrections, and our novel introspection method for effective self-correction. Experimental results demonstrate that our framework achieves a superior balance of safety, quality, and efficiency, attaining a low harmful response rate and minimal disruption to the user experience while maintaining high response quality. 

---
