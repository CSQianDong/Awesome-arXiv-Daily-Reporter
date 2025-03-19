# Temporal Consistency for LLM Reasoning Process Error Identification 

**Authors**: Jiacheng Guo, Yue Wu, Jiahao Qiu, Kaixuan Huang, Xinzhe Juan, Ling Yang, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14495)  

**Abstract**: Verification is crucial for effective mathematical reasoning. We present a new temporal consistency method where verifiers iteratively refine their judgments based on the previous assessment. Unlike one-round verification or multi-model debate approaches, our method leverages consistency in a sequence of self-reflection actions to improve verification accuracy. Empirical evaluations across diverse mathematical process error identification benchmarks (Mathcheck, ProcessBench, and PRM800K) show consistent performance improvements over baseline methods. When applied to the recent DeepSeek R1 distilled models, our method demonstrates strong performance, enabling 7B/8B distilled models to outperform all 70B/72B models and GPT-4o on ProcessBench. Notably, the distilled 14B model with our method achieves performance comparable to Deepseek-R1. Our codes are available at this https URL 

---
# Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations 

**Authors**: Ziwei Ji, Lei Yu, Yeskendir Koishekenov, Yejin Bang, Anthony Hartshorn, Alan Schelten, Cheng Zhang, Pascale Fung, Nicola Cancedda  

**Link**: [PDF](https://arxiv.org/pdf/2503.14477)  

**Abstract**: LLMs often adopt an assertive language style also when making false claims. Such ``overconfident hallucinations'' mislead users and erode trust. Achieving the ability to express in language the actual degree of uncertainty around a claim is therefore of great importance. We find that ``verbal uncertainty'' is governed by a single linear feature in the representation space of LLMs, and show that this has only moderate correlation with the actual ``semantic uncertainty'' of the model. We apply this insight and show that (1) the mismatch between semantic and verbal uncertainty is a better predictor of hallucinations than semantic uncertainty alone and (2) we can intervene on verbal uncertainty at inference time and reduce hallucinations on short-form answers, achieving an average relative reduction of 32%. 

---
# RWKV-7 "Goose" with Expressive Dynamic State Evolution 

**Authors**: Bo Peng, Ruichong Zhang, Daniel Goldstein, Eric Alcaide, Haowen Hou, Janna Lu, William Merrill, Guangyu Song, Kaifeng Tan, Saiteja Utpala, Nathan Wilce, Johan S. Wind, Tianyi Wu, Daniel Wuttke, Christian Zhou-Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.14456)  

**Abstract**: We present RWKV-7 "Goose", a new sequence modeling architecture, along with pre-trained language models that establish a new state-of-the-art in downstream performance at the 3 billion parameter scale on multilingual tasks, and match current SoTA English language performance despite being trained on dramatically fewer tokens than other top 3B models. Nevertheless, RWKV-7 models require only constant memory usage and constant inference time per token. RWKV-7 introduces a newly generalized formulation of the delta rule with vector-valued gating and in-context learning rates, as well as a relaxed value replacement rule. We show that RWKV-7 can perform state tracking and recognize all regular languages, while retaining parallelizability of training. This exceeds the capabilities of Transformers under standard complexity conjectures, which are limited to $\mathsf{TC}^0$. To demonstrate RWKV-7's language modeling capability, we also present an extended open source 3.1 trillion token multilingual corpus, and train four RWKV-7 models ranging from 0.19 billion to 2.9 billion parameters on this dataset.
To foster openness, reproduction, and adoption, we release our models and dataset component listing at this https URL, and our training and inference code at this https URL all under the Apache 2.0 License. 

---
# Splintering Nonconcatenative Languages for Better Tokenization 

**Authors**: Bar Gazit, Shaltiel Shmidman, Avi Shmidman, Yuval Pinter  

**Link**: [PDF](https://arxiv.org/pdf/2503.14433)  

**Abstract**: Common subword tokenization algorithms like BPE and UnigramLM assume that text can be split into meaningful units by concatenative measures alone. This is not true for languages such as Hebrew and Arabic, where morphology is encoded in root-template patterns, or Malay and Georgian, where split affixes are common. We present SPLINTER, a pre-processing step which rearranges text into a linear form that better represents such nonconcatenative morphologies, enabling meaningful contiguous segments to be found by the tokenizer. We demonstrate SPLINTER's merit using both intrinsic measures evaluating token vocabularies in Hebrew, Arabic, and Malay; as well as on downstream tasks using BERT-architecture models trained for Hebrew. 

---
# PLAY2PROMPT: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play 

**Authors**: Wei Fang, Yang Zhang, Kaizhi Qian, James Glass, Yada Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14432)  

**Abstract**: Large language models (LLMs) are increasingly integrated with specialized external tools, yet many tasks demand zero-shot tool usage with minimal or noisy documentation. Existing solutions rely on manual rewriting or labeled data for validation, making them inapplicable in true zero-shot settings. To address these challenges, we propose PLAY2PROMPT, an automated framework that systematically "plays" with each tool to explore its input-output behaviors. Through this iterative trial-and-error process, PLAY2PROMPT refines tool documentation and generates usage examples without any labeled data. These examples not only guide LLM inference but also serve as validation to further enhance tool utilization. Extensive experiments on real-world tasks demonstrate that PLAY2PROMPT significantly improves zero-shot tool performance across both open and closed models, offering a scalable and effective solution for domain-specific tool integration. 

---
# Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with Large Language Models 

**Authors**: Siwei Zhang, Yun Xiong, Yateng Tang, Xi Chen, Zian Jia, Zehao Gu, Jiarong Xu, Jiawei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14411)  

**Abstract**: Temporal graph neural networks (TGNNs) have shown remarkable performance in temporal graph modeling. However, real-world temporal graphs often possess rich textual information, giving rise to temporal text-attributed graphs (TTAGs). Such combination of dynamic text semantics and evolving graph structures introduces heightened complexity. Existing TGNNs embed texts statically and rely heavily on encoding mechanisms that biasedly prioritize structural information, overlooking the temporal evolution of text semantics and the essential interplay between semantics and structures for synergistic reinforcement. To tackle these issues, we present \textbf{Cross}, a novel framework that seamlessly extends existing TGNNs for TTAG modeling. The key idea is to employ the advanced large language models (LLMs) to extract the dynamic semantics in text space and then generate expressive representations unifying both semantics and structures. Specifically, we propose a Temporal Semantics Extractor in the {Cross} framework, which empowers the LLM to offer the temporal semantic understanding of node's evolving contexts of textual neighborhoods, facilitating semantic dynamics. Subsequently, we introduce the Semantic-structural Co-encoder, which collaborates with the above Extractor for synthesizing illuminating representations by jointly considering both semantic and structural information while encouraging their mutual reinforcement. Extensive experimental results on four public datasets and one practical industrial dataset demonstrate {Cross}'s significant effectiveness and robustness. 

---
# From "Hallucination" to "Suture": Insights from Language Philosophy to Enhance Large Language Models 

**Authors**: Qiantong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14392)  

**Abstract**: This paper explores hallucination phenomena in large language models (LLMs) through the lens of language philosophy and psychoanalysis. By incorporating Lacan's concepts of the "chain of signifiers" and "suture points," we propose the Anchor-RAG framework as a novel approach to mitigate hallucinations. In contrast to the predominant reliance on trial-and-error experiments, constant adjustments of mathematical formulas, or resource-intensive methods that emphasize quantity over quality, our approach returns to the fundamental principles of linguistics to analyze the root causes of hallucinations in LLMs. Drawing from robust theoretical foundations, we derive algorithms and models that are not only effective in reducing hallucinations but also enhance LLM performance and improve output quality. This paper seeks to establish a comprehensive theoretical framework for understanding hallucinations in LLMs and aims to challenge the prevalent "guess-and-test" approach and rat race mentality in the field. We aspire to pave the way for a new era of interpretable LLMs, offering deeper insights into the inner workings of language-based AI systems. 

---
# How much do LLMs learn from negative examples? 

**Authors**: Shadi Hamdan, Deniz Yuret  

**Link**: [PDF](https://arxiv.org/pdf/2503.14391)  

**Abstract**: Large language models (LLMs) undergo a three-phase training process: unsupervised pre-training, supervised fine-tuning (SFT), and learning from human feedback (RLHF/DPO). Notably, it is during the final phase that these models are exposed to negative examples -- incorrect, rejected, or suboptimal responses to queries. This paper delves into the role of negative examples in the training of LLMs, using a likelihood-ratio (Likra) model on multiple-choice question answering benchmarks to precisely manage the influence and the volume of negative examples. Our findings reveal three key insights: (1) During a critical phase in training, Likra with negative examples demonstrates a significantly larger improvement per training example compared to SFT using only positive examples. This leads to a sharp jump in the learning curve for Likra unlike the smooth and gradual improvement of SFT; (2) negative examples that are plausible but incorrect (near-misses) exert a greater influence; and (3) while training with positive examples fails to significantly decrease the likelihood of plausible but incorrect answers, training with negative examples more accurately identifies them. These results indicate a potentially significant role for negative examples in improving accuracy and reducing hallucinations for LLMs. 

---
# Good/Evil Reputation Judgment of Celebrities by LLMs via Retrieval Augmented Generation 

**Authors**: Rikuto Tsuchida, Hibiki Yokoyama, Takehito Utsuro  

**Link**: [PDF](https://arxiv.org/pdf/2503.14382)  

**Abstract**: The purpose of this paper is to examine whether large language models (LLMs) can understand what is good and evil with respect to judging good/evil reputation of celebrities. Specifically, we first apply a large language model (namely, ChatGPT) to the task of collecting sentences that mention the target celebrity from articles about celebrities on Web pages. Next, the collected sentences are categorized based on their contents by ChatGPT, where ChatGPT assigns a category name to each of those categories. Those assigned category names are referred to as "aspects" of each celebrity. Then, by applying the framework of retrieval augmented generation (RAG), we show that the large language model is quite effective in the task of judging good/evil reputation of aspects and descriptions of each celebrity. Finally, also in terms of proving the advantages of the proposed method over existing services incorporating RAG functions, we show that the proposed method of judging good/evil of aspects/descriptions of each celebrity significantly outperform an existing service incorporating RAG functions. 

---
# Spatio-Temporal Graph Neural Networks for Infant Language Acquisition Prediction 

**Authors**: Andrew Roxburgh, Floriana Grasso, Terry R. Payne  

**Link**: [PDF](https://arxiv.org/pdf/2503.14341)  

**Abstract**: Predicting the words that a child is going to learn next can be useful for boosting language acquisition, and such predictions have been shown to be possible with both neural network techniques (looking at changes in the vocabulary state over time) and graph model (looking at data pertaining to the relationships between words). However, these models do not fully capture the complexity of the language learning process of an infant when used in isolation. In this paper, we examine how a model of language acquisition for infants and young children can be constructed and adapted for use in a Spatio-Temporal Graph Convolutional Network (STGCN), taking into account the different types of linguistic relationships that occur during child language learning. We introduce a novel approach for predicting child vocabulary acquisition, and evaluate the efficacy of such a model with respect to the different types of linguistic relationships that occur during language acquisition, resulting in insightful observations on model calibration and norm selection. An evaluation of this model found that the mean accuracy of models for predicting new words when using sensorimotor relationships (0.733) and semantic relationships (0.729) were found to be superior to that observed with a 2-layer Feed-forward neural network. Furthermore, the high recall for some relationships suggested that some relationships (e.g. visual) were superior in identifying a larger proportion of relevant words that a child should subsequently learn than others (such as auditory). 

---
# DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal 

**Authors**: Vaibhav Aggarwal, Ojasv Kamal, Abhinav Japesh, Zhijing Jin, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2503.14269)  

**Abstract**: Large Language Models (LLMs) have revolutionized various domains, including natural language processing, data analysis, and software development, by enabling automation. In software engineering, LLM-powered coding agents have garnered significant attention due to their potential to automate complex development tasks, assist in debugging, and enhance productivity. However, existing approaches often struggle with sub-optimal decision-making, requiring either extensive manual intervention or inefficient compute scaling strategies. To improve coding agent performance, we present Dynamic Action Re-Sampling (DARS), a novel inference time compute scaling approach for coding agents, that is faster and more effective at recovering from sub-optimal decisions compared to baselines. While traditional agents either follow linear trajectories or rely on random sampling for scaling compute, our approach DARS works by branching out a trajectory at certain key decision points by taking an alternative action given the history of the trajectory and execution feedback of the previous attempt from that point. We evaluate our approach on SWE-Bench Lite benchmark, demonstrating that this scaling strategy achieves a pass@k score of 55% with Claude 3.5 Sonnet V2. Our framework achieves a pass@1 rate of 47%, outperforming state-of-the-art (SOTA) open-source frameworks. 

---
# JuDGE: Benchmarking Judgment Document Generation for Chinese Legal System 

**Authors**: Weihang Su, Baoqing Yue, Qingyao Ai, Yiran Hu, Jiaqi Li, Changyue Wang, Kaiyuan Zhang, Yueyue Wu, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14258)  

**Abstract**: This paper introduces JuDGE (Judgment Document Generation Evaluation), a novel benchmark for evaluating the performance of judgment document generation in the Chinese legal system. We define the task as generating a complete legal judgment document from the given factual description of the case. To facilitate this benchmark, we construct a comprehensive dataset consisting of factual descriptions from real legal cases, paired with their corresponding full judgment documents, which serve as the ground truth for evaluating the quality of generated documents. This dataset is further augmented by two external legal corpora that provide additional legal knowledge for the task: one comprising statutes and regulations, and the other consisting of a large collection of past judgment documents. In collaboration with legal professionals, we establish a comprehensive automated evaluation framework to assess the quality of generated judgment documents across various dimensions. We evaluate various baseline approaches, including few-shot in-context learning, fine-tuning, and a multi-source retrieval-augmented generation (RAG) approach, using both general and legal-domain LLMs. The experimental results demonstrate that, while RAG approaches can effectively improve performance in this task, there is still substantial room for further improvement. All the codes and datasets are available at: this https URL. 

---
# Towards Harmless Multimodal Assistants with Blind Preference Optimization 

**Authors**: Yongqi Li, Lu Yang, Jian Wang, Runyang You, Wenjie Li, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.14189)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in multimodal understanding, reasoning, and interaction. Given the extensive applications of MLLMs, the associated safety issues have become increasingly critical. Due to the effectiveness of preference optimization in aligning MLLMs with human preferences, there is an urgent need for safety-related preference data for MLLMs. To address this, we construct the MMSafe-PO preference dataset towards harmless multimodal assistants, featuring multimodal instructions, the conversational format, and ranked paired responses from human feedback. We also identify two insightful observations: modality co-defense and modality cheating, which illustrate that MLLMs possess a certain level of inherent defense while still presenting unique safety challenges. Based on these observations, we propose the Blind Preference Optimization (BPO) approach. Comprehensive experiments on three benchmarks show that BPO effectively enhances the safety capabilities of MLLMs. Notably, BPO significantly improves the safety rate of the base MLLM by 45.0%, outperforming the DPO approach. Additionally, applying BPO to the MMSafe-PO dataset greatly reduces the base MLLM's unsafe rate on other safety benchmarks (14.5% on MM-SafetyBench and 82.9% on HarmEval, demonstrating the effectiveness and robustness of both the dataset and the approach. We release code and data at this https URL. 

---
# AdaST: Dynamically Adapting Encoder States in the Decoder for End-to-End Speech-to-Text Translation 

**Authors**: Wuwei Huang, Dexin Wang, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.14185)  

**Abstract**: In end-to-end speech translation, acoustic representations learned by the encoder are usually fixed and static, from the perspective of the decoder, which is not desirable for dealing with the cross-modal and cross-lingual challenge in speech translation. In this paper, we show the benefits of varying acoustic states according to decoder hidden states and propose an adaptive speech-to-text translation model that is able to dynamically adapt acoustic states in the decoder. We concatenate the acoustic state and target word embedding sequence and feed the concatenated sequence into subsequent blocks in the decoder. In order to model the deep interaction between acoustic states and target hidden states, a speech-text mixed attention sublayer is introduced to replace the conventional cross-attention network. Experiment results on two widely-used datasets show that the proposed method significantly outperforms state-of-the-art neural speech translation models. 

---
# NERCat: Fine-Tuning for Enhanced Named Entity Recognition in Catalan 

**Authors**: Guillem Cadevall Ferreres, Marc Serrano Sanz, Marc Bardeli Gámez, Pol Gerdt Basullas, Francesc Tarres Ruiz, Raul Quijada Ferrero  

**Link**: [PDF](https://arxiv.org/pdf/2503.14173)  

**Abstract**: Named Entity Recognition (NER) is a critical component of Natural Language Processing (NLP) for extracting structured information from unstructured text. However, for low-resource languages like Catalan, the performance of NER systems often suffers due to the lack of high-quality annotated datasets. This paper introduces NERCat, a fine-tuned version of the GLiNER[1] model, designed to improve NER performance specifically for Catalan text. We used a dataset of manually annotated Catalan television transcriptions to train and fine-tune the model, focusing on domains such as politics, sports, and culture. The evaluation results show significant improvements in precision, recall, and F1-score, particularly for underrepresented named entity categories such as Law, Product, and Facility. This study demonstrates the effectiveness of domain-specific fine-tuning in low-resource languages and highlights the potential for enhancing Catalan NLP applications through manual annotation and high-quality datasets. 

---
# Synthetic Clarification and Correction Dialogues about Data-Centric Tasks -- A Teacher-Student Approach 

**Authors**: Christian Poelitz, Nick McKenna  

**Link**: [PDF](https://arxiv.org/pdf/2503.14167)  

**Abstract**: Real dialogues with AI assistants for solving data-centric tasks often follow dynamic, unpredictable paths due to imperfect information provided by the user or in the data, which must be caught and handled. Developing datasets which capture such user-AI interactions is difficult and time-consuming. In this work, we develop a novel framework for synthetically generating controlled, multi-turn conversations between a user and AI assistant for the task of table-based question answering, which can be generated from an existing dataset with fully specified table QA examples for any target domain. Each conversation aims to solve a table-based reasoning question through collaborative effort, modeling one of two real-world scenarios: (1) an AI-initiated clarification, or (2) a user-initiated correction. Critically, we employ a strong teacher LLM to verify the correctness of our synthetic conversations, ensuring high quality. We demonstrate synthetic datasets generated from TAT-QA and WikiTableQuestions as benchmarks of frontier LLMs. We find that even larger models struggle to effectively issuing clarification questions and accurately integrate user feedback for corrections. 

---
# CARE: A QLoRA-Fine Tuned Multi-Domain Chatbot With Fast Learning On Minimal Hardware 

**Authors**: Ankit Dutta, Nabarup Ghosh, Ankush Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.14136)  

**Abstract**: Large Language models have demonstrated excellent domain-specific question-answering capabilities when finetuned with a particular dataset of that specific domain. However, fine-tuning the models requires a significant amount of training time and a considerable amount of hardware. In this work, we propose CARE (Customer Assistance and Response Engine), a lightweight model made by fine-tuning Phi3.5-mini on very minimal hardware and data, designed to handle queries primarily across three domains: telecommunications support, medical support, and banking support. For telecommunications and banking, the chatbot addresses issues and problems faced by customers regularly in the above-mentioned domains. In the medical domain, CARE provides preliminary support by offering basic diagnoses and medical suggestions that a user might take before consulting a healthcare professional. Since CARE is built on Phi3.5-mini, it can be used even on mobile devices, increasing its usability. Our research also shows that CARE performs relatively well on various medical benchmarks, indicating that it can be used to make basic medical suggestions. 

---
# Wiki-Quantities and Wiki-Measurements: Datasets of Quantities and their Measurement Context from Wikipedia 

**Authors**: Jan Göpfert, Patrick Kuckertz, Jann M. Weinand, Detlef Stolten  

**Link**: [PDF](https://arxiv.org/pdf/2503.14090)  

**Abstract**: To cope with the large number of publications, more and more researchers are automatically extracting data of interest using natural language processing methods based on supervised learning. Much data, especially in the natural and engineering sciences, is quantitative, but there is a lack of datasets for identifying quantities and their context in text. To address this issue, we present two large datasets based on Wikipedia and Wikidata: Wiki-Quantities is a dataset consisting of over 1.2 million annotated quantities in the English-language Wikipedia. Wiki-Measurements is a dataset of 38,738 annotated quantities in the English-language Wikipedia along with their respective measured entity, property, and optional qualifiers. Manual validation of 100 samples each of Wiki-Quantities and Wiki-Measurements found 100% and 84-94% correct, respectively. The datasets can be used in pipeline approaches to measurement extraction, where quantities are first identified and then their measurement context. To allow reproduction of this work using newer or different versions of Wikipedia and Wikidata, we publish the code used to create the datasets along with the data. 

---
# Synthetic Data Generation Using Large Language Models: Advances in Text and Code 

**Authors**: Mihai Nadas, Laura Diosan, Andreea Tomescu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14023)  

**Abstract**: Large language models (LLMs) have unlocked new possibilities for generating synthetic training data in both natural language and code. By producing artificial but task-relevant examples, these models can significantly augment or even replace real-world datasets, especially when labeled data is scarce or sensitive. This paper surveys recent advances in using LLMs to create synthetic text and code, emphasizing prompt-based generation, retrieval-augmented pipelines, and iterative self-refinement. We show how these methods enrich low-resource tasks such as classification and question answering, as well as code-centric applications such as instruction tuning, code translation, and bug repair, by enabling automated verification of functional correctness. Alongside potential benefits like cost-effectiveness, broad coverage, and controllable diversity, we address challenges such as factual inaccuracies in generated text, lack of stylistic realism, and the risk of bias amplification. Proposed mitigations include filtering and weighting outputs and reinforcement learning with execution feedback for code. We conclude with open research directions like automated prompt engineering, cross-modal data synthesis, and robust evaluation frameworks, highlighting the importance of LLM-generated synthetic data in advancing AI while emphasizing ethical and quality safeguards. 

---
# The KoLMogorov Test: Compression by Code Generation 

**Authors**: Ori Yoran, Kunhao Zheng, Fabian Gloeckle, Jonas Gehring, Gabriel Synnaeve, Taco Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2503.13992)  

**Abstract**: Compression is at the heart of intelligence. A theoretically optimal way to compress any sequence of data is to find the shortest program that outputs that sequence and then halts. However, such 'Kolmogorov compression' is uncomputable, and code generating LLMs struggle to approximate this theoretical ideal, as it requires reasoning, planning and search capabilities beyond those of current models. In this work, we introduce the KoLMogorov-Test (KT), a compression-as-intelligence test for code generating LLMs. In KT a model is presented with a sequence of data at inference time, and asked to generate the shortest program that produces the sequence. We identify several benefits of KT for both evaluation and training: an essentially infinite number of problem instances of varying difficulty is readily available, strong baselines already exist, the evaluation metric (compression) cannot be gamed, and pretraining data contamination is highly unlikely. To evaluate current models, we use audio, text, and DNA data, as well as sequences produced by random synthetic programs. Current flagship models perform poorly - both GPT4-o and Llama-3.1-405B struggle on our natural and synthetic sequences. On our synthetic distribution, we are able to train code generation models with lower compression rates than previous approaches. Moreover, we show that gains on synthetic data generalize poorly to real data, suggesting that new innovations are necessary for additional gains on KT. 

---
# Empowering Smaller Models: Tuning LLaMA and Gemma with Chain-of-Thought for Ukrainian Exam Tasks 

**Authors**: Mykyta Syromiatnikov, Victoria Ruvinskaya, Nataliia Komleva  

**Link**: [PDF](https://arxiv.org/pdf/2503.13988)  

**Abstract**: Leading large language models have demonstrated impressive capabilities in reasoning-intensive tasks, such as standardized educational testing. However, they often require extensive training in low-resource settings with inaccessible infrastructure. Small or compact models, though more efficient, frequently lack sufficient support for underrepresented languages, leaving a performance gap in critical domains. This work explores the potential of parameter-efficient fine-tuning of compact open-weight language models to handle reasoning-intensive tasks in the underrepresented Ukrainian language, building on the findings of the ZNO-Eval benchmark. Parameter-efficient fine-tuning of LLaMA 3.1 (8 billion parameters), LLaMA 3.2 (3 billion parameters), and Gemma 2 (9 billion parameters) models on chain-of-thought solutions resulted in a modest test score improvement of up to 17.4% on complex matching tasks and 1.6% overall compared to tuning on answer letters alone, offering enhanced interpretability and robustness. In addition, the proposed tuning method with joint task topic and step-by-step solution generation outperforms standard chain-of-thought tuning in matching tasks and provides a 5.4% gain over the best LLaMA 3.2 model due to guiding the model to recall and apply domain-relevant information. Contrasting obtained results with zero-shot evaluations of leading open-weight and proprietary models such as Qwen, DeepSeek R1, OpenAI o1 and o3, Gemini, and Claude, highlight that fine-tuning LLaMA and Gemma models with 2,032 step-by-step solutions and 20 to 50 million trainable parameters on a single A100 GPU lets them outperform GPT-4o mini, Mistral Large, and larger open-weight models. This research also evaluates how merging the quantized adapter with the base model influences the generation quality. Source code and tuned models are available at this https URL. 

---
# Navigating Rifts in Human-LLM Grounding: Study and Benchmark 

**Authors**: Omar Shaikh, Hussein Mozannar, Gagan Bansal, Adam Fourney, Eric Horvitz  

**Link**: [PDF](https://arxiv.org/pdf/2503.13975)  

**Abstract**: Language models excel at following instructions but often struggle with the collaborative aspects of conversation that humans naturally employ. This limitation in grounding -- the process by which conversation participants establish mutual understanding -- can lead to outcomes ranging from frustrated users to serious consequences in high-stakes scenarios. To systematically study grounding challenges in human-LLM interactions, we analyze logs from three human-assistant datasets: WildChat, MultiWOZ, and Bing Chat. We develop a taxonomy of grounding acts and build models to annotate and forecast grounding behavior. Our findings reveal significant differences in human-human and human-LLM grounding: LLMs were three times less likely to initiate clarification and sixteen times less likely to provide follow-up requests than humans. Additionally, early grounding failures predicted later interaction breakdowns. Building on these insights, we introduce RIFTS: a benchmark derived from publicly available LLM interaction data containing situations where LLMs fail to initiate grounding. We note that current frontier models perform poorly on RIFTS, highlighting the need to reconsider how we train and prompt LLMs for human interaction. To this end, we develop a preliminary intervention that mitigates grounding failures. 

---
# ConSCompF: Consistency-focused Similarity Comparison Framework for Generative Large Language Models 

**Authors**: Alexey Karev, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13923)  

**Abstract**: Large language models (LLMs) have been one of the most important discoveries in machine learning in recent years. LLM-based artificial intelligence (AI) assistants, such as ChatGPT, have consistently attracted the attention from researchers, investors, and the general public, driving the rapid growth of this industry. With the frequent introduction of new LLMs to the market, it becomes increasingly difficult to differentiate between them, creating a demand for new LLM comparison methods.
In this research, the Consistency-focused Similarity Comparison Framework (ConSCompF) for generative large language models is proposed. It compares texts generated by two LLMs and produces a similarity score, indicating the overall degree of similarity between their responses. The main advantage of this framework is that it can operate on a small number of unlabeled data, such as chatbot instruction prompts, and does not require LLM developers to disclose any information about their product.
To evaluate the efficacy of ConSCompF, two experiments aimed at identifying similarities between multiple LLMs are conducted. Additionally, these experiments examine the correlation between the similarity scores generated by ConSCompF and the differences in the outputs produced by other benchmarking techniques, such as ROUGE-L. Finally, a series of few-shot LLM comparison experiments is conducted to evaluate the performance of ConSCompF in a few-shot LLM comparison scenario.
The proposed framework can be used for calculating similarity matrices of multiple LLMs, which can be effectively visualized using principal component analysis (PCA). The ConSCompF output may provide useful insights into data that might have been used during LLM training and help detect possible investment fraud attempts. 

---
# COMM:Concentrated Margin Maximization for Robust Document-Level Relation Extraction 

**Authors**: Zhichao Duan, Tengyu Pan, Zhenyu Li, Xiuxing Li, Jianyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13885)  

**Abstract**: Document-level relation extraction (DocRE) is the process of identifying and extracting relations between entities that span multiple sentences within a document. Due to its realistic settings, DocRE has garnered increasing research attention in recent years. Previous research has mostly focused on developing sophisticated encoding models to better capture the intricate patterns between entity pairs. While these advancements are undoubtedly crucial, an even more foundational challenge lies in the data itself. The complexity inherent in DocRE makes the labeling process prone to errors, compounded by the extreme sparsity of positive relation samples, which is driven by both the limited availability of positive instances and the broad diversity of positive relation types. These factors can lead to biased optimization processes, further complicating the task of accurate relation extraction. Recognizing these challenges, we have developed a robust framework called \textit{\textbf{COMM}} to better solve DocRE. \textit{\textbf{COMM}} operates by initially employing an instance-aware reasoning method to dynamically capture pertinent information of entity pairs within the document and extract relational features. Following this, \textit{\textbf{COMM}} takes into account the distribution of relations and the difficulty of samples to dynamically adjust the margins between prediction logits and the decision threshold, a process we call Concentrated Margin Maximization. In this way, \textit{\textbf{COMM}} not only enhances the extraction of relevant relational features but also boosts DocRE performance by addressing the specific challenges posed by the data. Extensive experiments and analysis demonstrate the versatility and effectiveness of \textit{\textbf{COMM}}, especially its robustness when trained on low-quality data (achieves \textgreater 10\% performance gains). 

---
# Enabling Inclusive Systematic Reviews: Incorporating Preprint Articles with Large Language Model-Driven Evaluations 

**Authors**: Rui Yang, Jiayi Tong, Haoyuan Wang, Hui Huang, Ziyang Hu, Peiyu Li, Nan Liu, Christopher J. Lindsell, Michael J. Pencina, Yong Chen, Chuan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.13857)  

**Abstract**: Background. Systematic reviews in comparative effectiveness research require timely evidence synthesis. Preprints accelerate knowledge dissemination but vary in quality, posing challenges for systematic reviews.
Methods. We propose AutoConfidence (automated confidence assessment), an advanced framework for predicting preprint publication, which reduces reliance on manual curation and expands the range of predictors, including three key advancements: (1) automated data extraction using natural language processing techniques, (2) semantic embeddings of titles and abstracts, and (3) large language model (LLM)-driven evaluation scores. Additionally, we employed two prediction models: a random forest classifier for binary outcome and a survival cure model that predicts both binary outcome and publication risk over time.
Results. The random forest classifier achieved AUROC 0.692 with LLM-driven scores, improving to 0.733 with semantic embeddings and 0.747 with article usage metrics. The survival cure model reached AUROC 0.716 with LLM-driven scores, improving to 0.731 with semantic embeddings. For publication risk prediction, it achieved a concordance index of 0.658, increasing to 0.667 with semantic embeddings.
Conclusion. Our study advances the framework for preprint publication prediction through automated data extraction and multiple feature integration. By combining semantic embeddings with LLM-driven evaluations, AudoConfidence enhances predictive performance while reducing manual annotation burden. The framework has the potential to facilitate systematic incorporation of preprint articles in evidence-based medicine, supporting researchers in more effective evaluation and utilization of preprint resources. 

---
# Spotting Persuasion: A Low-cost Model for Persuasion Detection in Political Ads on Social Media 

**Authors**: Elyas Meguellati, Stefano Civelli, Pietro Bernardelle, Shazia Sadiq, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2503.13844)  

**Abstract**: In the realm of political advertising, persuasion operates as a pivotal element within the broader framework of propaganda, exerting profound influences on public opinion and electoral outcomes. In this paper, we (1) introduce a lightweight model for persuasive text detection that achieves state-of-the-art performance in Subtask 3 of SemEval 2023 Task 3, while significantly reducing the computational resource requirements; and (2) leverage the proposed model to gain insights into political campaigning strategies on social media platforms by applying it to a real-world dataset we curated, consisting of Facebook political ads from the 2022 Australian Federal election campaign. Our study shows how subtleties can be found in persuasive political advertisements and presents a pragmatic approach to detect and analyze such strategies with limited resources, enhancing transparency in social media political campaigns. 

---
# Self-Vocabularizing Training for Neural Machine Translation 

**Authors**: Pin-Jie Lin, Ernie Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13837)  

**Abstract**: Past vocabulary learning techniques identify relevant vocabulary before training, relying on statistical and entropy-based assumptions that largely neglect the role of model training. Empirically, we observe that trained translation models are induced to use a byte-pair encoding (BPE) vocabulary subset distinct from the original BPE vocabulary, leading to performance improvements when retrained with the induced vocabulary. In this paper, we analyze this discrepancy in neural machine translation by examining vocabulary and entropy shifts during self-training--where each iteration generates a labeled dataset by pairing source sentences with the model's predictions to define a new vocabulary. Building on these insights, we propose self-vocabularizing training, an iterative method that self-selects a smaller, more optimal vocabulary, yielding up to a 1.49 BLEU improvement. Moreover, we find that deeper model architectures lead to both an increase in unique token usage and a 6-8% reduction in vocabulary size. 

---
# Mitigating KV Cache Competition to Enhance User Experience in LLM Inference 

**Authors**: Haiying Shen, Tanmoy Sen  

**Link**: [PDF](https://arxiv.org/pdf/2503.13773)  

**Abstract**: In Large Language Model (LLM) serving, the KV-cache (KVC) bottleneck causes high tail Time-to-First-Token (TTFT) and Time-Between-Tokens (TBT), impairing user experience, particularly in time-sensitive applications. However, satisfying both TTFT and TBT service-level objectives (SLOs) is challenging. To address this, we propose a system, named CacheOPT for mitigating KV Cache competition, based on key insights from our measurements, incorporating novel components. First, it estimates a request's output length, bounding the deviation with a high specified probability, adjusted based on the request arrival rate. Second, it allocates the estimated KVC demand to a request, and reuses other requests' allocated KVC to avoid preemptions while reducing waiting time. Third, it proactively allocates KVC before instead of at the time a request exhausts its allocation and reserves KVC globally to prevent preemptions. Fourth, it chooses a request that has long TBT SLO, long job remaining time and short preemption time to preempt. Fifth, it selects the shortest-latency strategy between swapping and recomputation for preemptions. Experiments show that CacheOPT achieves up to 3.29$\times$ and 2.83$\times$ lower tail TBT and tail TTFT, 47\% and 53\% higher TTFT and TBT SLO attainments, and supports up to 1.58$\times$ higher request arrival rate than the state-of-the-art methods. 

---
# AccelGen: Heterogeneous SLO-Guaranteed High-Throughput LLM Inference Serving for Diverse Applications 

**Authors**: Haiying Shen, Tanmoy Sen  

**Link**: [PDF](https://arxiv.org/pdf/2503.13737)  

**Abstract**: In this paper, we consider a mixed-prompt scenario for a large language model (LLM) inference serving system that supports diverse applications with both short prompts and long prompts and heterogeneous SLOs for iteration time. To improve throughput when handling long prompts, previous research introduces a chunking method, but has not addressed heterogeneous SLOs. To address the limitation, we propose AccelGen, a high-throughput LLM inference serving system with heterogeneous SLO guarantees for diverse applications. AccelGen introduces four core components: (1) SLO-guaranteed dynamic chunking, which dynamically adjusts chunk sizes to maximize GPU compute utilization while meeting iteration-level SLOs; (2) Iteration-level SLO-based task prioritization, which prioritizes tight-SLO requests and batches requests with similar SLOs; (3) Multi-resource-aware batching, which selects queued requests to maximize the utilizations of both GPU compute resource and key-value cache (KVC). Trace-driven real experiments demonstrate that AccelGen achieves 1.42-11.21X higher throughput, 1.43-13.71X higher goodput, 37-90% higher SLO attainment, and 1.61-12.22X lower response latency compared to the state-of-the-art approaches. It achieves performance near the Oracle, which optimally maximizes goodput. 

---
# CoDet-M4: Detecting Machine-Generated Code in Multi-Lingual, Multi-Generator and Multi-Domain Settings 

**Authors**: Daniil Orel, Dilshod Azizov, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2503.13733)  

**Abstract**: Large language models (LLMs) have revolutionized code generation, automating programming with remarkable efficiency. However, these advancements challenge programming skills, ethics, and assessment integrity, making the detection of LLM-generated code essential for maintaining accountability and standards. While, there has been some research on this problem, it generally lacks domain coverage and robustness, and only covers a small number of programming languages. To this end, we propose a framework capable of distinguishing between human- and LLM-written code across multiple programming languages, code generators, and domains. We use a large-scale dataset from renowned platforms and LLM-based code generators, alongside applying rigorous data quality checks, feature engineering, and comparative analysis using evaluation of traditional machine learning models, pre-trained language models (PLMs), and LLMs for code detection. We perform an evaluation on out-of-domain scenarios, such as detecting the authorship and hybrid authorship of generated code and generalizing to unseen models, domains, and programming languages. Moreover, our extensive experiments show that our framework effectively distinguishes human- from LLM-written code and sets a new benchmark for this task. 

---
# Atyaephyra at SemEval-2025 Task 4: Low-Rank NPO 

**Authors**: Jan Bronec, Jindřich Helcl  

**Link**: [PDF](https://arxiv.org/pdf/2503.13690)  

**Abstract**: We present a submission to the SemEval 2025 shared task on unlearning sensitive content from LLMs. Our approach employs negative preference optimization using low-rank adaptation. We show that we can utilize this combination to cheaply compute additional regularization terms, which help with unlearning stabilization. The results of our approach significantly exceed the shared task baselines. 

---
# Feature Extraction and Analysis for GPT-Generated Text 

**Authors**: A. Selvioğlu, V. Adanova, M. Atagoziev  

**Link**: [PDF](https://arxiv.org/pdf/2503.13687)  

**Abstract**: With the rise of advanced natural language models like GPT, distinguishing between human-written and GPT-generated text has become increasingly challenging and crucial across various domains, including academia. The long-standing issue of plagiarism has grown more pressing, now compounded by concerns about the authenticity of information, as it is not always clear whether the presented facts are genuine or fabricated. In this paper, we present a comprehensive study of feature extraction and analysis for differentiating between human-written and GPT-generated text. By applying machine learning classifiers to these extracted features, we evaluate the significance of each feature in detection. Our results demonstrate that human and GPT-generated texts exhibit distinct writing styles, which can be effectively captured by our features. Given sufficiently long text, the two can be differentiated with high accuracy. 

---
# Pensez: Less Data, Better Reasoning -- Rethinking French LLM 

**Authors**: Huy Hoang Ha  

**Link**: [PDF](https://arxiv.org/pdf/2503.13661)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in various natural language processing tasks. However, achieving strong performance in specialized domains like mathematical reasoning and non-English languages often requires extensive training on massive datasets. This paper investigates a contrasting approach: strategic fine-tuning on a small, high-quality, bilingual (English-French) dataset to enhance both the reasoning capabilities and French language proficiency of a large language model. Rather than relying on scale, we explore the hypothesis that targeted data curation and optimized training can achieve competitive, or even superior, performance. We demonstrate, through targeted supervised fine-tuning (SFT) on only 2,000 carefully selected samples, significant improvements in mathematical reasoning. Specifically, Pensez 7B exhibits an increase in accuracy of the base model up to 20% on the AIME25 and a 12% increase on a French MATH level 5 benchmark. These results challenge the prevailing assumption that massive datasets are aprerequisite for strong reasoning performance in LLMs, highlighting the potential of strategic data curation and optimized fine-tuning for enhancing both specialized skills and multilingual capabilities. Our findings have implications for the efficient development of high-performing, multilingual LLMs, especially in resource-constrained scenarios. 

---
# ML-SpecQD: Multi-Level Speculative Decoding with Quantized Drafts 

**Authors**: Evangelos Georganas, Dhiraj Kalamkar, Alexander Kozlov, Alexander Heinecke  

**Link**: [PDF](https://arxiv.org/pdf/2503.13565)  

**Abstract**: Speculative decoding (SD) has emerged as a method to accelerate LLM inference without sacrificing any accuracy over the 16-bit model inference. In a typical SD setup, the idea is to use a full-precision, small, fast model as "draft" to generate the next few tokens and use the "target" large model to verify the draft-generated tokens. The efficacy of this method heavily relies on the acceptance ratio of the draft-generated tokens and the relative token throughput of the draft versus the target model. Nevertheless, an efficient SD pipeline requires pre-training and aligning the draft model to the target model, making it impractical for LLM inference in a plug-and-play fashion. In this work, we propose using MXFP4 models as drafts in a plug-and-play fashion since the MXFP4 Weight-Only-Quantization (WOQ) merely direct-casts the BF16 target model weights to MXFP4. In practice, our plug-and-play solution gives speedups up to 2x over the BF16 baseline. Then we pursue an opportunity for further acceleration: the MXFP4 draft token generation itself can be accelerated via speculative decoding by using yet another smaller draft. We call our method ML-SpecQD: Multi-Level Speculative Decoding with Quantized Drafts since it recursively applies speculation for accelerating the draft-token generation. Combining Multi-Level Speculative Decoding with MXFP4 Quantized Drafts we outperform state-of-the-art speculative decoding, yielding speedups up to 2.72x over the BF16 baseline. 

---
# MES-RAG: Bringing Multi-modal, Entity-Storage, and Secure Enhancements to RAG 

**Authors**: Pingyu Wu, Daiheng Gao, Jing Tang, Huimin Chen, Wenbo Zhou, Weiming Zhang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13563)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves Large Language Models (LLMs) by using external knowledge, but it struggles with precise entity information retrieval. In this paper, we proposed MES-RAG framework, which enhances entity-specific query handling and provides accurate, secure, and consistent responses. MES-RAG introduces proactive security measures that ensure system integrity by applying protections prior to data access. Additionally, the system supports real-time multi-modal outputs, including text, images, audio, and video, seamlessly integrating into existing RAG architectures. Experimental results demonstrate that MES-RAG significantly improves both accuracy and recall, highlighting its effectiveness in advancing the security and utility of question-answering, increasing accuracy to 0.83 (+0.25) on targeted task. Our code and data are available at this https URL. 

---
# Towards Hierarchical Multi-Step Reward Models for Enhanced Reasoning in Large Language Models 

**Authors**: Teng Wang, Zhangyi Jiang, Zhenqi He, Wenhan Yang, Yanan Zheng, Zeyu Li, Zifan He, Shenyang Tong, Hailei Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.13551)  

**Abstract**: Recent studies show that Large Language Models (LLMs) achieve strong reasoning capabilities through supervised fine-tuning or reinforcement learning. However, a key approach, the Process Reward Model (PRM), suffers from reward hacking, making it unreliable in identifying the best intermediate steps. In this paper, we propose a novel reward model approach, Hierarchical Reward Model (HRM), which evaluates both individual and consecutive reasoning steps from fine-grained and coarse-grained level. HRM performs better in assessing reasoning coherence and self-reflection, particularly when the previous reasoning step is incorrect. Furthermore, to address the inefficiency of autonomous generating PRM training data via Monte Carlo Tree Search (MCTS), we introduce a lightweight and effective data augmentation strategy called Hierarchical Node Compression (HNC) based on node merging (combining two consecutive reasoning steps into one step) in the tree structure. This approach diversifies MCTS results for HRM with negligible computational overhead, enhancing label robustness by introducing noise. Empirical results on the PRM800K dataset demonstrate that HRM, in conjunction with HNC, achieves superior stability and reliability in evaluation compared to PRM. Furthermore, cross-domain evaluations on MATH500 and GSM8K confirm HRM's superior generalization and robustness across diverse reasoning tasks. The code for all experiments will be released at https: //github.com/tengwang0318/hierarchial_reward_model. 

---
# Agent-Enhanced Large Language Models for Researching Political Institutions 

**Authors**: Joseph R. Loffredo, Suyeol Yun  

**Link**: [PDF](https://arxiv.org/pdf/2503.13524)  

**Abstract**: The applications of Large Language Models (LLMs) in political science are rapidly expanding. This paper demonstrates how LLMs, when augmented with predefined functions and specialized tools, can serve as dynamic agents capable of streamlining tasks such as data collection, preprocessing, and analysis. Central to this approach is agentic retrieval-augmented generation (Agentic RAG), which equips LLMs with action-calling capabilities for interaction with external knowledge bases. Beyond information retrieval, LLM agents may incorporate modular tools for tasks like document summarization, transcript coding, qualitative variable classification, and statistical modeling. To demonstrate the potential of this approach, we introduce CongressRA, an LLM agent designed to support scholars studying the U.S. Congress. Through this example, we highlight how LLM agents can reduce the costs of replicating, testing, and extending empirical research using the domain-specific data that drives the study of political institutions. 

---
# Evaluating the Process Modeling Abilities of Large Language Models -- Preliminary Foundations and Results 

**Authors**: Peter Fettke, Constantin Houy  

**Link**: [PDF](https://arxiv.org/pdf/2503.13520)  

**Abstract**: Large language models (LLM) have revolutionized the processing of natural language. Although first benchmarks of the process modeling abilities of LLM are promising, it is currently under debate to what extent an LLM can generate good process models. In this contribution, we argue that the evaluation of the process modeling abilities of LLM is far from being trivial. Hence, available evaluation results must be taken carefully. For example, even in a simple scenario, not only the quality of a model should be taken into account, but also the costs and time needed for generation. Thus, an LLM does not generate one optimal solution, but a set of Pareto-optimal variants. Moreover, there are several further challenges which have to be taken into account, e.g. conceptualization of quality, validation of results, generalizability, and data leakage. We discuss these challenges in detail and discuss future experiments to tackle these challenges scientifically. 

---
# Examples as the Prompt: A Scalable Approach for Efficient LLM Adaptation in E-Commerce 

**Authors**: Jingying Zeng, Zhenwei Dai, Hui Liu, Samarth Varshney, Zhiji Liu, Chen Luo, Zhen Li, Qi He, Xianfeng Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13518)  

**Abstract**: Prompting LLMs offers an efficient way to guide output generation without explicit model training. In the e-commerce domain, prompting-based applications are widely used for tasks such as query understanding, recommender systems, and customer support. However, adapting LLMs to different tasks often requires extensive prompt engineering by domain experts, along with frequent updates to align with evolving business needs. Additionally, crafting fully unbiased natural language prompts remains a challenge for humans. To address these challenges, we propose a novel framework, Examples as the Prompt (EaP) which leverages labeled data to enhance prompts. Specifically, EaP automatically selects the most representative examples to maximize the few-shot capability of LLMs. It is efficient due to its unsupervised example selection and adaptive to potential data distribution shifts. We validate EaP on four real-world production use cases, demonstrating that it achieves comparable or even superior performance comparing to hand-crafted prompts designed by domain experts. Additionally, we introduce EaP_lite, which entirely replaces the natural language components of prompts with labeled examples. EaP_lite improves LLM inference speed by up to 70% without compromising performance. Latest online A/B test shows that using EaP and EaP_lite for data labeling can bring significant composite revenue gain by 0.06%. 

---
# CURIE: Evaluating LLMs On Multitask Scientific Long Context Understanding and Reasoning 

**Authors**: Hao Cui, Zahra Shamsi, Gowoon Cheon, Xuejian Ma, Shutong Li, Maria Tikhanovskaya, Peter Norgaard, Nayantara Mudur, Martyna Plomecka, Paul Raccuglia, Yasaman Bahri, Victor V. Albert, Pranesh Srinivasan, Haining Pan, Philippe Faist, Brian Rohr, Michael J. Statt, Dan Morris, Drew Purves, Elise Kleeman, Ruth Alcantara, Matthew Abraham, Muqthar Mohammad, Ean Phing VanLee, Chenfei Jiang, Elizabeth Dorfman, Eun-Ah Kim, Michael P Brenner, Viren Jain, Sameera Ponda, Subhashini Venugopalan  

**Link**: [PDF](https://arxiv.org/pdf/2503.13517)  

**Abstract**: Scientific problem-solving involves synthesizing information while applying expert knowledge. We introduce CURIE, a scientific long-Context Understanding,Reasoning and Information Extraction benchmark to measure the potential of Large Language Models (LLMs) in scientific problem-solving and assisting scientists in realistic workflows. This benchmark introduces ten challenging tasks with a total of 580 problems and solution pairs curated by experts in six disciplines - materials science, condensed matter physics, quantum computing, geospatial analysis, biodiversity, and proteins - covering both experimental and theoretical work-flows in science. We evaluate a range of closed and open LLMs on tasks in CURIE which requires domain expertise, comprehension of long in-context information,and multi-step reasoning. While Gemini Flash 2.0 and Claude-3 show consistent high comprehension across domains, the popular GPT-4o and command-R+ fail dramatically on protein sequencing tasks. With the best performance at 32% there is much room for improvement for all models. We hope that insights gained from CURIE can guide the future development of LLMs in sciences. Evaluation code and data are in this https URL 

---
# RAG-KG-IL: A Multi-Agent Hybrid Framework for Reducing Hallucinations and Enhancing LLM Reasoning through RAG and Incremental Knowledge Graph Learning Integration 

**Authors**: Hong Qing Yu, Frank McQuade  

**Link**: [PDF](https://arxiv.org/pdf/2503.13514)  

**Abstract**: This paper presents RAG-KG-IL, a novel multi-agent hybrid framework designed to enhance the reasoning capabilities of Large Language Models (LLMs) by integrating Retrieval-Augmented Generation (RAG) and Knowledge Graphs (KGs) with an Incremental Learning (IL) approach. Despite recent advancements, LLMs still face significant challenges in reasoning with structured data, handling dynamic knowledge evolution, and mitigating hallucinations, particularly in mission-critical domains. Our proposed RAG-KG-IL framework addresses these limitations by employing a multi-agent architecture that enables continuous knowledge updates, integrates structured knowledge, and incorporates autonomous agents for enhanced explainability and reasoning. The framework utilizes RAG to ensure the generated responses are grounded in verifiable information, while KGs provide structured domain knowledge for improved consistency and depth of understanding. The Incremental Learning approach allows for dynamic updates to the knowledge base without full retraining, significantly reducing computational overhead and improving the model's adaptability. We evaluate the framework using real-world case studies involving health-related queries, comparing it to state-of-the-art models like GPT-4o and a RAG-only baseline. Experimental results demonstrate that our approach significantly reduces hallucination rates and improves answer completeness and reasoning accuracy. The results underscore the potential of combining RAG, KGs, and multi-agent systems to create intelligent, adaptable systems capable of real-time knowledge integration and reasoning in complex domains. 

---
# Prompt Sentiment: The Catalyst for LLM Change 

**Authors**: Vishal Gandhi, Sagar Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2503.13510)  

**Abstract**: The rise of large language models (LLMs) has revolutionized natural language processing (NLP), yet the influence of prompt sentiment, a latent affective characteristic of input text, remains underexplored. This study systematically examines how sentiment variations in prompts affect LLM-generated outputs in terms of coherence, factuality, and bias. Leveraging both lexicon-based and transformer-based sentiment analysis methods, we categorize prompts and evaluate responses from five leading LLMs: Claude, DeepSeek, GPT-4, Gemini, and LLaMA. Our analysis spans six AI-driven applications, including content generation, conversational AI, legal and financial analysis, healthcare AI, creative writing, and technical documentation. By transforming prompts, we assess their impact on output quality. Our findings reveal that prompt sentiment significantly influences model responses, with negative prompts often reducing factual accuracy and amplifying bias, while positive prompts tend to increase verbosity and sentiment propagation. These results highlight the importance of sentiment-aware prompt engineering for ensuring fair and reliable AI-generated content. 

---
# It is Too Many Options: Pitfalls of Multiple-Choice Questions in Generative AI and Medical Education 

**Authors**: Shrutika Singh, Anton Alyakin, Daniel Alexander Alber, Jaden Stryker, Ai Phuong S Tong, Karl Sangwon, Nicolas Goff, Mathew de la Paz, Miguel Hernandez-Rovira, Ki Yun Park, Eric Claude Leuthardt, Eric Karl Oermann  

**Link**: [PDF](https://arxiv.org/pdf/2503.13508)  

**Abstract**: The performance of Large Language Models (LLMs) on multiple-choice question (MCQ) benchmarks is frequently cited as proof of their medical capabilities. We hypothesized that LLM performance on medical MCQs may in part be illusory and driven by factors beyond medical content knowledge and reasoning capabilities. To assess this, we created a novel benchmark of free-response questions with paired MCQs (FreeMedQA). Using this benchmark, we evaluated three state-of-the-art LLMs (GPT-4o, GPT-3.5, and LLama-3-70B-instruct) and found an average absolute deterioration of 39.43% in performance on free-response questions relative to multiple-choice (p = 1.3 * 10-5) which was greater than the human performance decline of 22.29%. To isolate the role of the MCQ format on performance, we performed a masking study, iteratively masking out parts of the question stem. At 100% masking, the average LLM multiple-choice performance was 6.70% greater than random chance (p = 0.002) with one LLM (GPT-4o) obtaining an accuracy of 37.34%. Notably, for all LLMs the free-response performance was near zero. Our results highlight the shortcomings in medical MCQ benchmarks for overestimating the capabilities of LLMs in medicine, and, broadly, the potential for improving both human and machine assessments using LLM-evaluated free-response questions. 

---
# NeurIPS 2023 LLM Efficiency Fine-tuning Competition 

**Authors**: Mark Saroufim, Yotam Perlitz, Leshem Choshen, Luca Antiga, Greg Bowyer, Christian Puhrsch, Driss Guessous, Supriya Rao, Geeta Chauhan, Ashvini Kumar, Jindal Pawan Kumar, Rajpoot Ankur Parikh, Joe Isaacson, Weiwei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13507)  

**Abstract**: Our analysis of the NeurIPS 2023 large language model (LLM) fine-tuning competition revealed the following trend: top-performing models exhibit significant overfitting on benchmark datasets, mirroring the broader issue of benchmark overfitting on popular leaderboards and that data curation is essential in order to get a high performing LLM. The competition, which consisted of two stages - an open evaluation stage with publicly available tasks and a closed evaluation stage with unseen tasks - allowed us to assess the generalizability of fine-tuned LLMs. Our results highlight the limitations of current benchmark-based evaluation schemes for generative models and demonstrate the need for more robust evaluation methods. Notably, the winning submissions utilized standard open-source libraries and focused primarily on data curation. To facilitate further research and promote reproducibility, we release all competition entries, Docker files, and evaluation infrastructure, providing a valuable resource for the community to explore fine-tuning, overfitting, and reproducibility in LLMs. 

---
# Ensemble Learning for Large Language Models in Text and Code Generation: A Survey 

**Authors**: Mari Ashiga, Wei Jie, Fan Wu, Vardan Voskanyan, Fateme Dinmohammadi, Paul Brookes, Jingzhi Gong, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13505)  

**Abstract**: Generative pretrained transformers (GPT) are the common large language models (LLMs) used for generating text from natural language inputs. However, the fixed properties of language parameters in individual LLMs can lead to inconsistencies in the generated outputs. This limitation also restricts the models' ability to represent diverse language patterns due to inherent biases. Moreover, many powerful LLMs are closed-source. This prevents organizations from integrating their data into these systems, raising concerns about data privacy and limiting industry applications. Inspired by the successful application of LLM ensemble models in text generation, recent literature has also investigated their potential in code generation. This article reviews these emerging LLM ensemble approaches. Our goal is to enhance readers' understanding of existing techniques and encourage further research and practical implementation, aiming to expand the real-world applications of LLM ensemble models in both text and code generation. We categorize these approaches into seven main methods: weight merging, knowledge fusion, mixture of experts, reward ensemble, output ensemble, routing, and cascading. From this list, we focus on four methods and models that show strong performance and potential for broader applications. We analyze their modeling steps, training methods, and output features to provide a clear understanding of their capabilities. Our findings highlight the benefits of LLM ensemble techniques. These include better representation of diversity, improved output quality, and greater flexibility in applications. This information offers valuable insights for selecting models for various real-world tasks involving text and code generation, and potentially applying methods to multimodal LLMs. 

---
# Gricean Norms as a Basis for Effective Collaboration 

**Authors**: Fardin Saad, Pradeep K. Murukannaiah, Munindar P. Singh  

**Link**: [PDF](https://arxiv.org/pdf/2503.14484)  

**Abstract**: Effective human-AI collaboration hinges not only on the AI agent's ability to follow explicit instructions but also on its capacity to navigate ambiguity, incompleteness, invalidity, and irrelevance in communication. Gricean conversational and inference norms facilitate collaboration by aligning unclear instructions with cooperative principles. We propose a normative framework that integrates Gricean norms and cognitive frameworks -- common ground, relevance theory, and theory of mind -- into large language model (LLM) based agents. The normative framework adopts the Gricean maxims of quantity, quality, relation, and manner, along with inference, as Gricean norms to interpret unclear instructions, which are: ambiguous, incomplete, invalid, or irrelevant. Within this framework, we introduce Lamoids, GPT-4 powered agents designed to collaborate with humans. To assess the influence of Gricean norms in human-AI collaboration, we evaluate two versions of a Lamoid: one with norms and one without. In our experiments, a Lamoid collaborates with a human to achieve shared goals in a grid world (Doors, Keys, and Gems) by interpreting both clear and unclear natural language instructions. Our results reveal that the Lamoid with Gricean norms achieves higher task accuracy and generates clearer, more accurate, and contextually relevant responses than the Lamoid without norms. This improvement stems from the normative framework, which enhances the agent's pragmatic reasoning, fostering effective human-AI collaboration and enabling context-aware communication in LLM-based agents. 

---
# Don't lie to your friends: Learning what you know from collaborative self-play 

**Authors**: Jacob Eisenstein, Reza Aghajani, Adam Fisch, Dheeru Dua, Fantine Huot, Mirella Lapata, Vicky Zayats, Jonathan Berant  

**Link**: [PDF](https://arxiv.org/pdf/2503.14481)  

**Abstract**: To be helpful assistants, AI agents must be aware of their own capabilities and limitations. This includes knowing when to answer from parametric knowledge versus using tools, when to trust tool outputs, and when to abstain or hedge. Such capabilities are hard to teach through supervised fine-tuning because they require constructing examples that reflect the agent's specific capabilities. We therefore propose a radically new approach to teaching agents what they know: \emph{collaborative self-play}. We construct multi-agent collaborations in which the group is rewarded for collectively arriving at correct answers. The desired meta-knowledge emerges from the incentives built into the structure of the interaction. We focus on small societies of agents that have access to heterogeneous tools (corpus-specific retrieval), and therefore must collaborate to maximize their success while minimizing their effort. Experiments show that group-level rewards for multi-agent communities can induce policies that \emph{transfer} to improve tool use and selective prediction in settings where individual agents are deployed in isolation. 

---
# DAPO: An Open-Source LLM Reinforcement Learning System at Scale 

**Authors**: Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, Mingxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14476)  

**Abstract**: Inference scaling empowers LLMs with unprecedented reasoning ability, with reinforcement learning as the core technique to elicit complex reasoning. However, key technical details of state-of-the-art reasoning LLMs are concealed (such as in OpenAI o1 blog and DeepSeek R1 technical report), thus the community still struggles to reproduce their RL training results. We propose the $\textbf{D}$ecoupled Clip and $\textbf{D}$ynamic s$\textbf{A}$mpling $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{DAPO}$) algorithm, and fully open-source a state-of-the-art large-scale RL system that achieves 50 points on AIME 2024 using Qwen2.5-32B base model. Unlike previous works that withhold training details, we introduce four key techniques of our algorithm that make large-scale LLM RL a success. In addition, we open-source our training code, which is built on the verl framework, along with a carefully curated and processed dataset. These components of our open-source system enhance reproducibility and support future research in large-scale LLM RL. 

---
# LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers 

**Authors**: Nikhil Abhyankar, Parshin Shojaee, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2503.14434)  

**Abstract**: Automated feature engineering plays a critical role in improving predictive model performance for tabular learning tasks. Traditional automated feature engineering methods are limited by their reliance on pre-defined transformations within fixed, manually designed search spaces, often neglecting domain knowledge. Recent advances using Large Language Models (LLMs) have enabled the integration of domain knowledge into the feature engineering process. However, existing LLM-based approaches use direct prompting or rely solely on validation scores for feature selection, failing to leverage insights from prior feature discovery experiments or establish meaningful reasoning between feature generation and data-driven performance. To address these challenges, we propose LLM-FE, a novel framework that combines evolutionary search with the domain knowledge and reasoning capabilities of LLMs to automatically discover effective features for tabular learning tasks. LLM-FE formulates feature engineering as a program search problem, where LLMs propose new feature transformation programs iteratively, and data-driven feedback guides the search process. Our results demonstrate that LLM-FE consistently outperforms state-of-the-art baselines, significantly enhancing the performance of tabular prediction models across diverse classification and regression benchmarks. 

---
# ExDDV: A New Dataset for Explainable Deepfake Detection in Video 

**Authors**: Vlad Hondru, Eduard Hogea, Darian Onchis, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14421)  

**Abstract**: The ever growing realism and quality of generated videos makes it increasingly harder for humans to spot deepfake content, who need to rely more and more on automatic deepfake detectors. However, deepfake detectors are also prone to errors, and their decisions are not explainable, leaving humans vulnerable to deepfake-based fraud and misinformation. To this end, we introduce ExDDV, the first dataset and benchmark for Explainable Deepfake Detection in Video. ExDDV comprises around 5.4K real and deepfake videos that are manually annotated with text descriptions (to explain the artifacts) and clicks (to point out the artifacts). We evaluate a number of vision-language models on ExDDV, performing experiments with various fine-tuning and in-context learning strategies. Our results show that text and click supervision are both required to develop robust explainable models for deepfake videos, which are able to localize and describe the observed artifacts. Our novel dataset and code to reproduce the results are available at this https URL. 

---
# Large Language Models for Virtual Human Gesture Selection 

**Authors**: Parisa Ghanad Torshizi, Laura B. Hensel, Ari Shapiro, Stacy C. Marsella  

**Link**: [PDF](https://arxiv.org/pdf/2503.14408)  

**Abstract**: Co-speech gestures convey a wide variety of meanings and play an important role in face-to-face human interactions. These gestures significantly influence the addressee's engagement, recall, comprehension, and attitudes toward the speaker. Similarly, they impact interactions between humans and embodied virtual agents. The process of selecting and animating meaningful gestures has thus become a key focus in the design of these agents. However, automating this gesture selection process poses a significant challenge. Prior gesture generation techniques have varied from fully automated, data-driven methods, which often struggle to produce contextually meaningful gestures, to more manual approaches that require crafting specific gesture expertise and are time-consuming and lack generalizability. In this paper, we leverage the semantic capabilities of Large Language Models to develop a gesture selection approach that suggests meaningful, appropriate co-speech gestures. We first describe how information on gestures is encoded into GPT-4. Then, we conduct a study to evaluate alternative prompting approaches for their ability to select meaningful, contextually relevant gestures and to align them appropriately with the co-speech utterance. Finally, we detail and demonstrate how this approach has been implemented within a virtual agent system, automating the selection and subsequent animation of the selected gestures for enhanced human-agent interactions. 

---
# VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation 

**Authors**: Shoubin Yu, Difan Liu, Ziqiao Ma, Yicong Hong, Yang Zhou, Hao Tan, Joyce Chai, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2503.14350)  

**Abstract**: Recent video diffusion models have enhanced video editing, but it remains challenging to handle instructional editing and diverse tasks (e.g., adding, removing, changing) within a unified framework. In this paper, we introduce VEGGIE, a Video Editor with Grounded Generation from Instructions, a simple end-to-end framework that unifies video concept editing, grounding, and reasoning based on diverse user instructions. Specifically, given a video and text query, VEGGIE first utilizes an MLLM to interpret user intentions in instructions and ground them to the video contexts, generating frame-specific grounded task queries for pixel-space responses. A diffusion model then renders these plans and generates edited videos that align with user intent. To support diverse tasks and complex instructions, we employ a curriculum learning strategy: first aligning the MLLM and video diffusion model with large-scale instructional image editing data, followed by end-to-end fine-tuning on high-quality multitask video data. Additionally, we introduce a novel data synthesis pipeline to generate paired instructional video editing data for model training. It transforms static image data into diverse, high-quality video editing samples by leveraging Image-to-Video models to inject dynamics. VEGGIE shows strong performance in instructional video editing with different editing skills, outperforming the best instructional baseline as a versatile model, while other models struggle with multi-tasking. VEGGIE also excels in video object grounding and reasoning segmentation, where other baselines fail. We further reveal how the multiple tasks help each other and highlight promising applications like zero-shot multimodal instructional and in-context video editing. 

---
# MoonCast: High-Quality Zero-Shot Podcast Generation 

**Authors**: Zeqian Ju, Dongchao Yang, Jianwei Yu, Kai Shen, Yichong Leng, Zhengtao Wang, Xu Tan, Xinyu Zhou, Tao Qin, Xiangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.14345)  

**Abstract**: Recent advances in text-to-speech synthesis have achieved notable success in generating high-quality short utterances for individual speakers. However, these systems still face challenges when extending their capabilities to long, multi-speaker, and spontaneous dialogues, typical of real-world scenarios such as podcasts. These limitations arise from two primary challenges: 1) long speech: podcasts typically span several minutes, exceeding the upper limit of most existing work; 2) spontaneity: podcasts are marked by their spontaneous, oral nature, which sharply contrasts with formal, written contexts; existing works often fall short in capturing this spontaneity. In this paper, we propose MoonCast, a solution for high-quality zero-shot podcast generation, aiming to synthesize natural podcast-style speech from text-only sources (e.g., stories, technical reports, news in TXT, PDF, or Web URL formats) using the voices of unseen speakers. To generate long audio, we adopt a long-context language model-based audio modeling approach utilizing large-scale long-context speech data. To enhance spontaneity, we utilize a podcast generation module to generate scripts with spontaneous details, which have been empirically shown to be as crucial as the text-to-speech modeling itself. Experiments demonstrate that MoonCast outperforms baselines, with particularly notable improvements in spontaneity and coherence. 

---
# PENCIL: Long Thoughts with Short Memory 

**Authors**: Chenxiao Yang, Nathan Srebro, David McAllester, Zhiyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.14337)  

**Abstract**: While recent works (e.g. o1, DeepSeek R1) have demonstrated great promise of using long Chain-of-Thought (CoT) to improve reasoning capabilities of language models, scaling it up during test-time is challenging due to inefficient memory usage -- intermediate computations accumulate indefinitely in context even no longer needed for future thoughts. We propose PENCIL, which incorporates a reduction mechanism into the autoregressive generation process, allowing the model to recursively clean up intermediate thoughts based on patterns learned from training. With this reduction mechanism, PENCIL significantly reduces the maximal context length required during generation, and thus can generate longer thoughts with limited memory, solving larger-scale problems given more thinking time. For example, we demonstrate PENCIL achieves 97\% accuracy on the challenging Einstein's puzzle -- a task even large models like GPT-4 struggle with -- using only a small 25M-parameter transformer with 2048 context length. Theoretically, we prove PENCIL can perform universal space-efficient computation by simulating Turing machines with optimal time and space complexity, and thus can solve arbitrary computational tasks that would otherwise be intractable given context window constraints. 

---
# DualToken: Towards Unifying Visual Understanding and Generation with Dual Visual Vocabularies 

**Authors**: Wei Song, Yuran Wang, Zijia Song, Yadong Li, Haoze Sun, Weipeng Chen, Zenan Zhou, Jianhua Xu, Jiaqi Wang, Kaicheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14324)  

**Abstract**: The differing representation spaces required for visual understanding and generation pose a challenge in unifying them within the autoregressive paradigm of large language models. A vision tokenizer trained for reconstruction excels at capturing low-level perceptual details, making it well-suited for visual generation but lacking high-level semantic representations for understanding tasks. Conversely, a vision encoder trained via contrastive learning aligns well with language but struggles to decode back into the pixel space for generation tasks. To bridge this gap, we propose DualToken, a method that unifies representations for both understanding and generation within a single tokenizer. However, directly integrating reconstruction and semantic objectives in a single tokenizer creates conflicts, leading to degraded performance in both reconstruction quality and semantic performance. Instead of forcing a single codebook to handle both semantic and perceptual information, DualToken disentangles them by introducing separate codebooks for high and low-level features, effectively transforming their inherent conflict into a synergistic relationship. As a result, DualToken achieves state-of-the-art performance in both reconstruction and semantic tasks while demonstrating remarkable effectiveness in downstream MLLM understanding and generation tasks. Notably, we also show that DualToken, as a unified tokenizer, surpasses the naive combination of two distinct types vision encoders, providing superior performance within a unified MLLM. 

---
# Benchmarking Failures in Tool-Augmented Language Models 

**Authors**: Eduardo Treviño, Hugo Contant, James Ngai, Graham Neubig, Zora Zhiruo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14227)  

**Abstract**: The integration of tools has extended the capabilities of language models (LMs) beyond vanilla text generation to versatile scenarios. However, tool-augmented language models (TaLMs) often assume 'perfect' information access and tool availability, which may not hold in the real world. To systematically study TaLMs' imperfections, we introduce the FAIL-TALMS benchmark, featuring two major failures: under-specified user queries and non-available tools. FAIL-TALMS contains 1,749 examples using 906 tools across 21 categories, including single- and multi-tool usage. We evaluate top-performing proprietary and open-source models, and find all current models except for Claude struggle to recognize missing tools or information. Further, to study possible mitigation of the failures, we enable real-time human interaction, named the Ask-and-Help (AAH) method, to provide missing information or replace non-functional tools. While AAH can help models solve tasks more correctly when queries are under-specified, it brings minimal benefit when complex tools are broken. 

---
# Speculative Decoding for Verilog: Speed and Quality, All in One 

**Authors**: Changran Xu, Yi Liu, Yunhao Zhou, Shan Huang, Ningyi Xu, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14153)  

**Abstract**: The rapid advancement of large language models (LLMs) has revolutionized code generation tasks across various programming languages. However, the unique characteristics of programming languages, particularly those like Verilog with specific syntax and lower representation in training datasets, pose significant challenges for conventional tokenization and decoding approaches. In this paper, we introduce a novel application of speculative decoding for Verilog code generation, showing that it can improve both inference speed and output quality, effectively achieving speed and quality all in one. Unlike standard LLM tokenization schemes, which often fragment meaningful code structures, our approach aligns decoding stops with syntactically significant tokens, making it easier for models to learn the token distribution. This refinement addresses inherent tokenization issues and enhances the model's ability to capture Verilog's logical constructs more effectively. Our experimental results show that our method achieves up to a 5.05x speedup in Verilog code generation and increases pass@10 functional accuracy on RTLLM by up to 17.19% compared to conventional training strategies. These findings highlight speculative decoding as a promising approach to bridge the quality gap in code generation for specialized programming languages. 

---
# Frac-Connections: Fractional Extension of Hyper-Connections 

**Authors**: Defa Zhu, Hongzhi Huang, Jundong Zhou, Zihao Huang, Yutao Zeng, Banggu Wu, Qiyang Min, Xun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.14125)  

**Abstract**: Residual connections are central to modern deep learning architectures, enabling the training of very deep networks by mitigating gradient vanishing. Hyper-Connections recently generalized residual connections by introducing multiple connection strengths at different depths, thereby addressing the seesaw effect between gradient vanishing and representation collapse. However, Hyper-Connections increase memory access costs by expanding the width of hidden states. In this paper, we propose Frac-Connections, a novel approach that divides hidden states into multiple parts rather than expanding their width. Frac-Connections retain partial benefits of Hyper-Connections while reducing memory consumption. To validate their effectiveness, we conduct large-scale experiments on language tasks, with the largest being a 7B MoE model trained on up to 3T tokens, demonstrating that Frac-Connections significantly outperform residual connections. 

---
# Growing a Twig to Accelerate Large Vision-Language Models 

**Authors**: Zhenwei Shao, Mingyang Wang, Zhou Yu, Wenwen Pan, Yan Yang, Tao Wei, Hongyuan Zhang, Ning Mao, Wei Chen, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14075)  

**Abstract**: Large vision-language models (VLMs) have demonstrated remarkable capabilities in open-world multimodal understanding, yet their high computational overheads pose great challenges for practical deployment. Some recent works have proposed methods to accelerate VLMs by pruning redundant visual tokens guided by the attention maps of VLM's early layers. Despite the success of these token pruning methods, they still suffer from two major shortcomings: (i) considerable accuracy drop due to insensitive attention signals in early layers, and (ii) limited speedup when generating long responses (e.g., 30 tokens). To address the limitations above, we present TwigVLM -- a simple and general architecture by growing a lightweight twig upon an early layer of the base VLM. Compared with most existing VLM acceleration methods purely based on visual token pruning, our TwigVLM not only achieves better accuracy retention by employing a twig-guided token pruning (TTP) strategy, but also yields higher generation speed by utilizing a self-speculative decoding (SSD) strategy. Taking LLaVA-1.5-7B as the base VLM, experimental results show that TwigVLM preserves 96% of the original performance after pruning 88.9% of visual tokens and achieves 154% speedup in generating long responses, delivering significantly better performance in terms of both accuracy and speed over the state-of-the-art VLM acceleration methods. Code will be made publicly available. 

---
# TextInVision: Text and Prompt Complexity Driven Visual Text Generation Benchmark 

**Authors**: Forouzan Fallah, Maitreya Patel, Agneet Chatterjee, Vlad I. Morariu, Chitta Baral, Yezhou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13730)  

**Abstract**: Generating images with embedded text is crucial for the automatic production of visual and multimodal documents, such as educational materials and advertisements. However, existing diffusion-based text-to-image models often struggle to accurately embed text within images, facing challenges in spelling accuracy, contextual relevance, and visual coherence. Evaluating the ability of such models to embed text within a generated image is complicated due to the lack of comprehensive benchmarks. In this work, we introduce TextInVision, a large-scale, text and prompt complexity driven benchmark designed to evaluate the ability of diffusion models to effectively integrate visual text into images. We crafted a diverse set of prompts and texts that consider various attributes and text characteristics. Additionally, we prepared an image dataset to test Variational Autoencoder (VAE) models across different character representations, highlighting that VAE architectures can also pose challenges in text generation within diffusion frameworks. Through extensive analysis of multiple models, we identify common errors and highlight issues such as spelling inaccuracies and contextual mismatches. By pinpointing the failure points across different prompts and texts, our research lays the foundation for future advancements in AI-generated multimodal content. 

---
# Does the Appearance of Autonomous Conversational Robots Affect User Spoken Behaviors in Real-World Conference Interactions? 

**Authors**: Zi Haur Pang, Yahui Fu, Divesh Lala, Mikey Elmers, Koji Inoue, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2503.13625)  

**Abstract**: We investigate the impact of robot appearance on users' spoken behavior during real-world interactions by comparing a human-like android, ERICA, with a less anthropomorphic humanoid, TELECO. Analyzing data from 42 participants at SIGDIAL 2024, we extracted linguistic features such as disfluencies and syntactic complexity from conversation transcripts. The results showed moderate effect sizes, suggesting that participants produced fewer disfluencies and employed more complex syntax when interacting with ERICA. Further analysis involving training classification models like Naïve Bayes, which achieved an F1-score of 71.60\%, and conducting feature importance analysis, highlighted the significant role of disfluencies and syntactic complexity in interactions with robots of varying human-like appearances. Discussing these findings within the frameworks of cognitive load and Communication Accommodation Theory, we conclude that designing robots to elicit more structured and fluent user speech can enhance their communicative alignment with humans. 

---
# Analytic Subspace Routing: How Recursive Least Squares Works in Continual Learning of Large Language Model 

**Authors**: Kai Tong, Kang Pan, Xiao Zhang, Erli Meng, Run He, Yawen Cui, Nuoyan Guo, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13575)  

**Abstract**: Large Language Models (LLMs) possess encompassing capabilities that can process diverse language-related tasks. However, finetuning on LLMs will diminish this general skills and continual finetuning will further cause severe degradation on accumulated knowledge. Recently, Continual Learning (CL) in Large Language Models (LLMs) arises which aims to continually adapt the LLMs to new tasks while maintaining previously learned knowledge and inheriting general skills. Existing techniques either leverage previous data to replay, leading to extra computational costs, or utilize a single parameter-efficient module to learn the downstream task, constraining new knowledge absorption with interference between different tasks. Toward these issues, this paper proposes Analytic Subspace Routing(ASR) to address these challenges. For each task, we isolate the learning within a subspace of deep layers' features via low-rank adaptation, eliminating knowledge interference between different tasks. Additionally, we propose an analytic routing mechanism to properly utilize knowledge learned in different subspaces. Our approach employs Recursive Least Squares to train a multi-task router model, allowing the router to dynamically adapt to incoming data without requiring access to historical data. Also, the router effectively assigns the current task to an appropriate subspace and has a non-forgetting property of previously learned tasks with a solid theoretical guarantee. Experimental results demonstrate that our method achieves near-perfect retention of prior knowledge while seamlessly integrating new information, effectively overcoming the core limitations of existing methods. Our code will be released after acceptance. 

---
# Pareidolic Illusions of Meaning: ChatGPT, Pseudolaw and the Triumph of Form over Substance 

**Authors**: Joe McIntyre  

**Link**: [PDF](https://arxiv.org/pdf/2503.13556)  

**Abstract**: The early 2020s has seen the rise of two strange and potentially quite impactful social phenomena, namely pseudolaw, where users rely upon pseudolegal arguments that mimic the form and ritual of legal argumentation but fundamentally distort the content of law, and generative AI/LLMs, which generate content that uses probabilistic calculations to create outputs that look like human generated text. This article argues that the juxtaposition of the two phenomena helps to reveal that they both share two fundamental traits as both elevate form and appearance over substance and content, and users of both routinely mistake the form for the substance. In drawing upon legal theory, computer science, linguistics and cognitive psychology, the article argues that both phenomena rely upon creating illusions of meaning that users mistake for the underlying primary phenomenon. I then explore four implications of this conception of both phenomena. Firstly, both rely on human tendencies of conceptual pareidolia resulting in the erroneous perception of meaningful linguistic legal patterns from nebulous inputs. Secondly, both rely upon the confidence heuristic, the human cognitive bias for treating confidence as a proxy for competence. Thirdly, both succeed when the primary concern is with the form of the output and not its content. Fourthly, both rely heavily upon the magical thinking of users and the desire for the promise of the approach to be real. The article argues that the legal context helps to reveal a solution for the problems caused by both phenomena as it is only where users possess sufficient legal and technological literacy that it becomes possible to reveal to them the illusionary nature of the phenomena. 

---
# LLM-Mediated Guidance of MARL Systems 

**Authors**: Philipp D. Siedler, Ian Gemp  

**Link**: [PDF](https://arxiv.org/pdf/2503.13553)  

**Abstract**: In complex multi-agent environments, achieving efficient learning and desirable behaviours is a significant challenge for Multi-Agent Reinforcement Learning (MARL) systems. This work explores the potential of combining MARL with Large Language Model (LLM)-mediated interventions to guide agents toward more desirable behaviours. Specifically, we investigate how LLMs can be used to interpret and facilitate interventions that shape the learning trajectories of multiple agents. We experimented with two types of interventions, referred to as controllers: a Natural Language (NL) Controller and a Rule-Based (RB) Controller. The NL Controller, which uses an LLM to simulate human-like interventions, showed a stronger impact than the RB Controller. Our findings indicate that agents particularly benefit from early interventions, leading to more efficient training and higher performance. Both intervention types outperform the baseline without interventions, highlighting the potential of LLM-mediated guidance to accelerate training and enhance MARL performance in challenging environments. 

---
# MentalChat16K: A Benchmark Dataset for Conversational Mental Health Assistance 

**Authors**: Jia Xu, Tianyi Wei, Bojian Hou, Patryk Orzechowski, Shu Yang, Ruochen Jin, Rachael Paulbeck, Joost Wagenaar, George Demiris, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.13509)  

**Abstract**: We introduce MentalChat16K, an English benchmark dataset combining a synthetic mental health counseling dataset and a dataset of anonymized transcripts from interventions between Behavioral Health Coaches and Caregivers of patients in palliative or hospice care. Covering a diverse range of conditions like depression, anxiety, and grief, this curated dataset is designed to facilitate the development and evaluation of large language models for conversational mental health assistance. By providing a high-quality resource tailored to this critical domain, MentalChat16K aims to advance research on empathetic, personalized AI solutions to improve access to mental health support services. The dataset prioritizes patient privacy, ethical considerations, and responsible data usage. MentalChat16K presents a valuable opportunity for the research community to innovate AI technologies that can positively impact mental well-being. 

---
# SciHorizon: Benchmarking AI-for-Science Readiness from Scientific Data to Large Language Models 

**Authors**: Chuan Qin, Xin Chen, Chengrui Wang, Pengmin Wu, Xi Chen, Yihang Cheng, Jingyi Zhao, Meng Xiao, Xiangchao Dong, Qingqing Long, Boya Pan, Han Wu, Chengzan Li, Yuanchun Zhou, Hui Xiong, Hengshu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13503)  

**Abstract**: In recent years, the rapid advancement of Artificial Intelligence (AI) technologies, particularly Large Language Models (LLMs), has revolutionized the paradigm of scientific discovery, establishing AI-for-Science (AI4Science) as a dynamic and evolving field. However, there is still a lack of an effective framework for the overall assessment of AI4Science, particularly from a holistic perspective on data quality and model capability. Therefore, in this study, we propose SciHorizon, a comprehensive assessment framework designed to benchmark the readiness of AI4Science from both scientific data and LLM perspectives. First, we introduce a generalizable framework for assessing AI-ready scientific data, encompassing four key dimensions: Quality, FAIRness, Explainability, and Compliance which are subdivided into 15 sub-dimensions. Drawing on data resource papers published between 2018 and 2023 in peer-reviewed journals, we present recommendation lists of AI-ready datasets for both Earth and Life Sciences, making a novel and original contribution to the field. Concurrently, to assess the capabilities of LLMs across multiple scientific disciplines, we establish 16 assessment dimensions based on five core indicators Knowledge, Understanding, Reasoning, Multimodality, and Values spanning Mathematics, Physics, Chemistry, Life Sciences, and Earth and Space Sciences. Using the developed benchmark datasets, we have conducted a comprehensive evaluation of over 20 representative open-source and closed source LLMs. All the results are publicly available and can be accessed online at this http URL. 

---
# Recent Developments in Deep Learning-based Author Name Disambiguation 

**Authors**: Francesca Cappelli, Giovanni Colavizza, Silvio Peroni  

**Link**: [PDF](https://arxiv.org/pdf/2503.13448)  

**Abstract**: Author Name Disambiguation (AND) is a critical task for digital libraries aiming to link existing authors with their respective publications. Due to the lack of persistent identifiers used by researchers and the presence of intrinsic linguistic challenges, such as homonymy, the development of Deep Learning algorithms to address this issue has become widespread. Many AND deep learning methods have been developed, and surveys exist comparing the approaches in terms of techniques, complexity, performance. However, none explicitly addresses AND methods in the context of deep learning in the latest years (i.e. timeframe 2016-2024). In this paper, we provide a systematic review of state-of-the-art AND techniques based on deep learning, highlighting recent improvements, challenges, and open issues in the field. We find that DL methods have significantly impacted AND by enabling the integration of structured and unstructured data, and hybrid approaches effectively balance supervised and unsupervised learning. 

---
