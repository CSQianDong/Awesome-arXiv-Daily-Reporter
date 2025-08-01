# How Well Can Reasoning Models Identify and Recover from Unhelpful Thoughts? 

**Authors**: Sohee Yang, Sang-Woo Lee, Nora Kassner, Daniela Gottesman, Sebastian Riedel, Mor Geva  

**Link**: [PDF](https://arxiv.org/pdf/2506.10979)  

**Abstract**: Recent reasoning models show the ability to reflect, backtrack, and self-validate their reasoning, which is crucial in spotting mistakes and arriving at accurate solutions. A natural question that arises is how effectively models can perform such self-reevaluation. We tackle this question by investigating how well reasoning models identify and recover from four types of unhelpful thoughts: uninformative rambling thoughts, thoughts irrelevant to the question, thoughts misdirecting the question as a slightly different question, and thoughts that lead to incorrect answers. We show that models are effective at identifying most unhelpful thoughts but struggle to recover from the same thoughts when these are injected into their thinking process, causing significant performance drops. Models tend to naively continue the line of reasoning of the injected irrelevant thoughts, which showcases that their self-reevaluation abilities are far from a general "meta-cognitive" awareness. Moreover, we observe non/inverse-scaling trends, where larger models struggle more than smaller ones to recover from short irrelevant thoughts, even when instructed to reevaluate their reasoning. We demonstrate the implications of these findings with a jailbreak experiment using irrelevant thought injection, showing that the smallest models are the least distracted by harmful-response-triggering thoughts. Overall, our findings call for improvement in self-reevaluation of reasoning models to develop better reasoning and safer systems. 

---
# AutoMind: Adaptive Knowledgeable Agent for Automated Data Science 

**Authors**: Yixin Ou, Yujie Luo, Jingsheng Zheng, Lanning Wei, Shuofei Qiao, Jintian Zhang, Da Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10974)  

**Abstract**: Large Language Model (LLM) agents have shown great potential in addressing real-world data science problems. LLM-driven data science agents promise to automate the entire machine learning pipeline, yet their real-world effectiveness remains limited. Existing frameworks depend on rigid, pre-defined workflows and inflexible coding strategies; consequently, they excel only on relatively simple, classical problems and fail to capture the empirical expertise that human practitioners bring to complex, innovative tasks. In this work, we introduce AutoMind, an adaptive, knowledgeable LLM-agent framework that overcomes these deficiencies through three key advances: (1) a curated expert knowledge base that grounds the agent in domain expert knowledge, (2) an agentic knowledgeable tree search algorithm that strategically explores possible solutions, and (3) a self-adaptive coding strategy that dynamically tailors code generation to task complexity. Evaluations on two automated data science benchmarks demonstrate that AutoMind delivers superior performance versus state-of-the-art baselines. Additional analyses confirm favorable effectiveness, efficiency, and qualitative solution quality, highlighting AutoMind as an efficient and robust step toward fully automated data science. 

---
# ChineseHarm-Bench: A Chinese Harmful Content Detection Benchmark 

**Authors**: Kangwei Liu, Siyuan Cheng, Bozhong Tian, Xiaozhuan Liang, Yuyang Yin, Meng Han, Ningyu Zhang, Bryan Hooi, Xi Chen, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10960)  

**Abstract**: Large language models (LLMs) have been increasingly applied to automated harmful content detection tasks, assisting moderators in identifying policy violations and improving the overall efficiency and accuracy of content review. However, existing resources for harmful content detection are predominantly focused on English, with Chinese datasets remaining scarce and often limited in scope. We present a comprehensive, professionally annotated benchmark for Chinese content harm detection, which covers six representative categories and is constructed entirely from real-world data. Our annotation process further yields a knowledge rule base that provides explicit expert knowledge to assist LLMs in Chinese harmful content detection. In addition, we propose a knowledge-augmented baseline that integrates both human-annotated knowledge rules and implicit knowledge from large language models, enabling smaller models to achieve performance comparable to state-of-the-art LLMs. Code and data are available at this https URL. 

---
# Domain2Vec: Vectorizing Datasets to Find the Optimal Data Mixture without Training 

**Authors**: Mozhi Zhang, Howe Tissue, Lu Wang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10952)  

**Abstract**: We introduce~\textsc{Domain2Vec}, a novel approach that decomposes any dataset into a linear combination of several \emph{meta-domains}, a new concept designed to capture the key underlying features of datasets. \textsc{Domain2Vec} maintains a vocabulary of meta-domains and uses a classifier to decompose any given dataset into a domain vector that corresponds to a distribution over this vocabulary. These domain vectors enable the identification of the optimal data mixture for language model (LM) pretraining in a training-free manner under the \emph{\textbf{D}istribution \textbf{A}lignment \textbf{A}ssumption} (DA$^{2}$), which suggests that when the data distributions of the training set and the validation set are better aligned, a lower validation loss is achieved. Moreover, \textsc{Domain2vec} can be seamlessly integrated into previous works to model the relationship between domain vectors and LM performance, greatly enhancing the efficiency and scalability of previous methods. Extensive experiments demonstrate that \textsc{Domain2Vec} helps find the data mixture that enhances downstream task performance with minimal computational overhead. Specifically, \textsc{Domain2Vec} achieves the same validation loss on Pile-CC using only $51.5\%$ of the computation required when training on the original mixture of The Pile dataset. Under equivalent compute budget, \textsc{Domain2Vec} improves downstream performance by an average of $2.83\%$. 

---
# Dynamic Epistemic Friction in Dialogue 

**Authors**: Timothy Obiso, Kenneth Lai, Abhijnan Nath, Nikhil Krishnaswamy, James Pustejovsky  

**Link**: [PDF](https://arxiv.org/pdf/2506.10934)  

**Abstract**: Recent developments in aligning Large Language Models (LLMs) with human preferences have significantly enhanced their utility in human-AI collaborative scenarios. However, such approaches often neglect the critical role of "epistemic friction," or the inherent resistance encountered when updating beliefs in response to new, conflicting, or ambiguous information. In this paper, we define dynamic epistemic friction as the resistance to epistemic integration, characterized by the misalignment between an agent's current belief state and new propositions supported by external evidence. We position this within the framework of Dynamic Epistemic Logic (Van Benthem and Pacuit, 2011), where friction emerges as nontrivial belief-revision during the interaction. We then present analyses from a situated collaborative task that demonstrate how this model of epistemic friction can effectively predict belief updates in dialogues, and we subsequently discuss how the model of belief alignment as a measure of epistemic resistance or friction can naturally be made more sophisticated to accommodate the complexities of real-world dialogue scenarios. 

---
# Decomposing MLP Activations into Interpretable Features via Semi-Nonnegative Matrix Factorization 

**Authors**: Or Shafran, Atticus Geiger, Mor Geva  

**Link**: [PDF](https://arxiv.org/pdf/2506.10920)  

**Abstract**: A central goal for mechanistic interpretability has been to identify the right units of analysis in large language models (LLMs) that causally explain their outputs. While early work focused on individual neurons, evidence that neurons often encode multiple concepts has motivated a shift toward analyzing directions in activation space. A key question is how to find directions that capture interpretable features in an unsupervised manner. Current methods rely on dictionary learning with sparse autoencoders (SAEs), commonly trained over residual stream activations to learn directions from scratch. However, SAEs often struggle in causal evaluations and lack intrinsic interpretability, as their learning is not explicitly tied to the computations of the model. Here, we tackle these limitations by directly decomposing MLP activations with semi-nonnegative matrix factorization (SNMF), such that the learned features are (a) sparse linear combinations of co-activated neurons, and (b) mapped to their activating inputs, making them directly interpretable. Experiments on Llama 3.1, Gemma 2 and GPT-2 show that SNMF derived features outperform SAEs and a strong supervised baseline (difference-in-means) on causal steering, while aligning with human-interpretable concepts. Further analysis reveals that specific neuron combinations are reused across semantically-related features, exposing a hierarchical structure in the MLP's activation space. Together, these results position SNMF as a simple and effective tool for identifying interpretable features and dissecting concept representations in LLMs. 

---
# Magistral 

**Authors**: Mistral-AI, Abhinav Rastogi, Albert Q. Jiang, Andy Lo, Gabrielle Berrada, Guillaume Lample, Jason Rute, Joep Barmentlo, Karmesh Yadav, Kartik Khandelwal, Khyathi Raghavi Chandu, Léonard Blier, Lucile Saulnier, Matthieu Dinot, Maxime Darrin, Neha Gupta, Roman Soletskyi, Sagar Vaze, Teven Le Scao, Yihan Wang, Adam Yang, Alexander H. Liu, Alexandre Sablayrolles, Amélie Héliou, Amélie Martin, Andy Ehrenberg, Anmol Agarwal, Antoine Roux, Arthur Darcet, Arthur Mensch, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault, Chris Bamford, Christian Wallenwein, Christophe Renaudin, Clémence Lanfranchi, Darius Dabert, Devon Mizelle, Diego de las Casas, Elliot Chane-Sane, Emilien Fugier, Emma Bou Hanna, Gauthier Delerce, Gauthier Guinet, Georgii Novikov, Guillaume Martin, Himanshu Jaju, Jan Ludziejewski, Jean-Hadrien Chabran, Jean-Malo Delignon, Joachim Studnia, Jonas Amar, Josselin Somerville Roberts, Julien Denize, Karan Saxena, Kush Jain, Lingxiao Zhao, Louis Martin, Luyu Gao, Lélio Renard Lavaud, Marie Pellat, Mathilde Guillaumin, Mathis Felardos, Maximilian Augustin, Mickaël Seznec, Nikhil Raghuraman, Olivier Duchenne, Patricia Wang, Patrick von Platen, Patryk Saffer, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Pavankumar Reddy Muddireddy, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Romain Sauvestre, Rémi Delacourt, Sanchit Gandhi, Sandeep Subramanian, Shashwat Dalal, Siddharth Gandhi, Soham Ghosh, Srijan Mishra, Sumukh Aithal, Szymon Antoniak, Thibault Schueller, Thibaut Lavril, Thomas Robert, Thomas Wang, Timothée Lacroix, Valeriia Nemychnikova, Victor Paltz, Virgile Richard, Wen-Ding Li, William Marshall, Xuanyu Zhang, Yunhao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10910)  

**Abstract**: We introduce Magistral, Mistral's first reasoning model and our own scalable reinforcement learning (RL) pipeline. Instead of relying on existing implementations and RL traces distilled from prior models, we follow a ground up approach, relying solely on our own models and infrastructure. Notably, we demonstrate a stack that enabled us to explore the limits of pure RL training of LLMs, present a simple method to force the reasoning language of the model, and show that RL on text data alone maintains most of the initial checkpoint's capabilities. We find that RL on text maintains or improves multimodal understanding, instruction following and function calling. We present Magistral Medium, trained for reasoning on top of Mistral Medium 3 with RL alone, and we open-source Magistral Small (Apache 2.0) which further includes cold-start data from Magistral Medium. 

---
# Beyond Gold Standards: Epistemic Ensemble of LLM Judges for Formal Mathematical Reasoning 

**Authors**: Lan Zhang, Marco Valentino, Andre Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2506.10903)  

**Abstract**: Autoformalization plays a crucial role in formal mathematical reasoning by enabling the automatic translation of natural language statements into formal languages. While recent advances using large language models (LLMs) have shown promising results, methods for automatically evaluating autoformalization remain underexplored. As one moves to more complex domains (e.g., advanced mathematics), human evaluation requires significant time and domain expertise, especially as the complexity of the underlying statements and background knowledge increases. LLM-as-a-judge presents a promising approach for automating such evaluation. However, existing methods typically employ coarse-grained and generic evaluation criteria, which limit their effectiveness for advanced formal mathematical reasoning, where quality hinges on nuanced, multi-granular dimensions. In this work, we take a step toward addressing this gap by introducing a systematic, automatic method to evaluate autoformalization tasks. The proposed method is based on an epistemically and formally grounded ensemble (EFG) of LLM judges, defined on criteria encompassing logical preservation (LP), mathematical consistency (MC), formal validity (FV), and formal quality (FQ), resulting in a transparent assessment that accounts for different contributing factors. We validate the proposed framework to serve as a proxy for autoformalization assessment within the domain of formal mathematics. Overall, our experiments demonstrate that the EFG ensemble of LLM judges is a suitable emerging proxy for evaluation, more strongly correlating with human assessments than a coarse-grained model, especially when assessing formal qualities. These findings suggest that LLM-as-judges, especially when guided by a well-defined set of atomic properties, could offer a scalable, interpretable, and reliable support for evaluating formal mathematical reasoning. 

---
# BioClinical ModernBERT: A State-of-the-Art Long-Context Encoder for Biomedical and Clinical NLP 

**Authors**: Thomas Sounack, Joshua Davis, Brigitte Durieux, Antoine Chaffin, Tom J. Pollard, Eric Lehman, Alistair E. W. Johnson, Matthew McDermott, Tristan Naumann, Charlotta Lindvall  

**Link**: [PDF](https://arxiv.org/pdf/2506.10896)  

**Abstract**: Encoder-based transformer models are central to biomedical and clinical Natural Language Processing (NLP), as their bidirectional self-attention makes them well-suited for efficiently extracting structured information from unstructured text through discriminative tasks. However, encoders have seen slower development compared to decoder models, leading to limited domain adaptation in biomedical and clinical settings. We introduce BioClinical ModernBERT, a domain-adapted encoder that builds on the recent ModernBERT release, incorporating long-context processing and substantial improvements in speed and performance for biomedical and clinical NLP. BioClinical ModernBERT is developed through continued pretraining on the largest biomedical and clinical corpus to date, with over 53.5 billion tokens, and addresses a key limitation of prior clinical encoders by leveraging 20 datasets from diverse institutions, domains, and geographic regions, rather than relying on data from a single source. It outperforms existing biomedical and clinical encoders on four downstream tasks spanning a broad range of use cases. We release both base (150M parameters) and large (396M parameters) versions of BioClinical ModernBERT, along with training checkpoints to support further research. 

---
# Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers 

**Authors**: Yixiao Huang, Hanlin Zhu, Tianyu Guo, Jiantao Jiao, Somayeh Sojoudi, Michael I. Jordan, Stuart Russell, Song Mei  

**Link**: [PDF](https://arxiv.org/pdf/2506.10887)  

**Abstract**: Large language models (LLMs) can acquire new knowledge through fine-tuning, but this process exhibits a puzzling duality: models can generalize remarkably from new facts, yet are also prone to hallucinating incorrect information. However, the reasons for this phenomenon remain poorly understood. In this work, we argue that both behaviors stem from a single mechanism known as out-of-context reasoning (OCR): the ability to deduce implications by associating concepts, even those without a causal link. Our experiments across five prominent LLMs confirm that OCR indeed drives both generalization and hallucination, depending on whether the associated concepts are causally related. To build a rigorous theoretical understanding of this phenomenon, we then formalize OCR as a synthetic factual recall task. We empirically show that a one-layer single-head attention-only transformer with factorized output and value matrices can learn to solve this task, while a model with combined weights cannot, highlighting the crucial role of matrix factorization. Our theoretical analysis shows that the OCR capability can be attributed to the implicit bias of gradient descent, which favors solutions that minimize the nuclear norm of the combined output-value matrix. This mathematical structure explains why the model learns to associate facts and implications with high sample efficiency, regardless of whether the correlation is causal or merely spurious. Ultimately, our work provides a theoretical foundation for understanding the OCR phenomenon, offering a new lens for analyzing and mitigating undesirable behaviors from knowledge injection. 

---
# Slimming Down LLMs Without Losing Their Minds 

**Authors**: Qingda  

**Link**: [PDF](https://arxiv.org/pdf/2506.10885)  

**Abstract**: This paper investigates and validates the impact of fine-tuning on large language model performance, focusing on parameter-efficient methods (LoRA and QLoRA). We evaluate model capabilities across three key domains: (1) commonsense reasoning (HellaSwag), (2) mathematical reasoning (GSM8K), and (3) multi-domain knowledge (MMLU-CS).
Our findings demonstrate that: (1) LoRA-based methods effectively improve task-specific performance while maintaining computational efficiency, and (2) performance strongly depends on alignment between fine-tuning dataset and benchmark tasks. The study provides both theoretical insights into parameter-efficient mechanisms and practical guidance for developers implementing efficient LLM adaptation with limited resources. 

---
# Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment 

**Authors**: Hongda Sun, Jiaren Peng, Wenzhong Yang, Liang He, Bo Du, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.10877)  

**Abstract**: Medical dialogue systems (MDS) have emerged as crucial online platforms for enabling multi-turn, context-aware conversations with patients. However, existing MDS often struggle to (1) identify relevant medical knowledge and (2) generate personalized, medically accurate responses. To address these challenges, we propose MedRef, a novel MDS that incorporates knowledge refining and dynamic prompt adjustment. First, we employ a knowledge refining mechanism to filter out irrelevant medical data, improving predictions of critical medical entities in responses. Additionally, we design a comprehensive prompt structure that incorporates historical details and evident details. To enable real-time adaptability to diverse patient conditions, we implement two key modules, Triplet Filter and Demo Selector, providing appropriate knowledge and demonstrations equipped in the system prompt. Extensive experiments on MedDG and KaMed benchmarks show that MedRef outperforms state-of-the-art baselines in both generation quality and medical entity accuracy, underscoring its effectiveness and reliability for real-world healthcare applications. 

---
# Analyzing the relationships between pretraining language, phonetic, tonal, and speaker information in self-supervised speech models 

**Authors**: Michele Gubian, Ioana Krehan, Oli Liu, James Kirby, Sharon Goldwater  

**Link**: [PDF](https://arxiv.org/pdf/2506.10855)  

**Abstract**: Analyses of self-supervised speech models have begun to reveal where and how they represent different types of information. However, almost all analyses have focused on English. Here, we examine how wav2vec2 models trained on four different languages encode both language-matched and non-matched speech. We use probing classifiers and geometric analyses to examine how phones, lexical tones, and speaker information are represented. We show that for all pretraining and test languages, the subspaces encoding phones, tones, and speakers are largely orthogonal, and that layerwise patterns of probing accuracy are similar, with a relatively small advantage for matched-language phone and tone (but not speaker) probes in the later layers. Our findings suggest that the structure of representations learned by wav2vec2 is largely independent of the speech material used during pretraining. 

---
# Accelerating Diffusion Large Language Models with SlowFast: The Three Golden Principles 

**Authors**: Qingyan Wei, Yaojie Zhang, Zhiyuan Liu, Dongrui Liu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10848)  

**Abstract**: Diffusion-based language models (dLLMs) have emerged as a promising alternative to traditional autoregressive LLMs by enabling parallel token generation and significantly reducing inference latency. However, existing sampling strategies for dLLMs, such as confidence-based or semi-autoregressive decoding, often suffer from static behavior, leading to suboptimal efficiency and limited flexibility. In this paper, we propose SlowFast Sampling, a novel dynamic sampling strategy that adaptively alternates between exploratory and accelerated decoding stages. Our method is guided by three golden principles: certainty principle, convergence principle, and positional principle, which govern when and where tokens can be confidently and efficiently decoded. We further integrate our strategy with dLLM-Cache to reduce redundant computation. Extensive experiments across benchmarks and models show that SlowFast Sampling achieves up to 15.63$\times$ speedup on LLaDA with minimal accuracy drop, and up to 34.22$\times$ when combined with caching. Notably, our approach outperforms strong autoregressive baselines like LLaMA3 8B in throughput, demonstrating that well-designed sampling can unlock the full potential of dLLMs for fast and high-quality generation. 

---
# CIIR@LiveRAG 2025: Optimizing Multi-Agent Retrieval Augmented Generation through Self-Training 

**Authors**: Alireza Salemi, Mukta Maddipatla, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2506.10844)  

**Abstract**: This paper presents mRAG, a multi-agent retrieval-augmented generation (RAG) framework composed of specialized agents for subtasks such as planning, searching, reasoning, and coordination. Our system uses a self-training paradigm with reward-guided trajectory sampling to optimize inter-agent collaboration and enhance response generation. Evaluated on DataMorgana-derived datasets during the SIGIR 2025 LiveRAG competition, mRAG outperforms conventional RAG baselines. We further analyze competition outcomes and showcase the framework's strengths with case studies, demonstrating its efficacy for complex, real-world RAG tasks. 

---
# ReCUT: Balancing Reasoning Length and Accuracy in LLMs via Stepwise Trails and Preference Optimization 

**Authors**: Zhensheng Jin, Xinze Li, Yifan Ji, Chunyi Peng, Zhenghao Liu, Qi Shi, Yukun Yan, Shuo Wang, Furong Peng, Ge Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10822)  

**Abstract**: Recent advances in Chain-of-Thought (CoT) prompting have substantially improved the reasoning capabilities of Large Language Models (LLMs). However, these methods often suffer from overthinking, leading to unnecessarily lengthy or redundant reasoning traces. Existing approaches attempt to mitigate this issue through curating multiple reasoning chains for training LLMs, but their effectiveness is often constrained by the quality of the generated data and prone to overfitting. To address the challenge, we propose Reasoning Compression ThroUgh Stepwise Trials (ReCUT), a novel method aimed at balancing the accuracy and length of reasoning trajectory. Specifically, ReCUT employs a stepwise exploration mechanism and a long-short switched sampling strategy, enabling LLMs to incrementally generate diverse reasoning paths. These paths are evaluated and used to construct preference pairs to train two specialized models (Gemini LLMs)-one optimized for reasoning accuracy, the other for shorter reasoning. A final integrated model is obtained by interpolating the parameters of these two models. Experimental results across multiple math reasoning datasets and backbone models demonstrate that ReCUT significantly reduces reasoning lengths by approximately 30-50%, while maintaining or improving reasoning accuracy compared to various baselines. All codes and data will be released via this https URL. 

---
# Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints 

**Authors**: Wei Sun, Tingyu Qu, Mingxiao Li, Jesse Davis, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2506.10800)  

**Abstract**: Efficiently updating multilingual knowledge in large language models (LLMs), while preserving consistent factual representations across languages, remains a long-standing and unresolved challenge. While deploying separate editing systems for each language might seem viable, this approach incurs substantial costs due to the need to manage multiple models. A more efficient solution involves integrating knowledge updates across all languages into a unified model. However, performing sequential edits across languages often leads to destructive parameter interference, significantly degrading multilingual generalization and the accuracy of injected knowledge. To address this challenge, we propose LangEdit, a novel null-space constrained framework designed to precisely isolate language-specific knowledge updates. The core innovation of LangEdit lies in its ability to project parameter updates for each language onto the orthogonal complement of previous updated subspaces. This approach mathematically guarantees update independence while preserving multilingual generalization capabilities. We conduct a comprehensive evaluation across three model architectures, six languages, and four downstream tasks, demonstrating that LangEdit effectively mitigates parameter interference and outperforms existing state-of-the-art editing methods. Our results highlight its potential for enabling efficient and accurate multilingual knowledge updates in LLMs. The code is available at this https URL. 

---
# Improving Named Entity Transcription with Contextual LLM-based Revision 

**Authors**: Viet Anh Trinh, Xinlu He, Jacob Whitehill  

**Link**: [PDF](https://arxiv.org/pdf/2506.10779)  

**Abstract**: With recent advances in modeling and the increasing amount of supervised training data, automatic speech recognition (ASR) systems have achieved remarkable performance on general speech. However, the word error rate (WER) of state-of-the-art ASR remains high for named entities. Since named entities are often the most critical keywords, misrecognizing them can affect all downstream applications, especially when the ASR system functions as the front end of a complex system. In this paper, we introduce a large language model (LLM) revision mechanism to revise incorrect named entities in ASR predictions by leveraging the LLM's reasoning ability as well as local context (e.g., lecture notes) containing a set of correct named entities. Finally, we introduce the NER-MIT-OpenCourseWare dataset, containing 45 hours of data from MIT courses for development and testing. On this dataset, our proposed technique achieves up to 30\% relative WER reduction for named entities. 

---
# Different Questions, Different Models: Fine-Grained Evaluation of Uncertainty and Calibration in Clinical QA with LLMs 

**Authors**: Alberto Testoni, Iacer Calixto  

**Link**: [PDF](https://arxiv.org/pdf/2506.10769)  

**Abstract**: Accurate and well-calibrated uncertainty estimates are essential for deploying large language models (LLMs) in high-stakes domains such as clinical decision support. We present a fine-grained evaluation of uncertainty estimation methods for clinical multiple-choice question answering, covering ten open-source LLMs (general-purpose, biomedical, and reasoning models) across two datasets, eleven medical specialties, and six question types. We compare standard single-generation and sampling-based methods, and present a case study exploring simple, single-pass estimators based on behavioral signals in reasoning traces. These lightweight methods approach the performance of Semantic Entropy while requiring only one generation. Our results reveal substantial variation across specialties and question types, underscoring the importance of selecting models based on both the nature of the question and model-specific strengths. 

---
# One Tokenizer To Rule Them All: Emergent Language Plasticity via Multilingual Tokenizers 

**Authors**: Diana Abagyan, Alejandro R. Salamanca, Andres Felipe Cruz-Salinas, Kris Cao, Hangyu Lin, Acyr Locatelli, Marzieh Fadaee, Ahmet Üstün, Sara Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2506.10766)  

**Abstract**: Pretraining massively multilingual Large Language Models (LLMs) for many languages at once is challenging due to limited model capacity, scarce high-quality data, and compute constraints. Moreover, the lack of language coverage of the tokenizer makes it harder to address the gap for new languages purely at the post-training stage. In this work, we study what relatively cheap interventions early on in training improve "language plasticity", or adaptation capabilities of the model post-training to new languages. We focus on tokenizer design and propose using a universal tokenizer that is trained for more languages than the primary pretraining languages to enable efficient adaptation in expanding language coverage after pretraining. Our systematic experiments across diverse groups of languages and different training strategies show that a universal tokenizer enables significantly higher language adaptation, with up to 20.2% increase in win rates compared to tokenizers specific to pretraining languages. Furthermore, a universal tokenizer also leads to better plasticity towards languages that are completely unseen in the tokenizer and pretraining, by up to 5% win rate gain. We achieve this adaptation to an expanded set of languages with minimal compromise in performance on the majority of languages included in pretraining. 

---
# TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora 

**Authors**: Priyanka Kargupta, Nan Zhang, Yunyi Zhang, Rui Zhang, Prasenjit Mitra, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.10737)  

**Abstract**: The rapid evolution of scientific fields introduces challenges in organizing and retrieving scientific literature. While expert-curated taxonomies have traditionally addressed this need, the process is time-consuming and expensive. Furthermore, recent automatic taxonomy construction methods either (1) over-rely on a specific corpus, sacrificing generalizability, or (2) depend heavily on the general knowledge of large language models (LLMs) contained within their pre-training datasets, often overlooking the dynamic nature of evolving scientific domains. Additionally, these approaches fail to account for the multi-faceted nature of scientific literature, where a single research paper may contribute to multiple dimensions (e.g., methodology, new tasks, evaluation metrics, benchmarks). To address these gaps, we propose TaxoAdapt, a framework that dynamically adapts an LLM-generated taxonomy to a given corpus across multiple dimensions. TaxoAdapt performs iterative hierarchical classification, expanding both the taxonomy width and depth based on corpus' topical distribution. We demonstrate its state-of-the-art performance across a diverse set of computer science conferences over the years to showcase its ability to structure and capture the evolution of scientific fields. As a multidimensional method, TaxoAdapt generates taxonomies that are 26.51% more granularity-preserving and 50.41% more coherent than the most competitive baselines judged by LLMs. 

---
# Beyond True or False: Retrieval-Augmented Hierarchical Analysis of Nuanced Claims 

**Authors**: Priyanka Kargupta, Runchu Tian, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.10728)  

**Abstract**: Claims made by individuals or entities are oftentimes nuanced and cannot be clearly labeled as entirely "true" or "false" -- as is frequently the case with scientific and political claims. However, a claim (e.g., "vaccine A is better than vaccine B") can be dissected into its integral aspects and sub-aspects (e.g., efficacy, safety, distribution), which are individually easier to validate. This enables a more comprehensive, structured response that provides a well-rounded perspective on a given problem while also allowing the reader to prioritize specific angles of interest within the claim (e.g., safety towards children). Thus, we propose ClaimSpect, a retrieval-augmented generation-based framework for automatically constructing a hierarchy of aspects typically considered when addressing a claim and enriching them with corpus-specific perspectives. This structure hierarchically partitions an input corpus to retrieve relevant segments, which assist in discovering new sub-aspects. Moreover, these segments enable the discovery of varying perspectives towards an aspect of the claim (e.g., support, neutral, or oppose) and their respective prevalence (e.g., "how many biomedical papers believe vaccine A is more transportable than B?"). We apply ClaimSpect to a wide variety of real-world scientific and political claims featured in our constructed dataset, showcasing its robustness and accuracy in deconstructing a nuanced claim and representing perspectives within a corpus. Through real-world case studies and human evaluation, we validate its effectiveness over multiple baselines. 

---
# PREMISE: Scalable and Strategic Prompt Optimization for Efficient Mathematical Reasoning in Large Models 

**Authors**: Ye Yu, Yaoning Yu, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10716)  

**Abstract**: Large reasoning models (LRMs) such as Claude 3.7 Sonnet and OpenAI o1 achieve strong performance on mathematical benchmarks using lengthy chain-of-thought (CoT) reasoning, but the resulting traces are often unnecessarily verbose. This inflates token usage and cost, limiting deployment in latency-sensitive or API-constrained settings. We introduce PREMISE (PRompt-based Efficient Mathematical Inference with Strategic Evaluation), a prompt-only framework that reduces reasoning overhead without modifying model weights. PREMISE combines trace-level diagnostics with gradient-inspired prompt optimization to minimize redundant computation while preserving answer accuracy. The approach jointly optimizes brevity and correctness through a multi-objective textual search that balances token length and answer validity. Unlike prior work, PREMISE runs in a single-pass black-box interface, so it can be applied directly to commercial LLMs. On GSM8K, SVAMP, and Math500 we match or exceed baseline accuracy ($96\%\rightarrow96\%$ with Claude, $91\%\rightarrow92\%$ with Gemini) while reducing reasoning tokens by up to $87.5\%$ and cutting dollar cost by $69$--$82\%$. These results show that prompt-level optimization is a practical and scalable path to efficient LRM inference without compromising reasoning quality. 

---
# Inferring Adjective Hypernyms with Language Models to Increase the Connectivity of Open English Wordnet 

**Authors**: Lorenzo Augello, John P. McCrae  

**Link**: [PDF](https://arxiv.org/pdf/2506.10715)  

**Abstract**: Open English Wordnet is a key resource published in OntoLex-lemon as part of the linguistic linked open data cloud. There are, however, many links missing in the resource, and in this paper, we look at how we can establish hypernymy between adjectives. We present a theoretical discussion of the hypernymy relation and how it differs for adjectives in contrast to nouns and verbs. We develop a new resource for adjective hypernymy and fine-tune large language models to predict adjective hypernymy, showing that the methodology of TaxoLLaMa can be adapted to this task. 

---
# Large Language Models for Detection of Life-Threatening Texts 

**Authors**: Thanh Thi Nguyen, Campbell Wilson, Janis Dalins  

**Link**: [PDF](https://arxiv.org/pdf/2506.10687)  

**Abstract**: Detecting life-threatening language is essential for safeguarding individuals in distress, promoting mental health and well-being, and preventing potential harm and loss of life. This paper presents an effective approach to identifying life-threatening texts using large language models (LLMs) and compares them with traditional methods such as bag of words, word embedding, topic modeling, and Bidirectional Encoder Representations from Transformers. We fine-tune three open-source LLMs including Gemma, Mistral, and Llama-2 using their 7B parameter variants on different datasets, which are constructed with class balance, imbalance, and extreme imbalance scenarios. Experimental results demonstrate a strong performance of LLMs against traditional methods. More specifically, Mistral and Llama-2 models are top performers in both balanced and imbalanced data scenarios while Gemma is slightly behind. We employ the upsampling technique to deal with the imbalanced data scenarios and demonstrate that while this method benefits traditional approaches, it does not have as much impact on LLMs. This study demonstrates a great potential of LLMs for real-world life-threatening language detection problems. 

---
# Spelling-out is not Straightforward: LLMs' Capability of Tokenization from Token to Characters 

**Authors**: Tatsuya Hiraoka, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2506.10641)  

**Abstract**: Large language models (LLMs) can spell out tokens character by character with high accuracy, yet they struggle with more complex character-level tasks, such as identifying compositional subcomponents within tokens. In this work, we investigate how LLMs internally represent and utilize character-level information during the spelling-out process. Our analysis reveals that, although spelling out is a simple task for humans, it is not handled in a straightforward manner by LLMs. Specifically, we show that the embedding layer does not fully encode character-level information, particularly beyond the first character. As a result, LLMs rely on intermediate and higher Transformer layers to reconstruct character-level knowledge, where we observe a distinct "breakthrough" in their spelling behavior. We validate this mechanism through three complementary analyses: probing classifiers, identification of knowledge neurons, and inspection of attention weights. 

---
# NeuralNexus at BEA 2025 Shared Task: Retrieval-Augmented Prompting for Mistake Identification in AI Tutors 

**Authors**: Numaan Naeem, Sarfraz Ahmad, Momina Ahsan, Hasan Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2506.10627)  

**Abstract**: This paper presents our system for Track 1: Mistake Identification in the BEA 2025 Shared Task on Pedagogical Ability Assessment of AI-powered Tutors. The task involves evaluating whether a tutor's response correctly identifies a mistake in a student's mathematical reasoning. We explore four approaches: (1) an ensemble of machine learning models over pooled token embeddings from multiple pretrained language models (LMs); (2) a frozen sentence-transformer using [CLS] embeddings with an MLP classifier; (3) a history-aware model with multi-head attention between token-level history and response embeddings; and (4) a retrieval-augmented few-shot prompting system with a large language model (LLM) i.e. GPT 4o. Our final system retrieves semantically similar examples, constructs structured prompts, and uses schema-guided output parsing to produce interpretable predictions. It outperforms all baselines, demonstrating the effectiveness of combining example-driven prompting with LLM reasoning for pedagogical feedback assessment. Our code is available at this https URL. 

---
# SDialog: A Python Toolkit for Synthetic Dialogue Generation and Analysis 

**Authors**: Sergio Burdisso, Esaú Villatoro-Tello, Petr Motlicek  

**Link**: [PDF](https://arxiv.org/pdf/2506.10622)  

**Abstract**: The advancement of conversational AI systems relies on the availability of high-quality, flexible, and reproducible synthetic dialogues for training, evaluation, and benchmarking. SDialog is a modular, extensible Python toolkit designed to address the challenges of synthetic dialogue generation and analysis. By leveraging instruction-tuned Large Language Models (LLMs), SDialog provides abstractions for personas, orchestration, and scenario management, enabling the creation of realistic, diverse, and controllable conversational data for research and development. SDialog supports workflows such as multi-agent simulation and scenario-driven generation, and represents a step forward in the standardization of tools and frameworks for synthetic data generation, a crucial advancement for ensuring reproducibility in today's fast-evolving research landscape. 

---
# Unsupervised Protoform Reconstruction through Parsimonious Rule-guided Heuristics and Evolutionary Search 

**Authors**: Promise Dodzi Kpoglu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10614)  

**Abstract**: We propose an unsupervised method for the reconstruction of protoforms i.e., ancestral word forms from which modern language forms are derived. While prior work has primarily relied on probabilistic models of phonological edits to infer protoforms from cognate sets, such approaches are limited by their predominantly data-driven nature. In contrast, our model integrates data-driven inference with rule-based heuristics within an evolutionary optimization framework. This hybrid approach leverages on both statistical patterns and linguistically motivated constraints to guide the reconstruction process. We evaluate our method on the task of reconstructing Latin protoforms using a dataset of cognates from five Romance languages. Experimental results demonstrate substantial improvements over established baselines across both character-level accuracy and phonological plausibility metrics. 

---
# Reliable Reasoning Path: Distilling Effective Guidance for LLM Reasoning with Knowledge Graphs 

**Authors**: Yilin Xiao, Chuang Zhou, Qinggang Zhang, Bo Li, Qing Li, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10508)  

**Abstract**: Large language models (LLMs) often struggle with knowledge-intensive tasks due to a lack of background knowledge and a tendency to hallucinate. To address these limitations, integrating knowledge graphs (KGs) with LLMs has been intensively studied. Existing KG-enhanced LLMs focus on supplementary factual knowledge, but still struggle with solving complex questions. We argue that refining the relationships among facts and organizing them into a logically consistent reasoning path is equally important as factual knowledge itself. Despite their potential, extracting reliable reasoning paths from KGs poses the following challenges: the complexity of graph structures and the existence of multiple generated paths, making it difficult to distinguish between useful and redundant ones. To tackle these challenges, we propose the RRP framework to mine the knowledge graph, which combines the semantic strengths of LLMs with structural information obtained through relation embedding and bidirectional distribution learning. Additionally, we introduce a rethinking module that evaluates and refines reasoning paths according to their significance. Experimental results on two public datasets show that RRP achieves state-of-the-art performance compared to existing baseline methods. Moreover, RRP can be easily integrated into various LLMs to enhance their reasoning abilities in a plug-and-play manner. By generating high-quality reasoning paths tailored to specific questions, RRP distills effective guidance for LLM reasoning. 

---
# Beyond Single-User Dialogue: Assessing Multi-User Dialogue State Tracking Capabilities of Large Language Models 

**Authors**: Sangmin Song, Juhwan Choi, JungMin Yun, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10504)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance in zero-shot dialogue state tracking (DST), reducing the need for task-specific training. However, conventional DST benchmarks primarily focus on structured user-agent conversations, failing to capture the complexities of real-world multi-user interactions. In this study, we assess the robustness of LLMs in multi-user DST while minimizing dataset construction costs. Inspired by recent advances in LLM-based data annotation, we extend an existing DST dataset by generating utterances of a second user based on speech act theory. Our methodology systematically incorporates a second user's utterances into conversations, enabling a controlled evaluation of LLMs in multi-user settings. Experimental results reveal a significant performance drop compared to single-user DST, highlighting the limitations of current LLMs in extracting and tracking dialogue states amidst multiple speakers. Our findings emphasize the need for future research to enhance LLMs for multi-user DST scenarios, paving the way for more realistic and robust DST models. 

---
# Surface Fairness, Deep Bias: A Comparative Study of Bias in Language Models 

**Authors**: Aleksandra Sorokovikova, Pavel Chizhov, Iuliia Eremenko, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2506.10491)  

**Abstract**: Modern language models are trained on large amounts of data. These data inevitably include controversial and stereotypical content, which contains all sorts of biases related to gender, origin, age, etc. As a result, the models express biased points of view or produce different results based on the assigned personality or the personality of the user. In this paper, we investigate various proxy measures of bias in large language models (LLMs). We find that evaluating models with pre-prompted personae on a multi-subject benchmark (MMLU) leads to negligible and mostly random differences in scores. However, if we reformulate the task and ask a model to grade the user's answer, this shows more significant signs of bias. Finally, if we ask the model for salary negotiation advice, we see pronounced bias in the answers. With the recent trend for LLM assistant memory and personalization, these problems open up from a different angle: modern LLM users do not need to pre-prompt the description of their persona since the model already knows their socio-demographics. 

---
# Table-Text Alignment: Explaining Claim Verification Against Tables in Scientific Papers 

**Authors**: Xanh Ho, Sunisth Kumar, Yun-Ang Wu, Florian Boudin, Atsuhiro Takasu, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.10486)  

**Abstract**: Scientific claim verification against tables typically requires predicting whether a claim is supported or refuted given a table. However, we argue that predicting the final label alone is insufficient: it reveals little about the model's reasoning and offers limited interpretability. To address this, we reframe table-text alignment as an explanation task, requiring models to identify the table cells essential for claim verification. We build a new dataset by extending the SciTab benchmark with human-annotated cell-level rationales. Annotators verify the claim label and highlight the minimal set of cells needed to support their decision. After the annotation process, we utilize the collected information and propose a taxonomy for handling ambiguous cases. Our experiments show that (i) incorporating table alignment information improves claim verification performance, and (ii) most LLMs, while often predicting correct labels, fail to recover human-aligned rationales, suggesting that their predictions do not stem from faithful reasoning. 

---
# Fast on the Easy, Deep on the Hard: Efficient Reasoning via Powered Length Penalty 

**Authors**: Zehui Ling, Deshu Chen, Hongwei Zhang, Yifeng Jiao, Xin Guo, Yuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10446)  

**Abstract**: Large language models (LLMs) have demonstrated significant advancements in reasoning capabilities, performing well on various challenging benchmarks. Techniques like Chain-of-Thought prompting have been introduced to further improve reasoning. However, these approaches frequently generate longer outputs, which in turn increase computational latency. Although some methods use reinforcement learning to shorten reasoning, they often apply uniform penalties without considering the problem's complexity, leading to suboptimal outcomes. In this study, we seek to enhance the efficiency of LLM reasoning by promoting conciseness for simpler problems while preserving sufficient reasoning for more complex ones for accuracy, thus improving the model's overall performance. Specifically, we manage the model's reasoning efficiency by dividing the reward function and including a novel penalty for output length. Our approach has yielded impressive outcomes in benchmark evaluations across three datasets: GSM8K, MATH500, and AIME2024. For the comparatively simpler datasets GSM8K and MATH500, our method has effectively shortened output lengths while preserving or enhancing accuracy. On the more demanding AIME2024 dataset, our approach has resulted in improved accuracy. 

---
# Beyond the Battlefield: Framing Analysis of Media Coverage in Conflict Reporting 

**Authors**: Avneet Kaur, Arnav Arora  

**Link**: [PDF](https://arxiv.org/pdf/2506.10421)  

**Abstract**: Framing used by news media, especially in times of conflict, can have substantial impact on readers' opinion, potentially aggravating the conflict itself. Current studies on the topic of conflict framing have limited insights due to their qualitative nature or only look at surface level generic frames without going deeper. In this work, we identify indicators of war and peace journalism, as outlined by prior work in conflict studies, in a corpus of news articles reporting on the Israel-Palestine war. For our analysis, we use computational approaches, using a combination of frame semantics and large language models to identify both communicative framing and its connection to linguistic framing. Our analysis reveals a higher focus on war based reporting rather than peace based. We also show substantial differences in reporting across the US, UK, and Middle Eastern news outlets in framing who the assailant and victims of the conflict are, surfacing biases within the media. 

---
# Burn After Reading: Do Multimodal Large Language Models Truly Capture Order of Events in Image Sequences? 

**Authors**: Yingjin Song, Yupei Du, Denis Paperno, Albert Gatt  

**Link**: [PDF](https://arxiv.org/pdf/2506.10415)  

**Abstract**: This paper introduces the TempVS benchmark, which focuses on temporal grounding and reasoning capabilities of Multimodal Large Language Models (MLLMs) in image sequences. TempVS consists of three main tests (i.e., event relation inference, sentence ordering and image ordering), each accompanied with a basic grounding test. TempVS requires MLLMs to rely on both visual and linguistic modalities to understand the temporal order of events. We evaluate 38 state-of-the-art MLLMs, demonstrating that models struggle to solve TempVS, with a substantial performance gap compared to human capabilities. We also provide fine-grained insights that suggest promising directions for future research. Our TempVS benchmark data and code are available at this https URL. 

---
# PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier 

**Authors**: Yuhua Jiang, Yuwen Xiong, Yufeng Yuan, Chao Xin, Wenyuan Xu, Yu Yue, Qianchuan Zhao, Lin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.10406)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in complex reasoning tasks, yet they still struggle to reliably verify the correctness of their own outputs. Existing solutions to this verification challenge often depend on separate verifier models or require multi-stage self-correction training pipelines, which limit scalability. In this paper, we propose Policy as Generative Verifier (PAG), a simple and effective framework that empowers LLMs to self-correct by alternating between policy and verifier roles within a unified multi-turn reinforcement learning (RL) paradigm. Distinct from prior approaches that always generate a second attempt regardless of model confidence, PAG introduces a selective revision mechanism: the model revises its answer only when its own generative verification step detects an error. This verify-then-revise workflow not only alleviates model collapse but also jointly enhances both reasoning and verification abilities. Extensive experiments across diverse reasoning benchmarks highlight PAG's dual advancements: as a policy, it enhances direct generation and self-correction accuracy; as a verifier, its self-verification outperforms self-consistency. 

---
# TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning 

**Authors**: Xiaohan Yu, Pu Jian, Chong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10380)  

**Abstract**: Retrieval-Augmented Generation (RAG) has demonstrated considerable effectiveness in open-domain question answering. However, when applied to heterogeneous documents, comprising both textual and tabular components, existing RAG approaches exhibit critical limitations. The prevailing practice of flattening tables and chunking strategies disrupts the intrinsic tabular structure, leads to information loss, and undermines the reasoning capabilities of LLMs in multi-hop, global queries. To address these challenges, we propose TableRAG, an hybrid framework that unifies textual understanding and complex manipulations over tabular data. TableRAG iteratively operates in four steps: context-sensitive query decomposition, text retrieval, SQL programming and execution, and compositional intermediate answer generation. We also develop HeteQA, a novel benchmark designed to evaluate the multi-hop heterogeneous reasoning capabilities. Experimental results demonstrate that TableRAG consistently outperforms existing baselines on both public datasets and our HeteQA, establishing a new state-of-the-art for heterogeneous document question answering. We release TableRAG at this https URL. 

---
# Code Execution as Grounded Supervision for LLM Reasoning 

**Authors**: Dongwon Jung, Wenxuan Zhou, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10343)  

**Abstract**: Training large language models (LLMs) with chain-of-thought (CoT) supervision has proven effective for enhancing their reasoning abilities. However, obtaining reliable and accurate reasoning supervision remains a significant challenge. We propose a scalable method for generating a high-quality CoT supervision dataset by leveraging the determinism of program execution. Unlike existing reasoning dataset generation methods that rely on costly human annotations or error-prone LLM-generated CoT, our approach extracts verifiable, step-by-step reasoning traces from code execution and transforms them into a natural language CoT reasoning. Experiments on reasoning benchmarks across various domains show that our method effectively equips LLMs with transferable reasoning abilities across diverse tasks. Furthermore, the ablation studies validate that our method produces highly accurate reasoning data and reduces overall token length during inference by reducing meaningless repetition and overthinking. 

---
# Scheduled Interleaved Speech-Text Training for Speech-to-Speech Translation with LLMs 

**Authors**: Hayato Futami, Emiru Tsunoo, Yosuke Kashiwagi, Yuki Ito, Hassan Shahmohammadi, Siddhant Arora, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.10299)  

**Abstract**: Speech-to-speech translation (S2ST) has been advanced with large language models (LLMs), which are fine-tuned on discrete speech units. In such approaches, modality adaptation from text to speech has been an issue. LLMs are trained on text-only data, which presents challenges to adapt them to speech modality with limited speech-to-speech data. To address the training difficulty, we propose scheduled interleaved speech--text training in this study. We use interleaved speech--text units instead of speech units during training, where aligned text tokens are interleaved at the word level. We gradually decrease the ratio of text as training progresses, to facilitate progressive modality adaptation from text to speech. We conduct experimental evaluations by fine-tuning LLaMA3.2-1B for S2ST on the CVSS dataset. We show that the proposed method consistently improves the translation performances, especially for languages with limited training data. 

---
# "Check My Work?": Measuring Sycophancy in a Simulated Educational Context 

**Authors**: Chuck Arvin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10297)  

**Abstract**: This study examines how user-provided suggestions affect Large Language Models (LLMs) in a simulated educational context, where sycophancy poses significant risks. Testing five different LLMs from the OpenAI GPT-4o and GPT-4.1 model classes across five experimental conditions, we show that response quality varies dramatically based on query framing. In cases where the student mentions an incorrect answer, the LLM correctness can degrade by as much as 15 percentage points, while mentioning the correct answer boosts accuracy by the same margin. Our results also show that this bias is stronger in smaller models, with an effect of up to 30% for the GPT-4.1-nano model, versus 8% for the GPT-4o model. Our analysis of how often LLMs "flip" their answer, and an investigation into token level probabilities, confirm that the models are generally changing their answers to answer choices mentioned by students in line with the sycophancy hypothesis. This sycophantic behavior has important implications for educational equity, as LLMs may accelerate learning for knowledgeable students while the same tools may reinforce misunderstanding for less knowledgeable students. Our results highlight the need to better understand the mechanism, and ways to mitigate, such bias in the educational context. 

---
# Flick: Few Labels Text Classification using K-Aware Intermediate Learning in Multi-Task Low-Resource Languages 

**Authors**: Ali Almutairi, Abdullah Alsuhaibani, Shoaib Jameel, Usman Naseem, Gelareh Mohammadi, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2506.10292)  

**Abstract**: Training deep learning networks with minimal supervision has gained significant research attention due to its potential to reduce reliance on extensive labelled data. While self-training methods have proven effective in semi-supervised learning, they remain vulnerable to errors from noisy pseudo labels. Moreover, most recent approaches to the few-label classification problem are either designed for resource-rich languages such as English or involve complex cascading models that are prone to overfitting. To address the persistent challenge of few-label text classification in truly low-resource linguistic contexts, where existing methods often struggle with noisy pseudo-labels and domain adaptation, we propose Flick. Unlike prior methods that rely on generic multi-cluster pseudo-labelling or complex cascading architectures, Flick leverages the fundamental insight that distilling high-confidence pseudo-labels from a broader set of initial clusters can dramatically improve pseudo-label quality, particularly for linguistically diverse, low-resource settings. Flick introduces a novel pseudo-label refinement component, a departure from traditional pseudo-labelling strategies by identifying and leveraging top-performing pseudo-label clusters. This component specifically learns to distil highly reliable pseudo-labels from an initial broad set by focusing on single-cluster cohesion and leveraging an adaptive top-k selection mechanism. This targeted refinement process is crucial for mitigating the propagation of errors inherent in low-resource data, allowing for robust fine-tuning of pre-trained language models with only a handful of true labels. We demonstrate Flick's efficacy across 14 diverse datasets, encompassing challenging low-resource languages such as Arabic, Urdu, and Setswana, alongside English, showcasing its superior performance and adaptability. 

---
# ClusterUCB: Efficient Gradient-Based Data Selection for Targeted Fine-Tuning of LLMs 

**Authors**: Zige Wang, Qi Zhu, Fei Mi, Minghui Xu, Ruochun Jin, Wenjing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10288)  

**Abstract**: Gradient-based data influence approximation has been leveraged to select useful data samples in the supervised fine-tuning of large language models. However, the computation of gradients throughout the fine-tuning process requires too many resources to be feasible in practice. In this paper, we propose an efficient gradient-based data selection framework with clustering and a modified Upper Confidence Bound (UCB) algorithm. Based on the intuition that data samples with similar gradient features will have similar influences, we first perform clustering on the training data pool. Then, we frame the inter-cluster data selection as a constrained computing budget allocation problem and consider it a multi-armed bandit problem. A modified UCB algorithm is leveraged to solve this problem. Specifically, during the iterative sampling process, historical data influence information is recorded to directly estimate the distributions of each cluster, and a cold start is adopted to balance exploration and exploitation. Experimental results on various benchmarks show that our proposed framework, ClusterUCB, can achieve comparable results to the original gradient-based data selection methods while greatly reducing computing consumption. 

---
# Do Language Models Have Bayesian Brains? Distinguishing Stochastic and Deterministic Decision Patterns within Large Language Models 

**Authors**: Andrea Yaoyun Cui, Pengfei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10268)  

**Abstract**: Language models are essentially probability distributions over token sequences. Auto-regressive models generate sentences by iteratively computing and sampling from the distribution of the next token. This iterative sampling introduces stochasticity, leading to the assumption that language models make probabilistic decisions, similar to sampling from unknown distributions. Building on this assumption, prior research has used simulated Gibbs sampling, inspired by experiments designed to elicit human priors, to infer the priors of language models. In this paper, we revisit a critical question: Do language models possess Bayesian brains? Our findings show that under certain conditions, language models can exhibit near-deterministic decision-making, such as producing maximum likelihood estimations, even with a non-zero sampling temperature. This challenges the sampling assumption and undermines previous methods for eliciting human-like priors. Furthermore, we demonstrate that without proper scrutiny, a system with deterministic behavior undergoing simulated Gibbs sampling can converge to a "false prior." To address this, we propose a straightforward approach to distinguish between stochastic and deterministic decision patterns in Gibbs sampling, helping to prevent the inference of misleading language model priors. We experiment on a variety of large language models to identify their decision patterns under various circumstances. Our results provide key insights in understanding decision making of large language models. 

---
# ToxSyn-PT: A Large-Scale Synthetic Dataset for Hate Speech Detection in Portuguese 

**Authors**: Iago Alves Brito, Julia Soares Dollis, Fernanda Bufon Färber, Diogo Fernandes Costa Silva, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2506.10245)  

**Abstract**: We present ToxSyn-PT, the first large-scale Portuguese corpus that enables fine-grained hate-speech classification across nine legally protected minority groups. The dataset contains 53,274 synthetic sentences equally distributed between minorities groups and toxicity labels. ToxSyn-PT is created through a novel four-stage pipeline: (1) a compact, manually curated seed; (2) few-shot expansion with an instruction-tuned LLM; (3) paraphrase-based augmentation; and (4) enrichment, plus additional neutral texts to curb overfitting to group-specific cues. The resulting corpus is class-balanced, stylistically diverse, and free from the social-media domain that dominate existing Portuguese datasets. Despite domain differences with traditional benchmarks, experiments on both binary and multi-label classification on the corpus yields strong results across five public Portuguese hate-speech datasets, demonstrating robust generalization even across domain boundaries. The dataset is publicly released to advance research on synthetic data and hate-speech detection in low-resource settings. 

---
# Classifying Unreliable Narrators with Large Language Models 

**Authors**: Anneliese Brei, Katharine Henry, Abhisheik Sharma, Shashank Srivastava, Snigdha Chaturvedi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10231)  

**Abstract**: Often when we interact with a first-person account of events, we consider whether or not the narrator, the primary speaker of the text, is reliable. In this paper, we propose using computational methods to identify unreliable narrators, i.e. those who unintentionally misrepresent information. Borrowing literary theory from narratology to define different types of unreliable narrators based on a variety of textual phenomena, we present TUNa, a human-annotated dataset of narratives from multiple domains, including blog posts, subreddit posts, hotel reviews, and works of literature. We define classification tasks for intra-narrational, inter-narrational, and inter-textual unreliabilities and analyze the performance of popular open-weight and proprietary LLMs for each. We propose learning from literature to perform unreliable narrator classification on real-world text data. To this end, we experiment with few-shot, fine-tuning, and curriculum learning settings. Our results show that this task is very challenging, and there is potential for using LLMs to identify unreliable narrators. We release our expert-annotated dataset and code and invite future research in this area. 

---
# TTT-Bench: A Benchmark for Evaluating Reasoning Ability with Simple and Novel Tic-Tac-Toe-style Games 

**Authors**: Prakamya Mishra, Jiang Liu, Jialian Wu, Xiaodong Yu, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2506.10209)  

**Abstract**: Large reasoning models (LRMs) have demonstrated impressive reasoning capabilities across a broad range of tasks including Olympiad-level mathematical problems, indicating evidence of their complex reasoning abilities. While many reasoning benchmarks focus on the STEM domain, the ability of LRMs to reason correctly in broader task domains remains underexplored. In this work, we introduce \textbf{TTT-Bench}, a new benchmark that is designed to evaluate basic strategic, spatial, and logical reasoning abilities in LRMs through a suite of four two-player Tic-Tac-Toe-style games that humans can effortlessly solve from a young age. We propose a simple yet scalable programmatic approach for generating verifiable two-player game problems for TTT-Bench. Although these games are trivial for humans, they require reasoning about the intentions of the opponent, as well as the game board's spatial configurations, to ensure a win. We evaluate a diverse set of state-of-the-art LRMs, and \textbf{discover that the models that excel at hard math problems frequently fail at these simple reasoning games}. Further testing reveals that our evaluated reasoning models score on average $\downarrow$ 41\% \& $\downarrow$ 5\% lower on TTT-Bench compared to MATH 500 \& AIME 2024 respectively, with larger models achieving higher performance using shorter reasoning traces, where most of the models struggle on long-term strategic reasoning situations on simple and new TTT-Bench tasks. 

---
# Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval 

**Authors**: Shubhashis Roy Dipta, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2506.10202)  

**Abstract**: Recent approaches have shown impressive proficiency in extracting and leveraging parametric knowledge from Large-Language Models (LLMs) and Vision-Language Models (VLMs). In this work, we consider how we can improve the identification and retrieval of videos related to complex real-world events by automatically extracting latent parametric knowledge about those events. We present Q2E: a Query-to-Event decomposition method for zero-shot multilingual text-to-video retrieval, adaptable across datasets, domains, LLMs, or VLMs. Our approach demonstrates that we can enhance the understanding of otherwise overly simplified human queries by decomposing the query using the knowledge embedded in LLMs and VLMs. We additionally show how to apply our approach to both visual and speech-based inputs. To combine this varied multimodal knowledge, we adopt entropy-based fusion scoring for zero-shot fusion. Through evaluations on two diverse datasets and multiple retrieval metrics, we demonstrate that Q2E outperforms several state-of-the-art baselines. Our evaluation also shows that integrating audio information can significantly improve text-to-video retrieval. We have released code and data for future research. 

---
# Can LLMs Generate Good Stories? Insights and Challenges from a Narrative Planning Perspective 

**Authors**: Yi Wang, Max Kreminski  

**Link**: [PDF](https://arxiv.org/pdf/2506.10161)  

**Abstract**: Story generation has been a prominent application of Large Language Models (LLMs). However, understanding LLMs' ability to produce high-quality stories remains limited due to challenges in automatic evaluation methods and the high cost and subjectivity of manual evaluation. Computational narratology offers valuable insights into what constitutes a good story, which has been applied in the symbolic narrative planning approach to story generation. This work aims to deepen the understanding of LLMs' story generation capabilities by using them to solve narrative planning problems. We present a benchmark for evaluating LLMs on narrative planning based on literature examples, focusing on causal soundness, character intentionality, and dramatic conflict. Our experiments show that GPT-4 tier LLMs can generate causally sound stories at small scales, but planning with character intentionality and dramatic conflict remains challenging, requiring LLMs trained with reinforcement learning for complex reasoning. The results offer insights on the scale of stories that LLMs can generate while maintaining quality from different aspects. Our findings also highlight interesting problem solving behaviors and shed lights on challenges and considerations for applying LLM narrative planning in game environments. 

---
# Measuring Corporate Human Capital Disclosures: Lexicon, Data, Code, and Research Opportunities 

**Authors**: Elizabeth Demers, Victor Xiaoqi Wang, Kean Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10155)  

**Abstract**: Human capital (HC) is increasingly important to corporate value creation. Unlike other assets, however, HC is not currently subject to well-defined measurement or disclosure rules. We use a machine learning algorithm (word2vec) trained on a confirmed set of HC disclosures to develop a comprehensive list of HC-related keywords classified into five subcategories (DEI; health and safety; labor relations and culture; compensation and benefits; and demographics and other) that capture the multidimensional nature of HC management. We share our lexicon, corporate HC disclosures, and the Python code used to develop the lexicon, and we provide detailed examples of using our data and code, including for fine-tuning a BERT model. Researchers can use our HC lexicon (or modify the code to capture another construct of interest) with their samples of corporate communications to address pertinent HC questions. We close with a discussion of future research opportunities related to HC management and disclosure. 

---
# Analyzing Emotions in Bangla Social Media Comments Using Machine Learning and LIME 

**Authors**: Bidyarthi Paul, SM Musfiqur Rahman, Dipta Biswas, Md. Ziaul Hasan, Md. Zahid Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2506.10154)  

**Abstract**: Research on understanding emotions in written language continues to expand, especially for understudied languages with distinctive regional expressions and cultural features, such as Bangla. This study examines emotion analysis using 22,698 social media comments from the EmoNoBa dataset. For language analysis, we employ machine learning models: Linear SVM, KNN, and Random Forest with n-gram data from a TF-IDF vectorizer. We additionally investigated how PCA affects the reduction of dimensionality. Moreover, we utilized a BiLSTM model and AdaBoost to improve decision trees. To make our machine learning models easier to understand, we used LIME to explain the predictions of the AdaBoost classifier, which uses decision trees. With the goal of advancing sentiment analysis in languages with limited resources, our work examines various techniques to find efficient techniques for emotion identification in Bangla. 

---
# When Large Language Models are Reliable for Judging Empathic Communication 

**Authors**: Aakriti Kumar, Nalin Poungpeth, Diyi Yang, Erina Farrell, Bruce Lambert, Matthew Groh  

**Link**: [PDF](https://arxiv.org/pdf/2506.10150)  

**Abstract**: Large language models (LLMs) excel at generating empathic responses in text-based conversations. But, how reliably do they judge the nuances of empathic communication? We investigate this question by comparing how experts, crowdworkers, and LLMs annotate empathic communication across four evaluative frameworks drawn from psychology, natural language processing, and communications applied to 200 real-world conversations where one speaker shares a personal problem and the other offers support. Drawing on 3,150 expert annotations, 2,844 crowd annotations, and 3,150 LLM annotations, we assess inter-rater reliability between these three annotator groups. We find that expert agreement is high but varies across the frameworks' sub-components depending on their clarity, complexity, and subjectivity. We show that expert agreement offers a more informative benchmark for contextualizing LLM performance than standard classification metrics. Across all four frameworks, LLMs consistently approach this expert level benchmark and exceed the reliability of crowdworkers. These results demonstrate how LLMs, when validated on specific tasks with appropriate benchmarks, can support transparency and oversight in emotionally sensitive applications including their use as conversational companions. 

---
# Unsupervised Elicitation of Language Models 

**Authors**: Jiaxin Wen, Zachary Ankner, Arushi Somani, Peter Hase, Samuel Marks, Jacob Goldman-Wetzler, Linda Petrini, Henry Sleight, Collin Burns, He He, Shi Feng, Ethan Perez, Jan Leike  

**Link**: [PDF](https://arxiv.org/pdf/2506.10139)  

**Abstract**: To steer pretrained language models for downstream tasks, today's post-training paradigm relies on humans to specify desired behaviors. However, for models with superhuman capabilities, it is difficult or impossible to get high-quality human supervision. To address this challenge, we introduce a new unsupervised algorithm, Internal Coherence Maximization (ICM), to fine-tune pretrained language models on their own generated labels, \emph{without external supervision}. On GSM8k-verification, TruthfulQA, and Alpaca reward modeling tasks, our method matches the performance of training on golden supervision and outperforms training on crowdsourced human supervision. On tasks where LMs' capabilities are strongly superhuman, our method can elicit those capabilities significantly better than training on human labels. Finally, we show that our method can improve the training of frontier LMs: we use our method to train an unsupervised reward model and use reinforcement learning to train a Claude 3.5 Haiku-based assistant. Both the reward model and the assistant outperform their human-supervised counterparts. 

---
# ChartReasoner: Code-Driven Modality Bridging for Long-Chain Reasoning in Chart Question Answering 

**Authors**: Caijun Jia, Nan Xu, Jingxuan Wei, Qingli Wang, Lei Wang, Bihui Yu, Junnan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10116)  

**Abstract**: Recently, large language models have shown remarkable reasoning capabilities through long-chain reasoning before responding. However, how to extend this capability to visual reasoning tasks remains an open challenge. Existing multimodal reasoning approaches transfer such visual reasoning task into textual reasoning task via several image-to-text conversions, which often lose critical structural and semantic information embedded in visualizations, especially for tasks like chart question answering that require a large amount of visual details. To bridge this gap, we propose ChartReasoner, a code-driven novel two-stage framework designed to enable precise, interpretable reasoning over charts. We first train a high-fidelity model to convert diverse chart images into structured ECharts codes, preserving both layout and data semantics as lossless as possible. Then, we design a general chart reasoning data synthesis pipeline, which leverages this pretrained transport model to automatically and scalably generate chart reasoning trajectories and utilizes a code validator to filter out low-quality samples. Finally, we train the final multimodal model using a combination of supervised fine-tuning and reinforcement learning on our synthesized chart reasoning dataset and experimental results on four public benchmarks clearly demonstrate the effectiveness of our proposed ChartReasoner. It can preserve the original details of the charts as much as possible and perform comparably with state-of-the-art open-source models while using fewer parameters, approaching the performance of proprietary systems like GPT-4o in out-of-domain settings. 

---
# When Meaning Stays the Same, but Models Drift: Evaluating Quality of Service under Token-Level Behavioral Instability in LLMs 

**Authors**: Xiao Li, Joel Kreuzwieser, Alan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2506.10095)  

**Abstract**: We investigate how large language models respond to prompts that differ only in their token-level realization but preserve the same semantic intent, a phenomenon we call prompt variance. We propose Prompt-Based Semantic Shift (PBSS), a diagnostic framework for measuring behavioral drift in LLMs under semantically equivalent prompt rewordings. Applied to ten constrained tasks, PBSS reveals consistent, model-specific response shifts, suggesting statistical regularities linked to tokenization and decoding. These results highlight an overlooked dimension of model evaluation stability under rephrasing and suggest that tokenization strategies and decoding dynamics may contribute to post-training quality of service instability. 

---
# Chat-of-Thought: Collaborative Multi-Agent System for Generating Domain Specific Information 

**Authors**: Christodoulos Constantinides, Shuxin Lin, Nianjun Zhou, Dhaval Patel  

**Link**: [PDF](https://arxiv.org/pdf/2506.10086)  

**Abstract**: This paper presents a novel multi-agent system called Chat-of-Thought, designed to facilitate the generation of Failure Modes and Effects Analysis (FMEA) documents for industrial assets. Chat-of-Thought employs multiple collaborative Large Language Model (LLM)-based agents with specific roles, leveraging advanced AI techniques and dynamic task routing to optimize the generation and validation of FMEA tables. A key innovation in this system is the introduction of a Chat of Thought, where dynamic, multi-persona-driven discussions enable iterative refinement of content. This research explores the application domain of industrial equipment monitoring, highlights key challenges, and demonstrates the potential of Chat-of-Thought in addressing these challenges through interactive, template-driven workflows and context-aware agent collaboration. 

---
# A quantum semantic framework for natural language processing 

**Authors**: Christopher J. Agostino, Quan Le Thien, Molly Apsel, Denizhan Pak, Elina Lesyk, Ashabari Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.10077)  

**Abstract**: Semantic degeneracy represents a fundamental property of natural language that extends beyond simple polysemy to encompass the combinatorial explosion of potential interpretations that emerges as semantic expressions increase in complexity. Large Language Models (LLMs) and other modern NLP systems face inherent limitations precisely because they operate within natural language itself, making them subject to the same interpretive constraints imposed by semantic degeneracy. In this work, we argue using Kolmogorov complexity that as an expression's complexity grows, the likelihood of any interpreting agent (human or LLM-powered AI) recovering the single intended meaning vanishes. This computational intractability suggests the classical view that linguistic forms possess meaning in and of themselves is flawed. We alternatively posit that meaning is instead actualized through an observer-dependent interpretive act. To test this, we conducted a semantic Bell inequality test using diverse LLM agents as ``computational cognitive systems'' to interpret ambiguous word pairs under varied contextual settings. Across several independent experiments, we found average CHSH expectation values ranging from 1.2 to 2.8, with several runs yielding values (e.g., 2.3-2.4) that significantly violate the classical boundary ($|S|\leq2$). This demonstrates that linguistic interpretation under ambiguity can exhibit non-classical contextuality, consistent with results from human cognition experiments. These results inherently imply that classical frequentist-based analytical approaches for natural language are necessarily lossy. Instead, we propose that Bayesian-style repeated sampling approaches can provide more practically useful and appropriate characterizations of linguistic meaning in context. 

---
# TaskCraft: Automated Generation of Agentic Tasks 

**Authors**: Dingfeng Shi, Jingyi Cao, Qianben Chen, Weichen Sun, Weizhen Li, Hongxuan Lu, Fangchen Dong, Tianrui Qin, King Zhu, Minghao Yang, Jian Yang, Ge Zhang, Jiaheng Liu, Changwang Zhang, Jun Wang, Yuchen Eleanor Jiang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.10055)  

**Abstract**: Agentic tasks, which require multi-step problem solving with autonomy, tool use, and adaptive reasoning, are becoming increasingly central to the advancement of NLP and AI. However, existing instruction data lacks tool interaction, and current agentic benchmarks rely on costly human annotation, limiting their scalability. We introduce \textsc{TaskCraft}, an automated workflow for generating difficulty-scalable, multi-tool, and verifiable agentic tasks with execution trajectories. TaskCraft expands atomic tasks using depth-based and width-based extensions to create structurally and hierarchically complex challenges. Empirical results show that these tasks improve prompt optimization in the generation workflow and enhance supervised fine-tuning of agentic foundation models. We present a large-scale synthetic dataset of approximately 36,000 tasks with varying difficulty to support future research on agent tuning and evaluation. 

---
# A Survey of Automatic Evaluation Methods on Text, Visual and Speech Generations 

**Authors**: Tian Lan, Yang-Hao Zhou, Zi-Ao Ma, Fanshu Sun, Rui-Qing Sun, Junyu Luo, Rong-Cheng Tu, Heyan Huang, Chen Xu, Zhijing Wu, Xian-Ling Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10019)  

**Abstract**: Recent advances in deep learning have significantly enhanced generative AI capabilities across text, images, and audio. However, automatically evaluating the quality of these generated outputs presents ongoing challenges. Although numerous automatic evaluation methods exist, current research lacks a systematic framework that comprehensively organizes these methods across text, visual, and audio modalities. To address this issue, we present a comprehensive review and a unified taxonomy of automatic evaluation methods for generated content across all three modalities; We identify five fundamental paradigms that characterize existing evaluation approaches across these domains. Our analysis begins by examining evaluation methods for text generation, where techniques are most mature. We then extend this framework to image and audio generation, demonstrating its broad applicability. Finally, we discuss promising directions for future research in cross-modal evaluation methodologies. 

---
# MMMG: A Massive, Multidisciplinary, Multi-Tier Generation Benchmark for Text-to-Image Reasoning 

**Authors**: Yuxuan Luo, Yuhui Yuan, Junwen Chen, Haonan Cai, Ziyi Yue, Yuwei Yang, Fatima Zohra Daha, Ji Li, Zhouhui Lian  

**Link**: [PDF](https://arxiv.org/pdf/2506.10963)  

**Abstract**: In this paper, we introduce knowledge image generation as a new task, alongside the Massive Multi-Discipline Multi-Tier Knowledge-Image Generation Benchmark (MMMG) to probe the reasoning capability of image generation models. Knowledge images have been central to human civilization and to the mechanisms of human learning--a fact underscored by dual-coding theory and the picture-superiority effect. Generating such images is challenging, demanding multimodal reasoning that fuses world knowledge with pixel-level grounding into clear explanatory visuals. To enable comprehensive evaluation, MMMG offers 4,456 expert-validated (knowledge) image-prompt pairs spanning 10 disciplines, 6 educational levels, and diverse knowledge formats such as charts, diagrams, and mind maps. To eliminate confounding complexity during evaluation, we adopt a unified Knowledge Graph (KG) representation. Each KG explicitly delineates a target image's core entities and their dependencies. We further introduce MMMG-Score to evaluate generated knowledge images. This metric combines factual fidelity, measured by graph-edit distance between KGs, with visual clarity assessment. Comprehensive evaluations of 16 state-of-the-art text-to-image generation models expose serious reasoning deficits--low entity fidelity, weak relations, and clutter--with GPT-4o achieving an MMMG-Score of only 50.20, underscoring the benchmark's difficulty. To spur further progress, we release FLUX-Reason (MMMG-Score of 34.45), an effective and open baseline that combines a reasoning LLM with diffusion models and is trained on 16,000 curated knowledge image-prompt pairs. 

---
# Build the web for agents, not agents for the web 

**Authors**: Xing Han Lù, Gaurav Kamath, Marius Mosbach, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.10953)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and multimodal counterparts have spurred significant interest in developing web agents -- AI systems capable of autonomously navigating and completing tasks within web environments. While holding tremendous promise for automating complex web interactions, current approaches face substantial challenges due to the fundamental mismatch between human-designed interfaces and LLM capabilities. Current methods struggle with the inherent complexity of web inputs, whether processing massive DOM trees, relying on screenshots augmented with additional information, or bypassing the user interface entirely through API interactions. This position paper advocates for a paradigm shift in web agent research: rather than forcing web agents to adapt to interfaces designed for humans, we should develop a new interaction paradigm specifically optimized for agentic capabilities. To this end, we introduce the concept of an Agentic Web Interface (AWI), an interface specifically designed for agents to navigate a website. We establish six guiding principles for AWI design, emphasizing safety, efficiency, and standardization, to account for the interests of all primary stakeholders. This reframing aims to overcome fundamental limitations of existing interfaces, paving the way for more efficient, reliable, and transparent web agent design, which will be a collaborative effort involving the broader ML community. 

---
# GUARD: Guided Unlearning and Retention via Data Attribution for Large Language Models 

**Authors**: Evelyn Ma, Duo Zhou, Peizhi Niu, Huiting Zhou, Huan Zhang, Olgica Milenkovic, S. Rasoul Etesami  

**Link**: [PDF](https://arxiv.org/pdf/2506.10946)  

**Abstract**: Unlearning in large language models (LLMs) is becoming increasingly important due to regulatory compliance, copyright protection, and privacy concerns. However, a key challenge in LLM unlearning is unintended forgetting, where the removal of specific data inadvertently impairs the utility of the model and its retention of valuable, desired information. While prior work has primarily focused on architectural innovations, the influence of data-level factors on unlearning performance remains underexplored. As a result, existing methods often suffer from degraded retention when forgetting high-impact data. To address this, we propose GUARD-a novel framework for Guided Unlearning And Retention via Data attribution. At its core, GUARD introduces a lightweight proxy data attribution metric tailored for LLM unlearning, which quantifies the "alignment" between the forget and retain sets while remaining computationally efficient. Building on this, we design a novel unlearning objective that assigns adaptive, nonuniform unlearning weights to samples, inversely proportional to their proxy attribution scores. Through such a reallocation of unlearning power, GUARD mitigates unintended losses in retention. We provide rigorous theoretical guarantees that GUARD significantly enhances retention while maintaining forgetting metrics comparable to prior methods. Extensive experiments on the TOFU benchmark across multiple LLM architectures demonstrate that GUARD substantially improves utility preservation while ensuring effective unlearning. Notably, GUARD reduces utility sacrifice on the Retain Set by up to 194.92% in terms of Truth Ratio when forgetting 10% of the training data. 

---
# VINCIE: Unlocking In-context Image Editing from Video 

**Authors**: Leigang Qu, Feng Cheng, Ziyan Yang, Qi Zhao, Shanchuan Lin, Yichun Shi, Yicong Li, Wenjie Wang, Tat-Seng Chua, Lu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10941)  

**Abstract**: In-context image editing aims to modify images based on a contextual sequence comprising text and previously generated images. Existing methods typically depend on task-specific pipelines and expert models (e.g., segmentation and inpainting) to curate training data. In this work, we explore whether an in-context image editing model can be learned directly from videos. We introduce a scalable approach to annotate videos as interleaved multimodal sequences. To effectively learn from this data, we design a block-causal diffusion transformer trained on three proxy tasks: next-image prediction, current segmentation prediction, and next-segmentation prediction. Additionally, we propose a novel multi-turn image editing benchmark to advance research in this area. Extensive experiments demonstrate that our model exhibits strong in-context image editing capabilities and achieves state-of-the-art results on two multi-turn image editing benchmarks. Despite being trained exclusively on videos, our model also shows promising abilities in multi-concept composition, story generation, and chain-of-editing applications. 

---
# Robustly Improving LLM Fairness in Realistic Settings via Interpretability 

**Authors**: Adam Karvonen, Samuel Marks  

**Link**: [PDF](https://arxiv.org/pdf/2506.10922)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes hiring applications, making decisions that directly impact people's careers and livelihoods. While prior studies suggest simple anti-bias prompts can eliminate demographic biases in controlled evaluations, we find these mitigations fail when realistic contextual details are introduced. We address these failures through internal bias mitigation: by identifying and neutralizing sensitive attribute directions within model activations, we achieve robust bias reduction across all tested scenarios. Across leading commercial (GPT-4o, Claude 4 Sonnet, Gemini 2.5 Flash) and open-source models (Gemma-2 27B, Gemma-3, Mistral-24B), we find that adding realistic context such as company names, culture descriptions from public careers pages, and selective hiring constraints (e.g.,``only accept candidates in the top 10\%") induces significant racial and gender biases (up to 12\% differences in interview rates). When these biases emerge, they consistently favor Black over White candidates and female over male candidates across all tested models and scenarios. Moreover, models can infer demographics and become biased from subtle cues like college affiliations, with these biases remaining invisible even when inspecting the model's chain-of-thought reasoning. To address these limitations, our internal bias mitigation identifies race and gender-correlated directions and applies affine concept editing at inference time. Despite using directions from a simple synthetic dataset, the intervention generalizes robustly, consistently reducing bias to very low levels (typically under 1\%, always below 2.5\%) while largely maintaining model performance. Our findings suggest that practitioners deploying LLMs for hiring should adopt more realistic evaluation methodologies and consider internal mitigation strategies for equitable outcomes. 

---
# Breaking Bad Molecules: Are MLLMs Ready for Structure-Level Molecular Detoxification? 

**Authors**: Fei Lin, Ziyang Gong, Cong Wang, Yonglin Tian, Tengchao Zhang, Xue Yang, Gen Luo, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10912)  

**Abstract**: Toxicity remains a leading cause of early-stage drug development failure. Despite advances in molecular design and property prediction, the task of molecular toxicity repair - generating structurally valid molecular alternatives with reduced toxicity - has not yet been systematically defined or benchmarked. To fill this gap, we introduce ToxiMol, the first benchmark task for general-purpose Multimodal Large Language Models (MLLMs) focused on molecular toxicity repair. We construct a standardized dataset covering 11 primary tasks and 560 representative toxic molecules spanning diverse mechanisms and granularities. We design a prompt annotation pipeline with mechanism-aware and task-adaptive capabilities, informed by expert toxicological knowledge. In parallel, we propose an automated evaluation framework, ToxiEval, which integrates toxicity endpoint prediction, synthetic accessibility, drug-likeness, and structural similarity into a high-throughput evaluation chain for repair success. We systematically assess nearly 30 mainstream general-purpose MLLMs and design multiple ablation studies to analyze key factors such as evaluation criteria, candidate diversity, and failure attribution. Experimental results show that although current MLLMs still face significant challenges on this task, they begin to demonstrate promising capabilities in toxicity understanding, semantic constraint adherence, and structure-aware molecule editing. 

---
# The Diffusion Duality 

**Authors**: Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, Volodymyr Kuleshov  

**Link**: [PDF](https://arxiv.org/pdf/2506.10892)  

**Abstract**: Uniform-state discrete diffusion models hold the promise of fast text generation due to their inherent ability to self-correct. However, they are typically outperformed by autoregressive models and masked diffusion models. In this work, we narrow this performance gap by leveraging a key insight: Uniform-state diffusion processes naturally emerge from an underlying Gaussian diffusion. Our method, Duo, transfers powerful techniques from Gaussian diffusion to improve both training and sampling. First, we introduce a curriculum learning strategy guided by the Gaussian process, doubling training speed by reducing variance. Models trained with curriculum learning surpass autoregressive models in zero-shot perplexity on 3 of 7 benchmarks. Second, we present Discrete Consistency Distillation, which adapts consistency distillation from the continuous to the discrete setting. This algorithm unlocks few-step generation in diffusion language models by accelerating sampling by two orders of magnitude. We provide the code and model checkpoints on the project page: this http URL 

---
# VideoDeepResearch: Long Video Understanding With Agentic Tool Using 

**Authors**: Huaying Yuan, Zheng Liu, Junjie Zhou, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.10821)  

**Abstract**: Long video understanding (LVU) presents a significant challenge for current multi-modal large language models (MLLMs) due to the task's inherent complexity and context window constraint. It is widely assumed that addressing LVU tasks requires foundation MLLMs with extended context windows, strong visual perception capabilities, and proficient domain expertise. In this work, we challenge this common belief by introducing VideoDeepResearch, a novel agentic framework for long video understanding. Our approach relies solely on a text-only large reasoning model (LRM) combined with a modular multi-modal toolkit, including multimodal retrievers and visual perceivers, all of which are readily available in practice. For each LVU task, the system formulates a problem-solving strategy through reasoning, while selectively accessing and utilizing essential video content via tool using. We conduct extensive experiments on popular LVU benchmarks, including MLVU, Video-MME, and LVBench. Our results demonstrate that VideoDeepResearch achieves substantial improvements over existing MLLM baselines, surpassing the previous state-of-the-art by 9.6%, 6.6%, and 3.9% on MLVU (test), LVBench, and LongVideoBench, respectively. These findings highlight the promise of agentic systems in overcoming key challenges in LVU problems. 

---
# FASCIST-O-METER: Classifier for Neo-fascist Discourse Online 

**Authors**: Rudy Alexandro Garrido Veliz, Martin Semmann, Chris Biemann, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2506.10789)  

**Abstract**: Neo-fascism is a political and societal ideology that has been having remarkable growth in the last decade in the United States of America (USA), as well as in other Western societies. It poses a grave danger to democracy and the minorities it targets, and it requires active actions against it to avoid escalation. This work presents the first-of-its-kind neo-fascist coding scheme for digital discourse in the USA societal context, overseen by political science researchers. Our work bridges the gap between Natural Language Processing (NLP) and political science against this phenomena. Furthermore, to test the coding scheme, we collect a tremendous amount of activity on the internet from notable neo-fascist groups (the forums of Iron March and this http URL), and the guidelines are applied to a subset of the collected posts. Through crowdsourcing, we annotate a total of a thousand posts that are labeled as neo-fascist or non-neo-fascist. With this labeled data set, we fine-tune and test both Small Language Models (SLMs) and Large Language Models (LLMs), obtaining the very first classification models for neo-fascist discourse. We find that the prevalence of neo-fascist rhetoric in this kind of forum is ever-present, making them a good target for future research. The societal context is a key consideration for neo-fascist speech when conducting NLP research. Finally, the work against this kind of political movement must be pressed upon and continued for the well-being of a democratic society. Disclaimer: This study focuses on detecting neo-fascist content in text, similar to other hate speech analyses, without labeling individuals or organizations. 

---
# Neural at ArchEHR-QA 2025: Agentic Prompt Optimization for Evidence-Grounded Clinical Question Answering 

**Authors**: Sai Prasanna Teja Reddy Bogireddy, Abrar Majeedi, Viswanatha Reddy Gajjala, Zhuoyan Xu, Siddhant Rai, Vaishnav Potlapalli  

**Link**: [PDF](https://arxiv.org/pdf/2506.10751)  

**Abstract**: Automated question answering (QA) over electronic health records (EHRs) can bridge critical information gaps for clinicians and patients, yet it demands both precise evidence retrieval and faithful answer generation under limited supervision. In this work, we present Neural, the runner-up in the BioNLP 2025 ArchEHR-QA shared task on evidence-grounded clinical QA. Our proposed method decouples the task into (1) sentence-level evidence identification and (2) answer synthesis with explicit citations. For each stage, we automatically explore the prompt space with DSPy's MIPROv2 optimizer, jointly tuning instructions and few-shot demonstrations on the development set. A self-consistency voting scheme further improves evidence recall without sacrificing precision. On the hidden test set, our method attains an overall score of 51.5, placing second stage while outperforming standard zero-shot and few-shot prompting by over 20 and 10 points, respectively. These results indicate that data-driven prompt optimization is a cost-effective alternative to model fine-tuning for high-stakes clinical QA, advancing the reliability of AI assistants in healthcare. 

---
# TeleMath: A Benchmark for Large Language Models in Telecom Mathematical Problem Solving 

**Authors**: Vincenzo Colle, Mohamed Sana, Nicola Piovesan, Antonio De Domenico, Fadhel Ayed, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2506.10674)  

**Abstract**: The increasing adoption of artificial intelligence in telecommunications has raised interest in the capability of Large Language Models (LLMs) to address domain-specific, mathematically intensive tasks. Although recent advancements have improved the performance of LLMs in general mathematical reasoning, their effectiveness within specialized domains, such as signal processing, network optimization, and performance analysis, remains largely unexplored. To address this gap, we introduce TeleMath, the first benchmark dataset specifically designed to evaluate LLM performance in solving mathematical problems with numerical solutions in the telecommunications domain. Comprising 500 question-answer (QnA) pairs, TeleMath covers a wide spectrum of topics in the telecommunications field. This paper outlines the proposed QnAs generation pipeline, starting from a selected seed of problems crafted by Subject Matter Experts. The evaluation of a wide range of open-source LLMs reveals that best performance on TeleMath is achieved by recent models explicitly designed for mathematical or logical reasoning. In contrast, general-purpose models, even those with a large number of parameters, often struggle with these challenges. We have released the dataset and the evaluation code to ease result reproducibility and support future research. 

---
# Robust Unsupervised Adaptation of a Speech Recogniser Using Entropy Minimisation and Speaker Codes 

**Authors**: Rogier C. van Dalen, Shucong Zhang, Titouan Parcollet, Sourav Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2506.10653)  

**Abstract**: Speech recognisers usually perform optimally only in a specific environment and need to be adapted to work well in another. For adaptation to a new speaker, there is often too little data for fine-tuning to be robust, and that data is usually unlabelled. This paper proposes a combination of approaches to make adaptation to a single minute of data robust. First, instead of estimating the adaptation parameters with cross-entropy on a single error-prone hypothesis or "pseudo-label", this paper proposes a novel loss function, the conditional entropy over complete hypotheses. Using multiple hypotheses makes adaptation more robust to errors in the initial recognition. Second, a "speaker code" characterises a speaker in a vector short enough that it requires little data to estimate. On a far-field noise-augmented version of Common Voice, the proposed scheme yields a 20% relative improvement in word error rate on one minute of adaptation data, increasing on 10 minutes to 29%. 

---
# Conversational Search: From Fundamentals to Frontiers in the LLM Era 

**Authors**: Fengran Mo, Chuan Meng, Mohammad Aliannejadi, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.10635)  

**Abstract**: Conversational search enables multi-turn interactions between users and systems to fulfill users' complex information needs. During this interaction, the system should understand the users' search intent within the conversational context and then return the relevant information through a flexible, dialogue-based interface. The recent powerful large language models (LLMs) with capacities of instruction following, content generation, and reasoning, attract significant attention and advancements, providing new opportunities and challenges for building up intelligent conversational search systems. This tutorial aims to introduce the connection between fundamentals and the emerging topics revolutionized by LLMs in the context of conversational search. It is designed for students, researchers, and practitioners from both academia and industry. Participants will gain a comprehensive understanding of both the core principles and cutting-edge developments driven by LLMs in conversational search, equipping them with the knowledge needed to contribute to the development of next-generation conversational search systems. 

---
# Deep Learning-Based Digitization of Overlapping ECG Images with Open-Source Python Code 

**Authors**: Reza Karbasi, Masoud Rahimi, Abdol-Hossein Vahabie, Hadi Moradi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10617)  

**Abstract**: This paper addresses the persistent challenge of accurately digitizing paper-based electrocardiogram (ECG) recordings, with a particular focus on robustly handling single leads compromised by signal overlaps-a common yet under-addressed issue in existing methodologies. We propose a two-stage pipeline designed to overcome this limitation. The first stage employs a U-Net based segmentation network, trained on a dataset enriched with overlapping signals and fortified with custom data augmentations, to accurately isolate the primary ECG trace. The subsequent stage converts this refined binary mask into a time-series signal using established digitization techniques, enhanced by an adaptive grid detection module for improved versatility across different ECG formats and scales. Our experimental results demonstrate the efficacy of our approach. The U-Net architecture achieves an IoU of 0.87 for the fine-grained segmentation task. Crucially, our proposed digitization method yields superior performance compared to a well-established baseline technique across both non-overlapping and challenging overlapping ECG samples. For non-overlapping signals, our method achieved a Mean Squared Error (MSE) of 0.0010 and a Pearson Correlation Coefficient (rho) of 0.9644, compared to 0.0015 and 0.9366, respectively, for the baseline. On samples with signal overlap, our method achieved an MSE of 0.0029 and a rho of 0.9641, significantly improving upon the baseline's 0.0178 and 0.8676. This work demonstrates an effective strategy to significantly enhance digitization accuracy, especially in the presence of signal overlaps, thereby laying a strong foundation for the reliable conversion of analog ECG records into analyzable digital data for contemporary research and clinical applications. The implementation is publicly available at this GitHub repository: this https URL. 

---
# Encoding call-by-push-value in the pi-calculus 

**Authors**: Benjamin Bennetzen, Nikolaj Rossander Kristensen, Peter Buus Steffensen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10584)  

**Abstract**: In this report we define an encoding of Levys call-by-push-value lambda-calculus (CBPV) in the pi-calculus, and prove that our encoding is both sound and complete. We present informal (by-hand) proofs of soundness, completeness, and all required lemmas. The encoding is specialized to the internal pi-calculus (pi-i-calculus) to circumvent certain challenges associated with using de Bruijn index in a formalization, and it also helps with bisimulation as early-, late- and open-bisimulation coincide in this setting, furthermore bisimulation is a congruence. Additionally, we argue that our encoding also satisfies the five criteria for good encodings proposed by Gorla, as well as show similarities between Milners and our encoding. This paper includes encodings from CBPV in the pi-i-calculus, asynchronous polyadic pi-calculus and the local pi-calculus. We begin a formalization of the proof in Coq for the soundness and completeness of the encoding in the pi-i-calculus. Not all lemmas used in the formalization are themselves formally proven. However, we argue that the non-proven lemmas are reasonable, as they are proven by hand, or amount to Coq formalities that are straightforward given informal arguments. 

---
# Scientists' First Exam: Probing Cognitive Abilities of MLLM via Perception, Understanding, and Reasoning 

**Authors**: Yuhao Zhou, Yiheng Wang, Xuming He, Ruoyao Xiao, Zhiwei Li, Qiantai Feng, Zijie Guo, Yuejin Yang, Hao Wu, Wenxuan Huang, Jiaqi Wei, Dan Si, Xiuqi Yao, Jia Bu, Haiwen Huang, Tianfan Fu, Shixiang Tang, Ben Fei, Dongzhan Zhou, Fenghua Ling, Yan Lu, Siqi Sun, Chenhui Li, Guanjie Zheng, Jiancheng Lv, Wenlong Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.10521)  

**Abstract**: Scientific discoveries increasingly rely on complex multimodal reasoning based on information-intensive scientific data and domain-specific expertise. Empowered by expert-level scientific benchmarks, scientific Multimodal Large Language Models (MLLMs) hold the potential to significantly enhance this discovery process in realistic workflows. However, current scientific benchmarks mostly focus on evaluating the knowledge understanding capabilities of MLLMs, leading to an inadequate assessment of their perception and reasoning abilities. To address this gap, we present the Scientists' First Exam (SFE) benchmark, designed to evaluate the scientific cognitive capacities of MLLMs through three interconnected levels: scientific signal perception, scientific attribute understanding, scientific comparative reasoning. Specifically, SFE comprises 830 expert-verified VQA pairs across three question types, spanning 66 multimodal tasks across five high-value disciplines. Extensive experiments reveal that current state-of-the-art GPT-o3 and InternVL-3 achieve only 34.08% and 26.52% on SFE, highlighting significant room for MLLMs to improve in scientific realms. We hope the insights obtained in SFE will facilitate further developments in AI-enhanced scientific discoveries. 

---
# Towards Robust Multimodal Emotion Recognition under Missing Modalities and Distribution Shifts 

**Authors**: Guowei Zhong, Ruohong Huan, Mingzhen Wu, Ronghua Liang, Peng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10452)  

**Abstract**: Recent advancements in Multimodal Emotion Recognition (MER) face challenges in addressing both modality missing and Out-Of-Distribution (OOD) data simultaneously. Existing methods often rely on specific models or introduce excessive parameters, which limits their practicality. To address these issues, we propose a novel robust MER framework, Causal Inference Distiller (CIDer), and introduce a new task, Random Modality Feature Missing (RMFM), to generalize the definition of modality missing. CIDer integrates two key components: a Model-Specific Self-Distillation (MSSD) module and a Model-Agnostic Causal Inference (MACI) module. MSSD enhances robustness under the RMFM task through a weight-sharing self-distillation approach applied across low-level features, attention maps, and high-level representations. Additionally, a Word-level Self-aligned Attention Module (WSAM) reduces computational complexity, while a Multimodal Composite Transformer (MCT) facilitates efficient multimodal fusion. To tackle OOD challenges, MACI employs a tailored causal graph to mitigate label and language biases using a Multimodal Causal Module (MCM) and fine-grained counterfactual texts. Notably, MACI can independently enhance OOD generalization with minimal additional parameters. Furthermore, we also introduce the new repartitioned MER OOD datasets. Experimental results demonstrate that CIDer achieves robust performance in both RMFM and OOD scenarios, with fewer parameters and faster training compared to state-of-the-art methods. The implementation of this work is publicly accessible at this https URL. 

---
# PAL: Probing Audio Encoders via LLMs -- A Study of Information Transfer from Audio Encoders to LLMs 

**Authors**: Tony Alex, Wish Suharitdamrong, Sara Atito, Armin Mustafa, Philip J. B. Jackson, Imran Razzak, Muhammad Awais  

**Link**: [PDF](https://arxiv.org/pdf/2506.10423)  

**Abstract**: The integration of audio perception capabilities into Large Language Models (LLMs) has enabled significant advances in Audio-LLMs. Although application-focused developments, particularly in curating training data for specific capabilities e.g., audio reasoning, have progressed rapidly, the underlying mechanisms that govern efficient transfer of rich semantic representations from audio encoders to LLMs remain under-explored. We conceptualize effective audio-LLM interaction as the LLM's ability to proficiently probe the audio encoder representations to satisfy textual queries. This paper presents a systematic investigation on how architectural design choices can affect that. Beginning with a standard Pengi/LLaVA-style audio-LLM architecture, we propose and evaluate several modifications guided by hypotheses derived from mechanistic interpretability studies and LLM operational principles. Our experiments demonstrate that: (1) delaying audio integration until the LLM's initial layers establish textual context that enhances its ability to probe the audio representations for relevant information; (2) the LLM can proficiently probe audio representations exclusively through LLM layer's attention submodule, without requiring propagation to its Feed-Forward Network (FFN) submodule; (3) an efficiently integrated ensemble of diverse audio encoders provides richer, complementary representations, thereby broadening the LLM's capacity to probe a wider spectrum of audio information. All hypotheses are evaluated using an identical three-stage training curriculum on a dataset of 5.6 million audio-text pairs, ensuring controlled comparisons. Our final architecture, which incorporates all proposed modifications, achieves relative improvements from 10\% to 60\% over the baseline, validating our approach to optimizing cross-modal information transfer in audio-LLMs. Project page: this https URL 

---
# Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series 

**Authors**: Ching Chang, Jeehyun Hwang, Yidan Shi, Haixin Wang, Wen-Chih Peng, Tien-Fu Chen, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10412)  

**Abstract**: Time series data in real-world applications such as healthcare, climate modeling, and finance are often irregular, multimodal, and messy, with varying sampling rates, asynchronous modalities, and pervasive missingness. However, existing benchmarks typically assume clean, regularly sampled, unimodal data, creating a significant gap between research and real-world deployment. We introduce Time-IMM, a dataset specifically designed to capture cause-driven irregularity in multimodal multivariate time series. Time-IMM represents nine distinct types of time series irregularity, categorized into trigger-based, constraint-based, and artifact-based mechanisms. Complementing the dataset, we introduce IMM-TSF, a benchmark library for forecasting on irregular multimodal time series, enabling asynchronous integration and realistic evaluation. IMM-TSF includes specialized fusion modules, including a timestamp-to-text fusion module and a multimodality fusion module, which support both recency-aware averaging and attention-based integration strategies. Empirical results demonstrate that explicitly modeling multimodality on irregular time series data leads to substantial gains in forecasting performance. Time-IMM and IMM-TSF provide a foundation for advancing time series analysis under real-world conditions. The dataset is publicly available at this https URL, and the benchmark library can be accessed at this https URL. 

---
# Discovering Hierarchical Latent Capabilities of Language Models via Causal Representation Learning 

**Authors**: Jikai Jin, Vasilis Syrgkanis, Sham Kakade, Hanlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10378)  

**Abstract**: Faithful evaluation of language model capabilities is crucial for deriving actionable insights that can inform model development. However, rigorous causal evaluations in this domain face significant methodological challenges, including complex confounding effects and prohibitive computational costs associated with extensive retraining. To tackle these challenges, we propose a causal representation learning framework wherein observed benchmark performance is modeled as a linear transformation of a few latent capability factors. Crucially, these latent factors are identified as causally interrelated after appropriately controlling for the base model as a common confounder. Applying this approach to a comprehensive dataset encompassing over 1500 models evaluated across six benchmarks from the Open LLM Leaderboard, we identify a concise three-node linear causal structure that reliably explains the observed performance variations. Further interpretation of this causal structure provides substantial scientific insights beyond simple numerical rankings: specifically, we reveal a clear causal direction starting from general problem-solving capabilities, advancing through instruction-following proficiency, and culminating in mathematical reasoning ability. Our results underscore the essential role of carefully controlling base model variations during evaluation, a step critical to accurately uncovering the underlying causal relationships among latent model capabilities. 

---
# Can We Infer Confidential Properties of Training Data from LLMs? 

**Authors**: Penguin Huang, Chhavi Yadav, Ruihan Wu, Kamalika Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2506.10364)  

**Abstract**: Large language models (LLMs) are increasingly fine-tuned on domain-specific datasets to support applications in fields such as healthcare, finance, and law. These fine-tuning datasets often have sensitive and confidential dataset-level properties -- such as patient demographics or disease prevalence -- that are not intended to be revealed. While prior work has studied property inference attacks on discriminative models (e.g., image classification models) and generative models (e.g., GANs for image data), it remains unclear if such attacks transfer to LLMs. In this work, we introduce PropInfer, a benchmark task for evaluating property inference in LLMs under two fine-tuning paradigms: question-answering and chat-completion. Built on the ChatDoctor dataset, our benchmark includes a range of property types and task configurations. We further propose two tailored attacks: a prompt-based generation attack and a shadow-model attack leveraging word frequency signals. Empirical evaluations across multiple pretrained LLMs show the success of our attacks, revealing a previously unrecognized vulnerability in LLMs. 

---
# An Analysis of Datasets, Metrics and Models in Keyphrase Generation 

**Authors**: Florian Boudin, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.10346)  

**Abstract**: Keyphrase generation refers to the task of producing a set of words or phrases that summarises the content of a document. Continuous efforts have been dedicated to this task over the past few years, spreading across multiple lines of research, such as model architectures, data resources, and use-case scenarios. Yet, the current state of keyphrase generation remains unknown as there has been no attempt to review and analyse previous work. In this paper, we bridge this gap by presenting an analysis of over 50 research papers on keyphrase generation, offering a comprehensive overview of recent progress, limitations, and open challenges. Our findings highlight several critical issues in current evaluation practices, such as the concerning similarity among commonly-used benchmark datasets and inconsistencies in metric calculations leading to overestimated performances. Additionally, we address the limited availability of pre-trained models by releasing a strong PLM-based model for keyphrase generation as an effort to facilitate future research. 

---
# Provably Learning from Language Feedback 

**Authors**: Wanqiao Xu, Allen Nie, Ruijie Zheng, Aditya Modi, Adith Swaminathan, Ching-An Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10341)  

**Abstract**: Interactively learning from observation and language feedback is an increasingly studied area driven by the emergence of large language model (LLM) agents. While impressive empirical demonstrations have been shown, so far a principled framing of these decision problems remains lacking. In this paper, we formalize the Learning from Language Feedback (LLF) problem, assert sufficient assumptions to enable learning despite latent rewards, and introduce $\textit{transfer eluder dimension}$ as a complexity measure to characterize the hardness of LLF problems. We show that transfer eluder dimension captures the intuition that information in the feedback changes the learning complexity of the LLF problem. We demonstrate cases where learning from rich language feedback can be exponentially faster than learning from reward. We develop a no-regret algorithm, called $\texttt{HELiX}$, that provably solves LLF problems through sequential interactions, with performance guarantees that scale with the transfer eluder dimension of the problem. Across several empirical domains, we show that $\texttt{HELiX}$ performs well even when repeatedly prompting LLMs does not work reliably. Our contributions mark a first step towards designing principled interactive learning algorithms from generic language feedback. 

---
# Detecting Sockpuppetry on Wikipedia Using Meta-Learning 

**Authors**: Luc Raszewski, Christine De Kock  

**Link**: [PDF](https://arxiv.org/pdf/2506.10314)  

**Abstract**: Malicious sockpuppet detection on Wikipedia is critical to preserving access to reliable information on the internet and preventing the spread of disinformation. Prior machine learning approaches rely on stylistic and meta-data features, but do not prioritise adaptability to author-specific behaviours. As a result, they struggle to effectively model the behaviour of specific sockpuppet-groups, especially when text data is limited. To address this, we propose the application of meta-learning, a machine learning technique designed to improve performance in data-scarce settings by training models across multiple tasks. Meta-learning optimises a model for rapid adaptation to the writing style of a new sockpuppet-group. Our results show that meta-learning significantly enhances the precision of predictions compared to pre-trained models, marking an advancement in combating sockpuppetry on open editing platforms. We release a new dataset of sockpuppet investigations to foster future research in both sockpuppetry and meta-learning fields. 

---
# AC/DC: LLM-based Audio Comprehension via Dialogue Continuation 

**Authors**: Yusuke Fujita, Tomoya Mizumoto, Atsushi Kojima, Lianbo Liu, Yui Sudo  

**Link**: [PDF](https://arxiv.org/pdf/2506.10312)  

**Abstract**: We propose an instruction-following audio comprehension model that leverages the dialogue continuation ability of large language models (LLMs). Instead of directly generating target captions in training data, the proposed method trains a model to produce responses as if the input caption triggered a dialogue. This dialogue continuation training mitigates the caption variation problem. Learning to continue a dialogue effectively captures the caption's meaning beyond its surface-level words. As a result, our model enables zero-shot instruction-following capability without multitask instruction tuning, even trained solely on audio captioning datasets. Experiments on AudioCaps, WavCaps, and Clotho datasets with AudioBench audio-scene question-answering tests demonstrate our model's ability to follow various unseen instructions. 

---
# Discrete Audio Tokens: More Than a Survey! 

**Authors**: Pooneh Mousavi, Gallil Maimon, Adel Moumen, Darius Petermann, Jiatong Shi, Haibin Wu, Haici Yang, Anastasia Kuznetsova, Artem Ploujnikov, Ricard Marxer, Bhuvana Ramabhadran, Benjamin Elizalde, Loren Lugosch, Jinyu Li, Cem Subakan, Phil Woodland, Minje Kim, Hung-yi Lee, Shinji Watanabe, Yossi Adi, Mirco Ravanelli  

**Link**: [PDF](https://arxiv.org/pdf/2506.10274)  

**Abstract**: Discrete audio tokens are compact representations that aim to preserve perceptual quality, phonetic content, and speaker characteristics while enabling efficient storage and inference, as well as competitive performance across diverse downstream this http URL provide a practical alternative to continuous features, enabling the integration of speech and audio into modern large language models (LLMs). As interest in token-based audio processing grows, various tokenization methods have emerged, and several surveys have reviewed the latest progress in the field. However, existing studies often focus on specific domains or tasks and lack a unified comparison across various benchmarks. This paper presents a systematic review and benchmark of discrete audio tokenizers, covering three domains: speech, music, and general audio. We propose a taxonomy of tokenization approaches based on encoder-decoder, quantization techniques, training paradigm, streamability, and application domains. We evaluate tokenizers on multiple benchmarks for reconstruction, downstream performance, and acoustic language modeling, and analyze trade-offs through controlled ablation studies. Our findings highlight key limitations, practical considerations, and open challenges, providing insight and guidance for future research in this rapidly evolving area. For more information, including our main results and tokenizer database, please refer to our website: this https URL. 

---
# Prompt Attacks Reveal Superficial Knowledge Removal in Unlearning Methods 

**Authors**: Yeonwoo Jang, Shariqah Hossain, Ashwin Sreevatsa, Diogo Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2506.10236)  

**Abstract**: In this work, we show that some machine unlearning methods may fail when subjected to straightforward prompt attacks. We systematically evaluate eight unlearning techniques across three model families, and employ output-based, logit-based, and probe analysis to determine to what extent supposedly unlearned knowledge can be retrieved. While methods like RMU and TAR demonstrate robust unlearning, ELM remains vulnerable to specific prompt attacks (e.g., Hindi filler text in original prompt recovering 57.3% accuracy). Our logit analysis also confirms that unlearned models are generally not hiding knowledge by modifying the way the answer is formatted, as the correlation between output and logit accuracy is strong. These results challenge prevailing assumptions about unlearning effectiveness and highlight the need for evaluation frameworks that can reliably distinguish between true knowledge removal and superficial output suppression. We also publicly make available our evaluation framework to easily evaluate prompting techniques to retrieve unlearning knowledge. 

---
# Disclosure Audits for LLM Agents 

**Authors**: Saswat Das, Jameson Sandler, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2506.10171)  

**Abstract**: Large Language Model agents have begun to appear as personal assistants, customer service bots, and clinical aides. While these applications deliver substantial operational benefits, they also require continuous access to sensitive data, which increases the likelihood of unauthorized disclosures. This study proposes an auditing framework for conversational privacy that quantifies and audits these risks. The proposed Conversational Manipulation for Privacy Leakage (CMPL) framework, is an iterative probing strategy designed to stress-test agents that enforce strict privacy directives. Rather than focusing solely on a single disclosure event, CMPL simulates realistic multi-turn interactions to systematically uncover latent vulnerabilities. Our evaluation on diverse domains, data modalities, and safety configurations demonstrate the auditing framework's ability to reveal privacy risks that are not deterred by existing single-turn defenses. In addition to introducing CMPL as a diagnostic tool, the paper delivers (1) an auditing procedure grounded in quantifiable risk metrics and (2) an open benchmark for evaluation of conversational privacy across agent implementations. 

---
# One Patient, Many Contexts: Scaling Medical AI Through Contextual Intelligence 

**Authors**: Michelle M. Li, Ben Y. Reis, Adam Rodman, Tianxi Cai, Noa Dagan, Ran D. Balicer, Joseph Loscalzo, Isaac S. Kohane, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2506.10157)  

**Abstract**: Medical foundation models, including language models trained on clinical notes, vision-language models on medical images, and multimodal models on electronic health records, can summarize clinical notes, answer medical questions, and assist in decision-making. Adapting these models to new populations, specialties, or settings typically requires fine-tuning, careful prompting, or retrieval from knowledge bases. This can be impractical, and limits their ability to interpret unfamiliar inputs and adjust to clinical situations not represented during training. As a result, models are prone to contextual errors, where predictions appear reasonable but fail to account for critical patient-specific or contextual information. These errors stem from a fundamental limitation that current models struggle with: dynamically adjusting their behavior across evolving contexts of medical care. In this Perspective, we outline a vision for context-switching in medical AI: models that dynamically adapt their reasoning without retraining to new specialties, populations, workflows, and clinical roles. We envision context-switching AI to diagnose, manage, and treat a wide range of diseases across specialties and regions, and expand access to medical care. 

---
# Omni-DPO: A Dual-Perspective Paradigm for Dynamic Preference Learning of LLMs 

**Authors**: Shangpin Peng, Weinong Wang, Zhuotao Tian, Senqiao Yang, Xing Wu, Haotian Xu, Chengquan Zhang, Takashi Isobe, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10054)  

**Abstract**: Direct Preference Optimization (DPO) has become a cornerstone of reinforcement learning from human feedback (RLHF) due to its simplicity and efficiency. However, existing DPO-based approaches typically treat all preference pairs uniformly, ignoring critical variations in their inherent quality and learning utility, leading to suboptimal data utilization and performance. To address this challenge, we propose Omni-DPO, a dual-perspective optimization framework that jointly accounts for (1) the inherent quality of each preference pair and (2) the model's evolving performance on those pairs. By adaptively weighting samples according to both data quality and the model's learning dynamics during training, Omni-DPO enables more effective training data utilization and achieves better performance. Experimental results on various models and benchmarks demonstrate the superiority and generalization capabilities of Omni-DPO. On textual understanding tasks, Gemma-2-9b-it finetuned with Omni-DPO beats the leading LLM, Claude 3 Opus, by a significant margin of 6.7 points on the Arena-Hard benchmark. On mathematical reasoning tasks, Omni-DPO consistently outperforms the baseline methods across all benchmarks, providing strong empirical evidence for the effectiveness and robustness of our approach. Code and models will be available at this https URL. 

---
# GenBreak: Red Teaming Text-to-Image Generators Using Large Language Models 

**Authors**: Zilong Wang, Xiang Zheng, Xiaosen Wang, Bo Wang, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10047)  

**Abstract**: Text-to-image (T2I) models such as Stable Diffusion have advanced rapidly and are now widely used in content creation. However, these models can be misused to generate harmful content, including nudity or violence, posing significant safety risks. While most platforms employ content moderation systems, underlying vulnerabilities can still be exploited by determined adversaries. Recent research on red-teaming and adversarial attacks against T2I models has notable limitations: some studies successfully generate highly toxic images but use adversarial prompts that are easily detected and blocked by safety filters, while others focus on bypassing safety mechanisms but fail to produce genuinely harmful outputs, neglecting the discovery of truly high-risk prompts. Consequently, there remains a lack of reliable tools for evaluating the safety of defended T2I models. To address this gap, we propose GenBreak, a framework that fine-tunes a red-team large language model (LLM) to systematically explore underlying vulnerabilities in T2I generators. Our approach combines supervised fine-tuning on curated datasets with reinforcement learning via interaction with a surrogate T2I model. By integrating multiple reward signals, we guide the LLM to craft adversarial prompts that enhance both evasion capability and image toxicity, while maintaining semantic coherence and diversity. These prompts demonstrate strong effectiveness in black-box attacks against commercial T2I generators, revealing practical and concerning safety weaknesses. 

---
# Evaluation empirique de la sécurisation et de l'alignement de ChatGPT et Gemini: analyse comparative des vulnérabilités par expérimentations de jailbreaks 

**Authors**: Rafaël Nouailles  

**Link**: [PDF](https://arxiv.org/pdf/2506.10029)  

**Abstract**: Large Language models (LLMs) are transforming digital usage, particularly in text generation, image creation, information retrieval and code development. ChatGPT, launched by OpenAI in November 2022, quickly became a reference, prompting the emergence of competitors such as Google's Gemini. However, these technological advances raise new cybersecurity challenges, including prompt injection attacks, the circumvention of regulatory measures (jailbreaking), the spread of misinformation (hallucinations) and risks associated with deep fakes. This paper presents a comparative analysis of the security and alignment levels of ChatGPT and Gemini, as well as a taxonomy of jailbreak techniques associated with experiments. 

---
# Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models 

**Authors**: Elena Sofia Ruzzetti, Giancarlo A. Xompero, Davide Venditti, Fabio Massimo Zanzotto  

**Link**: [PDF](https://arxiv.org/pdf/2506.10024)  

**Abstract**: Large Language Models (LLMs) memorize, and thus, among huge amounts of uncontrolled data, may memorize Personally Identifiable Information (PII), which should not be stored and, consequently, not leaked. In this paper, we introduce Private Memorization Editing (PME), an approach for preventing private data leakage that turns an apparent limitation, that is, the LLMs' memorization ability, into a powerful privacy defense strategy. While attacks against LLMs have been performed exploiting previous knowledge regarding their training data, our approach aims to exploit the same kind of knowledge in order to make a model more robust. We detect a memorized PII and then mitigate the memorization of PII by editing a model knowledge of its training data. We verify that our procedure does not affect the underlying language model while making it more robust against privacy Training Data Extraction attacks. We demonstrate that PME can effectively reduce the number of leaked PII in a number of configurations, in some cases even reducing the accuracy of the privacy attacks to zero. 

---
# From Threat to Tool: Leveraging Refusal-Aware Injection Attacks for Safety Alignment 

**Authors**: Kyubyung Chae, Hyunbin Jin, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10020)  

**Abstract**: Safely aligning large language models (LLMs) often demands extensive human-labeled preference data, a process that's both costly and time-consuming. While synthetic data offers a promising alternative, current methods frequently rely on complex iterative prompting or auxiliary models. To address this, we introduce Refusal-Aware Adaptive Injection (RAAI), a straightforward, training-free, and model-agnostic framework that repurposes LLM attack techniques. RAAI works by detecting internal refusal signals and adaptively injecting predefined phrases to elicit harmful, yet fluent, completions. Our experiments show RAAI effectively jailbreaks LLMs, increasing the harmful response rate from a baseline of 2.15% to up to 61.04% on average across four benchmarks. Crucially, fine-tuning LLMs with the synthetic data generated by RAAI improves model robustness against harmful prompts while preserving general capabilities on standard tasks like MMLU and ARC. This work highlights how LLM attack methodologies can be reframed as practical tools for scalable and controllable safety alignment. 

---
# Multimodal Large Language Models: A Survey 

**Authors**: Longzhen Han, Awes Mubarak, Almas Baimagambetov, Nikolaos Polatidis, Thar Baker  

**Link**: [PDF](https://arxiv.org/pdf/2506.10016)  

**Abstract**: Multimodal Large Language Models (MLLMs) have rapidly evolved beyond text generation, now spanning diverse output modalities including images, music, video, human motion, and 3D objects, by integrating language with other sensory modalities under unified architectures. This survey categorises six primary generative modalities and examines how foundational techniques, namely Self-Supervised Learning (SSL), Mixture of Experts (MoE), Reinforcement Learning from Human Feedback (RLHF), and Chain-of-Thought (CoT) prompting, enable cross-modal capabilities. We analyze key models, architectural trends, and emergent cross-modal synergies, while highlighting transferable techniques and unresolved challenges. Architectural innovations like transformers and diffusion models underpin this convergence, enabling cross-modal transfer and modular specialization. We highlight emerging patterns of synergy, and identify open challenges in evaluation, modularity, and structured reasoning. This survey offers a unified perspective on MLLM development and identifies critical paths toward more general-purpose, adaptive, and interpretable multimodal systems. 

---
# Multimodal Cinematic Video Synthesis Using Text-to-Image and Audio Generation Models 

**Authors**: Sridhar S, Nithin A, Shakeel Rifath, Vasantha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2506.10005)  

**Abstract**: Advances in generative artificial intelligence have altered multimedia creation, allowing for automatic cinematic video synthesis from text inputs. This work describes a method for creating 60-second cinematic movies incorporating Stable Diffusion for high-fidelity image synthesis, GPT-2 for narrative structuring, and a hybrid audio pipeline using gTTS and YouTube-sourced music. It uses a five-scene framework, which is augmented by linear frame interpolation, cinematic post-processing (e.g., sharpening), and audio-video synchronization to provide professional-quality results. It was created in a GPU-accelerated Google Colab environment using Python 3.11. It has a dual-mode Gradio interface (Simple and Advanced), which supports resolutions of up to 1024x768 and frame rates of 15-30 FPS. Optimizations such as CUDA memory management and error handling ensure reliability. The experiments demonstrate outstanding visual quality, narrative coherence, and efficiency, furthering text-to-video synthesis for creative, educational, and industrial applications. 

---
