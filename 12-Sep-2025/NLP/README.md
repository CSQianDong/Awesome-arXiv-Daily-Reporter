# CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models 

**Authors**: Runpeng Dai, Linfeng Song, Haolin Liu, Zhenwen Liang, Dian Yu, Haitao Mi, Zhaopeng Tu, Rui Liu, Tong Zheng, Hongtu Zhu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09675)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful paradigm for enhancing the reasoning ability of Large Language Models (LLMs). Yet current RLVR methods often explore poorly, leading to premature convergence and entropy collapse. To address this challenge, we introduce Curiosity-Driven Exploration (CDE), a framework that leverages the model's own intrinsic sense of curiosity to guide exploration. We formalize curiosity with signals from both the actor and the critic: for the actor, we use perplexity over its generated response, and for the critic, we use the variance of value estimates from a multi-head architecture. Both signals serve as an exploration bonus within the RLVR framework to guide the model. Our theoretical analysis shows that the actor-wise bonus inherently penalizes overconfident errors and promotes diversity among correct responses; moreover, we connect the critic-wise bonus to the well-established count-based exploration bonus in RL. Empirically, our method achieves an approximate +3 point improvement over standard RLVR using GRPO/PPO on AIME benchmarks. Further analysis identifies a calibration collapse mechanism within RLVR, shedding light on common LLM failure modes. 

---
# Steering MoE LLMs via Expert (De)Activation 

**Authors**: Mohsen Fayyaz, Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Ryan Rossi, Trung Bui, Hinrich Schütze, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.09660)  

**Abstract**: Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts. 

---
# All for One: LLMs Solve Mental Math at the Last Token With Information Transferred From Other Tokens 

**Authors**: Siddarth Mamidanna, Daking Rai, Ziyu Yao, Yilun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.09650)  

**Abstract**: Large language models (LLMs) demonstrate proficiency across numerous computational tasks, yet their inner workings remain unclear. In theory, the combination of causal self-attention and multilayer perceptron layers allows every token to access and compute information based on all preceding tokens. In practice, to what extent are such operations present? In this paper, on mental math tasks (i.e., direct math calculation via next-token prediction without explicit reasoning), we investigate this question in three steps: inhibiting input-specific token computations in the initial layers, restricting the routes of information transfer across token positions in the next few layers, and forcing all computation to happen at the last token in the remaining layers. With two proposed techniques, Context-Aware Mean Ablation (CAMA) and Attention-Based Peeking (ABP), we identify an All-for-One subgraph (AF1) with high accuracy on a wide variety of mental math tasks, where meaningful computation occurs very late (in terms of layer depth) and only at the last token, which receives information of other tokens in few specific middle layers. Experiments on a variety of models and arithmetic expressions show that this subgraph is sufficient and necessary for high model performance, transfers across different models, and works on a variety of input styles. Ablations on different CAMA and ABP alternatives reveal their unique advantages over other methods, which may be of independent interest. 

---
# Bridging the Capability Gap: Joint Alignment Tuning for Harmonizing LLM-based Multi-Agent Systems 

**Authors**: Minghang Zhu, Zhengliang Shi, Zhiwei Xu, Shiguang Wu, Lingjie Wang, Pengjie Ren, Zhaochun Ren, Zhumin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.09629)  

**Abstract**: The advancement of large language models (LLMs) has enabled the construction of multi-agent systems to solve complex tasks by dividing responsibilities among specialized agents, such as a planning agent for subgoal generation and a grounding agent for executing tool-use actions. Most existing methods typically fine-tune these agents independently, leading to capability gaps among them with poor coordination. To address this, we propose MOAT, a Multi-Agent Joint Alignment Tuning framework that improves agents collaboration through iterative alignment. MOAT alternates between two key stages: (1) Planning Agent Alignment, which optimizes the planning agent to generate subgoal sequences that better guide the grounding agent; and (2) Grounding Agent Improving, which fine-tunes the grounding agent using diverse subgoal-action pairs generated by the agent itself to enhance its generalization capablity. Theoretical analysis proves that MOAT ensures a non-decreasing and progressively convergent training process. Experiments across six benchmarks demonstrate that MOAT outperforms state-of-the-art baselines, achieving average improvements of 3.1% on held-in tasks and 4.4% on held-out tasks. 

---
# LAVA: Language Model Assisted Verbal Autopsy for Cause-of-Death Determination 

**Authors**: Yiqun T. Chen, Tyler H. McCormick, Li Liu, Abhirup Datta  

**Link**: [PDF](https://arxiv.org/pdf/2509.09602)  

**Abstract**: Verbal autopsy (VA) is a critical tool for estimating causes of death in resource-limited settings where medical certification is unavailable. This study presents LA-VA, a proof-of-concept pipeline that combines Large Language Models (LLMs) with traditional algorithmic approaches and embedding-based classification for improved cause-of-death prediction. Using the Population Health Metrics Research Consortium (PHMRC) dataset across three age categories (Adult: 7,580; Child: 1,960; Neonate: 2,438), we evaluate multiple approaches: GPT-5 predictions, LCVA baseline, text embeddings, and meta-learner ensembles. Our results demonstrate that GPT-5 achieves the highest individual performance with average test site accuracies of 48.6% (Adult), 50.5% (Child), and 53.5% (Neonate), outperforming traditional statistical machine learning baselines by 5-10%. Our findings suggest that simple off-the-shelf LLM-assisted approaches could substantially improve verbal autopsy accuracy, with important implications for global health surveillance in low-resource settings. 

---
# Fluent but Unfeeling: The Emotional Blind Spots of Language Models 

**Authors**: Bangzhao Shu, Isha Joshi, Melissa Karnaze, Anh C. Pham, Ishita Kakkar, Sindhu Kothe, Arpine Hovasapian, Mai ElSherief  

**Link**: [PDF](https://arxiv.org/pdf/2509.09593)  

**Abstract**: The versatility of Large Language Models (LLMs) in natural language understanding has made them increasingly popular in mental health research. While many studies explore LLMs' capabilities in emotion recognition, a critical gap remains in evaluating whether LLMs align with human emotions at a fine-grained level. Existing research typically focuses on classifying emotions into predefined, limited categories, overlooking more nuanced expressions. To address this gap, we introduce EXPRESS, a benchmark dataset curated from Reddit communities featuring 251 fine-grained, self-disclosed emotion labels. Our comprehensive evaluation framework examines predicted emotion terms and decomposes them into eight basic emotions using established emotion theories, enabling a fine-grained comparison. Systematic testing of prevalent LLMs under various prompt settings reveals that accurately predicting emotions that align with human self-disclosed emotions remains challenging. Qualitative analysis further shows that while certain LLMs generate emotion terms consistent with established emotion theories and definitions, they sometimes fail to capture contextual cues as effectively as human self-disclosures. These findings highlight the limitations of LLMs in fine-grained emotion alignment and offer insights for future research aimed at enhancing their contextual understanding. 

---
# Personality-Enhanced Social Recommendations in SAMI: Exploring the Role of Personality Detection in Matchmaking 

**Authors**: Brittany Harbison, Samuel Taubman, Travis Taylor, Ashok. K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2509.09583)  

**Abstract**: Social connection is a vital part of learning, yet online course environments present barriers to the organic formation of social groups. SAMI offers one solution by facilitating student connections, but its effectiveness is constrained by an incomplete Theory of Mind, limiting its ability to create an effective mental model of a student. One facet of this is its inability to intuit personality, which may influence the relevance of its recommendations. To explore this, we propose a personality detection model utilizing GPTs zero-shot capability to infer Big-Five personality traits from forum introduction posts, often encouraged in online courses. We benchmark its performance against established models, demonstrating its efficacy in this task. Furthermore, we integrate this model into SAMIs entity-based matchmaking system, enabling personality-informed social recommendations. Initial integration suggests personality traits can complement existing matching factors, though additional evaluation is required to determine their full impact on student engagement and match quality. 

---
# Prompting the Market? A Large-Scale Meta-Analysis of GenAI in Finance NLP (2022-2025) 

**Authors**: Paolo Pedinotti, Peter Baumann, Nathan Jessurun, Leslie Barrett, Enrico Santus  

**Link**: [PDF](https://arxiv.org/pdf/2509.09544)  

**Abstract**: Large Language Models (LLMs) have rapidly reshaped financial NLP, enabling new tasks and driving a proliferation of datasets and diversification of data sources. Yet, this transformation has outpaced traditional surveys. In this paper, we present MetaGraph, a generalizable methodology for extracting knowledge graphs from scientific literature and analyzing them to obtain a structured, queryable view of research trends. We define an ontology for financial NLP research and apply an LLM-based extraction pipeline to 681 papers (2022-2025), enabling large-scale, data-driven analysis. MetaGraph reveals three key phases: early LLM adoption and task/dataset innovation; critical reflection on LLM limitations; and growing integration of peripheral techniques into modular systems. This structured view offers both practitioners and researchers a clear understanding of how financial NLP has evolved - highlighting emerging trends, shifting priorities, and methodological shifts-while also demonstrating a reusable approach for mapping scientific progress in other domains. 

---
# DeMeVa at LeWiDi-2025: Modeling Perspectives with In-Context Learning and Label Distribution Learning 

**Authors**: Daniil Ignatev, Nan Li, Hugh Mee Wong, Anh Dang, Shane Kaszefski Yaschuk  

**Link**: [PDF](https://arxiv.org/pdf/2509.09524)  

**Abstract**: This system paper presents the DeMeVa team's approaches to the third edition of the Learning with Disagreements shared task (LeWiDi 2025; Leonardelli et al., 2025). We explore two directions: in-context learning (ICL) with large language models, where we compare example sampling strategies; and label distribution learning (LDL) methods with RoBERTa (Liu et al., 2019b), where we evaluate several fine-tuning methods. Our contributions are twofold: (1) we show that ICL can effectively predict annotator-specific annotations (perspectivist annotations), and that aggregating these predictions into soft labels yields competitive performance; and (2) we argue that LDL methods are promising for soft label predictions and merit further exploration by the perspectivist community. 

---
# Towards Explainable Job Title Matching: Leveraging Semantic Textual Relatedness and Knowledge Graphs 

**Authors**: Vadim Zadykian, Bruno Andrade, Haithem Afli  

**Link**: [PDF](https://arxiv.org/pdf/2509.09522)  

**Abstract**: Semantic Textual Relatedness (STR) captures nuanced relationships between texts that extend beyond superficial lexical similarity. In this study, we investigate STR in the context of job title matching - a key challenge in resume recommendation systems, where overlapping terms are often limited or misleading. We introduce a self-supervised hybrid architecture that combines dense sentence embeddings with domain-specific Knowledge Graphs (KGs) to improve both semantic alignment and explainability. Unlike previous work that evaluated models on aggregate performance, our approach emphasizes data stratification by partitioning the STR score continuum into distinct regions: low, medium, and high semantic relatedness. This stratified evaluation enables a fine-grained analysis of model performance across semantically meaningful subspaces. We evaluate several embedding models, both with and without KG integration via graph neural networks. The results show that fine-tuned SBERT models augmented with KGs produce consistent improvements in the high-STR region, where the RMSE is reduced by 25% over strong baselines. Our findings highlight not only the benefits of combining KGs with text embeddings, but also the importance of regional performance analysis in understanding model behavior. This granular approach reveals strengths and weaknesses hidden by global metrics, and supports more targeted model selection for use in Human Resources (HR) systems and applications where fairness, explainability, and contextual matching are essential. 

---
# Mitigating Language Barriers in Education: Developing Multilingual Digital Learning Materials with Machine Translation 

**Authors**: Lucie Poláková, Martin Popel, Věra Kloudová, Michal Novák, Mariia Anisimova, Jiří Balhar  

**Link**: [PDF](https://arxiv.org/pdf/2509.09473)  

**Abstract**: The EdUKate project combines digital education, linguistics, translation studies, and machine translation to develop multilingual learning materials for Czech primary and secondary schools. Launched through collaboration between a major Czech academic institution and the country's largest educational publisher, the project is aimed at translating up to 9,000 multimodal interactive exercises from Czech into Ukrainian, English, and German for an educational web portal. It emphasizes the development and evaluation of a direct Czech-Ukrainian machine translation system tailored to the educational domain, with special attention to processing formatted content such as XML and PDF and handling technical and scientific terminology. We present findings from an initial survey of Czech teachers regarding the needs of non-Czech-speaking students and describe the system's evaluation and implementation on the web portal. All resulting applications are freely available to students, educators, and researchers. 

---
# GrACE: A Generative Approach to Better Confidence Elicitation in Large Language Models 

**Authors**: Zhaohan Zhang, Ziquan Liu, Ioannis Patras  

**Link**: [PDF](https://arxiv.org/pdf/2509.09438)  

**Abstract**: Assessing the reliability of Large Language Models (LLMs) by confidence elicitation is a prominent approach to AI safety in high-stakes applications, such as healthcare and finance. Existing methods either require expensive computational overhead or suffer from poor calibration, making them impractical and unreliable for real-world deployment. In this work, we propose GrACE, a Generative Approach to Confidence Elicitation that enables scalable and reliable confidence elicitation for LLMs. GrACE adopts a novel mechanism in which the model expresses confidence by the similarity between the last hidden state and the embedding of a special token appended to the vocabulary, in real-time. We fine-tune the model for calibrating the confidence with calibration targets associated with accuracy. Experiments with three LLMs and two benchmark datasets show that the confidence produced by GrACE achieves the best discriminative capacity and calibration on open-ended generation tasks, outperforming six competing methods without resorting to additional sampling or an auxiliary model. Moreover, we propose two strategies for improving test-time scaling based on confidence induced by GrACE. Experimental results show that using GrACE not only improves the accuracy of the final decision but also significantly reduces the number of required samples in the test-time scaling scheme, indicating the potential of GrACE as a practical solution for deploying LLMs with scalable, reliable, and real-time confidence estimation. 

---
# Hierarchical Bracketing Encodings Work for Dependency Graphs 

**Authors**: Ana Ezquerro, Carlos Gómez-Rodríguez, David Vilares  

**Link**: [PDF](https://arxiv.org/pdf/2509.09388)  

**Abstract**: We revisit hierarchical bracketing encodings from a practical perspective in the context of dependency graph parsing. The approach encodes graphs as sequences, enabling linear-time parsing with $n$ tagging actions, and still representing reentrancies, cycles, and empty nodes. Compared to existing graph linearizations, this representation substantially reduces the label space while preserving structural information. We evaluate it on a multilingual and multi-formalism benchmark, showing competitive results and consistent improvements over other methods in exact match accuracy. 

---
# Modelling Analogies and Analogical Reasoning: Connecting Cognitive Science Theory and NLP Research 

**Authors**: Molly R Petersen, Claire E Stevenson, Lonneke van der Plas  

**Link**: [PDF](https://arxiv.org/pdf/2509.09381)  

**Abstract**: Analogical reasoning is an essential aspect of human cognition. In this paper, we summarize key theory about the processes underlying analogical reasoning from the cognitive science literature and relate it to current research in natural language processing. While these processes can be easily linked to concepts in NLP, they are generally not viewed through a cognitive lens. Furthermore, we show how these notions are relevant for several major challenges in NLP research, not directly related to analogy solving. This may guide researchers to better optimize relational understanding in text, as opposed to relying heavily on entity-level similarity. 

---
# MetaRAG: Metamorphic Testing for Hallucination Detection in RAG Systems 

**Authors**: Channdeth Sok, David Luz, Yacine Haddam  

**Link**: [PDF](https://arxiv.org/pdf/2509.09360)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in enterprise applications, yet their reliability remains limited by hallucinations, i.e., confident but factually incorrect information. Existing detection approaches, such as SelfCheckGPT and MetaQA, primarily target standalone LLMs and do not address the unique challenges of Retrieval-Augmented Generation (RAG) systems, where responses must be consistent with retrieved evidence. We therefore present MetaRAG, a metamorphic testing framework for hallucination detection in Retrieval-Augmented Generation (RAG) systems. MetaRAG operates in a real-time, unsupervised, black-box setting, requiring neither ground-truth references nor access to model internals, making it suitable for proprietary and high-stakes domains. The framework proceeds in four stages: (1) decompose answers into atomic factoids, (2) generate controlled mutations of each factoid using synonym and antonym substitutions, (3) verify each variant against the retrieved context (synonyms are expected to be entailed and antonyms contradicted), and (4) aggregate penalties for inconsistencies into a response-level hallucination score. Crucially for identity-aware AI, MetaRAG localizes unsupported claims at the factoid span where they occur (e.g., pregnancy-specific precautions, LGBTQ+ refugee rights, or labor eligibility), allowing users to see flagged spans and enabling system designers to configure thresholds and guardrails for identity-sensitive queries. Experiments on a proprietary enterprise dataset illustrate the effectiveness of MetaRAG for detecting hallucinations and enabling trustworthy deployment of RAG-based conversational agents. We also outline a topic-based deployment design that translates MetaRAG's span-level scores into identity-aware safeguards; this design is discussed but not evaluated in our experiments. 

---
# From scratch to silver: Creating trustworthy training data for patent-SDG classification using Large Language Models 

**Authors**: Grazia Sveva Ascione, Nicolò Tamagnone  

**Link**: [PDF](https://arxiv.org/pdf/2509.09303)  

**Abstract**: Classifying patents by their relevance to the UN Sustainable Development Goals (SDGs) is crucial for tracking how innovation addresses global challenges. However, the absence of a large, labeled dataset limits the use of supervised learning. Existing methods, such as keyword searches, transfer learning, and citation-based heuristics, lack scalability and generalizability. This paper frames patent-to-SDG classification as a weak supervision problem, using citations from patents to SDG-tagged scientific publications (NPL citations) as a noisy initial signal. To address its sparsity and noise, we develop a composite labeling function (LF) that uses large language models (LLMs) to extract structured concepts, namely functions, solutions, and applications, from patents and SDG papers based on a patent ontology. Cross-domain similarity scores are computed and combined using a rank-based retrieval approach. The LF is calibrated via a custom positive-only loss that aligns with known NPL-SDG links without penalizing discovery of new SDG associations. The result is a silver-standard, soft multi-label dataset mapping patents to SDGs, enabling the training of effective multi-label regression models. We validate our approach through two complementary strategies: (1) internal validation against held-out NPL-based labels, where our method outperforms several baselines including transformer-based models, and zero-shot LLM; and (2) external validation using network modularity in patent citation, co-inventor, and co-applicant graphs, where our labels reveal greater thematic, cognitive, and organizational coherence than traditional technological classifications. These results show that weak supervision and semantic alignment can enhance SDG classification at scale. 

---
# Agentic LLMs for Question Answering over Tabular Data 

**Authors**: Rishit Tyagi, Mohit Gupta, Rahul Bouri  

**Link**: [PDF](https://arxiv.org/pdf/2509.09234)  

**Abstract**: Question Answering over Tabular Data (Table QA) presents unique challenges due to the diverse structure, size, and data types of real-world tables. The SemEval 2025 Task 8 (DataBench) introduced a benchmark composed of large-scale, domain-diverse datasets to evaluate the ability of models to accurately answer structured queries. We propose a Natural Language to SQL (NL-to-SQL) approach leveraging large language models (LLMs) such as GPT-4o, GPT-4o-mini, and DeepSeek v2:16b to generate SQL queries dynamically. Our system follows a multi-stage pipeline involving example selection, SQL query generation, answer extraction, verification, and iterative refinement. Experiments demonstrate the effectiveness of our approach, achieving 70.5\% accuracy on DataBench QA and 71.6\% on DataBench Lite QA, significantly surpassing baseline scores of 26\% and 27\% respectively. This paper details our methodology, experimental results, and alternative approaches, providing insights into the strengths and limitations of LLM-driven Table QA. 

---
# Reading Between the Lines: Classifying Resume Seniority with Large Language Models 

**Authors**: Matan Cohen, Shira Shani, Eden Menahem, Yehudit Aperstein, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2509.09229)  

**Abstract**: Accurately assessing candidate seniority from resumes is a critical yet challenging task, complicated by the prevalence of overstated experience and ambiguous self-presentation. In this study, we investigate the effectiveness of large language models (LLMs), including fine-tuned BERT architectures, for automating seniority classification in resumes. To rigorously evaluate model performance, we introduce a hybrid dataset comprising both real-world resumes and synthetically generated hard examples designed to simulate exaggerated qualifications and understated seniority. Using the dataset, we evaluate the performance of Large Language Models in detecting subtle linguistic cues associated with seniority inflation and implicit expertise. Our findings highlight promising directions for enhancing AI-driven candidate evaluation systems and mitigating bias introduced by self-promotional language. The dataset is available for the research community at this https URL 

---
# CCF: A Context Compression Framework for Efficient Long-Sequence Language Modeling 

**Authors**: Wenhao Li, Bangcheng Sun, Weihao Ye, Tianyi Zhang, Daohai Yu, Fei Chao, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.09199)  

**Abstract**: Scaling language models to longer contexts is essential for capturing rich dependencies across extended discourse. However, naïve context extension imposes significant computational and memory burdens, often resulting in inefficiencies during both training and inference. In this work, we propose CCF, a novel context compression framework designed to enable efficient long-context modeling by learning hierarchical latent representations that preserve global semantics while aggressively reducing input redundancy. CCF integrates segment-wise semantic aggregation with key-value memory encoding, forming compact representations that support accurate reconstruction and long-range understanding. To further enhance scalability, we introduce a training-efficient optimization strategy that couples incremental segment decoding with sparse reservoir sampling, substantially reducing memory overhead without degrading performance. Empirical results on multiple long-context language modeling benchmarks demonstrate that CCF achieves competitive perplexity under high compression ratios, and significantly improves throughput and memory efficiency compared to existing approaches. These findings highlight the potential of structured compression for scalable and effective long-context language modeling. 

---
# GmSLM : Generative Marmoset Spoken Language Modeling 

**Authors**: Talia Sternberg, Michael London, David Omer, Yossi Adi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09198)  

**Abstract**: Marmoset monkeys exhibit complex vocal communication, challenging the view that nonhuman primates vocal communication is entirely innate, and show similar features of human speech, such as vocal labeling of others and turn-taking. Studying their vocal communication offers a unique opportunity to link it with brain activity-especially given the difficulty of accessing the human brain in speech and language research. Since Marmosets communicate primarily through vocalizations, applying standard LLM approaches is not straightforward. We introduce Generative Marmoset Spoken Language Modeling (GmSLM), an optimized spoken language model pipeline for Marmoset vocal communication. We designed a novel zero-shot evaluation metrics using unsupervised in-the-wild data, alongside weakly labeled conversational data, to assess GmSLM and demonstrate its advantage over a basic human-speech-based baseline. GmSLM generated vocalizations closely matched real resynthesized samples acoustically and performed well on downstream tasks. Despite being fully unsupervised, GmSLM effectively distinguish real from artificial conversations and may support further investigations of the neural basis of vocal communication and provides a practical framework linking vocalization and brain activity. We believe GmSLM stands to benefit future work in neuroscience, bioacoustics, and evolutionary biology. Samples are provided under: this http URL. 

---
# Improving Synthetic Data Training for Contextual Biasing Models with a Keyword-Aware Cost Function 

**Authors**: Chin Yuen Kwok, Jia Qi Yip, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2509.09197)  

**Abstract**: Rare word recognition can be improved by adapting ASR models to synthetic data that includes these words. Further improvements can be achieved through contextual biasing, which trains and adds a biasing module into the model architecture to prioritize rare words. While training the module on synthetic rare word data is more effective than using non-rare-word data, it can lead to overfitting due to artifacts in the synthetic audio. To address this, we enhance the TCPGen-based contextual biasing approach and propose a keyword-aware loss function that additionally focuses on biased words when training biasing modules. This loss includes a masked cross-entropy term for biased word prediction and a binary classification term for detecting biased word positions. These two terms complementarily support the decoding of biased words during inference. By adapting Whisper to 10 hours of synthetic data, our method reduced the word error rate on the NSC Part 2 test set from 29.71% to 11.81%. 

---
# Efficient Trie-based Biasing using K-step Prediction for Rare Word Recognition 

**Authors**: Chin Yuen Kwok, Jia Qi yip  

**Link**: [PDF](https://arxiv.org/pdf/2509.09196)  

**Abstract**: Contextual biasing improves rare word recognition of ASR models by prioritizing the output of rare words during decoding. A common approach is Trie-based biasing, which gives "bonus scores" to partial hypothesis (e.g. "Bon") that may lead to the generation of the rare word (e.g. "Bonham"). If the full word ("Bonham") isn't ultimately recognized, the system revokes those earlier bonuses. This revocation is limited to beam search and is computationally expensive, particularly for models with large decoders. To overcome these limitations, we propose adapting ASR models to look ahead and predict multiple steps at once. This avoids the revocation step entirely by better estimating whether a partial hypothesis will lead to the generation of the full rare word. By fine-tuning Whisper with only 10 hours of synthetic data, our method reduces the word error rate on the NSC Part 2 test set from 30.86% to 12.19%. 

---
# EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs 

**Authors**: Yuhao Zhang, Yuhao Du, Zhanchen Dai, Xiangnan Ma, Kaiqi Kou, Benyou Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.09174)  

**Abstract**: Speech-to-speech large language models (SLLMs) are attracting increasing attention. Derived from text-based large language models (LLMs), SLLMs often exhibit degradation in knowledge and reasoning capabilities. We hypothesize that this limitation arises because current training paradigms for SLLMs fail to bridge the acoustic-semantic gap in the feature representation space. To address this issue, we propose EchoX, which leverages semantic representations and dynamically generates speech training targets. This approach integrates both acoustic and semantic learning, enabling EchoX to preserve strong reasoning abilities as a speech LLM. Experimental results demonstrate that EchoX, with about six thousand hours of training data, achieves advanced performance on multiple knowledge-based question-answering benchmarks. The project is available at this https URL. 

---
# Target-oriented Multimodal Sentiment Classification with Counterfactual-enhanced Debiasing 

**Authors**: Zhiyue Liu, Fanrong Ma, Xin Ling  

**Link**: [PDF](https://arxiv.org/pdf/2509.09160)  

**Abstract**: Target-oriented multimodal sentiment classification seeks to predict sentiment polarity for specific targets from image-text pairs. While existing works achieve competitive performance, they often over-rely on textual content and fail to consider dataset biases, in particular word-level contextual biases. This leads to spurious correlations between text features and output labels, impairing classification accuracy. In this paper, we introduce a novel counterfactual-enhanced debiasing framework to reduce such spurious correlations. Our framework incorporates a counterfactual data augmentation strategy that minimally alters sentiment-related causal features, generating detail-matched image-text samples to guide the model's attention toward content tied to sentiment. Furthermore, for learning robust features from counterfactual data and prompting model decisions, we introduce an adaptive debiasing contrastive learning mechanism, which effectively mitigates the influence of biased words. Experimental results on several benchmark datasets show that our proposed method outperforms state-of-the-art baselines. 

---
# LITcoder: A General-Purpose Library for Building and Comparing Encoding Models 

**Authors**: Taha Binhuraib, Ruimin Gao, Anna A. Ivanova  

**Link**: [PDF](https://arxiv.org/pdf/2509.09152)  

**Abstract**: We introduce LITcoder, an open-source library for building and benchmarking neural encoding models. Designed as a flexible backend, LITcoder provides standardized tools for aligning continuous stimuli (e.g., text and speech) with brain data, transforming stimuli into representational features, mapping those features onto brain data, and evaluating the predictive performance of the resulting model on held-out data. The library implements a modular pipeline covering a wide array of methodological design choices, so researchers can easily compose, compare, and extend encoding models without reinventing core infrastructure. Such choices include brain datasets, brain regions, stimulus feature (both neural-net-based and control, such as word rate), downsampling approaches, and many others. In addition, the library provides built-in logging, plotting, and seamless integration with experiment tracking platforms such as Weights & Biases (W&B). We demonstrate the scalability and versatility of our framework by fitting a range of encoding models to three story listening datasets: LeBel et al. (2023), Narratives, and Little Prince. We also explore the methodological choices critical for building encoding models for continuous fMRI data, illustrating the importance of accounting for all tokens in a TR scan (as opposed to just taking the last one, even when contextualized), incorporating hemodynamic lag effects, using train-test splits that minimize information leakage, and accounting for head motion effects on encoding model predictivity. Overall, LITcoder lowers technical barriers to encoding model implementation, facilitates systematic comparisons across models and datasets, fosters methodological rigor, and accelerates the development of high-quality high-performance predictive models of brain activity.
Project page: this https URL 

---
# ViRanker: A BGE-M3 & Blockwise Parallel Transformer Cross-Encoder for Vietnamese Reranking 

**Authors**: Phuong-Nam Dang, Kieu-Linh Nguyen, Thanh-Hieu Pham  

**Link**: [PDF](https://arxiv.org/pdf/2509.09131)  

**Abstract**: This paper presents ViRanker, a cross-encoder reranking model tailored to the Vietnamese language. Built on the BGE-M3 encoder and enhanced with the Blockwise Parallel Transformer, ViRanker addresses the lack of competitive rerankers for Vietnamese, a low-resource language with complex syntax and diacritics. The model was trained on an 8 GB curated corpus and fine-tuned with hybrid hard-negative sampling to strengthen robustness. Evaluated on the MMARCO-VI benchmark, ViRanker achieves strong early-rank accuracy, surpassing multilingual baselines and competing closely with PhoRanker. By releasing the model openly on Hugging Face, we aim to support reproducibility and encourage wider adoption in real-world retrieval systems. Beyond Vietnamese, this study illustrates how careful architectural adaptation and data curation can advance reranking in other underrepresented languages. 

---
# Automated Classification of Tutors' Dialogue Acts Using Generative AI: A Case Study Using the CIMA Corpus 

**Authors**: Liqun He, Jiaqi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09125)  

**Abstract**: This study explores the use of generative AI for automating the classification of tutors' Dialogue Acts (DAs), aiming to reduce the time and effort required by traditional manual coding. This case study uses the open-source CIMA corpus, in which tutors' responses are pre-annotated into four DA categories. Both GPT-3.5-turbo and GPT-4 models were tested using tailored prompts. Results show that GPT-4 achieved 80% accuracy, a weighted F1-score of 0.81, and a Cohen's Kappa of 0.74, surpassing baseline performance and indicating substantial agreement with human annotations. These findings suggest that generative AI has strong potential to provide an efficient and accessible approach to DA classification, with meaningful implications for educational dialogue analysis. The study also highlights the importance of task-specific label definitions and contextual information in enhancing the quality of automated annotation. Finally, it underscores the ethical considerations associated with the use of generative AI and the need for responsible and transparent research practices. The script of this research is publicly available at this https URL. 

---
# Compass-v3: Scaling Domain-Specific LLMs for Multilingual E-Commerce in Southeast Asia 

**Authors**: Sophia Maria  

**Link**: [PDF](https://arxiv.org/pdf/2509.09121)  

**Abstract**: Large language models (LLMs) excel in general-domain applications, yet their performance often degrades in specialized tasks requiring domain-specific knowledge. E-commerce is particularly challenging, as its data are noisy, heterogeneous, multilingual, and highly dynamic. We present Compass-v3, a vertical-domain Mixture-of-Experts (MoE) model with 245B total parameters and 71B active per token, designed for Southeast Asian e-commerce. Compass-v3 adopts fewer but larger experts, combined with hardware-efficient optimizations-such as intra-node expert parallelism and a customized memcpy operator-to maximize GPU utilization. The model is trained on 12T tokens of curated multilingual corpora and large-scale synthetic e-commerce instructions using a mixed-training strategy. To enhance alignment, we propose Optimal-Transport Direct Preference Optimization (OTPO), which captures token-level distinctions and improves instruction adherence in commerce-specific scenarios. Extensive evaluations demonstrate that Compass-v3 delivers state-of-the-art e-commerce performance, surpassing DeepSeek-V3.1, GPT-4 series, and Qwen3-235B. Moreover, Compass-v3 demonstrates strong multilingual capability across low-resource Southeast Asian languages (Indonesian, Thai, Filipino, Vietnamese, Malay, Taglog) and Portuguese while sustaining competitive performance on general benchmarks. It has already been widely applied in Shopee's industrial-scale e-commerce platform and is gradually replacing OpenAI's traffic, now accounting for over 70\% of total LLM usage, highlighting its dual strengths in specialized commerce expertise and broad linguistic competence. 

---
# TigerCoder: A Novel Suite of LLMs for Code Generation in Bangla 

**Authors**: Nishat Raihan, Antonios Anastasopoulos, Marcos Zampieri  

**Link**: [PDF](https://arxiv.org/pdf/2509.09101)  

**Abstract**: Despite being the 5th most spoken language, Bangla remains underrepresented in Large Language Models (LLMs), particularly for code generation. This primarily stems from the scarcity of high-quality data to pre-train and/or finetune such models. Hence, we introduce the first dedicated family of Code LLMs for Bangla (1B & 9B). We offer three major contributions: (1) a comprehensive Bangla code instruction datasets for programming domain adaptation; (2) MBPP-Bangla, an evaluation benchmark for Bangla code generation; and (3) the TigerCoder-family of Code LLMs, achieving significant ~11-18% performance gains at Pass@1 over existing multilingual and general-purpose Bangla LLMs. Our findings show that curated, high-quality datasets can overcome limitations of smaller models for low-resource languages. We open-source all resources to advance further Bangla LLM research. 

---
# MR-UIE: Multi-Perspective Reasoning with Reinforcement Learning for Universal Information Extraction 

**Authors**: Zhongqiu Li, Shiquan Wang, Ruiyu Fang, Mengjiao Bao, Zhenhe Wu, Shuangyong Song, Yongxiang Li, Zhongjiang He  

**Link**: [PDF](https://arxiv.org/pdf/2509.09082)  

**Abstract**: Large language models (LLMs) demonstrate robust capabilities across diverse research domains. However, their performance in universal information extraction (UIE) remains insufficient, especially when tackling structured output scenarios that involve complex schema descriptions and require multi-step reasoning. While existing approaches enhance the performance of LLMs through in-context learning and instruction tuning, significant limitations nonetheless persist. To enhance the model's generalization ability, we propose integrating reinforcement learning (RL) with multi-perspective reasoning for information extraction (IE) tasks. Our work transitions LLMs from passive extractors to active reasoners, enabling them to understand not only what to extract but also how to reason. Experiments conducted on multiple IE benchmarks demonstrate that MR-UIE consistently elevates extraction accuracy across domains and surpasses state-of-the-art methods on several datasets. Furthermore, incorporating multi-perspective reasoning into RL notably enhances generalization in complex IE tasks, underscoring the critical role of reasoning in challenging scenarios. 

---
# Improving LLM Safety and Helpfulness using SFT and DPO: A Study on OPT-350M 

**Authors**: Piyush Pant  

**Link**: [PDF](https://arxiv.org/pdf/2509.09055)  

**Abstract**: This research investigates the effectiveness of alignment techniques, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and a combined SFT+DPO approach on improving the safety and helpfulness of the OPT-350M language model. Utilizing the Anthropic Helpful-Harmless RLHF dataset, we train and evaluate four models: the base OPT350M, an SFT model, a DPO model, and a model trained with both SFT and DPO. We introduce three key evaluation metrics: Harmlessness Rate (HmR), Helpfulness Rate (HpR), and a Combined Alignment Score (CAS), all derived from reward model outputs. The results show that while SFT outperforms DPO, The combined SFT+DPO model outperforms all others across all metrics, demonstrating the complementary nature of these techniques. Our findings also highlight challenges posed by noisy data, limited GPU resources, and training constraints. This study offers a comprehensive view of how fine-tuning strategies affect model alignment and provides a foundation for more robust alignment pipelines in future work. 

---
# Stated Preference for Interaction and Continued Engagement (SPICE): Evaluating an LLM's Willingness to Re-engage in Conversation 

**Authors**: Thomas Manuel Rost, Martina Figlia, Bernd Wallraff  

**Link**: [PDF](https://arxiv.org/pdf/2509.09043)  

**Abstract**: We introduce and evaluate Stated Preference for Interaction and Continued Engagement (SPICE), a simple diagnostic signal elicited by asking a Large Language Model a YES or NO question about its willingness to re-engage with a user's behavior after reviewing a short transcript. In a study using a 3-tone (friendly, unclear, abusive) by 10-interaction stimulus set, we tested four open-weight chat models across four framing conditions, resulting in 480 trials. Our findings show that SPICE sharply discriminates by user tone. Friendly interactions yielded a near-unanimous preference to continue (97.5% YES), while abusive interactions yielded a strong preference to discontinue (17.9% YES), with unclear interactions falling in between (60.4% YES). This core association remains decisive under multiple dependence-aware statistical tests, including Rao-Scott adjustment and cluster permutation tests. Furthermore, we demonstrate that SPICE provides a distinct signal from abuse classification. In trials where a model failed to identify abuse, it still overwhelmingly stated a preference not to continue the interaction (81% of the time). An exploratory analysis also reveals a significant interaction effect: a preamble describing the study context significantly impacts SPICE under ambiguity, but only when transcripts are presented as a single block of text rather than a multi-turn chat. The results validate SPICE as a robust, low-overhead, and reproducible tool for auditing model dispositions, complementing existing metrics by offering a direct, relational signal of a model's state. All stimuli, code, and analysis scripts are released to support replication. 

---
# Can Vision-Language Models Solve Visual Math Equations? 

**Authors**: Monjoy Narayan Choudhury, Junling Wang, Yifan Hou, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09013)  

**Abstract**: Despite strong performance in visual understanding and language-based reasoning, Vision-Language Models (VLMs) struggle with tasks requiring integrated perception and symbolic computation. We study this limitation through visual equation solving, where mathematical equations are embedded in images, variables are represented by object icons, and coefficients must be inferred by counting. While VLMs perform well on textual equations, they fail on visually grounded counterparts. To understand this gap, we decompose the task into coefficient counting and variable recognition, and find that counting is the primary bottleneck, even when recognition is accurate. We also observe that composing recognition and reasoning introduces additional errors, highlighting challenges in multi-step visual reasoning. Finally, as equation complexity increases, symbolic reasoning itself becomes a limiting factor. These findings reveal key weaknesses in current VLMs and point toward future improvements in visually grounded mathematical reasoning. 

---
# BRoverbs -- Measuring how much LLMs understand Portuguese proverbs 

**Authors**: Thales Sales Almeida, Giovana Kerche Bonás, João Guilherme Alves Santos  

**Link**: [PDF](https://arxiv.org/pdf/2509.08960)  

**Abstract**: Large Language Models (LLMs) exhibit significant performance variations depending on the linguistic and cultural context in which they are applied. This disparity signals the necessity of mature evaluation frameworks that can assess their capabilities in specific regional settings. In the case of Portuguese, existing evaluations remain limited, often relying on translated datasets that may not fully capture linguistic nuances or cultural references. Meanwhile, native Portuguese-language datasets predominantly focus on structured national exams or sentiment analysis of social media interactions, leaving gaps in evaluating broader linguistic understanding. To address this limitation, we introduce BRoverbs, a dataset specifically designed to assess LLM performance through Brazilian proverbs. Proverbs serve as a rich linguistic resource, encapsulating cultural wisdom, figurative expressions, and complex syntactic structures that challenge the model comprehension of regional expressions. BRoverbs aims to provide a new evaluation tool for Portuguese-language LLMs, contributing to advancing regionally informed benchmarking. The benchmark is available at this https URL. 

---
# Documents Are People and Words Are Items: A Psychometric Approach to Textual Data with Contextual Embeddings 

**Authors**: Jinsong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.08920)  

**Abstract**: This research introduces a novel psychometric method for analyzing textual data using large language models. By leveraging contextual embeddings to create contextual scores, we transform textual data into response data suitable for psychometric analysis. Treating documents as individuals and words as items, this approach provides a natural psychometric interpretation under the assumption that certain keywords, whose contextual meanings vary significantly across documents, can effectively differentiate documents within a corpus. The modeling process comprises two stages: obtaining contextual scores and performing psychometric analysis. In the first stage, we utilize natural language processing techniques and encoder based transformer models to identify common keywords and generate contextual scores. In the second stage, we employ various types of factor analysis, including exploratory and bifactor models, to extract and define latent factors, determine factor correlations, and identify the most significant words associated with each factor. Applied to the Wiki STEM corpus, our experimental results demonstrate the method's potential to uncover latent knowledge dimensions and patterns within textual data. This approach not only enhances the psychometric analysis of textual data but also holds promise for applications in fields rich in textual information, such as education, psychology, and law. 

---
# Automated Evidence Extraction and Scoring for Corporate Climate Policy Engagement: A Multilingual RAG Approach 

**Authors**: Imene Kolli, Ario Saeid Vaghefi, Chiara Colesanti Senni, Shantam Raj, Markus Leippold  

**Link**: [PDF](https://arxiv.org/pdf/2509.08907)  

**Abstract**: InfluenceMap's LobbyMap Platform monitors the climate policy engagement of over 500 companies and 250 industry associations, assessing each entity's support or opposition to science-based policy pathways for achieving the Paris Agreement's goal of limiting global warming to 1.5°C. Although InfluenceMap has made progress with automating key elements of the analytical workflow, a significant portion of the assessment remains manual, making it time- and labor-intensive and susceptible to human error. We propose an AI-assisted framework to accelerate the monitoring of corporate climate policy engagement by leveraging Retrieval-Augmented Generation to automate the most time-intensive extraction of relevant evidence from large-scale textual data. Our evaluation shows that a combination of layout-aware parsing, the Nomic embedding model, and few-shot prompting strategies yields the best performance in extracting and classifying evidence from multilingual corporate documents. We conclude that while the automated RAG system effectively accelerates evidence extraction, the nuanced nature of the analysis necessitates a human-in-the-loop approach where the technology augments, rather than replaces, expert judgment to ensure accuracy. 

---
# Noise or Nuance: An Investigation Into Useful Information and Filtering For LLM Driven AKBC 

**Authors**: Alex Clay, Ernesto Jiménez-Ruiz, Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2509.08903)  

**Abstract**: RAG and fine-tuning are prevalent strategies for improving the quality of LLM outputs. However, in constrained situations, such as that of the 2025 LM-KBC challenge, such techniques are restricted. In this work we investigate three facets of the triple completion task: generation, quality assurance, and LLM response parsing. Our work finds that in this constrained setting: additional information improves generation quality, LLMs can be effective at filtering poor quality triples, and the tradeoff between flexibility and consistency with LLM response parsing is setting dependent. 

---
# FLUX-Reason-6M & PRISM-Bench: A Million-Scale Text-to-Image Reasoning Dataset and Comprehensive Benchmark 

**Authors**: Rongyao Fang, Aldrich Yu, Chengqi Duan, Linjiang Huang, Shuai Bai, Yuxuan Cai, Kun Wang, Si Liu, Xihui Liu, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.09680)  

**Abstract**: The advancement of open-source text-to-image (T2I) models has been hindered by the absence of large-scale, reasoning-focused datasets and comprehensive evaluation benchmarks, resulting in a performance gap compared to leading closed-source systems. To address this challenge, We introduce FLUX-Reason-6M and PRISM-Bench (Precise and Robust Image Synthesis Measurement Benchmark). FLUX-Reason-6M is a massive dataset consisting of 6 million high-quality FLUX-generated images and 20 million bilingual (English and Chinese) descriptions specifically designed to teach complex reasoning. The image are organized according to six key characteristics: Imagination, Entity, Text rendering, Style, Affection, and Composition, and design explicit Generation Chain-of-Thought (GCoT) to provide detailed breakdowns of image generation steps. The whole data curation takes 15,000 A100 GPU days, providing the community with a resource previously unattainable outside of large industrial labs. PRISM-Bench offers a novel evaluation standard with seven distinct tracks, including a formidable Long Text challenge using GCoT. Through carefully designed prompts, it utilizes advanced vision-language models for nuanced human-aligned assessment of prompt-image alignment and image aesthetics. Our extensive evaluation of 19 leading models on PRISM-Bench reveals critical performance gaps and highlights specific areas requiring improvement. Our dataset, benchmark, and evaluation code are released to catalyze the next wave of reasoning-oriented T2I generation. Project page: this https URL . 

---
# ButterflyQuant: Ultra-low-bit LLM Quantization through Learnable Orthogonal Butterfly Transforms 

**Authors**: Bingxin Xu, Zhen Dong, Oussama Elachqar, Yuzhang Shang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09679)  

**Abstract**: Large language models require massive memory footprints, severely limiting deployment on consumer hardware. Quantization reduces memory through lower numerical precision, but extreme 2-bit quantization suffers from catastrophic performance loss due to outliers in activations. Rotation-based methods such as QuIP and QuaRot apply orthogonal transforms to eliminate outliers before quantization, using computational invariance: $\mathbf{y} = \mathbf{Wx} = (\mathbf{WQ}^T)(\mathbf{Qx})$ for orthogonal $\mathbf{Q}$. However, these methods use fixed transforms--Hadamard matrices achieving optimal worst-case coherence $\mu = 1/\sqrt{n}$--that cannot adapt to specific weight distributions. We identify that different transformer layers exhibit distinct outlier patterns, motivating layer-adaptive rotations rather than one-size-fits-all approaches. We propose ButterflyQuant, which replaces Hadamard rotations with learnable butterfly transforms parameterized by continuous Givens rotation angles. Unlike Hadamard's discrete $\{+1, -1\}$ entries that are non-differentiable and prohibit gradient-based learning, butterfly transforms' continuous parameterization enables smooth optimization while guaranteeing orthogonality by construction. This orthogonal constraint ensures theoretical guarantees in outlier suppression while achieving $O(n \log n)$ computational complexity with only $\frac{n \log n}{2}$ learnable parameters. We further introduce a uniformity regularization on post-transformation activations to promote smoother distributions amenable to quantization. Learning requires only 128 calibration samples and converges in minutes on a single GPU--a negligible one-time cost. On LLaMA-2-7B with 2-bit quantization, ButterflyQuant achieves 15.4 perplexity versus 22.1 for QuaRot. 

---
# SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning 

**Authors**: Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.09674)  

**Abstract**: Vision-Language-Action (VLA) models have recently emerged as a powerful paradigm for robotic manipulation. Despite substantial progress enabled by large-scale pretraining and supervised fine-tuning (SFT), these models face two fundamental challenges: (i) the scarcity and high cost of large-scale human-operated robotic trajectories required for SFT scaling, and (ii) limited generalization to tasks involving distribution shift. Recent breakthroughs in Large Reasoning Models (LRMs) demonstrate that reinforcement learning (RL) can dramatically enhance step-by-step reasoning capabilities, raising a natural question: Can RL similarly improve the long-horizon step-by-step action planning of VLA? In this work, we introduce SimpleVLA-RL, an efficient RL framework tailored for VLA models. Building upon veRL, we introduce VLA-specific trajectory sampling, scalable parallelization, multi-environment rendering, and optimized loss computation. When applied to OpenVLA-OFT, SimpleVLA-RL achieves SoTA performance on LIBERO and even outperforms $\pi_0$ on RoboTwin 1.0\&2.0 with the exploration-enhancing strategies we introduce. SimpleVLA-RL not only reduces dependence on large-scale data and enables robust generalization, but also remarkably surpasses SFT in real-world tasks. Moreover, we identify a novel phenomenon ``pushcut'' during RL training, wherein the policy discovers previously unseen patterns beyond those seen in the previous training process. Github: this https URL 

---
# Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations 

**Authors**: Zakaria El Kassimi, Fares Fourati, Mohamed-Slim Alouini  

**Link**: [PDF](https://arxiv.org/pdf/2509.09651)  

**Abstract**: We study question answering in the domain of radio regulations, a legally sensitive and high-stakes area. We propose a telecom-specific Retrieval-Augmented Generation (RAG) pipeline and introduce, to our knowledge, the first multiple-choice evaluation set for this domain, constructed from authoritative sources using automated filtering and human validation. To assess retrieval quality, we define a domain-specific retrieval metric, under which our retriever achieves approximately 97% accuracy. Beyond retrieval, our approach consistently improves generation accuracy across all tested models. In particular, while naively inserting documents without structured retrieval yields only marginal gains for GPT-4o (less than 1%), applying our pipeline results in nearly a 12% relative improvement. These findings demonstrate that carefully targeted grounding provides a simple yet strong baseline and an effective domain-specific solution for regulatory question answering. All code and evaluation scripts, along with our derived question-answer dataset, are available at this https URL. 

---
# DiFlow-TTS: Discrete Flow Matching with Factorized Speech Tokens for Low-Latency Zero-Shot Text-To-Speech 

**Authors**: Ngoc-Son Nguyen, Hieu-Nghia Huynh-Nguyen, Thanh V. T. Tran, Truong-Son Hy, Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2509.09631)  

**Abstract**: Zero-shot Text-to-Speech (TTS) aims to synthesize high-quality speech that mimics the voice of an unseen speaker using only a short reference sample, requiring not only speaker adaptation but also accurate modeling of prosodic attributes. Recent approaches based on language models, diffusion, and flow matching have shown promising results in zero-shot TTS, but still suffer from slow inference and repetition artifacts. Discrete codec representations have been widely adopted for speech synthesis, and recent works have begun to explore diffusion models in purely discrete settings, suggesting the potential of discrete generative modeling for speech synthesis. However, existing flow-matching methods typically embed these discrete tokens into a continuous space and apply continuous flow matching, which may not fully leverage the advantages of discrete representations. To address these challenges, we introduce DiFlow-TTS, which, to the best of our knowledge, is the first model to explore purely Discrete Flow Matching for speech synthesis. DiFlow-TTS explicitly models factorized speech attributes within a compact and unified architecture. It leverages in-context learning by conditioning on textual content, along with prosodic and acoustic attributes extracted from a reference speech, enabling effective attribute cloning in a zero-shot setting. In addition, the model employs a factorized flow prediction mechanism with distinct heads for prosody and acoustic details, allowing it to learn aspect-specific distributions. Experimental results demonstrate that DiFlow-TTS achieves promising performance in several key metrics, including naturalness, prosody, preservation of speaker style, and energy control. It also maintains a compact model size and achieves low-latency inference, generating speech up to 25.8 times faster than the latest existing baselines. 

---
# LLMs Don't Know Their Own Decision Boundaries: The Unreliability of Self-Generated Counterfactual Explanations 

**Authors**: Harry Mayne, Ryan Othniel Kearns, Yushi Yang, Andrew M. Bean, Eoin Delaney, Chris Russell, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09396)  

**Abstract**: To collaborate effectively with humans, language models must be able to explain their decisions in natural language. We study a specific type of self-explanation: self-generated counterfactual explanations (SCEs), where a model explains its prediction by modifying the input such that it would have predicted a different outcome. We evaluate whether LLMs can produce SCEs that are valid, achieving the intended outcome, and minimal, modifying the input no more than necessary. When asked to generate counterfactuals, we find that LLMs typically produce SCEs that are valid, but far from minimal, offering little insight into their decision-making behaviour. Worryingly, when asked to generate minimal counterfactuals, LLMs typically make excessively small edits that fail to change predictions. The observed validity-minimality trade-off is consistent across several LLMs, datasets, and evaluation settings. Our findings suggest that SCEs are, at best, an ineffective explainability tool and, at worst, can provide misleading insights into model behaviour. Proposals to deploy LLMs in high-stakes settings must consider the impact of unreliable self-explanations on downstream decision-making. Our code is available at this https URL. 

---
# OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning 

**Authors**: Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yuzheng Zhuang, Bowen Yang, He Zhu, Lingfeng Zhang, Pengwei Xie, David Gamaliel Arcos Bravo, Yingxue Zhang, Jianye Hao, Xingyue Quan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09332)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically this http URL address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: this https URL 

---
# Can Multimodal LLMs See Materials Clearly? A Multimodal Benchmark on Materials Characterization 

**Authors**: Zhengzhao Lai, Youbin Zheng, Zhenyang Cai, Haonan Lyu, Jinpu Yang, Hongqing Liang, Yan Hu, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09307)  

**Abstract**: Materials characterization is fundamental to acquiring materials information, revealing the processing-microstructure-property relationships that guide material design and optimization. While multimodal large language models (MLLMs) have recently shown promise in generative and predictive tasks within materials science, their capacity to understand real-world characterization imaging data remains underexplored. To bridge this gap, we present MatCha, the first benchmark for materials characterization image understanding, comprising 1,500 questions that demand expert-level domain expertise. MatCha encompasses four key stages of materials research comprising 21 distinct tasks, each designed to reflect authentic challenges faced by materials scientists. Our evaluation of state-of-the-art MLLMs on MatCha reveals a significant performance gap compared to human experts. These models exhibit degradation when addressing questions requiring higher-level expertise and sophisticated visual perception. Simple few-shot and chain-of-thought prompting struggle to alleviate these limitations. These findings highlight that existing MLLMs still exhibit limited adaptability to real-world materials characterization scenarios. We hope MatCha will facilitate future research in areas such as new material discovery and autonomous scientific agents. MatCha is available at this https URL. 

---
# Tree-OPO: Off-policy Monte Carlo Tree-Guided Advantage Optimization for Multistep Reasoning 

**Authors**: Bingning Huang, Tu Nguyen, Matthieu Zimmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.09284)  

**Abstract**: Recent advances in reasoning with large language models (LLMs) have shown the effectiveness of Monte Carlo Tree Search (MCTS) for generating high-quality intermediate trajectories, particularly in math and symbolic domains. Inspired by this, we explore how MCTS-derived trajectories, traditionally used for training value or reward models, can be repurposed to improve policy optimization in preference-based reinforcement learning (RL). Specifically, we focus on Group Relative Policy Optimization (GRPO), a recent algorithm that enables preference-consistent policy learning without value networks. We propose a staged GRPO training paradigm where completions are derived from partially revealed MCTS rollouts, introducing a novel tree-structured setting for advantage estimation. This leads to a rich class of prefix-conditioned reward signals, which we analyze theoretically and empirically. Our initial results indicate that while structured advantage estimation can stabilize updates and better reflect compositional reasoning quality, challenges such as advantage saturation and reward signal collapse remain. We propose heuristic and statistical solutions to mitigate these issues and discuss open challenges for learning under staged or tree-like reward structures. 

---
# Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents 

**Authors**: Jiawei Wang, Jiacai Liu, Yuqian Fu, Yingru Li, Xintao Wang, Yuan Lin, Yu Yue, Lin Zhang, Yang Wang, Ke Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09265)  

**Abstract**: In long-horizon tasks, recent agents based on Large Language Models (LLMs) face a significant challenge that sparse, outcome-based rewards make it difficult to assign credit to intermediate steps. Previous methods mainly focus on creating dense reward signals to guide learning, either through traditional reinforcement learning techniques like inverse reinforcement learning or by using Process Reward Models for step-by-step feedback. In this paper, we identify a fundamental problem in the learning dynamics of LLMs: the magnitude of policy gradients is inherently coupled with the entropy, which leads to inefficient small updates for confident correct actions and potentially destabilizes large updates for uncertain ones. To resolve this, we propose Entropy-Modulated Policy Gradients (EMPG), a framework that re-calibrates the learning signal based on step-wise uncertainty and the final task outcome. EMPG amplifies updates for confident correct actions, penalizes confident errors, and attenuates updates from uncertain steps to stabilize exploration. We further introduce a bonus term for future clarity that encourages agents to find more predictable solution paths. Through comprehensive experiments on three challenging agent tasks, WebShop, ALFWorld, and Deep Search, we demonstrate that EMPG achieves substantial performance gains and significantly outperforms strong policy gradient baselines. Project page is at this https URL 

---
# Identifying Key Features for Establishing Sustainable Agro-Tourism Centre: A Data Driven Approach 

**Authors**: Alka Gadakh, Vidya Kumbhar, Sonal Khosla, Kumar Karunendra  

**Link**: [PDF](https://arxiv.org/pdf/2509.09214)  

**Abstract**: Agro-tourism serves as a strategic economic model designed to facilitate rural development by diversifying income streams for local communities like farmers while promoting the conservation of indigenous cultural heritage and traditional agricultural practices. As a very booming subdomain of tourism, there is a need to study the strategies for the growth of Agro-tourism in detail. The current study has identified the important indicators for the growth and enhancement of agro-tourism. The study is conducted in two phases: identification of the important indicators through a comprehensive literature review and in the second phase state-of-the-art techniques were used to identify the important indicators for the growth of agro-tourism. The indicators are also called features synonymously, the machine learning models for feature selection were applied and it was observed that the Least Absolute Shrinkage and Selection Operator (LASSO) method combined with, the machine Learning Classifiers such as Logistic Regression (LR), Decision Trees (DT), Random Forest (RF) Tree, and Extreme Gradient Boosting (XGBOOST) models were used to suggest the growth of the agro-tourism. The results show that with the LASSO method, LR model gives the highest classification accuracy of 98% in 70-30% train-test data followed by RF with 95% accuracy. Similarly, in the 80-20% train-test data LR maintains the highest accuracy at 99%, while DT and XGBoost follow with 97% accuracy. 

---
# Bona fide Cross Testing Reveals Weak Spot in Audio Deepfake Detection Systems 

**Authors**: Chin Yuen Kwok, Jia Qi Yip, Zhen Qiu, Chi Hung Chi, Kwok Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2509.09204)  

**Abstract**: Audio deepfake detection (ADD) models are commonly evaluated using datasets that combine multiple synthesizers, with performance reported as a single Equal Error Rate (EER). However, this approach disproportionately weights synthesizers with more samples, underrepresenting others and reducing the overall reliability of EER. Additionally, most ADD datasets lack diversity in bona fide speech, often featuring a single environment and speech style (e.g., clean read speech), limiting their ability to simulate real-world conditions. To address these challenges, we propose bona fide cross-testing, a novel evaluation framework that incorporates diverse bona fide datasets and aggregates EERs for more balanced assessments. Our approach improves robustness and interpretability compared to traditional evaluation methods. We benchmark over 150 synthesizers across nine bona fide speech types and release a new dataset to facilitate further research at this https URL. 

---
# COCO-Urdu: A Large-Scale Urdu Image-Caption Dataset with Multimodal Quality Estimation 

**Authors**: Umair Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09014)  

**Abstract**: Urdu, spoken by over 250 million people, remains critically under-served in multimodal and vision-language research. The absence of large-scale, high-quality datasets has limited the development of Urdu-capable systems and reinforced biases in multilingual vision-language models trained primarily on high-resource languages. To address this gap, we present COCO-Urdu, a large-scale image-caption dataset derived from MS COCO, containing 59,000 images and 319,000 Urdu captions selected through stratified sampling to preserve the original distribution. Captions were translated using SeamlessM4T v2 and validated with a hybrid multimodal quality estimation framework that integrates COMET-Kiwi for translation quality, CLIP-based similarity for visual grounding, and BERTScore with back-translation for semantic consistency; low-scoring captions were iteratively refined using open-source large language models. We further benchmark COCO-Urdu on BLEU, SacreBLEU, and chrF, reporting consistently strong results. To the best of our knowledge, COCO-Urdu is the largest publicly available Urdu captioning dataset. By releasing both the dataset and the quality estimation pipeline, we aim to reduce language bias in multimodal research and establish a foundation for inclusive vision-language systems. 

---
# Open-sci-ref-0.01: open and reproducible reference baselines for language model and dataset comparison 

**Authors**: Marianna Nezhurina, Taishi Nakamura, Timur Carstensen, Niccolò Ajroldi, Ville Komulainen, David Salinas, Jenia Jitsev  

**Link**: [PDF](https://arxiv.org/pdf/2509.09009)  

**Abstract**: We introduce open-sci-ref, a family of dense transformer models trained as research baselines across multiple model (0.13B to 1.7B parameters) and token scales (up to 1T) on 8 recent open reference datasets. Evaluating the models on various standardized benchmarks, our training runs set establishes reference points that enable researchers to assess the sanity and quality of alternative training approaches across scales and datasets. Intermediate checkpoints allow comparison and studying of the training dynamics. The established reference baselines allow training procedures to be compared through their scaling trends, aligning them on a common compute axis. Comparison of open reference datasets reveals that training on NemoTron-CC HQ consistently outperforms other reference datasets, followed by DCLM-baseline and FineWeb-Edu. In addition to intermediate training checkpoints, the release includes logs, code, and downstream evaluations to simplify reproduction, standardize comparison, and facilitate future research. 

---
# Generative Engine Optimization: How to Dominate AI Search 

**Authors**: Mahe Chen, Xiaoxuan Wang, Kaiwen Chen, Nick Koudas  

**Link**: [PDF](https://arxiv.org/pdf/2509.08919)  

**Abstract**: The rapid adoption of generative AI-powered search engines like ChatGPT, Perplexity, and Gemini is fundamentally reshaping information retrieval, moving from traditional ranked lists to synthesized, citation-backed answers. This shift challenges established Search Engine Optimization (SEO) practices and necessitates a new paradigm, which we term Generative Engine Optimization (GEO).
This paper presents a comprehensive comparative analysis of AI Search and traditional web search (Google). Through a series of large-scale, controlled experiments across multiple verticals, languages, and query paraphrases, we quantify critical differences in how these systems source information. Our key findings reveal that AI Search exhibit a systematic and overwhelming bias towards Earned media (third-party, authoritative sources) over Brand-owned and Social content, a stark contrast to Google's more balanced mix. We further demonstrate that AI Search services differ significantly from each other in their domain diversity, freshness, cross-language stability, and sensitivity to phrasing.
Based on these empirical results, we formulate a strategic GEO agenda. We provide actionable guidance for practitioners, emphasizing the critical need to: (1) engineer content for machine scannability and justification, (2) dominate earned media to build AI-perceived authority, (3) adopt engine-specific and language-aware strategies, and (4) overcome the inherent "big brand bias" for niche players. Our work provides the foundational empirical analysis and a strategic framework for achieving visibility in the new generative search landscape. 

---
# Recurrence Meets Transformers for Universal Multimodal Retrieval 

**Authors**: Davide Caffagni, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2509.08897)  

**Abstract**: With the rapid advancement of multimodal retrieval and its application in LLMs and multimodal LLMs, increasingly complex retrieval tasks have emerged. Existing methods predominantly rely on task-specific fine-tuning of vision-language models and are limited to single-modality queries or documents. In this paper, we propose ReT-2, a unified retrieval model that supports multimodal queries, composed of both images and text, and searches across multimodal document collections where text and images coexist. ReT-2 leverages multi-layer representations and a recurrent Transformer architecture with LSTM-inspired gating mechanisms to dynamically integrate information across layers and modalities, capturing fine-grained visual and textual details. We evaluate ReT-2 on the challenging M2KR and M-BEIR benchmarks across different retrieval configurations. Results demonstrate that ReT-2 consistently achieves state-of-the-art performance across diverse settings, while offering faster inference and reduced memory usage compared to prior approaches. When integrated into retrieval-augmented generation pipelines, ReT-2 also improves downstream performance on Encyclopedic-VQA and InfoSeek datasets. Our source code and trained models are publicly available at: this https URL 

---
# A vibe coding learning design to enhance EFL students' talking to, through, and about AI 

**Authors**: David James Woo, Kai Guo, Yangyang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.08854)  

**Abstract**: This innovative practice article reports on the piloting of vibe coding (using natural language to create software applications with AI) for English as a Foreign Language (EFL) education. We developed a human-AI meta-languaging framework with three dimensions: talking to AI (prompt engineering), talking through AI (negotiating authorship), and talking about AI (mental models of AI). Using backward design principles, we created a four-hour workshop where two students designed applications addressing authentic EFL writing challenges. We adopted a case study methodology, collecting data from worksheets and video recordings, think-aloud protocols, screen recordings, and AI-generated images. Contrasting cases showed one student successfully vibe coding a functional application cohering to her intended design, while another encountered technical difficulties with major gaps between intended design and actual functionality. Analysis reveals differences in students' prompt engineering approaches, suggesting different AI mental models and tensions in attributing authorship. We argue that AI functions as a beneficial languaging machine, and that differences in how students talk to, through, and about AI explain vibe coding outcome variations. Findings indicate that effective vibe coding instruction requires explicit meta-languaging scaffolding, teaching structured prompt engineering, facilitating critical authorship discussions, and developing vocabulary for articulating AI mental models. 

---
# Automated Unity Game Template Generation from GDDs via NLP and Multi-Modal LLMs 

**Authors**: Amna Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2509.08847)  

**Abstract**: This paper presents a novel framework for automated game template generation by transforming Game Design Documents (GDDs) into functional Unity game prototypes using Natural Language Processing (NLP) and multi-modal Large Language Models (LLMs). We introduce an end-to-end system that parses GDDs, extracts structured game specifications, and synthesizes Unity-compatible C# code that implements the core mechanics, systems, and architecture defined in the design documentation. Our approach combines a fine-tuned LLaMA-3 model specialized for Unity code generation with a custom Unity integration package that streamlines the implementation process. Evaluation results demonstrate significant improvements over baseline models, with our fine-tuned model achieving superior performance (4.8/5.0 average score) compared to state-of-the-art LLMs across compilation success, GDD adherence, best practices adoption, and code modularity metrics. The generated templates demonstrate high adherence to GDD specifications across multiple game genres. Our system effectively addresses critical gaps in AI-assisted game development, positioning LLMs as valuable tools in streamlining the transition from game design to implementation. 

---
