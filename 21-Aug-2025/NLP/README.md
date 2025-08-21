# Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs 

**Authors**: Haokun Lin, Haobo Xu, Yichen Wu, Ziyu Guo, Renrui Zhang, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.14896)  

**Abstract**: Recent advances in diffusion large language models (dLLMs) have introduced a promising alternative to autoregressive (AR) LLMs for natural language generation tasks, leveraging full attention and denoising-based decoding strategies. However, the deployment of these models on edge devices remains challenging due to their massive parameter scale and high resource demands. While post-training quantization (PTQ) has emerged as a widely adopted technique for compressing AR LLMs, its applicability to dLLMs remains largely unexplored. In this work, we present the first systematic study on quantizing diffusion-based language models. We begin by identifying the presence of activation outliers, characterized by abnormally large activation values that dominate the dynamic range. These outliers pose a key challenge to low-bit quantization, as they make it difficult to preserve precision for the majority of values. More importantly, we implement state-of-the-art PTQ methods and conduct a comprehensive evaluation across multiple task types and model variants. Our analysis is structured along four key dimensions: bit-width, quantization method, task category, and model type. Through this multi-perspective evaluation, we offer practical insights into the quantization behavior of dLLMs under different configurations. We hope our findings provide a foundation for future research in efficient dLLM deployment. All codes and experimental setups will be released to support the community. 

---
# MedReseacher-R1: Expert-Level Medical Deep Researcher via A Knowledge-Informed Trajectory Synthesis Framework 

**Authors**: Ailing Yu, Lan Yao, Jingnan Liu, Zhe Chen, Jiajun Yin, Yuan Wang, Xinhao Liao, Zhiling Ye, Ji Li, Yun Yue, Hansong Xiao, Hualei Zhou, Chunxiao Guo, Peng Wei, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14880)  

**Abstract**: Recent developments in Large Language Model (LLM)-based agents have shown impressive capabilities spanning multiple domains, exemplified by deep research systems that demonstrate superior performance on complex information-seeking and synthesis tasks. While general-purpose deep research agents have shown impressive capabilities, they struggle significantly with medical domain challenges, as evidenced by leading proprietary systems achieving limited accuracy on complex medical benchmarks. The key limitations are: (1) the model lacks sufficient dense medical knowledge for clinical reasoning, and (2) the framework is constrained by the absence of specialized retrieval tools tailored for medical this http URL present a medical deep research agent that addresses these challenges through two core innovations. First, we develop a novel data synthesis framework using medical knowledge graphs, extracting the longest chains from subgraphs around rare medical entities to generate complex multi-hop question-answer pairs. Second, we integrate a custom-built private medical retrieval engine alongside general-purpose tools, enabling accurate medical information synthesis. Our approach generates 2100+ diverse trajectories across 12 medical specialties, each averaging 4.2 tool this http URL a two-stage training paradigm combining supervised fine-tuning and online reinforcement learning with composite rewards, our MedResearcher-R1-32B model demonstrates exceptional performance, establishing new state-of-the-art results on medical benchmarks while maintaining competitive performance on general deep research tasks. Our work demonstrates that strategic domain-specific innovations in architecture, tool design, and training data construction can enable smaller open-source models to outperform much larger proprietary systems in specialized domains. 

---
# Long Chain-of-Thought Reasoning Across Languages 

**Authors**: Josh Barua, Seun Eisape, Kayo Yin, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2508.14828)  

**Abstract**: Scaling inference through long chains-of-thought (CoTs) has unlocked impressive reasoning capabilities in large language models (LLMs), yet the reasoning process remains almost exclusively English-centric. We construct translated versions of two popular English reasoning datasets, fine-tune Qwen 2.5 (7B) and Qwen 3 (8B) models, and present a systematic study of long CoT generation across French, Japanese, Latvian, and Swahili. Our experiments reveal three key findings. First, the efficacy of using English as a pivot language varies by language: it provides no benefit for French, improves performance when used as the reasoning language for Japanese and Latvian, and proves insufficient for Swahili where both task comprehension and reasoning remain poor. Second, extensive multilingual pretraining in Qwen 3 narrows but does not eliminate the cross-lingual performance gap. A lightweight fine-tune using only 1k traces still improves performance by over 30\% in Swahili. Third, data quality versus scale trade-offs are language dependent: small, carefully curated datasets suffice for English and French, whereas larger but noisier corpora prove more effective for Swahili and Latvian. Together, these results clarify when and why long CoTs transfer across languages and provide translated datasets to foster equitable multilingual reasoning research. 

---
# Evaluating Retrieval-Augmented Generation vs. Long-Context Input for Clinical Reasoning over EHRs 

**Authors**: Skatje Myers, Dmitriy Dligach, Timothy A. Miller, Samantha Barr, Yanjun Gao, Matthew Churpek, Anoop Mayampurath, Majid Afshar  

**Link**: [PDF](https://arxiv.org/pdf/2508.14817)  

**Abstract**: Electronic health records (EHRs) are long, noisy, and often redundant, posing a major challenge for the clinicians who must navigate them. Large language models (LLMs) offer a promising solution for extracting and reasoning over this unstructured text, but the length of clinical notes often exceeds even state-of-the-art models' extended context windows. Retrieval-augmented generation (RAG) offers an alternative by retrieving task-relevant passages from across the entire EHR, potentially reducing the amount of required input tokens. In this work, we propose three clinical tasks designed to be replicable across health systems with minimal effort: 1) extracting imaging procedures, 2) generating timelines of antibiotic use, and 3) identifying key diagnoses. Using EHRs from actual hospitalized patients, we test three state-of-the-art LLMs with varying amounts of provided context, using either targeted text retrieval or the most recent clinical notes. We find that RAG closely matches or exceeds the performance of using recent notes, and approaches the performance of using the models' full context while requiring drastically fewer input tokens. Our results suggest that RAG remains a competitive and efficient approach even as newer models become capable of handling increasingly longer amounts of text. 

---
# TransLLM: A Unified Multi-Task Foundation Framework for Urban Transportation via Learnable Prompting 

**Authors**: Jiaming Leng, Yunying Bi, Chuan Qin, Bing Yin, Yanyong Zhang, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14782)  

**Abstract**: Urban transportation systems encounter diverse challenges across multiple tasks, such as traffic forecasting, electric vehicle (EV) charging demand prediction, and taxi dispatch. Existing approaches suffer from two key limitations: small-scale deep learning models are task-specific and data-hungry, limiting their generalizability across diverse scenarios, while large language models (LLMs), despite offering flexibility through natural language interfaces, struggle with structured spatiotemporal data and numerical reasoning in transportation domains. To address these limitations, we propose TransLLM, a unified foundation framework that integrates spatiotemporal modeling with large language models through learnable prompt composition. Our approach features a lightweight spatiotemporal encoder that captures complex dependencies via dilated temporal convolutions and dual-adjacency graph attention networks, seamlessly interfacing with LLMs through structured embeddings. A novel instance-level prompt routing mechanism, trained via reinforcement learning, dynamically personalizes prompts based on input characteristics, moving beyond fixed task-specific templates. The framework operates by encoding spatiotemporal patterns into contextual representations, dynamically composing personalized prompts to guide LLM reasoning, and projecting the resulting representations through specialized output layers to generate task-specific predictions. Experiments across seven datasets and three tasks demonstrate the exceptional effectiveness of TransLLM in both supervised and zero-shot settings. Compared to ten baseline models, it delivers competitive performance on both regression and planning problems, showing strong generalization and cross-task adaptability. Our code is available at this https URL. 

---
# Evaluating Multilingual and Code-Switched Alignment in LLMs via Synthetic Natural Language Inference 

**Authors**: Samir Abdaljalil, Erchin Serpedin, Khalid Qaraqe, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2508.14735)  

**Abstract**: Large language models (LLMs) are increasingly applied in multilingual contexts, yet their capacity for consistent, logically grounded alignment across languages remains underexplored. We present a controlled evaluation framework for multilingual natural language inference (NLI) that generates synthetic, logic-based premise-hypothesis pairs and translates them into a typologically diverse set of languages. This design enables precise control over semantic relations and allows testing in both monolingual and mixed-language (code-switched) conditions. Surprisingly, code-switching does not degrade, and can even improve, performance, suggesting that translation-induced lexical variation may serve as a regularization signal. We validate semantic preservation through embedding-based similarity analyses and cross-lingual alignment visualizations, confirming the fidelity of translated pairs. Our findings expose both the potential and the brittleness of current LLM cross-lingual reasoning, and identify code-switching as a promising lever for improving multilingual robustness. Code available at: this https URL 

---
# Transplant Then Regenerate: A New Paradigm for Text Data Augmentation 

**Authors**: Guangzhan Wang, Hongyu Zhang, Beijun Shen, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14723)  

**Abstract**: Data augmentation is a critical technique in deep learning. Traditional methods like Back-translation typically focus on lexical-level rephrasing, which primarily produces variations with the same semantics. While large language models (LLMs) have enhanced text augmentation by their "knowledge emergence" capability, controlling the style and structure of these outputs remains challenging and requires meticulous prompt engineering. In this paper, we propose LMTransplant, a novel text augmentation paradigm leveraging LLMs. The core idea of LMTransplant is transplant-then-regenerate: incorporating seed text into a context expanded by LLM, and asking the LLM to regenerate a variant based on the expanded context. This strategy allows the model to create more diverse and creative content-level variants by fully leveraging the knowledge embedded in LLMs, while preserving the core attributes of the original text. We evaluate LMTransplant across various text-related tasks, demonstrating its superior performance over existing text augmentation methods. Moreover, LMTransplant demonstrates exceptional scalability as the size of augmented data grows. 

---
# The Digital Sous Chef -- A Comparative Study on Fine-Tuning Language Models for Recipe Generation 

**Authors**: Shubham Pundhir, Ganesh Bagler  

**Link**: [PDF](https://arxiv.org/pdf/2508.14718)  

**Abstract**: We established a rigorous benchmark for text-based recipe generation, a fundamental task in natural language generation. We present a comprehensive comparative study contrasting a fine-tuned GPT-2 large (774M) model against the GPT-2 small (124M) model and traditional LSTM/RNN baselines on the 5-cuisine corpus from RecipeDB. Our key contribution is a targeted tokenization strategy that augments the vocabulary with 23 common fraction tokens and custom structural markers. This approach addresses a critical limitation of generic tokenizers by preserving essential recipe structures and precise numerical quantities, thereby enhancing domain specificity. Performance is evaluated using a comprehensive suite of seven automatic metrics spanning fluency (BLEU-4, METEOR), coherence (ROUGE-L), semantic relevance (BERTScore), and diversity. Our experiments show that the large transformer-based approach yields a >20% relative improvement in BERTScore (F1) (0.92 vs 0.72) over the best recurrent baseline, while reducing perplexity by 69.8%. We conclude with a discussion of remaining challenges, particularly regarding factual accuracy, and outline how this foundational study paves the way for integrating real-world constraints and multi-modal inputs in advanced recipe generation research. 

---
# ShizhenGPT: Towards Multimodal LLMs for Traditional Chinese Medicine 

**Authors**: Junying Chen, Zhenyang Cai, Zhiheng Liu, Yunjin Yang, Rongsheng Wang, Qingying Xiao, Xiangyi Feng, Zhan Su, Jing Guo, Xiang Wan, Guangjun Yu, Haizhou Li, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14706)  

**Abstract**: Despite the success of large language models (LLMs) in various domains, their potential in Traditional Chinese Medicine (TCM) remains largely underexplored due to two critical barriers: (1) the scarcity of high-quality TCM data and (2) the inherently multimodal nature of TCM diagnostics, which involve looking, listening, smelling, and pulse-taking. These sensory-rich modalities are beyond the scope of conventional LLMs. To address these challenges, we present ShizhenGPT, the first multimodal LLM tailored for TCM. To overcome data scarcity, we curate the largest TCM dataset to date, comprising 100GB+ of text and 200GB+ of multimodal data, including 1.2M images, 200 hours of audio, and physiological signals. ShizhenGPT is pretrained and instruction-tuned to achieve deep TCM knowledge and multimodal reasoning. For evaluation, we collect recent national TCM qualification exams and build a visual benchmark for Medicinal Recognition and Visual Diagnosis. Experiments demonstrate that ShizhenGPT outperforms comparable-scale LLMs and competes with larger proprietary models. Moreover, it leads in TCM visual understanding among existing multimodal LLMs and demonstrates unified perception across modalities like sound, pulse, smell, and vision, paving the way toward holistic multimodal perception and diagnosis in TCM. Datasets, models, and code are publicly available. We hope this work will inspire further exploration in this field. 

---
# Improving in-context learning with a better scoring function 

**Authors**: Omar Naim, Swarnadeep Bhar, Jérôme Bolte, Nicholas Asher  

**Link**: [PDF](https://arxiv.org/pdf/2508.14685)  

**Abstract**: Large language models (LLMs) exhibit a remarkable capacity to learn by analogy, known as in-context learning (ICL). However, recent studies have revealed limitations in this ability. In this paper, we examine these limitations on tasks involving first-order quantifiers such as {\em all} and {\em some}, as well as on ICL with linear functions. We identify Softmax, the scoring function in attention mechanism, as a contributing factor to these constraints. To address this, we propose \textbf{scaled signed averaging (SSA)}, a novel alternative to Softmax. Empirical results show that SSA dramatically improves performance on our target tasks. Furthermore, we evaluate both encoder-only and decoder-only transformers models with SSA, demonstrating that they match or exceed their Softmax-based counterparts across a variety of linguistic probing tasks. 

---
# Continuous sentiment scores for literary and multilingual contexts 

**Authors**: Laurits Lyngbaek, Pascale Feldkamp, Yuri Bizzoni, Kristoffer Nielbo, Kenneth Enevoldsen  

**Link**: [PDF](https://arxiv.org/pdf/2508.14620)  

**Abstract**: Sentiment Analysis is widely used to quantify sentiment in text, but its application to literary texts poses unique challenges due to figurative language, stylistic ambiguity, as well as sentiment evocation strategies. Traditional dictionary-based tools often underperform, especially for low-resource languages, and transformer models, while promising, typically output coarse categorical labels that limit fine-grained analysis. We introduce a novel continuous sentiment scoring method based on concept vector projection, trained on multilingual literary data, which more effectively captures nuanced sentiment expressions across genres, languages, and historical periods. Our approach outperforms existing tools on English and Danish texts, producing sentiment scores whose distribution closely matches human ratings, enabling more accurate analysis and sentiment arc modeling in literature. 

---
# Filling the Gap for Uzbek: Creating Translation Resources for Southern Uzbek 

**Authors**: Mukhammadsaid Mamasaidov, Azizullah Aral, Abror Shopulatov, Mironshoh Inomjonov  

**Link**: [PDF](https://arxiv.org/pdf/2508.14586)  

**Abstract**: Southern Uzbek (uzs) is a Turkic language variety spoken by around 5 million people in Afghanistan and differs significantly from Northern Uzbek (uzn) in phonology, lexicon, and orthography. Despite the large number of speakers, Southern Uzbek is underrepresented in natural language processing. We present new resources for Southern Uzbek machine translation, including a 997-sentence FLORES+ dev set, 39,994 parallel sentences from dictionary, literary, and web sources, and a fine-tuned NLLB-200 model (lutfiy). We also propose a post-processing method for restoring Arabic-script half-space characters, which improves handling of morphological boundaries. All datasets, models, and tools are released publicly to support future work on Southern Uzbek and other low-resource languages. 

---
# Towards Skeletal and Signer Noise Reduction in Sign Language Production via Quaternion-Based Pose Encoding and Contrastive Learning 

**Authors**: Guilhem Fauré, Mostafa Sadeghi, Sam Bigeard, Slim Ouni  

**Link**: [PDF](https://arxiv.org/pdf/2508.14574)  

**Abstract**: One of the main challenges in neural sign language production (SLP) lies in the high intra-class variability of signs, arising from signer morphology and stylistic variety in the training data. To improve robustness to such variations, we propose two enhancements to the standard Progressive Transformers (PT) architecture (Saunders et al., 2020). First, we encode poses using bone rotations in quaternion space and train with a geodesic loss to improve the accuracy and clarity of angular joint movements. Second, we introduce a contrastive loss to structure decoder embeddings by semantic similarity, using either gloss overlap or SBERT-based sentence similarity, aiming to filter out anatomical and stylistic features that do not convey relevant semantic information. On the Phoenix14T dataset, the contrastive loss alone yields a 16% improvement in Probability of Correct Keypoint over the PT baseline. When combined with quaternion-based pose encoding, the model achieves a 6% reduction in Mean Bone Angle Error. These results point to the benefit of incorporating skeletal structure modeling and semantically guided contrastive objectives on sign pose representations into the training of Transformer-based SLP models. 

---
# EmoTale: An Enacted Speech-emotion Dataset in Danish 

**Authors**: Maja J. Hjuler, Harald V. Skat-Rørdam, Line H. Clemmensen, Sneha Das  

**Link**: [PDF](https://arxiv.org/pdf/2508.14548)  

**Abstract**: While multiple emotional speech corpora exist for commonly spoken languages, there is a lack of functional datasets for smaller (spoken) languages, such as Danish. To our knowledge, Danish Emotional Speech (DES), published in 1997, is the only other database of Danish emotional speech. We present EmoTale; a corpus comprising Danish and English speech recordings with their associated enacted emotion annotations. We demonstrate the validity of the dataset by investigating and presenting its predictive power using speech emotion recognition (SER) models. We develop SER models for EmoTale and the reference datasets using self-supervised speech model (SSLM) embeddings and the openSMILE feature extractor. We find the embeddings superior to the hand-crafted features. The best model achieves an unweighted average recall (UAR) of 64.1% on the EmoTale corpus using leave-one-speaker-out cross-validation, comparable to the performance on DES. 

---
# Reasoning is about giving reasons 

**Authors**: Krunal Shah, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2508.14488)  

**Abstract**: Convincing someone of the truth value of a premise requires understanding and articulating the core logical structure of the argument which proves or disproves the premise. Understanding the logical structure of an argument refers to understanding the underlying "reasons" which make up the proof or disproof of the premise - as a function of the "logical atoms" in the argument. While it has been shown that transformers can "chain" rules to derive simple arguments, the challenge of articulating the "reasons" remains. Not only do current approaches to chaining rules suffer in terms of their interpretability, they are also quite constrained in their ability to accommodate extensions to theoretically equivalent reasoning tasks - a model trained to chain rules cannot support abduction or identify contradictions. In this work we suggest addressing these shortcomings by identifying an intermediate representation (which we call the Representation of the Logical Structure (RLS) of the argument) that possesses an understanding of the logical structure of a natural language argument - the logical atoms in the argument and the rules incorporating them. Given the logical structure, reasoning is deterministic and easy to compute. Therefore, our approach supports all forms of reasoning that depend on the logical structure of the natural language argument, including arbitrary depths of reasoning, on-the-fly mistake rectification and interactive discussion with respect to an argument. We show that we can identify and extract the logical structure of natural language arguments in three popular reasoning datasets with high accuracies, thus supporting explanation generation and extending the reasoning capabilities significantly. 

---
# In2x at WMT25 Translation Task 

**Authors**: Lei Pang, Hanyi Mao, Quanjia Xiao, HaiXiao Liu, Xiangyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14472)  

**Abstract**: This paper presents the open-system submission by the In2x research team for the WMT25 General Machine Translation Shared Task. Our submission focuses on Japanese-related translation tasks, aiming to explore a generalizable paradigm for extending large language models (LLMs) to other languages. This paradigm encompasses aspects such as data construction methods and reward model design. The ultimate goal is to enable large language model systems to achieve exceptional performance in low-resource or less commonly spoken languages. 

---
# NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model 

**Authors**: NVIDIA, Aarti Basant, Abhijit Khairnar, Abhijit Paithankar, Abhinav Khattar, Adi Renduchintala, Adithya Renduchintala, Aditya Malte, Akhiad Bercovich, Akshay Hazare, Alejandra Rico, Aleksander Ficek, Alex Kondratenko, Alex Shaposhnikov, Ali Taghibakhshi, Amelia Barton, Ameya Sunil Mahabaleshwarkar, Amy Shen, Andrew Tao, Ann Guan, Anna Shors, Anubhav Mandarwal, Arham Mehta, Arun Venkatesan, Ashton Sharabiani, Ashwath Aithal, Ashwin Poojary, Ayush Dattagupta, Balaram Buddharaju, Banghua Zhu, Barnaby Simkin, Bilal Kartal, Bita Darvish Rouhani, Bobby Chen, Boris Ginsburg, Brandon Norick, Brian Yu, Bryan Catanzaro, Charles Wang, Charlie Truong, Chetan Mungekar, Chintan Patel, Chris Alexiuk, Christian Munley, Christopher Parisien, Dan Su, Daniel Afrimi, Daniel Korzekwa, Daniel Rohrer, Daria Gitman, David Mosallanezhad, Deepak Narayanan, Dima Rekesh, Dina Yared, Dmytro Pykhtar, Dong Ahn, Duncan Riach, Eileen Long, Elliott Ning, Eric Chung, Erick Galinkin, Evelina Bakhturina, Gargi Prasad, Gerald Shen, Haim Elisha, Harsh Sharma, Hayley Ross, Helen Ngo, Herman Sahota, Hexin Wang, Hoo Chang Shin, Hua Huang, Iain Cunningham, Igor Gitman, Ivan Moshkov, Jaehun Jung, Jan Kautz, Jane Polak Scowcroft, Jared Casper, Jimmy Zhang, Jinze Xue, Jocelyn Huang, Joey Conway, John Kamalu, Jonathan Cohen, Joseph Jennings, Julien Veron Vialard, Junkeun Yi, Jupinder Parmar, Kari Briski, Katherine Cheung, Katherine Luna, Keith Wyss, Keshav Santhanam, Kezhi Kong, Krzysztof Pawelec, Kumar Anik, Kunlun Li, Kushan Ahmadian, Lawrence McAfee  

**Link**: [PDF](https://arxiv.org/pdf/2508.14444)  

**Abstract**: We introduce Nemotron-Nano-9B-v2, a hybrid Mamba-Transformer language model designed to increase throughput for reasoning workloads while achieving state-of-the-art accuracy compared to similarly-sized models. Nemotron-Nano-9B-v2 builds on the Nemotron-H architecture, in which the majority of the self-attention layers in the common Transformer architecture are replaced with Mamba-2 layers, to achieve improved inference speed when generating the long thinking traces needed for reasoning. We create Nemotron-Nano-9B-v2 by first pre-training a 12-billion-parameter model (Nemotron-Nano-12B-v2-Base) on 20 trillion tokens using an FP8 training recipe. After aligning Nemotron-Nano-12B-v2-Base, we employ the Minitron strategy to compress and distill the model with the goal of enabling inference on up to 128k tokens on a single NVIDIA A10G GPU (22GiB of memory, bfloat16 precision). Compared to existing similarly-sized models (e.g., Qwen3-8B), we show that Nemotron-Nano-9B-v2 achieves on-par or better accuracy on reasoning benchmarks while achieving up to 6x higher inference throughput in reasoning settings like 8k input and 16k output tokens. We are releasing Nemotron-Nano-9B-v2, Nemotron-Nano12B-v2-Base, and Nemotron-Nano-9B-v2-Base checkpoints along with the majority of our pre- and post-training datasets on Hugging Face. 

---
# Knowledge Graph-Infused Fine-Tuning for Structured Reasoning in Large Language Models 

**Authors**: Wuyang Zhang, Yexin Tian, Xiandong Meng, Mengjie Wang, Junliang Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.14427)  

**Abstract**: This paper addresses the problems of missing reasoning chains and insufficient entity-level semantic understanding in large language models when dealing with tasks that require structured knowledge. It proposes a fine-tuning algorithm framework based on knowledge graph injection. The method builds on pretrained language models and introduces structured graph information for auxiliary learning. A graph neural network is used to encode entities and their relations, constructing a graph-based semantic representation. A fusion mechanism is then designed to jointly model the knowledge graph embeddings with the contextual representations from the language model. To enhance the robustness of knowledge integration, a gating mechanism is introduced to dynamically balance the contributions of linguistic semantics and structural knowledge. This effectively mitigates conflicts between different representational spaces. During training, a joint loss function is constructed to account for both task performance and structural alignment objectives. This helps improve the accuracy of entity prediction and semantic reasoning. The study also includes a series of systematic sensitivity experiments. It evaluates the effects of learning rate, graph coverage, and structural perturbations on model performance. The results further validate the effectiveness and stability of the proposed method across tasks such as entity recognition, question answering, and language generation. Experimental findings show that the proposed structure-aware fine-tuning framework significantly enhances the model's ability to represent complex semantic units. It demonstrates better semantic consistency and contextual logic modeling in scenarios involving structural reasoning and entity extraction. 

---
# Cognitive Surgery: The Awakening of Implicit Territorial Awareness in LLMs 

**Authors**: Yinghan Zhou, Weifeng Zhu, Juan Wen, Wanli Peng, Zhengxian Wu, Yiming Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.14408)  

**Abstract**: Large language models (LLMs) have been shown to possess a degree of self-recognition capability-the ability to identify whether a given text was generated by themselves. Prior work has demonstrated that this capability is reliably expressed under the Pair Presentation Paradigm (PPP), where the model is presented with two texts and asked to choose which one it authored. However, performance deteriorates sharply under the Individual Presentation Paradigm (IPP), where the model is given a single text to judge authorship. Although this phenomenon has been observed, its underlying causes have not been systematically analyzed. In this paper, we first replicate existing findings to confirm that LLMs struggle to distinguish self- from other-generated text under IPP. We then investigate the reasons for this failure and attribute it to a phenomenon we term Implicit Territorial Awareness (ITA)-the model's latent ability to distinguish self- and other-texts in representational space, which remains unexpressed in its output behavior. To awaken the ITA of LLMs, we propose Cognitive Surgery (CoSur), a novel framework comprising four main modules: representation extraction, territory construction, authorship discrimination and cognitive editing. Experimental results demonstrate that our proposed method improves the performance of three different LLMs in the IPP scenario, achieving average accuracies of 83.25%, 66.19%, and 88.01%, respectively. 

---
# DEPTH: Hallucination-Free Relation Extraction via Dependency-Aware Sentence Simplification and Two-tiered Hierarchical Refinement 

**Authors**: Yupei Yang, Fan Feng, Lin Yang, Wanxi Deng, Lin Qu, Biwei Huang, Shikui Tu, Lei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14391)  

**Abstract**: Relation extraction enables the construction of structured knowledge for many downstream applications. While large language models (LLMs) have shown great promise in this domain, most existing methods concentrate on relation classification, which predicts the semantic relation type between a related entity pair. However, we observe that LLMs often struggle to reliably determine whether a relation exists, especially in cases involving complex sentence structures or intricate semantics, which leads to spurious predictions. Such hallucinations can introduce noisy edges in knowledge graphs, compromising the integrity of structured knowledge and downstream reliability. To address these challenges, we propose DEPTH, a framework that integrates Dependency-aware sEntence simPlification and Two-tiered Hierarchical refinement into the relation extraction pipeline. Given a sentence and its candidate entity pairs, DEPTH operates in two stages: (1) the Grounding module extracts relations for each pair by leveraging their shortest dependency path, distilling the sentence into a minimal yet coherent relational context that reduces syntactic noise while preserving key semantics; (2) the Refinement module aggregates all local predictions and revises them based on a holistic understanding of the sentence, correcting omissions and inconsistencies. We further introduce a causality-driven reward model that mitigates reward hacking by disentangling spurious correlations, enabling robust fine-tuning via reinforcement learning with human feedback. Experiments on six benchmarks demonstrate that DEPTH reduces the average hallucination rate to 7.0\% while achieving a 17.2\% improvement in average F1 score over state-of-the-art baselines. 

---
# Credence Calibration Game? Calibrating Large Language Models through Structured Play 

**Authors**: Ke Fang, Tianyi Zhao, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.14390)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in decision-critical domains, it becomes essential to ensure that their confidence estimates faithfully correspond to their actual correctness. Existing calibration methods have primarily focused on post-hoc adjustments or auxiliary model training; however, many of these approaches necessitate additional supervision or parameter updates. In this work, we propose a novel prompt-based calibration framework inspired by the Credence Calibration Game. Our method establishes a structured interaction loop wherein LLMs receive feedback based on the alignment of their predicted confidence with correctness. Through feedback-driven prompting and natural language summaries of prior performance, our framework dynamically improves model calibration. Extensive experiments across models and game configurations demonstrate consistent improvements in evaluation metrics. Our results highlight the potential of game-based prompting as an effective strategy for LLM calibration. Code and data are available at this https URL. 

---
# ZPD-SCA: Unveiling the Blind Spots of LLMs in Assessing Students' Cognitive Abilities 

**Authors**: Wenhan Dong, Zhen Sun, Yuemeng Zhao, Zifan Peng, Jun Wu, Jingyi Zheng, Yule Liu, Xinlei He, Yu Wang, Ruiming Wang, Xinyi Huang, Lei Mo  

**Link**: [PDF](https://arxiv.org/pdf/2508.14377)  

**Abstract**: Large language models (LLMs) have demonstrated potential in educational applications, yet their capacity to accurately assess the cognitive alignment of reading materials with students' developmental stages remains insufficiently explored. This gap is particularly critical given the foundational educational principle of the Zone of Proximal Development (ZPD), which emphasizes the need to match learning resources with Students' Cognitive Abilities (SCA). Despite the importance of this alignment, there is a notable absence of comprehensive studies investigating LLMs' ability to evaluate reading comprehension difficulty across different student age groups, especially in the context of Chinese language education. To fill this gap, we introduce ZPD-SCA, a novel benchmark specifically designed to assess stage-level Chinese reading comprehension difficulty. The benchmark is annotated by 60 Special Grade teachers, a group that represents the top 0.15% of all in-service teachers nationwide. Experimental results reveal that LLMs perform poorly in zero-shot learning scenarios, with Qwen-max and GLM even falling below the probability of random guessing. When provided with in-context examples, LLMs performance improves substantially, with some models achieving nearly double the accuracy of their zero-shot baselines. These results reveal that LLMs possess emerging abilities to assess reading difficulty, while also exposing limitations in their current training for educationally aligned judgment. Notably, even the best-performing models display systematic directional biases, suggesting difficulties in accurately aligning material difficulty with SCA. Furthermore, significant variations in model performance across different genres underscore the complexity of task. We envision that ZPD-SCA can provide a foundation for evaluating and improving LLMs in cognitively aligned educational applications. 

---
# ISCA: A Framework for Interview-Style Conversational Agents 

**Authors**: Charles Welch, Allison Lahnala, Vasudha Varadarajan, Lucie Flek, Rada Mihalcea, J. Lomax Boyd, João Sedoc  

**Link**: [PDF](https://arxiv.org/pdf/2508.14344)  

**Abstract**: We present a low-compute non-generative system for implementing interview-style conversational agents which can be used to facilitate qualitative data collection through controlled interactions and quantitative analysis. Use cases include applications to tracking attitude formation or behavior change, where control or standardization over the conversational flow is desired. We show how our system can be easily adjusted through an online administrative panel to create new interviews, making the tool accessible without coding. Two case studies are presented as example applications, one regarding the Expressive Interviewing system for COVID-19 and the other a semi-structured interview to survey public opinion on emerging neurotechnology. Our code is open-source, allowing others to build off of our work and develop extensions for additional functionality. 

---
# Beyond Semantic Similarity: Reducing Unnecessary API Calls via Behavior-Aligned Retriever 

**Authors**: Yixin Chen, Ying Xiong, Shangyu Wu, Yufei Cui, Xue Liu, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.14323)  

**Abstract**: Tool-augmented large language models (LLMs) leverage external functions to extend their capabilities, but inaccurate function calls can lead to inefficiencies and increased this http URL methods address this challenge by fine-tuning LLMs or using demonstration-based prompting, yet they often suffer from high training overhead and fail to account for inconsistent demonstration samples, which misguide the model's invocation behavior. In this paper, we trained a behavior-aligned retriever (BAR), which provides behaviorally consistent demonstrations to help LLMs make more accurate tool-using decisions. To train the BAR, we construct a corpus including different function-calling behaviors, i.e., calling or this http URL use the contrastive learning framework to train the BAR with customized positive/negative pairs and a dual-negative contrastive loss, ensuring robust retrieval of behaviorally consistent this http URL demonstrate that our approach significantly reduces erroneous function calls while maintaining high task performance, offering a cost-effective and efficient solution for tool-augmented LLMs. 

---
# SurveyGen-I: Consistent Scientific Survey Generation with Evolving Plans and Memory-Guided Writing 

**Authors**: Jing Chen, Zhiheng Yang, Yixian Shen, Jie Liu, Adam Belloum, Chrysa Papagainni, Paola Grosso  

**Link**: [PDF](https://arxiv.org/pdf/2508.14317)  

**Abstract**: Survey papers play a critical role in scientific communication by consolidating progress across a field. Recent advances in Large Language Models (LLMs) offer a promising solution by automating key steps in the survey-generation pipeline, such as retrieval, structuring, and summarization. However, existing LLM-based approaches often struggle with maintaining coherence across long, multi-section surveys and providing comprehensive citation coverage. To address these limitations, we introduce SurveyGen-I, an automatic survey generation framework that combines coarse-to-fine retrieval, adaptive planning, and memory-guided generation. SurveyGen-I first performs survey-level retrieval to construct the initial outline and writing plan, and then dynamically refines both during generation through a memory mechanism that stores previously written content and terminology, ensuring coherence across subsections. When the system detects insufficient context, it triggers fine-grained subsection-level retrieval. During generation, SurveyGen-I leverages this memory mechanism to maintain coherence across subsections. Experiments across four scientific domains demonstrate that SurveyGen-I consistently outperforms previous works in content quality, consistency, and citation coverage. 

---
# Zero-knowledge LLM hallucination detection and mitigation through fine-grained cross-model consistency 

**Authors**: Aman Goel, Daniel Schwartz, Yanjun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14314)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, but they remain susceptible to hallucinations--generating content that appears plausible but contains factual inaccuracies. We present Finch-Zk, a black-box framework that leverages FINe-grained Cross-model consistency to detect and mitigate Hallucinations in LLM outputs without requiring external knowledge sources. Finch-Zk introduces two key innovations: 1) a cross-model consistency checking strategy that reveals fine-grained inaccuracies by comparing responses generated by diverse models from semantically-equivalent prompts, and 2) a targeted mitigation technique that applies precise corrections to problematic segments while preserving accurate content. Experiments on the FELM dataset show Finch-Zk improves hallucination detection F1 scores by 6-39\% compared to existing approaches. For mitigation, Finch-Zk achieves 7-8 absolute percentage points improvement in answer accuracy on the GPQA-diamond dataset when applied to state-of-the-art models like Llama 4 Maverick and Claude 4 Sonnet. Extensive evaluation across multiple models demonstrates that Finch-Zk provides a practical, deployment-ready safeguard for enhancing factual reliability in production LLM systems. 

---
# A Joint Multitask Model for Morpho-Syntactic Parsing 

**Authors**: Demian Inostroza, Mel Mistica, Ekaterina Vylomova, Chris Guest, Kemal Kurniawan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14307)  

**Abstract**: We present a joint multitask model for the UniDive 2025 Morpho-Syntactic Parsing shared task, where systems predict both morphological and syntactic analyses following novel UD annotation scheme. Our system uses a shared XLM-RoBERTa encoder with three specialized decoders for content word identification, dependency parsing, and morphosyntactic feature prediction. Our model achieves the best overall performance on the shared task's leaderboard covering nine typologically diverse languages, with an average MSLAS score of 78.7 percent, LAS of 80.1 percent, and Feats F1 of 90.3 percent. Our ablation studies show that matching the task's gold tokenization and content word identification are crucial to model performance. Error analysis reveals that our model struggles with core grammatical cases (particularly Nom-Acc) and nominal features across languages. 

---
# Tokens with Meaning: A Hybrid Tokenization Approach for NLP 

**Authors**: M. Ali Bayram, Ali Arda Fincan, Ahmet Semih Gümüş, Sercan Karakaş, Banu Diri, Savaş Yıldırım, Demircan Çelik  

**Link**: [PDF](https://arxiv.org/pdf/2508.14292)  

**Abstract**: Tokenization plays a pivotal role in natural language processing (NLP), shaping how text is segmented and interpreted by language models. While subword methods such as Byte Pair Encoding (BPE) and WordPiece have been effective, they often struggle with morphologically rich and agglutinative languages because they rely on frequency rather than linguistic structure. We introduce a hybrid tokenization framework that combines rule-based morphological analysis with statistical subword segmentation. The method uses phonological normalization, root-affix dictionaries, and a novel algorithm that balances morpheme preservation with vocabulary efficiency. It assigns shared identifiers to phonologically variant affixes (e.g., -ler and -lar) and altered root forms (e.g., kitap vs. kitabı), reducing redundancy while maintaining semantic integrity. Special tokens are added for whitespace and case, including an UPPERCASE marker to avoid vocabulary inflation from capitalization. BPE is integrated for out-of-vocabulary coverage without harming morphological coherence. On the TR-MMLU benchmark, the tokenizer achieves the highest Turkish Token Percentage (90.29\%) and Pure Token Percentage (85.8\%). Comparisons with tokenizers from LLaMA, Gemma, and GPT show more linguistically meaningful and coherent tokens. Although demonstrated on Turkish, the approach is language-independent and adaptable to other languages, offering a practical path toward more interpretable and effective multilingual NLP systems. 

---
# GRILE: A Benchmark for Grammar Reasoning and Explanation in Romanian LLMs 

**Authors**: Adrian-Marius Dumitran, Alexandra-Mihaela Danila, Angela-Liliana Dumitran  

**Link**: [PDF](https://arxiv.org/pdf/2508.14279)  

**Abstract**: LLMs (Large language models) have revolutionized NLP (Natural Language Processing), yet their pedagogical value for low-resource languages remains unclear. We present GRILE (Grammar Romanian Inference and Language Explanations) , the first open benchmark of 1,151 multiple-choice questions harvested from Romanian high-stakes exams (National Evaluation, Baccalaureate, university admissions). GRILE enables us to probe two complementary abilities of seven state-of-the-art multilingual and Romanian-specific LLMs: (i) selecting the correct answer, and (ii) producing linguistically accurate explanations. While Gemini 2.5 Pro reaches 83% accuracy, most open-weight models stay below 65%, and 48% of their explanations contain factual or pedagogical flaws according to expert review. A detailed error analysis pinpoints systematic weaknesses in morphology and in applying the latest DOOM3 orthographic norms. All data, code and a public web demo are released to catalyze future research. Our findings expose open challenges for trustworthy educational NLP in low-resource settings and establish GRILE as a new test-bed for controllable explanation generation and evaluation. 

---
# Disentangling concept semantics via multilingual averaging in Sparse Autoencoders 

**Authors**: Cliff O'Reilly, Ernesto Jimenez-Ruiz, Tillman Weyde  

**Link**: [PDF](https://arxiv.org/pdf/2508.14275)  

**Abstract**: Connecting LLMs with formal knowledge representation and reasoning is a promising approach to address their shortcomings. Embeddings and sparse autoencoders are widely used to represent textual content, but the semantics are entangled with syntactic and language-specific information. We propose a method that isolates concept semantics in Large Langue Models by averaging concept activations derived via Sparse Autoencoders. We create English text representations from OWL ontology classes, translate the English into French and Chinese and then pass these texts as prompts to the Gemma 2B LLM. Using the open source Gemma Scope suite of Sparse Autoencoders, we obtain concept activations for each class and language version. We average the different language activations to derive a conceptual average. We then correlate the conceptual averages with a ground truth mapping between ontology classes. Our results give a strong indication that the conceptual average aligns to the true relationship between classes when compared with a single language by itself. The result hints at a new technique which enables mechanistic interpretation of internal network states with higher accuracy. 

---
# Let's Use ChatGPT To Write Our Paper! Benchmarking LLMs To Write the Introduction of a Research Paper 

**Authors**: Krishna Garg, Firoz Shaikh, Sambaran Bandyopadhyay, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2508.14273)  

**Abstract**: As researchers increasingly adopt LLMs as writing assistants, generating high-quality research paper introductions remains both challenging and essential. We introduce Scientific Introduction Generation (SciIG), a task that evaluates LLMs' ability to produce coherent introductions from titles, abstracts, and related works. Curating new datasets from NAACL 2025 and ICLR 2025 papers, we assess five state-of-the-art models, including both open-source (DeepSeek-v3, Gemma-3-12B, LLaMA 4-Maverick, MistralAI Small 3.1) and closed-source GPT-4o systems, across multiple dimensions: lexical overlap, semantic similarity, content coverage, faithfulness, consistency, citation correctness, and narrative quality. Our comprehensive framework combines automated metrics with LLM-as-a-judge evaluations. Results demonstrate LLaMA-4 Maverick's superior performance on most metrics, particularly in semantic similarity and faithfulness. Moreover, three-shot prompting consistently outperforms fewer-shot approaches. These findings provide practical insights into developing effective research writing assistants and set realistic expectations for LLM-assisted academic writing. To foster reproducibility and future research, we will publicly release all code and datasets. 

---
# Comparing energy consumption and accuracy in text classification inference 

**Authors**: Johannes Zschache, Tilman Hartwig  

**Link**: [PDF](https://arxiv.org/pdf/2508.14170)  

**Abstract**: The increasing deployment of large language models (LLMs) in natural language processing (NLP) tasks raises concerns about energy efficiency and sustainability. While prior research has largely focused on energy consumption during model training, the inference phase has received comparatively less attention. This study systematically evaluates the trade-offs between model accuracy and energy consumption in text classification inference across various model architectures and hardware configurations. Our empirical analysis shows that the best-performing model in terms of accuracy can also be energy-efficient, while larger LLMs tend to consume significantly more energy with lower classification accuracy. We observe substantial variability in inference energy consumption ($<$mWh to $>$kWh), influenced by model type, model size, and hardware specifications. Additionally, we find a strong correlation between inference energy consumption and model runtime, indicating that execution time can serve as a practical proxy for energy usage in settings where direct measurement is not feasible. These findings have implications for sustainable AI development, providing actionable insights for researchers, industry practitioners, and policymakers seeking to balance performance and resource efficiency in NLP applications. 

---
# DPad: Efficient Diffusion Language Models with Suffix Dropout 

**Authors**: Xinhua Chen, Sitao Huang, Cong Guo, Chiyue Wei, Yintao He, Jianyi Zhang, Hai "Hellen" Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.14148)  

**Abstract**: Diffusion-based Large Language Models (dLLMs) parallelize text generation by framing decoding as a denoising process, but suffer from high computational overhead since they predict all future suffix tokens at each step while retaining only a small fraction. We propose Diffusion Scratchpad (DPad), a training-free method that restricts attention to a small set of nearby suffix tokens, preserving fidelity while eliminating redundancy. DPad integrates two strategies: (i) a sliding window, which maintains a fixed-length suffix window, and (ii) distance-decay dropout, which deterministically removes distant suffix tokens before attention computation. This simple design is compatible with existing optimizations such as prefix caching and can be implemented with only a few lines of code. Comprehensive evaluations across multiple benchmarks on LLaDA-1.5 and Dream models demonstrate that DPad delivers up to $\mathbf{61.4\times}$ speedup over vanilla dLLMs while maintaining comparable accuracy, highlighting its potential for efficient and scalable long-sequence inference. Our code is available at this https URL. 

---
# MMReview: A Multidisciplinary and Multimodal Benchmark for LLM-Based Peer Review Automation 

**Authors**: Xian Gao, Jiacheng Ruan, Zongyun Zhang, Jingsheng Gao, Ting Liu, Yuzhuo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14146)  

**Abstract**: With the rapid growth of academic publications, peer review has become an essential yet time-consuming responsibility within the research community. Large Language Models (LLMs) have increasingly been adopted to assist in the generation of review comments; however, current LLM-based review tasks lack a unified evaluation benchmark to rigorously assess the models' ability to produce comprehensive, accurate, and human-aligned assessments, particularly in scenarios involving multimodal content such as figures and tables. To address this gap, we propose \textbf{MMReview}, a comprehensive benchmark that spans multiple disciplines and modalities. MMReview includes multimodal content and expert-written review comments for 240 papers across 17 research domains within four major academic disciplines: Artificial Intelligence, Natural Sciences, Engineering Sciences, and Social Sciences. We design a total of 13 tasks grouped into four core categories, aimed at evaluating the performance of LLMs and Multimodal LLMs (MLLMs) in step-wise review generation, outcome formulation, alignment with human preferences, and robustness to adversarial input manipulation. Extensive experiments conducted on 16 open-source models and 5 advanced closed-source models demonstrate the thoroughness of the benchmark. We envision MMReview as a critical step toward establishing a standardized foundation for the development of automated peer review systems. 

---
# DLLMQuant: Quantizing Diffusion-based Large Language Models 

**Authors**: Chen Xu, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14090)  

**Abstract**: Diffusion-based large language models (DLLMs) have shown promise for non-autoregressive text generation, but their deployment is constrained by large model sizes and heavy computational costs. Post-training quantization (PTQ), a widely used method for compressing and accelerating Large Language Models (LLMs), suffers from severe accuracy degradation and reduced generalization performance when directly applied to DLLMs (e.g., AWQ suffers a 16% accuracy drop on LLADA under W4A4). This paper explores how DLLMs' key mechanisms - dynamic masking, iterative generation, bidirectional attention - clash with quantization. We identify three core issues: 1) Iterative generation and dynamic masking ratios lead to distinct token distributions across decoding steps, which are not adequately captured by existing PTQ calibration methods; 2) Quantization errors are accumulated and amplified progressively during iteration in DLLMs, causing quantized models to perform worse as decoding steps progress; 3) Unmasked tokens stabilize while masked remain probabilistic, making overall feature distribution incompatible with existing PTQ methods. To address these issues, we propose DLLMQuant, a PTQ framework tailored for DLLMs, which incorporates three novel techniques: 1) Temporal-Mask Adaptive Sampling (TMAS), a calibration method that accounts for both time and mask factors, with the capacity to capture distributions across timesteps. 2) Interaction-Aware Activation Quantization (IA-AQ), which utilizes bidirectional attention's interaction signals to dynamically allocate quantization resources. 3) Certainty-Guided Quantization (CGQ), which integrates mask status and token scores as key weighting criteria into error compensation, making weight quantization more suitable for DLLMs. Experiments show that DLLMQuant achieves significant performance gains while enhancing efficiency. 

---
# Punctuation and Predicates in Language Models 

**Authors**: Sonakshi Chauhan, Maheep Chaudhary, Koby Choy, Samuel Nellessen, Nandi Schoots  

**Link**: [PDF](https://arxiv.org/pdf/2508.14067)  

**Abstract**: In this paper we explore where information is collected and how it is propagated throughout layers in large language models (LLMs). We begin by examining the surprising computational importance of punctuation tokens which previous work has identified as attention sinks and memory aids. Using intervention-based techniques, we evaluate the necessity and sufficiency (for preserving model performance) of punctuation tokens across layers in GPT-2, DeepSeek, and Gemma. Our results show stark model-specific differences: for GPT-2, punctuation is both necessary and sufficient in multiple layers, while this holds far less in DeepSeek and not at all in Gemma. Extending beyond punctuation, we ask whether LLMs process different components of input (e.g., subjects, adjectives, punctuation, full sentences) by forming early static summaries reused across the network, or if the model remains sensitive to changes in these components across layers. Extending beyond punctuation, we investigate whether different reasoning rules are processed differently by LLMs. In particular, through interchange intervention and layer-swapping experiments, we find that conditional statements (if, then), and universal quantification (for all) are processed very differently. Our findings offer new insight into the internal mechanisms of punctuation usage and reasoning in LLMs and have implications for interpretability. 

---
# Assessing and Mitigating Data Memorization Risks in Fine-Tuned Large Language Models 

**Authors**: Badrinath Ramakrishnan, Akshaya Balaji  

**Link**: [PDF](https://arxiv.org/pdf/2508.14062)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse natural language processing tasks, but their tendency to memorize training data poses significant privacy risks, particularly during fine-tuning processes. This paper presents a comprehensive empirical analysis of data memorization in fine-tuned LLMs and introduces a novel multi-layered privacy protection framework. Through controlled experiments on modern LLM architectures including GPT-2, Phi-3, and Gemma-2, we demonstrate that fine-tuning with repeated sensitive data increases privacy leakage rates from baseline levels of 0-5% to 60-75%, representing a 64.2% average increase across tested models. We propose and rigorously evaluate four complementary privacy protection methods: semantic data deduplication, differential privacy during generation, entropy-based filtering, and pattern-based content filtering. Our experimental results show that these techniques can reduce data leakage to 0% while maintaining 94.7% of original model utility. 

---
# Confidence Estimation for Text-to-SQL in Large Language Models 

**Authors**: Sepideh Entezari Maleki, Mohammadreza Pourreza, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2508.14056)  

**Abstract**: Confidence estimation for text-to-SQL aims to assess the reliability of model-generated SQL queries without having access to gold answers. We study this problem in the context of large language models (LLMs), where access to model weights and gradients is often constrained. We explore both black-box and white-box confidence estimation strategies, evaluating their effectiveness on cross-domain text-to-SQL benchmarks. Our evaluation highlights the superior performance of consistency-based methods among black-box models and the advantage of SQL-syntax-aware approaches for interpreting LLM logits in white-box settings. Furthermore, we show that execution-based grounding of queries provides a valuable supplementary signal, improving the effectiveness of both approaches. 

---
# T-REX: Table -- Refute or Entail eXplainer 

**Authors**: Tim Luka Horstmann, Baptiste Geisenberger, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2508.14055)  

**Abstract**: Verifying textual claims against structured tabular data is a critical yet challenging task in Natural Language Processing with broad real-world impact. While recent advances in Large Language Models (LLMs) have enabled significant progress in table fact-checking, current solutions remain inaccessible to non-experts. We introduce T-REX (T-REX: Table -- Refute or Entail eXplainer), the first live, interactive tool for claim verification over multimodal, multilingual tables using state-of-the-art instruction-tuned reasoning LLMs. Designed for accuracy and transparency, T-REX empowers non-experts by providing access to advanced fact-checking technology. The system is openly available online. 

---
# Contrastive Analysis of Constituent Order Preferences Within Adverbial Roles in English and Chinese News: A Large-Language-Model-Driven Approach 

**Authors**: Yiran Rex Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.14054)  

**Abstract**: Based on comparable English-Chinese news corpora annotated by Large Language Model (LLM), this paper attempts to explore the differences in constituent order of English-Chinese news from the perspective of functional chunks with adverbial roles, and analyze their typical positional preferences and distribution patterns. It is found that: (1) English news prefers linear narrative of core information first, and functional chunks are mostly post-positioned, while Chinese news prefers overall presentation mode of background first, and functional chunks are often pre-positioned; (2) In SVO structure, both English and Chinese news show differences in the distribution of functional chunks, but the tendency of Chinese pre-positioning is more significant, while that of English post-positioning is relatively mild; (3) When function blocks are co-occurring, both English and Chinese news show high flexibility, and the order adjustment is driven by information and pragmatic purposes. The study reveals that word order has both systematic preference and dynamic adaptability, providing new empirical support for contrastive study of English-Chinese information structure. 

---
# Benchmarking Sociolinguistic Diversity in Swahili NLP: A Taxonomy-Guided Approach 

**Authors**: Kezia Oketch, John P. Lalor, Ahmed Abbasi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14051)  

**Abstract**: We introduce the first taxonomy-guided evaluation of Swahili NLP, addressing gaps in sociolinguistic diversity. Drawing on health-related psychometric tasks, we collect a dataset of 2,170 free-text responses from Kenyan speakers. The data exhibits tribal influences, urban vernacular, code-mixing, and loanwords. We develop a structured taxonomy and use it as a lens for examining model prediction errors across pre-trained and instruction-tuned language models. Our findings advance culturally grounded evaluation frameworks and highlight the role of sociolinguistic variation in shaping model performance. 

---
# From Image Captioning to Visual Storytelling 

**Authors**: Admitos Passadakis, Yingjin Song, Albert Gatt  

**Link**: [PDF](https://arxiv.org/pdf/2508.14045)  

**Abstract**: Visual Storytelling is a challenging multimodal task between Vision & Language, where the purpose is to generate a story for a stream of images. Its difficulty lies on the fact that the story should be both grounded to the image sequence but also narrative and coherent. The aim of this work is to balance between these aspects, by treating Visual Storytelling as a superset of Image Captioning, an approach quite different compared to most of prior relevant studies. This means that we firstly employ a vision-to-language model for obtaining captions of the input images, and then, these captions are transformed into coherent narratives using language-to-language methods. Our multifarious evaluation shows that integrating captioning and storytelling under a unified framework, has a positive impact on the quality of the produced stories. In addition, compared to numerous previous studies, this approach accelerates training time and makes our framework readily reusable and reproducible by anyone interested. Lastly, we propose a new metric/tool, named ideality, that can be used to simulate how far some results are from an oracle model, and we apply it to emulate human-likeness in visual storytelling. 

---
# Virtual Community: An Open World for Humans, Robots, and Society 

**Authors**: Qinhong Zhou, Hongxin Zhang, Xiangye Lin, Zheyuan Zhang, Yutian Chen, Wenjun Liu, Zunzhe Zhang, Sunli Chen, Lixing Fang, Qiushi Lyu, Xinyu Sun, Jincheng Yang, Zeyuan Wang, Bao Chi Dang, Zhehuan Chen, Daksha Ladia, Jiageng Liu, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14893)  

**Abstract**: The rapid progress in AI and Robotics may lead to a profound societal transformation, as humans and robots begin to coexist within shared communities, introducing both opportunities and challenges. To explore this future, we present Virtual Community-an open-world platform for humans, robots, and society-built on a universal physics engine and grounded in real-world 3D scenes. With Virtual Community, we aim to study embodied social intelligence at scale: 1) How robots can intelligently cooperate or compete; 2) How humans develop social relations and build community; 3) More importantly, how intelligent robots and humans can co-exist in an open world. To support these, Virtual Community features: 1) An open-source multi-agent physics simulator that supports robots, humans, and their interactions within a society; 2) A large-scale, real-world aligned community generation pipeline, including vast outdoor space, diverse indoor scenes, and a community of grounded agents with rich characters and appearances. Leveraging Virtual Community, we propose two novel challenges. The Community Planning Challenge evaluates multi-agent reasoning and planning ability in open-world settings, such as cooperating to help agents with daily activities and efficiently connecting other agents. The Community Robot Challenge requires multiple heterogeneous robots to collaborate in solving complex open-world tasks. We evaluate various baselines on these tasks and demonstrate the challenges in both high-level open-world task planning and low-level cooperation controls. We hope that Virtual Community will unlock further study of human-robot coexistence within open-world environments. 

---
# The Prompting Brain: Neurocognitive Markers of Expertise in Guiding Large Language Models 

**Authors**: Hend Al-Khalifa, Raneem Almansour, Layan Abdulrahman Alhuasini, Alanood Alsaleh, Mohamad-Hani Temsah, Mohamad-Hani_Temsah, Ashwag Rafea S Alruwaili  

**Link**: [PDF](https://arxiv.org/pdf/2508.14869)  

**Abstract**: Prompt engineering has rapidly emerged as a critical skill for effective interaction with large language models (LLMs). However, the cognitive and neural underpinnings of this expertise remain largely unexplored. This paper presents findings from a cross-sectional pilot fMRI study investigating differences in brain functional connectivity and network activity between experts and intermediate prompt engineers. Our results reveal distinct neural signatures associated with higher prompt engineering literacy, including increased functional connectivity in brain regions such as the left middle temporal gyrus and the left frontal pole, as well as altered power-frequency dynamics in key cognitive networks. These findings offer initial insights into the neurobiological basis of prompt engineering proficiency. We discuss the implications of these neurocognitive markers in Natural Language Processing (NLP). Understanding the neural basis of human expertise in interacting with LLMs can inform the design of more intuitive human-AI interfaces, contribute to cognitive models of LLM interaction, and potentially guide the development of AI systems that better align with human cognitive workflows. This interdisciplinary approach aims to bridge the gap between human cognition and machine intelligence, fostering a deeper understanding of how humans learn and adapt to complex AI systems. 

---
# Privileged Self-Access Matters for Introspection in AI 

**Authors**: Siyuan Song, Harvey Lederman, Jennifer Hu, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2508.14802)  

**Abstract**: Whether AI models can introspect is an increasingly important practical question. But there is no consensus on how introspection is to be defined. Beginning from a recently proposed ''lightweight'' definition, we argue instead for a thicker one. According to our proposal, introspection in AI is any process which yields information about internal states through a process more reliable than one with equal or lower computational cost available to a third party. Using experiments where LLMs reason about their internal temperature parameters, we show they can appear to have lightweight introspection while failing to meaningfully introspect per our proposed definition. 

---
# MCP-Universe: Benchmarking Large Language Models with Real-World Model Context Protocol Servers 

**Authors**: Ziyang Luo, Zhiqi Shen, Wenzhuo Yang, Zirui Zhao, Prathyusha Jwalapuram, Amrita Saha, Doyen Sahoo, Silvio Savarese, Caiming Xiong, Junnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14704)  

**Abstract**: The Model Context Protocol has emerged as a transformative standard for connecting large language models to external data sources and tools, rapidly gaining adoption across major AI providers and development platforms. However, existing benchmarks are overly simplistic and fail to capture real application challenges such as long-horizon reasoning and large, unfamiliar tool spaces. To address this critical gap, we introduce MCP-Universe, the first comprehensive benchmark specifically designed to evaluate LLMs in realistic and hard tasks through interaction with real-world MCP servers. Our benchmark encompasses 6 core domains spanning 11 different MCP servers: Location Navigation, Repository Management, Financial Analysis, 3D Design, Browser Automation, and Web Searching. To ensure rigorous evaluation, we implement execution-based evaluators, including format evaluators for agent format compliance, static evaluators for time-invariant content matching, and dynamic evaluators that automatically retrieve real-time ground truth for temporally sensitive tasks. Through extensive evaluation of leading LLMs, we find that even SOTA models such as GPT-5 (43.72%), Grok-4 (33.33%) and Claude-4.0-Sonnet (29.44%) exhibit significant performance limitations. In addition, our benchmark poses a significant long-context challenge for LLM agents, as the number of input tokens increases rapidly with the number of interaction steps. Moreover, it introduces an unknown-tools challenge, as LLM agents often lack familiarity with the precise usage of the MCP servers. Notably, enterprise-level agents like Cursor cannot achieve better performance than standard ReAct frameworks. Beyond evaluation, we open-source our extensible evaluation framework with UI support, enabling researchers and practitioners to seamlessly integrate new agents and MCP servers while fostering innovation in the rapidly evolving MCP ecosystem. 

---
# Who Sees What? Structured Thought-Action Sequences for Epistemic Reasoning in LLMs 

**Authors**: Luca Annese, Sabrina Patania, Silvia Serino, Tom Foulsham, Silvia Rossi, Azzurra Ruggeri, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2508.14564)  

**Abstract**: Recent advances in large language models (LLMs) and reasoning frameworks have opened new possibilities for improving the perspective -taking capabilities of autonomous agents. However, tasks that involve active perception, collaborative reasoning, and perspective taking (understanding what another agent can see or knows) pose persistent challenges for current LLM-based systems. This study investigates the potential of structured examples derived from transformed solution graphs generated by the Fast Downward planner to improve the performance of LLM-based agents within a ReAct framework. We propose a structured solution-processing pipeline that generates three distinct categories of examples: optimal goal paths (G-type), informative node paths (E-type), and step-by-step optimal decision sequences contrasting alternative actions (L-type). These solutions are further converted into ``thought-action'' examples by prompting an LLM to explicitly articulate the reasoning behind each decision. While L-type examples slightly reduce clarification requests and overall action steps, they do not yield consistent improvements. Agents are successful in tasks requiring basic attentional filtering but struggle in scenarios that required mentalising about occluded spaces or weighing the costs of epistemic actions. These findings suggest that structured examples alone are insufficient for robust perspective-taking, underscoring the need for explicit belief tracking, cost modelling, and richer environments to enable socially grounded collaboration in LLM-based agents. 

---
# DuPO: Enabling Reliable LLM Self-Verification via Dual Preference Optimization 

**Authors**: Shuaijie She, Yu Bao, Yu Lu, Lu Xu, Tao Li, Wenhao Zhu, Shujian Huang, Shanbo Cheng, Lu Lu, Yuxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14460)  

**Abstract**: We present DuPO, a dual learning-based preference optimization framework that generates annotation-free feedback via a generalized duality. DuPO addresses two key limitations: Reinforcement Learning with Verifiable Rewards (RLVR)'s reliance on costly labels and applicability restricted to verifiable tasks, and traditional dual learning's restriction to strictly dual task pairs (e.g., translation and back-translation). Specifically, DuPO decomposes a primal task's input into known and unknown components, then constructs its dual task to reconstruct the unknown part using the primal output and known information (e.g., reversing math solutions to recover hidden variables), broadening applicability to non-invertible tasks. The quality of this reconstruction serves as a self-supervised reward to optimize the primal task, synergizing with LLMs' ability to instantiate both tasks via a single model. Empirically, DuPO achieves substantial gains across diverse tasks: it enhances the average translation quality by 2.13 COMET over 756 directions, boosts the mathematical reasoning accuracy by an average of 6.4 points on three challenge benchmarks, and enhances performance by 9.3 points as an inference-time reranker (trading computation for accuracy). These results position DuPO as a scalable, general, and annotation-free paradigm for LLM optimization. 

---
# GLASS: Test-Time Acceleration for LLMs via Global-Local Neural Importance Aggregation 

**Authors**: Amirmohsen Sattarifard, Sepehr Lavasani, Ehsan Imani, Kunlin Zhang, Hanlin Xu, Fengyu Sun, Negar Hassanpour, Chao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.14302)  

**Abstract**: Deploying Large Language Models (LLMs) on edge hardware demands aggressive, prompt-aware dynamic pruning to reduce computation without degrading quality. Static or predictor-based schemes either lock in a single sparsity pattern or incur extra runtime overhead, and recent zero-shot methods that rely on statistics from a single prompt fail on short prompt and/or long generation scenarios. We introduce A/I-GLASS: Activation- and Impact-based Global-Local neural importance Aggregation for feed-forward network SparSification, two training-free methods that dynamically select FFN units using a rank-aggregation of prompt local and model-intrinsic global neuron statistics. Empirical results across multiple LLMs and benchmarks demonstrate that GLASS significantly outperforms prior training-free methods, particularly in challenging long-form generation scenarios, without relying on auxiliary predictors or adding any inference overhead. 

---
# MultiFuzz: A Dense Retrieval-based Multi-Agent System for Network Protocol Fuzzing 

**Authors**: Youssef Maklad, Fares Wael, Ali Hamdi, Wael Elsersy, Khaled Shaban  

**Link**: [PDF](https://arxiv.org/pdf/2508.14300)  

**Abstract**: Traditional protocol fuzzing techniques, such as those employed by AFL-based systems, often lack effectiveness due to a limited semantic understanding of complex protocol grammars and rigid seed mutation strategies. Recent works, such as ChatAFL, have integrated Large Language Models (LLMs) to guide protocol fuzzing and address these limitations, pushing protocol fuzzers to wider exploration of the protocol state space. But ChatAFL still faces issues like unreliable output, LLM hallucinations, and assumptions of LLM knowledge about protocol specifications. This paper introduces MultiFuzz, a novel dense retrieval-based multi-agent system designed to overcome these limitations by integrating semantic-aware context retrieval, specialized agents, and structured tool-assisted reasoning. MultiFuzz utilizes agentic chunks of protocol documentation (RFC Documents) to build embeddings in a vector database for a retrieval-augmented generation (RAG) pipeline, enabling agents to generate more reliable and structured outputs, enhancing the fuzzer in mutating protocol messages with enhanced state coverage and adherence to syntactic constraints. The framework decomposes the fuzzing process into modular groups of agents that collaborate through chain-of-thought reasoning to dynamically adapt fuzzing strategies based on the retrieved contextual knowledge. Experimental evaluations on the Real-Time Streaming Protocol (RTSP) demonstrate that MultiFuzz significantly improves branch coverage and explores deeper protocol states and transitions over state-of-the-art (SOTA) fuzzers such as NSFuzz, AFLNet, and ChatAFL. By combining dense retrieval, agentic coordination, and language model reasoning, MultiFuzz establishes a new paradigm in autonomous protocol fuzzing, offering a scalable and extensible foundation for future research in intelligent agentic-based fuzzing systems. 

---
# Measuring LLM Code Generation Stability via Structural Entropy 

**Authors**: Yewei Song, Tiezhu Sun, Xunzhu Tang, Prateek Rajput, Tegawende F. Bissyande, Jacques Klein  

**Link**: [PDF](https://arxiv.org/pdf/2508.14288)  

**Abstract**: Assessing the stability of code generation from large language models (LLMs) is essential for judging their reliability in real-world development. We extend prior "structural-entropy concepts" to the program domain by pairing entropy with abstract syntax tree (AST) analysis. For any fixed prompt, we collect the multiset of depth-bounded subtrees of AST in each generated program and treat their relative frequencies as a probability distribution. We then measure stability in two complementary ways: (i) Jensen-Shannon divergence, a symmetric, bounded indicator of structural overlap, and (ii) a Structural Cross-Entropy ratio that highlights missing high-probability patterns. Both metrics admit structural-only and token-aware variants, enabling separate views on control-flow shape and identifier-level variability. Unlike pass@k, BLEU, or CodeBLEU, our metrics are reference-free, language-agnostic, and execution-independent. We benchmark several leading LLMs on standard code generation tasks, demonstrating that AST-driven structural entropy reveals nuances in model consistency and robustness. The method runs in O(n,d) time with no external tests, providing a lightweight addition to the code-generation evaluation toolkit. 

---
# Two Birds with One Stone: Multi-Task Detection and Attribution of LLM-Generated Text 

**Authors**: Zixin Rao, Youssef Mohamed, Shang Liu, Zeyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14190)  

**Abstract**: Large Language Models (LLMs), such as GPT-4 and Llama, have demonstrated remarkable abilities in generating natural language. However, they also pose security and integrity challenges. Existing countermeasures primarily focus on distinguishing AI-generated content from human-written text, with most solutions tailored for English. Meanwhile, authorship attribution--determining which specific LLM produced a given text--has received comparatively little attention despite its importance in forensic analysis. In this paper, we present DA-MTL, a multi-task learning framework that simultaneously addresses both text detection and authorship attribution. We evaluate DA-MTL on nine datasets and four backbone models, demonstrating its strong performance across multiple languages and LLM sources. Our framework captures each task's unique characteristics and shares insights between them, which boosts performance in both tasks. Additionally, we conduct a thorough analysis of cross-modal and cross-lingual patterns and assess the framework's robustness against adversarial obfuscation techniques. Our findings offer valuable insights into LLM behavior and the generalization of both detection and authorship attribution. 

---
# FinAgentBench: A Benchmark Dataset for Agentic Retrieval in Financial Question Answering 

**Authors**: Chanyeol Choi, Jihoon Kwon, Alejandro Lopez-Lira, Chaewoon Kim, Minjae Kim, Juneha Hwang, Jaeseon Ha, Hojun Choi, Suyeol Yun, Yongjin Kim, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.14052)  

**Abstract**: Accurate information retrieval (IR) is critical in the financial domain, where investors must identify relevant information from large collections of documents. Traditional IR methods-whether sparse or dense-often fall short in retrieval accuracy, as it requires not only capturing semantic similarity but also performing fine-grained reasoning over document structure and domain-specific knowledge. Recent advances in large language models (LLMs) have opened up new opportunities for retrieval with multi-step reasoning, where the model ranks passages through iterative reasoning about which information is most relevant to a given query. However, there exists no benchmark to evaluate such capabilities in the financial domain. To address this gap, we introduce FinAgentBench, the first large-scale benchmark for evaluating retrieval with multi-step reasoning in finance -- a setting we term agentic retrieval. The benchmark consists of 3,429 expert-annotated examples on S&P-100 listed firms and assesses whether LLM agents can (1) identify the most relevant document type among candidates, and (2) pinpoint the key passage within the selected document. Our evaluation framework explicitly separates these two reasoning steps to address context limitations. This design enables to provide a quantitative basis for understanding retrieval-centric LLM behavior in finance. We evaluate a suite of state-of-the-art models and further demonstrated how targeted fine-tuning can significantly improve agentic retrieval performance. Our benchmark provides a foundation for studying retrieval-centric LLM behavior in complex, domain-specific tasks for finance. We will release the dataset publicly upon acceptance of the paper and plan to expand and share dataset for the full S&P 500 and beyond. 

---
# MahaTTS: A Unified Framework for Multilingual Text-to-Speech Synthesis 

**Authors**: Jaskaran Singh, Amartya Roy Chowdhury, Raghav Prabhakar, Varshul C. W  

**Link**: [PDF](https://arxiv.org/pdf/2508.14049)  

**Abstract**: Current Text-to-Speech models pose a multilingual challenge, where most of the models traditionally focus on English and European languages, thereby hurting the potential to provide access to information to many more people. To address this gap, we introduce MahaTTS-v2 a Multilingual Multi-speaker Text-To-Speech (TTS) system that has excellent multilingual expressive capabilities in Indic languages. The model has been trained on around 20K hours of data specifically focused on Indian languages. Our approach leverages Wav2Vec2.0 tokens for semantic extraction, and a Language Model (LM) for text-to-semantic modeling. Additionally, we have used a Conditional Flow Model (CFM) for semantics to melspectogram generation. The experimental results indicate the effectiveness of the proposed approach over other frameworks. Our code is available at this https URL 

---
# RAG-Boost: Retrieval-Augmented Generation Enhanced LLM-based Speech Recognition 

**Authors**: Pengcheng Wang, Sheng Li, Takahiro Shinozaki  

**Link**: [PDF](https://arxiv.org/pdf/2508.14048)  

**Abstract**: In this paper, we propose RAG-Boost (ST-ShinozakiLab Task I system), which enhances the baseline LLM-based ASR system of the MLC-SLM Challenge (task I) with a retrieval-augmented generation (RAG) module on the fly. Each partial ASR hypothesis queries a vector store of audio-text pairs and domain terms, and the retrieved results are fused with the live ASR hypotheses to fix recognition errors. The fused hypotheses are passed to the LLM, yielding improved responses. 

---
