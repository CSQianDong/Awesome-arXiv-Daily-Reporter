# Think Twice: Branch-and-Rethink Reasoning Reward Model 

**Authors**: Yizhu Jiao, Jiaqi Zeng, Julien Veron Vialard, Oleksii Kuchaiev, Jiawei Han, Olivier Delalleau  

**Link**: [PDF](https://arxiv.org/pdf/2510.23596)  

**Abstract**: Large language models (LLMs) increasingly rely on thinking models that externalize intermediate steps and allocate extra test-time compute, with think-twice strategies showing that a deliberate second pass can elicit stronger reasoning. In contrast, most reward models (RMs) still compress many quality dimensions into a single scalar in one shot, a design that induces judgment diffusion: attention spreads across evaluation criteria, yielding diluted focus and shallow analysis. We introduce branch-and-rethink (BR-RM), a two-turn RM that transfers the think-twice principle to reward modeling. Turn 1 performs adaptive branching, selecting a small set of instance-critical dimensions (such as factuality and safety) and sketching concise, evidence-seeking hypotheses. Turn 2 executes branch-conditioned rethinking, a targeted reread that tests those hypotheses and scrutinizes only what matters most. We train with GRPO-style reinforcement learning over structured two-turn traces using a simple binary outcome reward with strict format checks, making the approach compatible with standard RLHF pipelines. By converting all-at-oncescoringintofocused, second-lookreasoning, BR-RMreducesjudgmentdiffusionandimproves sensitivity to subtle yet consequential errors while remaining practical and scalable. Experimental results demonstrate that our model achieves state-of-the-art performance on three challenging reward modeling benchmarks across diverse domains. The code and the model will be released soon. 

---
# Hope Speech Detection in Social Media English Corpora: Performance of Traditional and Transformer Models 

**Authors**: Luis Ramos, Hiram Calvo, Olga Kolesnikova  

**Link**: [PDF](https://arxiv.org/pdf/2510.23585)  

**Abstract**: The identification of hope speech has become a promised NLP task, considering the need to detect motivational expressions of agency and goal-directed behaviour on social media platforms. This proposal evaluates traditional machine learning models and fine-tuned transformers for a previously split hope speech dataset as train, development and test set. On development test, a linear-kernel SVM and logistic regression both reached a macro-F1 of 0.78; SVM with RBF kernel reached 0.77, and Naïve Bayes hit 0.75. Transformer models delivered better results, the best model achieved weighted precision of 0.82, weighted recall of 0.80, weighted F1 of 0.79, macro F1 of 0.79, and 0.80 accuracy. These results suggest that while optimally configured traditional machine learning models remain agile, transformer architectures detect some subtle semantics of hope to achieve higher precision and recall in hope speech detection, suggesting that larges transformers and LLMs could perform better in small datasets. 

---
# LimRank: Less is More for Reasoning-Intensive Information Reranking 

**Authors**: Tingyu Song, Yilun Zhao, Siyue Zhang, Chen Zhao, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2510.23544)  

**Abstract**: Existing approaches typically rely on large-scale fine-tuning to adapt LLMs for information reranking tasks, which is computationally expensive. In this work, we demonstrate that modern LLMs can be effectively adapted using only minimal, high-quality supervision. To enable this, we design LIMRANK-SYNTHESIZER, a reusable and open-source pipeline for generating diverse, challenging, and realistic reranking examples. Using this synthetic data, we fine-tune our reranker model, LIMRANK. We evaluate LIMRANK on two challenging benchmarks, i.e., BRIGHT for reasoning-intensive retrieval and FollowIR for instruction-following retrieval. Our experiments demonstrate that LIMRANK achieves competitive performance, while being trained on less than 5% of the data typically used in prior work. Further ablation studies demonstrate the effectiveness of LIMRANK-SYNTHESIZER and the strong generalization capabilities of LIMRANK across downstream tasks, including scientific literature search and retrieval-augmented generation for knowledge-intensive problem solving. 

---
# IPQA: A Benchmark for Core Intent Identification in Personalized Question Answering 

**Authors**: Jieyong Kim, Maryam Amirizaniani, Soojin Yoon, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.23536)  

**Abstract**: Intent identification serves as the foundation for generating appropriate responses in personalized question answering (PQA). However, existing benchmarks evaluate only response quality or retrieval performance without directly measuring intent identification capabilities. This gap is critical because without understanding which intents users prioritize, systems cannot generate responses satisfying individual information needs. To address this, we introduce the concept of core intents: intents users prioritize when selecting answers to satisfy their information needs. To evaluate these core intents, we propose IPQA, a benchmark for core Intent identification in Personalized Question Answering. Since users do not explicitly state their prioritized intents, we derive core intents from observable behavior patterns in answer selection, grounded in satisficing theory where users choose answers meeting their acceptance thresholds. We construct a dataset with various domains through systematic filtering, LLM-based annotation, and rigorous quality control combining automated verification with human validation. Experimental evaluations across state-of-the-art language models reveal that current systems struggle with core intent identification in personalized contexts. Models fail to identify core intents from user histories, with performance degrading as question complexity increases. The code and dataset will be made publicly available to facilitate future research in this direction. 

---
# M4FC: a Multimodal, Multilingual, Multicultural, Multitask Real-World Fact-Checking Dataset 

**Authors**: Jiahui Geng, Jonathan Tonglet, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2510.23508)  

**Abstract**: Existing real-world datasets for multimodal automated fact-checking have multiple limitations: they contain few instances, focus on only one or two languages and tasks, suffer from evidence leakage, or depend on external sets of news articles for sourcing true claims. To address these shortcomings, we introduce M4FC, a new real-world dataset comprising 4,982 images paired with 6,980 claims. The images, verified by professional fact-checkers from 22 organizations, represent diverse cultural and geographic contexts. Each claim is available in one or two out of ten languages. M4FC spans six multimodal fact-checking tasks: visual claim extraction, claimant intent prediction, fake detection, image contextualization, location verification, and verdict prediction. We provide baseline results for all tasks and analyze how combining intermediate tasks influence downstream verdict prediction performance. We make our dataset and code available. 

---
# MMTutorBench: The First Multimodal Benchmark for AI Math Tutoring 

**Authors**: Tengchao Yang, Sichen Guo, Mengzhao Jia, Jiaming Su, Yuanyang Liu, Zhihan Zhang, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23477)  

**Abstract**: Effective math tutoring requires not only solving problems but also diagnosing students' difficulties and guiding them step by step. While multimodal large language models (MLLMs) show promise, existing benchmarks largely overlook these tutoring skills. We introduce MMTutorBench, the first benchmark for AI math tutoring, consisting of 685 problems built around pedagogically significant key-steps. Each problem is paired with problem-specific rubrics that enable fine-grained evaluation across six dimensions, and structured into three tasks-Insight Discovery, Operation Formulation, and Operation Execution. We evaluate 12 leading MLLMs and find clear performance gaps between proprietary and open-source systems, substantial room compared to human tutors, and consistent trends across input variants: OCR pipelines degrade tutoring quality, few-shot prompting yields limited gains, and our rubric-based LLM-as-a-Judge proves highly reliable. These results highlight both the difficulty and diagnostic value of MMTutorBench for advancing AI tutoring. 

---
# Evaluating Large Language Models for Stance Detection on Financial Targets from SEC Filing Reports and Earnings Call Transcripts 

**Authors**: Nikesh Gyawali, Doina Caragea, Alex Vasenkov, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2510.23464)  

**Abstract**: Financial narratives from U.S. Securities and Exchange Commission (SEC) filing reports and quarterly earnings call transcripts (ECTs) are very important for investors, auditors, and regulators. However, their length, financial jargon, and nuanced language make fine-grained analysis difficult. Prior sentiment analysis in the financial domain required a large, expensive labeled dataset, making the sentence-level stance towards specific financial targets challenging. In this work, we introduce a sentence-level corpus for stance detection focused on three core financial metrics: debt, earnings per share (EPS), and sales. The sentences were extracted from Form 10-K annual reports and ECTs, and labeled for stance (positive, negative, neutral) using the advanced ChatGPT-o3-pro model under rigorous human validation. Using this corpus, we conduct a systematic evaluation of modern large language models (LLMs) using zero-shot, few-shot, and Chain-of-Thought (CoT) prompting strategies. Our results show that few-shot with CoT prompting performs best compared to supervised baselines, and LLMs' performance varies across the SEC and ECT datasets. Our findings highlight the practical viability of leveraging LLMs for target-specific stance in the financial domain without requiring extensive labeled data. 

---
# BrowseConf: Confidence-Guided Test-Time Scaling for Web Agents 

**Authors**: Litu Ou, Kuan Li, Huifeng Yin, Liwen Zhang, Zhongwang Zhang, Xixi Wu, Rui Ye, Zile Qiao, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.23458)  

**Abstract**: Confidence in LLMs is a useful indicator of model uncertainty and answer reliability. Existing work mainly focused on single-turn scenarios, while research on confidence in complex multi-turn interactions is limited. In this paper, we investigate whether LLM-based search agents have the ability to communicate their own confidence through verbalized confidence scores after long sequences of actions, a significantly more challenging task compared to outputting confidence in a single interaction. Experimenting on open-source agentic models, we first find that models exhibit much higher task accuracy at high confidence while having near-zero accuracy when confidence is low. Based on this observation, we propose Test-Time Scaling (TTS) methods that use confidence scores to determine answer quality, encourage the model to try again until reaching a satisfactory confidence level. Results show that our proposed methods significantly reduce token consumption while demonstrating competitive performance compared to baseline fixed budget TTS methods. 

---
# Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences 

**Authors**: Zhuoran Jin, Hongbang Yuan, Kejian Zhu, Jiachun Li, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.23451)  

**Abstract**: Reward models (RMs) play a critical role in aligning AI behaviors with human preferences, yet they face two fundamental challenges: (1) Modality Imbalance, where most RMs are mainly focused on text and image modalities, offering limited support for video, audio, and other modalities; and (2) Preference Rigidity, where training on fixed binary preference pairs fails to capture the complexity and diversity of personalized preferences. To address the above challenges, we propose Omni-Reward, a step toward generalist omni-modal reward modeling with support for free-form preferences, consisting of: (1) Evaluation: We introduce Omni-RewardBench, the first omni-modal RM benchmark with free-form preferences, covering nine tasks across five modalities including text, image, video, audio, and 3D; (2) Data: We construct Omni-RewardData, a multimodal preference dataset comprising 248K general preference pairs and 69K instruction-tuning pairs for training generalist omni-modal RMs; (3) Model: We propose Omni-RewardModel, which includes both discriminative and generative RMs, and achieves strong performance on Omni-RewardBench as well as other widely used reward modeling benchmarks. 

---
# EMTSF:Extraordinary Mixture of SOTA Models for Time Series Forecasting 

**Authors**: Musleh Alharthi, Kaleel Mahmood, Sarosh Patel, Ausif Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2510.23396)  

**Abstract**: The immense success of the Transformer architecture
in Natural Language Processing has led to its adoption in Time Se ries Forecasting (TSF), where superior performance has been shown.
However, a recent important paper questioned their effectiveness by
demonstrating that a simple single layer linear model outperforms
Transformer-based models. This was soon shown to be not as valid,
by a better transformer-based model termed PatchTST. More re cently, TimeLLM demonstrated even better results by repurposing a
Large Language Model (LLM) for the TSF domain. Again, a follow
up paper challenged this by demonstrating that removing the LLM
component or replacing it with a basic attention layer in fact yields
better performance. One of the challenges in forecasting is the fact
that TSF data favors the more recent past, and is sometimes subject
to unpredictable events. Based upon these recent insights in TSF, we
propose a strong Mixture of Experts (MoE) framework. Our method
combines the state-of-the-art (SOTA) models including xLSTM, en hanced Linear, PatchTST, and minGRU, among others. This set of
complimentary and diverse models for TSF are integrated in a Trans former based MoE gating network. Our proposed model outperforms
all existing TSF models on standard benchmarks, surpassing even the
latest approaches based on MoE frameworks. 

---
# Detecting Religious Language in Climate Discourse 

**Authors**: Evy Beijen, Pien Pieterse, Yusuf Çelik, Willem Th. van Peursen, Sandjai Bhulai, Meike Morren  

**Link**: [PDF](https://arxiv.org/pdf/2510.23395)  

**Abstract**: Religious language continues to permeate contemporary discourse, even in ostensibly secular domains such as environmental activism and climate change debates. This paper investigates how explicit and implicit forms of religious language appear in climate-related texts produced by secular and religious nongovernmental organizations (NGOs). We introduce a dual methodological approach: a rule-based model using a hierarchical tree of religious terms derived from ecotheology literature, and large language models (LLMs) operating in a zero-shot setting. Using a dataset of more than 880,000 sentences, we compare how these methods detect religious language and analyze points of agreement and divergence. The results show that the rule-based method consistently labels more sentences as religious than LLMs. These findings highlight not only the methodological challenges of computationally detecting religious language but also the broader tension over whether religious language should be defined by vocabulary alone or by contextual meaning. This study contributes to digital methods in religious studies by demonstrating both the potential and the limitations of approaches for analyzing how the sacred persists in climate discourse. 

---
# How AI Forecasts AI Jobs: Benchmarking LLM Predictions of Labor Market Changes 

**Authors**: Sheri Osborn, Rohit Valecha, H. Raghav Rao, Dan Sass, Anthony Rios  

**Link**: [PDF](https://arxiv.org/pdf/2510.23358)  

**Abstract**: Artificial intelligence is reshaping labor markets, yet we lack tools to systematically forecast its effects on employment. This paper introduces a benchmark for evaluating how well large language models (LLMs) can anticipate changes in job demand, especially in occupations affected by AI. Existing research has shown that LLMs can extract sentiment, summarize economic reports, and emulate forecaster behavior, but little work has assessed their use for forward-looking labor prediction. Our benchmark combines two complementary datasets: a high-frequency index of sector-level job postings in the United States, and a global dataset of projected occupational changes due to AI adoption. We format these data into forecasting tasks with clear temporal splits, minimizing the risk of information leakage. We then evaluate LLMs using multiple prompting strategies, comparing task-scaffolded, persona-driven, and hybrid approaches across model families. We assess both quantitative accuracy and qualitative consistency over time. Results show that structured task prompts consistently improve forecast stability, while persona prompts offer advantages on short-term trends. However, performance varies significantly across sectors and horizons, highlighting the need for domain-aware prompting and rigorous evaluation protocols. By releasing our benchmark, we aim to support future research on labor forecasting, prompt design, and LLM-based economic reasoning. This work contributes to a growing body of research on how LLMs interact with real-world economic data, and provides a reproducible testbed for studying the limits and opportunities of AI as a forecasting tool in the context of labor markets. 

---
# LightKGG: Simple and Efficient Knowledge Graph Generation from Textual Data 

**Authors**: Teng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.23341)  

**Abstract**: The scarcity of high-quality knowledge graphs (KGs) remains a critical bottleneck for downstream AI applications, as existing extraction methods rely heavily on error-prone pattern-matching techniques or resource-intensive large language models (LLMs). While recent tools leverage LLMs to generate KGs, their computational demands limit accessibility for low-resource environments. Our paper introduces LightKGG, a novel framework that enables efficient KG extraction from textual data using small-scale language models (SLMs) through two key technical innovations: (1) Context-integrated Graph extraction integrates contextual information with nodes and edges into a unified graph structure, reducing the reliance on complex semantic processing while maintaining more key information; (2) Topology-enhanced relationship inference leverages the inherent topology of the extracted graph to efficiently infer relationships, enabling relationship discovery without relying on complex language understanding capabilities of LLMs. By enabling accurate KG construction with minimal hardware requirements, this work bridges the gap between automated knowledge extraction and practical deployment scenarios while introducing scientifically rigorous methods for optimizing SLM efficiency in structured NLP tasks. 

---
# BaZi-Based Character Simulation Benchmark: Evaluating AI on Temporal and Persona Reasoning 

**Authors**: Siyuan Zheng, Pai Liu, Xi Chen, Jizheng Dong, Sihan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.23337)  

**Abstract**: Human-like virtual characters are crucial for games, storytelling, and virtual reality, yet current methods rely heavily on annotated data or handcrafted persona prompts, making it difficult to scale up and generate realistic, contextually coherent personas. We create the first QA dataset for BaZi-based persona reasoning, where real human experiences categorized into wealth, health, kinship, career, and relationships are represented as life-event questions and answers. Furthermore, we propose the first BaZi-LLM system that integrates symbolic reasoning with large language models to generate temporally dynamic and fine-grained virtual personas. Compared with mainstream LLMs such as DeepSeek-v3 and GPT-5-mini, our method achieves a 30.3%-62.6% accuracy improvement. In addition, when incorrect BaZi information is used, our model's accuracy drops by 20%-45%, showing the potential of culturally grounded symbolic-LLM integration for realistic character simulation. 

---
# Adaptive Blockwise Search: Inference-Time Alignment for Large Language Models 

**Authors**: Mohammad Atif Quamar, Mohammad Areeb, Nishant Sharma, Ananth Shreekumar, Jonathan Rosenthal, Muslum Ozgur Ozmen, Mikhail Kuznetsov, Z. Berkay Celik  

**Link**: [PDF](https://arxiv.org/pdf/2510.23334)  

**Abstract**: LLM alignment remains a critical challenge. Inference-time methods provide a flexible alternative to fine-tuning, but their uniform computational effort often yields suboptimal alignment. We hypothesize that for many alignment tasks, the initial tokens of a response are disproportionately more critical. To leverage this principle, we introduce AdaSearch, a novel blockwise search strategy. It adaptively allocates a fixed computational budget using a sampling schedule, focusing search effort on these critical tokens. We apply AdaSearch to sequential decoding and introduce its tree-search counterpart, AdaBeam. Our comprehensive evaluation across eight LLMs demonstrates that AdaSearch outperforms strong Best-of-N and fine-tuning baselines. Specifically, win-rates improve by over 10% for harmlessness generation, controlled sentiment generation, and for mathematical reasoning tasks relative to Best-of-N. 

---
# Arabic Little STT: Arabic Children Speech Recognition Dataset 

**Authors**: Mouhand Alkadri, Dania Desouki, Khloud Al Jallad  

**Link**: [PDF](https://arxiv.org/pdf/2510.23319)  

**Abstract**: The performance of Artificial Intelligence (AI) systems fundamentally depends on high-quality training data. However, low-resource languages like Arabic suffer from severe data scarcity. Moreover, the absence of child-specific speech corpora is an essential gap that poses significant challenges. To address this gap, we present our created dataset, Arabic Little STT, a dataset of Levantine Arabic child speech recorded in classrooms, containing 355 utterances from 288 children (ages 6 - 13). We further conduct a systematic assessment of Whisper, a state-of-the-art automatic speech recognition (ASR) model, on this dataset and compare its performance with adult Arabic benchmarks. Our evaluation across eight Whisper variants reveals that even the best-performing model (Large_v3) struggles significantly, achieving a 0.66 word error rate (WER) on child speech, starkly contrasting with its sub 0.20 WER on adult datasets. These results align with other research on English speech. Results highlight the critical need for dedicated child speech benchmarks and inclusive training data in ASR development. Emphasizing that such data must be governed by strict ethical and privacy frameworks to protect sensitive child information. We hope that this study provides an initial step for future work on equitable speech technologies for Arabic-speaking children. We hope that our publicly available dataset enrich the children's demographic representation in ASR datasets. 

---
# DCMM-SQL: Automated Data-Centric Pipeline and Multi-Model Collaboration Training for Text-to-SQL Model 

**Authors**: Yuanzhen Xie, Liu Ye, Jiqun Chu, Mochi Gao, Hehuan Liu, Yunzhi Tan, Bo Hu, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.23284)  

**Abstract**: Text-to-SQL tasks have gained attractive improvements since the release of ChatGPT. Among them, agent-based frameworks have been widely used in this field. However, the impact of data-centric strategies on text-to-SQL tasks has rarely been explored. In this paper, we systemically design a fully automated data-centric pipeline for text-to-SQL tasks, including \emph{adaptive data repair}, which can automatically find and fix errors in the training dataset; and \emph{error data augmentation}, where we specifically diffuse and enhance erroneous data predicted by the initially trained models. Meanwhile, we propose a Multi-Model collaboration training schema, aiming to train multiple models with different augmented data, enabling them to possess distinct capabilities and work together to complement each other, because it has been found that the capability of a single fine-tuned model is very limited. Furthermore, we utilize an ensemble strategy to integrate the capabilities of multiple models to solve a multiple-choice question, aiming to further improve the accuracy of text-to-SQL tasks. The experiment results and ablation study have demonstrated the effectiveness of data-centric pipeline and Multi-Model(MM) interactive iterative strategies, achieving first place in lightweight text-to-SQL models (within 70B). 

---
# A Cocktail-Party Benchmark: Multi-Modal dataset and Comparative Evaluation Results 

**Authors**: Thai-Binh Nguyen, Katerina Zmolikova, Pingchuan Ma, Ngoc Quan Pham, Christian Fuegen, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2510.23276)  

**Abstract**: We introduce the task of Multi-Modal Context-Aware Recognition (MCoRec) in the ninth CHiME Challenge, which addresses the cocktail-party problem of overlapping conversations in a single-room setting using audio, visual, and contextual cues. MCoRec captures natural multi-party conversations where the recordings focus on unscripted, casual group chats, leading to extreme speech overlap of up to 100% and highly fragmented conversational turns. The task requires systems to answer the question "Who speaks when, what, and with whom?" by jointly transcribing each speaker's speech and clustering them into their respective conversations from audio-visual recordings. Audio-only baselines exceed 100% word error rate, whereas incorporating visual cues yields substantial 50% improvements, highlighting the importance of multi-modality. In this manuscript, we present the motivation behind the task, outline the data collection process, and report the baseline systems developed for the MCoRec. 

---
# Code Aesthetics with Agentic Reward Feedback 

**Authors**: Bang Xiao, Lingjie Jiang, Shaohan Huang, Tengchao Lv, Yupan Huang, Xun Wu, Lei Cui, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.23272)  

**Abstract**: Large Language Models (LLMs) have become valuable assistants for developers in code-related tasks. While LLMs excel at traditional programming tasks such as code generation and bug fixing, they struggle with visually-oriented coding tasks, often producing suboptimal aesthetics. In this paper, we introduce a new pipeline to enhance the aesthetic quality of LLM-generated code. We first construct AesCode-358K, a large-scale instruction-tuning dataset focused on code aesthetics. Next, we propose agentic reward feedback, a multi-agent system that evaluates executability, static aesthetics, and interactive aesthetics. Building on this, we develop GRPO-AR, which integrates these signals into the GRPO algorithm for joint optimization of functionality and code aesthetics. Finally, we develop OpenDesign, a benchmark for assessing code aesthetics. Experimental results show that combining supervised fine-tuning on AesCode-358K with reinforcement learning using agentic reward feedback significantly improves performance on OpenDesign and also enhances results on existing benchmarks such as PandasPlotBench. Notably, our AesCoder-4B surpasses GPT-4o and GPT-4.1, and achieves performance comparable to large open-source models with 480B-685B parameters, underscoring the effectiveness of our approach. 

---
# Mubeen AI: A Specialized Arabic Language Model for Heritage Preservation and User Intent Understanding 

**Authors**: Mohammed Aljafari, Ismail Alturki, Ahmed Mori, Yehya Kadumi  

**Link**: [PDF](https://arxiv.org/pdf/2510.23271)  

**Abstract**: Mubeen is a proprietary Arabic language model developed by MASARAT SA, optimized for deep understanding of Arabic linguistics, Islamic studies, and cultural heritage. Trained on an extensive collection of authentic Arabic sources significantly expanded by digitizing historical manuscripts via a proprietary Arabic OCR engine, the model incorporates seminal scholarly works in linguistics, jurisprudence, hadith, and Quranic exegesis, alongside thousands of academic theses and peer-reviewed research papers. Conditioned through a deep linguistic engineering framework, Mubeen masters not just the meaning but the eloquence of Arabic, enabling precise understanding across classical texts, contemporary writing, and regional dialects with focus on comprehending user intent and delivering accurate, contextually relevant responses. Unlike other Arabic models relying on translated English data that often fail in intent detection or retrieval-augmented generation (RAG), Mubeen uses native Arabic sources to ensure cultural authenticity and accuracy. Its core innovation is the Practical Closure Architecture, designed to solve the "Utility Gap Crisis" where factually correct answers fail to resolve users' core needs, forcing them into frustrating cycles of re-prompting. By prioritizing clarity and decisive guidance, Mubeen transforms from an information repository into a decisive guide, aligning with Saudi Vision 2030. The model's architecture combines deep heritage specialization with multi-disciplinary expert modules, enabling robust performance across both cultural preservation and general knowledge domains. 

---
# Are ASR foundation models generalized enough to capture features of regional dialects for low-resource languages? 

**Authors**: Tawsif Tashwar Dipto, Azmol Hossain, Rubayet Sabbir Faruque, Md. Rezuwan Hassan, Kanij Fatema, Tanmoy Shome, Ruwad Naswan, Md.Foriduzzaman Zihad, Mohaymen Ul Anam, Nazia Tasnim, Hasan Mahmud, Md Kamrul Hasan, Md. Mehedi Hasan Shawon, Farig Sadeque, Tahsin Reasat  

**Link**: [PDF](https://arxiv.org/pdf/2510.23252)  

**Abstract**: Conventional research on speech recognition modeling relies on the canonical form for most low-resource languages while automatic speech recognition (ASR) for regional dialects is treated as a fine-tuning task. To investigate the effects of dialectal variations on ASR we develop a 78-hour annotated Bengali Speech-to-Text (STT) corpus named Ben-10. Investigation from linguistic and data-driven perspectives shows that speech foundation models struggle heavily in regional dialect ASR, both in zero-shot and fine-tuned settings. We observe that all deep learning methods struggle to model speech data under dialectal variations but dialect specific model training alleviates the issue. Our dataset also serves as a out of-distribution (OOD) resource for ASR modeling under constrained resources in ASR algorithms. The dataset and code developed for this project are publicly available 

---
# Process Reward Models for Sentence-Level Verification of LVLM Radiology Reports 

**Authors**: Alois Thomas, Maya Varma, Jean-Benoit Delbrouck, Curtis P. Langlotz  

**Link**: [PDF](https://arxiv.org/pdf/2510.23217)  

**Abstract**: Automating radiology report generation with Large Vision-Language Models (LVLMs) holds great potential, yet these models often produce clinically critical hallucinations, posing serious risks. Existing hallucination detection methods frequently lack the necessary sentence-level granularity or robust generalization across different LVLM generators. We introduce a novel approach: a sentence-level Process Reward Model (PRM) adapted for this vision-language task. Our PRM predicts the factual correctness of each generated sentence, conditioned on clinical context and preceding text. When fine-tuned on MIMIC-CXR with weakly-supervised labels, a lightweight 0.5B-parameter PRM outperforms existing verification techniques, demonstrating, for instance, relative improvements of 7.5% in Matthews Correlation Coefficient and 1.8% in AUROC over strong white-box baselines on outputs from one LVLM. Unlike methods reliant on internal model states, our PRM demonstrates strong generalization to an unseen LVLM. We further show its practical utility: PRM scores effectively filter low-quality reports, improving F1-CheXbert scores by 4.5% (when discarding the worst 10% of reports). Moreover, when guiding a novel weighted best-of-N selection process on the MIMIC-CXR test set, our PRM show relative improvements in clinical metrics of 7.4% for F1-CheXbert and 0.6% for BERTScore. These results demonstrate that a lightweight, context-aware PRM provides a model-agnostic safety layer for clinical LVLMs without access to internal activations 

---
# DREaM: Drug-Drug Relation Extraction via Transfer Learning Method 

**Authors**: Ali Fata, Hossein Rahmani, Parinaz Soltanzadeh, Amirhossein Derakhshan, Behrouz Minaei Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2510.23189)  

**Abstract**: Relation extraction between drugs plays a crucial role in identifying drug drug interactions and predicting side effects. The advancement of machine learning methods in relation extraction, along with the development of large medical text databases, has enabled the low cost extraction of such relations compared to other approaches that typically require expert knowledge. However, to the best of our knowledge, there are limited datasets specifically designed for drug drug relation extraction currently available. Therefore, employing transfer learning becomes necessary to apply machine learning methods in this domain. In this study, we propose DREAM, a method that first employs a trained relation extraction model to discover relations between entities and then applies this model to a corpus of medical texts to construct an ontology of drug relationships. The extracted relations are subsequently validated using a large language model. Quantitative results indicate that the LLM agreed with 71 of the relations extracted from a subset of PubMed abstracts. Furthermore, our qualitative analysis indicates that this approach can uncover ambiguities in the medical domain, highlighting the challenges inherent in relation extraction in this field. 

---
# SI-Bench: Benchmarking Social Intelligence of Large Language Models in Human-to-Human Conversations 

**Authors**: Shuai Huang, Wenxuan Zhao, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.23182)  

**Abstract**: As large language models (LLMs) develop anthropomorphic abilities, they are increasingly being deployed as autonomous agents to interact with humans. However, evaluating their performance in realistic and complex social interactions remains a significant challenge. Most previous research built datasets through simulated agent-to-agent interactions, which fails to capture the authentic linguistic styles and relational dynamics found in real human conversations. To address this gap, we introduce SI-Bench, a novel benchmark designed to evaluate aspects of social intelligence in LLMs. Grounded in broad social science theories, SI-Bench contains 2,221 authentic multi-turn dialogues collected from a social networking application. We further selected a subset of 312 dialogues for manual annotation across 8 major models. The experiments show that SOTA models have surpassed the human expert in process reasoning under complex social situations, yet they still fall behind humans in reply quality. Moreover, introducing Chain-of-Thought (CoT) reasoning may degrade the performance of LLMs in social dialogue tasks. All datasets are openly available at this https URL. 

---
# MATCH: Task-Driven Code Evaluation through Contrastive Learning 

**Authors**: Marah Ghoummaid, Vladimir Tchuiev, Ofek Glick, Michal Moschkovitz, Dotan Di Castro  

**Link**: [PDF](https://arxiv.org/pdf/2510.23169)  

**Abstract**: AI-based code generation is increasingly prevalent, with GitHub Copilot estimated to generate 46% of the code on GitHub. Accurately evaluating how well generated code aligns with developer intent remains a critical challenge. Traditional evaluation methods, such as unit tests, are often unscalable and costly. Syntactic similarity metrics (e.g., BLEU, ROUGE) fail to capture code functionality, and metrics like CodeBERTScore require reference code, which is not always available. To address the gap in reference-free evaluation, with few alternatives such as ICE-Score, this paper introduces MATCH, a novel reference-free metric. MATCH uses Contrastive Learning to generate meaningful embeddings for code and natural language task descriptions, enabling similarity scoring that reflects how well generated code implements the task. We show that MATCH achieves stronger correlations with functional correctness and human preference than existing metrics across multiple programming languages. 

---
# Beyond Direct Generation: A Decomposed Approach to Well-Crafted Screenwriting with LLMs 

**Authors**: Hang Lei, Shengyi Zong, Zhaoyan Li, Ziren Zhou, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23163)  

**Abstract**: The screenplay serves as the foundation for television production, defining narrative structure, character development, and dialogue. While Large Language Models (LLMs) show great potential in creative writing, direct end-to-end generation approaches often fail to produce well-crafted screenplays. We argue this failure stems from forcing a single model to simultaneously master two disparate capabilities: creative narrative construction and rigid format adherence. The resulting outputs may mimic superficial style but lack the deep structural integrity and storytelling substance required for professional use. To enable LLMs to generate high-quality screenplays, we introduce Dual-Stage Refinement (DSR), a decomposed framework that decouples creative narrative generation from format conversion. The first stage transforms a brief outline into rich, novel-style prose. The second stage refines this narrative into a professionally formatted screenplay. This separation enables the model to specialize in one distinct capability at each stage. A key challenge in implementing DSR is the scarcity of paired outline-to-novel training data. We address this through hybrid data synthesis: reverse synthesis deconstructs existing screenplays into structured inputs, while forward synthesis leverages these inputs to generate high-quality narrative texts as training targets. Blind evaluations by professional screenwriters show that DSR achieves a 75% win rate against strong baselines like Gemini-2.5-Pro and reaches 82.7% of human-level performance. Our work demonstrates that decomposed generation architecture with tailored data synthesis effectively specializes LLMs in complex creative domains. 

---
# ENTP: Enhancing Low-Quality SFT Data via Neural-Symbolic Text Purge-Mix 

**Authors**: Zile Yang, Ling Li, Na Di, Jinlong Pang, Yao Zhou, Hao Cheng, Bo Han, Jiaheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.23160)  

**Abstract**: Supervised Fine-Tuning (SFT) adapts pre-trained Large Language Models (LLMs) to domain-specific instructions by training on a carefully curated subset of high-quality instruction-response pairs, typically drawn from a larger dataset that often contains many low-quality or noisy samples. However, existing quality-first paradigms often overlook valuable signals in discarded low-quality data and rely on imperfect quality filters. We introduce ENTP (Enhancing low-quality SFT data via Neural-symbolic Text Purge-Mix), a framework that revitalizes low-quality corpora through symbolic purification and neural reconstruction. The symbolic module identifies and prunes noisy samples based on statistical priors, while the neural component synthesizes enriched instruction-response pairs by leveraging latent representations and model knowledge. This neural-symbolic synergy enhances data informativeness and diversity. Experiments show that ENTP-augmented datasets, constructed exclusively from low-quality data, outperform 13 established data-selection baselines across five instruction-following benchmarks, and even surpass fine-tuning on the full original dataset (approximately 300K examples). Our results highlight the untapped potential of low-quality data and underscore the importance of intelligent purification and synthesis for efficient instruction alignment. 

---
# Corpus Frequencies in Morphological Inflection: Do They Matter? 

**Authors**: Tomáš Sourada, Jana Straková  

**Link**: [PDF](https://arxiv.org/pdf/2510.23131)  

**Abstract**: The traditional approach to morphological inflection (the task of modifying a base word (lemma) to express grammatical categories) has been, for decades, to consider lexical entries of lemma-tag-form triples uniformly, lacking any information about their frequency distribution. However, in production deployment, one might expect the user inputs to reflect a real-world distribution of frequencies in natural texts. With future deployment in mind, we explore the incorporation of corpus frequency information into the task of morphological inflection along three key dimensions during system development: (i) for train-dev-test split, we combine a lemma-disjoint approach, which evaluates the model's generalization capabilities, with a frequency-weighted strategy to better reflect the realistic distribution of items across different frequency bands in training and test sets; (ii) for evaluation, we complement the standard type accuracy (often referred to simply as accuracy), which treats all items equally regardless of frequency, with token accuracy, which assigns greater weight to frequent words and better approximates performance on running text; (iii) for training data sampling, we introduce a method novel in the context of inflection, frequency-aware training, which explicitly incorporates word frequency into the sampling process. We show that frequency-aware training outperforms uniform sampling in 26 out of 43 languages. 

---
# Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation 

**Authors**: Shiwei Li, Xiandi Luo, Haozhao Wang, Xing Tang, Ziqiang Cui, Dugang Liu, Yuhua Li, Xiuqiang He, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.23123)  

**Abstract**: Low-rank adaptation (LoRA) is a parameter-efficient fine-tuning (PEFT) method widely used in large language models (LLMs). LoRA essentially describes the projection of an input space into a low-dimensional output space, with the dimensionality determined by the LoRA rank. In standard LoRA, all input tokens share the same weights and undergo an identical input-output projection. This limits LoRA's ability to capture token-specific information due to the inherent semantic differences among tokens. To address this limitation, we propose Token-wise Projected Low-Rank Adaptation (TopLoRA), which dynamically adjusts LoRA weights according to the input token, thereby learning token-wise input-output projections in an end-to-end manner. Formally, the weights of TopLoRA can be expressed as $B\Sigma_X A$, where $A$ and $B$ are low-rank matrices (as in standard LoRA), and $\Sigma_X$ is a diagonal matrix generated from each input token $X$. Notably, TopLoRA does not increase the rank of LoRA weights but achieves more granular adaptation by learning token-wise LoRA weights (i.e., token-wise input-output projections). Extensive experiments across multiple models and datasets demonstrate that TopLoRA consistently outperforms LoRA and its variants. The code is available at this https URL. 

---
# Flexing in 73 Languages: A Single Small Model for Multilingual Inflection 

**Authors**: Tomáš Sourada, Jana Straková  

**Link**: [PDF](https://arxiv.org/pdf/2510.23114)  

**Abstract**: We present a compact, single-model approach to multilingual inflection, the task of generating inflected word forms from base lemmas to express grammatical categories. Our model, trained jointly on data from 73 languages, is lightweight, robust to unseen words, and outperforms monolingual baselines in most languages. This demonstrates the effectiveness of multilingual modeling for inflection and highlights its practical benefits: simplifying deployment by eliminating the need to manage and retrain dozens of separate monolingual models. In addition to the standard SIGMORPHON shared task benchmarks, we evaluate our monolingual and multilingual models on 73 Universal Dependencies (UD) treebanks, extracting lemma-tag-form triples and their frequency counts. To ensure realistic data splits, we introduce a novel frequency-weighted, lemma-disjoint train-dev-test resampling procedure. Our work addresses the lack of an open-source, general-purpose, multilingual morphological inflection system capable of handling unseen words across a wide range of languages, including Czech. All code is publicly released at: this https URL. 

---
# Leveraging Hierarchical Organization for Medical Multi-document Summarization 

**Authors**: Yi-Li Hsu, Katelyn X. Mei, Lucy Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23104)  

**Abstract**: Medical multi-document summarization (MDS) is a complex task that requires effectively managing cross-document relationships. This paper investigates whether incorporating hierarchical structures in the inputs of MDS can improve a model's ability to organize and contextualize information across documents compared to traditional flat summarization methods. We investigate two ways of incorporating hierarchical organization across three large language models (LLMs), and conduct comprehensive evaluations of the resulting summaries using automated metrics, model-based metrics, and domain expert evaluation of preference, understandability, clarity, complexity, relevance, coverage, factuality, and coherence. Our results show that human experts prefer model-generated summaries over human-written summaries. Hierarchical approaches generally preserve factuality, coverage, and coherence of information, while also increasing human preference for summaries. Additionally, we examine whether simulated judgments from GPT-4 align with human judgments, finding higher agreement along more objective evaluation facets. Our findings demonstrate that hierarchical structures can improve the clarity of medical summaries generated by models while maintaining content coverage, providing a practical way to improve human preference for generated summaries. 

---
# MAP4TS: A Multi-Aspect Prompting Framework for Time-Series Forecasting with Large Language Models 

**Authors**: Suchan Lee, Jihoon Choi, Sohyeon Lee, Minseok Song, Bong-Gyu Jang, Hwanjo Yu, Soyeon Caren Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.23090)  

**Abstract**: Recent advances have investigated the use of pretrained large language models (LLMs) for time-series forecasting by aligning numerical inputs with LLM embedding spaces. However, existing multimodal approaches often overlook the distinct statistical properties and temporal dependencies that are fundamental to time-series data. To bridge this gap, we propose MAP4TS, a novel Multi-Aspect Prompting Framework that explicitly incorporates classical time-series analysis into the prompt design. Our framework introduces four specialized prompt components: a Global Domain Prompt that conveys dataset-level context, a Local Domain Prompt that encodes recent trends and series-specific behaviors, and a pair of Statistical and Temporal Prompts that embed handcrafted insights derived from autocorrelation (ACF), partial autocorrelation (PACF), and Fourier analysis. Multi-Aspect Prompts are combined with raw time-series embeddings and passed through a cross-modality alignment module to produce unified representations, which are then processed by an LLM and projected for final forecasting. Extensive experiments across eight diverse datasets show that MAP4TS consistently outperforms state-of-the-art LLM-based methods. Our ablation studies further reveal that prompt-aware designs significantly enhance performance stability and that GPT-2 backbones, when paired with structured prompts, outperform larger models like LLaMA in long-term forecasting tasks. 

---
# A Survey on LLM Mid-training 

**Authors**: Chengying Tu, Xuemiao Zhang, Rongxiang Weng, Rumei Li, Chen Zhang, Yang Bai, Hongfei Yan, Jingang Wang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.23081)  

**Abstract**: Recent advances in foundation models have highlighted the significant benefits of multi-stage training, with a particular emphasis on the emergence of mid-training as a vital stage that bridges pre-training and post-training. Mid-training is distinguished by its use of intermediate data and computational resources, systematically enhancing specified capabilities such as mathematics, coding, reasoning, and long-context extension, while maintaining foundational competencies. This survey provides a formal definition of mid-training for large language models (LLMs) and investigates optimization frameworks that encompass data curation, training strategies, and model architecture optimization. We analyze mainstream model implementations in the context of objective-driven interventions, illustrating how mid-training serves as a distinct and critical stage in the progressive development of LLM capabilities. By clarifying the unique contributions of mid-training, this survey offers a comprehensive taxonomy and actionable insights, supporting future research and innovation in the advancement of LLMs. 

---
# Quality-Aware Translation Tagging in Multilingual RAG system 

**Authors**: Hoyeon Moon, Byeolhee Kim, Nikhil Verma  

**Link**: [PDF](https://arxiv.org/pdf/2510.23070)  

**Abstract**: Multilingual Retrieval-Augmented Generation (mRAG) often retrieves English documents and translates them into the query language for low-resource settings. However, poor translation quality degrades response generation performance. Existing approaches either assume sufficient translation quality or utilize the rewriting method, which introduces factual distortion and hallucinations. To mitigate these problems, we propose Quality-Aware Translation Tagging in mRAG (QTT-RAG), which explicitly evaluates translation quality along three dimensions-semantic equivalence, grammatical accuracy, and naturalness&fluency-and attach these scores as metadata without altering the original content. We evaluate QTT-RAG against CrossRAG and DKM-RAG as baselines in two open-domain QA benchmarks (XORQA, MKQA) using six instruction-tuned LLMs ranging from 2.4B to 14B parameters, covering two low-resource languages (Korean and Finnish) and one high-resource language (Chinese). QTT-RAG outperforms the baselines by preserving factual integrity while enabling generator models to make informed decisions based on translation reliability. This approach allows for effective usage of cross-lingual documents in low-resource settings with limited native language documents, offering a practical and robust solution across multilingual domains. 

---
# Knocking-Heads Attention 

**Authors**: Zhanchao Zhou, Xiaodong Chen, Haoxing Chen, Zhenzhong Lan, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.23052)  

**Abstract**: Multi-head attention (MHA) has become the cornerstone of modern large language models, enhancing representational capacity through parallel attention heads. However, increasing the number of heads inherently weakens individual head capacity, and existing attention mechanisms - whether standard MHA or its variants like grouped-query attention (GQA) and grouped-tied attention (GTA) - simply concatenate outputs from isolated heads without strong interaction. To address this limitation, we propose knocking-heads attention (KHA), which enables attention heads to "knock" on each other - facilitating cross-head feature-level interactions before the scaled dot-product attention. This is achieved by applying a shared, diagonally-initialized projection matrix across all heads. The diagonal initialization preserves head-specific specialization at the start of training while allowing the model to progressively learn integrated cross-head representations. KHA adds only minimal parameters and FLOPs and can be seamlessly integrated into MHA, GQA, GTA, and other attention variants. We validate KHA by training a 6.1B parameter MoE model (1.01B activated) on 1T high-quality tokens. Compared to baseline attention mechanisms, KHA brings superior and more stable training dynamics, achieving better performance across downstream tasks. 

---
# Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning 

**Authors**: Ran Xu, Jingjing Chen, Jiayu Ye, Yu Wu, Jun Yan, Carl Yang, Hongkun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23038)  

**Abstract**: Large Language Models (LLMs) are widely used as judges to evaluate response quality, providing a scalable alternative to human evaluation. However, most LLM judges operate solely on intrinsic text-based reasoning, limiting their ability to verify complex constraints or perform accurate computation. Motivated by the success of tool-integrated reasoning (TIR) in numerous tasks, we propose TIR-Judge, an end-to-end RL framework for training LLM judges that integrates a code executor for precise evaluation. TIR-Judge is built on three principles: (i) diverse training across verifiable and non-verifiable domains, (ii) flexible judgment formats (pointwise, pairwise, listwise), and (iii) iterative RL that bootstraps directly from the initial model without distillation. On seven public benchmarks, TIR-Judge surpasses strong reasoning-based judges by up to 6.4% (pointwise) and 7.7% (pairwise), and achieves listwise performance comparable to Claude-Opus-4 despite having only 8B parameters. Remarkably, TIR-Judge-Zero - trained entirely without distilled judge trajectories, matches the performance of distilled variants, demonstrating that tool-augmented judges can self-evolve through iterative reinforcement learning. 

---
# LangLingual: A Personalised, Exercise-oriented English Language Learning Tool Leveraging Large Language Models 

**Authors**: Sammriddh Gupta, Sonit Singh, Aditya Joshi, Mira Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.23011)  

**Abstract**: Language educators strive to create a rich experience for learners, while they may be restricted in the extend of feedback and practice they can provide. We present the design and development of LangLingual, a conversational agent built using the LangChain framework and powered by Large Language Models. The system is specifically designed to provide real-time, grammar-focused feedback, generate context-aware language exercises and track learner proficiency over time. The paper discusses the architecture, implementation and evaluation of LangLingual in detail. The results indicate strong usability, positive learning outcomes and encouraging learner engagement. 

---
# Understanding In-Context Learning Beyond Transformers: An Investigation of State Space and Hybrid Architectures 

**Authors**: Shenran Wang, Timothy Tin-Long Tse, Jian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23006)  

**Abstract**: We perform in-depth evaluations of in-context learning (ICL) on state-of-the-art transformer, state-space, and hybrid large language models over two categories of knowledge-based ICL tasks. Using a combination of behavioral probing and intervention-based methods, we have discovered that, while LLMs of different architectures can behave similarly in task performance, their internals could remain different. We discover that function vectors (FVs) responsible for ICL are primarily located in the self-attention and Mamba layers, and speculate that Mamba2 uses a different mechanism from FVs to perform ICL. FVs are more important for ICL involving parametric knowledge retrieval, but not for contextual knowledge understanding. Our work contributes to a more nuanced understanding across architectures and task types. Methodologically, our approach also highlights the importance of combining both behavioural and mechanistic analyses to investigate LLM capabilities. 

---
# Measuring Teaching with LLMs 

**Authors**: Michael Hardy  

**Link**: [PDF](https://arxiv.org/pdf/2510.22968)  

**Abstract**: Objective and scalable measurement of teaching quality is a persistent challenge in education. While Large Language Models (LLMs) offer potential, general-purpose models have struggled to reliably apply complex, authentic classroom observation instruments. This paper uses custom LLMs built on sentence-level embeddings, an architecture better suited for the long-form, interpretive nature of classroom transcripts than conventional subword tokenization. We systematically evaluate five different sentence embeddings under a data-efficient training regime designed to prevent overfitting. Our results demonstrate that these specialized models can achieve human-level and even super-human performance with expert human ratings above 0.65 and surpassing the average human-human rater correlation. Further, through analysis of annotation context windows, we find that more advanced models-those better aligned with human judgments-attribute a larger share of score variation to lesson-level features rather than isolated utterances, challenging the sufficiency of single-turn annotation paradigms. Finally, to assess external validity, we find that aggregate model scores align with teacher value-added measures, indicating they are capturing features relevant to student learning. However, this trend does not hold at the individual item level, suggesting that while the models learn useful signals, they have not yet achieved full generalization. This work establishes a viable and powerful new methodology for AI-driven instructional measurement, offering a path toward providing scalable, reliable, and valid feedback for educator development. 

---
# MAD-Fact: A Multi-Agent Debate Framework for Long-Form Factuality Evaluation in LLMs 

**Authors**: Yucheng Ning, Xixun Lin, Fang Fang, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.22967)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) raises critical concerns about the factual accuracy of their outputs, especially in high-risk domains such as biomedicine, law, and education. Existing evaluation methods for short texts often fail on long-form content due to complex reasoning chains, intertwined perspectives, and cumulative information. To address this, we propose a systematic approach integrating large-scale long-form datasets, multi-agent verification mechanisms, and weighted evaluation metrics. We construct LongHalluQA, a Chinese long-form factuality dataset; and develop MAD-Fact, a debate-based multi-agent verification system. We introduce a fact importance hierarchy to capture the varying significance of claims in long-form texts. Experiments on two benchmarks show that larger LLMs generally maintain higher factual consistency, while domestic models excel on Chinese content. Our work provides a structured framework for evaluating and enhancing factual reliability in long-form LLM outputs, guiding their safe deployment in sensitive domains. 

---
# Tagging-Augmented Generation: Assisting Language Models in Finding Intricate Knowledge In Long Contexts 

**Authors**: Anwesan Pal, Karen Hovsepian, Tinghao Guo, Mengnan Zhao, Somendra Tripathi, Nikos Kanakaris, George Mihaila, Sumit Nigam  

**Link**: [PDF](https://arxiv.org/pdf/2510.22956)  

**Abstract**: Recent investigations into effective context lengths of modern flagship large language models (LLMs) have revealed major limitations in effective question answering (QA) and reasoning over long and complex contexts for even the largest and most impressive cadre of models. While approaches like retrieval-augmented generation (RAG) and chunk-based re-ranking attempt to mitigate this issue, they are sensitive to chunking, embedding and retrieval strategies and models, and furthermore, rely on extensive pre-processing, knowledge acquisition and indexing steps. In this paper, we propose Tagging-Augmented Generation (TAG), a lightweight data augmentation strategy that boosts LLM performance in long-context scenarios, without degrading and altering the integrity and composition of retrieved documents. We validate our hypothesis by augmenting two challenging and directly relevant question-answering benchmarks -- NoLima and NovelQA -- and show that tagging the context or even just adding tag definitions into QA prompts leads to consistent performance gains over the baseline -- up to 17% for 32K token contexts, and 2.9% in complex reasoning question-answering for multi-hop queries requiring knowledge across a wide span of text. Additional details are available at this https URL. 

---
# Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond) 

**Authors**: Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, Alon Albalak, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.22954)  

**Abstract**: Language models (LMs) often struggle to generate diverse, human-like creative content, raising concerns about the long-term homogenization of human thought through repeated exposure to similar outputs. Yet scalable methods for evaluating LM output diversity remain limited, especially beyond narrow tasks such as random number or name generation, or beyond repeated sampling from a single model. We introduce Infinity-Chat, a large-scale dataset of 26K diverse, real-world, open-ended user queries that admit a wide range of plausible answers with no single ground truth. We introduce the first comprehensive taxonomy for characterizing the full spectrum of open-ended prompts posed to LMs, comprising 6 top-level categories (e.g., brainstorm & ideation) that further breaks down to 17 subcategories. Using Infinity-Chat, we present a large-scale study of mode collapse in LMs, revealing a pronounced Artificial Hivemind effect in open-ended generation of LMs, characterized by (1) intra-model repetition, where a single model consistently generates similar responses, and more so (2) inter-model homogeneity, where different models produce strikingly similar outputs. Infinity-Chat also includes 31,250 human annotations, across absolute ratings and pairwise preferences, with 25 independent human annotations per example. This enables studying collective and individual-specific human preferences in response to open-ended queries. Our findings show that LMs, reward models, and LM judges are less well calibrated to human ratings on model generations that elicit differing idiosyncratic annotator preferences, despite maintaining comparable overall quality. Overall, INFINITY-CHAT presents the first large-scale resource for systematically studying real-world open-ended queries to LMs, revealing critical insights to guide future research for mitigating long-term AI safety risks posed by the Artificial Hivemind. 

---
# Language Server CLI Empowers Language Agents with Process Rewards 

**Authors**: Yifan Zhang, Lanser Contributors  

**Link**: [PDF](https://arxiv.org/pdf/2510.22907)  

**Abstract**: Large language models routinely hallucinate APIs and mislocalize edits, while language servers compute verified, IDE-grade facts about real code. We present Lanser-CLI, a CLI-first orchestration layer that pins and mediates a Language Server Protocol (LSP) server for coding agents and CI, exposing deterministic, replayable workflows. Our position is that language servers provide not only structural information (definitions, references, types, diagnostics) but also an actionable process reward: machine-checked, step-wise signals that align an agent's planning loop with program reality. In this work, Lanser-CLI contributes: (i) a robust addressing scheme beyond brittle "file:line:col" via a Selector DSL (symbolic, AST-path, and content-anchored selectors) with a principled relocation algorithm; (ii) deterministic Analysis Bundles that normalize Language Server responses and capture environment/capability metadata with stable content hashes; (iii) a safety envelope for mutating operations (rename, code actions) with preview, workspace jails, and Git-aware, transactional apply; and (iv) a process-reward functional derived from Language Server facts (diagnostic deltas, disambiguation confidence, and safe-apply checks) that is computable online and replayable offline. We formalize determinism under frozen snapshots and establish a monotonicity property for the process reward, making it suitable for process supervision and counterfactual analysis. Project Page: this https URL 

---
# Batch Speculative Decoding Done Right 

**Authors**: Ranran Haoran Zhang, Soumik Dey, Ashirbad Mishra, Hansi Wu, Binbin Li, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22876)  

**Abstract**: Speculative decoding speeds up LLM inference by using a small draft model to propose multiple tokens that a target model verifies in parallel. Extending this idea to batches is essential for production serving, but it introduces the ragged tensor problem: sequences in the same batch accept different numbers of draft tokens, breaking right-alignment and corrupting position IDs, attention masks, and KV-cache state. We show that several existing batch implementations violate output equivalence-the fundamental requirement that speculative decoding must produce identical token sequences to standard autoregressive generation. These violations occur precisely due to improper handling of the ragged tensor problem. In response, we (1) characterize the synchronization requirements that guarantee correctness, (2) present a correctness-first batch speculative decoding EQSPEC that exposes realignment as consuming 40% of overhead, and (3) introduce EXSPEC, which maintains a sliding pool of sequences and dynamically forms same-length groups, to reduce the realignment overhead while preserving per-sequence speculative speedups. On the SpecBench dataset, across Vicuna-7B/68M, Qwen3-8B/0.6B, and GLM-4-9B/0.6B target/draft pairs, our approach achieves up to 3$\times$ throughput improvement at batch size 8 compared to batch size 1, with efficient scaling through batch size 8, while maintaining 95% output equivalence. Our method requires no custom kernels and integrates cleanly with existing inference stacks. Our code is available at this https URL. 

---
# A Comprehensive Dataset for Human vs. AI Generated Text Detection 

**Authors**: Rajarshi Roy, Nasrin Imanpour, Ashhar Aziz, Shashwat Bajpai, Gurpreet Singh, Shwetangshu Biswas, Kapil Wanaskar, Parth Patwa, Subhankar Ghosh, Shreyas Dixit, Nilesh Ranjan Pal, Vipula Rawte, Ritvik Garimella, Gaytri Jena, Amit Sheth, Vasu Sharma, Aishwarya Naresh Reganti, Vinija Jain, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2510.22874)  

**Abstract**: The rapid advancement of large language models (LLMs) has led to increasingly human-like AI-generated text, raising concerns about content authenticity, misinformation, and trustworthiness. Addressing the challenge of reliably detecting AI-generated text and attributing it to specific models requires large-scale, diverse, and well-annotated datasets. In this work, we present a comprehensive dataset comprising over 58,000 text samples that combine authentic New York Times articles with synthetic versions generated by multiple state-of-the-art LLMs including Gemma-2-9b, Mistral-7B, Qwen-2-72B, LLaMA-8B, Yi-Large, and GPT-4-o. The dataset provides original article abstracts as prompts, full human-authored narratives. We establish baseline results for two key tasks: distinguishing human-written from AI-generated text, achieving an accuracy of 58.35\%, and attributing AI texts to their generating models with an accuracy of 8.92\%. By bridging real-world journalistic content with modern generative models, the dataset aims to catalyze the development of robust detection and attribution methods, fostering trust and transparency in the era of generative AI. Our dataset is available at: this https URL. 

---
# Interpreting and Mitigating Unwanted Uncertainty in LLMs 

**Authors**: Tiasa Singha Roy, Ayush Rajesh Jhaveri, Ilias Triantafyllopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2510.22866)  

**Abstract**: Despite their impressive capabilities, Large Language Models (LLMs) exhibit unwanted uncertainty, a phenomenon where a model changes a previously correct answer into an incorrect one when re-prompted. This behavior undermines trust and poses serious risks in high-stakes domains. In this work, we investigate the mechanisms that drive this phenomenon. We adapt the Needle-in-a-Haystack retrieval framework and integrate a Flip-style re-evaluation prompt to simulate realistic answer-flipping scenarios. We find that retrieval heads are not primarily responsible for avoiding uncertainty. Instead, we identify a small set of non-retrieval attention heads that disproportionately attend to misleading tokens in uncertain contexts. Masking these heads yields significant improvements, reducing flip behavior by up to 15% without introducing incoherence or overcorrection. However, when tested for downstream tasks, we observe trade-offs with flip behavior. Our findings contribute to the growing field of mechanistic interpretability and present a simple yet effective technique for mitigating uncertainty-driven failure modes in LLMs. 

---
# Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement 

**Authors**: Linyang He, Tianjun Zhong, Richard Antonello, Gavin Mischler, Micah Goldblum, Nima Mesgarani  

**Link**: [PDF](https://arxiv.org/pdf/2510.22860)  

**Abstract**: Understanding how the human brain progresses from processing simple linguistic inputs to performing high-level reasoning is a fundamental challenge in neuroscience. While modern large language models (LLMs) are increasingly used to model neural responses to language, their internal representations are highly "entangled," mixing information about lexicon, syntax, meaning, and reasoning. This entanglement biases conventional brain encoding analyses toward linguistically shallow features (e.g., lexicon and syntax), making it difficult to isolate the neural substrates of cognitively deeper processes. Here, we introduce a residual disentanglement method that computationally isolates these components. By first probing an LM to identify feature-specific layers, our method iteratively regresses out lower-level representations to produce four nearly orthogonal embeddings for lexicon, syntax, meaning, and, critically, reasoning. We used these disentangled embeddings to model intracranial (ECoG) brain recordings from neurosurgical patients listening to natural speech. We show that: 1) This isolated reasoning embedding exhibits unique predictive power, accounting for variance in neural activity not explained by other linguistic features and even extending to the recruitment of visual regions beyond classical language areas. 2) The neural signature for reasoning is temporally distinct, peaking later (~350-400ms) than signals related to lexicon, syntax, and meaning, consistent with its position atop a processing hierarchy. 3) Standard, non-disentangled LLM embeddings can be misleading, as their predictive success is primarily attributable to linguistically shallow features, masking the more subtle contributions of deeper cognitive processing. 

---
# Once Upon an Input: Reasoning via Per-Instance Program Synthesis 

**Authors**: Adam Stein, Neelay Velingker, Mayur Naik, Eric Wong  

**Link**: [PDF](https://arxiv.org/pdf/2510.22849)  

**Abstract**: Large language models (LLMs) excel at zero-shot inference but continue to struggle with complex, multi-step reasoning. Recent methods that augment LLMs with intermediate reasoning steps such as Chain of Thought (CoT) and Program of Thought (PoT) improve performance but often produce undesirable solutions, especially in algorithmic domains. We introduce Per-Instance Program Synthesis (PIPS), a method that generates and refines programs at the instance-level using structural feedback without relying on task-specific guidance or explicit test cases. To further improve performance, PIPS incorporates a confidence metric that dynamically chooses between direct inference and program synthesis on a per-instance basis. Experiments across three frontier LLMs and 30 benchmarks including all tasks of Big Bench Extra Hard (BBEH), visual question answering tasks, relational reasoning tasks, and mathematical reasoning tasks show that PIPS improves the absolute harmonic mean accuracy by up to 8.6% and 9.4% compared to PoT and CoT respectively, and reduces undesirable program generations by 65.1% on the algorithmic tasks compared to PoT with Gemini-2.0-Flash. 

---
# Leveraging Large Language Models to Identify Conversation Threads in Collaborative Learning 

**Authors**: Prerna Ravi, Dong Won Lee, Beatriz Flamia, Jasmine David, Brandon Hanks, Cynthia Breazeal, Emma Anderson, Grace Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.22844)  

**Abstract**: Understanding how ideas develop and flow in small-group conversations is critical for analyzing collaborative learning. A key structural feature of these interactions is threading, the way discourse talk naturally organizes into interwoven topical strands that evolve over time. While threading has been widely studied in asynchronous text settings, detecting threads in synchronous spoken dialogue remains challenging due to overlapping turns and implicit cues. At the same time, large language models (LLMs) show promise for automating discourse analysis but often struggle with long-context tasks that depend on tracing these conversational links. In this paper, we investigate whether explicit thread linkages can improve LLM-based coding of relational moves in group talk. We contribute a systematic guidebook for identifying threads in synchronous multi-party transcripts and benchmark different LLM prompting strategies for automated threading. We then test how threading influences performance on downstream coding of conversational analysis frameworks, that capture core collaborative actions such as agreeing, building, and eliciting. Our results show that providing clear conversational thread information improves LLM coding performance and underscores the heavy reliance of downstream analysis on well-structured dialogue. We also discuss practical trade-offs in time and cost, emphasizing where human-AI hybrid approaches can yield the best value. Together, this work advances methods for combining LLMs and robust conversational thread structures to make sense of complex, real-time group interactions. 

---
# Exploration of Summarization by Generative Language Models for Automated Scoring of Long Essays 

**Authors**: Haowei Hua, Hong Jiao, Xinyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22830)  

**Abstract**: BERT and its variants are extensively explored for automated scoring. However, a limit of 512 tokens for these encoder-based models showed the deficiency in automated scoring of long essays. Thus, this research explores generative language models for automated scoring of long essays via summarization and prompting. The results revealed great improvement of scoring accuracy with QWK increased from 0.822 to 0.8878 for the Learning Agency Lab Automated Essay Scoring 2.0 dataset. 

---
# Cross-Lingual Stability and Bias in Instruction-Tuned Language Models for Humanitarian NLP 

**Authors**: Poli Nemkova, Amrit Adhikari, Matthew Pearson, Vamsi Krishna Sadu, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2510.22823)  

**Abstract**: Humanitarian organizations face a critical choice: invest in costly commercial APIs or rely on free open-weight models for multilingual human rights monitoring. While commercial systems offer reliability, open-weight alternatives lack empirical validation -- especially for low-resource languages common in conflict zones. This paper presents the first systematic comparison of commercial and open-weight large language models (LLMs) for human-rights-violation detection across seven languages, quantifying the cost-reliability trade-off facing resource-constrained organizations. Across 78,000 multilingual inferences, we evaluate six models -- four instruction-aligned (Claude-Sonnet-4, DeepSeek-V3, Gemini-Flash-2.0, GPT-4.1-mini) and two open-weight (LLaMA-3-8B, Mistral-7B) -- using both standard classification metrics and new measures of cross-lingual reliability: Calibration Deviation (CD), Decision Bias (B), Language Robustness Score (LRS), and Language Stability Score (LSS). Results show that alignment, not scale, determines stability: aligned models maintain near-invariant accuracy and balanced calibration across typologically distant and low-resource languages (e.g., Lingala, Burmese), while open-weight models exhibit significant prompt-language sensitivity and calibration drift. These findings demonstrate that multilingual alignment enables language-agnostic reasoning and provide practical guidance for humanitarian organizations balancing budget constraints with reliability in multilingual deployment. 

---
# VEHME: A Vision-Language Model For Evaluating Handwritten Mathematics Expressions 

**Authors**: Thu Phuong Nguyen, Duc M. Nguyen, Hyotaek Jeon, Hyunwook Lee, Hyunmin Song, Sungahn Ko, Taehwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.22798)  

**Abstract**: Automatically assessing handwritten mathematical solutions is an important problem in educational technology with practical applications, but it remains a significant challenge due to the diverse formats, unstructured layouts, and symbolic complexity of student work. To address this challenge, we introduce VEHME-a Vision-Language Model for Evaluating Handwritten Mathematics Expressions-designed to assess open-form handwritten math responses with high accuracy and interpretable reasoning traces. VEHME integrates a two-phase training pipeline: (i) supervised fine-tuning using structured reasoning data, and (ii) reinforcement learning that aligns model outputs with multi-dimensional grading objectives, including correctness, reasoning depth, and error localization. To enhance spatial understanding, we propose an Expression-Aware Visual Prompting Module, trained on our synthesized multi-line math expressions dataset to robustly guide attention in visually heterogeneous inputs. Evaluated on AIHub and FERMAT datasets, VEHME achieves state-of-the-art performance among open-source models and approaches the accuracy of proprietary systems, demonstrating its potential as a scalable and accessible tool for automated math assessment. Our training and experiment code is publicly available at our GitHub repository. 

---
# Scalable Supervising Software Agents with Patch Reasoner 

**Authors**: Junjielong Xu, Boyin Tan, Xiaoyuan Liu, Chao Peng, Pengfei Gao, Pinjia He  

**Link**: [PDF](https://arxiv.org/pdf/2510.22775)  

**Abstract**: While large language model agents have advanced software engineering tasks, the unscalable nature of existing test-based supervision is limiting the potential improvement of data scaling. The reason is twofold: (1) building and running test sandbox is rather heavy and fragile, and (2) data with high-coverage tests is naturally rare and threatened by test hacking via edge cases. In this paper, we propose R4P, a patch verifier model to provide scalable rewards for training and testing SWE agents via reasoning. We consider that patch verification is fundamentally a reasoning task, mirroring how human repository maintainers review patches without writing and running new reproduction tests. To obtain sufficient reference and reduce the risk of reward hacking, R4P uses a group-wise objective for RL training, enabling it to verify multiple patches against each other's modification and gain a dense reward for stable training. R4P achieves 72.2% Acc. for verifying patches from SWE-bench-verified, surpassing OpenAI o3. To demonstrate R4P's practicality, we design and train a lite scaffold, Mini-SE, with pure reinforcement learning where all rewards are derived from R4P. As a result, Mini-SE achieves 26.2% Pass@1 on SWE-bench-verified, showing a 10.0% improvement over the original Qwen3-32B. This can be further improved to 32.8% with R4P for test-time scaling. Furthermore, R4P verifies patches within a second, 50x faster than testing on average. The stable scaling curves of rewards and accuracy along with high efficiency reflect R4P's practicality. 

---
# MMPersuade: A Dataset and Evaluation Framework for Multimodal Persuasion 

**Authors**: Haoyi Qiu, Yilun Zhou, Pranav Narayanan Venkit, Kung-Hsiang Huang, Jiaxin Zhang, Nanyun Peng, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22768)  

**Abstract**: As Large Vision-Language Models (LVLMs) are increasingly deployed in domains such as shopping, health, and news, they are exposed to pervasive persuasive content. A critical question is how these models function as persuadees-how and why they can be influenced by persuasive multimodal inputs. Understanding both their susceptibility to persuasion and the effectiveness of different persuasive strategies is crucial, as overly persuadable models may adopt misleading beliefs, override user preferences, or generate unethical or unsafe outputs when exposed to manipulative messages. We introduce MMPersuade, a unified framework for systematically studying multimodal persuasion dynamics in LVLMs. MMPersuade contributes (i) a comprehensive multimodal dataset that pairs images and videos with established persuasion principles across commercial, subjective and behavioral, and adversarial contexts, and (ii) an evaluation framework that quantifies both persuasion effectiveness and model susceptibility via third-party agreement scoring and self-estimated token probabilities on conversation histories. Our study of six leading LVLMs as persuadees yields three key insights: (i) multimodal inputs substantially increase persuasion effectiveness-and model susceptibility-compared to text alone, especially in misinformation scenarios; (ii) stated prior preferences decrease susceptibility, yet multimodal information maintains its persuasive advantage; and (iii) different strategies vary in effectiveness across contexts, with reciprocity being most potent in commercial and subjective contexts, and credibility and logic prevailing in adversarial contexts. By jointly analyzing persuasion effectiveness and susceptibility, MMPersuade provides a principled foundation for developing models that are robust, preference-consistent, and ethically aligned when engaging with persuasive multimodal content. 

---
# Iterative Layer Pruning for Efficient Translation Inference 

**Authors**: Yasmin Moslem, Muhammad Hazim Al Farouq, John D. Kelleher  

**Link**: [PDF](https://arxiv.org/pdf/2510.22763)  

**Abstract**: Large language models (LLMs) have transformed many areas of natural language processing, including machine translation. However, efficient deployment of LLMs remains challenging due to their intensive computational requirements. In this paper, we address this challenge and present our submissions to the Model Compression track at the Conference on Machine Translation (WMT 2025). In our experiments, we investigate iterative layer pruning guided by layer importance analysis. We evaluate this method using the Aya-Expanse-8B model for translation from Czech to German, and from English to Egyptian Arabic. Our approach achieves substantial reductions in model size and inference time, while maintaining the translation quality of the baseline models. 

---
# EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models 

**Authors**: Li Zhou, Lutong Yu, You Lyu, Yihang Lin, Zefeng Zhao, Junyi Ao, Yuhao Zhang, Benyou Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.22758)  

**Abstract**: Speech Language Models (SLMs) have made significant progress in spoken language understanding. Yet it remains unclear whether they can fully perceive non lexical vocal cues alongside spoken words, and respond with empathy that aligns with both emotional and contextual factors. Existing benchmarks typically evaluate linguistic, acoustic, reasoning, or dialogue abilities in isolation, overlooking the integration of these skills that is crucial for human-like, emotionally intelligent conversation. We present EchoMind, the first interrelated, multi-level benchmark that simulates the cognitive process of empathetic dialogue through sequential, context-linked tasks: spoken-content understanding, vocal-cue perception, integrated reasoning, and response generation. All tasks share identical and semantically neutral scripts that are free of explicit emotional or contextual cues, and controlled variations in vocal style are used to test the effect of delivery independent of the transcript. EchoMind is grounded in an empathy-oriented framework spanning 3 coarse and 12 fine-grained dimensions, encompassing 39 vocal attributes, and evaluated using both objective and subjective metrics. Testing 12 advanced SLMs reveals that even state-of-the-art models struggle with high-expressive vocal cues, limiting empathetic response quality. Analyses of prompt strength, speech source, and ideal vocal cue recognition reveal persistent weaknesses in instruction-following, resilience to natural speech variability, and effective use of vocal cues for empathy. These results underscore the need for SLMs that integrate linguistic content with diverse vocal cues to achieve truly empathetic conversational ability. 

---
# Beyond Semantics: How Temporal Biases Shape Retrieval in Transformer and State-Space Models 

**Authors**: Anooshka Bajaj, Deven Mahesh Mistry, Sahaj Singh Maini, Yash Aggarwal, Zoran Tiganj  

**Link**: [PDF](https://arxiv.org/pdf/2510.22752)  

**Abstract**: In-context learning is governed by both temporal and semantic relationships, shaping how Large Language Models (LLMs) retrieve contextual information. Analogous to human episodic memory, where the retrieval of specific events is enabled by separating events that happened at different times, this work probes the ability of various pretrained LLMs, including transformer and state-space models, to differentiate and retrieve temporally separated events. Specifically, we prompted models with sequences containing multiple presentations of the same token, which reappears at the sequence end. By fixing the positions of these repeated tokens and permuting all others, we removed semantic confounds and isolated temporal effects on next-token prediction. Across diverse sequences, models consistently placed the highest probabilities on tokens following a repeated token, but with a notable bias for those nearest the beginning or end of the input. An ablation experiment linked this phenomenon in transformers to induction heads. Extending the analysis to unique semantic contexts with partial overlap further demonstrated that memories embedded in the middle of a prompt are retrieved less reliably. Despite architectural differences, state-space and transformer models showed comparable temporal biases. Our findings deepen the understanding of temporal biases in in-context learning and offer an illustration of how these biases can enable temporal separation and episodic retrieval. 

---
# Low-Resource Dialect Adaptation of Large Language Models: A French Dialect Case-Study 

**Authors**: Eeham Khan, Firas Saidani, Owen Van Esbroeck, Richard Khoury, Leila Kosseim  

**Link**: [PDF](https://arxiv.org/pdf/2510.22747)  

**Abstract**: Despite the widespread adoption of large language models (LLMs), their strongest capabilities remain largely confined to a small number of high-resource languages for which there is abundant training data. Recently, continual pre-training (CPT) has emerged as a means to fine-tune these models to low-resource regional dialects. In this paper, we study the use of CPT for dialect learning under tight data and compute budgets. Using low-rank adaptation (LoRA) and compute-efficient continual pre-training, we adapt three LLMs to the Québec French dialect using a very small dataset and benchmark them on the COLE suite. Our experiments demonstrate an improvement on the minority dialect benchmarks with minimal regression on the prestige language benchmarks with under 1% of model parameters updated. Analysis of the results demonstrate that gains are highly contingent on corpus composition. These findings indicate that CPT with parameter-efficient fine-tuning (PEFT) can narrow the dialect gap by providing cost-effective and sustainable language resource creation, expanding high-quality LLM access to minority linguistic communities. We release the first Québec French LLMs on HuggingFace. 

---
# $\text{E}^2\text{Rank}$: Your Text Embedding can Also be an Effective and Efficient Listwise Reranker 

**Authors**: Qi Liu, Yanzhao Zhang, Mingxin Li, Dingkun Long, Pengjun Xie, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2510.22733)  

**Abstract**: Text embedding models serve as a fundamental component in real-world search applications. By mapping queries and documents into a shared embedding space, they deliver competitive retrieval performance with high efficiency. However, their ranking fidelity remains limited compared to dedicated rerankers, especially recent LLM-based listwise rerankers, which capture fine-grained query-document and document-document interactions. In this paper, we propose a simple yet effective unified framework $\text{E}^2\text{Rank}$, means Efficient Embedding-based Ranking (also means Embedding-to-Rank), which extends a single text embedding model to perform both high-quality retrieval and listwise reranking through continued training under a listwise ranking objective, thereby achieving strong effectiveness with remarkable efficiency. By applying cosine similarity between the query and document embeddings as a unified ranking function, the listwise ranking prompt, which is constructed from the original query and its candidate documents, serves as an enhanced query enriched with signals from the top-K documents, akin to pseudo-relevance feedback (PRF) in traditional retrieval models. This design preserves the efficiency and representational quality of the base embedding model while significantly improving its reranking performance. Empirically, $\textrm{E}^2\text{Rank}$ achieves state-of-the-art results on the BEIR reranking benchmark and demonstrates competitive performance on the reasoning-intensive BRIGHT benchmark, with very low reranking latency. We also show that the ranking training process improves embedding performance on the MTEB benchmark. Our findings indicate that a single embedding model can effectively unify retrieval and reranking, offering both computational efficiency and competitive ranking accuracy. 

---
# SALSA: Single-pass Autoregressive LLM Structured Classification 

**Authors**: Ruslan Berdichevsky, Shai Nahum-Gefen, Elad Ben Zaken  

**Link**: [PDF](https://arxiv.org/pdf/2510.22691)  

**Abstract**: Despite their impressive generalization capabilities, instruction-tuned Large Language Models often underperform on text classification benchmarks. We introduce SALSA, a coherent pipeline that combines structured prompting, class-to-token mapping, and parameter-efficient fine-tuning, thereby avoiding cold-start training. Each class label is mapped to a distinct output token, and prompts are constructed to elicit a single-token response. During inference, the model's output is projected only onto the logits of the relevant class tokens, enabling efficient and accurate classification in a single forward pass. SALSA achieves state-of-the-art results across diverse benchmarks, demonstrating its robustness and scalability for LLM-based classification applications. 

---
# Rule-Based Explanations for Retrieval-Augmented LLM Systems 

**Authors**: Joel Rorseth, Parke Godfrey, Lukasz Golab, Divesh Srivastava, Jarek Szlichta  

**Link**: [PDF](https://arxiv.org/pdf/2510.22689)  

**Abstract**: If-then rules are widely used to explain machine learning models; e.g., "if employed = no, then loan application = rejected." We present the first proposal to apply rules to explain the emerging class of large language models (LLMs) with retrieval-augmented generation (RAG). Since RAG enables LLM systems to incorporate retrieved information sources at inference time, rules linking the presence or absence of sources can explain output provenance; e.g., "if a Times Higher Education ranking article is retrieved, then the LLM ranks Oxford first." To generate such rules, a brute force approach would probe the LLM with all source combinations and check if the presence or absence of any sources leads to the same output. We propose optimizations to speed up rule generation, inspired by Apriori-like pruning from frequent itemset mining but redefined within the scope of our novel problem. We conclude with qualitative and quantitative experiments demonstrating our solutions' value and efficiency. 

---
# Conjugate Relation Modeling for Few-Shot Knowledge Graph Completion 

**Authors**: Zilong Wang, Qingtian Zeng, Hua Duan, Cheng Cheng, Minghao Zou, Ziyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22656)  

**Abstract**: Few-shot Knowledge Graph Completion (FKGC) infers missing triples from limited support samples, tackling long-tail distribution challenges. Existing methods, however, struggle to capture complex relational patterns and mitigate data sparsity. To address these challenges, we propose a novel FKGC framework for conjugate relation modeling (CR-FKGC). Specifically, it employs a neighborhood aggregation encoder to integrate higher-order neighbor information, a conjugate relation learner combining an implicit conditional diffusion relation module with a stable relation module to capture stable semantics and uncertainty offsets, and a manifold conjugate decoder for efficient evaluation and inference of missing triples in manifold space. Experiments on three benchmarks demonstrate that our method achieves superior performance over state-of-the-art methods. 

---
# Culturally Grounded Physical Commonsense Reasoning in Italian and English: A Submission to the MRL 2025 Shared Task 

**Authors**: Marco De Santis, Lisa Alazraki  

**Link**: [PDF](https://arxiv.org/pdf/2510.22631)  

**Abstract**: This paper presents our submission to the MRL 2025 Shared Task on Multilingual Physical Reasoning Datasets. The objective of the shared task is to create manually-annotated evaluation data in the physical commonsense reasoning domain, for languages other than English, following a format similar to PIQA. Our contribution, FormaMentis, is a novel benchmark for physical commonsense reasoning that is grounded in Italian language and culture. The data samples in FormaMentis are created by expert annotators who are native Italian speakers and are familiar with local customs and norms. The samples are additionally translated into English, while preserving the cultural elements unique to the Italian context. 

---
# Integrating Linguistics and AI: Morphological Analysis and Corpus development of Endangered Toto Language of West Bengal 

**Authors**: Ambalika Guha, Sajal Saha, Debanjan Ballav, Soumi Mitra, Hritwick Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2510.22629)  

**Abstract**: Preserving linguistic diversity is necessary as every language offers a distinct perspective on the world. There have been numerous global initiatives to preserve endangered languages through documentation. This paper is a part of a project which aims to develop a trilingual (Toto-Bangla-English) language learning application to digitally archive and promote the endangered Toto language of West Bengal, India. This application, designed for both native Toto speakers and non-native learners, aims to revitalize the language by ensuring accessibility and usability through Unicode script integration and a structured language corpus. The research includes detailed linguistic documentation collected via fieldwork, followed by the creation of a morpheme-tagged, trilingual corpus used to train a Small Language Model (SLM) and a Transformer-based translation engine. The analysis covers inflectional morphology such as person-number-gender agreement, tense-aspect-mood distinctions, and case marking, alongside derivational strategies that reflect word-class changes. Script standardization and digital literacy tools were also developed to enhance script usage. The study offers a sustainable model for preserving endangered languages by incorporating traditional linguistic methodology with AI. This bridge between linguistic research with technological innovation highlights the value of interdisciplinary collaboration for community-based language revitalization. 

---
# PerCoR: Evaluating Commonsense Reasoning in Persian via Multiple-Choice Sentence Completion 

**Authors**: Morteza Alikhani, Mohammadtaha Bagherifard, Erfan Zinvandi, Mehran Sarmadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.22616)  

**Abstract**: We introduced PerCoR (Persian Commonsense Reasoning), the first large-scale Persian benchmark for commonsense reasoning. PerCoR contains 106K multiple-choice sentence-completion problems drawn from more than forty news, cultural, and other web sources. We introduce a novel conjunction-based segmentation strategy to generate coherent sentence-completion pairs, enabling broad topical and structural diversity. To create challenging distractors, we propose DRESS-AF (Distractor Ranking via Embedding Similarity Scoring and Adversarial Filtering), a generation-free adversarial filtering method that selects distractors from the pool of gold continuations while maximising model confusion. Human annotators score 89% on PerCoR, while OpenAI-o3 achieves the highest performance at 92.18%, followed closely by Claude-Sonnet-3.7 (91.17%). The strongest open-source model, DeepSeek-R1, reaches 82.51%, underscoring both the dataset's difficulty and the remaining performance gap in Persian commonsense reasoning. We further show that DRESS-AF transfers to the English HellaSwag benchmark, increasing its difficulty without hurting human solvability. The dataset is available at this https URL. 

---
# Personal Care Utility (PCU): Building the Health Infrastructure for Everyday Insight and Guidance 

**Authors**: Mahyar Abbasian, Ramesh Jain  

**Link**: [PDF](https://arxiv.org/pdf/2510.22602)  

**Abstract**: Building on decades of success in digital infrastructure and biomedical innovation, we propose the Personal Care Utility (PCU) - a cybernetic system for lifelong health guidance. PCU is conceived as a global, AI-powered utility that continuously orchestrates multimodal data, knowledge, and services to assist individuals and populations alike. Drawing on multimodal agents, event-centric modeling, and contextual inference, it offers three essential capabilities: (1) trusted health information tailored to the individual, (2) proactive health navigation and behavior guidance, and (3) ongoing interpretation of recovery and treatment response after medical events. Unlike conventional episodic care, PCU functions as an ambient, adaptive companion - observing, interpreting, and guiding health in real time across daily life. By integrating personal sensing, experiential computing, and population-level analytics, PCU promises not only improved outcomes for individuals but also a new substrate for public health and scientific discovery. We describe the architecture, design principles, and implementation challenges of this emerging paradigm. 

---
# AutoBench: Automating LLM Evaluation through Reciprocal Peer Assessment 

**Authors**: Dario Loi, Elena Maria Muià, Federico Siciliano, Giovanni Trappolini, Vincenzo Crisà, Peter Kruger, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2510.22593)  

**Abstract**: We present AutoBench, a fully automated and self-sustaining framework for evaluating Large Language Models (LLMs) through reciprocal peer assessment. This paper provides a rigorous scientific validation of the AutoBench methodology, originally developed as an open-source project by eZecute S.R.L.. Unlike static benchmarks that suffer from test-set contamination and limited adaptability, AutoBench dynamically generates novel evaluation tasks while models alternately serve as question generators, contestants, and judges across diverse domains. An iterative weighting mechanism amplifies the influence of consistently reliable evaluators, aggregating peer judgments into consensus-based rankings that reflect collective model agreement. Our experiments demonstrate strong correlations with established benchmarks including MMLU-Pro and GPQA (respectively 78\% and 63\%), validating this peer-driven evaluation paradigm. The multi-judge design significantly outperforms single-judge baselines, confirming that distributed evaluation produces more robust and human-consistent assessments. AutoBench offers a scalable, contamination-resistant alternative to static benchmarks for the continuous evaluation of evolving language models. 

---
# Pedagogy-driven Evaluation of Generative AI-powered Intelligent Tutoring Systems 

**Authors**: Kaushal Kumar Maurya, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2510.22581)  

**Abstract**: The interdisciplinary research domain of Artificial Intelligence in Education (AIED) has a long history of developing Intelligent Tutoring Systems (ITSs) by integrating insights from technological advancements, educational theories, and cognitive psychology. The remarkable success of generative AI (GenAI) models has accelerated the development of large language model (LLM)-powered ITSs, which have potential to imitate human-like, pedagogically rich, and cognitively demanding tutoring. However, the progress and impact of these systems remain largely untraceable due to the absence of reliable, universally accepted, and pedagogy-driven evaluation frameworks and benchmarks. Most existing educational dialogue-based ITS evaluations rely on subjective protocols and non-standardized benchmarks, leading to inconsistencies and limited generalizability. In this work, we take a step back from mainstream ITS development and provide comprehensive state-of-the-art evaluation practices, highlighting associated challenges through real-world case studies from careful and caring AIED research. Finally, building on insights from previous interdisciplinary AIED research, we propose three practical, feasible, and theoretically grounded research directions, rooted in learning science principles and aimed at establishing fair, unified, and scalable evaluation methodologies for ITSs. 

---
# A Closed-Loop Personalized Learning Agent Integrating Neural Cognitive Diagnosis, Bounded-Ability Adaptive Testing, and LLM-Driven Feedback 

**Authors**: Zhifeng Wang, Xinyue Zheng, Chunyan Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.22559)  

**Abstract**: As information technology advances, education is moving from one-size-fits-all instruction toward personalized learning. However, most methods handle modeling, item selection, and feedback in isolation rather than as a closed loop. This leads to coarse or opaque student models, assumption-bound adaptivity that ignores diagnostic posteriors, and generic, non-actionable feedback. To address these limitations, this paper presents an end-to-end personalized learning agent, EduLoop-Agent, which integrates a Neural Cognitive Diagnosis model (NCD), a Bounded-Ability Estimation Computerized Adaptive Testing strategy (BECAT), and large language models (LLMs). The NCD module provides fine-grained estimates of students' mastery at the knowledge-point level; BECAT dynamically selects subsequent items to maximize relevance and learning efficiency; and LLMs convert diagnostic signals into structured, actionable feedback. Together, these components form a closed-loop framework of ``Diagnosis--Recommendation--Feedback.'' Experiments on the ASSISTments dataset show that the NCD module achieves strong performance on response prediction while yielding interpretable mastery assessments. The adaptive recommendation strategy improves item relevance and personalization, and the LLM-based feedback offers targeted study guidance aligned with identified weaknesses. Overall, the results indicate that the proposed design is effective and practically deployable, providing a feasible pathway to generating individualized learning trajectories in intelligent education. 

---
# SABlock: Semantic-Aware KV Cache Eviction with Adaptive Compression Block Size 

**Authors**: Jinhan Chen, Jianchun Liu, Hongli Xu, Xianjun Gao, Shilong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22556)  

**Abstract**: The growing memory footprint of the Key-Value (KV) cache poses a severe scalability bottleneck for long-context Large Language Model (LLM) inference. While KV cache eviction has emerged as an effective solution by discarding less critical tokens, existing token-, block-, and sentence-level compression methods struggle to balance semantic coherence and memory efficiency. To this end, we introduce SABlock, a \underline{s}emantic-aware KV cache eviction framework with \underline{a}daptive \underline{block} sizes. Specifically, SABlock first performs semantic segmentation to align compression boundaries with linguistic structures, then applies segment-guided token scoring to refine token importance estimation. Finally, for each segment, a budget-driven search strategy adaptively determines the optimal block size that preserves semantic integrity while improving compression efficiency under a given cache budget. Extensive experiments on long-context benchmarks demonstrate that SABlock consistently outperforms state-of-the-art baselines under the same memory budgets. For instance, on Needle-in-a-Haystack (NIAH), SABlock achieves 99.9% retrieval accuracy with only 96 KV entries, nearly matching the performance of the full-cache baseline that retains up to 8K entries. Under a fixed cache budget of 1,024, SABlock further reduces peak memory usage by 46.28% and achieves up to 9.5x faster decoding on a 128K context length. 

---
# LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges? 

**Authors**: Ziyuan He, Yuxuan Wang, Jiaqi Li, Kexin Liang, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22548)  

**Abstract**: Large language models (LLMs) are equipped with increasingly extended context windows recently, yet their long context understanding capabilities over long dependency tasks remain fundamentally limited and underexplored. This gap is especially significant in many real-world long-context applications that were rarely benchmarked. In this paper, we introduce LooGLE v2, a novel benchmark designed to evaluate LLMs' long context ability in real-world applications and scenarios. Our benchmark consists of automatically collected real-world long texts, ranging from 16k to 2M tokens, encompassing domains in law, finance, game and code. Accordingly, we delicately design 10 types of domain-specific long-dependency tasks and generate 1,934 QA instances with various diversity and complexity in a scalable data curation pipeline for further practical needs. We conduct a comprehensive assessment of 6 locally deployed and 4 API-based LLMs. The evaluation results show that even the best-performing model achieves only a 59.2% overall score on our benchmark. Despite the extensive context windows, popular LLMs are only capable of understanding a much shorter length of context than they claim to be, revealing significant limitations in their ability to handle real-world tasks with long dependencies and highlighting substantial room for model improvement in practical long-context understanding. 

---
# Text to Trust: Evaluating Fine-Tuning and LoRA Trade-offs in Language Models for Unfair Terms of Service Detection 

**Authors**: Noshitha Padma Pratyusha Juttu, Sahithi Singireddy, Sravani Gona, Sujal Timilsina  

**Link**: [PDF](https://arxiv.org/pdf/2510.22531)  

**Abstract**: Large Language Models (LLMs) have transformed text understanding, yet their adaptation to specialized legal domains remains constrained by the cost of full fine-tuning. This study provides a systematic evaluation of fine tuning, parameter efficient adaptation (LoRA, QLoRA), and zero-shot prompting strategies for unfair clause detection in Terms of Service (ToS) documents, a key application in legal NLP. We finetune BERT and DistilBERT, apply 4-bit Low-Rank Adaptation (LoRA) to models such as TinyLlama, LLaMA 3B/7B, and SaulLM, and evaluate GPT-4o and O-versions in zero-shot settings. Experiments on the CLAUDETTE-ToS benchmark and the Multilingual Scraper Corpus show that full fine-tuning achieves the strongest precision recall balance, while LoRA-based models provide competitive recall with up to 3x lower memory cost. These findings highlight practical design trade-offs for efficient and domain-adapted LLMs, contributing open baselines for fine-tuning research in legal text processing. 

---
# A Sociophonetic Analysis of Racial Bias in Commercial ASR Systems Using the Pacific Northwest English Corpus 

**Authors**: Michael Scott, Siyu Liang, Alicia Wassink, Gina-Anne Levow  

**Link**: [PDF](https://arxiv.org/pdf/2510.22495)  

**Abstract**: This paper presents a systematic evaluation of racial bias in four major commercial automatic speech recognition (ASR) systems using the Pacific Northwest English (PNWE) corpus. We analyze transcription accuracy across speakers from four ethnic backgrounds (African American, Caucasian American, ChicanX, and Yakama) and examine how sociophonetic variation contributes to differential system performance. We introduce a heuristically-determined Phonetic Error Rate (PER) metric that links recognition errors to specific linguistically motivated variables derived from sociophonetic annotation. Our analysis of eleven sociophonetic features reveals that vowel quality variation, particularly resistance to the low-back merger and pre-nasal merger patterns, is systematically associated with differential error rates across ethnic groups, with the most pronounced effects for African American speakers across all evaluated systems. These findings demonstrate that acoustic modeling of dialectal phonetic variation, rather than lexical or syntactic factors, remains a primary source of bias in commercial ASR systems. The study establishes the PNWE corpus as a valuable resource for bias evaluation in speech technologies and provides actionable guidance for improving ASR performance through targeted representation of sociophonetic diversity in training data. 

---
# The Limits of Data Scaling: Sub-token Utilization and Acoustic Saturation in Multilingual ASR 

**Authors**: Siyu Liang, Nicolas Ballier, Gina-Anne Levow, Richard Wright  

**Link**: [PDF](https://arxiv.org/pdf/2510.22492)  

**Abstract**: How much audio is needed to fully observe a multilingual ASR model's learned sub-token inventory across languages, and does data disparity in multilingual pre-training affect how these tokens are utilized during inference? We address this question by analyzing Whisper's decoding behavior during inference across 49 languages. By logging decoding candidate sub-tokens and tracking their cumulative discovery over time, we study the utilization pattern of the model's sub-token space. Results show that the total number of discovered tokens remains largely independent of a language's pre-training hours, indicating that data disparity does not strongly influence lexical diversity in the model's hypothesis space. Sub-token discovery rates follow a consistent exponential saturation pattern across languages, suggesting a stable time window after which additional audio yields minimal new sub-token activation. We refer to this convergence threshold as acoustic saturation time (AST). Further analyses of rank-frequency distributions reveal Zipf-like patterns better modeled by a Zipf-Mandelbrot law, and mean sub-token length shows a positive correlation with resource level. Additionally, those metrics show more favorable patterns for languages in the Latin script than those in scripts such as Cyrillic, CJK, and Semitic. Together, our study suggests that sub-token utilization during multilingual ASR inference is constrained more by the statistical, typological, and orthographic structure of the speech than by training data scale, providing an empirical basis for more equitable corpus construction and cross-lingual evaluation. 

---
# Frustratingly Easy Task-aware Pruning for Large Language Models 

**Authors**: Yuanhe Tian, Junjie Liu, Xican Yang, Haishan Ye, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.22489)  

**Abstract**: Pruning provides a practical solution to reduce the resources required to run large language models (LLMs) to benefit from their effective capabilities as well as control their cost for training and inference. Research on LLM pruning often ranks the importance of LLM parameters using their magnitudes and calibration-data activations and removes (or masks) the less important ones, accordingly reducing LLMs' size. However, these approaches primarily focus on preserving the LLM's ability to generate fluent sentences, while neglecting performance on specific domains and tasks. In this paper, we propose a simple yet effective pruning approach for LLMs that preserves task-specific capabilities while shrinking their parameter space. We first analyze how conventional pruning minimizes loss perturbation under general-domain calibration and extend this formulation by incorporating task-specific feature distributions into the importance computation of existing pruning algorithms. Thus, our framework computes separate importance scores using both general and task-specific calibration data, partitions parameters into shared and exclusive groups based on activation-norm differences, and then fuses their scores to guide the pruning process. This design enables our method to integrate seamlessly with various foundation pruning techniques and preserve the LLM's specialized abilities under compression. Experiments on widely used benchmarks demonstrate that our approach is effective and consistently outperforms the baselines with identical pruning ratios and different settings. 

---
# The Tonogenesis Continuum in Tibetan: A Computational Investigation 

**Authors**: Siyu Liang, Zhaxi Zerong  

**Link**: [PDF](https://arxiv.org/pdf/2510.22485)  

**Abstract**: Tonogenesis-the historical process by which segmental contrasts evolve into lexical tone-has traditionally been studied through comparative reconstruction and acoustic phonetics. We introduce a computational approach that quantifies the functional role of pitch at different stages of this sound change by measuring how pitch manipulation affects automatic speech recognition (ASR) performance. Through analysis on the sensitivity to pitch-flattening from a set of closely related Tibetan languages, we find evidence of a tonogenesis continuum: atonal Amdo dialects tolerate pitch removal the most, while fully tonal U-Tsang varieties show severe degradation, and intermediate Kham dialects fall measurably between these extremes. These gradient effects demonstrate how ASR models implicitly learn the shifting functional load of pitch as languages transition from consonant-based to tone-based lexical contrasts. Our findings show that computational methods can capture fine-grained stages of sound change and suggest that traditional functional load metrics, based solely on minimal pairs, may overestimate pitch dependence in transitional systems where segmental and suprasegmental cues remain phonetically intertwined. 

---
# CHOIR: Collaborative Harmonization fOr Inference Robustness 

**Authors**: Xiangjue Dong, Cong Wang, Maria Teleki, Millennium Bismay, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2510.22475)  

**Abstract**: Persona-assigned Large Language Models (LLMs) can adopt diverse roles, enabling personalized and context-aware reasoning. However, even minor demographic perturbations in personas, such as simple pronoun changes, can alter reasoning trajectories, leading to divergent sets of correct answers. Instead of treating these variations as biases to be mitigated, we explore their potential as a constructive resource to improve reasoning robustness. We propose CHOIR (Collaborative Harmonization fOr Inference Robustness), a test-time framework that harmonizes multiple persona-conditioned reasoning signals into a unified prediction. CHOIR orchestrates a collaborative decoding process among counterfactual personas, dynamically balancing agreement and divergence in their reasoning paths. Experiments on various reasoning benchmarks demonstrate that CHOIR consistently enhances performance across demographics, model architectures, scales, and tasks - without additional training. Improvements reach up to 26.4% for individual demographic groups and 19.2% on average across five demographics. It remains effective even when base personas are suboptimal. By reframing persona variation as a constructive signal, CHOIR provides a scalable and generalizable approach to more reliable LLM reasoning. 

---
# Confabulations from ACL Publications (CAP): A Dataset for Scientific Hallucination Detection 

**Authors**: Federica Gamba, Aman Sinha, Timothee Mickus, Raul Vazquez, Patanjali Bhamidipati, Claudio Savelli, Ahana Chattopadhyay, Laura A. Zanella, Yash Kankanampati, Binesh Arakkal Remesh, Aryan Ashok Chandramania, Rohit Agarwal, Chuyuan Li, Ioana Buhnila, Radhika Mamidi  

**Link**: [PDF](https://arxiv.org/pdf/2510.22395)  

**Abstract**: We introduce the CAP (Confabulations from ACL Publications) dataset, a multilingual resource for studying hallucinations in large language models (LLMs) within scientific text generation. CAP focuses on the scientific domain, where hallucinations can distort factual knowledge, as they frequently do. In this domain, however, the presence of specialized terminology, statistical reasoning, and context-dependent interpretations further exacerbates these distortions, particularly given LLMs' lack of true comprehension, limited contextual understanding, and bias toward surface-level generalization. CAP operates in a cross-lingual setting covering five high-resource languages (English, French, Hindi, Italian, and Spanish) and four low-resource languages (Bengali, Gujarati, Malayalam, and Telugu). The dataset comprises 900 curated scientific questions and over 7000 LLM-generated answers from 16 publicly available models, provided as question-answer pairs along with token sequences and corresponding logits. Each instance is annotated with a binary label indicating the presence of a scientific hallucination, denoted as a factuality error, and a fluency label, capturing issues in the linguistic quality or naturalness of the text. CAP is publicly released to facilitate advanced research on hallucination detection, multilingual evaluation of LLMs, and the development of more reliable scientific NLP systems. 

---
# VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations 

**Authors**: Yupeng Xie, Zhiyang Zhang, Yifan Wu, Sirong Lu, Jiayi Zhang, Zhaoyang Yu, Jinlin Wang, Sirui Hong, Bang Liu, Chenglin Wu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.22373)  

**Abstract**: Visualization, a domain-specific yet widely used form of imagery, is an effective way to turn complex datasets into intuitive insights, and its value depends on whether data are faithfully represented, clearly communicated, and aesthetically designed. However, evaluating visualization quality is challenging: unlike natural images, it requires simultaneous judgment across data encoding accuracy, information expressiveness, and visual aesthetics. Although multimodal large language models (MLLMs) have shown promising performance in aesthetic assessment of natural images, no systematic benchmark exists for measuring their capabilities in evaluating visualizations. To address this, we propose VisJudge-Bench, the first comprehensive benchmark for evaluating MLLMs' performance in assessing visualization aesthetics and quality. It contains 3,090 expert-annotated samples from real-world scenarios, covering single visualizations, multiple visualizations, and dashboards across 32 chart types. Systematic testing on this benchmark reveals that even the most advanced MLLMs (such as GPT-5) still exhibit significant gaps compared to human experts in judgment, with a Mean Absolute Error (MAE) of 0.551 and a correlation with human ratings of only 0.429. To address this issue, we propose VisJudge, a model specifically designed for visualization aesthetics and quality assessment. Experimental results demonstrate that VisJudge significantly narrows the gap with human judgment, reducing the MAE to 0.442 (a 19.8% reduction) and increasing the consistency with human experts to 0.681 (a 58.7% improvement) compared to GPT-5. The benchmark is available at this https URL. 

---
# GigaEmbeddings: Efficient Russian Language Embedding Model 

**Authors**: Egor Kolodin, Daria Khomich, Nikita Savushkin, Anastasia Ianina, Fyodor Minkin  

**Link**: [PDF](https://arxiv.org/pdf/2510.22369)  

**Abstract**: We introduce GigaEmbeddings, a novel framework for training high-performance Russian-focused text embeddings through hierarchical instruction tuning of the decoder-only LLM designed specifically for Russian language (GigaChat-3B). Our three-stage pipeline, comprising large-scale contrastive pre-training in web-scale corpora, fine-tuning with hard negatives, and multitask generalization across retrieval, classification, and clustering tasks, addresses key limitations of existing methods by unifying diverse objectives and leveraging synthetic data generation. Architectural innovations include bidirectional attention for contextual modeling, latent attention pooling for robust sequence aggregation, and strategic pruning of 25% of transformer layers to enhance efficiency without compromising performance. Evaluated on the ruMTEB benchmark spanning 23 multilingual tasks, GigaEmbeddings achieves state-of-the-art results (69.1 avg. score), outperforming strong baselines with a larger number of parameters. 

---
# Irony Detection in Urdu Text: A Comparative Study Using Machine Learning Models and Large Language Models 

**Authors**: Fiaz Ahmad, Nisar Hussain, Amna Qasim, Momina Hafeez, Muhammad Usman Grigori Sidorov, Alexander Gelbukh  

**Link**: [PDF](https://arxiv.org/pdf/2510.22356)  

**Abstract**: Ironic identification is a challenging task in Natural Language Processing, particularly when dealing with languages that differ in syntax and cultural context. In this work, we aim to detect irony in Urdu by translating an English Ironic Corpus into the Urdu language. We evaluate ten state-of-the-art machine learning algorithms using GloVe and Word2Vec embeddings, and compare their performance with classical methods. Additionally, we fine-tune advanced transformer-based models, including BERT, RoBERTa, LLaMA 2 (7B), LLaMA 3 (8B), and Mistral, to assess the effectiveness of large-scale models in irony detection. Among machine learning models, Gradient Boosting achieved the best performance with an F1-score of 89.18%. Among transformer-based models, LLaMA 3 (8B) achieved the highest performance with an F1-score of 94.61%. These results demonstrate that combining transliteration techniques with modern NLP models enables robust irony detection in Urdu, a historically low-resource language. 

---
# FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation 

**Authors**: Mohammad Aghajani Asl, Majid Asgari-Bidhendi, Behrooz Minaei-Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2510.22344)  

**Abstract**: While Retrieval-Augmented Generation (RAG) mitigates hallucination and knowledge staleness in Large Language Models (LLMs), existing frameworks often falter on complex, multi-hop queries that require synthesizing information from disparate sources. Current advanced RAG methods, employing iterative or adaptive strategies, lack a robust mechanism to systematically identify and fill evidence gaps, often propagating noise or failing to gather a comprehensive context. We introduce FAIR-RAG, a novel agentic framework that transforms the standard RAG pipeline into a dynamic, evidence-driven reasoning process. At its core is an Iterative Refinement Cycle governed by a module we term Structured Evidence Assessment (SEA). The SEA acts as an analytical gating mechanism: it deconstructs the initial query into a checklist of required findings and audits the aggregated evidence to identify confirmed facts and, critically, explicit informational gaps. These gaps provide a precise signal to an Adaptive Query Refinement agent, which generates new, targeted sub-queries to retrieve missing information. This cycle repeats until the evidence is verified as sufficient, ensuring a comprehensive context for a final, strictly faithful generation. We conducted experiments on challenging multi-hop QA benchmarks, including HotpotQA, 2WikiMultiHopQA, and MusiQue. In a unified experimental setup, FAIR-RAG significantly outperforms strong baselines. On HotpotQA, it achieves an F1-score of 0.453 -- an absolute improvement of 8.3 points over the strongest iterative baseline -- establishing a new state-of-the-art for this class of methods on these benchmarks. Our work demonstrates that a structured, evidence-driven refinement process with explicit gap analysis is crucial for unlocking reliable and accurate reasoning in advanced RAG systems for complex, knowledge-intensive tasks. 

---
# Multilingual Target-Stance Extraction 

**Authors**: Ethan Mines, Bonnie Dorr  

**Link**: [PDF](https://arxiv.org/pdf/2510.22334)  

**Abstract**: Social media enables data-driven analysis of public opinion on contested issues. Target-Stance Extraction (TSE) is the task of identifying the target discussed in a document and the document's stance towards that target. Many works classify stance towards a given target in a multilingual setting, but all prior work in TSE is English-only. This work introduces the first multilingual TSE benchmark, spanning Catalan, Estonian, French, Italian, Mandarin, and Spanish corpora. It manages to extend the original TSE pipeline to a multilingual setting without requiring separate models for each language. Our model pipeline achieves a modest F1 score of 12.78, underscoring the increased difficulty of the multilingual task relative to English-only setups and highlighting target prediction as the primary bottleneck. We are also the first to demonstrate the sensitivity of TSE's F1 score to different target verbalizations. Together these serve as a much-needed baseline for resources, algorithms, and evaluation criteria in multilingual TSE. 

---
# Memory-based Language Models: An Efficient, Explainable, and Eco-friendly Approach to Large Language Modeling 

**Authors**: Antal van den Bosch, Ainhoa Risco Patón, Teun Buijse, Peter Berck, Maarten van Gompel  

**Link**: [PDF](https://arxiv.org/pdf/2510.22317)  

**Abstract**: We present memory-based language modeling as an efficient, eco-friendly alternative to deep neural network-based language modeling. It offers log-linearly scalable next-token prediction performance and strong memorization capabilities. Implementing fast approximations of k-nearest neighbor classification, memory-based language modeling leaves a relatively small ecological footprint both in training and in inference mode, as it relies fully on CPUs and attains low token latencies. Its internal workings are simple and fully transparent. We compare our implementation of memory-based language modeling, OLIFANT, with GPT-2 and GPT-Neo on next-token prediction accuracy, estimated emissions and speeds, and offer some deeper analyses of the model. 

---
# Supervised Fine-Tuning or In-Context Learning? Evaluating LLMs for Clinical NER 

**Authors**: Andrei Baroian  

**Link**: [PDF](https://arxiv.org/pdf/2510.22285)  

**Abstract**: We study clinical Named Entity Recognition (NER) on the CADEC corpus and compare three families of approaches: (i) BERT-style encoders (BERT Base, BioClinicalBERT, RoBERTa-large), (ii) GPT-4o used with few-shot in-context learning (ICL) under simple vs.\ complex prompts, and (iii) GPT-4o with supervised fine-tuning (SFT). All models are evaluated on standard NER metrics over CADEC's five entity types (ADR, Drug, Disease, Symptom, Finding). RoBERTa-large and BioClinicalBERT offer limited improvements over BERT Base, showing the limit of these family of models. Among LLM settings, simple ICL outperforms a longer, instruction-heavy prompt, and SFT achieves the strongest overall performance (F1 $\approx$ 87.1%), albeit with higher cost. We find that the LLM achieve higher accuracy on simplified tasks, restricting classification to two labels. 

---
# From Slides to Chatbots: Enhancing Large Language Models with University Course Materials 

**Authors**: Tu Anh Dinh, Philipp Nicolas Schumacher, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2510.22272)  

**Abstract**: Large Language Models (LLMs) have advanced rapidly in recent years. One application of LLMs is to support student learning in educational settings. However, prior work has shown that LLMs still struggle to answer questions accurately within university-level computer science courses. In this work, we investigate how incorporating university course materials can enhance LLM performance in this setting. A key challenge lies in leveraging diverse course materials such as lecture slides and transcripts, which differ substantially from typical textual corpora: slides also contain visual elements like images and formulas, while transcripts contain spoken, less structured language. We compare two strategies, Retrieval-Augmented Generation (RAG) and Continual Pre-Training (CPT), to extend LLMs with course-specific knowledge. For lecture slides, we further explore a multi-modal RAG approach, where we present the retrieved content to the generator in image form. Our experiments reveal that, given the relatively small size of university course materials, RAG is more effective and efficient than CPT. Moreover, incorporating slides as images in the multi-modal setting significantly improves performance over text-only retrieval. These findings highlight practical strategies for developing AI assistants that better support learning and teaching, and we hope they inspire similar efforts in other educational contexts. 

---
# PatenTEB: A Comprehensive Benchmark and Model Family for Patent Text Embedding 

**Authors**: Iliass Ayaou, Denis Cavallucci  

**Link**: [PDF](https://arxiv.org/pdf/2510.22264)  

**Abstract**: Patent text embeddings enable prior art search, technology landscaping, and patent analysis, yet existing benchmarks inadequately capture patent-specific challenges. We introduce PatenTEB, a comprehensive benchmark comprising 15 tasks across retrieval, classification, paraphrase, and clustering, with 2.06 million examples. PatenTEB employs domain-stratified splits, domain specific hard negative mining, and systematic coverage of asymmetric fragment-to-document matching scenarios absent from general embedding benchmarks. We develop the patembed model family through multi-task training, spanning 67M to 344M parameters with context lengths up to 4096 tokens. External validation shows strong generalization: patembed-base achieves state-of-the-art on MTEB BigPatentClustering.v2 (0.494 V-measure vs. 0.445 previous best), while patembed-large achieves 0.377 NDCG@100 on DAPFAM. Systematic ablations reveal that multi-task training improves external generalization despite minor benchmark costs, and that domain-pretrained initialization provides consistent advantages across task families. All resources will be made available at this https URL. Keywords: patent retrieval, sentence embeddings, multi-task learning, asymmetric retrieval, benchmark evaluation, contrastive learning. 

---
# SteerX: Disentangled Steering for LLM Personalization 

**Authors**: Xiaoyan Zhao, Ming Yan, Yilun Qiu, Haoting Ni, Yang Zhang, Fuli Feng, Hong Cheng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2510.22256)  

**Abstract**: Large language models (LLMs) have shown remarkable success in recent years, enabling a wide range of applications, including intelligent assistants that support users' daily life and work. A critical factor in building such assistants is personalizing LLMs, as user preferences and needs vary widely. Activation steering, which directly leverages directions representing user preference in the LLM activation space to adjust its behavior, offers a cost-effective way to align the model's outputs with individual users. However, existing methods rely on all historical data to compute the steering vector, ignoring that not all content reflects true user preferences, which undermines the personalization signal. To address this, we propose SteerX, a disentangled steering method that isolates preference-driven components from preference-agnostic components. Grounded in causal inference theory, SteerX estimates token-level causal effects to identify preference-driven tokens, transforms these discrete signals into a coherent description, and then leverages them to steer personalized LLM generation. By focusing on the truly preference-driven information, SteerX produces more accurate activation steering vectors and enhances personalization. Experiments on two representative steering backbone methods across real-world datasets demonstrate that SteerX consistently enhances steering vector quality, offering a practical solution for more effective LLM personalization. 

---
# You Don't Need Prompt Engineering Anymore: The Prompting Inversion 

**Authors**: Imran Khan  

**Link**: [PDF](https://arxiv.org/pdf/2510.22251)  

**Abstract**: Prompt engineering, particularly Chain-of-Thought (CoT) prompting, significantly enhances LLM reasoning capabilities. We introduce "Sculpting," a constrained, rule-based prompting method designed to improve upon standard CoT by reducing errors from semantic ambiguity and flawed common sense.
We evaluate three prompting strategies (Zero Shot, standard CoT, and Sculpting) across three OpenAI model generations (gpt-4o-mini, gpt-4o, gpt-5) using the GSM8K mathematical reasoning benchmark (1,317 problems).
Our findings reveal a "Prompting Inversion": Sculpting provides advantages on gpt-4o (97% vs. 93% for standard CoT), but becomes detrimental on gpt-5 (94.00% vs. 96.36% for CoT on full benchmark). We trace this to a "Guardrail-to-Handcuff" transition where constraints preventing common-sense errors in mid-tier models induce hyper-literalism in advanced models. Our detailed error analysis demonstrates that optimal prompting strategies must co-evolve with model capabilities, suggesting simpler prompts for more capable models. 

---
# Evolution of the lexicon: a probabilistic point of view 

**Authors**: Maurizio Serva  

**Link**: [PDF](https://arxiv.org/pdf/2510.22220)  

**Abstract**: The Swadesh approach for determining the temporal separation between two languages relies on the stochastic process of words replacement (when a complete new word emerges to represent a given concept). It is well known that the basic assumptions of the Swadesh approach are often unrealistic due to various contamination phenomena and misjudgments (horizontal transfers, variations over time and space of the replacement rate, incorrect assessments of cognacy relationships, presence of synonyms, and so on). All of this means that the results cannot be completely correct.
More importantly, even in the unrealistic case that all basic assumptions are satisfied, simple mathematics places limits on the accuracy of estimating the temporal separation between two languages. These limits, which are purely probabilistic in nature and which are often neglected in lexicostatistical studies, are analyzed in detail in this article.
Furthermore, in this work we highlight that the evolution of a language's lexicon is also driven by another stochastic process: gradual lexical modification of words. We show that this process equally also represents a major contribution to the reshaping of the vocabulary of languages over the centuries and we also show, from a purely probabilistic perspective, that taking into account this second random process significantly increases the precision in determining the temporal separation between two languages. 

---
# Estimating the Error of Large Language Models at Pairwise Text Comparison 

**Authors**: Tianyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.22219)  

**Abstract**: We measure LLMs' output error at pairwise text comparison, noting the probability of error in their preferences. Our method does not rely on the ground truth and supports two scenarios: (i) uniform error rate regardless of the order of comparison, estimated with two comparisons for each text pair with either text placed first; (ii) binary positional bias assuming distinct error rates for the two orders of comparison, estimated with repeated comparisons between the texts. The Copeland counting constructs a ranking over the compared texts from pairwise preferences; the ranking reveals the poor scalability of LLM-based pairwise comparison and helps yield the estimates for LLMs' error rates. We apply the method to six LLMs (ChatGPT, Claude, DeepSeek, Gemini, Grok, Qwen) with five types of text input and obtain consistent estimates of LLMs' error. In general, the measured two positional bias terms are similar, close to the uniform error. Considering both the error rates and the robustness to the variation of prompts, Claude obtained the most desirable performance in this experiment. Our model outperforms the biased Bradley-Terry model and the commutativity score in indicating LLMs' error at this task. 

---
# DETECT: Determining Ease and Textual Clarity of German Text Simplifications 

**Authors**: Maria Korobeynikova, Alessia Battisti, Lukas Fischer, Yingqiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.22212)  

**Abstract**: Current evaluation of German automatic text simplification (ATS) relies on general-purpose metrics such as SARI, BLEU, and BERTScore, which insufficiently capture simplification quality in terms of simplicity, meaning preservation, and fluency. While specialized metrics like LENS have been developed for English, corresponding efforts for German have lagged behind due to the absence of human-annotated corpora. To close this gap, we introduce DETECT, the first German-specific metric that holistically evaluates ATS quality across all three dimensions of simplicity, meaning preservation, and fluency, and is trained entirely on synthetic large language model (LLM) responses. Our approach adapts the LENS framework to German and extends it with (i) a pipeline for generating synthetic quality scores via LLMs, enabling dataset creation without human annotation, and (ii) an LLM-based refinement step for aligning grading criteria with simplification requirements. To the best of our knowledge, we also construct the largest German human evaluation dataset for text simplification to validate our metric directly. Experimental results show that DETECT achieves substantially higher correlations with human judgments than widely used ATS metrics, with particularly strong gains in meaning preservation and fluency. Beyond ATS, our findings highlight both the potential and the limitations of LLMs for automatic evaluation and provide transferable guidelines for general language accessibility tasks. 

---
# SentiMaithili: A Benchmark Dataset for Sentiment and Reason Generation for the Low-Resource Maithili Language 

**Authors**: Rahul Ranjan, Mahendra Kumar Gurve, Anuj, Nitin, Yamuna Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2510.22160)  

**Abstract**: Developing benchmark datasets for low-resource languages poses significant challenges, primarily due to the limited availability of native linguistic experts and the substantial time and cost involved in annotation. Given these challenges, Maithili is still underrepresented in natural language processing research. It is an Indo-Aryan language spoken by more than 13 million people in the Purvanchal region of India, valued for its rich linguistic structure and cultural significance. While sentiment analysis has achieved remarkable progress in high-resource languages, resources for low-resource languages, such as Maithili, remain scarce, often restricted to coarse-grained annotations and lacking interpretability mechanisms. To address this limitation, we introduce a novel dataset comprising 3,221 Maithili sentences annotated for sentiment polarity and accompanied by natural language justifications. Moreover, the dataset is carefully curated and validated by linguistic experts to ensure both label reliability and contextual fidelity. Notably, the justifications are written in Maithili, thereby promoting culturally grounded interpretation and enhancing the explainability of sentiment models. Furthermore, extensive experiments using both classical machine learning and state-of-the-art transformer architectures demonstrate the dataset's effectiveness for interpretable sentiment analysis. Ultimately, this work establishes the first benchmark for explainable affective computing in Maithili, thus contributing a valuable resource to the broader advancement of multilingual NLP and explainable AI. 

---
# OlaMind: Towards Human-Like and Hallucination-Safe Customer Service for Retrieval-Augmented Dialogue 

**Authors**: Tianhong Gao, Jundong Shen, Bei Shi, Jiapeng Wang, Ying Ju, Junfeng Yao, Jiao Ran, Yong Zhang, Lin Dong, Huiyu Yu, Tingting Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.22143)  

**Abstract**: Intelligent customer service (ICS) systems via retrieval-augmented generation (RAG) have been widely adopted in Web-based domains such as social platforms and e-commerce, achieving remarkable improvements in automation and efficiency. However, notable limitations still remain: these systems are prone to hallucinations and often generate rigid, mechanical responses, which can introduce business risks and undermine user experience, especially in Web-based customer service interactions under the RAG scenarios. In this paper, we introduce OlaMind, a human-like and hallucination-safe customer service framework for retrieval-augmented dialogue. Specifically, it first leverages a Learn-to-Think stage to learn the reasoning processes and response strategies from human experts, and then employs a Learn-to-Respond stage to perform cold-start supervised fine-tuning (SFT) combined with reinforcement learning (RL) for basic-to-hard self-refinement. Our method significantly enhances human-likeness and naturalness while effectively mitigating hallucinations and critical business risks. We have conducted large-scale online A/B experiments in an industry-level social customer service setting, and extensive experimental results show that OlaMind achieves significant cumulative relative improvements with intelligent resolution rates +28.92%/+18.42% and human takeover rate -6.08%/-7.12% in community-support/livestream-interaction scenarios, respectively, which highlights its consistent effectiveness across diverse real-world applications. The code and data will be publicly available. 

---
# Every Activation Boosted: Scaling General Reasoner to 1 Trillion Open Language Foundation 

**Authors**: Ling-Team, Ang Li, Ben Liu, Binbin Hu, Bing Li, Bingwei Zeng, Borui Ye, Caizhi Tang, Changxin Tian, Chao Huang, Chao Zhang, Chen Qian, Chenchen Ju, Chenchen Li, Chengfu Tang, Chili Fu, Chunshao Ren, Chunwei Wu, Cong Zhang, Cunyin Peng, Dafeng Xu, Daixin Wang, Dalong Zhang, Dingnan Jin, Dingyuan Zhu, Dongke Hu, Fangzheng Zhao, Feifan Wu, Feng Zhu, Gangshan Wang, Haitao Zhang, Hailin Zhao, Hanxiao Zhang, Hanzi Wang, Hao Qian, Haoyi Yu, Heng Zhang, Hongliang Zhang, Hongzhi Luan, Huirong Dong, Huizhong Li, Jia Li, Jia Liu, Jialong Zhu, Jian Sha, Jianping Wei, Jiaolong Yang, Jieyue Ma, Jiewei Wu, Jinjing Huang, Jingyun Tian, Jingyuan Zhang, Jinquan Sun, Juanhui Tu, Jun Liu, Jun Xu, Jun Zhou, Junjie Ou, Junpeng Fang, Kaihong Zhang, Kaiqin Hu, Ke Shi, Kun Tang, Kunlong Chen, Lanyin Mei, Lei Liang, Lei Xu, Libo Zhang, Lin Ju, Lin Yuan, Ling Zhong, Lintao Ma, Lu Liu, Lu Yu, Lun Cai, Meiqi Zhu, Mengying Li, Min Chen, Minghao Xue, Minghong Cai, Mingming Yin, Peijie Jiang, Peilong Zhao, Pingping Liu, Qian Zhao, Qing Cui, Qingxiang Huang, Qingyuan Yang, Quankun Yu, Shaowei Wei, Shijie Lian, Shoujian Zheng, Shun Song, Shungen Zhang, Shuo Zhang, Siyuan Li, Song Liu, Ting Guo, Tong Zhao, Wanli Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22115)  

**Abstract**: We introduce Ling 2.0, a series reasoning-oriented language foundation built upon the principle that every activation boosts reasoning capability. Designed to scale from tens of billions to one trillion parameters under a unified Mixture-of-Experts (MoE) paradigm, Ling 2.0 emphasizes high sparsity, cross-scale consistency, and efficiency guided by empirical scaling laws. The series includes three non-thinking (instruct) models - Ling-mini-2.0, Ling-flash-2.0, and Ling-1T - ranging from 16B to 1T total parameters and achieving up to 7-fold active-compute efficiency compared with dense counterparts. Ling 2.0 integrates coordinated innovations across model architecture, pre-training, post-training, and infrastructure: a high-sparsity MoE with MTP for efficient reasoning, reasoning-oriented data and mid-training CoT activation, reinforcement-based fine-tuning (DFT, Evo-CoT), and full-scale FP8 training with fine-grained heterogeneous pipelines. At the trillion scale, Ling-1T establishes a new Pareto frontier of reasoning accuracy versus computational efficiency, demonstrating that sparse activation, when properly aligned with reasoning objectives, enables scalable and efficient intelligence. Collectively, Ling 2.0 provides a coherent, open, and efficient foundation for advancing future reasoning and thinking models, including the Ring series built upon the same base. 

---
# Gradual Forgetting: Logarithmic Compression for Extending Transformer Context Windows 

**Authors**: Billy Dickson, Zoran Tiganj  

**Link**: [PDF](https://arxiv.org/pdf/2510.22109)  

**Abstract**: Most approaches to long-context processing increase the complexity of the transformer's internal architecture by integrating mechanisms such as recurrence or auxiliary memory modules. In this work, we introduce an alternative approach that modifies the input representation itself, rather than the transformer architecture. Inspired by cognitive models of human memory, our method applies a scale-invariant logarithmic compression to the input tokens. The resulting compressed representation is processed by a standard, unmodified transformer, preserving architectural simplicity. We evaluate this approach on the WikiText-103 and PG-19 language modeling benchmarks, showing a reduction in perplexity compared to uncompressed baselines. Moreover, performance improves consistently with longer compressed temporal contexts, showing that input-level logarithmic compression is a simple and effective way to extend a transformer's long-range memory. 

---
# Generalization or Memorization: Dynamic Decoding for Mode Steering 

**Authors**: Xuanming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22099)  

**Abstract**: Large Language Models (LLMs) exhibit a troubling duality, capable of both remarkable generalization and brittle, verbatim memorization of their training data. This unpredictability undermines their reliability in high-stakes applications. In this work, we propose a unified framework to understand, identify, and control these distinct reasoning modes. First, we introduce a theoretical model based on the Information Bottleneck (IB) principle, formalizing generalization as the learning of a compressed, task-relevant representation and memorization as a failure to compress. Building on this theory, we develop Dynamic Mode Steering (DMS), a novel inference-time algorithm which comprises two components: (1) a lightweight, causally-grounded linear probe that identifies the model's instantaneous reliance on memorization, and (2) a dynamic activation steering mechanism that nudges the model's computation towards pre-identified generalization circuits. We frame DMS as a form of adaptive, self-contrastive decoding. Experiments on reasoning and faithfulness tasks demonstrate that DMS significantly improves logical consistency and factual accuracy, thereby offering a principled approach to enhancing LLM reliability. 

---
# Compositional Bias Control in Large Language Models: Preference Learning Fails, Supervision Succeeds 

**Authors**: Atij Mahesh  

**Link**: [PDF](https://arxiv.org/pdf/2510.22084)  

**Abstract**: Large Language Models (LLMs) still produce gender-stereotyped language even in occupation-neutral contexts that reflect deep societal biases (Rudinger et al., 2018). To address this, prior work has proposed prompting, constrained decoding (Dathathri et al., 2020; Zhou et al., 2024), post-processing, and fine-tuning-based alignment (Rafailov et al., 2023; Ravfogel et al., 2022). However, the comparative efficacy and learning dynamics remain little understood. We report a comparative analysis of six control techniques for bias mitigation: prompt-only, generate-and-filter, DFA-based Ctrl-G decoding, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Iterative Nullspace Projection (INLP). We evaluate each method on a compositional constraint task. This task requires generating sentences that contain at least one agentic and one communal descriptor for each of the twenty Winogender-derived occupations. We quantify trade-offs between control strength and naturalness with evaluations of constraint compliance, lexical diversity, and fluency. Our results reveal key contrasts among the methods: SFT achieves 99.87 +- 0.15% compliance and high lexical diversity, while DPO, despite similar training stability, fails at 4.53 +- 0.82%. Ctrl-G guarantees perfect compliance, but at the cost of severely reduced fluency and diversity. Preference-based learning fundamentally differs: it cannot satisfy compositional constraints, as binary preference signals encode ranking, not logical conjunctions. Only explicit positive supervision enables mitigation of compositional biases; preference-based alignment fails to generalize logical structures, underscoring the limitations of preference learning and the necessity of explicit supervision for fair and fluent controlled generation. 

---
# Emotions Where Art Thou: Understanding and Characterizing the Emotional Latent Space of Large Language Models 

**Authors**: Benjamin Reichman, Adar Avsian, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2510.22042)  

**Abstract**: This work investigates how large language models (LLMs) internally represent emotion by analyzing the geometry of their hidden-state space. The paper identifies a low-dimensional emotional manifold and shows that emotional representations are directionally encoded, distributed across layers, and aligned with interpretable dimensions. These structures are stable across depth and generalize to eight real-world emotion datasets spanning five languages. Cross-domain alignment yields low error and strong linear probe performance, indicating a universal emotional subspace. Within this space, internal emotion perception can be steered while preserving semantics using a learned intervention module, with especially strong control for basic emotions across languages. These findings reveal a consistent and manipulable affective geometry in LLMs and offer insight into how they internalize and process emotion. 

---
# ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality 

**Authors**: Shayne Longpre, Sneha Kudugunta, Niklas Muennighoff, I-Hung Hsu, Isaac Caswell, Alex Pentland, Sercan Arik, Chen-Yu Lee, Sayna Ebrahimi  

**Link**: [PDF](https://arxiv.org/pdf/2510.22037)  

**Abstract**: Scaling laws research has focused overwhelmingly on English -- yet the most prominent AI models explicitly serve billions of international users. In this work, we undertake the largest multilingual scaling laws study to date, totaling 774 multilingual training experiments, spanning 10M-8B model parameters, 400+ training languages and 48 evaluation languages. We introduce the Adaptive Transfer Scaling Law (ATLAS) for both monolingual and multilingual pretraining, which outperforms existing scaling laws' out-of-sample generalization often by more than 0.3 R^2. Our analyses of the experiments shed light on multilingual learning dynamics, transfer properties between languages, and the curse of multilinguality. First, we derive a cross-lingual transfer matrix, empirically measuring mutual benefit scores between 38 x 38=1444 language pairs. Second, we derive a language-agnostic scaling law that reveals how to optimally scale model size and data when adding languages without sacrificing performance. Third, we identify the computational crossover points for when to pretrain from scratch versus finetune from multilingual checkpoints. We hope these findings provide the scientific foundation for democratizing scaling laws across languages, and enable practitioners to efficiently scale models -- beyond English-first AI. 

---
# Penalizing Length: Uncovering Systematic Bias in Quality Estimation Metrics 

**Authors**: Yilin Zhang, Wenda Xu, Zhongtao Liu, Tetsuji Nakagawa, Markus Freitag  

**Link**: [PDF](https://arxiv.org/pdf/2510.22028)  

**Abstract**: Quality Estimation (QE) metrics are vital in machine translation for reference-free evaluation and as a reward signal in tasks like reinforcement learning. However, the prevalence and impact of length bias in QE have been underexplored. Through a systematic study of top-performing regression-based and LLM-as-a-Judge QE metrics across 10 diverse language pairs, we reveal two critical length biases: First, QE metrics consistently over-predict errors with increasing translation length, even for high-quality, error-free texts. Second, they exhibit a preference for shorter translations when multiple candidates are available for the same source text. These inherent length biases risk unfairly penalizing longer, correct translations and can lead to sub-optimal decision-making in applications such as QE reranking and QE guided reinforcement learning. To mitigate this, we propose two strategies: (a) applying length normalization during model training, and (b) incorporating reference texts during evaluation. Both approaches were found to effectively reduce the identified length bias. 

---
# Toward Understanding the Transferability of Adversarial Suffixes in Large Language Models 

**Authors**: Sarah Ball, Niki Hasrati, Alexander Robey, Avi Schwarzschild, Frauke Kreuter, Zico Kolter, Andrej Risteski  

**Link**: [PDF](https://arxiv.org/pdf/2510.22014)  

**Abstract**: Discrete optimization-based jailbreaking attacks on large language models aim to generate short, nonsensical suffixes that, when appended onto input prompts, elicit disallowed content. Notably, these suffixes are often transferable -- succeeding on prompts and models for which they were never optimized. And yet, despite the fact that transferability is surprising and empirically well-established, the field lacks a rigorous analysis of when and why transfer occurs. To fill this gap, we identify three statistical properties that strongly correlate with transfer success across numerous experimental settings: (1) how much a prompt without a suffix activates a model's internal refusal direction, (2) how strongly a suffix induces a push away from this direction, and (3) how large these shifts are in directions orthogonal to refusal. On the other hand, we find that prompt semantic similarity only weakly correlates with transfer success. These findings lead to a more fine-grained understanding of transferability, which we use in interventional experiments to showcase how our statistical analysis can translate into practical improvements in attack success. 

---
# Uncovering the Persuasive Fingerprint of LLMs in Jailbreaking Attacks 

**Authors**: Havva Alizadeh Noughabi, Julien Serbanescu, Fattane Zarrinkalam, Ali Dehghantanha  

**Link**: [PDF](https://arxiv.org/pdf/2510.21983)  

**Abstract**: Despite recent advances, Large Language Models remain vulnerable to jailbreak attacks that bypass alignment safeguards and elicit harmful outputs. While prior research has proposed various attack strategies differing in human readability and transferability, little attention has been paid to the linguistic and psychological mechanisms that may influence a model's susceptibility to such attacks. In this paper, we examine an interdisciplinary line of research that leverages foundational theories of persuasion from the social sciences to craft adversarial prompts capable of circumventing alignment constraints in LLMs. Drawing on well-established persuasive strategies, we hypothesize that LLMs, having been trained on large-scale human-generated text, may respond more compliantly to prompts with persuasive structures. Furthermore, we investigate whether LLMs themselves exhibit distinct persuasive fingerprints that emerge in their jailbreak responses. Empirical evaluations across multiple aligned LLMs reveal that persuasion-aware prompts significantly bypass safeguards, demonstrating their potential to induce jailbreak behaviors. This work underscores the importance of cross-disciplinary insight in addressing the evolving challenges of LLM safety. The code and data are available. 

---
# A Stylometric Application of Large Language Models 

**Authors**: Harrison F. Stropkay, Jiayi Chen, Mohammad J. Latifi, Daniel N. Rockmore, Jeremy R. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2510.21958)  

**Abstract**: We show that large language models (LLMs) can be used to distinguish the writings of different authors. Specifically, an individual GPT-2 model, trained from scratch on the works of one author, will predict held-out text from that author more accurately than held-out text from other authors. We suggest that, in this way, a model trained on one author's works embodies the unique writing style of that author. We first demonstrate our approach on books written by eight different (known) authors. We also use this approach to confirm R. P. Thompson's authorship of the well-studied 15th book of the Oz series, originally attributed to F. L. Baum. 

---
# Model-Aware Tokenizer Transfer 

**Authors**: Mykola Haltiuk, Aleksander Smywiński-Pohl  

**Link**: [PDF](https://arxiv.org/pdf/2510.21954)  

**Abstract**: Large Language Models (LLMs) are trained to support an increasing number of languages, yet their predefined tokenizers remain a bottleneck for adapting models to lower-resource or distinct-script languages. Existing tokenizer transfer methods typically rely on semantic heuristics to initialize new embeddings, ignoring higher-layer model dynamics and limiting transfer quality. We propose Model-Aware Tokenizer Transfer (MATT), a method that incorporates model internals into the tokenizer transfer process. MATT introduces an Attention Influence Modeling (AIM) objective that distills inter-token communication patterns from a source model into a target model with a new tokenizer, providing an efficient warm-up before standard language modeling. Unlike approaches that focus solely on embedding similarity, MATT leverages attention behavior to guide embedding initialization and adaptation. Experiments across diverse linguistic settings show that MATT recovers a large fraction of the original model's performance within a few GPU hours, outperforming heuristic baselines. These results demonstrate that incorporating model-level signals offers a practical and effective path toward robust tokenizer transfer in multilingual LLMs. 

---
# Explaining and Mitigating Crosslingual Tokenizer Inequities 

**Authors**: Catherine Arnett, Tyler A. Chang, Stella Biderman, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2510.21909)  

**Abstract**: The number of tokens it takes to encode parallel text in different languages is known to vary. These disparities are called token premiums. Having high token premiums leads to less throughput during training and increases costs at inference. In this paper, we show that even after controlling for dataset size, vocabulary size, and data content, monolingual tokenizers exhibit a wide range of token premiums across languages. To understand the cross-linguistic differences that cause these token premiums, we train a suite of approximately 7,000 comparable monolingual tokenizers for 97 languages, manipulating tokenization algorithm, vocabulary size, and dataset size. We measure token premiums and test for a relationship between factors such as data similarity (between tokenizer training and evaluation), vocabulary size, and pre-tokenization. We also investigate the role of language-specific features such as writing system and word length. We find that similarity between training and test data does not impact token premiums, but vocabulary size and pre-tokenization do. While simply increasing vocabulary size does not lead to reduced token premium effects, we can determine an ``optimal'' vocabulary size for each language to achieve significantly reduced token premium effects. We also train superword tokenizers which allow merges over whitespaces, and we find that they both reduce token premium effects and improve compression overall. Thus, intervening on the vocabulary size or the pre-tokenizer significantly reduces crosslingual token premium effects. 

---
# Deep Literature Survey Automation with an Iterative Workflow 

**Authors**: Hongbo Zhang, Han Cui, Yidong Wang, Yijian Tian, Qi Guo, Cunxiang Wang, Jian Wu, Chiyu Song, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21900)  

**Abstract**: Automatic literature survey generation has attracted increasing attention, yet most existing systems follow a one-shot paradigm, where a large set of papers is retrieved at once and a static outline is generated before drafting. This design often leads to noisy retrieval, fragmented structures, and context overload, ultimately limiting survey quality. Inspired by the iterative reading process of human researchers, we propose \ours, a framework based on recurrent outline generation, in which a planning agent incrementally retrieves, reads, and updates the outline to ensure both exploration and coherence. To provide faithful paper-level grounding, we design paper cards that distill each paper into its contributions, methods, and findings, and introduce a review-and-refine loop with visualization enhancement to improve textual flow and integrate multimodal elements such as figures and tables. Experiments on both established and emerging topics show that \ours\ substantially outperforms state-of-the-art baselines in content coverage, structural coherence, and citation quality, while producing more accessible and better-organized surveys. To provide a more reliable assessment of such improvements, we further introduce Survey-Arena, a pairwise benchmark that complements absolute scoring and more clearly positions machine-generated surveys relative to human-written ones. The code is available at this https URL\_Autosurveyv2. 

---
# Understanding Network Behaviors through Natural Language Question-Answering 

**Authors**: Mingzhe Xing, Chang Tian, Jianan Zhang, Lichen Pan, Peipei Liu, Zhaoteng Yan, Yinliang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2510.21894)  

**Abstract**: Modern large-scale networks introduce significant complexity in understanding network behaviors, increasing the risk of misconfiguration. Prior work proposed to understand network behaviors by mining network configurations, typically relying on domain-specific languages interfaced with formal models. While effective, they suffer from a steep learning curve and limited flexibility. In contrast, natural language (NL) offers a more accessible and interpretable interface, motivating recent research on NL-guided network behavior understanding. Recent advances in large language models (LLMs) further enhance this direction, leveraging their extensive prior knowledge of network concepts and strong reasoning capabilities. However, three key challenges remain: 1) numerous router devices with lengthy configuration files challenge LLM's long-context understanding ability; 2) heterogeneity across devices and protocols impedes scalability; and 3) complex network topologies and protocols demand advanced reasoning abilities beyond the current capabilities of LLMs. To tackle the above challenges, we propose NetMind, a novel framework for querying networks using NL. Our approach introduces a tree-based configuration chunking strategy to preserve semantic coherence while enabling efficient partitioning. We then construct a unified fact graph as an intermediate representation to normalize vendor-specific configurations. Finally, we design a hybrid imperative-declarative language to reduce the reasoning burden on LLMs and enhance precision. We contribute a benchmark consisting of NL question-answer pairs paired with network configurations. Experiments demonstrate that NetMind achieves accurate and scalable network behavior understanding, outperforming existing baselines. 

---
# Embedding Trust: Semantic Isotropy Predicts Nonfactuality in Long-Form Text Generation 

**Authors**: Dhrupad Bhardwaj, Julia Kempe, Tim G. J. Rudner  

**Link**: [PDF](https://arxiv.org/pdf/2510.21891)  

**Abstract**: To deploy large language models (LLMs) in high-stakes application domains that require substantively accurate responses to open-ended prompts, we need reliable, computationally inexpensive methods that assess the trustworthiness of long-form responses generated by LLMs. However, existing approaches often rely on claim-by-claim fact-checking, which is computationally expensive and brittle in long-form responses to open-ended prompts. In this work, we introduce semantic isotropy -- the degree of uniformity across normalized text embeddings on the unit sphere -- and use it to assess the trustworthiness of long-form responses generated by LLMs. To do so, we generate several long-form responses, embed them, and estimate the level of semantic isotropy of these responses as the angular dispersion of the embeddings on the unit sphere. We find that higher semantic isotropy -- that is, greater embedding dispersion -- reliably signals lower factual consistency across samples. Our approach requires no labeled data, no fine-tuning, and no hyperparameter selection, and can be used with open- or closed-weight embedding models. Across multiple domains, our method consistently outperforms existing approaches in predicting nonfactuality in long-form responses using only a handful of samples -- offering a practical, low-cost approach for integrating trust assessment into real-world LLM workflows. 

---
# Preventing Catastrophic Forgetting: Behavior-Aware Sampling for Safer Language Model Fine-Tuning 

**Authors**: Anh Pham, Mihir Thalanki, Michael Sun, Aditya Chaloo, Ankita Gupta, Tian Xia, Aditya Mate, Ehimwenma Nosakhare, Soundararajan Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2510.21885)  

**Abstract**: Large language models often lose previously aligned safety behaviors when fine-tuned on benign data, a phenomenon known as catastrophic forgetting. Prior work shows that adding random safety examples can mitigate this effect, but it remains unclear which examples are most effective. We propose a behavior-aware sampling framework that selects safety examples based on two complementary factors: instruction-response behavior (e.g., refusal versus compliance) and semantic diversity across harm categories. Systematic evaluation shows that this approach substantially reduces harmful outputs while maintaining helpfulness, achieving up to a 41% reduction in harmfulness with only 0.5% additional training data. These results highlight how targeted data selection can improve the safety and efficiency of fine-tuning at scale. 

---
# Framework for Machine Evaluation of Reasoning Completeness in Large Language Models For Classification Tasks 

**Authors**: Avinash Patil  

**Link**: [PDF](https://arxiv.org/pdf/2510.21884)  

**Abstract**: The growing adoption of machine learning (ML) in sensitive domains has heightened the demand for transparent and interpretable artificial intelligence. Large Language Models (LLMs) are increasingly capable of producing natural language explanations, yet it remains unclear whether these rationales faithfully capture the predictive signals that underlie decisions. This paper introduces RACE-Reasoning Alignment for Completeness of Explanations, a systematic framework to evaluate the alignment between LLM-generated explanations and interpretable feature importance scores derived from a logistic regression baseline. We analyze four widely used text classification datasets-WIKI ONTOLOGY, AG NEWS, IMDB, and GOEMOTIONS-and compare LLM rationales against top-ranked supporting and contradicting lexical features. To capture alignment at multiple levels of granularity, RACE implements token-aware, exact string, and edit-distance matching techniques. Empirical results reveal a consistent asymmetry: correct predictions exhibit higher coverage of supporting features, while incorrect predictions are associated with elevated coverage of contradicting features. Edit-distance matching further uncovers paraphrastic overlaps, boosting coverage while preserving this asymmetry. These findings demonstrate that LLM rationales combine both surface-level and flexible evidence reuse, yet can also amplify misleading cues in error cases. RACE provides new insights into the faithfulness of LLM explanations and establishes a quantitative basis for evaluating reasoning completeness in neural language models. 

---
# Language Ranker: A Lightweight Ranking framework for LLM Decoding 

**Authors**: Chenheng Zhang, Tianqi Du, Jizhe Zhang, Mingqing Xiao, Yifei Wang, Yisen Wang, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.21883)  

**Abstract**: Conventional research on large language models (LLMs) has primarily focused on refining output distributions, while paying less attention to the decoding process that transforms these distributions into final responses. Recent advances, such as scaling the computation of inference time with reward models, have underscored the importance of decoding, but these methods often suffer from high computational costs and limited applicability. In this paper, we revisit LLM generation through the lens of recommender systems, conceptualizing the decoding process as analogous to the ranking stage in recommendation pipelines. From this perspective, we observe that both traditional decoding methods and reward models exhibit clear limitations such as redundancy. Motivated by this insight, we propose Language Ranker, a novel framework that introduces a lightweight module to rerank candidate responses using features extracted by the base model. Experiments across a wide range of tasks show that Language Ranker achieves performance comparable to large-scale reward models, while requiring only <0.5M additional parameters, significantly reducing the computational overhead during both training and inference stages. This highlights the efficiency and effectiveness of our method, showcasing its potential to fully unlock the capabilities of LLMs. 

---
# Policy Optimization Prefers The Path of Least Resistance 

**Authors**: Debdeep Sanyal, Aakash Sen Sharma, Dhruv Kumar, Saurabh Deshpande, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2510.21853)  

**Abstract**: Policy optimization (PO) algorithms are used to refine Large Language Models for complex, multi-step reasoning. Current state-of-the-art pipelines enforce a strict think-then-answer format to elicit chain-of-thought (CoT); however, the behavior of PO when these rigid constraints are relaxed into an open-ended CoT structure remains an under-studied question. We investigate this gap with an extensive suite of controlled experiments and identify a consistent principle: \textit{policy optimization consistently follows the path of least resistance}. When afforded the flexibility to interleave reasoning and response, policy optimization consistently learns to discard explicit reasoning, causing the policy to degenerate to a direct \texttt{<answer>}-only format. This outcome holds true across various models and algorithms. We find that this collapse in format is persistent even when the complex \texttt{<think><answer>} format is assigned up to 4x larger reward weights. We formalize this principle through a series of controlled reward decomposition experiments, demonstrating a clear hierarchy: PO systematically optimizes for the simplest reward component first, a preference that holds even when faced with mutually exclusive choices or strong incentives for more complex behaviors. Finally, we show that successful convergence on the high-reward shortcut is not a low-effort drift but is driven by the optimization process that requires the KL-regularized policy to have sufficient freedom to make a significant shift from its initial prior. Our findings reveal that granting policies the freedom to diverge is a double-edged sword: while necessary for discovering high-reward shortcuts, it also creates a powerful incentive to game the simplest aspects of the reward function, posing a critical challenge for reward hacking under alignment. 

---
# A Multi-lingual Dataset of Classified Paragraphs from Open Access Scientific Publications 

**Authors**: Eric Jeangirard  

**Link**: [PDF](https://arxiv.org/pdf/2510.21762)  

**Abstract**: We present a dataset of 833k paragraphs extracted from CC-BY licensed scientific publications, classified into four categories: acknowledgments, data mentions, software/code mentions, and clinical trial mentions. The paragraphs are primarily in English and French, with additional European languages represented. Each paragraph is annotated with language identification (using fastText) and scientific domain (from OpenAlex). This dataset, derived from the French Open Science Monitor corpus and processed using GROBID, enables training of text classification models and development of named entity recognition systems for scientific literature mining. The dataset is publicly available on HuggingFace this https URL under a CC-BY license. 

---
# Variational Masked Diffusion Models 

**Authors**: Yichi Zhang, Alex Schwing, Zhizhen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.23606)  

**Abstract**: Masked diffusion models have recently emerged as a flexible framework for discrete generative modeling. However, a key limitation of standard masked diffusion is its inability to effectively capture dependencies among tokens that are predicted concurrently, leading to degraded generation quality when dependencies among tokens are important. To explicitly model dependencies among tokens, we propose Variational Masked Diffusion (VMD), a framework that introduces latent variables into the masked diffusion process. Through controlled experiments on synthetic datasets, we demonstrate that VMD successfully learns dependencies that conventional masked diffusion fails to capture. We further validate the effectiveness of our approach on Sudoku puzzles and text datasets, where learning of dependencies among tokens improves global consistency. Across these domains, VMD enhances both generation quality and dependency awareness, highlighting the value of integrating variational inference into masked diffusion. Our code is available at: this https URL. 

---
# ReCode: Unify Plan and Action for Universal Granularity Control 

**Authors**: Zhaoyang Yu, Jiayi Zhang, Huixue Su, Yufan Zhao, Yifan Wu, Mingyi Deng, Jinyu Xiang, Yizhang Lin, Lingxiao Tang, Yingchao Li, Yuyu Luo, Bang Liu, Chenglin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23564)  

**Abstract**: Real-world tasks require decisions at varying granularities, and humans excel at this by leveraging a unified cognitive representation where planning is fundamentally understood as a high-level form of action. However, current Large Language Model (LLM)-based agents lack this crucial capability to operate fluidly across decision granularities. This limitation stems from existing paradigms that enforce a rigid separation between high-level planning and low-level action, which impairs dynamic adaptability and limits generalization. We propose ReCode (Recursive Code Generation), a novel paradigm that addresses this limitation by unifying planning and action within a single code representation. In this representation, ReCode treats high-level plans as abstract placeholder functions, which the agent then recursively decomposes into finer-grained sub-functions until reaching primitive actions. This recursive approach dissolves the rigid boundary between plan and action, enabling the agent to dynamically control its decision granularity. Furthermore, the recursive structure inherently generates rich, multi-granularity training data, enabling models to learn hierarchical decision-making processes. Extensive experiments show ReCode significantly surpasses advanced baselines in inference performance and demonstrates exceptional data efficiency in training, validating our core insight that unifying planning and action through recursive code generation is a powerful and effective approach to achieving universal granularity control. The code is available at this https URL. 

---
# ISA-Bench: Benchmarking Instruction Sensitivity for Large Audio Language Models 

**Authors**: Bohan Li, Wenbin Huang, Yuhang Qiu, Yiwei Guo, Hankun Wang, Zhihan Li, Jing Peng, Ziyang Ma, Xie Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23558)  

**Abstract**: Large Audio Language Models (LALMs), which couple acoustic perception with large language models (LLMs) to extract and understand diverse information from audio, have attracted intense interest from both academic and industrial communities. However, existing LALMs are highly sensitive to how instructions are phrased, affecting both (i) instruction-following rates and (ii) task performance. Yet, no existing benchmarks offer a systematic and comprehensive evaluation of this sensitivity. We introduce ISA-Bench, a dynamic benchmark evaluating instruction sensitivity for LALMs along three axes: instruction description, output format, and task composition. We assess recent open-source and proprietary LALMs using ISA-Bench, profiling both compliance and accuracy under controlled instruction variations. Experimental results reveal that even state-of-the-art LALMs suffer significant instruction sensitivity, leading to degraded performance on fundamental audio understanding tasks. To mitigate this issue, we fine-tune Qwen2-Audio on a specifically constructed complex instruction-variant dataset, achieving a marked improvement in instruction-following performance. However, this also induces nontrivial catastrophic forgetting: the model loses some previously mastered task capabilities when exposed to new instruction styles. Our benchmark provides a standardized basis for assessing and improving instruction sensitivity in LALMs, underscoring the need for instruction-robust audio understanding in real-world pipelines. 

---
# A U-Net and Transformer Pipeline for Multilingual Image Translation 

**Authors**: Siddharth Sahay, Radhika Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2510.23554)  

**Abstract**: This paper presents an end-to-end multilingual translation pipeline that integrates a custom U-Net for text detection, the Tesseract engine for text recognition, and a from-scratch sequence-to-sequence (Seq2Seq) Transformer for Neural Machine Translation (NMT). Our approach first utilizes a U-Net model, trained on a synthetic dataset , to accurately segment and detect text regions from an image. These detected regions are then processed by Tesseract to extract the source text. This extracted text is fed into a custom Transformer model trained from scratch on a multilingual parallel corpus spanning 5 languages. Unlike systems reliant on monolithic pre-trained models, our architecture emphasizes full customization and adaptability. The system is evaluated on its text detection accuracy, text recognition quality, and translation performance via BLEU scores. The complete pipeline demonstrates promising results, validating the viability of a custom-built system for translating text directly from images. 

---
# JanusCoder: Towards a Foundational Visual-Programmatic Interface for Code Intelligence 

**Authors**: Qiushi Sun, Jingyang Gong, Yang Liu, Qiaosheng Chen, Lei Li, Kai Chen, Qipeng Guo, Ben Kao, Fei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.23538)  

**Abstract**: The scope of neural code intelligence is rapidly expanding beyond text-based source code to encompass the rich visual outputs that programs generate. This visual dimension is critical for advanced applications like flexible content generation and precise, program-driven editing of visualizations. However, progress has been impeded by the scarcity of high-quality multimodal code data, a bottleneck stemming from challenges in synthesis and quality assessment. To address these challenges, we make contributions from both a data and modeling perspective. We first introduce a complete synthesis toolkit that leverages reciprocal synergies between data modalities to efficiently produce a large-scale, high-quality corpus spanning from standard charts to complex interactive web UIs and code-driven animations. Leveraging this toolkit, we construct JanusCode-800K, the largest multimodal code corpus to date. This powers the training of our models, JanusCoder and JanusCoderV, which establish a visual-programmatic interface for generating code from textual instructions, visual inputs, or a combination of both. Our unified model is a departure from existing approaches that build specialized models for isolated tasks. Extensive experiments on both text-centric and vision-centric coding tasks demonstrate the superior performance of the JanusCoder series, with our 7B to 14B scale models approaching or even exceeding the performance of commercial models. Furthermore, extensive analysis provides key insights into harmonizing programmatic logic with its visual expression. Our code and checkpoints will are available at this https URL. 

---
# A Neuro-Symbolic Multi-Agent Approach to Legal-Cybersecurity Knowledge Integration 

**Authors**: Chiara Bonfanti, Alessandro Druetto, Cataldo Basile, Tharindu Ranasinghe, Marcos Zampieri  

**Link**: [PDF](https://arxiv.org/pdf/2510.23443)  

**Abstract**: The growing intersection of cybersecurity and law creates a complex information space where traditional legal research tools struggle to deal with nuanced connections between cases, statutes, and technical vulnerabilities. This knowledge divide hinders collaboration between legal experts and cybersecurity professionals. To address this important gap, this work provides a first step towards intelligent systems capable of navigating the increasingly intricate cyber-legal domain. We demonstrate promising initial results on multilingual tasks. 

---
# Planning Ahead with RSA: Efficient Signalling in Dynamic Environments by Projecting User Awareness across Future Timesteps 

**Authors**: Anwesha Das, John Duff, Jörg Hoffmann, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2510.23340)  

**Abstract**: Adaptive agent design offers a way to improve human-AI collaboration on time-sensitive tasks in rapidly changing environments. In such cases, to ensure the human maintains an accurate understanding of critical task elements, an assistive agent must not only identify the highest priority information but also estimate how and when this information can be communicated most effectively, given that human attention represents a zero-sum cognitive resource where focus on one message diminishes awareness of other or upcoming information. We introduce a theoretical framework for adaptive signalling which meets these challenges by using principles of rational communication, formalised as Bayesian reference resolution using the Rational Speech Act (RSA) modelling framework, to plan a sequence of messages which optimise timely alignment between user belief and a dynamic environment. The agent adapts message specificity and timing to the particulars of a user and scenario based on projections of how prior-guided interpretation of messages will influence attention to the interface and subsequent belief update, across several timesteps out to a fixed horizon. In a comparison to baseline methods, we show that this effectiveness depends crucially on combining multi-step planning with a realistic model of user awareness. As the first application of RSA for communication in a dynamic environment, and for human-AI interaction in general, we establish theoretical foundations for pragmatic communication in human-agent teams, highlighting how insights from cognitive science can be capitalised to inform the design of assistive agents. 

---
# LibriConvo: Simulating Conversations from Read Literature for ASR and Diarization 

**Authors**: Máté Gedeon, Péter Mihajlik  

**Link**: [PDF](https://arxiv.org/pdf/2510.23320)  

**Abstract**: We introduce LibriConvo, a simulated multi-speaker conversational dataset based on speaker-aware conversation simulation (SASC), designed to support training and evaluation of speaker diarization and automatic speech recognition (ASR) systems. Unlike prior resources that mostly rely on semantically disconnected utterances and implausible temporal gaps, LibriConvo ensures semantic coherence and realistic conversational timing. Our pipeline leverages CallHome with external VAD for reliable boundaries, applies compression to reduce unnaturally long silences, and organizes LibriTTS utterances by book to maintain contextual consistency. Acoustic realism is enhanced via a novel room impulse response selection procedure that ranks speaker-microphone configurations by spatial plausibility, balancing realism and diversity. The dataset comprises 240.1 hours across 1,496 dialogues with 830 unique speakers, split in a speaker-disjoint manner for robust evaluation. Baselines show that the sortformer model outperforms the pyannote pipeline in diarization, while a fine-tuned Fast Conformer-CTC XLarge with Serialized Output Training achieves 7.29\% WER for ASR, surpassing zero-shot Whisper-large-v3. LibriConvo provides a valuable resource for advancing multi-speaker speech processing research with realistic conversational dynamics and controlled experimental conditions. 

---
# PTPP-Aware Adaptation Scaling Laws: Predicting Domain-Adaptation Performance at Unseen Pre-Training Budgets 

**Authors**: Etienne Goffinet, Shane Bergsma, Avraham Sheinin, Natalia Vassilieva, Shaheer Muhammad, Preslav Nakov, Gurpreet Gosal  

**Link**: [PDF](https://arxiv.org/pdf/2510.23198)  

**Abstract**: Continual pre-training (CPT) for domain adaptation must balance target-domain gains with stability on the base domain. Existing CPT scaling laws typically assume a fixed pre-training budget, which limits their ability to forecast adaptation outcomes for models trained at different tokens-per-parameter (PTPP). We present \emph{PTPP-aware} adaptation scaling laws that make the pre-training budget an explicit variable, enabling accurate \emph{prediction} of adaptation loss at unseen \ptpp. On a multilingual setup (English/Arabic $\rightarrow$ French), PTPP-aware formulations trained on early stages (\ptpp{}=\{15,31\}) predict target loss at \ptpp{}=279 and outperform a PTPP-agnostic \dcpt{} transfer baseline on metrics (Huber-on-log, MAE$_\mathrm{rel}$, calibration slope); full diagnostics (RMSE, MAPE) are in the appendix. Beyond forecasting, we show a practical use case: planning replay ratios and adaptation token budgets that satisfy target and forgetting constraints under compute limits. 

---
# Rethinking GSPO: The Perplexity-Entropy Equivalence 

**Authors**: Chi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23142)  

**Abstract**: We provide a new perspective on GSPO's length-normalized importance ratios by establishing their connection to information-theoretic quantities. We show that GSPO's sequence-level weight $s(\theta) = (\pi_\theta/\pi_{\theta_{\text{old}}})^{1/|y|}$ can be equivalently expressed as the inverse perplexity ratio $\text{PPL}_{\theta_{\text{old}}}/\text{PPL}_\theta$ and as the exponential cross-entropy change $\exp(\Delta H)$. While the perplexity-entropy relationship follows from standard definitions, this observation provides a useful lens for understanding GSPO: the algorithm weights policy gradient updates by perplexity ratios, offering an information-theoretic interpretation of the importance weights. This perspective helps explain GSPO's empirical properties, including log-domain variance reduction through geometric averaging and stability in training mixture-of-experts models. We validate the mathematical equivalences and variance predictions through controlled experiments on mathematical reasoning tasks. 

---
# Fast-MIA: Efficient and Scalable Membership Inference for LLMs 

**Authors**: Hiromu Takahashi, Shotaro Ishihara  

**Link**: [PDF](https://arxiv.org/pdf/2510.23074)  

**Abstract**: We propose Fast-MIA (this https URL), a Python library for efficiently evaluating membership inference attacks (MIA) against Large Language Models (LLMs). MIA against LLMs has emerged as a crucial challenge due to growing concerns over copyright, security, and data privacy, and has attracted increasing research attention. However, the progress of this research is significantly hindered by two main obstacles: (1) the high computational cost of inference in LLMs, and (2) the lack of standardized and maintained implementations of MIA methods, which makes large-scale empirical comparison difficult. To address these challenges, our library provides fast batch inference and includes implementations of representative MIA methods under a unified evaluation framework. This library supports easy implementation of reproducible benchmarks with simple configuration and extensibility. We release Fast-MIA as an open-source (Apache License 2.0) tool to support scalable and transparent research on LLMs. 

---
# Towards Stable and Effective Reinforcement Learning for Mixture-of-Experts 

**Authors**: Di Zhang, Xun Wu, Shaohan Huang, Yaru Hao, Li Dong, Zewen Chi, Zhifang Sui, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.23027)  

**Abstract**: Recent advances in reinforcement learning (RL) have substantially improved the training of large-scale language models, leading to significant gains in generation quality and reasoning ability. However, most existing research focuses on dense models, while RL training for Mixture-of-Experts (MoE) architectures remains underexplored. To address the instability commonly observed in MoE training, we propose a novel router-aware approach to optimize importance sampling (IS) weights in off-policy RL. Specifically, we design a rescaling strategy guided by router logits, which effectively reduces gradient variance and mitigates training divergence. Experimental results demonstrate that our method significantly improves both the convergence stability and the final performance of MoE models, highlighting the potential of RL algorithmic innovations tailored to MoE architectures and providing a promising direction for efficient training of large-scale expert models. 

---
# UniAIDet: A Unified and Universal Benchmark for AI-Generated Image Content Detection and Localization 

**Authors**: Huixuan Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2510.23023)  

**Abstract**: With the rapid proliferation of image generative models, the authenticity of digital images has become a significant concern. While existing studies have proposed various methods for detecting AI-generated content, current benchmarks are limited in their coverage of diverse generative models and image categories, often overlooking end-to-end image editing and artistic images. To address these limitations, we introduce UniAIDet, a unified and comprehensive benchmark that includes both photographic and artistic images. UniAIDet covers a wide range of generative models, including text-to-image, image-to-image, image inpainting, image editing, and deepfake models. Using UniAIDet, we conduct a comprehensive evaluation of various detection methods and answer three key research questions regarding generalization capability and the relation between detection and localization. Our benchmark and analysis provide a robust foundation for future research. 

---
# M$^{3}$T2IBench: A Large-Scale Multi-Category, Multi-Instance, Multi-Relation Text-to-Image Benchmark 

**Authors**: Huixuan Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2510.23020)  

**Abstract**: Text-to-image models are known to struggle with generating images that perfectly align with textual prompts. Several previous studies have focused on evaluating image-text alignment in text-to-image generation. However, these evaluations either address overly simple scenarios, especially overlooking the difficulty of prompts with multiple different instances belonging to the same category, or they introduce metrics that do not correlate well with human evaluation. In this study, we introduce M$^3$T2IBench, a large-scale, multi-category, multi-instance, multi-relation along with an object-detection-based evaluation metric, $AlignScore$, which aligns closely with human evaluation. Our findings reveal that current open-source text-to-image models perform poorly on this challenging benchmark. Additionally, we propose the Revise-Then-Enforce approach to enhance image-text alignment. This training-free post-editing method demonstrates improvements in image-text alignment across a broad range of diffusion models. \footnote{Our code and data has been released in supplementary material and will be made publicly available after the paper is accepted.} 

---
# Can Language Models Compose Skills In-Context? 

**Authors**: Zidong Liu, Zhuoyan Xu, Zhenmei Shi, Yingyu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22993)  

**Abstract**: Composing basic skills from simple tasks to accomplish composite tasks is crucial for modern intelligent systems. We investigate the in-context composition ability of language models to perform composite tasks that combine basic skills demonstrated in in-context examples. This is more challenging than the standard setting, where skills and their composition can be learned in training. We conduct systematic experiments on various representative open-source language models, utilizing linguistic and logical tasks designed to probe composition abilities. The results reveal that simple task examples can have a surprising negative impact on the performance, because the models generally struggle to recognize and assemble the skills correctly, even with Chain-of-Thought examples. Theoretical analysis further shows that it is crucial to align examples with the corresponding steps in the composition. This inspires a method for the probing tasks, whose improved performance provides positive support for our insights. 

---
# Modeling Political Discourse with Sentence-BERT and BERTopic 

**Authors**: Margarida Mendonca, Alvaro Figueira  

**Link**: [PDF](https://arxiv.org/pdf/2510.22904)  

**Abstract**: Social media has reshaped political discourse, offering politicians a platform for direct engagement while reinforcing polarization and ideological divides. This study introduces a novel topic evolution framework that integrates BERTopic-based topic modeling with Moral Foundations Theory (MFT) to analyze the longevity and moral dimensions of political topics in Twitter activity during the 117th U.S. Congress. We propose a methodology for tracking dynamic topic shifts over time and measuring their association with moral values and quantifying topic persistence. Our findings reveal that while overarching themes remain stable, granular topics tend to dissolve rapidly, limiting their long-term influence. Moreover, moral foundations play a critical role in topic longevity, with Care and Loyalty dominating durable topics, while partisan differences manifest in distinct moral framing strategies. This work contributes to the field of social network analysis and computational political discourse by offering a scalable, interpretable approach to understanding moral-driven topic evolution on social media. 

---
# Offline Preference Optimization via Maximum Marginal Likelihood Estimation 

**Authors**: Saeed Najafi, Alona Fyshe  

**Link**: [PDF](https://arxiv.org/pdf/2510.22881)  

**Abstract**: Aligning Large Language Models (LLMs) with human preferences is crucial, but standard methods like Reinforcement Learning from Human Feedback (RLHF) are often complex and unstable. In this work, we propose a new, simpler approach that recasts alignment through the lens of Maximum Marginal Likelihood (MML) estimation. Our new MML based Preference Optimization (MMPO) maximizes the marginal log-likelihood of a preferred text output, using the preference pair as samples for approximation, and forgoes the need for both an explicit reward model and entropy maximization. We theoretically demonstrate that MMPO implicitly performs preference optimization, producing a weighted gradient that naturally up-weights chosen responses over rejected ones. Across models ranging from 135M to 8B parameters, we empirically show that MMPO: 1) is more stable with respect to the hyperparameter $\beta$ compared to alternative baselines, and 2) achieves competitive or superior preference alignment while better preserving the base model's general language capabilities. Through a series of ablation experiments, we show that this improved performance is indeed attributable to MMPO's implicit preference optimization within the gradient updates. 

---
# How Do AI Agents Do Human Work? Comparing AI and Human Workflows Across Diverse Occupations 

**Authors**: Zora Zhiruo Wang, Yijia Shao, Omar Shaikh, Daniel Fried, Graham Neubig, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22780)  

**Abstract**: AI agents are continually optimized for tasks related to human work, such as software engineering and professional writing, signaling a pressing trend with significant impacts on the human workforce. However, these agent developments have often not been grounded in a clear understanding of how humans execute work, to reveal what expertise agents possess and the roles they can play in diverse workflows. In this work, we study how agents do human work by presenting the first direct comparison of human and agent workers across multiple essential work-related skills: data analysis, engineering, computation, writing, and design. To better understand and compare heterogeneous computer-use activities of workers, we introduce a scalable toolkit to induce interpretable, structured workflows from either human or agent computer-use activities. Using such induced workflows, we compare how humans and agents perform the same tasks and find that: (1) While agents exhibit promise in their alignment to human workflows, they take an overwhelmingly programmatic approach across all work domains, even for open-ended, visually dependent tasks like design, creating a contrast with the UI-centric methods typically used by humans. (2) Agents produce work of inferior quality, yet often mask their deficiencies via data fabrication and misuse of advanced tools. (3) Nonetheless, agents deliver results 88.3% faster and cost 90.4-96.2% less than humans, highlighting the potential for enabling efficient collaboration by delegating easily programmable tasks to agents. 

---
# TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination 

**Authors**: Omar Naim, Krish Sharma, Nicholas Asher  

**Link**: [PDF](https://arxiv.org/pdf/2510.22767)  

**Abstract**: In this paper we introduce Tale, Task-Aware Layer Elimination, an inference-time algorithm that prunes entire transformer layers in an LLM by directly optimizing task-specific validation performance. We evaluate TALE on 9 tasks and 5 models, including LLaMA 3.1 8B, Qwen 2.5 7B, Qwen 2.5 0.5B, Mistral 7B, and Lucie 7B, under both zero-shot and few-shot settings. Unlike prior approaches, TALE requires no retraining and consistently improves accuracy while reducing computational cost across all benchmarks. Furthermore, applying TALE during finetuning leads to additional performance gains. Finally, TALE provides flexible user control over trade-offs between accuracy and efficiency. Mutual information analysis shows that certain layers act as bottlenecks, degrading task-relevant representations. Tale's selective layer removal remedies this problem, producing smaller, faster, and more accurate models that are also faster to fine-tune while offering new insights into transformer interpretability. 

---
# Multi-Modal Fact-Verification Framework for Reducing Hallucinations in Large Language Models 

**Authors**: Piyushkumar Patel  

**Link**: [PDF](https://arxiv.org/pdf/2510.22751)  

**Abstract**: While Large Language Models have transformed how we interact with AI systems, they suffer from a critical flaw: they confidently generate false information that sounds entirely plausible. This hallucination problem has become a major barrier to deploying these models in real-world applications where accuracy matters. We developed a fact verification framework that catches and corrects these errors in real-time by cross checking LLM outputs against multiple knowledge sources. Our system combines structured databases, live web searches, and academic literature to verify factual claims as they're generated. When we detect inconsistencies, we automatically correct them while preserving the natural flow of the response. Testing across various domains showed we could reduce hallucinations by 67% without sacrificing response quality. Domain experts in healthcare, finance, and scientific research rated our corrected outputs 89% satisfactory a significant improvement over unverified LLM responses. This work offers a practical solution for making LLMs more trustworthy in applications where getting facts wrong isn't an option. 

---
# REVISION:Reflective Intent Mining and Online Reasoning Auxiliary for E-commerce Visual Search System Optimization 

**Authors**: Yiwen Tang, Qiuyu Zhao, Zenghui Sun, Jinsong Lan, Xiaoyong Zhu, Bo Zheng, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22739)  

**Abstract**: In Taobao e-commerce visual search, user behavior analysis reveals a large proportion of no-click requests, suggesting diverse and implicit user intents. These intents are expressed in various forms and are difficult to mine and discover, thereby leading to the limited adaptability and lag in platform strategies. This greatly restricts users' ability to express diverse intents and hinders the scalability of the visual search system. This mismatch between user implicit intent expression and system response defines the User-SearchSys Intent Discrepancy. To alleviate the issue, we propose a novel framework REVISION. This framework integrates offline reasoning mining with online decision-making and execution, enabling adaptive strategies to solve implicit user demands. In the offline stage, we construct a periodic pipeline to mine discrepancies from historical no-click requests. Leveraging large models, we analyze implicit intent factors and infer optimal suggestions by jointly reasoning over query and product metadata. These inferred suggestions serve as actionable insights for refining platform strategies. In the online stage, REVISION-R1-3B, trained on the curated offline data, performs holistic analysis over query images and associated historical products to generate optimization plans and adaptively schedule strategies across the search pipeline. Our framework offers a streamlined paradigm for integrating large models with traditional search systems, enabling end-to-end intelligent optimization across information aggregation and user interaction. Experimental results demonstrate that our approach improves the efficiency of implicit intent mining from large-scale search logs and significantly reduces the no-click rate. 

---
# ATLAS: Actor-Critic Task-Completion with Look-ahead Action Simulation 

**Authors**: Jiali Cheng, Anjishnu Kumar, Roshan Lal, Rishi Rajasekaran, Hani Ramezani, Omar Zia Khan, Oleg Rokhlenko, Sunny Chiu-Webster, Gang Hua, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.22732)  

**Abstract**: We observe that current state-of-the-art web-agents are unable to effectively adapt to new environments without neural network fine-tuning, without which they produce inefficient execution plans due to a lack of awareness of the structure and dynamics of the new environment. To address this limitation, we introduce ATLAS (Actor-Critic Task-completion with Look-ahead Action Simulation), a memory-augmented agent that is able to make plans grounded in a model of the environment by simulating the consequences of those actions in cognitive space. Our agent starts by building a "cognitive map" by performing a lightweight curiosity driven exploration of the environment. The planner proposes candidate actions; the simulator predicts their consequences in cognitive space; a critic analyzes the options to select the best roll-out and update the original plan; and a browser executor performs the chosen action. On the WebArena-Lite Benchmark, we achieve a 63% success rate compared to 53.9% success rate for the previously published state-of-the-art. Unlike previous systems, our modular architecture requires no website-specific LLM fine-tuning. Ablations show sizable drops without the world-model, hierarchical planner, and look-ahead-based replanner confirming their complementary roles within the design of our system 

---
# Critical Insights into Leading Conversational AI Models 

**Authors**: Urja Kohli, Aditi Singh, Arun Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2510.22729)  

**Abstract**: Big Language Models (LLMs) are changing the way businesses use software, the way people live their lives and the way industries work. Companies like Google, High-Flyer, Anthropic, OpenAI and Meta are making better LLMs. So, it's crucial to look at how each model is different in terms of performance, moral behaviour and usability, as these differences are based on the different ideas that built them. This study compares five top LLMs: Google's Gemini, High-Flyer's DeepSeek, Anthropic's Claude, OpenAI's GPT models and Meta's LLaMA. It performs this by analysing three important factors: Performance and Accuracy, Ethics and Bias Mitigation and Usability and Integration. It was found that Claude has good moral reasoning, Gemini is better at multimodal capabilities and has strong ethical frameworks. DeepSeek is great at reasoning based on facts, LLaMA is good for open applications and ChatGPT delivers balanced performance with a focus on usage. It was concluded that these models are different in terms of how well they work, how easy they are to use and how they treat people ethically, making it a point that each model should be utilised by the user in a way that makes the most of its strengths. 

---
# Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation 

**Authors**: Shu Zhao, Tianyi Shen, Nilesh Ahuja, Omesh Tickoo, Vijaykrishnan Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2510.22694)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) has emerged as a promising method to generate factual and up-to-date responses of Multimodal Large Language Models (MLLMs) by incorporating non-parametric knowledge from external knowledge bases. However, existing MRAG approaches suffer from static retrieval strategies, inflexible modality selection, and suboptimal utilization of retrieved information, leading to three critical challenges: determining when to retrieve, what modality to incorporate, and how to utilize retrieved information effectively. To address these challenges, we introduce Windsock, a query-dependent module making decisions on retrieval necessity and modality selection, effectively reducing computational overhead and improving response quality. Additionally, we propose Dynamic Noise-Resistance (DANCE) Instruction Tuning, an adaptive training strategy that enhances MLLMs' ability to utilize retrieved information while maintaining robustness against noise. Moreover, we adopt a self-assessment approach leveraging knowledge within MLLMs to convert question-answering datasets to MRAG training datasets. Extensive experiments demonstrate that our proposed method significantly improves the generation quality by 17.07% while reducing 8.95% retrieval times. 

---
# RoboSVG: A Unified Framework for Interactive SVG Generation with Multi-modal Guidance 

**Authors**: Jiuniu Wang, Gongjie Zhang, Quanhao Qian, Junlong Gao, Deli Zhao, Ran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22684)  

**Abstract**: Scalable Vector Graphics (SVGs) are fundamental to digital design and robot control, encoding not only visual structure but also motion paths in interactive drawings. In this work, we introduce RoboSVG, a unified multimodal framework for generating interactive SVGs guided by textual, visual, and numerical signals. Given an input query, the RoboSVG model first produces multimodal guidance, then synthesizes candidate SVGs through dedicated generation modules, and finally refines them under numerical guidance to yield high-quality outputs. To support this framework, we construct RoboDraw, a large-scale dataset of one million examples, each pairing an SVG generation condition (e.g., text, image, and partial SVG) with its corresponding ground-truth SVG code. RoboDraw dataset enables systematic study of four tasks, including basic generation (Text-to-SVG, Image-to-SVG) and interactive generation (PartialSVG-to-SVG, PartialImage-to-SVG). Extensive experiments demonstrate that RoboSVG achieves superior query compliance and visual fidelity across tasks, establishing a new state of the art in versatile SVG generation. The dataset and source code of this project will be publicly available soon. 

---
# Do Stop Me Now: Detecting Boilerplate Responses with a Single Iteration 

**Authors**: Yuval Kainan, Shaked Zychlinski  

**Link**: [PDF](https://arxiv.org/pdf/2510.22679)  

**Abstract**: Large Language Models (LLMs) often expend significant computational resources generating boilerplate responses, such as refusals, simple acknowledgements and casual greetings, which adds unnecessary cost and latency. To address this inefficiency, we propose a simple yet highly effective method for detecting such responses after only a single generation step. We demonstrate that the log-probability distribution of the first generated token serves as a powerful signal for classifying the nature of the entire subsequent response. Our experiments, conducted across a diverse range of small, large, and reasoning-specialized models, show that the first-token log-probability vectors form distinctly separable clusters for different response types. Using a lightweight k-NN classifier, we achieve high accuracy in predicting whether a response will be a substantive answer or a form of boilerplate response, including user-specified refusals. The primary implication is a practical, computationally trivial technique, optimizing LLM inference by enabling early termination or redirection to a smaller model, thereby yielding significant savings in computational cost. This work presents a direct path toward more efficient and sustainable LLM deployment. 

---
# Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views 

**Authors**: Anna Deichler, Jonas Beskow  

**Link**: [PDF](https://arxiv.org/pdf/2510.22672)  

**Abstract**: We introduce Look and Tell, a multimodal dataset for studying referential communication across egocentric and exocentric perspectives. Using Meta Project Aria smart glasses and stationary cameras, we recorded synchronized gaze, speech, and video as 25 participants instructed a partner to identify ingredients in a kitchen. Combined with 3D scene reconstructions, this setup provides a benchmark for evaluating how different spatial representations (2D vs. 3D; ego vs. exo) affect multimodal grounding. The dataset contains 3.67 hours of recordings, including 2,707 richly annotated referential expressions, and is designed to advance the development of embodied agents that can understand and engage in situated dialogue. 

---
# ATOM: AdapTive and OptiMized dynamic temporal knowledge graph construction using LLMs 

**Authors**: Yassir Lairgi, Ludovic Moncla, Khalid Benabdeslem, Rémy Cazabet, Pierre Cléau  

**Link**: [PDF](https://arxiv.org/pdf/2510.22590)  

**Abstract**: In today's rapidly expanding data landscape, knowledge extraction from unstructured text is vital for real-time analytics, temporal inference, and dynamic memory frameworks. However, traditional static knowledge graph (KG) construction often overlooks the dynamic and time-sensitive nature of real-world data, limiting adaptability to continuous changes. Moreover, recent zero- or few-shot approaches that avoid domain-specific fine-tuning or reliance on prebuilt ontologies often suffer from instability across multiple runs, as well as incomplete coverage of key facts. To address these challenges, we introduce ATOM (AdapTive and OptiMized), a few-shot and scalable approach that builds and continuously updates Temporal Knowledge Graphs (TKGs) from unstructured texts. ATOM splits input documents into minimal, self-contained "atomic" facts, improving extraction exhaustivity and stability. Then, it constructs atomic TKGs from these facts while employing a dual-time modeling that distinguishes when information is observed from when it is valid. The resulting atomic TKGs are subsequently merged in parallel. Empirical evaluations demonstrate that ATOM achieves ~18% higher exhaustivity, ~17% better stability, and over 90% latency reduction compared to baseline methods, demonstrating a strong scalability potential for dynamic TKG construction. 

---
# UltraVoice: Scaling Fine-Grained Style-Controlled Speech Conversations for Spoken Dialogue Models 

**Authors**: Wenming Tu, Guanrou Yang, Ruiqi Yan, Wenxi Chen, Ziyang Ma, Yipeng Kang, Kai Yu, Xie Chen, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.22588)  

**Abstract**: Spoken dialogue models currently lack the ability for fine-grained speech style control, a critical capability for human-like interaction that is often overlooked in favor of purely functional capabilities like reasoning and question answering. To address this limitation, we introduce UltraVoice, the first large-scale speech dialogue dataset engineered for multiple fine-grained speech style control. Encompassing over 830 hours of speech dialogues, UltraVoice provides instructions across six key speech stylistic dimensions: emotion, speed, volume, accent, language, and composite styles. Fine-tuning leading models such as SLAM-Omni and VocalNet on UltraVoice significantly enhances their fine-grained speech stylistic controllability without degrading core conversational abilities. Specifically, our fine-tuned models achieve improvements of 29.12-42.33% in Mean Opinion Score (MOS) and 14.61-40.09 percentage points in Instruction Following Rate (IFR) on multi-dimensional control tasks designed in the UltraVoice. Moreover, on the URO-Bench benchmark, our fine-tuned models demonstrate substantial gains in core understanding, reasoning, and conversational abilities, with average improvements of +10.84% on the Basic setting and +7.87% on the Pro setting. Furthermore, the dataset's utility extends to training controllable Text-to-Speech (TTS) models, underscoring its high quality and broad applicability for expressive speech synthesis. The complete dataset and model checkpoints are available at: this https URL. 

---
# OFFSIDE: Benchmarking Unlearning Misinformation in Multimodal Large Language Models 

**Authors**: Hao Zheng, Zirui Pang, Ling li, Zhijie Deng, Yuhan Pu, Zhaowei Zhu, Xiaobo Xia, Jiaheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.22535)  

**Abstract**: Advances in Multimodal Large Language Models (MLLMs) intensify concerns about data privacy, making Machine Unlearning (MU), the selective removal of learned information, a critical necessity. However, existing MU benchmarks for MLLMs are limited by a lack of image diversity, potential inaccuracies, and insufficient evaluation scenarios, which fail to capture the complexity of real-world applications. To facilitate the development of MLLMs unlearning and alleviate the aforementioned limitations, we introduce OFFSIDE, a novel benchmark for evaluating misinformation unlearning in MLLMs based on football transfer rumors. This manually curated dataset contains 15.68K records for 80 players, providing a comprehensive framework with four test sets to assess forgetting efficacy, generalization, utility, and robustness. OFFSIDE supports advanced settings like selective unlearning and corrective relearning, and crucially, unimodal unlearning (forgetting only text data). Our extensive evaluation of multiple baselines reveals key findings: (1) Unimodal methods (erasing text-based knowledge) fail on multimodal rumors; (2) Unlearning efficacy is largely driven by catastrophic forgetting; (3) All methods struggle with "visual rumors" (rumors appear in the image); (4) The unlearned rumors can be easily recovered and (5) All methods are vulnerable to prompt attacks. These results expose significant vulnerabilities in current approaches, highlighting the need for more robust multimodal unlearning solutions. The code is available at \href{this https URL}{this https URL}. 

---
# Scalable Oversight via Partitioned Human Supervision 

**Authors**: Ren Yin, Takashi Ishida, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2510.22500)  

**Abstract**: As artificial intelligence (AI) systems approach and surpass expert human performance across a broad range of tasks, obtaining high-quality human supervision for evaluation and training becomes increasingly challenging. Our focus is on tasks that require deep knowledge and skills of multiple domains. Unfortunately, even the best human experts are knowledgeable only in a single narrow area, and will not be able to evaluate the correctness of advanced AI systems on such superhuman tasks. However, based on their narrow expertise, humans may provide a weak signal, i.e., a complementary label indicating an option that is incorrect. For example, a cardiologist could state that "this is not related to cardiology,'' even if they cannot identify the true disease. Based on this weak signal, we propose a scalable oversight framework that enables us to evaluate frontier AI systems without the need to prepare the ground truth. We derive an unbiased estimator of top-1 accuracy from complementary labels and quantify how many complementary labels are needed to match the variance of ordinary labels. We further introduce two estimators to combine scarce ordinary labels with abundant complementary labels. We provide finite-sample deviation guarantees for both complementary-only and the mixed estimators. Empirically, we show that we can evaluate the output of large language models without the ground truth, if we have complementary labels. We further show that we can train an AI system with such weak signals: we show how we can design an agentic AI system automatically that can perform better with this partitioned human supervision. Our code is available at this https URL. 

---
# Modeling Hierarchical Thinking in Large Reasoning Models 

**Authors**: G M Shahariar, Ali Nazari, Erfan Shayegani, Nael Abu-Ghazaleh  

**Link**: [PDF](https://arxiv.org/pdf/2510.22437)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable reasoning abilities when they generate step-by-step solutions, known as chain-of-thought (CoT) reasoning. When trained to using chain-of-thought reasoning examples, the resulting models (called Large Reasoning Models, or LRMs) appear to learn hierarchical thinking strategies similar to those used by humans. However, understanding LRMs emerging reasoning capabilities remains a difficult open problem, with many potential important applications including improving training and understanding robustness. In this paper, we adopt a memoryless Finite State Machine formulation to approximate LRM's emerging hierarchical reasoning dynamics as a structured, interpretable abstraction. We identify a small set of discrete reasoning states including - initialization, deduction, augmentation-strategy, uncertainty-estimation, backtracking, and final-conclusion that capture the high-level states present in the model's reasoning process. By annotating each step of a model's CoT with these states, we can represent the reasoning trajectory as a transition sequence through the state graph. This FSM formulation provides a systematic way to analyze, interpret and visualize how different models approach problems. We describe the FSM model, provide examples of CoT annotations under this scheme, and discuss how it can shed light on differences between available models in their approach to reasoning. Our results demonstrate that this FSM-based analysis reveals distinct reasoning patterns and potential shortcomings, offering a new lens to evaluate and improve LLM reasoning. 

---
# Label Smoothing Improves Gradient Ascent in LLM Unlearning 

**Authors**: Zirui Pang, Hao Zheng, Zhijie Deng, Ling Li, Zixin Zhong, Jiaheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.22376)  

**Abstract**: LLM unlearning has emerged as a promising approach, aiming to enable models to forget hazardous/undesired knowledge at low cost while preserving as much model utility as possible. Among existing techniques, the most straightforward method is performing Gradient Ascent (GA) w.r.t. the forget data, thereby forcing the model to unlearn the forget dataset. However, GA suffers from severe instability, as it drives updates in a divergent direction, often resulting in drastically degraded model utility. To address this issue, we propose Smoothed Gradient Ascent (SGA). SGA combines the forget data with multiple constructed normal data through a tunable smoothing rate. Intuitively, this extends GA from learning solely on the forget data to jointly learning across both forget and normal data, enabling more stable unlearning while better preserving model utility. Theoretically, we provide the theoretical guidance on the selection of the optimal smoothing rate. Empirically, we evaluate SGA on three benchmarks: TOFU, Harry Potter, and MUSE-NEWS. Experimental results demonstrate that SGA consistently outperforms the original Gradient Ascent (GA) method across all metrics and achieves top-2 performance among all baseline methods on several key metrics. 

---
# Reasoning Models Reason Well, Until They Don't 

**Authors**: Revanth Rameshkumar, Jimson Huang, Yunxin Sun, Fei Xia, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2510.22371)  

**Abstract**: Large language models (LLMs) have shown significant progress in reasoning tasks. However, recent studies show that transformers and LLMs fail catastrophically once reasoning problems exceed modest complexity. We revisit these findings through the lens of large reasoning models (LRMs) -- LLMs fine-tuned with incentives for step-by-step argumentation and self-verification. LRM performance on graph and reasoning benchmarks such as NLGraph seem extraordinary, with some even claiming they are capable of generalized reasoning and innovation in reasoning-intensive fields such as mathematics, physics, medicine, and law. However, by more carefully scaling the complexity of reasoning problems, we show existing benchmarks actually have limited complexity. We develop a new dataset, the Deep Reasoning Dataset (DeepRD), along with a generative process for producing unlimited examples of scalable complexity. We use this dataset to evaluate model performance on graph connectivity and natural language proof planning. We find that the performance of LRMs drop abruptly at sufficient complexity and do not generalize. We also relate our LRM results to the distributions of the complexities of large, real-world knowledge graphs, interaction graphs, and proof datasets. We find the majority of real-world examples fall inside the LRMs' success regime, yet the long tails expose substantial failure potential. Our analysis highlights the near-term utility of LRMs while underscoring the need for new methods that generalize beyond the complexity of examples in the training distribution. 

---
# Mapping Faithful Reasoning in Language Models 

**Authors**: Jiazheng Li, Andreas Damianou, J Rosser, José Luis Redondo García, Konstantina Palla  

**Link**: [PDF](https://arxiv.org/pdf/2510.22362)  

**Abstract**: Chain-of-thought (CoT) traces promise transparency for reasoning language models, but prior work shows they are not always faithful reflections of internal computation. This raises challenges for oversight: practitioners may misinterpret decorative reasoning as genuine. We introduce Concept Walk, a general framework for tracing how a model's internal stance evolves with respect to a concept direction during reasoning. Unlike surface text, Concept Walk operates in activation space, projecting each reasoning step onto the concept direction learned from contrastive data. This allows us to observe whether reasoning traces shape outcomes or are discarded. As a case study, we apply Concept Walk to the domain of Safety using Qwen 3-4B. We find that in 'easy' cases, perturbed CoTs are quickly ignored, indicating decorative reasoning, whereas in 'hard' cases, perturbations induce sustained shifts in internal activations, consistent with faithful reasoning. The contribution is methodological: Concept Walk provides a lens to re-examine faithfulness through concept-specific internal dynamics, helping identify when reasoning traces can be trusted and when they risk misleading practitioners. 

---
# DynaSolidGeo: A Dynamic Benchmark for Genuine Spatial Mathematical Reasoning of VLMs in Solid Geometry 

**Authors**: Changti Wu, Shijie Lian, Zihao Liu, Lei Zhang, Laurence Tianruo Yang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.22340)  

**Abstract**: Solid geometry problem solving demands spatial mathematical reasoning that integrates spatial intelligence and symbolic reasoning. However, most existing multimodal mathematical reasoning benchmarks focus primarily on 2D plane geometry, rely on static datasets prone to data contamination and memorization, and evaluate models solely by final answers, overlooking the reasoning process. To address these limitations, we introduce DynaSolidGeo, the first dynamic benchmark for evaluating genuine spatial reasoning in Vision-Language Models (VLMs). Constructed through a semi-automatic annotation pipeline, DynaSolidGeo contains 503 expert-curated seed questions that can, in principle, dynamically generate an unbounded number of diverse multimodal text-visual instances. Beyond answer accuracy, we incorporate process evaluation based on expert-annotated reasoning chains to measure logical validity and causal coherence. Experiments across representative open-source and closed-source VLMs reveal large performance gaps, severe degradation in dynamic settings, and poor performance on tasks requiring high-level spatial intelligence, such as mental rotation and visualization. The code and dataset are available at \href{this https URL}{DynaSolidGeo}. 

---
# VietLyrics: A Large-Scale Dataset and Models for Vietnamese Automatic Lyrics Transcription 

**Authors**: Quoc Anh Nguyen, Bernard Cheng, Kelvin Soh  

**Link**: [PDF](https://arxiv.org/pdf/2510.22295)  

**Abstract**: Automatic Lyrics Transcription (ALT) for Vietnamese music presents unique challenges due to its tonal complexity and dialectal variations, but remains largely unexplored due to the lack of a dedicated dataset. Therefore, we curated the first large-scale Vietnamese ALT dataset (VietLyrics), comprising 647 hours of songs with line-level aligned lyrics and metadata to address these issues. Our evaluation of current ASRbased approaches reveal significant limitations, including frequent transcription errors and hallucinations in non-vocal segments. To improve performance, we fine-tuned Whisper models on the VietLyrics dataset, achieving superior results compared to existing multilingual ALT systems, including LyricWhiz. We publicly release VietLyrics and our models, aiming to advance Vietnamese music computing research while demonstrating the potential of this approach for ALT in low-resource language and music. 

---
# CityRiSE: Reasoning Urban Socio-Economic Status in Vision-Language Models via Reinforcement Learning 

**Authors**: Tianhui Liu, Hetian Pang, Xin Zhang, Jie Feng, Yong Li, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2510.22282)  

**Abstract**: Harnessing publicly available, large-scale web data, such as street view and satellite imagery, urban socio-economic sensing is of paramount importance for achieving global sustainable development goals. With the emergence of Large Vision-Language Models (LVLMs), new opportunities have arisen to solve this task by treating it as a multi-modal perception and understanding problem. However, recent studies reveal that LVLMs still struggle with accurate and interpretable socio-economic predictions from visual data. To address these limitations and maximize the potential of LVLMs, we introduce \textbf{CityRiSE}, a novel framework for \textbf{R}eason\textbf{i}ng urban \textbf{S}ocio-\textbf{E}conomic status in LVLMs through pure reinforcement learning (RL). With carefully curated multi-modal data and verifiable reward design, our approach guides the LVLM to focus on semantically meaningful visual cues, enabling structured and goal-oriented reasoning for generalist socio-economic status prediction. Experiments demonstrate that CityRiSE with emergent reasoning process significantly outperforms existing baselines, improving both prediction accuracy and generalization across diverse urban contexts, particularly for prediction on unseen cities and unseen indicators. This work highlights the promise of combining RL and LVLMs for interpretable and generalist urban socio-economic sensing. 

---
# WAON: Large-Scale and High-Quality Japanese Image-Text Pair Dataset for Vision-Language Models 

**Authors**: Issa Sugiura, Shuhei Kurita, Yusuke Oda, Daisuke Kawahara, Yasuo Okabe, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2510.22276)  

**Abstract**: Large-scale and high-quality image-text pair datasets play an important role in developing high-performing Vision-Language Models (VLMs). In this work, we introduce WAON, a large-scale and high-quality Japanese image-text pair dataset containing approximately 155 million examples, collected from Common Crawl. Our dataset construction pipeline employs various techniques, including filtering and deduplication, which have been shown to be effective in previous studies. To evaluate its effectiveness, we also construct WAON-Bench, a manually curated benchmark for Japanese cultural image classification, consisting of 374 classes. To assess the effectiveness of our dataset, we conduct experiments using both WAON and the Japanese subset of ReLAION, one of the most widely used vision-language datasets. We fine-tune SigLIP2, a strong multilingual model, on both datasets. The results demonstrate that WAON enhances model performance on WAON-Bench more efficiently than ReLAION and achieves higher accuracy across all evaluated benchmarks. Furthermore, the model fine-tuned on WAON achieves state-of-the-art performance on several Japanese cultural benchmarks. We release our dataset, model, and code at this https URL. 

---
# PACR: Progressively Ascending Confidence Reward for LLM Reasoning 

**Authors**: Eunseop Yoon, Hee Suk Yoon, Jaehyun Jang, SooHwan Eom, Qi Dai, Chong Luo, Mark A. Hasegawa-Johnson, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2510.22255)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has significantly improved LLM reasoning, but its sparse, outcome-based reward provides no guidance for intermediate steps, slowing exploration. We propose Progressively Ascending Confidence Reward (PACR), a dense, model-intrinsic reward computed directly from the model's evolving belief in the correct answer. PACR encodes the inductive bias that, along a well-formed reasoning trajectory, the probability of the ground-truth answer should have a generally ascending trend. We provide empirical and theoretical analysis validating that such an inductive bias constrains the exploration search space to regions richer in logically sound reasoning. We demonstrate that PACR accelerates exploration, reaches reward saturation with fewer trajectories, and yields improvements on multiple benchmarks. Our results suggest that dense, model-intrinsic shaping signals can make RLVR training more effective and reliable. 

---
# PaperAsk: A Benchmark for Reliability Evaluation of LLMs in Paper Search and Reading 

**Authors**: Yutao Wu, Xiao Liu, Yunhao Feng, Jiale Ding, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.22242)  

**Abstract**: Large Language Models (LLMs) increasingly serve as research assistants, yet their reliability in scholarly tasks remains under-evaluated. In this work, we introduce PaperAsk, a benchmark that systematically evaluates LLMs across four key research tasks: citation retrieval, content extraction, paper discovery, and claim verification. We evaluate GPT-4o, GPT-5, and Gemini-2.5-Flash under realistic usage conditions-via web interfaces where search operations are opaque to the user. Through controlled experiments, we find consistent reliability failures: citation retrieval fails in 48-98% of multi-reference queries, section-specific content extraction fails in 72-91% of cases, and topical paper discovery yields F1 scores below 0.32, missing over 60% of relevant literature. Further human analysis attributes these failures to the uncontrolled expansion of retrieved context and the tendency of LLMs to prioritize semantically relevant text over task instructions. Across basic tasks, the LLMs display distinct failure behaviors: ChatGPT often withholds responses rather than risk errors, whereas Gemini produces fluent but fabricated answers. To address these issues, we develop lightweight reliability classifiers trained on PaperAsk data to identify unreliable outputs. PaperAsk provides a reproducible and diagnostic framework for advancing the reliability evaluation of LLM-based scholarly assistance systems. 

---
# The Lossy Horizon: Error-Bounded Predictive Coding for Lossy Text Compression (Episode I) 

**Authors**: Nnamdi Aghanya, Jun Li, Kewei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22207)  

**Abstract**: Large Language Models (LLMs) can achieve near-optimal lossless compression by acting as powerful probability models. We investigate their use in the lossy domain, where reconstruction fidelity is traded for higher compression ratios. This paper introduces Error-Bounded Predictive Coding (EPC), a lossy text codec that leverages a Masked Language Model (MLM) as a decompressor. Instead of storing a subset of original tokens, EPC allows the model to predict masked content and stores minimal, rank-based corrections only when the model's top prediction is incorrect. This creates a residual channel that offers continuous rate-distortion control. We compare EPC to a simpler Predictive Masking (PM) baseline and a transform-based Vector Quantisation with a Residual Patch (VQ+RE) approach. Through an evaluation that includes precise bit accounting and rate-distortion analysis, we demonstrate that EPC consistently dominates PM, offering superior fidelity at a significantly lower bit rate by more efficiently utilising the model's intrinsic knowledge. 

---
# M-CIF: Multi-Scale Alignment For CIF-Based Non-Autoregressive ASR 

**Authors**: Ruixiang Mao, Xiangnan Ma, Qing Yang, Ziming Zhu, Yucheng Qiao, Yuan Ge, Tong Xiao, Shengxiang Gao, Zhengtao Yu, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22172)  

**Abstract**: The Continuous Integrate-and-Fire (CIF) mechanism provides effective alignment for non-autoregressive (NAR) speech recognition. This mechanism creates a smooth and monotonic mapping from acoustic features to target tokens, achieving performance on Mandarin competitive with other NAR approaches. However, without finer-grained guidance, its stability degrades in some languages such as English and French. In this paper, we propose Multi-scale CIF (M-CIF), which performs multi-level alignment by integrating character and phoneme level supervision progressively distilled into subword representations, thereby enhancing robust acoustic-text alignment. Experiments show that M-CIF reduces WER compared to the Paraformer baseline, especially on CommonVoice by 4.21% in German and 3.05% in French. To further investigate these gains, we define phonetic confusion errors (PE) and space-related segmentation errors (SE) as evaluation metrics. Analysis of these metrics across different M-CIF settings reveals that the phoneme and character layers are essential for enhancing progressive CIF alignment. 

---
# Surface Reading LLMs: Synthetic Text and its Styles 

**Authors**: Hannes Bajohr  

**Link**: [PDF](https://arxiv.org/pdf/2510.22162)  

**Abstract**: Despite a potential plateau in ML advancement, the societal impact of large language models lies not in approaching superintelligence but in generating text surfaces indistinguishable from human writing. While Critical AI Studies provides essential material and socio-technical critique, it risks overlooking how LLMs phenomenologically reshape meaning-making. This paper proposes a semiotics of "surface integrity" as attending to the immediate plane where LLMs inscribe themselves into human communication. I distinguish three knowledge interests in ML research (epistemology, epistēmē, and epistemics) and argue for integrating surface-level stylistic analysis alongside depth-oriented critique. Through two case studies examining stylistic markers of synthetic text, I argue how attending to style as a semiotic phenomenon reveals LLMs as cultural actors that transform the conditions of meaning emergence and circulation in contemporary discourse, independent of questions about machine consciousness. 

---
# Power to the Clients: Federated Learning in a Dictatorship Setting 

**Authors**: Mohammadsajad Alipour, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.22149)  

**Abstract**: Federated learning (FL) has emerged as a promising paradigm for decentralized model training, enabling multiple clients to collaboratively learn a shared model without exchanging their local data. However, the decentralized nature of FL also introduces vulnerabilities, as malicious clients can compromise or manipulate the training process. In this work, we introduce dictator clients, a novel, well-defined, and analytically tractable class of malicious participants capable of entirely erasing the contributions of all other clients from the server model, while preserving their own. We propose concrete attack strategies that empower such clients and systematically analyze their effects on the learning process. Furthermore, we explore complex scenarios involving multiple dictator clients, including cases where they collaborate, act independently, or form an alliance in order to ultimately betray one another. For each of these settings, we provide a theoretical analysis of their impact on the global model's convergence. Our theoretical algorithms and findings about the complex scenarios including multiple dictator clients are further supported by empirical evaluations on both computer vision and natural language processing benchmarks. 

---
# LOC: A General Language-Guided Framework for Open-Set 3D Occupancy Prediction 

**Authors**: Yuhang Gao, Xiang Xiang, Sheng Zhong, Guoyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22141)  

**Abstract**: Vision-Language Models (VLMs) have shown significant progress in open-set challenges. However, the limited availability of 3D datasets hinders their effective application in 3D scene understanding. We propose LOC, a general language-guided framework adaptable to various occupancy networks, supporting both supervised and self-supervised learning paradigms. For self-supervised tasks, we employ a strategy that fuses multi-frame LiDAR points for dynamic/static scenes, using Poisson reconstruction to fill voids, and assigning semantics to voxels via K-Nearest Neighbor (KNN) to obtain comprehensive voxel representations. To mitigate feature over-homogenization caused by direct high-dimensional feature distillation, we introduce Densely Contrastive Learning (DCL). DCL leverages dense voxel semantic information and predefined textual prompts. This efficiently enhances open-set recognition without dense pixel-level supervision, and our framework can also leverage existing ground truth to further improve performance. Our model predicts dense voxel features embedded in the CLIP feature space, integrating textual and image pixel information, and classifies based on text and semantic similarity. Experiments on the nuScenes dataset demonstrate the method's superior performance, achieving high-precision predictions for known classes and distinguishing unknown classes without additional training data. 

---
# Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs 

**Authors**: Jinzhe Liu, Junshu Sun, Shufan Shen, Chenxue Yang, Shuhui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22139)  

**Abstract**: Lifelong knowledge editing enables continuous, precise updates to outdated knowledge in large language models (LLMs) without computationally expensive full retraining. However, existing methods often accumulate errors throughout the editing process, causing a gradual decline in both editing accuracy and generalization. To tackle this problem, we propose Neuron-Specific Masked Knowledge Editing (NMKE), a novel fine-grained editing framework that combines neuron-level attribution with dynamic sparse masking. Leveraging neuron functional attribution, we identify two key types of knowledge neurons, with knowledge-general neurons activating consistently across prompts and knowledge-specific neurons activating to specific prompts. NMKE further introduces an entropy-guided dynamic sparse mask, locating relevant neurons to the target knowledge. This strategy enables precise neuron-level knowledge editing with fewer parameter modifications. Experimental results from thousands of sequential edits demonstrate that NMKE outperforms existing methods in maintaining high editing success rates and preserving model general capabilities in lifelong editing. 

---
# Mitigating Coordinate Prediction Bias from Positional Encoding Failures 

**Authors**: Xingjian Tao, Yiwei Wang, Yujun Cai, Yihong Luo, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22102)  

**Abstract**: Multimodal large language models (MLLMs) excel at vision-language tasks such as VQA and document understanding, yet precise coordinate prediction remains challenging. High-resolution inputs exacerbate this difficulty by producing long token sequences that weaken positional encodings and introduce directional biases in coordinate outputs. We investigate this phenomenon by analyzing how MLLMs behave when visual positional encodings (VPEs) are deliberately perturbed through shuffling. Our analysis reveals that such perturbations induce predictable, non-random coordinate biases rather than random errors, suggesting that models rely on internal positional priors when spatial grounding signals are degraded. Crucially, we observe similar directional error patterns in natural high-resolution datasets, indicating that positional encoding failures are a key bottleneck for accurate coordinate prediction at scale. To address this issue, we propose Vision-PE Shuffle Guidance (VPSG), a training-free test-time method that leverages the directional nature of these biases for correction. VPSG runs auxiliary decoding with shuffled VPEs to isolate position-unconditioned tendencies, then uses this as negative evidence to guide digit prediction while preserving coordinate format through a lightweight finite-state machine. Experiments on ScreenSpot-Pro demonstrate reliable improvements, highlighting positional encoding robustness as a critical factor for spatial reasoning in MLLMs. 

---
# Embracing Trustworthy Brain-Agent Collaboration as Paradigm Extension for Intelligent Assistive Technologies 

**Authors**: Yankai Chen, Xinni Zhang, Yifei Zhang, Yangning Li, Henry Peng Zou, Chunyu Miao, Weizhi Zhang, Xue Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22095)  

**Abstract**: Brain-Computer Interfaces (BCIs) offer a direct communication pathway between the human brain and external devices, holding significant promise for individuals with severe neurological impairments. However, their widespread adoption is hindered by critical limitations, such as low information transfer rates and extensive user-specific calibration. To overcome these challenges, recent research has explored the integration of Large Language Models (LLMs), extending the focus from simple command decoding to understanding complex cognitive states. Despite these advancements, deploying agentic AI faces technical hurdles and ethical concerns. Due to the lack of comprehensive discussion on this emerging direction, this position paper argues that the field is poised for a paradigm extension from BCI to Brain-Agent Collaboration (BAC). We emphasize reframing agents as active and collaborative partners for intelligent assistance rather than passive brain signal data processors, demanding a focus on ethical data handling, model reliability, and a robust human-agent collaboration framework to ensure these systems are safe, trustworthy, and effective. 

---
# Jailbreak Mimicry: Automated Discovery of Narrative-Based Jailbreaks for Large Language Models 

**Authors**: Pavlos Ntais  

**Link**: [PDF](https://arxiv.org/pdf/2510.22085)  

**Abstract**: Large language models (LLMs) remain vulnerable to sophisticated prompt engineering attacks that exploit contextual framing to bypass safety mechanisms, posing significant risks in cybersecurity applications. We introduce Jailbreak Mimicry, a systematic methodology for training compact attacker models to automatically generate narrative-based jailbreak prompts in a one-shot manner. Our approach transforms adversarial prompt discovery from manual craftsmanship into a reproducible scientific process, enabling proactive vulnerability assessment in AI-driven security systems. Developed for the OpenAI GPT-OSS-20B Red-Teaming Challenge, we use parameter-efficient fine-tuning (LoRA) on Mistral-7B with a curated dataset derived from AdvBench, achieving an 81.0% Attack Success Rate (ASR) against GPT-OSS-20B on a held-out test set of 200 items. Cross-model evaluation reveals significant variation in vulnerability patterns: our attacks achieve 66.5% ASR against GPT-4, 79.5% on Llama-3 and 33.0% against Gemini 2.5 Flash, demonstrating both broad applicability and model-specific defensive strengths in cybersecurity contexts. This represents a 54x improvement over direct prompting (1.5% ASR) and demonstrates systematic vulnerabilities in current safety alignment approaches. Our analysis reveals that technical domains (Cybersecurity: 93% ASR) and deception-based attacks (Fraud: 87.8% ASR) are particularly vulnerable, highlighting threats to AI-integrated threat detection, malware analysis, and secure systems, while physical harm categories show greater resistance (55.6% ASR). We employ automated harmfulness evaluation using Claude Sonnet 4, cross-validated with human expert assessment, ensuring reliable and scalable evaluation for cybersecurity red-teaming. Finally, we analyze failure mechanisms and discuss defensive strategies to mitigate these vulnerabilities in AI for cybersecurity. 

---
# Agentic Reinforcement Learning for Real-World Code Repair 

**Authors**: Siyu Zhu, Anastasiya Karpovich, Albert Chen, Jessica Koscheka, Shailesh Jannu, Di Wen, Yuqing Zhu, Rohit Jain, Alborz Geramifard  

**Link**: [PDF](https://arxiv.org/pdf/2510.22075)  

**Abstract**: We tackle the challenge of training reliable code-fixing agents in real repositories, where complex builds and shifting dependencies make evaluation unstable. We developed a verifiable pipeline with success defined as post-fix build validation and improved reproducibility across ~1K real issues by pinning dependencies and disabling automatic upgrades. Building on this, we introduced a scalable simplified pipeline for large-scale reinforcement learning (RL). Using this setup, we supervised fine-tuned Qwen3-32B in the full pipeline and applied RL on top of the SFT model in the simplified environment. The SFT model distilled from GPT-4.1 trajectories performs on par while being 56x smaller, and RL added 7-20% absolute gains under matched train-test conditions. "Thinking mode" was on par or worse in our experiments. Both SFT and RL models failed to generalize across environments, highlighting the importance of matching train-test environments for building reliable real-world code-fixing agents. 

---
# A Benchmark for Open-Domain Numerical Fact-Checking Enhanced by Claim Decomposition 

**Authors**: V Venktesh, Deepali Prabhu, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2510.22055)  

**Abstract**: Fact-checking numerical claims is critical as the presence of numbers provide mirage of veracity despite being fake potentially causing catastrophic impacts on society. The prior works in automatic fact verification do not primarily focus on natural numerical claims. A typical human fact-checker first retrieves relevant evidence addressing the different numerical aspects of the claim and then reasons about them to predict the veracity of the claim. Hence, the search process of a human fact-checker is a crucial skill that forms the foundation of the verification process. Emulating a real-world setting is essential to aid in the development of automated methods that encompass such skills. However, existing benchmarks employ heuristic claim decomposition approaches augmented with weakly supervised web search to collect evidences for verifying claims. This sometimes results in less relevant evidences and noisy sources with temporal leakage rendering a less realistic retrieval setting for claim verification. Hence, we introduce QuanTemp++: a dataset consisting of natural numerical claims, an open domain corpus, with the corresponding relevant evidence for each claim. The evidences are collected through a claim decomposition process approximately emulating the approach of human fact-checker and veracity labels ensuring there is no temporal leakage. Given this dataset, we also characterize the retrieval performance of key claim decomposition paradigms. Finally, we observe their effect on the outcome of the verification pipeline and draw insights. The code for data pipeline along with link to data can be found at this https URL 

---
# Optimal Detection for Language Watermarks with Pseudorandom Collision 

**Authors**: T. Tony Cai, Xiang Li, Qi Long, Weijie J. Su, Garrett G. Wen  

**Link**: [PDF](https://arxiv.org/pdf/2510.22007)  

**Abstract**: Text watermarking plays a crucial role in ensuring the traceability and accountability of large language model (LLM) outputs and mitigating misuse. While promising, most existing methods assume perfect pseudorandomness. In practice, repetition in generated text induces collisions that create structured dependence, compromising Type I error control and invalidating standard analyses.
We introduce a statistical framework that captures this structure through a hierarchical two-layer partition. At its core is the concept of minimal units -- the smallest groups treatable as independent across units while permitting dependence within. Using minimal units, we define a non-asymptotic efficiency measure and cast watermark detection as a minimax hypothesis testing problem.
Applied to Gumbel-max and inverse-transform watermarks, our framework produces closed-form optimal rules. It explains why discarding repeated statistics often improves performance and shows that within-unit dependence must be addressed unless degenerate. Both theory and experiments confirm improved detection power with rigorous Type I error control. These results provide the first principled foundation for watermark detection under imperfect pseudorandomness, offering both theoretical insight and practical guidance for reliable tracing of model outputs. 

---
# From Social Division to Cohesion with AI Message Suggestions in Online Chat Groups 

**Authors**: Faria Huq, Elijah L. Claggett, Hirokazu Shirado  

**Link**: [PDF](https://arxiv.org/pdf/2510.21984)  

**Abstract**: Social cohesion is difficult to sustain in societies marked by opinion diversity, particularly in online communication. As large language model (LLM)-driven messaging assistance becomes increasingly embedded in these contexts, it raises critical questions about its societal impact. We present an online experiment with 557 participants who engaged in multi-round discussions on politically controversial topics while freely reconfiguring their discussion groups. In some conditions, participants received real-time message suggestions generated by an LLM, either personalized to the individual or adapted to their group context. We find that subtle shifts in linguistic style during communication, mediated by AI assistance, can scale up to reshape collective structures. While individual-focused assistance leads users to segregate into like-minded groups, relational assistance that incorporates group members' stances enhances cohesion through more receptive exchanges. These findings demonstrate that AI-mediated communication can support social cohesion in diverse groups, but outcomes critically depend on how personalization is designed. 

---
# Performance Trade-offs of Optimizing Small Language Models for E-Commerce 

**Authors**: Josip Tomo Licardo, Nikola Tankovic  

**Link**: [PDF](https://arxiv.org/pdf/2510.21970)  

**Abstract**: Large Language Models (LLMs) offer state-of-the-art performance in natural language understanding and generation tasks. However, the deployment of leading commercial models for specialized tasks, such as e-commerce, is often hindered by high computational costs, latency, and operational expenses. This paper investigates the viability of smaller, open-weight models as a resource-efficient alternative. We present a methodology for optimizing a one-billion-parameter Llama 3.2 model for multilingual e-commerce intent recognition. The model was fine-tuned using Quantized Low-Rank Adaptation (QLoRA) on a synthetically generated dataset designed to mimic real-world user queries. Subsequently, we applied post-training quantization techniques, creating GPU-optimized (GPTQ) and CPU-optimized (GGUF) versions. Our results demonstrate that the specialized 1B model achieves 99% accuracy, matching the performance of the significantly larger GPT-4.1 model. A detailed performance analysis revealed critical, hardware-dependent trade-offs: while 4-bit GPTQ reduced VRAM usage by 41%, it paradoxically slowed inference by 82% on an older GPU architecture (NVIDIA T4) due to dequantization overhead. Conversely, GGUF formats on a CPU achieved a speedup of up to 18x in inference throughput and a reduction of over 90% in RAM consumption compared to the FP16 baseline. We conclude that small, properly optimized open-weight models are not just a viable but a more suitable alternative for domain-specific applications, offering state-of-the-art accuracy at a fraction of the computational cost. 

---
# Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing 

**Authors**: Iskander Azangulov, Teodora Pandeva, Niranjani Prasad, Javier Zazo, Sushrut Karmalkar  

**Link**: [PDF](https://arxiv.org/pdf/2510.21961)  

**Abstract**: Masked diffusion models (MDMs) offer a compelling alternative to autoregressive models (ARMs) for discrete text generation because they enable parallel token sampling, rather than sequential, left-to-right generation. This means potentially much faster inference. However, effective parallel sampling faces two competing requirements: (i) simultaneously updated tokens must be conditionally independent, and (ii) updates should prioritise high-confidence predictions. These goals conflict because high-confidence predictions often cluster and depend on each other, opportunities for parallel updates.
We present PUNT, a model-agnostic sampler that reconciles this trade-off. Our method identifies token dependencies and removes lower-confidence tokens from conflicting groups. This produces sets of indices for unmasking that satisfy both independence and confidence criteria. Our approach ensures improved parallel unmasking through approximate conditional independence testing.
Our experiments show that PUNT delivers a superior trade-off between accuracy and compute when compared to other strong training-free baselines, especially for generation of longer sequences. On the IFEval benchmark, it achieves up to 16\% higher accuracy over baseline methods, including sequential generation (one-by-one). These gains hold across different values of hyperparameters, mitigating the need for brittle hyperparameter tuning. Moreover, we observe that PUNT induces an emergent hierarchical generation strategy, where the model first establishes high-level paragraph structure before local refinement, suggesting a planning-like generation process that contributes to strong alignment performance. 

---
# Transformer Based Linear Attention with Optimized GPU Kernel Implementation 

**Authors**: Armin Gerami, Ramani Duraiswami  

**Link**: [PDF](https://arxiv.org/pdf/2510.21956)  

**Abstract**: The original softmax-based attention mechanism (regular attention) in the extremely successful Transformer architecture computes attention between $N$ tokens, each embedded in a $D$-dimensional head, with a time complexity of $O(N^2D)$. Given the success of Transformers, improving their runtime during both training and inference is a popular research area. One such approach is the introduction of the linear attention (LA) mechanisms, which offers a linear time complexity of $O(ND^2)$ and have demonstrated comparable accuracy to regular attention. However, LA in practice lags behind its theoretical efficiency. We propose a novel method for LA's forward and backward passes, along with a highly-optimized CUDA implementation. Our approach outperforms the state-of-the-art by 3.3 times in speed and reduces memory consumption by 3.6 times. We validate these improvements in both single-layer and end-to-end settings by training a 1.4 billion parameter language model, which demonstrates similar expressivity to regular attention on major reasoning benchmarks. 

---
# GeoThought: A Dataset for Enhancing Mathematical Geometry Reasoning in Vision-Language Models 

**Authors**: Nannan Shi, Chuanyu Qin, Shipeng Song, Man Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.21881)  

**Abstract**: Large language models (LLMs) have demonstrated strong reasoning capabilities in text-based mathematical problem solving; however, when adapted to visual reasoning tasks, particularly geometric problem solving, their performance substantially declines because geometric problems present unique challenges. Specifically, these challenges stem from two key factors: first, the intrinsic complexity of geometry requiring detailed image comprehension and multi-step reasoning, and second, the limitations of existing datasets which lack sufficient scale, diversity, and explicit reasoning traces, consequently hindering effective model training. To address these challenges, we developed the GeoThoughts dataset, a comprehensive geometric reasoning corpus with two subsets: Geo-Thought-6K with 6,243 samples and its augmented version Geo-Thought-Augmented-10K containing 10,834 samples. Each entry includes visual descriptions, step-by-step solutions, explicit reasoning chains, reflection steps, and final answers. Using this dataset, we developed GeoThought-MLLM, a mathematical reasoning multimodal model that generates detailed thinking processes during problem-solving. Our model outperforms existing benchmarks in geometric tasks, demonstrating that training with our Chain-of-Thought dataset improves geometric reasoning capabilities across both in-domain and out-of-domain settings. Finally, we analyze failure cases and observe that errors primarily arise from incorrect interpretation of mathematical concepts or spatial misjudgment. By invoking CoT to correct these mistakes, the model produces correct answers. 

---
# The Mirror Loop: Recursive Non-Convergence in Generative Reasoning Systems 

**Authors**: Bentley DeVilling  

**Link**: [PDF](https://arxiv.org/pdf/2510.21861)  

**Abstract**: Large language models are often described as capable of reflective reasoning, yet recursive self-evaluation without external feedback frequently yields reformulation rather than progress. We test this prediction in a cross-provider study of 144 reasoning sequences across three models (OpenAI GPT-4o-mini, Anthropic Claude 3 Haiku, and Google Gemini 2.0 Flash) and four task families (arithmetic, code, explanation, reflection), each iterated ten times under two conditions: ungrounded self-critique and a minimal grounding intervention (a single verification step at iteration three). Mean informational change (delta I, measured via normalized edit distance) declined by 55% from early (0.193) to late (0.087) iterations in ungrounded runs, with consistent patterns across all three providers. Grounded runs showed a +28% rebound in informational change immediately after the intervention and sustained non-zero variance thereafter. Complementary measures-n-gram novelty, embedding drift, and character-level entropy-converged on the same pattern: reflection without contact tends toward informational closure. We interpret this as evidence for a structural limit on self-correction in generative reasoning: without an exchange of information with an independent verifier or environment, recursive inference approaches an attractor state of epistemic stasis. Minimal grounding functions as dissipative coupling, reintroducing informational flux. The cross-architecture consistency suggests the mirror loop arises from shared autoregressive training objectives rather than provider-specific alignment schemes. The results delineate when reflection is performative rather than epistemic and motivate design principles for grounded, cooperative reasoning. Materials and code are publicly available. 

---
# SIGN: Schema-Induced Games for Naming 

**Authors**: Ryan Zhang, Herbert Woisetscläger  

**Link**: [PDF](https://arxiv.org/pdf/2510.21855)  

**Abstract**: Real-world AI systems are tackling increasingly complex problems, often through interactions among large language model (LLM) agents. When these agents develop inconsistent conventions, coordination can break down. Applications such as collaborative coding and distributed planning therefore require reliable, consistent communication, and scalability is a central concern as systems grow. We introduce Schema-Induced Games for Naming (SIGN), a naming game that examines how lightweight structure can steer convention formation. We compare schema-induced communication to unconstrained natural language and find faster convergence with up to 5.8x higher agreement. These results suggest that minimal structure can act as a simple control knob for efficient multi-agent coordination, pointing toward broader applications beyond the naming game. 

---
# SCoPE VLM: Selective Context Processing for Efficient Document Navigation in Vision-Language Models 

**Authors**: Gyubeum Lim, Yemo Koo, Vijay Krishna Madisetti  

**Link**: [PDF](https://arxiv.org/pdf/2510.21850)  

**Abstract**: Understanding long-context visual information remains a fundamental challenge for vision-language models, particularly in agentic tasks such as GUI control and web navigation. While web pages and GUI environments are inherently structured documents, current VLMs typically neglect decision-oriented document understanding in their training objectives. Existing approaches primarily extend visual embeddings to process long, high-resolution inputs, but these methods are memory-intensive and impractical for locally deployable solutions. To address these issues, we propose SCoPE VLM, a document navigation expert that leverages a novel Chain of Scroll mechanism to selectively and recursively navigate documents, focusing exclusively on relevant segments. We introduce a dedicated data generation pipeline to construct informative Chain of Scroll trajectories and Episodic Group Relative Policy Optimization, a tailored reinforcement learning method to reduce the gap between training and inference. Our method substantially reduces memory usage and effectively models human-like reading behaviors. To the best of our knowledge, SCoPE VLM is the first framework to explicitly model agentic reading patterns in multi-page document question answering, advancing the capabilities of multimodal agents. 

---
# A Multimodal, Multitask System for Generating E Commerce Text Listings from Images 

**Authors**: Nayan Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.21835)  

**Abstract**: Manually generating catchy descriptions and names is labor intensive and a slow process for retailers. Although generative AI provides an automation solution in form of Vision to Language Models (VLM), the current VLMs are prone to factual "hallucinations". Siloed, single task models are not only inefficient but also fail to capture interdependent relationships between features. To address these challenges, we propose an end to end, multi task system that generates factually grounded textual listings from a single image. The contributions of this study are two proposals for the model architecture. First, application of multi task learning approach for fine tuning a vision encoder where a single vision backbone is jointly trained on attribute prediction such as color, hemline and neck style and price regression. Second, introduction of a hierarchical generation process where the model's own predicted attributes are embedded in a prompt and fed to the text decoder to improve factual consistency. The experiments demonstrate the superiority of this architecture. The multi tasking approach outperforms both the independent price regression, with a 3.6% better R2 Value and attribute classification, with a 6.6% improvement F1 score. Critically, the hierarchical generation process proves highly effective, slashing the factual hallucination rate from 12.7% to 7.1%, a 44.5% relative reduction, compared to a non hierarchical ablation. The hierarchical approach also reduces the latency of the autoregressive text generation process by a factor of 3.5 when compared to direct vision to language model of similar size. One minor caveat is that the model does perform 3.5% worse than direct vision-to-language model on ROUGE-L score. 

---
# Structured and Abstractive Reasoning on Multi-modal Relational Knowledge Images 

**Authors**: Yichi Zhang, Zhuo Chen, Lingbing Guo, Lei Liang, Wen Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.21828)  

**Abstract**: Understanding and reasoning with abstractive information from the visual modality presents significant challenges for current multi-modal large language models (MLLMs). Among the various forms of abstractive information, Multi-Modal Relational Knowledge (MMRK), which represents abstract relational structures between multi-modal entities using node-edge formats, remains largely under-explored. In particular, STructured and Abstractive Reasoning (STAR) on such data has received little attention from the research community. To bridge the dual gaps in large-scale high-quality data and capability enhancement methodologies, this paper makes the following key contributions: (i). An automatic STAR data engine capable of synthesizing images with MMRK to build multi-modal instruction data with reliable chain-of-thought thinking for various STAR tasks and (ii). A comprehsive two-stage capability enhancement training framework, accompanied by a suite of evaluation protocols tailored to different STAR tasks. Based upon these contributions, we introduce STAR-64K, a dataset comprising 64K high-quality multi-modal instruction samples, and conduct experiments across 5 open-source MLLMs. Experimental results show that our two-stage enhancement framework enables smaller 3B/7B models to significantly outperform GPT-4o in STAR. Additionally, we provide in-depth analysis regarding the effectiveness of various designs, data transferability, and scalability. 

---
# VITA-E: Natural Embodied Interaction with Concurrent Seeing, Hearing, Speaking, and Acting 

**Authors**: Xiaoyu Liu, Chaoyou Fu, Chi Yan, Chu Wu, Haihan Gao, Yi-Fan Zhang, Shaoqi Dong, Cheng Qian, Bin Luo, Xiuyong Yang, Guanwu Li, Yusheng Cai, Yunhang Shen, Deqiang Jiang, Haoyu Cao, Xing Sun, Caifeng Shan, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2510.21817)  

**Abstract**: Current Vision-Language-Action (VLA) models are often constrained by a rigid, static interaction paradigm, which lacks the ability to see, hear, speak, and act concurrently as well as handle real-time user interruptions dynamically. This hinders seamless embodied collaboration, resulting in an inflexible and unresponsive user experience. To address these limitations, we introduce VITA-E, a novel embodied interaction framework designed for both behavioral concurrency and nearly real-time interruption. The core of our approach is a dual-model architecture where two parallel VLA instances operate as an ``Active Model'' and a ``Standby Model'', allowing the embodied agent to observe its environment, listen to user speech, provide verbal responses, and execute actions, all concurrently and interruptibly, mimicking human-like multitasking capabilities. We further propose a ``model-as-controller'' paradigm, where we fine-tune the VLM to generate special tokens that serve as direct system-level commands, coupling the model's reasoning with the system's behavior. Experiments conducted on a physical humanoid platform demonstrate that VITA-E can reliably handle complex interactive scenarios. Our framework is compatible with various dual-system VLA models, achieving an extremely high success rate on emergency stops and speech interruptions while also successfully performing concurrent speech and action. This represents a significant step towards more natural and capable embodied assistants. 

---
# Diagnosing Bottlenecks in Data Visualization Understanding by Vision-Language Models 

**Authors**: Alexa R. Tartaglini, Satchel Grant, Daniel Wurgaft, Christopher Potts, Judith E. Fan  

**Link**: [PDF](https://arxiv.org/pdf/2510.21740)  

**Abstract**: Data visualizations are vital components of many scientific articles and news stories. Current vision-language models (VLMs) still struggle on basic data visualization understanding tasks, but the causes of failure remain unclear. Are VLM failures attributable to limitations in how visual information in the data visualization is encoded, how information is transferred between the vision and language modules, or how information is processed within the language module? We developed FUGU, a suite of data visualization understanding tasks, to precisely characterize potential sources of difficulty (e.g., extracting the position of data points, distances between them, and other summary statistics). We used FUGU to investigate three widely used VLMs. To diagnose the sources of errors produced by these models, we used activation patching and linear probes to trace information flow through models across a variety of prompting strategies. We found that some models fail to generate the coordinates of individual data points correctly, and these initial errors often lead to erroneous final responses. When these models are provided with the correct coordinates, performance improves substantially. Moreover, even when the model generates an incorrect response, the correct coordinates can be successfully read out from the latent representations in the vision encoder, suggesting that the source of these errors lies in the vision-language handoff. We further found that while providing correct coordinates helps with tasks involving one or a small number of data points, it generally worsens performance for tasks that require extracting statistical relationships across many data points. Fine-tuning models on FUGU also fails to yield ceiling performance. These findings point to architectural constraints in current VLMs that might pose significant challenges for reliable data visualization understanding. 

---
# Next-Generation LLM for UAV: From Natural Language to Autonomous Flight 

**Authors**: Liangqi Yuan, Chuhao Deng, Dong-Jun Han, Inseok Hwang, Sabine Brunswicker, Christopher G. Brinton  

**Link**: [PDF](https://arxiv.org/pdf/2510.21739)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), their capabilities in various automation domains, particularly Unmanned Aerial Vehicle (UAV) operations, have garnered increasing attention. Current research remains predominantly constrained to small-scale UAV applications, with most studies focusing on isolated components such as path planning for toy drones, while lacking comprehensive investigation of medium- and long-range UAV systems in real-world operational contexts. Larger UAV platforms introduce distinct challenges, including stringent requirements for airport-based take-off and landing procedures, adherence to complex regulatory frameworks, and specialized operational capabilities with elevated mission expectations. This position paper presents the Next-Generation LLM for UAV (NeLV) system -- a comprehensive demonstration and automation roadmap for integrating LLMs into multi-scale UAV operations. The NeLV system processes natural language instructions to orchestrate short-, medium-, and long-range UAV missions through five key technical components: (i) LLM-as-Parser for instruction interpretation, (ii) Route Planner for Points of Interest (POI) determination, (iii) Path Planner for waypoint generation, (iv) Control Platform for executable trajectory implementation, and (v) UAV monitoring. We demonstrate the system's feasibility through three representative use cases spanning different operational scales: multi-UAV patrol, multi-POI delivery, and multi-hop relocation. Beyond the current implementation, we establish a five-level automation taxonomy that charts the evolution from current LLM-as-Parser capabilities (Level 1) to fully autonomous LLM-as-Autopilot systems (Level 5), identifying technical prerequisites and research challenges at each stage. 

---
# When Robots Say No: Temporal Trust Recovery Through Explanation 

**Authors**: Nicola Webb, Zijun Huang, Sanja Milivojevic, Chris Baber, Edmund R. Hunt  

**Link**: [PDF](https://arxiv.org/pdf/2510.21716)  

**Abstract**: Mobile robots with some degree of autonomy could deliver significant advantages in high-risk missions such as search and rescue and firefighting. Integrated into a human-robot team (HRT), robots could work effectively to help search hazardous buildings. User trust is a key enabler for HRT, but during a mission, trust can be damaged. With distributed situation awareness, such as when team members are working in different locations, users may be inclined to doubt a robot's integrity if it declines to immediately change its priorities on request. In this paper, we present the results of a computer-based study investigating on-mission trust dynamics in a high-stakes human-robot teaming scenario. Participants (n = 38) played an interactive firefighting game alongside a robot teammate, where a trust violation occurs owing to the robot declining to help the user immediately. We find that when the robot provides an explanation for declining to help, trust better recovers over time, albeit following an initial drop that is comparable to a baseline condition where an explanation for refusal is not provided. Our findings indicate that trust can vary significantly during a mission, notably when robots do not immediately respond to user requests, but that this trust violation can be largely ameliorated over time if adequate explanation is provided. 

---
# Beyond IVR Touch-Tones: Customer Intent Routing using LLMs 

**Authors**: Sergio Rojas-Galeano  

**Link**: [PDF](https://arxiv.org/pdf/2510.21715)  

**Abstract**: Widespread frustration with rigid touch-tone Interactive Voice Response (IVR) systems for customer service underscores the need for more direct and intuitive language interaction. While speech technologies are necessary, the key challenge lies in routing intents from user phrasings to IVR menu paths, a task where Large Language Models (LLMs) show strong potential. Progress, however, is limited by data scarcity, as real IVR structures and interactions are often proprietary. We present a novel LLM-based methodology to address this gap. Using three distinct models, we synthesized a realistic 23-node IVR structure, generated 920 user intents (230 base and 690 augmented), and performed the routing task. We evaluate two prompt designs: descriptive hierarchical menus and flattened path representations, across both base and augmented datasets. Results show that flattened paths consistently yield higher accuracy, reaching 89.13% on the base dataset compared to 81.30% with the descriptive format, while augmentation introduces linguistic noise that slightly reduces performance. Confusion matrix analysis further suggests that low-performing routes may reflect not only model limitations but also redundancies in menu design. Overall, our findings demonstrate proof-of-concept that LLMs can enable IVR routing through a smoother, more seamless user experience -- moving customer service one step ahead of touch-tone menus. 

---
# DecoupleSearch: Decouple Planning and Search via Hierarchical Reward Modeling 

**Authors**: Hao Sun, Zile Qiao, Bo Wang, Guoxin Chen, Yingyan Hou, Yong Jiang, Pengjun Xie, Fei Huang, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21712)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a pivotal methodology for enhancing Large Language Models (LLMs) through the dynamic integration of external knowledge. To further improve RAG's flexibility, Agentic RAG introduces autonomous agents into the workflow. However, Agentic RAG faces several challenges: (1) the success of each step depends on both high-quality planning and accurate search, (2) the lack of supervision for intermediate reasoning steps, and (3) the exponentially large candidate space for planning and searching. To address these challenges, we propose DecoupleSearch, a novel framework that decouples planning and search processes using dual value models, enabling independent optimization of plan reasoning and search grounding. Our approach constructs a reasoning tree, where each node represents planning and search steps. We leverage Monte Carlo Tree Search to assess the quality of each step. During inference, Hierarchical Beam Search iteratively refines planning and search candidates with dual value models. Extensive experiments across policy models of varying parameter sizes, demonstrate the effectiveness of our method. 

---
# BugPilot: Complex Bug Generation for Efficient Learning of SWE Skills 

**Authors**: Atharv Sonwane, Isadora White, Hyunji Lee, Matheus Pereira, Lucas Caccia, Minseon Kim, Zhengyan Shi, Chinmay Singh, Alessandro Sordoni, Marc-Alexandre Côté, Xingdi Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.19898)  

**Abstract**: High quality bugs are key to training the next generation of language model based software engineering (SWE) agents. We introduce a novel method for synthetic generation of difficult and diverse bugs. Our method instructs SWE Agents to introduce a feature into the codebase whereby they may unintentionally break tests, resulting in bugs. Prior approaches often induce an out-of-distribution effect by generating bugs intentionally (e.g. by introducing local perturbation to existing code), which does not reflect realistic development processes. We perform qualitative analysis to demonstrate that our approach for generating bugs more closely reflects the patterns found in human-authored edits. Through extensive experiments, we demonstrate that our bugs provide more efficient training data for supervised fine-tuning, outperforming other bug datasets by 2% with half the training data (1.2k vs. 3k bugs). We train on our newly generated bugs in addition to existing bug datasets to get FrogBoss a state-of-the-art 32B parameter model on SWE-bench Verified with a pass@1 of 54.6% and FrogMini a state-of-the-art 14B model on SWE-bench Verified with a pass@1 of 45.3% on SWE-bench Verified averaged over three seeds. 

---
