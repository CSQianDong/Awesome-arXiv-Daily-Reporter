# Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies 

**Authors**: Massimiliano Pronesti, Joao Bettencourt-Silva, Paul Flanagan, Alessandra Pascale, Oisin Redmond, Anya Belz, Yufang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06186)  

**Abstract**: Extracting scientific evidence from biomedical studies for clinical research questions (e.g., Does stem cell transplantation improve quality of life in patients with medically refractory Crohn's disease compared to placebo?) is a crucial step in synthesising biomedical evidence. In this paper, we focus on the task of document-level scientific evidence extraction for clinical questions with conflicting evidence. To support this task, we create a dataset called CochraneForest, leveraging forest plots from Cochrane systematic reviews. It comprises 202 annotated forest plots, associated clinical research questions, full texts of studies, and study-specific conclusions. Building on CochraneForest, we propose URCA (Uniform Retrieval Clustered Augmentation), a retrieval-augmented generation framework designed to tackle the unique challenges of evidence extraction. Our experiments show that URCA outperforms the best existing methods by up to 10.3% in F1 score on this task. However, the results also underscore the complexity of CochraneForest, establishing it as a challenging testbed for advancing automated evidence synthesis systems. 

---
# Estimating Quality in Therapeutic Conversations: A Multi-Dimensional Natural Language Processing Framework 

**Authors**: Alice Rueda, Argyrios Perivolaris, Niloy Roy, Dylan Weston, Sarmed Shaya, Zachary Cote, Martin Ivanov, Bazen G. Teferra, Yuqi Wu, Sirisha Rambhatla, Divya Sharma, Andrew Greenshaw, Rakesh Jetly, Yanbo Zhang, Bo Cao, Reza Samavi, Sridhar Krishnan, Venkat Bhat  

**Link**: [PDF](https://arxiv.org/pdf/2505.06151)  

**Abstract**: Engagement between client and therapist is a critical determinant of therapeutic success. We propose a multi-dimensional natural language processing (NLP) framework that objectively classifies engagement quality in counseling sessions based on textual transcripts. Using 253 motivational interviewing transcripts (150 high-quality, 103 low-quality), we extracted 42 features across four domains: conversational dynamics, semantic similarity as topic alignment, sentiment classification, and question detection. Classifiers, including Random Forest (RF), Cat-Boost, and Support Vector Machines (SVM), were hyperparameter tuned and trained using a stratified 5-fold cross-validation and evaluated on a holdout test set. On balanced (non-augmented) data, RF achieved the highest classification accuracy (76.7%), and SVM achieved the highest AUC (85.4%). After SMOTE-Tomek augmentation, performance improved significantly: RF achieved up to 88.9% accuracy, 90.0% F1-score, and 94.6% AUC, while SVM reached 81.1% accuracy, 83.1% F1-score, and 93.6% AUC. The augmented data results reflect the potential of the framework in future larger-scale applications. Feature contribution revealed conversational dynamics and semantic similarity between clients and therapists were among the top contributors, led by words uttered by the client (mean and standard deviation). The framework was robust across the original and augmented datasets and demonstrated consistent improvements in F1 scores and recall. While currently text-based, the framework supports future multimodal extensions (e.g., vocal tone, facial affect) for more holistic assessments. This work introduces a scalable, data-driven method for evaluating engagement quality of the therapy session, offering clinicians real-time feedback to enhance the quality of both virtual and in-person therapeutic interactions. 

---
# A Scaling Law for Token Efficiency in LLM Fine-Tuning Under Fixed Compute Budgets 

**Authors**: Ryan Lagasse, Aidan Kiernans, Avijit Ghosh, Shiri Dori-Hacohen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06150)  

**Abstract**: We introduce a scaling law for fine-tuning large language models (LLMs) under fixed compute budgets that explicitly accounts for data composition. Conventional approaches measure training data solely by total tokens, yet the number of examples and their average token length -- what we term \emph{dataset volume} -- play a decisive role in model performance. Our formulation is tuned following established procedures. Experiments on the BRICC dataset \cite{salavati2024reducing} and subsets of the MMLU dataset \cite{hendrycks2021measuringmassivemultitasklanguage}, evaluated under multiple subsampling strategies, reveal that data composition significantly affects token efficiency. These results motivate refined scaling laws for practical LLM fine-tuning in resource-constrained settings. 

---
# Can Prompting LLMs Unlock Hate Speech Detection across Languages? A Zero-shot and Few-shot Study 

**Authors**: Faeze Ghorbanpour, Daryna Dementieva, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2505.06149)  

**Abstract**: Despite growing interest in automated hate speech detection, most existing approaches overlook the linguistic diversity of online content. Multilingual instruction-tuned large language models such as LLaMA, Aya, Qwen, and BloomZ offer promising capabilities across languages, but their effectiveness in identifying hate speech through zero-shot and few-shot prompting remains underexplored. This work evaluates LLM prompting-based detection across eight non-English languages, utilizing several prompting techniques and comparing them to fine-tuned encoder models. We show that while zero-shot and few-shot prompting lag behind fine-tuned encoder models on most of the real-world evaluation sets, they achieve better generalization on functional tests for hate speech detection. Our study also reveals that prompt design plays a critical role, with each language often requiring customized prompting techniques to maximize performance. 

---
# Towards Robust Few-Shot Text Classification Using Transformer Architectures and Dual Loss Strategies 

**Authors**: Xu Han, Yumeng Sun, Weiqiang Huang, Hongye Zheng, Junliang Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.06145)  

**Abstract**: Few-shot text classification has important application value in low-resource environments. This paper proposes a strategy that combines adaptive fine-tuning, contrastive learning, and regularization optimization to improve the classification performance of Transformer-based models. Experiments on the FewRel 2.0 dataset show that T5-small, DeBERTa-v3, and RoBERTa-base perform well in few-shot tasks, especially in the 5-shot setting, which can more effectively capture text features and improve classification accuracy. The experiment also found that there are significant differences in the classification difficulty of different relationship categories. Some categories have fuzzy semantic boundaries or complex feature distributions, making it difficult for the standard cross entropy loss to learn the discriminative information required to distinguish categories. By introducing contrastive loss and regularization loss, the generalization ability of the model is enhanced, effectively alleviating the overfitting problem in few-shot environments. In addition, the research results show that the use of Transformer models or generative architectures with stronger self-attention mechanisms can help improve the stability and accuracy of few-shot classification. 

---
# LLMs Get Lost In Multi-Turn Conversation 

**Authors**: Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, Jennifer Neville  

**Link**: [PDF](https://arxiv.org/pdf/2505.06120)  

**Abstract**: Large Language Models (LLMs) are conversational interfaces. As such, LLMs have the potential to assist their users not only when they can fully specify the task at hand, but also to help them define, explore, and refine what they need through multi-turn conversational exchange. Although analysis of LLM conversation logs has confirmed that underspecification occurs frequently in user instructions, LLM evaluation has predominantly focused on the single-turn, fully-specified instruction setting. In this work, we perform large-scale simulation experiments to compare LLM performance in single- and multi-turn settings. Our experiments confirm that all the top open- and closed-weight LLMs we test exhibit significantly lower performance in multi-turn conversations than single-turn, with an average drop of 39% across six generation tasks. Analysis of 200,000+ simulated conversations decomposes the performance degradation into two components: a minor loss in aptitude and a significant increase in unreliability. We find that LLMs often make assumptions in early turns and prematurely attempt to generate final solutions, on which they overly rely. In simpler terms, we discover that *when LLMs take a wrong turn in a conversation, they get lost and do not recover*. 

---
# Multimodal Sentiment Analysis on CMU-MOSEI Dataset using Transformer-based Models 

**Authors**: Jugal Gajjar, Kaustik Ranaware  

**Link**: [PDF](https://arxiv.org/pdf/2505.06110)  

**Abstract**: This project performs multimodal sentiment analysis using the CMU-MOSEI dataset, using transformer-based models with early fusion to integrate text, audio, and visual modalities. We employ BERT-based encoders for each modality, extracting embeddings that are concatenated before classification. The model achieves strong performance, with 97.87\% 7-class accuracy and a 0.9682 F1-score on the test set, demonstrating the effectiveness of early fusion in capturing cross-modal interactions. The training utilized Adam optimization (lr=1e-4), dropout (0.3), and early stopping to ensure generalization and robustness. Results highlight the superiority of transformer architectures in modeling multimodal sentiment, with a low MAE (0.1060) indicating precise sentiment intensity prediction. Future work may compare fusion strategies or enhance interpretability. This approach utilizes multimodal learning by effectively combining linguistic, acoustic, and visual cues for sentiment analysis. 

---
# Attention on Multiword Expressions: A Multilingual Study of BERT-based Models with Regard to Idiomaticity and Microsyntax 

**Authors**: Iuliia Zaitova, Vitalii Hirak, Badr M. Abdullah, Dietrich Klakow, Bernd Möbius, Tania Avgustinova  

**Link**: [PDF](https://arxiv.org/pdf/2505.06062)  

**Abstract**: This study analyzes the attention patterns of fine-tuned encoder-only models based on the BERT architecture (BERT-based models) towards two distinct types of Multiword Expressions (MWEs): idioms and microsyntactic units (MSUs). Idioms present challenges in semantic non-compositionality, whereas MSUs demonstrate unconventional syntactic behavior that does not conform to standard grammatical categorizations. We aim to understand whether fine-tuning BERT-based models on specific tasks influences their attention to MWEs, and how this attention differs between semantic and syntactic tasks. We examine attention scores to MWEs in both pre-trained and fine-tuned BERT-based models. We utilize monolingual models and datasets in six Indo-European languages - English, German, Dutch, Polish, Russian, and Ukrainian. Our results show that fine-tuning significantly influences how models allocate attention to MWEs. Specifically, models fine-tuned on semantic tasks tend to distribute attention to idiomatic expressions more evenly across layers. Models fine-tuned on syntactic tasks show an increase in attention to MSUs in the lower layers, corresponding with syntactic processing requirements. 

---
# Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information 

**Authors**: Joshua Harris, Fan Grayson, Felix Feldman, Timothy Laurence, Toby Nonnenmacher, Oliver Higgins, Leo Loman, Selina Patel, Thomas Finnie, Samuel Collins, Michael Borowitz  

**Link**: [PDF](https://arxiv.org/pdf/2505.06046)  

**Abstract**: As Large Language Models (LLMs) become widely accessible, a detailed understanding of their knowledge within specific domains becomes necessary for successful real world use. This is particularly critical in public health, where failure to retrieve relevant, accurate, and current information could significantly impact UK residents. However, currently little is known about LLM knowledge of UK Government public health information. To address this issue, this paper introduces a new benchmark, PubHealthBench, with over 8000 questions for evaluating LLMs' Multiple Choice Question Answering (MCQA) and free form responses to public health queries, created via an automated pipeline. We also release a new dataset of the extracted UK Government public health guidance documents used as source text for PubHealthBench. Assessing 24 LLMs on PubHealthBench we find the latest private LLMs (GPT-4.5, GPT-4.1 and o1) have a high degree of knowledge, achieving >90% in the MCQA setup, and outperform humans with cursory search engine use. However, in the free form setup we see lower performance with no model scoring >75%. Therefore, whilst there are promising signs that state of the art (SOTA) LLMs are an increasingly accurate source of public health information, additional safeguards or tools may still be needed when providing free form responses on public health topics. 

---
# Unilogit: Robust Machine Unlearning for LLMs Using Uniform-Target Self-Distillation 

**Authors**: Stefan Vasilev, Christian Herold, Baohao Liao, Seyyed Hadi Hashemi, Shahram Khadivi, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2505.06027)  

**Abstract**: This paper introduces Unilogit, a novel self-distillation method for machine unlearning in Large Language Models. Unilogit addresses the challenge of selectively forgetting specific information while maintaining overall model utility, a critical task in compliance with data privacy regulations like GDPR. Unlike prior methods that rely on static hyperparameters or starting model outputs, Unilogit dynamically adjusts target logits to achieve a uniform probability for the target token, leveraging the current model's outputs for more accurate self-distillation targets. This approach not only eliminates the need for additional hyperparameters but also enhances the model's ability to approximate the golden targets. Extensive experiments on public benchmarks and an in-house e-commerce dataset demonstrate Unilogit's superior performance in balancing forget and retain objectives, outperforming state-of-the-art methods such as NPO and UnDIAL. Our analysis further reveals Unilogit's robustness across various scenarios, highlighting its practical applicability and effectiveness in achieving efficacious machine unlearning. 

---
# Do Not Change Me: On Transferring Entities Without Modification in Neural Machine Translation -- a Multilingual Perspective 

**Authors**: Dawid Wisniewski, Mikolaj Pokrywka, Zofia Rostek  

**Link**: [PDF](https://arxiv.org/pdf/2505.06010)  

**Abstract**: Current machine translation models provide us with high-quality outputs in most scenarios. However, they still face some specific problems, such as detecting which entities should not be changed during translation. In this paper, we explore the abilities of popular NMT models, including models from the OPUS project, Google Translate, MADLAD, and EuroLLM, to preserve entities such as URL addresses, IBAN numbers, or emails when producing translations between four languages: English, German, Polish, and Ukrainian. We investigate the quality of popular NMT models in terms of accuracy, discuss errors made by the models, and examine the reasons for errors. Our analysis highlights specific categories, such as emojis, that pose significant challenges for many models considered. In addition to the analysis, we propose a new multilingual synthetic dataset of 36,000 sentences that can help assess the quality of entity transfer across nine categories and four aforementioned languages. 

---
# Exploring the Feasibility of Multilingual Grammatical Error Correction with a Single LLM up to 9B parameters: A Comparative Study of 17 Models 

**Authors**: Dawid Wisniewski, Antoni Solarski, Artur Nowakowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.06004)  

**Abstract**: Recent language models can successfully solve various language-related tasks, and many understand inputs stated in different languages. In this paper, we explore the performance of 17 popular models used to correct grammatical issues in texts stated in English, German, Italian, and Swedish when using a single model to correct texts in all those languages. We analyze the outputs generated by these models, focusing on decreasing the number of grammatical errors while keeping the changes small. The conclusions drawn help us understand what problems occur among those models and which models can be recommended for multilingual grammatical error correction tasks. We list six models that improve grammatical correctness in all four languages and show that Gemma 9B is currently the best performing one for the languages considered. 

---
# An Exploratory Analysis on the Explanatory Potential of Embedding-Based Measures of Semantic Transparency for Malay Word Recognition 

**Authors**: M. Maziyah Mohamed, R. H. Baayen  

**Link**: [PDF](https://arxiv.org/pdf/2505.05973)  

**Abstract**: Studies of morphological processing have shown that semantic transparency is crucial for word recognition. Its computational operationalization is still under discussion. Our primary objectives are to explore embedding-based measures of semantic transparency, and assess their impact on reading. First, we explored the geometry of complex words in semantic space. To do so, we conducted a t-distributed Stochastic Neighbor Embedding clustering analysis on 4,226 Malay prefixed words. Several clusters were observed for complex words varied by their prefix class. Then, we derived five simple measures, and investigated whether they were significant predictors of lexical decision latencies. Two sets of Linear Discriminant Analyses were run in which the prefix of a word is predicted from either word embeddings or shift vectors (i.e., a vector subtraction of the base word from the derived word). The accuracy with which the model predicts the prefix of a word indicates the degree of transparency of the prefix. Three further measures were obtained by comparing embeddings between each word and all other words containing the same prefix (i.e., centroid), between each word and the shift from their base word, and between each word and the predicted word of the Functional Representations of Affixes in Compositional Semantic Space model. In a series of Generalized Additive Mixed Models, all measures predicted decision latencies after accounting for word frequency, word length, and morphological family size. The model that included the correlation between each word and their centroid as a predictor provided the best fit to the data. 

---
# Towards Developmentally Plausible Rewards: Communicative Success as a Learning Signal for Interactive Language Models 

**Authors**: Lennart Stöpler, Rufat Asadli, Mitja Nikolaus, Ryan Cotterell, Alex Warstadt  

**Link**: [PDF](https://arxiv.org/pdf/2505.05970)  

**Abstract**: We propose a method for training language models in an interactive setting inspired by child language acquisition. In our setting, a speaker attempts to communicate some information to a listener in a single-turn dialogue and receives a reward if communicative success is achieved. Unlike earlier related work using image--caption data for interactive reference games, we operationalize communicative success in a more abstract language-only question--answering setting. First, we present a feasibility study demonstrating that our reward provides an indirect signal about grammaticality. Second, we conduct experiments using reinforcement learning to fine-tune language models. We observe that cognitively plausible constraints on the communication channel lead to interpretable changes in speaker behavior. However, we do not yet see improvements on linguistic evaluations from our training regime. We outline potential modifications to the task design and training configuration that could better position future work to use our methodology to observe the benefits of interaction on language learning in computational cognitive models. 

---
# NeoQA: Evidence-based Question Answering with Generated News Events 

**Authors**: Max Glockner, Xiang Jiang, Leonardo F. R. Ribeiro, Iryna Gurevych, Markus Dreyer  

**Link**: [PDF](https://arxiv.org/pdf/2505.05949)  

**Abstract**: Evaluating Retrieval-Augmented Generation (RAG) in large language models (LLMs) is challenging because benchmarks can quickly become stale. Questions initially requiring retrieval may become answerable from pretraining knowledge as newer models incorporate more recent information during pretraining, making it difficult to distinguish evidence-based reasoning from recall. We introduce NeoQA (News Events for Out-of-training Question Answering), a benchmark designed to address this issue. To construct NeoQA, we generated timelines and knowledge bases of fictional news events and entities along with news articles and Q\&A pairs to prevent LLMs from leveraging pretraining knowledge, ensuring that no prior evidence exists in their training data. We propose our dataset as a new platform for evaluating evidence-based question answering, as it requires LLMs to generate responses exclusively from retrieved evidence and only when sufficient evidence is available. NeoQA enables controlled evaluation across various evidence scenarios, including cases with missing or misleading details. Our findings indicate that LLMs struggle to distinguish subtle mismatches between questions and evidence, and suffer from short-cut reasoning when key information required to answer a question is missing from the evidence, underscoring key limitations in evidence-based reasoning. 

---
# Summarisation of German Judgments in conjunction with a Class-based Evaluation 

**Authors**: Bianca Steffes, Nils Torben Wiedemann, Alexander Gratz, Pamela Hochreither, Jana Elina Meyer, Katharina Luise Schilke  

**Link**: [PDF](https://arxiv.org/pdf/2505.05947)  

**Abstract**: The automated summarisation of long legal documents can be a great aid for legal experts in their daily work. We automatically create summaries (guiding principles) of German judgments by fine-tuning a decoder-based large language model. We enrich the judgments with information about legal entities before the training. For the evaluation of the created summaries, we define a set of evaluation classes which allows us to measure their language, pertinence, completeness and correctness. Our results show that employing legal entities helps the generative model to find the relevant content, but the quality of the created summaries is not yet sufficient for a use in practice. 

---
# Elastic Weight Consolidation for Full-Parameter Continual Pre-Training of Gemma2 

**Authors**: Vytenis Šliogeris, Povilas Daniušis, Artūras Nakvosas  

**Link**: [PDF](https://arxiv.org/pdf/2505.05946)  

**Abstract**: This technical report describes an experiment on autoregressive pre-training of Gemma2 2 billion parameter large language model (LLM) with 10\% on the Lithuanian language component of CulturaX from the point of view of continual learning. We apply elastic weight consolidation (EWC) to the full set of the model's parameters and investigate language understanding benchmarks, consisting of Arc, Belebele, Gsm8K, Hellaswag, MMLU, TruthfulQA, and Winogrande sets (both in English and Lithuanian versions), and perplexity benchmarks. We empirically demonstrate that EWC regularisation allows us not only to mitigate catastrophic forgetting effects but also that it is potentially beneficial for learning of the new task with LLMs. 

---
# Symbol-based entity marker highlighting for enhanced text mining in materials science with generative AI 

**Authors**: Junhyeong Lee, Jong Min Yuk, Chan-Woo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.05864)  

**Abstract**: The construction of experimental datasets is essential for expanding the scope of data-driven scientific discovery. Recent advances in natural language processing (NLP) have facilitated automatic extraction of structured data from unstructured scientific literature. While existing approaches-multi-step and direct methods-offer valuable capabilities, they also come with limitations when applied independently. Here, we propose a novel hybrid text-mining framework that integrates the advantages of both methods to convert unstructured scientific text into structured data. Our approach first transforms raw text into entity-recognized text, and subsequently into structured form. Furthermore, beyond the overall data structuring framework, we also enhance entity recognition performance by introducing an entity marker-a simple yet effective technique that uses symbolic annotations to highlight target entities. Specifically, our entity marker-based hybrid approach not only consistently outperforms previous entity recognition approaches across three benchmark datasets (MatScholar, SOFC, and SOFC slot NER) but also improve the quality of final structured data-yielding up to a 58% improvement in entity-level F1 score and up to 83% improvement in relation-level F1 score compared to direct approach. 

---
# Tell Me Who Your Students Are: GPT Can Generate Valid Multiple-Choice Questions When Students' (Mis)Understanding Is Hinted 

**Authors**: Machi Shimmei, Masaki Uto, Yuichiroh Matsubayashi, Kentaro Inui, Aditi Mallavarapu, Noboru Matsuda  

**Link**: [PDF](https://arxiv.org/pdf/2505.05815)  

**Abstract**: The primary goal of this study is to develop and evaluate an innovative prompting technique, AnaQuest, for generating multiple-choice questions (MCQs) using a pre-trained large language model. In AnaQuest, the choice items are sentence-level assertions about complex concepts. The technique integrates formative and summative assessments. In the formative phase, students answer open-ended questions for target concepts in free text. For summative assessment, AnaQuest analyzes these responses to generate both correct and incorrect assertions. To evaluate the validity of the generated MCQs, Item Response Theory (IRT) was applied to compare item characteristics between MCQs generated by AnaQuest, a baseline ChatGPT prompt, and human-crafted items. An empirical study found that expert instructors rated MCQs generated by both AI models to be as valid as those created by human instructors. However, IRT-based analysis revealed that AnaQuest-generated questions - particularly those with incorrect assertions (foils) - more closely resembled human-crafted items in terms of difficulty and discrimination than those produced by ChatGPT. 

---
# Sparse Attention Remapping with Clustering for Efficient LLM Decoding on PIM 

**Authors**: Zehao Fan, Garrett Gagnon, Zhenyu Liu, Liu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05772)  

**Abstract**: Transformer-based models are the foundation of modern machine learning, but their execution, particularly during autoregressive decoding in large language models (LLMs), places significant pressure on memory systems due to frequent memory accesses and growing key-value (KV) caches. This creates a bottleneck in memory bandwidth, especially as context lengths increase. Processing-in-memory (PIM) architectures are a promising solution, offering high internal bandwidth and compute parallelism near memory. However, current PIM designs are primarily optimized for dense attention and struggle with the dynamic, irregular access patterns introduced by modern KV cache sparsity techniques. Consequently, they suffer from workload imbalance, reducing throughput and resource utilization. In this work, we propose STARC, a novel sparsity-optimized data mapping scheme tailored specifically for efficient LLM decoding on PIM architectures. STARC clusters KV pairs by semantic similarity and maps them to contiguous memory regions aligned with PIM bank structures. During decoding, queries retrieve relevant tokens at cluster granularity by matching against precomputed centroids, enabling selective attention and parallel processing without frequent reclustering or data movement overhead. Experiments on the HBM-PIM system show that, compared to common token-wise sparsity methods, STARC reduces attention-layer latency by 19%--31% and energy consumption by 19%--27%. Under a KV cache budget of 1024, it achieves up to 54%--74% latency reduction and 45%--67% energy reduction compared to full KV cache retrieval. Meanwhile, STARC maintains model accuracy comparable to state-of-the-art sparse attention methods, demonstrating its effectiveness in enabling efficient and hardware-friendly long-context LLM inference on PIM architectures. 

---
# Insertion Language Models: Sequence Generation with Arbitrary-Position Insertions 

**Authors**: Dhruvesh Patel, Aishwarya Sahoo, Avinash Amballa, Tahira Naseem, Tim G. J. Rudner, Andrew McCallum  

**Link**: [PDF](https://arxiv.org/pdf/2505.05755)  

**Abstract**: Autoregressive models (ARMs), which predict subsequent tokens one-by-one ``from left to right,'' have achieved significant success across a wide range of sequence generation tasks. However, they struggle to accurately represent sequences that require satisfying sophisticated constraints or whose sequential dependencies are better addressed by out-of-order generation. Masked Diffusion Models (MDMs) address some of these limitations, but the process of unmasking multiple tokens simultaneously in MDMs can introduce incoherences, and MDMs cannot handle arbitrary infilling constraints when the number of tokens to be filled in is not known in advance. In this work, we introduce Insertion Language Models (ILMs), which learn to insert tokens at arbitrary positions in a sequence -- that is, they select jointly both the position and the vocabulary element to be inserted. By inserting tokens one at a time, ILMs can represent strong dependencies between tokens, and their ability to generate sequences in arbitrary order allows them to accurately model sequences where token dependencies do not follow a left-to-right sequential structure. To train ILMs, we propose a tailored network parameterization and use a simple denoising objective. Our empirical evaluation demonstrates that ILMs outperform both ARMs and MDMs on common planning tasks. Furthermore, we show that ILMs outperform MDMs and perform on par with ARMs in an unconditional text generation task while offering greater flexibility than MDMs in arbitrary-length text infilling. 

---
# TopicVD: A Topic-Based Dataset of Video-Guided Multimodal Machine Translation for Documentaries 

**Authors**: Jinze Lv, Jian Chen, Zi Long, Xianghua Fu, Yin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.05714)  

**Abstract**: Most existing multimodal machine translation (MMT) datasets are predominantly composed of static images or short video clips, lacking extensive video data across diverse domains and topics. As a result, they fail to meet the demands of real-world MMT tasks, such as documentary translation. In this study, we developed TopicVD, a topic-based dataset for video-supported multimodal machine translation of documentaries, aiming to advance research in this field. We collected video-subtitle pairs from documentaries and categorized them into eight topics, such as economy and nature, to facilitate research on domain adaptation in video-guided MMT. Additionally, we preserved their contextual information to support research on leveraging the global context of documentaries in video-guided MMT. To better capture the shared semantics between text and video, we propose an MMT model based on a cross-modal bidirectional attention module. Extensive experiments on the TopicVD dataset demonstrate that visual information consistently improves the performance of the NMT model in documentary translation. However, the MMT model's performance significantly declines in out-of-domain scenarios, highlighting the need for effective domain adaptation methods. Additionally, experiments demonstrate that global context can effectively improve translation performance. % Dataset and our implementations are available at this https URL 

---
# Assessing Robustness to Spurious Correlations in Post-Training Language Models 

**Authors**: Julia Shuieh, Prasann Singhal, Apaar Shanker, John Heyer, George Pu, Samuel Denton  

**Link**: [PDF](https://arxiv.org/pdf/2505.05704)  

**Abstract**: Supervised and preference-based fine-tuning techniques have become popular for aligning large language models (LLMs) with user intent and correctness criteria. However, real-world training data often exhibits spurious correlations -- arising from biases, dataset artifacts, or other "shortcut" features -- that can compromise a model's performance or generalization. In this paper, we systematically evaluate three post-training algorithms -- Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and KTO (Kahneman-Tversky Optimization) -- across a diverse set of synthetic tasks and spuriousness conditions. Our tasks span mathematical reasoning, constrained instruction-following, and document-grounded question answering. We vary the degree of spurious correlation (10% vs. 90%) and investigate two forms of artifacts: "Feature Ambiguity" and "Distributional Narrowness." Our results show that the models often but not always degrade under higher spuriousness. The preference-based methods (DPO/KTO) can demonstrate relative robustness in mathematical reasoning tasks. By contrast, SFT maintains stronger performance in complex, context-intensive tasks. These findings highlight that no single post-training strategy universally outperforms in all scenarios; the best choice depends on the type of target task and the nature of spurious correlations. 

---
# Exploration of COVID-19 Discourse on Twitter: American Politician Edition 

**Authors**: Cindy Kim, Daniela Puchall, Jiangyi Liang, Jiwon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.05687)  

**Abstract**: The advent of the COVID-19 pandemic has undoubtedly affected the political scene worldwide and the introduction of new terminology and public opinions regarding the virus has further polarized partisan stances. Using a collection of tweets gathered from leading American political figures online (Republican and Democratic), we explored the partisan differences in approach, response, and attitude towards handling the international crisis. Implementation of the bag-of-words, bigram, and TF-IDF models was used to identify and analyze keywords, topics, and overall sentiments from each party. Results suggest that Democrats are more concerned with the casualties of the pandemic, and give more medical precautions and recommendations to the public whereas Republicans are more invested in political responsibilities such as keeping the public updated through media and carefully watching the progress of the virus. We propose a systematic approach to predict and distinguish a tweet's political stance (left or right leaning) based on its COVID-19 related terms using different classification algorithms on different language models. 

---
# Privacy-Preserving Transformers: SwiftKey's Differential Privacy Implementation 

**Authors**: Abdelrahman Abouelenin, Mohamed Abdelrehim, Raffy Fahim, Amr Hendy, Mohamed Afify  

**Link**: [PDF](https://arxiv.org/pdf/2505.05648)  

**Abstract**: In this paper we train a transformer using differential privacy (DP) for language modeling in SwiftKey. We run multiple experiments to balance the trade-off between the model size, run-time speed and accuracy. We show that we get small and consistent gains in the next-word-prediction and accuracy with graceful increase in memory and speed compared to the production GRU. This is obtained by scaling down a GPT2 architecture to fit the required size and a two stage training process that builds a seed model on general data and DP finetunes it on typing data. The transformer is integrated using ONNX offering both flexibility and efficiency. 

---
# KG-HTC: Integrating Knowledge Graphs into LLMs for Effective Zero-shot Hierarchical Text Classification 

**Authors**: Qianbo Zang, Christophe Zgrzendek, Igor Tchappi, Afshin Khadangi, Johannes Sedlmeir  

**Link**: [PDF](https://arxiv.org/pdf/2505.05583)  

**Abstract**: Hierarchical Text Classification (HTC) involves assigning documents to labels organized within a taxonomy. Most previous research on HTC has focused on supervised methods. However, in real-world scenarios, employing supervised HTC can be challenging due to a lack of annotated data. Moreover, HTC often faces issues with large label spaces and long-tail distributions. In this work, we present Knowledge Graphs for zero-shot Hierarchical Text Classification (KG-HTC), which aims to address these challenges of HTC in applications by integrating knowledge graphs with Large Language Models (LLMs) to provide structured semantic context during classification. Our method retrieves relevant subgraphs from knowledge graphs related to the input text using a Retrieval-Augmented Generation (RAG) approach. Our KG-HTC can enhance LLMs to understand label semantics at various hierarchy levels. We evaluate KG-HTC on three open-source HTC datasets: WoS, DBpedia, and Amazon. Our experimental results show that KG-HTC significantly outperforms three baselines in the strict zero-shot setting, particularly achieving substantial improvements at deeper levels of the hierarchy. This evaluation demonstrates the effectiveness of incorporating structured knowledge into LLMs to address HTC's challenges in large label spaces and long-tailed label distributions. Our code is available at: this https URL. 

---
# Neuro-Symbolic Concepts 

**Authors**: Jiayuan Mao, Joshua B. Tenenbaum, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06191)  

**Abstract**: This article presents a concept-centric paradigm for building agents that can learn continually and reason flexibly. The concept-centric agent utilizes a vocabulary of neuro-symbolic concepts. These concepts, such as object, relation, and action concepts, are grounded on sensory inputs and actuation outputs. They are also compositional, allowing for the creation of novel concepts through their structural combination. To facilitate learning and reasoning, the concepts are typed and represented using a combination of symbolic programs and neural network representations. Leveraging such neuro-symbolic concepts, the agent can efficiently learn and recombine them to solve various tasks across different domains, ranging from 2D images, videos, 3D scenes, and robotic manipulation tasks. This concept-centric framework offers several advantages, including data efficiency, compositional generalization, continual learning, and zero-shot transfer. 

---
# From Millions of Tweets to Actionable Insights: Leveraging LLMs for User Profiling 

**Authors**: Vahid Rahimzadeh, Ali Hamzehpour, Azadeh Shakery, Masoud Asadpour  

**Link**: [PDF](https://arxiv.org/pdf/2505.06184)  

**Abstract**: Social media user profiling through content analysis is crucial for tasks like misinformation detection, engagement prediction, hate speech monitoring, and user behavior modeling. However, existing profiling techniques, including tweet summarization, attribute-based profiling, and latent representation learning, face significant limitations: they often lack transferability, produce non-interpretable features, require large labeled datasets, or rely on rigid predefined categories that limit adaptability. We introduce a novel large language model (LLM)-based approach that leverages domain-defining statements, which serve as key characteristics outlining the important pillars of a domain as foundations for profiling. Our two-stage method first employs semi-supervised filtering with a domain-specific knowledge base, then generates both abstractive (synthesized descriptions) and extractive (representative tweet selections) user profiles. By harnessing LLMs' inherent knowledge with minimal human validation, our approach is adaptable across domains while reducing the need for large labeled datasets. Our method generates interpretable natural language user profiles, condensing extensive user data into a scale that unlocks LLMs' reasoning and knowledge capabilities for downstream social network tasks. We contribute a Persian political Twitter (X) dataset and an LLM-based evaluation framework with human validation. Experimental results show our method significantly outperforms state-of-the-art LLM-based and traditional methods by 9.8%, demonstrating its effectiveness in creating flexible, adaptable, and interpretable user profiles. 

---
# Differentiating Emigration from Return Migration of Scholars Using Name-Based Nationality Detection Models 

**Authors**: Faeze Ghorbanpour, Thiago Zordan Malaguth, Aliakbar Akbaritabar  

**Link**: [PDF](https://arxiv.org/pdf/2505.06107)  

**Abstract**: Most web and digital trace data do not include information about an individual's nationality due to privacy concerns. The lack of data on nationality can create challenges for migration research. It can lead to a left-censoring issue since we are uncertain about the migrant's country of origin. Once we observe an emigration event, if we know the nationality, we can differentiate it from return migration. We propose methods to detect the nationality with the least available data, i.e., full names. We use the detected nationality in comparison with the country of academic origin, which is a common approach in studying the migration of researchers. We gathered 2.6 million unique name-nationality pairs from Wikipedia and categorized them into families of nationalities with three granularity levels to use as our training data. Using a character-based machine learning model, we achieved a weighted F1 score of 84% for the broadest and 67% for the most granular, country-level categorization. In our empirical study, we used the trained and tested model to assign nationality to 8+ million scholars' full names in Scopus data. Our results show that using the country of first publication as a proxy for nationality underestimates the size of return flows, especially for countries with a more diverse academic workforce, such as the USA, Australia, and Canada. We found that around 48% of emigration from the USA was return migration once we used the country of name origin, in contrast to 33% based on academic origin. In the most recent period, 79% of scholars whose affiliation has consistently changed from the USA to China, and are considered emigrants, have Chinese names in contrast to 41% with a Chinese academic origin. Our proposed methods for addressing left-censoring issues are beneficial for other research that uses digital trace data to study migration. 

---
# Short-circuiting Shortcuts: Mechanistic Investigation of Shortcuts in Text Classification 

**Authors**: Leon Eshuijs, Shihan Wang, Antske Fokkens  

**Link**: [PDF](https://arxiv.org/pdf/2505.06032)  

**Abstract**: Reliance on spurious correlations (shortcuts) has been shown to underlie many of the successes of language models. Previous work focused on identifying the input elements that impact prediction. We investigate how shortcuts are actually processed within the model's decision-making mechanism. We use actor names in movie reviews as controllable shortcuts with known impact on the outcome. We use mechanistic interpretability methods and identify specific attention heads that focus on shortcuts. These heads gear the model towards a label before processing the complete input, effectively making premature decisions that bypass contextual analysis. Based on these findings, we introduce Head-based Token Attribution (HTA), which traces intermediate decisions back to input tokens. We show that HTA is effective in detecting shortcuts in LLMs and enables targeted mitigation by selectively deactivating shortcut-related attention heads. 

---
# Evolutionary ecology of words 

**Authors**: Reiji Suzuki, Takaya Arita  

**Link**: [PDF](https://arxiv.org/pdf/2505.05863)  

**Abstract**: We propose a model for the evolutionary ecology of words as one attempt to extend evolutionary game theory and agent-based models by utilizing the rich linguistic expressions of Large Language Models (LLMs). Our model enables the emergence and evolution of diverse and infinite options for interactions among agents. Within the population, each agent possesses a short word (or phrase) generated by an LLM and moves within a spatial environment. When agents become adjacent, the outcome of their interaction is determined by the LLM based on the relationship between their words, with the loser's word being replaced by the winner's. Word mutations, also based on LLM outputs, may occur. We conducted preliminary experiments assuming that ``strong animal species" would survive. The results showed that from an initial population consisting of well-known species, many species emerged both gradually and in a punctuated equilibrium manner. Each trial demonstrated the unique evolution of diverse populations, with one type of large species becoming dominant, such as terrestrial animals, marine life, or extinct species, which were ecologically specialized and adapted ones across diverse extreme habitats. We also conducted a long-term experiment with a large population, demonstrating the emergence and coexistence of diverse species. 

---
# An empathic GPT-based chatbot to talk about mental disorders with Spanish teenagers 

**Authors**: Alba María Mármol-Romero, Manuel García-Vega, Miguel Ángel García-Cumbreras, Arturo Montejo-Ráez  

**Link**: [PDF](https://arxiv.org/pdf/2505.05828)  

**Abstract**: This paper presents a chatbot-based system to engage young Spanish people in the awareness of certain mental disorders through a self-disclosure technique. The study was carried out in a population of teenagers aged between 12 and 18 years. The dialogue engine mixes closed and open conversations, so certain controlled messages are sent to focus the chat on a specific disorder, which will change over time. Once a set of trial questions is answered, the system can initiate the conversation on the disorder under the focus according to the user's sensibility to that disorder, in an attempt to establish a more empathetic communication. Then, an open conversation based on the GPT-3 language model is initiated, allowing the user to express themselves with more freedom. The results show that these systems are of interest to young people and could help them become aware of certain mental disorders. 

---
# BMMDetect: A Multimodal Deep Learning Framework for Comprehensive Biomedical Misconduct Detection 

**Authors**: Yize Zhou, Jie Zhang, Meijie Wang, Lun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05763)  

**Abstract**: Academic misconduct detection in biomedical research remains challenging due to algorithmic narrowness in existing methods and fragmented analytical pipelines. We present BMMDetect, a multimodal deep learning framework that integrates journal metadata (SJR, institutional data), semantic embeddings (PubMedBERT), and GPT-4o-mined textual attributes (methodological statistics, data anomalies) for holistic manuscript evaluation. Key innovations include: (1) multimodal fusion of domain-specific features to reduce detection bias; (2) quantitative evaluation of feature importance, identifying journal authority metrics (e.g., SJR-index) and textual anomalies (e.g., statistical outliers) as dominant predictors; and (3) the BioMCD dataset, a large-scale benchmark with 13,160 retracted articles and 53,411 controls. BMMDetect achieves 74.33% AUC, outperforming single-modality baselines by 8.6%, and demonstrates transferability across biomedical subfields. This work advances scalable, interpretable tools for safeguarding research integrity. 

---
# Harnessing LLMs Explanations to Boost Surrogate Models in Tabular Data Classification 

**Authors**: Ruxue Shi, Hengrui Gu, Xu Shen, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05744)  

**Abstract**: Large Language Models (LLMs) have shown remarkable ability in solving complex tasks, making them a promising tool for enhancing tabular learning. However, existing LLM-based methods suffer from high resource requirements, suboptimal demonstration selection, and limited interpretability, which largely hinder their prediction performance and application in the real world. To overcome these problems, we propose a novel in-context learning framework for tabular prediction. The core idea is to leverage the explanations generated by LLMs to guide a smaller, locally deployable Surrogate Language Model (SLM) to make interpretable tabular predictions. Specifically, our framework mainly involves three stages: (i) Post Hoc Explanation Generation, where LLMs are utilized to generate explanations for question-answer pairs in candidate demonstrations, providing insights into the reasoning behind the answer. (ii) Post Hoc Explanation-Guided Demonstrations Selection, which utilizes explanations generated by LLMs to guide the process of demonstration selection from candidate demonstrations. (iii) Post Hoc Explanation-Guided Interpretable SLM Prediction, which utilizes the demonstrations obtained in step (ii) as in-context and merges corresponding explanations as rationales to improve the performance of SLM and guide the model to generate interpretable outputs. Experimental results highlight the framework's effectiveness, with an average accuracy improvement of 5.31% across various tabular datasets in diverse domains. 

---
# Multimodal Integrated Knowledge Transfer to Large Language Models through Preference Optimization with Biomedical Applications 

**Authors**: Da Wu, Zhanliang Wang, Quan Nguyen, Zhuoran Xu, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05736)  

**Abstract**: The scarcity of high-quality multimodal biomedical data limits the ability to effectively fine-tune pretrained Large Language Models (LLMs) for specialized biomedical tasks. To address this challenge, we introduce MINT (Multimodal Integrated kNowledge Transfer), a framework that aligns unimodal large decoder models with domain-specific decision patterns from multimodal biomedical data through preference optimization. While MINT supports different optimization techniques, we primarily implement it with the Odds Ratio Preference Optimization (ORPO) framework as its backbone. This strategy enables the aligned LLMs to perform predictive tasks using text-only or image-only inputs while retaining knowledge learnt from multimodal data. MINT leverages an upstream multimodal machine learning (MML) model trained on high-quality multimodal data to transfer domain-specific insights to downstream text-only or image-only LLMs. We demonstrate its effectiveness through two key applications: (1) Rare genetic disease prediction from texts, where MINT uses a multimodal encoder model, trained on facial photos and clinical notes, to generate a preference dataset for aligning a lightweight Llama 3.2-3B-Instruct. Despite relying on text input only, the MINT-derived model outperforms models trained with SFT, RAG, or DPO, and even outperforms Llama 3.1-405B-Instruct. (2) Tissue type classification using cell nucleus images, where MINT uses a vision-language foundation model as the preference generator, containing knowledge learnt from both text and histopathological images to align downstream image-only models. The resulting MINT-derived model significantly improves the performance of Llama 3.2-Vision-11B-Instruct on tissue type classification. In summary, MINT provides an effective strategy to align unimodal LLMs with high-quality multimodal expertise through preference optimization. 

---
# Prompted Meta-Learning for Few-shot Knowledge Graph Completion 

**Authors**: Han Wu, Jie Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05684)  

**Abstract**: Few-shot knowledge graph completion (KGC) has obtained significant attention due to its practical applications in real-world scenarios, where new knowledge often emerges with limited available data. While most existing methods for few-shot KGC have predominantly focused on leveraging relational information, rich semantics inherent in KGs have been largely overlooked. To address this gap, we propose a novel prompted meta-learning (PromptMeta) framework that seamlessly integrates meta-semantics with relational information for few-shot KGC. PrompMeta has two key innovations: (1) a meta-semantic prompt pool that captures and consolidates high-level meta-semantics, enabling effective knowledge transfer and adaptation to rare and newly emerging relations. (2) a learnable fusion prompt that dynamically combines meta-semantic information with task-specific relational information tailored to different few-shot tasks. Both components are optimized together with model parameters within a meta-learning framework. Extensive experiments on two benchmark datasets demonstrate the effectiveness of our approach. 

---
# Adaptive Stress Testing Black-Box LLM Planners 

**Authors**: Neeloy Chakraborty, John Pohovey, Melkior Ornik, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2505.05665)  

**Abstract**: Large language models (LLMs) have recently demonstrated success in generalizing across decision-making tasks including planning, control and prediction, but their tendency to hallucinate unsafe and undesired outputs poses risks. We argue that detecting such failures is necessary, especially in safety-critical scenarios. Existing black-box methods often detect hallucinations by identifying inconsistencies across multiple samples. Many of these approaches typically introduce prompt perturbations like randomizing detail order or generating adversarial inputs, with the intuition that a confident model should produce stable outputs. We first perform a manual case study showing that other forms of perturbations (e.g., adding noise, removing sensor details) cause LLMs to hallucinate in a driving environment. We then propose a novel method for efficiently searching the space of prompt perturbations using Adaptive Stress Testing (AST) with Monte-Carlo Tree Search (MCTS). Our AST formulation enables discovery of scenarios and prompts that cause language models to act with high uncertainty. By generating MCTS prompt perturbation trees across diverse scenarios, we show that offline analyses can be used at runtime to automatically generate prompts that influence model uncertainty, and to inform real-time trust assessments of an LLM. 

---
# X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP 

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey  

**Link**: [PDF](https://arxiv.org/pdf/2505.05528)  

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{this https URL}{GitHub repository}. 

---
