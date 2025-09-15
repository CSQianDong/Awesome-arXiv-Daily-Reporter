# WhisTLE: Deeply Supervised, Text-Only Domain Adaptation for Pretrained Speech Recognition Transformers 

**Authors**: Akshat Pandey, Karun Kumar, Raphael Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10452)  

**Abstract**: Pretrained automatic speech recognition (ASR) models such as Whisper perform well but still need domain adaptation to handle unseen vocabulary and parlance. In many real-world settings, collecting speech data is impractical, necessitating text-only adaptation. We propose WhisTLE, a deeply supervised, text-only adaptation method for pretrained encoder-decoder ASR models. WhisTLE trains a variational autoencoder (VAE) to model encoder outputs from text and fine-tunes the decoder using the learned text-to-latent encoder, optionally combined with text-to-speech (TTS) adaptation. At inference, the original encoder is restored, incurring no extra runtime cost. Across four out-of-domain datasets and four ASR models, WhisTLE with TTS reduces word error rate (WER) by 12.3% relative to TTS-only adaptation and outperforms all non-WhisTLE baselines in 27 of 32 scenarios. 

---
# DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL 

**Authors**: Rui Lu, Zhenyu Hou, Zihan Wang, Hanchen Zhang, Xiao Liu, Yujiang Li, Shi Feng, Jie Tang, Yuxiao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.10446)  

**Abstract**: Augmenting large language models (LLMs) with browsing tools substantially improves their potential as deep search agents to solve complex, real-world tasks. Yet, open LLMs still perform poorly in such settings due to limited long-horizon reasoning capacity with browsing tools and the lack of sufficiently difficult supervised data. To address these challenges, we present DeepDive to advance deep search agents. First, we propose a strategy to automatically synthesize complex, difficult, and hard-to-find questions from open knowledge graphs. Second, we apply end-to-end multi-turn reinforcement learning (RL) to enhance LLMs' long-horizon reasoning with deep search. Experiments show that DeepDive-32B achieves a new open-source competitive result on BrowseComp, outperforming WebSailor, DeepSeek-R1-Browse, and Search-o1. We demonstrate that multi-turn RL training improves deep search ability and significantly contributes to the performance improvements across multiple benchmarks. We observe that DeepDive enables test-time scaling of tool calls and parallel sampling. All datasets, models, and code are publicly available at this https URL. 

---
# RefactorCoderQA: Benchmarking LLMs for Multi-Domain Coding Question Solutions in Cloud and Edge Deployment 

**Authors**: Shadikur Rahman, Aroosa Hameed, Gautam Srivastava, Syed Muhammad Danish  

**Link**: [PDF](https://arxiv.org/pdf/2509.10436)  

**Abstract**: To optimize the reasoning and problem-solving capabilities of Large Language Models (LLMs), we propose a novel cloud-edge collaborative architecture that enables a structured, multi-agent prompting framework. This framework comprises three specialized components: GuideLLM, a lightweight model deployed at the edge to provide methodological guidance; SolverLLM, a more powerful model hosted in the cloud responsible for generating code solutions; and JudgeLLM, an automated evaluator for assessing solution correctness and quality. To evaluate and demonstrate the effectiveness of this architecture in realistic settings, we introduce RefactorCoderQA, a comprehensive benchmark designed to evaluate and enhance the performance of Large Language Models (LLMs) across multi-domain coding tasks. Motivated by the limitations of existing benchmarks, RefactorCoderQA systematically covers various technical domains, including Software Engineering, Data Science, Machine Learning, and Natural Language Processing, using authentic coding challenges from Stack Overflow. Extensive experiments reveal that our fine-tuned model, RefactorCoder-MoE, achieves state-of-the-art performance, significantly outperforming leading open-source and commercial baselines with an overall accuracy of 76.84%. Human evaluations further validate the interpretability, accuracy, and practical relevance of the generated solutions. In addition, we evaluate system-level metrics, such as throughput and latency, to gain deeper insights into the performance characteristics and trade-offs of the proposed architecture. 

---
# Long Context Automated Essay Scoring with Language Models 

**Authors**: Christopher Ormerod, Gitit Kehat  

**Link**: [PDF](https://arxiv.org/pdf/2509.10417)  

**Abstract**: Transformer-based language models are architecturally constrained to process text of a fixed maximum length. Essays written by higher-grade students frequently exceed the maximum allowed length for many popular open-source models. A common approach to addressing this issue when using these models for Automated Essay Scoring is to truncate the input text. This raises serious validity concerns as it undermines the model's ability to fully capture and evaluate organizational elements of the scoring rubric, which requires long contexts to assess. In this study, we evaluate several models that incorporate architectural modifications of the standard transformer architecture to overcome these length limitations using the Kaggle ASAP 2.0 dataset. The models considered in this study include fine-tuned versions of XLNet, Longformer, ModernBERT, Mamba, and Llama models. 

---
# Is In-Context Learning Learning? 

**Authors**: Adrian de Wynter  

**Link**: [PDF](https://arxiv.org/pdf/2509.10414)  

**Abstract**: In-context learning (ICL) allows some autoregressive models to solve tasks via next-token prediction and without needing further training. This has led to claims about these model's ability to solve (learn) unseen tasks with only a few shots (exemplars) in the prompt. However, deduction does not always imply learning, as ICL does not explicitly encode a given observation. Instead, the models rely on their prior knowledge and the exemplars given, if any. We argue that, mathematically, ICL does constitute learning, but its full characterisation requires empirical work. We then carry out a large-scale analysis of ICL ablating out or accounting for memorisation, pretraining, distributional shifts, and prompting style and phrasing. We find that ICL is an effective learning paradigm, but limited in its ability to learn and generalise to unseen tasks. We note that, in the limit where exemplars become more numerous, accuracy is insensitive to exemplar distribution, model, prompt style, and the input's linguistic features. Instead, it deduces patterns from regularities in the prompt, which leads to distributional sensitivity, especially in prompting styles such as chain-of-thought. Given the varied accuracies on formally similar tasks, we conclude that autoregression's ad-hoc encoding is not a robust mechanism, and suggests limited all-purpose generalisability. 

---
# Dropping Experts, Recombining Neurons: Retraining-Free Pruning for Sparse Mixture-of-Experts LLMs 

**Authors**: Yixiao Zhou, Ziyu Zhao, Dongzhou Cheng, zhiliang wu, Jie Gui, Yi Yang, Fei Wu, Yu Cheng, Hehe Fan  

**Link**: [PDF](https://arxiv.org/pdf/2509.10377)  

**Abstract**: Sparse Mixture-of-Experts (SMoE) architectures are widely used in large language models (LLMs) due to their computational efficiency. However, though only a few experts are activated for each token, SMoE still requires loading all expert parameters, leading to high memory usage and challenges in deployment. Previous work has tried to reduce the overhead by pruning and merging experts, but primarily focused on expert-level operations, leaving neuron-level structure underexplored. We propose DERN (Dropping Experts, Recombining Neurons), a task-agnostic and retraining-free framework for expert pruning and reconstruction. We observe that experts are often misaligned and contain semantic conflicts at the neuron level, which poses challenges for direct merging. To solve this, DERN works in three steps: it first prunes redundant experts using router statistics; then it decomposes them into neuron-level expert segments, assigning each segment to its most compatible retained expert; and finally, it merges segments within each retained expert to build a compact representation. Experiments on Mixtral, Qwen, and DeepSeek SMoE models show that DERN improves performance by more than 5% on commonsense reasoning and MMLU benchmarks under 50% expert sparsity, without extra training. It also greatly reduces the number of experts and memory usage, making SMoE LLMs easier to deploy in practice. 

---
# SI-FACT: Mitigating Knowledge Conflict via Self-Improving Faithfulness-Aware Contrastive Tuning 

**Authors**: Shengqiang Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10208)  

**Abstract**: Large Language Models often generate unfaithful responses in knowledge intensive tasks due to knowledge conflict,that is,a preference for relying on internal parametric knowledge rather than the provided this http URL address this issue,we propose a novel self improving framework,Self Improving Faithfulness Aware Contrastive this http URL framework uses a self instruct mechanism that allows the base LLM to automatically generate high quality,structured contrastive learning data,including anchor samples,semantically equivalent positive samples,and negative samples simulating unfaithful this http URL approach significantly reduces the cost of manual this http URL,contrastive learning is applied to train the model,enabling it to pull faithful responses closer and push unfaithful responses farther apart in the representation this http URL on knowledge conflict evaluation benchmarks ECARE KRE and COSE KRE show that the SI FACT model based on Llama3 8B Instruct improves the Contextual Recall Rate by 6.2% over the best baseline method,while significantly reducing dependence on internal this http URL results indicate that SI FACT provides strong effectiveness and high data efficiency in enhancing the contextual faithfulness of LLMs,offering a practical pathway toward building more proactive and trustworthy language models. 

---
# Beyond Token Limits: Assessing Language Model Performance on Long Text Classification 

**Authors**: Miklós Sebők, Viktor Kovács, Martin Bánóczy, Daniel Møller Eriksen, Nathalie Neptune, Philippe Roussille  

**Link**: [PDF](https://arxiv.org/pdf/2509.10199)  

**Abstract**: The most widely used large language models in the social sciences (such as BERT, and its derivatives, e.g. RoBERTa) have a limitation on the input text length that they can process to produce predictions. This is a particularly pressing issue for some classification tasks, where the aim is to handle long input texts. One such area deals with laws and draft laws (bills), which can have a length of multiple hundred pages and, therefore, are not particularly amenable for processing with models that can only handle e.g. 512 tokens. In this paper, we show results from experiments covering 5 languages with XLM-RoBERTa, Longformer, GPT-3.5, GPT-4 models for the multiclass classification task of the Comparative Agendas Project, which has a codebook of 21 policy topic labels from education to health care. Results show no particular advantage for the Longformer model, pre-trained specifically for the purposes of handling long inputs. The comparison between the GPT variants and the best-performing open model yielded an edge for the latter. An analysis of class-level factors points to the importance of support and substance overlaps between specific categories when it comes to performance on long text inputs. 

---
# Incongruent Positivity: When Miscalibrated Positivity Undermines Online Supportive Conversations 

**Authors**: Leen Almajed, Abeer ALdayel  

**Link**: [PDF](https://arxiv.org/pdf/2509.10184)  

**Abstract**: In emotionally supportive conversations, well-intended positivity can sometimes misfire, leading to responses that feel dismissive, minimizing, or unrealistically optimistic. We examine this phenomenon of incongruent positivity as miscalibrated expressions of positive support in both human and LLM generated responses. To this end, we collected real user-assistant dialogues from Reddit across a range of emotional intensities and generated additional responses using large language models for the same context. We categorize these conversations by intensity into two levels: Mild, which covers relationship tension and general advice, and Severe, which covers grief and anxiety conversations. This level of categorization enables a comparative analysis of how supportive responses vary across lower and higher stakes contexts. Our analysis reveals that LLMs are more prone to unrealistic positivity through dismissive and minimizing tone, particularly in high-stakes contexts. To further study the underlying dimensions of this phenomenon, we finetune LLMs on datasets with strong and weak emotional reactions. Moreover, we developed a weakly supervised multilabel classifier ensemble (DeBERTa and MentalBERT) that shows improved detection of incongruent positivity types across two sorts of concerns (Mild and Severe). Our findings shed light on the need to move beyond merely generating generic positive responses and instead study the congruent support measures to balance positive affect with emotional acknowledgment. This approach offers insights into aligning large language models with affective expectations in the online supportive dialogue, paving the way toward context-aware and trust preserving online conversation systems. 

---
# Benchmark of stylistic variation in LLM-generated texts 

**Authors**: Jiří Milička, Anna Marklová, Václav Cvrček  

**Link**: [PDF](https://arxiv.org/pdf/2509.10179)  

**Abstract**: This study investigates the register variation in texts written by humans and comparable texts produced by large language models (LLMs). Biber's multidimensional analysis (MDA) is applied to a sample of human-written texts and AI-created texts generated to be their counterparts to find the dimensions of variation in which LLMs differ most significantly and most systematically from humans. As textual material, a new LLM-generated corpus AI-Brown is used, which is comparable to BE-21 (a Brown family corpus representing contemporary British English). Since all languages except English are underrepresented in the training data of frontier LLMs, similar analysis is replicated on Czech using AI-Koditex corpus and Czech multidimensional model. Examined were 16 frontier models in various settings and prompts, with emphasis placed on the difference between base models and instruction-tuned models. Based on this, a benchmark is created through which models can be compared with each other and ranked in interpretable dimensions. 

---
# Towards Reliable and Interpretable Document Question Answering via VLMs 

**Authors**: Alessio Chen, Simone Giovannini, Andrea Gemelli, Fabio Coppini, Simone Marinai  

**Link**: [PDF](https://arxiv.org/pdf/2509.10129)  

**Abstract**: Vision-Language Models (VLMs) have shown strong capabilities in document understanding, particularly in identifying and extracting textual information from complex documents. Despite this, accurately localizing answers within documents remains a major challenge, limiting both interpretability and real-world applicability. To address this, we introduce \textit{DocExplainerV0}, a plug-and-play bounding-box prediction module that decouples answer generation from spatial localization. This design makes it applicable to existing VLMs, including proprietary systems where fine-tuning is not feasible. Through systematic evaluation, we provide quantitative insights into the gap between textual accuracy and spatial grounding, showing that correct answers often lack reliable localization. Our standardized framework highlights these shortcomings and establishes a benchmark for future research toward more interpretable and robust document information extraction VLMs. 

---
# Population-Aligned Persona Generation for LLM-based Social Simulation 

**Authors**: Zhengyu Hu, Zheyuan Xiao, Max Xiong, Yuxuan Lei, Tianfu Wang, Jianxun Lian, Kaize Ding, Ziang Xiao, Nicholas Jing Yuan, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.10127)  

**Abstract**: Recent advances in large language models (LLMs) have enabled human-like social simulations at unprecedented scale and fidelity, offering new opportunities for computational social science. A key challenge, however, is the construction of persona sets that authentically represent the diversity and distribution of real-world populations. Most existing LLM-based social simulation studies focus primarily on designing agentic frameworks and simulation environments, often overlooking the complexities of persona generation and the potential biases introduced by unrepresentative persona sets. In this paper, we propose a systematic framework for synthesizing high-quality, population-aligned persona sets for LLM-driven social simulation. Our approach begins by leveraging LLMs to generate narrative personas from long-term social media data, followed by rigorous quality assessment to filter out low-fidelity profiles. We then apply importance sampling to achieve global alignment with reference psychometric distributions, such as the Big Five personality traits. To address the needs of specific simulation contexts, we further introduce a task-specific module that adapts the globally aligned persona set to targeted subpopulations. Extensive experiments demonstrate that our method significantly reduces population-level bias and enables accurate, flexible social simulation for a wide range of research and policy applications. 

---
# Prominence-aware automatic speech recognition for conversational speech 

**Authors**: Julian Linke, Barbara Schuppler  

**Link**: [PDF](https://arxiv.org/pdf/2509.10116)  

**Abstract**: This paper investigates prominence-aware automatic speech recognition (ASR) by combining prominence detection and speech recognition for conversational Austrian German. First, prominence detectors were developed by fine-tuning wav2vec2 models to classify word-level prominence. The detector was then used to automatically annotate prosodic prominence in a large corpus. Based on those annotations, we trained novel prominence-aware ASR systems that simultaneously transcribe words and their prominence levels. The integration of prominence information did not change performance compared to our baseline ASR system, while reaching a prominence detection accuracy of 85.53% for utterances where the recognized word sequence was correct. This paper shows that transformer-based models can effectively encode prosodic information and represents a novel contribution to prosody-enhanced ASR, with potential applications for linguistic research and prosody-informed dialogue systems. 

---
# Scaling Arabic Medical Chatbots Using Synthetic Data: Enhancing Generative AI with Synthetic Patient Records 

**Authors**: Abdulrahman Allam, Seif Ahmed, Ali Hamdi, Khaled Shaban  

**Link**: [PDF](https://arxiv.org/pdf/2509.10108)  

**Abstract**: The development of medical chatbots in Arabic is significantly constrained by the scarcity of large-scale, high-quality annotated datasets. While prior efforts compiled a dataset of 20,000 Arabic patient-doctor interactions from social media to fine-tune large language models (LLMs), model scalability and generalization remained limited. In this study, we propose a scalable synthetic data augmentation strategy to expand the training corpus to 100,000 records. Using advanced generative AI systems ChatGPT-4o and Gemini 2.5 Pro we generated 80,000 contextually relevant and medically coherent synthetic question-answer pairs grounded in the structure of the original dataset. These synthetic samples were semantically filtered, manually validated, and integrated into the training pipeline. We fine-tuned five LLMs, including Mistral-7B and AraGPT2, and evaluated their performance using BERTScore metrics and expert-driven qualitative assessments. To further analyze the effectiveness of synthetic sources, we conducted an ablation study comparing ChatGPT-4o and Gemini-generated data independently. The results showed that ChatGPT-4o data consistently led to higher F1-scores and fewer hallucinations across all models. Overall, our findings demonstrate the viability of synthetic augmentation as a practical solution for enhancing domain-specific language models in-low resource medical NLP, paving the way for more inclusive, scalable, and accurate Arabic healthcare chatbot systems. 

---
# Arabic Large Language Models for Medical Text Generation 

**Authors**: Abdulrahman Allam, Seif Ahmed, Ali Hamdi, Ammar Mohammed  

**Link**: [PDF](https://arxiv.org/pdf/2509.10095)  

**Abstract**: Efficient hospital management systems (HMS) are critical worldwide to address challenges such as overcrowding, limited resources, and poor availability of urgent health care. Existing methods often lack the ability to provide accurate, real-time medical advice, particularly for irregular inputs and underrepresented languages. To overcome these limitations, this study proposes an approach that fine-tunes large language models (LLMs) for Arabic medical text generation. The system is designed to assist patients by providing accurate medical advice, diagnoses, drug recommendations, and treatment plans based on user input. The research methodology required the collection of a unique dataset from social media platforms, capturing real-world medical conversations between patients and doctors. The dataset, which includes patient complaints together with medical advice, was properly cleaned and preprocessed to account for multiple Arabic dialects. Fine-tuning state-of-the-art generative models, such as Mistral-7B-Instruct-v0.2, LLaMA-2-7B, and GPT-2 Medium, optimized the system's ability to generate reliable medical text. Results from evaluations indicate that the fine-tuned Mistral-7B model outperformed the other models, achieving average BERT (Bidirectional Encoder Representations from Transformers) Score values in precision, recall, and F1-scores of 68.5\%, 69.08\%, and 68.5\%, respectively. Comparative benchmarking and qualitative assessments validate the system's ability to produce coherent and relevant medical replies to informal input. This study highlights the potential of generative artificial intelligence (AI) in advancing HMS, offering a scalable and adaptable solution for global healthcare challenges, especially in linguistically and culturally diverse environments. 

---
# Querying Climate Knowledge: Semantic Retrieval for Scientific Discovery 

**Authors**: Mustapha Adamu, Qi Zhang, Huitong Pan, Longin Jan Latecki, Eduard C. Dragut  

**Link**: [PDF](https://arxiv.org/pdf/2509.10087)  

**Abstract**: The growing complexity and volume of climate science literature make it increasingly difficult for researchers to find relevant information across models, datasets, regions, and variables. This paper introduces a domain-specific Knowledge Graph (KG) built from climate publications and broader scientific texts, aimed at improving how climate knowledge is accessed and used. Unlike keyword based search, our KG supports structured, semantic queries that help researchers discover precise connections such as which models have been validated in specific regions or which datasets are commonly used with certain teleconnection patterns. We demonstrate how the KG answers such questions using Cypher queries, and outline its integration with large language models in RAG systems to improve transparency and reliability in climate-related question answering. This work moves beyond KG construction to show its real world value for climate researchers, model developers, and others who rely on accurate, contextual scientific information. 

---
# Established Psychometric vs. Ecologically Valid Questionnaires: Rethinking Psychological Assessments in Large Language Models 

**Authors**: Dongmin Choi, Woojung Song, Jongwook Han, Eun-Ju Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2509.10078)  

**Abstract**: Researchers have applied established psychometric questionnaires (e.g., BFI, PVQ) to measure the personality traits and values reflected in the responses of Large Language Models (LLMs). However, concerns have been raised about applying these human-designed questionnaires to LLMs. One such concern is their lack of ecological validity--the extent to which survey questions adequately reflect and resemble real-world contexts in which LLMs generate texts in response to user queries. However, it remains unclear how established questionnaires and ecologically valid questionnaires differ in their outcomes, and what insights these differences may provide. In this paper, we conduct a comprehensive comparative analysis of the two types of questionnaires. Our analysis reveals that established questionnaires (1) yield substantially different profiles of LLMs from ecologically valid ones, deviating from the psychological characteristics expressed in the context of user queries, (2) suffer from insufficient items for stable measurement, (3) create misleading impressions that LLMs possess stable constructs, and (4) yield exaggerated profiles for persona-prompted LLMs. Overall, our work cautions against the use of established psychological questionnaires for LLMs. Our code will be released upon publication. 

---
# !MSA at BAREC Shared Task 2025: Ensembling Arabic Transformers for Readability Assessment 

**Authors**: Mohamed Basem, Mohamed Younes, Seif Ahmed, Abdelrahman Moustafa  

**Link**: [PDF](https://arxiv.org/pdf/2509.10040)  

**Abstract**: We present MSAs winning system for the BAREC 2025 Shared Task on fine-grained Arabic readability assessment, achieving first place in six of six tracks. Our approach is a confidence-weighted ensemble of four complementary transformer models (AraBERTv2, AraELECTRA, MARBERT, and CAMeLBERT) each fine-tuned with distinct loss functions to capture diverse readability signals. To tackle severe class imbalance and data scarcity, we applied weighted training, advanced preprocessing, SAMER corpus relabeling with our strongest model, and synthetic data generation via Gemini 2.5 Flash, adding about 10,000 rare-level samples. A targeted post-processing step corrected prediction distribution skew, delivering a 6.3 percent Quadratic Weighted Kappa (QWK) gain. Our system reached 87.5 percent QWK at the sentence level and 87.4 percent at the document level, demonstrating the power of model and loss diversity, confidence-informed fusion, and intelligent augmentation for robust Arabic readability prediction. 

---
# Linguistic trajectories of bipolar disorder on social media 

**Authors**: Laurin Plank, Armin Zlomuzica  

**Link**: [PDF](https://arxiv.org/pdf/2509.10035)  

**Abstract**: Language provides valuable markers of affective disorders such as bipolar disorder (BD), yet clinical assessments remain limited in scale. In response, analyses of social media (SM) language have gained prominence due to their high temporal resolution and longitudinal scope. Here, we introduce a method to determine the timing of users' diagnoses and apply it to study language trajectories from 3 years before to 21 years after BD diagnosis - contrasted with uses reporting unipolar depression (UD) and non-affected users (HC). We show that BD diagnosis is accompanied by pervasive linguistic alterations reflecting mood disturbance, psychiatric comorbidity, substance abuse, hospitalization, medical comorbidities, unusual thought content, and disorganized thought. We further observe recurring mood-related language changes across two decades after the diagnosis, with a pronounced 12-month periodicity suggestive of seasonal mood episodes. Finally, trend-level evidence suggests an increased periodicity in users estimated to be female. In sum, our findings provide evidence for language alterations in the acute and chronic phase of BD. This validates and extends recent efforts leveraging SM for scalable monitoring of mental health. 

---
# Multi-Intent Recognition in Dialogue Understanding: A Comparison Between Smaller Open-Source LLMs 

**Authors**: Adnan Ahmad, Philine Kowol, Stefan Hillmann, Sebastian Möller  

**Link**: [PDF](https://arxiv.org/pdf/2509.10010)  

**Abstract**: In this paper, we provide an extensive analysis of multi-label intent classification using Large Language Models (LLMs) that are open-source, publicly available, and can be run in consumer hardware. We use the MultiWOZ 2.1 dataset, a benchmark in the dialogue system domain, to investigate the efficacy of three popular open-source pre-trained LLMs, namely LLama2-7B-hf, Mistral-7B-v0.1, and Yi-6B. We perform the classification task in a few-shot setup, giving 20 examples in the prompt with some instructions. Our approach focuses on the differences in performance of these models across several performance metrics by methodically assessing these models on multi-label intent classification tasks. Additionally, we compare the performance of the instruction-based fine-tuning approach with supervised learning using the smaller transformer model BertForSequenceClassification as a baseline. To evaluate the performance of the models, we use evaluation metrics like accuracy, precision, and recall as well as micro, macro, and weighted F1 score. We also report the inference time, VRAM requirements, etc. The Mistral-7B-v0.1 outperforms two other generative models on 11 intent classes out of 14 in terms of F-Score, with a weighted average of 0.50. It also has relatively lower Humming Loss and higher Jaccard Similarity, making it the winning model in the few-shot setting. We find BERT based supervised classifier having superior performance compared to the best performing few-shot generative LLM. The study provides a framework for small open-source LLMs in detecting complex multi-intent dialogues, enhancing the Natural Language Understanding aspect of task-oriented chatbots. 

---
# Unsupervised Hallucination Detection by Inspecting Reasoning Processes 

**Authors**: Ponhvoan Srey, Xiaobao Wu, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10004)  

**Abstract**: Unsupervised hallucination detection aims to identify hallucinated content generated by large language models (LLMs) without relying on labeled data. While unsupervised methods have gained popularity by eliminating labor-intensive human annotations, they frequently rely on proxy signals unrelated to factual correctness. This misalignment biases detection probes toward superficial or non-truth-related aspects, limiting generalizability across datasets and scenarios. To overcome these limitations, we propose IRIS, an unsupervised hallucination detection framework, leveraging internal representations intrinsic to factual correctness. IRIS prompts the LLM to carefully verify the truthfulness of a given statement, and obtain its contextualized embedding as informative features for training. Meanwhile, the uncertainty of each response is considered a soft pseudolabel for truthfulness. Experimental results demonstrate that IRIS consistently outperforms existing unsupervised methods. Our approach is fully unsupervised, computationally low cost, and works well even with few training data, making it suitable for real-time detection. 

---
# CMHG: A Dataset and Benchmark for Headline Generation of Minority Languages in China 

**Authors**: Guixian Xu, Zeli Su, Ziyin Zhang, Jianing Liu, XU Han, Ting Zhang, Yushuang Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.09990)  

**Abstract**: Minority languages in China, such as Tibetan, Uyghur, and Traditional Mongolian, face significant challenges due to their unique writing systems, which differ from international standards. This discrepancy has led to a severe lack of relevant corpora, particularly for supervised tasks like headline generation. To address this gap, we introduce a novel dataset, Chinese Minority Headline Generation (CMHG), which includes 100,000 entries for Tibetan, and 50,000 entries each for Uyghur and Mongolian, specifically curated for headline generation tasks. Additionally, we propose a high-quality test set annotated by native speakers, designed to serve as a benchmark for future research in this domain. We hope this dataset will become a valuable resource for advancing headline generation in Chinese minority languages and contribute to the development of related benchmarks. 

---
# Large Language Models Meet Legal Artificial Intelligence: A Survey 

**Authors**: Zhitian Hou, Zihan Ye, Nanli Zeng, Tianyong Hao, Kun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.09969)  

**Abstract**: Large Language Models (LLMs) have significantly advanced the development of Legal Artificial Intelligence (Legal AI) in recent years, enhancing the efficiency and accuracy of legal tasks. To advance research and applications of LLM-based approaches in legal domain, this paper provides a comprehensive review of 16 legal LLMs series and 47 LLM-based frameworks for legal tasks, and also gather 15 benchmarks and 29 datasets to evaluate different legal capabilities. Additionally, we analyse the challenges and discuss future directions for LLM-based approaches in the legal domain. We hope this paper provides a systematic introduction for beginners and encourages future research in this field. Resources are available at this https URL. 

---
# Emulating Public Opinion: A Proof-of-Concept of AI-Generated Synthetic Survey Responses for the Chilean Case 

**Authors**: Bastián González-Bustamante, Nando Verelst, Carla Cisternas  

**Link**: [PDF](https://arxiv.org/pdf/2509.09871)  

**Abstract**: Large Language Models (LLMs) offer promising avenues for methodological and applied innovations in survey research by using synthetic respondents to emulate human answers and behaviour, potentially mitigating measurement and representation errors. However, the extent to which LLMs recover aggregate item distributions remains uncertain and downstream applications risk reproducing social stereotypes and biases inherited from training data. We evaluate the reliability of LLM-generated synthetic survey responses against ground-truth human responses from a Chilean public opinion probabilistic survey. Specifically, we benchmark 128 prompt-model-question triplets, generating 189,696 synthetic profiles, and pool performance metrics (i.e., accuracy, precision, recall, and F1-score) in a meta-analysis across 128 question-subsample pairs to test for biases along key sociodemographic dimensions. The evaluation spans OpenAI's GPT family and o-series reasoning models, as well as Llama and Qwen checkpoints. Three results stand out. First, synthetic responses achieve excellent performance on trust items (F1-score and accuracy > 0.90). Second, GPT-4o, GPT-4o-mini and Llama 4 Maverick perform comparably on this task. Third, synthetic-human alignment is highest among respondents aged 45-59. Overall, LLM-based synthetic samples approximate responses from a probabilistic sample, though with substantial item-level heterogeneity. Capturing the full nuance of public opinion remains challenging and requires careful calibration and additional distributional tests to ensure algorithmic fidelity and reduce errors. 

---
# Topic-Guided Reinforcement Learning with LLMs for Enhancing Multi-Document Summarization 

**Authors**: Chuyuan Li, Austin Xu, Shafiq Joty, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2509.09852)  

**Abstract**: A key challenge in Multi-Document Summarization (MDS) is effectively integrating information from multiple sources while maintaining coherence and topical relevance. While Large Language Models have shown impressive results in single-document summarization, their performance on MDS still leaves room for improvement. In this paper, we propose a topic-guided reinforcement learning approach to improve content selection in MDS. We first show that explicitly prompting models with topic labels enhances the informativeness of the generated summaries. Building on this insight, we propose a novel topic reward within the Group Relative Policy Optimization (GRPO) framework to measure topic alignment between the generated summary and source documents. Experimental results on the Multi-News and Multi-XScience datasets demonstrate that our method consistently outperforms strong baselines, highlighting the effectiveness of leveraging topical cues in MDS. 

---
# Pragmatic Frames Evoked by Gestures: A FrameNet Brasil Approach to Multimodality in Turn Organization 

**Authors**: Helen de Andrade Abreu, Tiago Timponi Torrent, Ely Edison da Silva Matos  

**Link**: [PDF](https://arxiv.org/pdf/2509.09804)  

**Abstract**: This paper proposes a framework for modeling multimodal conversational turn organization via the proposition of correlations between language and interactive gestures, based on analysis as to how pragmatic frames are conceptualized and evoked by communicators. As a means to provide evidence for the analysis, we developed an annotation methodology to enrich a multimodal dataset (annotated for semantic frames) with pragmatic frames modeling conversational turn organization. Although conversational turn organization has been studied by researchers from diverse fields, the specific strategies, especially gestures used by communicators, had not yet been encoded in a dataset that can be used for machine learning. To fill this gap, we enriched the Frame2 dataset with annotations of gestures used for turn organization. The Frame2 dataset features 10 episodes from the Brazilian TV series Pedro Pelo Mundo annotated for semantic frames evoked in both video and text. This dataset allowed us to closely observe how communicators use interactive gestures outside a laboratory, in settings, to our knowledge, not previously recorded in related literature. Our results have confirmed that communicators involved in face-to-face conversation make use of gestures as a tool for passing, taking and keeping conversational turns, and also revealed variations of some gestures that had not been documented before. We propose that the use of these gestures arises from the conceptualization of pragmatic frames, involving mental spaces, blending and conceptual metaphors. In addition, our data demonstrate that the annotation of pragmatic frames contributes to a deeper understanding of human cognition and language. 

---
# HEFT: A Coarse-to-Fine Hierarchy for Enhancing the Efficiency and Accuracy of Language Model Reasoning 

**Authors**: Brennen Hill  

**Link**: [PDF](https://arxiv.org/pdf/2509.09801)  

**Abstract**: The adaptation of large language models (LLMs) to specialized reasoning tasks is fundamentally constrained by computational resources. Parameter-Efficient Fine-Tuning (PEFT) methods have emerged as a powerful solution, yet the landscape of these techniques is diverse, with distinct methods operating in either the model's weight space or its representation space. This paper investigates the hypothesis that a synergistic combination of these paradigms can unlock superior performance and efficiency. We introduce HEFT (Hierarchical Efficient Fine-Tuning), a novel hierarchical adaptation strategy that composes two distinct PEFT methods in a coarse-to-fine manner: first, a broad, foundational adaptation in the weight space using Low-Rank Adaptation (LoRA), followed by a precise, surgical refinement of internal activations using Representation Fine-Tuning (ReFT). We evaluate this approach by fine-tuning a Llama-2-7B model on the BoolQ benchmark, a challenging dataset for inferential reasoning. Our results reveal a profound synergistic effect. A model fine-tuned for only three epochs with our HEFT strategy achieves an accuracy of 85.17\%, exceeding the performance of models trained for 20 epochs with either LoRA-only (85.05\%) or ReFT-only (83.36\%) methodologies. This work demonstrates that the thoughtful composition of PEFT methods is a potent algorithmic innovation, offering a more efficient and effective path toward advancing the reasoning capabilities of language models. By achieving superior results with a fraction of the computational budget, our findings present a principled approach to overcoming the obstacles inherent in adapting large-scale models for complex cognitive tasks. 

---
# Discrimination by LLMs: Cross-lingual Bias Assessment and Mitigation in Decision-Making and Summarisation 

**Authors**: Willem Huijzer, Jieying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.09735)  

**Abstract**: The rapid integration of Large Language Models (LLMs) into various domains raises concerns about societal inequalities and information bias. This study examines biases in LLMs related to background, gender, and age, with a focus on their impact on decision-making and summarization tasks. Additionally, the research examines the cross-lingual propagation of these biases and evaluates the effectiveness of prompt-instructed mitigation strategies. Using an adapted version of the dataset by Tamkin et al. (2023) translated into Dutch, we created 151,200 unique prompts for the decision task and 176,400 for the summarisation task. Various demographic variables, instructions, salience levels, and languages were tested on GPT-3.5 and GPT-4o. Our analysis revealed that both models were significantly biased during decision-making, favouring female gender, younger ages, and certain backgrounds such as the African-American background. In contrast, the summarisation task showed minimal evidence of bias, though significant age-related differences emerged for GPT-3.5 in English. Cross-lingual analysis showed that bias patterns were broadly similar between English and Dutch, though notable differences were observed across specific demographic categories. The newly proposed mitigation instructions, while unable to eliminate biases completely, demonstrated potential in reducing them. The most effective instruction achieved a 27\% mean reduction in the gap between the most and least favorable demographics. Notably, contrary to GPT-3.5, GPT-4o displayed reduced biases for all prompts in English, indicating the specific potential for prompt-based mitigation within newer models. This research underscores the importance of cautious adoption of LLMs and context-specific bias testing, highlighting the need for continued development of effective mitigation strategies to ensure responsible deployment of AI. 

---
# MCP-AgentBench: Evaluating Real-World Language Agent Performance with MCP-Mediated Tools 

**Authors**: Zikang Guo, Benfeng Xu, Chiwei Zhu, Wentao Hong, Xiaorui Wang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2509.09734)  

**Abstract**: The Model Context Protocol (MCP) is rapidly emerging as a pivotal open standard, designed to enhance agent-tool integration and interoperability, and is positioned to unlock a new era of powerful, interconnected, and genuinely utilitarian agentic AI. However, despite MCP's growing adoption, existing benchmarks often fail to capture real-world agent performance within this new paradigm, leading to a distorted perception of their true operational value and an inability to reliably differentiate proficiencies. To bridge this critical evaluation gap, we introduce MCP-AgentBench -- a comprehensive benchmark specifically engineered to rigorously assess language agent capabilities in MCP-mediated tool interactions. Core contributions of MCP-AgentBench include: the establishment of a robust MCP testbed comprising 33 operational servers with 188 distinct tools; the development of a benchmark featuring 600 systematically designed queries distributed across 6 distinct categories of varying interaction complexity; and the introduction of MCP-Eval, a novel outcome-oriented evaluation methodology prioritizing real-world task success. Through extensive empirical evaluation of leading language agents, we provide foundational insights. MCP-AgentBench aims to equip the research community with a standardized and reliable framework to build, validate, and advance agents capable of fully leveraging MCP's transformative benefits, thereby accelerating progress toward truly capable and interoperable AI systems. 

---
# Benchmarking Vision-Language Models on Chinese Ancient Documents: From OCR to Knowledge Reasoning 

**Authors**: Haiyang Yu, Yuchuan Wu, Fan Shi, Lei Liao, Jinghui Lu, Xiaodong Ge, Han Wang, Minghan Zhuo, Xuecheng Wu, Xiang Fei, Hao Feng, Guozhi Tang, An-Lan Wang, Hanshen Zhu, Yangfan He, Quanhuan Liang, Liyuan Meng, Chao Feng, Can Huang, Jingqun Tang, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.09731)  

**Abstract**: Chinese ancient documents, invaluable carriers of millennia of Chinese history and culture, hold rich knowledge across diverse fields but face challenges in digitization and understanding, i.e., traditional methods only scan images, while current Vision-Language Models (VLMs) struggle with their visual and linguistic complexity. Existing document benchmarks focus on English printed texts or simplified Chinese, leaving a gap for evaluating VLMs on ancient Chinese documents. To address this, we present AncientDoc, the first benchmark for Chinese ancient documents, designed to assess VLMs from OCR to knowledge reasoning. AncientDoc includes five tasks (page-level OCR, vernacular translation, reasoning-based QA, knowledge-based QA, linguistic variant QA) and covers 14 document types, over 100 books, and about 3,000 pages. Based on AncientDoc, we evaluate mainstream VLMs using multiple metrics, supplemented by a human-aligned large language model for scoring. 

---
# MultimodalHugs: Enabling Sign Language Processing in Hugging Face 

**Authors**: Gerard Sant, Zifan Jiang, Carlos Escolano, Amit Moryossef, Mathias Müller, Rico Sennrich, Sarah Ebling  

**Link**: [PDF](https://arxiv.org/pdf/2509.09729)  

**Abstract**: In recent years, sign language processing (SLP) has gained importance in the general field of Natural Language Processing. However, compared to research on spoken languages, SLP research is hindered by complex ad-hoc code, inadvertently leading to low reproducibility and unfair comparisons. Existing tools that are built for fast and reproducible experimentation, such as Hugging Face, are not flexible enough to seamlessly integrate sign language experiments. This view is confirmed by a survey we conducted among SLP researchers.
To address these challenges, we introduce MultimodalHugs, a framework built on top of Hugging Face that enables more diverse data modalities and tasks, while inheriting the well-known advantages of the Hugging Face ecosystem. Even though sign languages are our primary focus, MultimodalHugs adds a layer of abstraction that makes it more widely applicable to other use cases that do not fit one of the standard templates of Hugging Face. We provide quantitative experiments to illustrate how MultimodalHugs can accommodate diverse modalities such as pose estimation data for sign languages, or pixel data for text characters. 

---
# A meta-analysis on the performance of machine-learning based language models for sentiment analysis 

**Authors**: Elena Rohde, Jonas Klingwort, Christian Borgs  

**Link**: [PDF](https://arxiv.org/pdf/2509.09728)  

**Abstract**: This paper presents a meta-analysis evaluating ML performance in sentiment analysis for Twitter data. The study aims to estimate the average performance, assess heterogeneity between and within studies, and analyze how study characteristics influence model performance. Using PRISMA guidelines, we searched academic databases and selected 195 trials from 20 studies with 12 study features. Overall accuracy, the most reported performance metric, was analyzed using double arcsine transformation and a three-level random effects model. The average overall accuracy of the AIC-optimized model was 0.80 [0.76, 0.84]. This paper provides two key insights: 1) Overall accuracy is widely used but often misleading due to its sensitivity to class imbalance and the number of sentiment classes, highlighting the need for normalization. 2) Standardized reporting of model performance, including reporting confusion matrices for independent test sets, is essential for reliable comparisons of ML classifiers across studies, which seems far from common practice. 

---
# A Role-Aware Multi-Agent Framework for Financial Education Question Answering with LLMs 

**Authors**: Andy Zhu, Yingjun Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.09727)  

**Abstract**: Question answering (QA) plays a central role in financial education, yet existing large language model (LLM) approaches often fail to capture the nuanced and specialized reasoning required for financial problem-solving. The financial domain demands multistep quantitative reasoning, familiarity with domain-specific terminology, and comprehension of real-world scenarios. We present a multi-agent framework that leverages role-based prompting to enhance performance on domain-specific QA. Our framework comprises a Base Generator, an Evidence Retriever, and an Expert Reviewer agent that work in a single-pass iteration to produce a refined answer. We evaluated our framework on a set of 3,532 expert-designed finance education questions from this http URL, an online learning platform. We leverage retrieval-augmented generation (RAG) for contextual evidence from 6 finance textbooks and prompting strategies for a domain-expert reviewer. Our experiments indicate that critique-based refinement improves answer accuracy by 6.6-8.3% over zero-shot Chain-of-Thought baselines, with the highest performance from Gemini-2.0-Flash. Furthermore, our method enables GPT-4o-mini to achieve performance comparable to the finance-tuned FinGPT-mt_Llama3-8B_LoRA. Our results show a cost-effective approach to enhancing financial QA and offer insights for further research in multi-agent financial LLM systems. 

---
# Natural Language Translation of Formal Proofs through Informalization of Proof Steps and Recursive Summarization along Proof Structure 

**Authors**: Seiji Hattori, Takuya Matsuzaki, Makoto Fujiwara  

**Link**: [PDF](https://arxiv.org/pdf/2509.09726)  

**Abstract**: This paper proposes a natural language translation method for machine-verifiable formal proofs that leverages the informalization (verbalization of formal language proof steps) and summarization capabilities of LLMs. For evaluation, it was applied to formal proof data created in accordance with natural language proofs taken from an undergraduate-level textbook, and the quality of the generated natural language proofs was analyzed in comparison with the original natural language proofs. Furthermore, we will demonstrate that this method can output highly readable and accurate natural language proofs by applying it to existing formal proof library of the Lean proof assistant. 

---
# BIBERT-Pipe on Biomedical Nested Named Entity Linking at BioASQ 2025 

**Authors**: Chunyu Li, Xindi Zheng, Siqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09725)  

**Abstract**: Entity linking (EL) for biomedical text is typically benchmarked on English-only corpora with flat mentions, leaving the more realistic scenario of nested and multilingual mentions largely unexplored. We present our system for the BioNNE 2025 Multilingual Biomedical Nested Named Entity Linking shared task (English & Russian), closing this gap with a lightweight pipeline that keeps the original EL model intact and modifies only three task-aligned components: Two-stage retrieval-ranking. We leverage the same base encoder model in both stages: the retrieval stage uses the original pre-trained model, while the ranking stage applies domain-specific fine-tuning. Boundary cues. In the ranking stage, we wrap each mention with learnable [Ms] / [Me] tags, providing the encoder with an explicit, language-agnostic span before robustness to overlap and nesting. Dataset augmentation. We also automatically expand the ranking training corpus with three complementary data sources, enhancing coverage without extra manual annotation. On the BioNNE 2025 leaderboard, our two stage system, bilingual bert (BIBERT-Pipe), ranks third in the multilingual track, demonstrating the effectiveness and competitiveness of these minimal yet principled modifications. Code are publicly available at this https URL. 

---
# DiTTO-LLM: Framework for Discovering Topic-based Technology Opportunities via Large Language Model 

**Authors**: Wonyoung Kim, Sujeong Seo, Juhyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.09724)  

**Abstract**: Technology opportunities are critical information that serve as a foundation for advancements in technology, industry, and innovation. This paper proposes a framework based on the temporal relationships between technologies to identify emerging technology opportunities. The proposed framework begins by extracting text from a patent dataset, followed by mapping text-based topics to discover inter-technology relationships. Technology opportunities are then identified by tracking changes in these topics over time. To enhance efficiency, the framework leverages a large language model to extract topics and employs a prompt for a chat-based language model to support the discovery of technology opportunities. The framework was evaluated using an artificial intelligence patent dataset provided by the United States Patent and Trademark Office. The experimental results suggest that artificial intelligence technology is evolving into forms that facilitate everyday accessibility. This approach demonstrates the potential of the proposed framework to identify future technology opportunities. 

---
# ALIGNS: Unlocking nomological networks in psychological measurement through a large language model 

**Authors**: Kai R. Larsen, Sen Yan, Roland Müller, Lan Sang, Mikko Rönkkö, Ravi Starzl, Donald Edmondson  

**Link**: [PDF](https://arxiv.org/pdf/2509.09723)  

**Abstract**: Psychological measurement is critical to many disciplines. Despite advances in measurement, building nomological networks, theoretical maps of how concepts and measures relate to establish validity, remains a challenge 70 years after Cronbach and Meehl proposed them as fundamental to validation. This limitation has practical consequences: clinical trials may fail to detect treatment effects, and public policy may target the wrong outcomes. We introduce Analysis of Latent Indicators to Generate Nomological Structures (ALIGNS), a large language model-based system trained with validated questionnaire measures. ALIGNS provides three comprehensive nomological networks containing over 550,000 indicators across psychology, medicine, social policy, and other fields. This represents the first application of large language models to solve a foundational problem in measurement validation. We report classification accuracy tests used to develop the model, as well as three evaluations. In the first evaluation, the widely used NIH PROMIS anxiety and depression instruments are shown to converge into a single dimension of emotional distress. The second evaluation examines child temperament measures and identifies four potential dimensions not captured by current frameworks, and questions one existing dimension. The third evaluation, an applicability check, engages expert psychometricians who assess the system's importance, accessibility, and suitability. ALIGNS is freely available at this http URL, complementing traditional validation methods with large-scale nomological analysis. 

---
# Investigating Symbolic Triggers of Hallucination in Gemma Models Across HaluEval and TruthfulQA 

**Authors**: Naveen Lamba, Sanju Tiwari, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2509.09715)  

**Abstract**: Hallucination in Large Language Models (LLMs) is a well studied problem. However, the properties that make LLM intrinsically vulnerable to hallucinations have not been identified and studied. This research identifies and characterizes the key properties, allowing us to pinpoint vulnerabilities within the model's internal mechanisms. To solidify on these properties, we utilized two established datasets, HaluEval and TruthfulQA and convert their existing format of question answering into various other formats to narrow down these properties as the reason for the hallucinations. Our findings reveal that hallucination percentages across symbolic properties are notably high for Gemma-2-2B, averaging 79.0% across tasks and datasets. With increased model scale, hallucination drops to 73.6% for Gemma-2-9B and 63.9% for Gemma-2-27B, reflecting a 15 percentage point reduction overall. Although the hallucination rate decreases as the model size increases, a substantial amount of hallucination caused by symbolic properties still persists. This is especially evident for modifiers (ranging from 84.76% to 94.98%) and named entities (ranging from 83.87% to 93.96%) across all Gemma models and both datasets. These findings indicate that symbolic elements continue to confuse the models, pointing to a fundamental weakness in how these LLMs process such inputs--regardless of their scale. 

---
# How Small Transformation Expose the Weakness of Semantic Similarity Measures 

**Authors**: Serge Lionel Nikiema, Albérick Euraste Djire, Abdoul Aziz Bonkoungou, Micheline Bénédicte Moumoula, Jordan Samhi, Abdoul Kader Kabore, Jacques Klein, Tegawendé F. Bissyande  

**Link**: [PDF](https://arxiv.org/pdf/2509.09714)  

**Abstract**: This research examines how well different methods measure semantic similarity, which is important for various software engineering applications such as code search, API recommendations, automated code reviews, and refactoring tools. While large language models are increasingly used for these similarity assessments, questions remain about whether they truly understand semantic relationships or merely recognize surface patterns.
The study tested 18 different similarity measurement approaches, including word-based methods, embedding techniques, LLM-based systems, and structure-aware algorithms. The researchers created a systematic testing framework that applies controlled changes to text and code to evaluate how well each method handles different types of semantic relationships.
The results revealed significant issues with commonly used metrics. Some embedding-based methods incorrectly identified semantic opposites as similar up to 99.9 percent of the time, while certain transformer-based approaches occasionally rated opposite meanings as more similar than synonymous ones. The study found that embedding methods' poor performance often stemmed from how they calculate distances; switching from Euclidean distance to cosine similarity improved results by 24 to 66 percent. LLM-based approaches performed better at distinguishing semantic differences, producing low similarity scores (0.00 to 0.29) for genuinely different meanings, compared to embedding methods that incorrectly assigned high scores (0.82 to 0.99) to dissimilar content. 

---
# HANRAG: Heuristic Accurate Noise-resistant Retrieval-Augmented Generation for Multi-hop Question Answering 

**Authors**: Duolin Sun, Dan Yang, Yue Shen, Yihan Jiao, Zhehao Tan, Jie Feng, Lianzhen Zhong, Jian Wang, Peng Wei, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09713)  

**Abstract**: The Retrieval-Augmented Generation (RAG) approach enhances question-answering systems and dialogue generation tasks by integrating information retrieval (IR) technologies with large language models (LLMs). This strategy, which retrieves information from external knowledge bases to bolster the response capabilities of generative models, has achieved certain successes. However, current RAG methods still face numerous challenges when dealing with multi-hop queries. For instance, some approaches overly rely on iterative retrieval, wasting too many retrieval steps on compound queries. Additionally, using the original complex query for retrieval may fail to capture content relevant to specific sub-queries, resulting in noisy retrieved content. If the noise is not managed, it can lead to the problem of noise accumulation. To address these issues, we introduce HANRAG, a novel heuristic-based framework designed to efficiently tackle problems of varying complexity. Driven by a powerful revelator, HANRAG routes queries, decomposes them into sub-queries, and filters noise from retrieved documents. This enhances the system's adaptability and noise resistance, making it highly capable of handling diverse queries. We compare the proposed framework against other leading industry methods across various benchmarks. The results demonstrate that our framework obtains superior performance in both single-hop and multi-hop question-answering tasks. 

---
# The Thinking Therapist: Training Large Language Models to Deliver Acceptance and Commitment Therapy using Supervised Fine-Tuning and Odds Ratio Policy Optimization 

**Authors**: Talha Tahir  

**Link**: [PDF](https://arxiv.org/pdf/2509.09712)  

**Abstract**: Acceptance and Commitment Therapy (ACT) is a third-wave cognitive behavioral therapy with emerging evidence of efficacy in several psychiatric conditions. This study investigates the impact of post-training methodology and explicit reasoning on the ability of a small open-weight large language model (LLM) to deliver ACT. Using 50 sets of synthetic ACT transcripts generated by Mistral-Large, we trained Llama-3.2-3b-Instruct with two distinct approaches, supervised fine-tuning (SFT) and odds ratio policy optimization (ORPO), each with and without an explicit chain-of-thought (COT) reasoning step. Performance was evaluated by comparing these four post-trained variants against the base Instruct model. These models were benchmarked in simulated therapy sessions, with performance quantitatively assessed on the ACT Fidelity Measure (ACT-FM) and the Therapist Empathy Scale (TES) by an LLM judge that had been fine-tuned on human evaluations. Our findings demonstrate that the ORPO-trained models significantly outperformed both their SFT and Instruct counterparts on ACT fidelity ($\chi^2(5) = 185.15, p < .001$) and therapeutic empathy ($\chi^2(5) = 140.37, p < .001$). The effect of COT was conditional as it provided a significant benefit to SFT models, improving ACT-FM scores by an average of 2.68 points ($p < .001$), while offering no discernible advantage to the superior ORPO or instruct-tuned variants. We posit that the superiority of ORPO stems from its ability to learn the therapeutic `process' over imitating `content,' a key aspect of ACT, while COT acts as a necessary scaffold for models trained only via imitation. This study establishes that preference-aligned policy optimization can effectively instill ACT competencies in small LLMs, and that the utility of explicit reasoning is highly dependent on the underlying training paradigm. 

---
# Psychiatry-Bench: A Multi-Task Benchmark for LLMs in Psychiatry 

**Authors**: Aya E. Fouda, Abdelrahamn A. Hassan, Radwa J. Hanafy, Mohammed E. Fouda  

**Link**: [PDF](https://arxiv.org/pdf/2509.09711)  

**Abstract**: Large language models (LLMs) hold great promise in enhancing psychiatric practice, from improving diagnostic accuracy to streamlining clinical documentation and therapeutic support. However, existing evaluation resources heavily rely on small clinical interview corpora, social media posts, or synthetic dialogues, which limits their clinical validity and fails to capture the full complexity of psychiatric reasoning. In this work, we introduce PsychiatryBench, a rigorously curated benchmark grounded exclusively in authoritative, expert-validated psychiatric textbooks and casebooks. PsychiatryBench comprises eleven distinct question-answering tasks ranging from diagnostic reasoning and treatment planning to longitudinal follow-up, management planning, clinical approach, sequential case analysis, and multiple-choice/extended matching formats totaling over 5,300 expert-annotated items. We evaluate a diverse set of frontier LLMs (including Google Gemini, DeepSeek, LLaMA 3, and QWQ-32) alongside leading open-source medical models (e.g., OpenBiloLLM, MedGemma) using both conventional metrics and an "LLM-as-judge" similarity scoring framework. Our results reveal substantial gaps in clinical consistency and safety, particularly in multi-turn follow-up and management tasks, underscoring the need for specialized model tuning and more robust evaluation paradigms. PsychiatryBench offers a modular, extensible platform for benchmarking and improving LLM performance in high-stakes mental health applications. 

---
# Generating Individual Travel Diaries Using Large Language Models Informed by Census and Land-Use Data 

**Authors**: Sepehr Golrokh Amin, Devin Rhoads, Fatemeh Fakhrmoosavi, Nicholas E. Lownes, John N. Ivan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09710)  

**Abstract**: This study introduces a Large Language Model (LLM) scheme for generating individual travel diaries in agent-based transportation models. While traditional approaches rely on large quantities of proprietary household travel surveys, the method presented in this study generates personas stochastically from open-source American Community Survey (ACS) and Smart Location Database (SLD) data, then synthesizes diaries through direct prompting. This study features a novel one-to-cohort realism score: a composite of four metrics (Trip Count Score, Interval Score, Purpose Score, and Mode Score) validated against the Connecticut Statewide Transportation Study (CSTS) diaries, matched across demographic variables. The validation utilizes Jensen-Shannon Divergence to measure distributional similarities between generated and real diaries. When compared to diaries generated with classical methods (Negative Binomial for trip generation; Multinomial Logit for mode/purpose) calibrated on the validation set, LLM-generated diaries achieve comparable overall realism (LLM mean: 0.485 vs. 0.455). The LLM excels in determining trip purpose and demonstrates greater consistency (narrower realism score distribution), while classical models lead in numerical estimates of trip count and activity duration. Aggregate validation confirms the LLM's statistical representativeness (LLM mean: 0.612 vs. 0.435), demonstrating LLM's zero-shot viability and establishing a quantifiable metric of diary realism for future synthetic diary evaluation systems. 

---
# Assisting Research Proposal Writing with Large Language Models: Evaluation and Refinement 

**Authors**: Jing Ren, Weiqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09709)  

**Abstract**: Large language models (LLMs) like ChatGPT are increasingly used in academic writing, yet issues such as incorrect or fabricated references raise ethical concerns. Moreover, current content quality evaluations often rely on subjective human judgment, which is labor-intensive and lacks objectivity, potentially compromising the consistency and reliability. In this study, to provide a quantitative evaluation and enhance research proposal writing capabilities of LLMs, we propose two key evaluation metrics--content quality and reference validity--and an iterative prompting method based on the scores derived from these two metrics. Our extensive experiments show that the proposed metrics provide an objective, quantitative framework for assessing ChatGPT's writing performance. Additionally, iterative prompting significantly enhances content quality while reducing reference inaccuracies and fabrications, addressing critical ethical challenges in academic contexts. 

---
# Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal 

**Authors**: Nirmalendu Prakash, Yeo Wei Jie, Amir Abdullah, Ranjan Satapathy, Erik Cambria, Roy Ka Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.09708)  

**Abstract**: Refusal on harmful prompts is a key safety behaviour in instruction-tuned large language models (LLMs), yet the internal causes of this behaviour remain poorly understood. We study two public instruction-tuned models, Gemma-2-2B-IT and LLaMA-3.1-8B-IT, using sparse autoencoders (SAEs) trained on residual-stream activations. Given a harmful prompt, we search the SAE latent space for feature sets whose ablation flips the model from refusal to compliance, demonstrating causal influence and creating a jailbreak. Our search proceeds in three stages: (1) Refusal Direction: find a refusal-mediating direction and collect SAE features near that direction; (2) Greedy Filtering: prune to a minimal set; and (3) Interaction Discovery: fit a factorization machine (FM) that captures nonlinear interactions among the remaining active features and the minimal set. This pipeline yields a broad set of jailbreak-critical features, offering insight into the mechanistic basis of refusal. Moreover, we find evidence of redundant features that remain dormant unless earlier features are suppressed. Our findings highlight the potential for fine-grained auditing and targeted intervention in safety behaviours by manipulating the interpretable latent space. 

---
# The Non-Determinism of Small LLMs: Evidence of Low Answer Consistency in Repetition Trials of Standard Multiple-Choice Benchmarks 

**Authors**: Claudio Pinhanez, Paulo Cavalin, Cassia Sanctos, Marcelo Grave, Yago Primerano  

**Link**: [PDF](https://arxiv.org/pdf/2509.09705)  

**Abstract**: This work explores the consistency of small LLMs (2B-8B parameters) in answering multiple times the same question. We present a study on known, open-source LLMs responding to 10 repetitions of questions from the multiple-choice benchmarks MMLU-Redux and MedQA, considering different inference temperatures, small vs. medium models (50B-80B), finetuned vs. base models, and other parameters. We also look into the effects of requiring multi-trial answer consistency on accuracy and the trade-offs involved in deciding which model best provides both of them. To support those studies, we propose some new analytical and graphical tools. Results show that the number of questions which can be answered consistently vary considerably among models but are typically in the 50%-80% range for small models at low inference temperatures. Also, accuracy among consistent answers seems to reasonably correlate with overall accuracy. Results for medium-sized models seem to indicate much higher levels of answer consistency. 

---
# Temporal Preferences in Language Models for Long-Horizon Assistance 

**Authors**: Ali Mazyaki, Mohammad Naghizadeh, Samaneh Ranjkhah Zonouzaghi, Hossein Setareh  

**Link**: [PDF](https://arxiv.org/pdf/2509.09704)  

**Abstract**: We study whether language models (LMs) exhibit future- versus present-oriented preferences in intertemporal choice and whether those preferences can be systematically manipulated. Using adapted human experimental protocols, we evaluate multiple LMs on time-tradeoff tasks and benchmark them against a sample of human decision makers. We introduce an operational metric, the Manipulability of Time Orientation (MTO), defined as the change in an LM's revealed time preference between future- and present-oriented prompts. In our tests, reasoning-focused models (e.g., DeepSeek-Reasoner and grok-3-mini) choose later options under future-oriented prompts but only partially personalize decisions across identities or geographies. Moreover, models that correctly reason about time orientation internalize a future orientation for themselves as AI decision makers. We discuss design implications for AI assistants that should align with heterogeneous, long-horizon goals and outline a research agenda on personalized contextual calibration and socially aware deployment. 

---
# CTCC: A Robust and Stealthy Fingerprinting Framework for Large Language Models via Cross-Turn Contextual Correlation Backdoor 

**Authors**: Zhenhua Xu, Xixiang Zhao, Xubin Yue, Shengwei Tian, Changting Lin, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.09703)  

**Abstract**: The widespread deployment of large language models (LLMs) has intensified concerns around intellectual property (IP) protection, as model theft and unauthorized redistribution become increasingly feasible. To address this, model fingerprinting aims to embed verifiable ownership traces into LLMs. However, existing methods face inherent trade-offs between stealthness, robustness, and generalizability, being either detectable via distributional shifts, vulnerable to adversarial modifications, or easily invalidated once the fingerprint is revealed. In this work, we introduce CTCC, a novel rule-driven fingerprinting framework that encodes contextual correlations across multiple dialogue turns, such as counterfactual, rather than relying on token-level or single-turn triggers. CTCC enables fingerprint verification under black-box access while mitigating false positives and fingerprint leakage, supporting continuous construction under a shared semantic rule even if partial triggers are exposed. Extensive experiments across multiple LLM architectures demonstrate that CTCC consistently achieves stronger stealth and robustness than prior work. Our findings position CTCC as a reliable and practical solution for ownership verification in real-world LLM deployment scenarios. Our code and data are publicly available at <this https URL. 

---
# Creativity Benchmark: A benchmark for marketing creativity for LLM models 

**Authors**: Ninad Bhat, Kieran Browne, Pip Bingemann  

**Link**: [PDF](https://arxiv.org/pdf/2509.09702)  

**Abstract**: We introduce Creativity Benchmark, an evaluation framework for large language models (LLMs) in marketing creativity. The benchmark covers 100 brands (12 categories) and three prompt types (Insights, Ideas, Wild Ideas). Human pairwise preferences from 678 practising creatives over 11,012 anonymised comparisons, analysed with Bradley-Terry models, show tightly clustered performance with no model dominating across brands or prompt types: the top-bottom spread is $\Delta\theta \approx 0.45$, which implies a head-to-head win probability of $0.61$; the highest-rated model beats the lowest only about $61\%$ of the time. We also analyse model diversity using cosine distances to capture intra- and inter-model variation and sensitivity to prompt reframing. Comparing three LLM-as-judge setups with human rankings reveals weak, inconsistent correlations and judge-specific biases, underscoring that automated judges cannot substitute for human evaluation. Conventional creativity tests also transfer only partially to brand-constrained tasks. Overall, the results highlight the need for expert human evaluation and diversity-aware workflows. 

---
# Optimal Multi-Task Learning at Regularization Horizon for Speech Translation Task 

**Authors**: JungHo Jung, Junhyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.09701)  

**Abstract**: End-to-end speech-to-text translation typically suffers from the scarcity of paired speech-text data. One way to overcome this shortcoming is to utilize the bitext data from the Machine Translation (MT) task and perform Multi-Task Learning (MTL). In this paper, we formulate MTL from a regularization perspective and explore how sequences can be regularized within and across modalities. By thoroughly investigating the effect of consistency regularization (different modality) and R-drop (same modality), we show how they respectively contribute to the total regularization. We also demonstrate that the coefficient of MT loss serves as another source of regularization in the MTL setting. With these three sources of regularization, we introduce the optimal regularization contour in the high-dimensional space, called the regularization horizon. Experiments show that tuning the hyperparameters within the regularization horizon achieves near state-of-the-art performance on the MuST-C dataset. 

---
# Cross-Layer Attention Probing for Fine-Grained Hallucination Detection 

**Authors**: Malavika Suresh, Rahaf Aljundi, Ikechukwu Nkisi-Orji, Nirmalie Wiratunga  

**Link**: [PDF](https://arxiv.org/pdf/2509.09700)  

**Abstract**: With the large-scale adoption of Large Language Models (LLMs) in various applications, there is a growing reliability concern due to their tendency to generate inaccurate text, i.e. hallucinations. In this work, we propose Cross-Layer Attention Probing (CLAP), a novel activation probing technique for hallucination detection, which processes the LLM activations across the entire residual stream as a joint sequence. Our empirical evaluations using five LLMs and three tasks show that CLAP improves hallucination detection compared to baselines on both greedy decoded responses as well as responses sampled at higher temperatures, thus enabling fine-grained detection, i.e. the ability to disambiguate hallucinations and non-hallucinations among different sampled responses to a given prompt. This allows us to propose a detect-then-mitigate strategy using CLAP to reduce hallucinations and improve LLM reliability compared to direct mitigation approaches. Finally, we show that CLAP maintains high reliability even when applied out-of-distribution. 

---
# Structured Information Matters: Explainable ICD Coding with Patient-Level Knowledge Graphs 

**Authors**: Mingyang Li, Viktor Schlegel, Tingting Mu, Warren Del-Pinto, Goran Nenadic  

**Link**: [PDF](https://arxiv.org/pdf/2509.09699)  

**Abstract**: Mapping clinical documents to standardised clinical vocabularies is an important task, as it provides structured data for information retrieval and analysis, which is essential to clinical research, hospital administration and improving patient care. However, manual coding is both difficult and time-consuming, making it impractical at scale. Automated coding can potentially alleviate this burden, improving the availability and accuracy of structured clinical data. The task is difficult to automate, as it requires mapping to high-dimensional and long-tailed target spaces, such as the International Classification of Diseases (ICD). While external knowledge sources have been readily utilised to enhance output code representation, the use of external resources for representing the input documents has been underexplored. In this work, we compute a structured representation of the input documents, making use of document-level knowledge graphs (KGs) that provide a comprehensive structured view of a patient's condition. The resulting knowledge graph efficiently represents the patient-centred input documents with 23\% of the original text while retaining 90\% of the information. We assess the effectiveness of this graph for automated ICD-9 coding by integrating it into the state-of-the-art ICD coding architecture PLM-ICD. Our experiments yield improved Macro-F1 scores by up to 3.20\% on popular benchmarks, while improving training efficiency. We attribute this improvement to different types of entities and relationships in the KG, and demonstrate the improved explainability potential of the approach over the text-only baseline. 

---
# Abduct, Act, Predict: Scaffolding Causal Inference for Automated Failure Attribution in Multi-Agent Systems 

**Authors**: Alva West, Yixuan Weng, Minjun Zhu, Zhen Lin, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10401)  

**Abstract**: Failure attribution in multi-agent systems -- pinpointing the exact step where a decisive error occurs -- is a critical yet unsolved challenge. Current methods treat this as a pattern recognition task over long conversation logs, leading to critically low step-level accuracy (below 17\%), which renders them impractical for debugging complex systems. Their core weakness is a fundamental inability to perform robust counterfactual reasoning: to determine if correcting a single action would have actually averted the task failure. To bridge this counterfactual inference gap, we introduce Abduct-Act-Predict (A2P) Scaffolding, a novel agent framework that transforms failure attribution from pattern recognition into a structured causal inference task. A2P explicitly guides a large language model through a formal three-step reasoning process within a single inference pass: (1) Abduction, to infer the hidden root causes behind an agent's actions; (2) Action, to define a minimal corrective intervention; and (3) Prediction, to simulate the subsequent trajectory and verify if the intervention resolves the failure. This structured approach leverages the holistic context of the entire conversation while imposing a rigorous causal logic on the model's analysis. Our extensive experiments on the Who\&When benchmark demonstrate its efficacy. On the Algorithm-Generated dataset, A2P achieves 47.46\% step-level accuracy, a 2.85$\times$ improvement over the 16.67\% of the baseline. On the more complex Hand-Crafted dataset, it achieves 29.31\% step accuracy, a 2.43$\times$ improvement over the baseline's 12.07\%. By reframing the problem through a causal lens, A2P Scaffolding provides a robust, verifiable, and significantly more accurate solution for automated failure attribution. 

---
# Error Analysis in a Modular Meeting Transcription System 

**Authors**: Peter Vieting, Simon Berger, Thilo von Neumann, Christoph Boeddeker, Ralf Schlüter, Reinhold Haeb-Umbach  

**Link**: [PDF](https://arxiv.org/pdf/2509.10143)  

**Abstract**: Meeting transcription is a field of high relevance and remarkable progress in recent years. Still, challenges remain that limit its performance. In this work, we extend a previously proposed framework for analyzing leakage in speech separation with proper sensitivity to temporal locality. We show that there is significant leakage to the cross channel in areas where only the primary speaker is active. At the same time, the results demonstrate that this does not affect the final performance much as these leaked parts are largely ignored by the voice activity detection (VAD). Furthermore, different segmentations are compared showing that advanced diarization approaches are able to reduce the gap to oracle segmentation by a third compared to a simple energy-based VAD. We additionally reveal what factors contribute to the remaining difference. The results represent state-of-the-art performance on LibriCSS among systems that train the recognition module on LibriSpeech data only. 

---
# VARCO-VISION-2.0 Technical Report 

**Authors**: Young-rok Cha, Jeongho Ju, SunYoung Park, Jong-Hyeon Lee, Younghyun Yu, Youngjune Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.10105)  

**Abstract**: We introduce VARCO-VISION-2.0, an open-weight bilingual vision-language model (VLM) for Korean and English with improved capabilities compared to the previous model VARCO-VISION-14B. The model supports multi-image understanding for complex inputs such as documents, charts, and tables, and delivers layoutaware OCR by predicting both textual content and its spatial location. Trained with a four-stage curriculum with memory-efficient techniques, the model achieves enhanced multimodal alignment, while preserving core language abilities and improving safety via preference optimization. Extensive benchmark evaluations demonstrate strong spatial grounding and competitive results for both languages, with the 14B model achieving 8th place on the OpenCompass VLM leaderboard among models of comparable scale. Alongside the 14B-scale model, we release a 1.7B version optimized for on-device deployment. We believe these models advance the development of bilingual VLMs and their practical applications. Two variants of VARCO-VISION-2.0 are available at Hugging Face: a full-scale 14B model and a lightweight 1.7B model. 

---
# Unified Learnable 2D Convolutional Feature Extraction for ASR 

**Authors**: Peter Vieting, Benedikt Hilmes, Ralf Schlüter, Hermann Ney  

**Link**: [PDF](https://arxiv.org/pdf/2509.10031)  

**Abstract**: Neural front-ends represent a promising approach to feature extraction for automatic speech recognition (ASR) systems as they enable to learn specifically tailored features for different tasks. Yet, many of the existing techniques remain heavily influenced by classical methods. While this inductive bias may ease the system design, our work aims to develop a more generic front-end for feature extraction. Furthermore, we seek to unify the front-end architecture contrasting with existing approaches that apply a composition of several layer topologies originating from different sources. The experiments systematically show how to reduce the influence of existing techniques to achieve a generic front-end. The resulting 2D convolutional front-end is parameter-efficient and suitable for a scenario with limited computational resources unlike large models pre-trained on unlabeled audio. The results demonstrate that this generic unified approach is not only feasible but also matches the performance of existing supervised learnable feature extractors. 

---
# Whisper Has an Internal Word Aligner 

**Authors**: Sung-Lin Yeh, Yen Meng, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09987)  

**Abstract**: There is an increasing interest in obtaining accurate word-level timestamps from strong automatic speech recognizers, in particular Whisper. Existing approaches either require additional training or are simply not competitive. The evaluation in prior work is also relatively loose, typically using a tolerance of more than 200 ms. In this work, we discover attention heads in Whisper that capture accurate word alignments and are distinctively different from those that do not. Moreover, we find that using characters produces finer and more accurate alignments than using wordpieces. Based on these findings, we propose an unsupervised approach to extracting word alignments by filtering attention heads while teacher forcing Whisper with characters. Our approach not only does not require training but also produces word alignments that are more accurate than prior work under a stricter tolerance between 20 ms and 100 ms. 

---
# Vibe Check: Understanding the Effects of LLM-Based Conversational Agents' Personality and Alignment on User Perceptions in Goal-Oriented Tasks 

**Authors**: Hasibur Rahman, Smit Desai  

**Link**: [PDF](https://arxiv.org/pdf/2509.09870)  

**Abstract**: Large language models (LLMs) enable conversational agents (CAs) to express distinctive personalities, raising new questions about how such designs shape user perceptions. This study investigates how personality expression levels and user-agent personality alignment influence perceptions in goal-oriented tasks. In a between-subjects experiment (N=150), participants completed travel planning with CAs exhibiting low, medium, or high expression across the Big Five traits, controlled via our novel Trait Modulation Keys framework. Results revealed an inverted-U relationship: medium expression produced the most positive evaluations across Intelligence, Enjoyment, Anthropomorphism, Intention to Adopt, Trust, and Likeability, significantly outperforming both extremes. Personality alignment further enhanced outcomes, with Extraversion and Emotional Stability emerging as the most influential traits. Cluster analysis identified three distinct compatibility profiles, with "Well-Aligned" users reporting substantially positive perceptions. These findings demonstrate that personality expression and strategic trait alignment constitute optimal design targets for CA personality, offering design implications as LLM-based CAs become increasingly prevalent. 

---
# LLMs as Agentic Cooperative Players in Multiplayer UNO 

**Authors**: Yago Romano Matinez, Jesse Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2509.09867)  

**Abstract**: LLMs promise to assist humans -- not just by answering questions, but by offering useful guidance across a wide range of tasks. But how far does that assistance go? Can a large language model based agent actually help someone accomplish their goal as an active participant? We test this question by engaging an LLM in UNO, a turn-based card game, asking it not to win but instead help another player to do so. We built a tool that allows decoder-only LLMs to participate as agents within the RLCard game environment. These models receive full game-state information and respond using simple text prompts under two distinct prompting strategies. We evaluate models ranging from small (1B parameters) to large (70B parameters) and explore how model scale impacts performance. We find that while all models were able to successfully outperform a random baseline when playing UNO, few were able to significantly aid another player. 

---
# Latency and Token-Aware Test-Time Compute 

**Authors**: Jenny Y. Huang, Mehul Damani, Yousef El-Kurdi, Ramon Astudillo, Wei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.09864)  

**Abstract**: Inference-time scaling has emerged as a powerful way to improve large language model (LLM) performance by generating multiple candidate responses and selecting among them. However, existing work on dynamic allocation for test-time compute typically considers only parallel generation methods such as best-of-N, overlooking incremental decoding methods like beam search, and has largely ignored latency, focusing only on token usage. We formulate inference-time scaling as a problem of dynamic compute allocation and method selection, where the system must decide which strategy to apply and how much compute to allocate on a per-query basis. Our framework explicitly incorporates both token cost and wall-clock latency, the latter being critical for user experience and particularly for agentic workflows where models must issue multiple queries efficiently. Experiments on reasoning benchmarks show that our approach consistently outperforms static strategies, achieving favorable accuracy-cost trade-offs while remaining practical for deployment. 

---
# Executable Ontologies: Synthesizing Event Semantics with Dataflow Architecture 

**Authors**: Aleksandr Boldachev  

**Link**: [PDF](https://arxiv.org/pdf/2509.09775)  

**Abstract**: This paper presents boldsea, Boldachev's semantic-event approach -- an architecture for modeling complex dynamic systems using executable ontologies -- semantic models that act as dynamic structures, directly controlling process execution. We demonstrate that integrating event semantics with a dataflow architecture addresses the limitations of traditional Business Process Management (BPM) systems and object-oriented semantic technologies. The paper presents the formal BSL (boldsea Semantic Language), including its BNF grammar, and outlines the boldsea-engine's architecture, which directly interprets semantic models as executable algorithms without compilation. It enables the modification of event models at runtime, ensures temporal transparency, and seamlessly merges data and business logic within a unified semantic framework. 

---
# HypoGeneAgent: A Hypothesis Language Agent for Gene-Set Cluster Resolution Selection Using Perturb-seq Datasets 

**Authors**: Ying Yuan, Xing-Yue Monica Ge, Aaron Archer Waterman, Tommaso Biancalani, David Richmond, Yogesh Pandit, Avtar Singh, Russell Littman, Jin Liu, Jan-Christian Huetter, Vladimir Ermakov  

**Link**: [PDF](https://arxiv.org/pdf/2509.09740)  

**Abstract**: Large-scale single-cell and Perturb-seq investigations routinely involve clustering cells and subsequently annotating each cluster with Gene-Ontology (GO) terms to elucidate the underlying biological programs. However, both stages, resolution selection and functional annotation, are inherently subjective, relying on heuristics and expert curation. We present HYPOGENEAGENT, a large language model (LLM)-driven framework, transforming cluster annotation into a quantitatively optimizable task. Initially, an LLM functioning as a gene-set analyst analyzes the content of each gene program or perturbation module and generates a ranked list of GO-based hypotheses, accompanied by calibrated confidence scores. Subsequently, we embed every predicted description with a sentence-embedding model, compute pair-wise cosine similarities, and let the agent referee panel score (i) the internal consistency of the predictions, high average similarity within the same cluster, termed intra-cluster agreement (ii) their external distinctiveness, low similarity between clusters, termed inter-cluster separation. These two quantities are combined to produce an agent-derived resolution score, which is maximized when clusters exhibit simultaneous coherence and mutual exclusivity. When applied to a public K562 CRISPRi Perturb-seq dataset as a preliminary test, our Resolution Score selects clustering granularities that exhibit alignment with known pathway compared to classical metrics such silhouette score, modularity score for gene functional enrichment summary. These findings establish LLM agents as objective adjudicators of cluster resolution and functional annotation, thereby paving the way for fully automated, context-aware interpretation pipelines in single-cell multi-omics studies. 

---
# Improving MLLM Historical Record Extraction with Test-Time Image 

**Authors**: Taylor Archibald, Tony Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2509.09722)  

**Abstract**: We present a novel ensemble framework that stabilizes LLM based text extraction from noisy historical documents. We transcribe multiple augmented variants of each image with Gemini 2.0 Flash and fuse these outputs with a custom Needleman Wunsch style aligner that yields both a consensus transcription and a confidence score. We present a new dataset of 622 Pennsylvania death records, and demonstrate our method improves transcription accuracy by 4 percentage points relative to a single shot baseline. We find that padding and blurring are the most useful for improving accuracy, while grid warp perturbations are best for separating high and low confidence cases. The approach is simple, scalable, and immediately deployable to other document collections and transcription models. 

---
# VStyle: A Benchmark for Voice Style Adaptation with Spoken Instructions 

**Authors**: Jun Zhan, Mingyang Han, Yuxuan Xie, Chen Wang, Dong Zhang, Kexin Huang, Haoxiang Shi, DongXiao Wang, Tengtao Song, Qinyuan Cheng, Shimin Li, Jun Song, Xipeng Qiu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.09716)  

**Abstract**: Spoken language models (SLMs) have emerged as a unified paradigm for speech understanding and generation, enabling natural human machine interaction. However, while most progress has focused on semantic accuracy and instruction following, the ability of SLMs to adapt their speaking style based on spoken instructions has received limited attention. We introduce Voice Style Adaptation (VSA), a new task that examines whether SLMs can modify their speaking style, such as timbre, prosody, or persona following natural language spoken commands. To study this task, we present VStyle, a bilingual (Chinese & English) benchmark covering four categories of speech generation: acoustic attributes, natural language instruction, role play, and implicit empathy. We also introduce the Large Audio Language Model as a Judge (LALM as a Judge) framework, which progressively evaluates outputs along textual faithfulness, style adherence, and naturalness, ensuring reproducible and objective assessment. Experiments on commercial systems and open source SLMs demonstrate that current models face clear limitations in controllable style adaptation, highlighting both the novelty and challenge of this task. By releasing VStyle and its evaluation toolkit, we aim to provide the community with a foundation for advancing human centered spoken interaction. The dataset and code are publicly available at \href{this https URL}{project's homepage}. 

---
# LLM-Based Instance-Driven Heuristic Bias In the Context of a Biased Random Key Genetic Algorithm 

**Authors**: Camilo Chacón Sartori, Martín Isla Pino, Pedro Pinacho-Davidson, Christian Blum  

**Link**: [PDF](https://arxiv.org/pdf/2509.09707)  

**Abstract**: Integrating Large Language Models (LLMs) within metaheuristics opens a novel path for solving complex combinatorial optimization problems. While most existing approaches leverage LLMs for code generation to create or refine specific heuristics, they often overlook the structural properties of individual problem instances. In this work, we introduce a novel framework that integrates LLMs with a Biased Random-Key Genetic Algorithm (BRKGA) to solve the NP-hard Longest Run Subsequence problem. Our approach extends the instance-driven heuristic bias paradigm by introducing a human-LLM collaborative process to co-design and implement a set of computationally efficient metrics. The LLM analyzes these instance-specific metrics to generate a tailored heuristic bias, which steers the BRKGA toward promising areas of the search space. We conduct a comprehensive experimental evaluation, including rigorous statistical tests, convergence and behavioral analyses, and targeted ablation studies, comparing our method against a standard BRKGA baseline across 1,050 generated instances of varying complexity. Results show that our top-performing hybrid, BRKGA+Llama-4-Maverick, achieves statistically significant improvements over the baseline, particularly on the most complex instances. Our findings confirm that leveraging an LLM to produce an a priori, instance-driven heuristic bias is a valuable approach for enhancing metaheuristics in complex optimization domains. 

---
# Differential Robustness in Transformer Language Models: Empirical Evaluation Under Adversarial Text Attacks 

**Authors**: Taniya Gidatkar, Oluwaseun Ajao, Matthew Shardlow  

**Link**: [PDF](https://arxiv.org/pdf/2509.09706)  

**Abstract**: This study evaluates the resilience of large language models (LLMs) against adversarial attacks, specifically focusing on Flan-T5, BERT, and RoBERTa-Base. Using systematically designed adversarial tests through TextFooler and BERTAttack, we found significant variations in model robustness. RoBERTa-Base and FlanT5 demonstrated remarkable resilience, maintaining accuracy even when subjected to sophisticated attacks, with attack success rates of 0%. In contrast. BERT-Base showed considerable vulnerability, with TextFooler achieving a 93.75% success rate in reducing model accuracy from 48% to just 3%. Our research reveals that while certain LLMs have developed effective defensive mechanisms, these safeguards often require substantial computational resources. This study contributes to the understanding of LLM security by identifying existing strengths and weaknesses in current safeguarding approaches and proposes practical recommendations for developing more efficient and effective defensive strategies. 

---
# Personas within Parameters: Fine-Tuning Small Language Models with Low-Rank Adapters to Mimic User Behaviors 

**Authors**: Himanshu Thakur, Eshani Agrawal, Smruthi Mukund  

**Link**: [PDF](https://arxiv.org/pdf/2509.09689)  

**Abstract**: A long-standing challenge in developing accurate recommendation models is simulating user behavior, mainly due to the complex and stochastic nature of user interactions. Towards this, one promising line of work has been the use of Large Language Models (LLMs) for simulating user behavior. However, aligning these general-purpose large pre-trained models with user preferences necessitates: (i) effectively and continously parsing large-scale tabular user-item interaction data, (ii) overcoming pre-training-induced inductive biases to accurately learn user specific knowledge, and (iii) achieving the former two at scale for millions of users. While most previous works have focused on complex methods to prompt an LLM or fine-tune it on tabular interaction datasets, our approach shifts the focus to extracting robust textual user representations using a frozen LLM and simulating cost-effective, resource-efficient user agents powered by fine-tuned Small Language Models (SLMs). Further, we showcase a method for training multiple low-rank adapters for groups of users or \textit{persona}, striking an optimal balance between scalability and performance of user behavior agents. Our experiments provide compelling empirical evidence of the efficacy of our methods, demonstrating that user agents developed using our approach have the potential to bridge the gap between offline metrics and real-world performance of recommender systems. 

---
# AI-Powered Assistant for Long-Term Access to RHIC Knowledge 

**Authors**: Mohammad Atif, Vincent Garonne, Eric Lancon, Jerome Lauret, Alexandr Prozorov, Michal Vranovsky  

**Link**: [PDF](https://arxiv.org/pdf/2509.09688)  

**Abstract**: As the Relativistic Heavy Ion Collider (RHIC) at Brookhaven National Laboratory concludes 25 years of operation, preserving not only its vast data holdings ($\sim$1 ExaByte) but also the embedded scientific knowledge becomes a critical priority. The RHIC Data and Analysis Preservation Plan (DAPP) introduces an AI-powered assistant system that provides natural language access to documentation, workflows, and software, with the aim of supporting reproducibility, education, and future discovery. Built upon Large Language Models using Retrieval-Augmented Generation and the Model Context Protocol, this assistant indexes structured and unstructured content from RHIC experiments and enables domain-adapted interaction. We report on the deployment, computational performance, ongoing multi-experiment integration, and architectural features designed for a sustainable and explainable long-term AI access. Our experience illustrates how modern AI/ML tools can transform the usability and discoverability of scientific legacy data. 

---
# Text-to-SQL Oriented to the Process Mining Domain: A PT-EN Dataset for Query Translation 

**Authors**: Bruno Yui Yamate, Thais Rodrigues Neubauer, Marcelo Fantinato, Sarajane Marques Peres  

**Link**: [PDF](https://arxiv.org/pdf/2509.09684)  

**Abstract**: This paper introduces text-2-SQL-4-PM, a bilingual (Portuguese-English) benchmark dataset designed for the text-to-SQL task in the process mining domain. Text-to-SQL conversion facilitates natural language querying of databases, increasing accessibility for users without SQL expertise and productivity for those that are experts. The text-2-SQL-4-PM dataset is customized to address the unique challenges of process mining, including specialized vocabularies and single-table relational structures derived from event logs. The dataset comprises 1,655 natural language utterances, including human-generated paraphrases, 205 SQL statements, and ten qualifiers. Methods include manual curation by experts, professional translations, and a detailed annotation process to enable nuanced analyses of task complexity. Additionally, a baseline study using GPT-3.5 Turbo demonstrates the feasibility and utility of the dataset for text-to-SQL applications. The results show that text-2-SQL-4-PM supports evaluation of text-to-SQL implementations, offering broader applicability for semantic parsing and other natural language processing tasks. 

---
# DB3 Team's Solution For Meta KDD Cup' 25 

**Authors**: Yikuan Xia, Jiazun Chen, Yirui Zhan, Suifeng Zhao, Weipeng Jiang, Chaorui Zhang, Wei Han, Bo Bai, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.09681)  

**Abstract**: This paper presents the db3 team's winning solution for the Meta CRAG-MM Challenge 2025 at KDD Cup'25. Addressing the challenge's unique multi-modal, multi-turn question answering benchmark (CRAG-MM), we developed a comprehensive framework that integrates tailored retrieval pipelines for different tasks with a unified LLM-tuning approach for hallucination control. Our solution features (1) domain-specific retrieval pipelines handling image-indexed knowledge graphs, web sources, and multi-turn conversations; and (2) advanced refusal training using SFT, DPO, and RL. The system achieved 2nd place in Task 1, 2nd place in Task 2, and 1st place in Task 3, securing the grand prize for excellence in ego-centric queries through superior handling of first-person perspective challenges. 

---
# Clip Your Sequences Fairly: Enforcing Length Fairness for Sequence-Level RL 

**Authors**: Hanyi Mao, Quanjia Xiao, Lei Pang, Haixiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09177)  

**Abstract**: We propose FSPO (Fair Sequence Policy Optimization), a sequence-level reinforcement learning method for LLMs that enforces length-fair clipping directly in the importance-sampling (IS) weight space. We revisit sequence-level RL methods and identify a mismatch when PPO/GRPO-style clipping is transplanted to sequences: a fixed clip range systematically reweights short vs. long responses, distorting the effective objective. Theoretically, we formalize length fairness via a Length Reweighting Error (LRE) and prove that small LRE yields a directional cosine guarantee between the clipped and true updates. FSPO introduces a simple, Gaussian-motivated remedy: we clip the sequence log-IS ratio with a band that applies a KL-corrected drift term and scales as $\sqrt{L}$. Empirically, FSPO flattens clip rates across length bins, stabilizes training, and outperforms all baselines across multiple evaluation datasets. 

---
