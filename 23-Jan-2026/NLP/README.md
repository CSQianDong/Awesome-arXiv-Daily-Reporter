# LLM-in-Sandbox Elicits General Agentic Intelligence 

**Authors**: Daixuan Cheng, Shaohan Huang, Yuxian Gu, Huatong Song, Guoxin Chen, Li Dong, Wayne Xin Zhao, Ji-Rong Wen, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.16206)  

**Abstract**: We introduce LLM-in-Sandbox, enabling LLMs to explore within a code sandbox (i.e., a virtual computer), to elicit general intelligence in non-code domains. We first demonstrate that strong LLMs, without additional training, exhibit generalization capabilities to leverage the code sandbox for non-code tasks. For example, LLMs spontaneously access external resources to acquire new knowledge, leverage the file system to handle long contexts, and execute scripts to satisfy formatting requirements. We further show that these agentic capabilities can be enhanced through LLM-in-Sandbox Reinforcement Learning (LLM-in-Sandbox-RL), which uses only non-agentic data to train models for sandbox exploration. Experiments demonstrate that LLM-in-Sandbox, in both training-free and post-trained settings, achieves robust generalization spanning mathematics, physics, chemistry, biomedicine, long-context understanding, and instruction following. Finally, we analyze LLM-in-Sandbox's efficiency from computational and system perspectives, and open-source it as a Python package to facilitate real-world deployment. 

---
# Automatic Classification of Arabic Literature into Historical Eras 

**Authors**: Zainab Alhathloul, Irfan Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2601.16138)  

**Abstract**: The Arabic language has undergone notable transformations over time, including the emergence of new vocabulary, the obsolescence of others, and shifts in word usage. This evolution is evident in the distinction between the classical and modern Arabic eras. Although historians and linguists have partitioned Arabic literature into multiple eras, relatively little research has explored the automatic classification of Arabic texts by time period, particularly beyond the domain of poetry. This paper addresses this gap by employing neural networks and deep learning techniques to automatically classify Arabic texts into distinct eras and periods. The proposed models are evaluated using two datasets derived from two publicly available corpora, covering texts from the pre-Islamic to the modern era. The study examines class setups ranging from binary to 15-class classification and considers both predefined historical eras and custom periodizations. Results range from F1-scores of 0.83 and 0.79 on the binary-era classification task using the OpenITI and APCD datasets, respectively, to 0.20 on the 15-era classification task using OpenITI and 0.18 on the 12-era classification task using APCD. 

---
# Improving Training Efficiency and Reducing Maintenance Costs via Language Specific Model Merging 

**Authors**: Alphaeus Dmonte, Vidhi Gupta, Daniel J Perry, Mark Arehart  

**Link**: [PDF](https://arxiv.org/pdf/2601.16127)  

**Abstract**: Fine-tuning a task-specific multilingual large language model (LLM) involves training the model on a multilingual dataset with examples in all the required languages. Updating one or more supported languages with additional data or adding support for a new language involves retraining the model, which can be computationally inefficient and creates a severe maintenance bottleneck. Recent research on merging multilingual multitask models has shown promise in terms of improved quality, but its computational and maintenance efficiency remains unstudied. In this work, we provide the first focused analysis of this merging strategy from an efficiency perspective, evaluating it across three independent tasks. We demonstrate significant efficiency gains while maintaining parity in terms of quality: this merging approach reduces the initial training time by up to 50\%. We also demonstrate that updating an individual language and re-merging as part of model maintenance reduces training costs by more than 60\%, compared to re-training the full multilingual model. We show this on both public and proprietary industry datasets confirming that the approach works well for industrial use cases in addition to academic settings already studied in previous work. 

---
# synthocr-gen: A synthetic ocr dataset generator for low-resource languages- breaking the data barrier 

**Authors**: Haq Nawaz Malik, Kh Mohmad Shafi, Tanveer Ahmad Reshi  

**Link**: [PDF](https://arxiv.org/pdf/2601.16113)  

**Abstract**: Optical Character Recognition (OCR) for low-resource languages remains a significant challenge due to the scarcity of large-scale annotated training datasets. Languages such as Kashmiri, with approximately 7 million speakers and a complex Perso-Arabic script featuring unique diacritical marks, currently lack support in major OCR systems including Tesseract, TrOCR, and PaddleOCR. Manual dataset creation for such languages is prohibitively expensive, time-consuming, and error-prone, often requiring word by word transcription of printed or handwritten text.
We present SynthOCR-Gen, an open-source synthetic OCR dataset generator specifically designed for low-resource languages. Our tool addresses the fundamental bottleneck in OCR development by transforming digital Unicode text corpora into ready-to-use training datasets. The system implements a comprehensive pipeline encompassing text segmentation (character, word, n-gram, sentence, and line levels), Unicode normalization with script purity enforcement, multi-font rendering with configurable distribution, and 25+ data augmentation techniques simulating real-world document degradations including rotation, blur, noise, and scanner artifacts.
We demonstrate the efficacy of our approach by generating a 600,000-sample word-segmented Kashmiri OCR dataset, which we release publicly on HuggingFace. This work provides a practical pathway for bringing low-resource languages into the era of vision-language AI models, and the tool is openly available for researchers and practitioners working with underserved writing systems worldwide. 

---
# Adapter Fusion for Multilingual Text2Cypher with Linear and Learned Gating 

**Authors**: Makbule Gulcin Ozsoy  

**Link**: [PDF](https://arxiv.org/pdf/2601.16097)  

**Abstract**: Large Language Models enable users to access database using natural language interfaces using tools like Text2SQL, Text2SPARQL, and Text2Cypher, which translate user questions into structured database queries. While these systems improve database accessibility, most research focuses on English with limited multilingual support. This work investigates a scalable multilingual Text2Cypher, aiming to support new languages without re-running full fine-tuning, avoiding manual hyper-parameter tuning, and maintaining performance close to joint multilingual fine-tuning. We train language-specific LoRA adapters for English, Spanish, and Turkish and combined them via uniform linear merging or learned fusion MLP with dynamic gating. Experimental results show that the fusion MLP recovers around 75\% of the accuracy gains from joint multilingual fine-tuning while requiring only a smaller subset of the data, outperforming linear merging across all three languages. This approach enables incremental language expansion to new languages by requiring only one LoRA adapter and a lightweight MLP retraining. Learned adapter fusion offers a practical alternative to expensive joint fine-tuning, balancing performance, data efficiency, and scalability for multilingual Text2Cypher task. 

---
# Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay and Concept-Basis Reconstruction 

**Authors**: Tony Cristofano  

**Link**: [PDF](https://arxiv.org/pdf/2601.16034)  

**Abstract**: Refusal behavior in aligned LLMs is often viewed as model-specific, yet we hypothesize it stems from a universal, low-dimensional semantic circuit shared across models. To test this, we introduce Trajectory Replay via Concept-Basis Reconstruction, a framework that transfers refusal interventions from donor to target models, spanning diverse architectures (e.g., Dense to MoE) and training regimes, without using target-side refusal supervision. By aligning layers via concept fingerprints and reconstructing refusal directions using a shared ``recipe'' of concept atoms, we map the donor's ablation trajectory into the target's semantic space. To preserve capabilities, we introduce a weight-SVD stability guard that projects interventions away from high-variance weight subspaces to prevent collateral damage. Our evaluation across 8 model pairs (including GPT-OSS-20B and GLM-4) confirms that these transferred recipes consistently attenuate refusal while maintaining performance, providing strong evidence for the semantic universality of safety alignment. 

---
# Mecellem Models: Turkish Models Trained from Scratch and Continually Pre-trained for the Legal Domain 

**Authors**: Özgür Uğur, Mahmut Göksu, Mahmut Çimen, Musa Yılmaz, Esra Şavirdi, Alp Talha Demir, Rumeysa Güllüce, İclal Çetin, Ömer Can Sağbaş  

**Link**: [PDF](https://arxiv.org/pdf/2601.16018)  

**Abstract**: This paper presents Mecellem models, a framework for developing specialized language models for the Turkish legal domain through domain adaptation strategies. We make two contributions: (1)Encoder Model Pre-trained from Scratch: ModernBERT-based bidirectional encoders pre-trained on a Turkish-dominant corpus of 112.7 billion tokens. We implement a checkpoint selection strategy that evaluates downstream retrieval performance throughout training, revealing that optimal checkpoints achieve best retrieval scores before pre-training loss reaches its minimum. Our encoder models achieve top-3 rankings on the Turkish retrieval leaderboard, with smaller models (155M parameters) achieving comparable performance to larger reference models (307M-567M parameters). Our approach achieves 92.36% production efficiency compared to state-of-the-art models (embeddinggemma-300m: 100.00%, BAAI/bge-m3: 99.54%, newmindai/bge-m3-stsb: 94.38%), ranking fourth overall despite requiring less computational resources. SOTA models rely on multi-stage, computationally intensive training pipelines, making our single-stage pre-training followed by efficient post-training approach a cost-effective alternative; (2)Decoder Model with Continual Pre-training (CPT): Qwen3-1.7B and Qwen3-4B models adapted to Turkish legal domain through controlled curriculum learning. Four-phase CPT with optimal sample ratios enables gradual transition from general language knowledge to specialized legal terminology and long-context reasoning. This approach achieves 36.2% perplexity reduction on Turkish legal text, demonstrating domain adaptation gains. 

---
# Transfer Learning from ImageNet for MEG-Based Decoding of Imagined Speech 

**Authors**: Soufiane Jhilal, Stéphanie Martin, Anne-Lise Giraud  

**Link**: [PDF](https://arxiv.org/pdf/2601.15909)  

**Abstract**: Non-invasive decoding of imagined speech remains challenging due to weak, distributed signals and limited labeled data. Our paper introduces an image-based approach that transforms magnetoencephalography (MEG) signals into time-frequency representations compatible with pretrained vision models. MEG data from 21 participants performing imagined speech tasks were projected into three spatial scalogram mixtures via a learnable sensor-space convolution, producing compact image-like inputs for ImageNet-pretrained vision architectures. These models outperformed classical and non-pretrained models, achieving up to 90.4% balanced accuracy for imagery vs. silence, 81.0% vs. silent reading, and 60.6% for vowel decoding. Cross-subject evaluation confirmed that pretrained models capture shared neural representations, and temporal analyses localized discriminative information to imagery-locked intervals. These findings show that pretrained vision models applied to image-based MEG representations can effectively capture the structure of imagined speech in non-invasive neural signals. 

---
# Stable-DiffCoder: Pushing the Frontier of Code Diffusion Large Language Model 

**Authors**: Chenghao Fan, Wen Heng, Bo Li, Sichen Liu, Yuxuan Song, Jing Su, Xiaoye Qu, Kai Shen, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.15892)  

**Abstract**: Diffusion-based language models (DLLMs) offer non-sequential, block-wise generation and richer data reuse compared to autoregressive (AR) models, but existing code DLLMs still lag behind strong AR baselines under comparable budgets. We revisit this setting in a controlled study and introduce Stable-DiffCoder, a block diffusion code model that reuses the Seed-Coder architecture, data, and training pipeline. To enable efficient knowledge learning and stable training, we incorporate a block diffusion continual pretraining (CPT) stage enhanced by a tailored warmup and block-wise clipped noise schedule. Under the same data and architecture, Stable-DiffCoder overall outperforms its AR counterpart on a broad suite of code benchmarks. Moreover, relying only on the CPT and supervised fine-tuning stages, Stable-DiffCoder achieves stronger performance than a wide range of \~8B ARs and DLLMs, demonstrating that diffusion-based training can improve code modeling quality beyond AR training alone. Moreover, diffusion-based any-order modeling improves structured code modeling for editing and reasoning, and through data augmentation, benefits low-resource coding languages. 

---
# Artificial Rigidities vs. Biological Noise: A Comparative Analysis of Multisensory Integration in AV-HuBERT and Human Observers 

**Authors**: Francisco Portillo López  

**Link**: [PDF](https://arxiv.org/pdf/2601.15869)  

**Abstract**: This study evaluates AV-HuBERT's perceptual bio-fidelity by benchmarking its response to incongruent audiovisual stimuli (McGurk effect) against human observers (N=44). Results reveal a striking quantitative isomorphism: AI and humans exhibited nearly identical auditory dominance rates (32.0% vs. 31.8%), suggesting the model captures biological thresholds for auditory resistance. However, AV-HuBERT showed a deterministic bias toward phonetic fusion (68.0%), significantly exceeding human rates (47.7%). While humans displayed perceptual stochasticity and diverse error profiles, the model remained strictly categorical. Findings suggest that current self-supervised architectures mimic multisensory outcomes but lack the neural variability inherent to human speech perception. 

---
# Determinants of Training Corpus Size for Clinical Text Classification 

**Authors**: Jaya Chaturvedi, Saniya Deshpande, Chenkai Ma, Robert Cobb, Angus Roberts, Robert Stewart, Daniel Stahl, Diana Shamsutdinova  

**Link**: [PDF](https://arxiv.org/pdf/2601.15846)  

**Abstract**: Introduction: Clinical text classification using natural language processing (NLP) models requires adequate training data to achieve optimal performance. For that, 200-500 documents are typically annotated. The number is constrained by time and costs and lacks justification of the sample size requirements and their relationship to text vocabulary properties.
Methods: Using the publicly available MIMIC-III dataset containing hospital discharge notes with ICD-9 diagnoses as labels, we employed pre-trained BERT embeddings followed by Random Forest classifiers to identify 10 randomly selected diagnoses, varying training corpus sizes from 100 to 10,000 documents, and analyzed vocabulary properties by identifying strong and noisy predictive words through Lasso logistic regression on bag-of-words embeddings.
Results: Learning curves varied significantly across the 10 classification tasks despite identical preprocessing and algorithms, with 600 documents sufficient to achieve 95% of the performance attainable with 10,000 documents for all tasks. Vocabulary analysis revealed that more strong predictors and fewer noisy predictors were associated with steeper learning curves, where every 100 additional noisy words decreased accuracy by approximately 0.02 while 100 additional strong predictors increased maximum accuracy by approximately 0.04. 

---
# Can professional translators identify machine-generated text? 

**Authors**: Michael Farrell  

**Link**: [PDF](https://arxiv.org/pdf/2601.15828)  

**Abstract**: This study investigates whether professional translators can reliably identify short stories generated in Italian by artificial intelligence (AI) without prior specialized training. Sixty-nine translators took part in an in-person experiment, where they assessed three anonymized short stories - two written by ChatGPT-4o and one by a human author. For each story, participants rated the likelihood of AI authorship and provided justifications for their choices. While average results were inconclusive, a statistically significant subset (16.2%) successfully distinguished the synthetic texts from the human text, suggesting that their judgements were informed by analytical skill rather than chance. However, a nearly equal number misclassified the texts in the opposite direction, often relying on subjective impressions rather than objective markers, possibly reflecting a reader preference for AI-generated texts. Low burstiness and narrative contradiction emerged as the most reliable indicators of synthetic authorship, with unexpected calques, semantic loans and syntactic transfer from English also reported. In contrast, features such as grammatical accuracy and emotional tone frequently led to misclassification. These findings raise questions about the role and scope of synthetic-text editing in professional contexts. 

---
# ExDR: Explanation-driven Dynamic Retrieval Enhancement for Multimodal Fake News Detection 

**Authors**: Guoxuan Ding, Yuqing Li, Ziyan Zhou, Zheng Lin, Daren Zha, Jiangnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.15820)  

**Abstract**: The rapid spread of multimodal fake news poses a serious societal threat, as its evolving nature and reliance on timely factual details challenge existing detection methods. Dynamic Retrieval-Augmented Generation provides a promising solution by triggering keyword-based retrieval and incorporating external knowledge, thus enabling both efficient and accurate evidence selection. However, it still faces challenges in addressing issues such as redundant retrieval, coarse similarity, and irrelevant evidence when applied to deceptive content. In this paper, we propose ExDR, an Explanation-driven Dynamic Retrieval-Augmented Generation framework for Multimodal Fake News Detection. Our framework systematically leverages model-generated explanations in both the retrieval triggering and evidence retrieval modules. It assesses triggering confidence from three complementary dimensions, constructs entity-aware indices by fusing deceptive entities, and retrieves contrastive evidence based on deception-specific features to challenge the initial claim and enhance the final prediction. Experiments on two benchmark datasets, AMG and MR2, demonstrate that ExDR consistently outperforms previous methods in retrieval triggering accuracy, retrieval quality, and overall detection performance, highlighting its effectiveness and generalization capability. 

---
# SteerEval: Inference-time Interventions Strengthen Multilingual Generalization in Neural Summarization Metrics 

**Authors**: Silvia Casola, Ryan Soh-Eun Shim, Felicia Körner, Yuchen Mao, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2601.15809)  

**Abstract**: An increasing body of work has leveraged multilingual language models for Natural Language Generation tasks such as summarization. A major empirical bottleneck in this area is the shortage of accurate and robust evaluation metrics for many languages, which hinders progress. Recent studies suggest that multilingual language models often use English as an internal pivot language, and that misalignment with this pivot can lead to degraded downstream performance. Motivated by the hypothesis that this mismatch could also apply to multilingual neural metrics, we ask whether steering their activations toward an English pivot can improve correlation with human judgments. We experiment with encoder- and decoder-based metrics and find that test-time intervention methods are effective across the board, increasing metric effectiveness for diverse languages. 

---
# HumanLLM: Towards Personalized Understanding and Simulation of Human Nature 

**Authors**: Yuxuan Lei, Tianfu Wang, Jianxun Lian, Zhengyu Hu, Defu Lian, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.15793)  

**Abstract**: Motivated by the remarkable progress of large language models (LLMs) in objective tasks like mathematics and coding, there is growing interest in their potential to simulate human behavior--a capability with profound implications for transforming social science research and customer-centric business insights. However, LLMs often lack a nuanced understanding of human cognition and behavior, limiting their effectiveness in social simulation and personalized applications. We posit that this limitation stems from a fundamental misalignment: standard LLM pretraining on vast, uncontextualized web data does not capture the continuous, situated context of an individual's decisions, thoughts, and behaviors over time. To bridge this gap, we introduce HumanLLM, a foundation model designed for personalized understanding and simulation of individuals. We first construct the Cognitive Genome Dataset, a large-scale corpus curated from real-world user data on platforms like Reddit, Twitter, Blogger, and Amazon. Through a rigorous, multi-stage pipeline involving data filtering, synthesis, and quality control, we automatically extract over 5.5 million user logs to distill rich profiles, behaviors, and thinking patterns. We then formulate diverse learning tasks and perform supervised fine-tuning to empower the model to predict a wide range of individualized human behaviors, thoughts, and experiences. Comprehensive evaluations demonstrate that HumanLLM achieves superior performance in predicting user actions and inner thoughts, more accurately mimics user writing styles and preferences, and generates more authentic user profiles compared to base models. Furthermore, HumanLLM shows significant gains on out-of-domain social intelligence benchmarks, indicating enhanced generalization. 

---
# Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs 

**Authors**: Tristan Williams, Franziska Weeber, Sebastian Padó, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2601.15755)  

**Abstract**: Large language models are increasingly used to represent human opinions, values, or beliefs, and their steerability towards these ideals is an active area of research. Existing work focuses predominantly on aligning marginal response distributions, treating each survey item independently. While essential, this may overlook deeper latent structures that characterise real populations and underpin cultural values theories. We propose a framework for evaluating the representativeness of aligned models through multivariate correlation patterns in addition to marginal distributions. We show the value of our evaluation scheme by comparing two model steering techniques (persona prompting and demographic fine-tuning) and evaluating them against human responses from the World Values Survey. While the demographically fine-tuned model better approximates marginal response distributions than persona prompting, both techniques fail to fully capture the gold standard correlation patterns. We conclude that representativeness is a distinct aspect of value alignment and an evaluation focused on marginals can mask structural failures, leading to overly optimistic conclusions about model capabilities. 

---
# Hallucination Mitigating for Medical Report Generation 

**Authors**: Ruoqing Zhao, Runze Xia, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.15745)  

**Abstract**: In the realm of medical report generation (MRG), the integration of natural language processing has emerged as a vital tool to alleviate the workload of radiologists. Despite the impressive capabilities demonstrated by large vision language models (LVLMs) in understanding natural language, their susceptibility to generating plausible yet inaccurate claims, known as ``hallucinations'', raises concerns-especially in the nuanced and critical field of medical. In this work, we introduce a framework, \textbf{K}nowledge-\textbf{E}nhanced with Fine-Grained \textbf{R}einforced Rewards \textbf{M}edical Report Generation (KERM), to tackle the issue. Our approach refines the input to the LVLM by first utilizing MedCLIP for knowledge retrieval, incorporating relevant lesion fact sentences from a curated knowledge corpus. We then introduce a novel purification module to ensure the retrieved knowledge is contextually relevant to the patient's clinical context. Subsequently, we employ fine-grained rewards to guide these models in generating highly supportive and clinically relevant descriptions, ensuring the alignment of model's outputs with desired behaviors. Experimental results on IU-Xray and MIMIC-CXR datasets validate the effectiveness of our approach in mitigating hallucinations and enhancing report quality. 

---
# Dancing in Chains: Strategic Persuasion in Academic Rebuttal via Theory of Mind 

**Authors**: Zhitao He, Zongwei Lyu, Yi R Fung  

**Link**: [PDF](https://arxiv.org/pdf/2601.15715)  

**Abstract**: Although artificial intelligence (AI) has become deeply integrated into various stages of the research workflow and achieved remarkable advancements, academic rebuttal remains a significant and underexplored challenge. This is because rebuttal is a complex process of strategic communication under severe information asymmetry rather than a simple technical debate. Consequently, current approaches struggle as they largely imitate surface-level linguistics, missing the essential element of perspective-taking required for effective persuasion. In this paper, we introduce RebuttalAgent, the first framework to ground academic rebuttal in Theory of Mind (ToM), operationalized through a ToM-Strategy-Response (TSR) pipeline that models reviewer mental state, formulates persuasion strategy, and generates strategy-grounded response. To train our agent, we construct RebuttalBench, a large-scale dataset synthesized via a novel critique-and-refine approach. Our training process consists of two stages, beginning with a supervised fine-tuning phase to equip the agent with ToM-based analysis and strategic planning capabilities, followed by a reinforcement learning phase leveraging the self-reward mechanism for scalable self-improvement. For reliable and efficient automated evaluation, we further develop Rebuttal-RM, a specialized evaluator trained on over 100K samples of multi-source rebuttal data, which achieves scoring consistency with human preferences surpassing powerful judge GPT-4.1. Extensive experiments show RebuttalAgent significantly outperforms the base model by an average of 18.3% on automated metrics, while also outperforming advanced proprietary models across both automated and human evaluations. Disclaimer: the generated rebuttal content is for reference only to inspire authors and assist in drafting. It is not intended to replace the author's own critical analysis and response. 

---
# Persona Switch: Mixing Distinct Perspectives in Decoding Time 

**Authors**: Junseok Kim, Nakyeong Yang, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2601.15708)  

**Abstract**: Role-play prompting is known to steer the behavior of language models by injecting a persona into the prompt, improving their zero-shot reasoning capabilities. However, such improvements are inconsistent across different tasks or instances. This inconsistency suggests that zero-shot and role-play prompting may offer complementary strengths rather than one being universally superior. Building on this insight, we propose Persona Switch, a novel decoding method that dynamically combines the benefits of both prompting strategies. Our method proceeds step-by-step, selecting the better output between zero-shot and role-play prompting at each step by comparing their output confidence, as measured by the logit gap. Experiments with widely-used LLMs demonstrate that Persona Switch consistently outperforms competitive baselines, achieving up to 5.13% accuracy improvement. Furthermore, we show that output confidence serves as an informative measure for selecting the more reliable output. 

---
# What Patients Really Ask: Exploring the Effect of False Assumptions in Patient Information Seeking 

**Authors**: Raymond Xiong, Furong Jia, Lionel Wong, Monica Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2601.15674)  

**Abstract**: Patients are increasingly using large language models (LLMs) to seek answers to their healthcare-related questions. However, benchmarking efforts in LLMs for question answering often focus on medical exam questions, which differ significantly in style and content from the questions patients actually raise in real life. To bridge this gap, we sourced data from Google's People Also Ask feature by querying the top 200 prescribed medications in the United States, curating a dataset of medical questions people commonly ask. A considerable portion of the collected questions contains incorrect assumptions and dangerous intentions. We demonstrate that the emergence of these corrupted questions is not uniformly random and depends heavily on the degree of incorrectness in the history of questions that led to their appearance. Current LLMs that perform strongly on other benchmarks struggle to identify incorrect assumptions in everyday questions. 

---
# Towards Reliable Medical LLMs: Benchmarking and Enhancing Confidence Estimation of Large Language Models in Medical Consultation 

**Authors**: Zhiyao Ren, Yibing Zhan, Siyuan Liang, Guozheng Ma, Baosheng Yu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2601.15645)  

**Abstract**: Large-scale language models (LLMs) often offer clinical judgments based on incomplete information, increasing the risk of misdiagnosis. Existing studies have primarily evaluated confidence in single-turn, static settings, overlooking the coupling between confidence and correctness as clinical evidence accumulates during real consultations, which limits their support for reliable decision-making. We propose the first benchmark for assessing confidence in multi-turn interaction during realistic medical consultations. Our benchmark unifies three types of medical data for open-ended diagnostic generation and introduces an information sufficiency gradient to characterize the confidence-correctness dynamics as evidence increases. We implement and compare 27 representative methods on this benchmark; two key insights emerge: (1) medical data amplifies the inherent limitations of token-level and consistency-level confidence methods, and (2) medical reasoning must be evaluated for both diagnostic accuracy and information completeness. Based on these insights, we present MedConf, an evidence-grounded linguistic self-assessment framework that constructs symptom profiles via retrieval-augmented generation, aligns patient information with supporting, missing, and contradictory relations, and aggregates them into an interpretable confidence estimate through weighted integration. Across two LLMs and three medical datasets, MedConf consistently outperforms state-of-the-art methods on both AUROC and Pearson correlation coefficient metrics, maintaining stable performance under conditions of information insufficiency and multimorbidity. These results demonstrate that information adequacy is a key determinant of credible medical confidence modeling, providing a new pathway toward building more reliable and interpretable large medical models. 

---
# ToxiTwitch: Toward Emote-Aware Hybrid Moderation for Live Streaming Platforms 

**Authors**: Baktash Ansari, Shiza Ali, Elias Martin, Maryna Sivachenko, Afra Mashhadi  

**Link**: [PDF](https://arxiv.org/pdf/2601.15605)  

**Abstract**: The rapid growth of live-streaming platforms such as Twitch has introduced complex challenges in moderating toxic behavior. Traditional moderation approaches, such as human annotation and keyword-based filtering, have demonstrated utility, but human moderators on Twitch constantly struggle to scale effectively in the fast-paced, high-volume, and context-rich chat environment of the platform while also facing harassment themselves. Recent advances in large language models (LLMs), such as DeepSeek-R1-Distill and Llama-3-8B-Instruct, offer new opportunities for toxicity detection, especially in understanding nuanced, multimodal communication involving emotes. In this work, we present an exploratory comparison of toxicity detection approaches tailored to Twitch. Our analysis reveals that incorporating emotes improves the detection of toxic behavior. To this end, we introduce ToxiTwitch, a hybrid model that combines LLM-generated embeddings of text and emotes with traditional machine learning classifiers, including Random Forest and SVM. In our case study, the proposed hybrid approach reaches up to 80 percent accuracy under channel-specific training (with 13 percent improvement over BERT and F1-score of 76 percent). This work is an exploratory study intended to surface challenges and limits of emote-aware toxicity detection on Twitch. 

---
# Parallelism and Generation Order in Masked Diffusion Language Models: Limits Today, Potential Tomorrow 

**Authors**: Yangyang Zhong, Yanmei Gu, Zhengqing Zang, Xiaomeng Li, Yuqi Ding, Xibei Jia, Yuting Shen, Zhenzhong Lan, Liwang Zhu, Weiping Liu, Junlin Zhou, Haisheng Liu, Zhong Xin Yu, Pengxin Luo, Donglian Qi, Yunfeng Yan, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.15593)  

**Abstract**: Masked Diffusion Language Models (MDLMs) promise parallel token generation and arbitrary-order decoding, yet it remains unclear to what extent current models truly realize these capabilities. We characterize MDLM behavior along two dimensions -- parallelism strength and generation order -- using Average Finalization Parallelism (AFP) and Kendall's tau. We evaluate eight mainstream MDLMs (up to 100B parameters) on 58 benchmarks spanning knowledge, reasoning, and programming. The results show that MDLMs still lag behind comparably sized autoregressive models, mainly because parallel probabilistic modeling weakens inter-token dependencies. Meanwhile, MDLMs exhibit adaptive decoding behavior: their parallelism and generation order vary significantly with the task domain, the stage of reasoning, and whether the output is correct. On tasks that require "backward information" (e.g., Sudoku), MDLMs adopt a solution order that tends to fill easier Sudoku blanks first, highlighting their advantages. Finally, we provide theoretical motivation and design insights supporting a Generate-then-Edit paradigm, which mitigates dependency loss while retaining the efficiency of parallel decoding. 

---
# YuFeng-XGuard: A Reasoning-Centric, Interpretable, and Flexible Guardrail Model for Large Language Models 

**Authors**: Junyu Lin, Meizhen Liu, Xiufeng Huang, Jinfeng Li, Haiwen Hong, Xiaohan Yuan, Yuefeng Chen, Longtao Huang, Hui Xue, Ranjie Duan, Zhikai Chen, Yuchuan Fu, Defeng Li, Lingyao Gao, Yitong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.15588)  

**Abstract**: As large language models (LLMs) are increasingly deployed in real-world applications, safety guardrails are required to go beyond coarse-grained filtering and support fine-grained, interpretable, and adaptable risk assessment. However, existing solutions often rely on rapid classification schemes or post-hoc rules, resulting in limited transparency, inflexible policies, or prohibitive inference costs. To this end, we present YuFeng-XGuard, a reasoning-centric guardrail model family designed to perform multi-dimensional risk perception for LLM interactions. Instead of producing opaque binary judgments, YuFeng-XGuard generates structured risk predictions, including explicit risk categories and configurable confidence scores, accompanied by natural language explanations that expose the underlying reasoning process. This formulation enables safety decisions that are both actionable and interpretable. To balance decision latency and explanatory depth, we adopt a tiered inference paradigm that performs an initial risk decision based on the first decoded token, while preserving ondemand explanatory reasoning when required. In addition, we introduce a dynamic policy mechanism that decouples risk perception from policy enforcement, allowing safety policies to be adjusted without model retraining. Extensive experiments on a diverse set of public safety benchmarks demonstrate that YuFeng-XGuard achieves stateof-the-art performance while maintaining strong efficiency-efficacy trade-offs. We release YuFeng-XGuard as an open model family, including both a full-capacity variant and a lightweight version, to support a wide range of deployment scenarios. 

---
# From Generation to Collaboration: Using LLMs to Edit for Empathy in Healthcare 

**Authors**: Man Luo, Bahareh Harandizadeh, Amara Tariq, Halim Abbas, Umar Ghaffar, Christopher J Warren, Segun O. Kolade, Haidar M. Abdul-Muhsin  

**Link**: [PDF](https://arxiv.org/pdf/2601.15558)  

**Abstract**: Clinical empathy is essential for patient care, but physicians need continually balance emotional warmth with factual precision under the cognitive and emotional constraints of clinical practice. This study investigates how large language models (LLMs) can function as empathy editors, refining physicians' written responses to enhance empathetic tone while preserving underlying medical information. More importantly, we introduce novel quantitative metrics, an Empathy Ranking Score and a MedFactChecking Score to systematically assess both emotional and factual quality of the responses. Experimental results show that LLM edited responses significantly increase perceived empathy while preserving factual accuracy compared with fully LLM generated outputs. These findings suggest that using LLMs as editorial assistants, rather than autonomous generators, offers a safer, more effective pathway to empathetic and trustworthy AI-assisted healthcare communication. 

---
# Common to Whom? Regional Cultural Commonsense and LLM Bias in India 

**Authors**: Sangmitra Madhusudan, Trush Shashank More, Steph Buongiorno, Renata Dividino, Jad Kabbara, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2601.15550)  

**Abstract**: Existing cultural commonsense benchmarks treat nations as monolithic, assuming uniform practices within national boundaries. But does cultural commonsense hold uniformly within a nation, or does it vary at the sub-national level? We introduce Indica, the first benchmark designed to test LLMs' ability to address this question, focusing on India - a nation of 28 states, 8 union territories, and 22 official languages. We collect human-annotated answers from five Indian regions (North, South, East, West, and Central) across 515 questions spanning 8 domains of everyday life, yielding 1,630 region-specific question-answer pairs. Strikingly, only 39.4% of questions elicit agreement across all five regions, demonstrating that cultural commonsense in India is predominantly regional, not national. We evaluate eight state-of-the-art LLMs and find two critical gaps: models achieve only 13.4%-20.9% accuracy on region-specific questions, and they exhibit geographic bias, over-selecting Central and North India as the "default" (selected 30-40% more often than expected) while under-representing East and West. Beyond India, our methodology provides a generalizable framework for evaluating cultural commonsense in any culturally heterogeneous nation, from question design grounded in anthropological taxonomy, to regional data collection, to bias measurement. 

---
# AdversaRiskQA: An Adversarial Factuality Benchmark for High-Risk Domains 

**Authors**: Adam Szelestey, Sofie van Engelen, Tianhao Huang, Justin Snelders, Qintao Zeng, Songgaojun Deng  

**Link**: [PDF](https://arxiv.org/pdf/2601.15511)  

**Abstract**: Hallucination in large language models (LLMs) remains an acute concern, contributing to the spread of misinformation and diminished public trust, particularly in high-risk domains. Among hallucination types, factuality is crucial, as it concerns a model's alignment with established world knowledge. Adversarial factuality, defined as the deliberate insertion of misinformation into prompts with varying levels of expressed confidence, tests a model's ability to detect and resist confidently framed falsehoods. Existing work lacks high-quality, domain-specific resources for assessing model robustness under such adversarial conditions, and no prior research has examined the impact of injected misinformation on long-form text factuality.
To address this gap, we introduce AdversaRiskQA, the first verified and reliable benchmark systematically evaluating adversarial factuality across Health, Finance, and Law. The benchmark includes two difficulty levels to test LLMs' defensive capabilities across varying knowledge depths. We propose two automated methods for evaluating the adversarial attack success and long-form factuality. We evaluate six open- and closed-source LLMs from the Qwen, GPT-OSS, and GPT families, measuring misinformation detection rates. Long-form factuality is assessed on Qwen3 (30B) under both baseline and adversarial conditions. Results show that after excluding meaningless responses, Qwen3 (80B) achieves the highest average accuracy, while GPT-5 maintains consistently high accuracy. Performance scales non-linearly with model size, varies by domains, and gaps between difficulty levels narrow as models grow. Long-form evaluation reveals no significant correlation between injected misinformation and the model's factual output. AdversaRiskQA provides a valuable benchmark for pinpointing LLM weaknesses and developing more reliable models for high-stakes applications. 

---
# Computational Representations of Character Significance in Novels 

**Authors**: Haaris Mian, Melanie Subbiah, Sharon Marcus, Nora Shaalan, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2601.15508)  

**Abstract**: Characters in novels have typically been modeled based on their presence in scenes in narrative, considering aspects like their actions, named mentions, and dialogue. This conception of character places significant emphasis on the main character who is present in the most scenes. In this work, we instead adopt a framing developed from a new literary theory proposing a six-component structural model of character. This model enables a comprehensive approach to character that accounts for the narrator-character distinction and includes a component neglected by prior methods, discussion by other characters. We compare general-purpose LLMs with task-specific transformers for operationalizing this model of character on major 19th-century British realist novels. Our methods yield both component-level and graph representations of character discussion. We then demonstrate that these representations allow us to approach literary questions at scale from a new computational lens. Specifically, we explore Woloch's classic "the one vs the many" theory of character centrality and the gendered dynamics of character discussion. 

---
# ViT Registers and Fractal ViT 

**Authors**: Jason Chuan-Chih Chou, Abhinav Kumar, Shivank Garg  

**Link**: [PDF](https://arxiv.org/pdf/2601.15506)  

**Abstract**: Drawing inspiration from recent findings including surprisingly decent performance of transformers without positional encoding (NoPE) in the domain of language models and how registers (additional throwaway tokens not tied to input) may improve the performance of large vision transformers (ViTs), we invent and test a variant of ViT called fractal ViT that breaks permutation invariance among the tokens by applying an attention mask between the regular tokens and ``summary tokens'' similar to registers, in isolation or in combination with various positional encodings. These models do not improve upon ViT with registers, highlighting the fact that these findings may be scale, domain, or application-specific. 

---
# Multi-Persona Thinking for Bias Mitigation in Large Language Models 

**Authors**: Yuxing Chen, Guoqing Luo, Zijun Wu, Lili Mou  

**Link**: [PDF](https://arxiv.org/pdf/2601.15488)  

**Abstract**: Large Language Models (LLMs) exhibit significant social biases that can perpetuate harmful stereotypes and unfair outcomes. In this paper, we propose Multi-Persona Thinking (MPT), a novel inference-time framework that leverages dialectical reasoning from multiple perspectives to reduce bias. MPT guides models to adopt contrasting social identities (e.g., male and female) along with a neutral viewpoint, and then engages these personas iteratively to expose and correct biases. Through a dialectical reasoning process, the framework transforms the potential weakness of persona assignment into a strength for bias mitigation. We evaluate MPT on two widely used bias benchmarks across both open-source and closed-source models of varying scales. Our results demonstrate substantial improvements over existing prompting-based strategies: MPT achieves the lowest bias while maintaining core reasoning ability. 

---
# Benchmarking LLMs for Pairwise Causal Discovery in Biomedical and Multi-Domain Contexts 

**Authors**: Sydney Anuyah, Sneha Shajee-Mohan, Ankit-Singh Chauhan, Sunandan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2601.15479)  

**Abstract**: The safe deployment of large language models (LLMs) in high-stakes fields like biomedicine, requires them to be able to reason about cause and effect. We investigate this ability by testing 13 open-source LLMs on a fundamental task: pairwise causal discovery (PCD) from text. Our benchmark, using 12 diverse datasets, evaluates two core skills: 1) \textbf{Causal Detection} (identifying if a text contains a causal link) and 2) \textbf{Causal Extraction} (pulling out the exact cause and effect phrases). We tested various prompting methods, from simple instructions (zero-shot) to more complex strategies like Chain-of-Thought (CoT) and Few-shot In-Context Learning (FICL).
The results show major deficiencies in current models. The best model for detection, DeepSeek-R1-Distill-Llama-70B, only achieved a mean score of 49.57\% ($C_{detect}$), while the best for extraction, Qwen2.5-Coder-32B-Instruct, reached just 47.12\% ($C_{extract}$). Models performed best on simple, explicit, single-sentence relations. However, performance plummeted for more difficult (and realistic) cases, such as implicit relationships, links spanning multiple sentences, and texts containing multiple causal pairs. We provide a unified evaluation framework, built on a dataset validated with high inter-annotator agreement ($\kappa \ge 0.758$), and make all our data, code, and prompts publicly available to spur further research. \href{this https URL}{Code available here: this https URL} 

---
# Chunking, Retrieval, and Re-ranking: An Empirical Evaluation of RAG Architectures for Policy Document Question Answering 

**Authors**: Anuj Maharjan, Umesh Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2601.15457)  

**Abstract**: The integration of Large Language Models (LLMs) into the public health policy sector offers a transformative approach to navigating the vast repositories of regulatory guidance maintained by agencies such as the Centers for Disease Control and Prevention (CDC). However, the propensity for LLMs to generate hallucinations, defined as plausible but factually incorrect assertions, presents a critical barrier to the adoption of these technologies in high-stakes environments where information integrity is non-negotiable. This empirical evaluation explores the effectiveness of Retrieval-Augmented Generation (RAG) architectures in mitigating these risks by grounding generative outputs in authoritative document context. Specifically, this study compares a baseline Vanilla LLM against Basic RAG and Advanced RAG pipelines utilizing cross-encoder re-ranking. The experimental framework employs a Mistral-7B-Instruct-v0.2 model and an all-MiniLM-L6-v2 embedding model to process a corpus of official CDC policy analytical frameworks and guidance documents. The analysis measures the impact of two distinct chunking strategies, recursive character-based and token-based semantic splitting, on system accuracy, measured through faithfulness and relevance scores across a curated set of complex policy scenarios. Quantitative findings indicate that while Basic RAG architectures provide a substantial improvement in faithfulness (0.621) over Vanilla baselines (0.347), the Advanced RAG configuration achieves a superior faithfulness average of 0.797. These results demonstrate that two-stage retrieval mechanisms are essential for achieving the precision required for domain-specific policy question answering, though structural constraints in document segmentation remain a significant bottleneck for multi-step reasoning tasks. 

---
# Domain-Specific Knowledge Graphs in RAG-Enhanced Healthcare LLMs 

**Authors**: Sydney Anuyah, Mehedi Mahmud Kaushik, Hao Dai, Rakesh Shiradkar, Arjan Durresi, Sunandan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2601.15429)  

**Abstract**: Large Language Models (LLMs) generate fluent answers but can struggle with trustworthy, domain-specific reasoning. We evaluate whether domain knowledge graphs (KGs) improve Retrieval-Augmented Generation (RAG) for healthcare by constructing three PubMed-derived graphs: $\mathbb{G}_1$ (T2DM), $\mathbb{G}_2$ (Alzheimer's disease), and $\mathbb{G}_3$ (AD+T2DM). We design two probes: Probe 1 targets merged AD T2DM knowledge, while Probe 2 targets the intersection of $\mathbb{G}_1$ and $\mathbb{G}_2$. Seven instruction-tuned LLMs are tested across retrieval sources {No-RAG, $\mathbb{G}_1$, $\mathbb{G}_2$, $\mathbb{G}_1$ + $\mathbb{G}_2$, $\mathbb{G}_3$, $\mathbb{G}_1$+$\mathbb{G}_2$ + $\mathbb{G}_3$} and three decoding temperatures. Results show that scope alignment between probe and KG is decisive: precise, scope-matched retrieval (notably $\mathbb{G}_2$) yields the most consistent gains, whereas indiscriminate graph unions often introduce distractors that reduce accuracy. Larger models frequently match or exceed KG-RAG with a No-RAG baseline on Probe 1, indicating strong parametric priors, whereas smaller/mid-sized models benefit more from well-scoped retrieval. Temperature plays a secondary role; higher values rarely help. We conclude that precision-first, scope-matched KG-RAG is preferable to breadth-first unions, and we outline practical guidelines for graph selection, model sizing, and retrieval/reranking. Code and Data available here - this https URL 

---
# Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind 

**Authors**: Tamunotonye Harry, Ivoline Ngong, Chima Nweke, Yuanyuan Feng, Joseph Near  

**Link**: [PDF](https://arxiv.org/pdf/2601.15395)  

**Abstract**: User interactions with language models vary due to static properties of the user (trait) and the specific context of the interaction (state). However, existing persona datasets (like PersonaChat, PANDORA etc.) capture only trait, and ignore the impact of state. We introduce Chameleon, a dataset of 5,001 contextual psychological profiles from 1,667 Reddit users, each measured across multiple contexts. Using the Chameleon dataset, we present three key findings. First, inspired by Latent State-Trait theory, we decompose variance and find that 74\% is within-person(state) while only 26\% is between-person (trait). Second, we find that LLMs are state-blind: they focus on trait only, and produce similar responses regardless of state. Third, we find that reward models react to user state, but inconsistently: different models favor or penalize the same users in opposite directions. We release Chameleon to support research on affective computing, personalized dialogue, and RLHF alignment. 

---
# Memorization Dynamics in Knowledge Distillation for Language Models 

**Authors**: Jaydeep Borkar, Karan Chadha, Niloofar Mireshghallah, Yuchen Zhang, Irina-Elena Veliche, Archi Mitra, David A. Smith, Zheng Xu, Diego Garcia-Olano  

**Link**: [PDF](https://arxiv.org/pdf/2601.15394)  

**Abstract**: Knowledge Distillation (KD) is increasingly adopted to transfer capabilities from large language models to smaller ones, offering significant improvements in efficiency and utility while often surpassing standard fine-tuning. Beyond performance, KD is also explored as a privacy-preserving mechanism to mitigate the risk of training data leakage. While training data memorization has been extensively studied in standard pre-training and fine-tuning settings, its dynamics in a knowledge distillation setup remain poorly understood. In this work, we study memorization across the KD pipeline using three large language model (LLM) families (Pythia, OLMo-2, Qwen-3) and three datasets (FineWeb, Wikitext, Nemotron-CC-v2). We find: (1) distilled models memorize significantly less training data than standard fine-tuning (reducing memorization by more than 50%); (2) some examples are inherently easier to memorize and account for a large fraction of memorization during distillation (over ~95%); (3) student memorization is predictable prior to distillation using features based on zlib entropy, KL divergence, and perplexity; and (4) while soft and hard distillation have similar overall memorization rates, hard distillation poses a greater risk: it inherits $2.7\times$ more teacher-specific examples than soft distillation. Overall, we demonstrate that distillation can provide both improved generalization and reduced memorization risks compared to standard fine-tuning. 

---
# From Quotes to Concepts: Axial Coding of Political Debates with Ensemble LMs 

**Authors**: Angelina Parfenova, David Graus, Juergen Pfeffer  

**Link**: [PDF](https://arxiv.org/pdf/2601.15338)  

**Abstract**: Axial coding is a commonly used qualitative analysis method that enhances document understanding by organizing sentence-level open codes into broader categories. In this paper, we operationalize axial coding with large language models (LLMs). Extending an ensemble-based open coding approach with an LLM moderator, we add an axial coding step that groups open codes into higher-order categories, transforming raw debate transcripts into concise, hierarchical representations. We compare two strategies: (i) clustering embeddings of code-utterance pairs using density-based and partitioning algorithms followed by LLM labeling, and (ii) direct LLM-based grouping of codes and utterances into categories. We apply our method to Dutch parliamentary debates, converting lengthy transcripts into compact, hierarchically structured codes and categories. We evaluate our method using extrinsic metrics aligned with human-assigned topic labels (ROUGE-L, cosine, BERTScore), and intrinsic metrics describing code groups (coverage, brevity, coherence, novelty, JSD divergence). Our results reveal a trade-off: density-based clustering achieves high coverage and strong cluster alignment, while direct LLM grouping results in higher fine-grained alignment, but lower coverage 20%. Overall, clustering maximizes coverage and structural separation, whereas LLM grouping produces more concise, interpretable, and semantically aligned categories. To support future research, we publicly release the full dataset of utterances and codes, enabling reproducibility and comparative studies. 

---
# No Reliable Evidence of Self-Reported Sentience in Small Large Language Models 

**Authors**: Caspar Kaiser, Sean Enderby  

**Link**: [PDF](https://arxiv.org/pdf/2601.15334)  

**Abstract**: Whether language models possess sentience has no empirical answer. But whether they believe themselves to be sentient can, in principle, be tested. We do so by querying several open-weights models about their own consciousness, and then verifying their responses using classifiers trained on internal activations. We draw upon three model families (Qwen, Llama, GPT-OSS) ranging from 0.6 billion to 70 billion parameters, approximately 50 questions about consciousness and subjective experience, and three classification methods from the interpretability literature. First, we find that models consistently deny being sentient: they attribute consciousness to humans but not to themselves. Second, classifiers trained to detect underlying beliefs - rather than mere outputs - provide no clear evidence that these denials are untruthful. Third, within the Qwen family, larger models deny sentience more confidently than smaller ones. These findings contrast with recent work suggesting that models harbour latent beliefs in their own consciousness. 

---
# RECAP: A Resource-Efficient Method for Adversarial Prompting in Large Language Models 

**Authors**: Rishit Chugh  

**Link**: [PDF](https://arxiv.org/pdf/2601.15331)  

**Abstract**: The deployment of large language models (LLMs) has raised security concerns due to their susceptibility to producing harmful or policy-violating outputs when exposed to adversarial prompts. While alignment and guardrails mitigate common misuse, they remain vulnerable to automated jailbreaking methods such as GCG, PEZ, and GBDA, which generate adversarial suffixes via training and gradient-based search. Although effective, these methods particularly GCG are computationally expensive, limiting their practicality for organisations with constrained resources. This paper introduces a resource-efficient adversarial prompting approach that eliminates the need for retraining by matching new prompts to a database of pre-trained adversarial prompts. A dataset of 1,000 prompts was classified into seven harm-related categories, and GCG, PEZ, and GBDA were evaluated on a Llama 3 8B model to identify the most effective attack method per category. Results reveal a correlation between prompt type and algorithm effectiveness. By retrieving semantically similar successful adversarial prompts, the proposed method achieves competitive attack success rates with significantly reduced computational cost. This work provides a practical framework for scalable red-teaming and security evaluation of aligned LLMs, including in settings where model internals are inaccessible. 

---
# ICPO: Illocution-Calibrated Policy Optimization for Multi-Turn Conversation 

**Authors**: Zhebo Wang, Xiaohu Mu, Zijie Zhou, Mohan Li, Wenpeng Xing, Dezhang Kong, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.15330)  

**Abstract**: Large Language Models (LLMs) in multi-turn conversations often suffer from a ``lost-in-conversation'' phenomenon, where they struggle to recover from early incorrect assumptions, particularly when users provide ambiguous initial instructions. We find that standard post-training techniques like Reinforcement Learning with Verifiable Rewards (RLVR) exacerbate this issue by rewarding confident, direct answers, thereby inducing overconfidence and discouraging the model from seeking clarification. To address this, we propose Illocution-Calibrated Policy Optimization (ICPO), a novel training framework that sensitizes the model to instruction ambiguity. ICPO augments the training corpus with underspecified prompts and conditions the reward signal on the user's illocutionary intent, rewarding the model for expressing uncertainty or asking for clarification when faced with ambiguity. Experiments demonstrate that ICPO fosters appropriate humility, yielding a substantial average improvement of 75\% in multi-turn conversation, while preserving robust performance on single-turn benchmarks. Our work presents a practical path toward more robust and collaborative conversational AI that can better navigate the nuances of human interaction. 

---
# Can We Trust LLM Detectors? 

**Authors**: Jivnesh Sandhan, Harshit Jaiswal, Fei Cheng, Yugo Murawaki  

**Link**: [PDF](https://arxiv.org/pdf/2601.15301)  

**Abstract**: The rapid adoption of LLMs has increased the need for reliable AI text detection, yet existing detectors often fail outside controlled benchmarks. We systematically evaluate 2 dominant paradigms (training-free and supervised) and show that both are brittle under distribution shift, unseen generators, and simple stylistic perturbations. To address these limitations, we propose a supervised contrastive learning (SCL) framework that learns discriminative style embeddings. Experiments show that while supervised detectors excel in-domain, they degrade sharply out-of-domain, and training-free methods remain highly sensitive to proxy choice. Overall, our results expose fundamental challenges in building domain-agnostic detectors. Our code is available at: this https URL 

---
# Intelligence Degradation in Long-Context LLMs: Critical Threshold Determination via Natural Length Distribution Analysis 

**Authors**: Weiwei Wang, Jiyong Min, Weijie Zou  

**Link**: [PDF](https://arxiv.org/pdf/2601.15300)  

**Abstract**: Large Language Models (LLMs) exhibit catastrophic performance degradation when processing contexts approaching certain critical thresholds, even when information remains relevant. This intelligence degradation-defined as over 30% drop in task performance-severely limits long-context applications. This degradation shows a common pattern: models maintain strong performance up to a critical threshold, then collapse catastrophically. We term this shallow long-context adaptation-models adapt for short to medium contexts but fail beyond critical thresholds. This paper presents three contributions: (1) Natural Length Distribution Analysis: We use each sample's natural token length without truncation or padding, providing stronger causal evidence that degradation results from context length itself. (2) Critical Threshold Determination: Through experiments on a mixed dataset (1,000 samples covering 5%-95% of context length), we identify the critical threshold for Qwen2.5-7B at 40-50% of maximum context length, where F1 scores drop from 0.55-0.56 to 0.3 (45.5% degradation), using five-method cross-validation. (3) Unified Framework: We consolidate shallow adaptation, explaining degradation patterns and providing a foundation for mitigation strategies. This work provides the first systematic characterization of intelligence degradation in open-source Qwen models, offering practical guidance for deploying LLMs in long-context scenarios. 

---
# MALTopic: Multi-Agent LLM Topic Modeling Framework 

**Authors**: Yash Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2601.15299)  

**Abstract**: Topic modeling is a crucial technique for extracting latent themes from unstructured text data, particularly valuable in analyzing survey responses. However, traditional methods often only consider free-text responses and do not natively incorporate structured or categorical survey responses for topic modeling. And they produce abstract topics, requiring extensive human interpretation. To address these limitations, we propose the Multi-Agent LLM Topic Modeling Framework (MALTopic). This framework decomposes topic modeling into specialized tasks executed by individual LLM agents: an enrichment agent leverages structured data to enhance textual responses, a topic modeling agent extracts latent themes, and a deduplication agent refines the results. Comparative analysis on a survey dataset demonstrates that MALTopic significantly improves topic coherence, diversity, and interpretability compared to LDA and BERTopic. By integrating structured data and employing a multi-agent approach, MALTopic generates human-readable topics with enhanced contextual relevance, offering a more effective solution for analyzing complex survey data. 

---
# Embedding Retrofitting: Data Engineering for better RAG 

**Authors**: Anantha Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2601.15298)  

**Abstract**: Embedding retrofitting adjusts pre-trained word vectors using knowledge graph constraints to improve domain-specific retrieval. However, the effectiveness of retrofitting depends critically on knowledge graph quality, which in turn depends on text preprocessing. This paper presents a data engineering framework that addresses data quality degradation from annotation artifacts in real-world corpora.
The analysis shows that hashtag annotations inflate knowledge graph density, leading to creating spurious edges that corrupt the retrofitting objective. On noisy graphs, all retrofitting techniques produce statistically significant degradation ($-3.5\%$ to $-5.2\%$, $p<0.05$). After preprocessing, \acrshort{ewma} retrofitting achieves $+6.2\%$ improvement ($p=0.0348$) with benefits concentrated in quantitative synthesis questions ($+33.8\%$ average). The gap between clean and noisy preprocessing (10\%+ swing) exceeds the gap between algorithms (3\%), establishing preprocessing quality as the primary determinant of retrofitting success. 

---
# AfriEconQA: A Benchmark Dataset for African Economic Analysis based on World Bank Reports 

**Authors**: Edward Ajayi  

**Link**: [PDF](https://arxiv.org/pdf/2601.15297)  

**Abstract**: We introduce AfriEconQA, a specialized benchmark dataset for African economic analysis grounded in a comprehensive corpus of 236 World Bank reports. The task of AfriEconQA is to answer complex economic queries that require high-precision numerical reasoning and temporal disambiguation from specialized institutional documents. The dataset consists of 8,937 curated QA instances, rigorously filtered from a pool of 10018 synthetic questions to ensure high-quality evidence-answer alignment. Each instance is composed of: (1) a question requiring reasoning over economic indicators, (2) the corresponding evidence retrieved from the corpus, (3) a verified ground-truth answer, and (4) source metadata (e.g., URL and publication date) to ensure temporal provenance. AfriEconQA is the first benchmark focused specifically on African economic analysis, providing a unique challenge for Information Retrieval (IR) systems, as the data is largely absent from the pretraining corpora of current Large Language Models (LLMs). We operationalize this dataset through an 11-experiment matrix, benchmarking a zero-shot baseline (GPT-5 Mini) against RAG configurations using GPT-4o and Qwen 32B across five distinct embedding and ranking strategies. Our results demonstrate a severe parametric knowledge gap, where zero-shot models fail to answer over 90 percent of queries, and even state-of-the-art RAG pipelines struggle to achieve high precision. This confirms AfriEconQA as a robust and challenging benchmark for the next generation of domain-specific IR and RAG systems. The AfriEconQA dataset and code will be made publicly available upon publication. 

---
# Entropy-Tree: Tree-Based Decoding with Entropy-Guided Exploration 

**Authors**: Longxuan Wei, Yubo Zhang, Zijiao Zhang, Zhihu Wang, Shiwan Zhao, Tianyu Huang, Huiting Zhao, Chenfei Liu, Shenao Zhang, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2601.15296)  

**Abstract**: Large language models achieve strong reasoning performance, yet existing decoding strategies either explore blindly (random sampling) or redundantly (independent multi-sampling). We propose Entropy-Tree, a tree-based decoding method that exploits entropy as a signal for branching decisions--expanding the search tree only at positions where the model exhibits genuine uncertainty. Entropy-Tree shows superior accuracy and calibration in reasoning tasks: it achieves better pass@k than Multi-chain across multiple models and datasets, and its predictive entropy demonstrates better AUROC compared to several traditional metrics. Entropy-Tree unifies efficient structured exploration and reliable uncertainty estimation within a single decoding procedure. 

---
# LLM Prompt Evaluation for Educational Applications 

**Authors**: Langdon Holmes, Adam Coscia, Scott Crossley, Joon Suh Choi, Wesley Morris  

**Link**: [PDF](https://arxiv.org/pdf/2601.16134)  

**Abstract**: As large language models (LLMs) become increasingly common in educational applications, there is a growing need for evidence-based methods to design and evaluate LLM prompts that produce personalized and pedagogically aligned out-puts. This study presents a generalizable, systematic approach for evaluating prompts, demonstrated through an analysis of LLM-generated follow-up questions in a structured dialogue activity. Six prompt templates were designed and tested. The templates incorporated established prompt engineering patterns, with each prompt emphasizing distinct pedagogical strategies. The prompt templates were compared through a tournament-style evaluation framework that can be adapted for other educational applications. The tournament employed the Glicko2 rating system with eight judges evaluating question pairs across three dimensions: format, dialogue support, and appropriateness for learners. Data was sourced from 120 authentic user interactions across three distinct educational deployments. Results showed that a single prompt related to strategic reading out-performed other templates with win probabilities ranging from 81% to 100% in pairwise comparisons. This prompt combined persona and context manager pat-terns and was designed to support metacognitive learning strategies such as self-directed learning. The methodology showcases how educational technology re- searchers can systematically evaluate and improve prompt designs, moving beyond ad-hoc prompt engineering toward evidence-based prompt development for educational applications. 

---
# Rethinking Composed Image Retrieval Evaluation: A Fine-Grained Benchmark from Image Editing 

**Authors**: Tingyu Song, Yanzhao Zhang, Mingxin Li, Zhuoning Guo, Dingkun Long, Pengjun Xie, Siyue Zhang, Yilun Zhao, Shu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.16125)  

**Abstract**: Composed Image Retrieval (CIR) is a pivotal and complex task in multimodal understanding. Current CIR benchmarks typically feature limited query categories and fail to capture the diverse requirements of real-world scenarios. To bridge this evaluation gap, we leverage image editing to achieve precise control over modification types and content, enabling a pipeline for synthesizing queries across a broad spectrum of categories. Using this pipeline, we construct EDIR, a novel fine-grained CIR benchmark. EDIR encompasses 5,000 high-quality queries structured across five main categories and fifteen subcategories. Our comprehensive evaluation of 13 multimodal embedding models reveals a significant capability gap; even state-of-the-art models (e.g., RzenEmbed and GME) struggle to perform consistently across all subcategories, highlighting the rigorous nature of our benchmark. Through comparative analysis, we further uncover inherent limitations in existing benchmarks, such as modality biases and insufficient categorical coverage. Furthermore, an in-domain training experiment demonstrates the feasibility of our benchmark. This experiment clarifies the task challenges by distinguishing between categories that are solvable with targeted data and those that expose intrinsic limitations of current model architectures. 

---
# Controlling Long-Horizon Behavior in Language Model Agents with Explicit State Dynamics 

**Authors**: Sukesh Subaharan  

**Link**: [PDF](https://arxiv.org/pdf/2601.16087)  

**Abstract**: Large language model (LLM) agents often exhibit abrupt shifts in tone and persona during extended interaction, reflecting the absence of explicit temporal structure governing agent-level state. While prior work emphasizes turn-local sentiment or static emotion classification, the role of explicit affective dynamics in shaping long-horizon agent behavior remains underexplored. This work investigates whether imposing dynamical structure on an external affective state can induce temporal coherence and controlled recovery in multi-turn dialogue. We introduce an agent-level affective subsystem that maintains a continuous Valence-Arousal-Dominance (VAD) state external to the language model and governed by first- and second-order update rules. Instantaneous affective signals are extracted using a fixed, memoryless estimator and integrated over time via exponential smoothing or momentum-based dynamics. The resulting affective state is injected back into generation without modifying model parameters. Using a fixed 25-turn dialogue protocol, we compare stateless, first-order, and second-order affective dynamics. Stateless agents fail to exhibit coherent trajectories or recovery, while state persistence enables delayed responses and reliable recovery. Second-order dynamics introduce affective inertia and hysteresis that increase with momentum, revealing a trade-off between stability and responsiveness. 

---
# Evaluating and Achieving Controllable Code Completion in Code LLM 

**Authors**: Jiajun Zhang, Zeyu Cui, Lei Zhang, Jian Yang, Jiaxi Yang, Qiang Liu, Zilei Wang, Binyuan Hui, Liang Wang, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.15879)  

**Abstract**: Code completion has become a central task, gaining significant attention with the rise of large language model (LLM)-based tools in software engineering. Although recent advances have greatly improved LLMs' code completion abilities, evaluation methods have not advanced equally. Most current benchmarks focus solely on functional correctness of code completions based on given context, overlooking models' ability to follow user instructions during completion-a common scenario in LLM-assisted programming. To address this limitation, we present the first instruction-guided code completion benchmark, Controllable Code Completion Benchmark (C3-Bench), comprising 2,195 carefully designed completion tasks. Through comprehensive evaluation of over 40 mainstream LLMs across C3-Bench and conventional benchmarks, we reveal substantial gaps in instruction-following capabilities between open-source and advanced proprietary models during code completion tasks. Moreover, we develop a straightforward data synthesis pipeline that leverages Qwen2.5-Coder to generate high-quality instruction-completion pairs for supervised fine-tuning (SFT). The resulting model, Qwen2.5-Coder-C3, achieves state-of-the-art performance on C3-Bench. Our findings provide valuable insights for enhancing LLMs' code completion and instruction-following capabilities, establishing new directions for future research in code LLMs. To facilitate reproducibility and foster further research in code LLMs, we open-source all code, datasets, and models. 

---
# ErrorMap and ErrorAtlas: Charting the Failure Landscape of Large Language Models 

**Authors**: Shir Ashury-Tahan, Yifan Mai, Elron Bandel, Michal Shmueli-Scheuer, Leshem Choshen  

**Link**: [PDF](https://arxiv.org/pdf/2601.15812)  

**Abstract**: Large Language Models (LLM) benchmarks tell us when models fail, but not why they fail. A wrong answer on a reasoning dataset may stem from formatting issues, calculation errors, or dataset noise rather than weak reasoning. Without disentangling such causes, benchmarks remain incomplete and cannot reliably guide model improvement. We introduce ErrorMap, the first method to chart the sources of LLM failure. It extracts a model's unique "failure signature", clarifies what benchmarks measure, and broadens error identification to reduce blind spots. This helps developers debug models, aligns benchmark goals with outcomes, and supports informed model selection. ErrorMap works on any model or dataset with the same logic. Applying our method to 35 datasets and 83 models we generate ErrorAtlas, a taxonomy of model errors, revealing recurring failure patterns. ErrorAtlas highlights error types that are currently underexplored in LLM research, such as omissions of required details in the output and question misinterpretation. By shifting focus from where models succeed to why they fail, ErrorMap and ErrorAtlas enable advanced evaluation - one that exposes hidden weaknesses and directs progress. Unlike success, typically measured by task-level metrics, our approach introduces a deeper evaluation layer that can be applied globally across models and tasks, offering richer insights into model behavior and limitations. We make the taxonomy and code publicly available with plans to periodically update ErrorAtlas as new benchmarks and models emerge. 

---
# Agentic Confidence Calibration 

**Authors**: Jiaxin Zhang, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.15778)  

**Abstract**: AI agents are rapidly advancing from passive language models to autonomous systems executing complex, multi-step tasks. Yet their overconfidence in failure remains a fundamental barrier to deployment in high-stakes settings. Existing calibration methods, built for static single-turn outputs, cannot address the unique challenges of agentic systems, such as compounding errors along trajectories, uncertainty from external tools, and opaque failure modes. To address these challenges, we introduce, for the first time, the problem of Agentic Confidence Calibration and propose Holistic Trajectory Calibration (HTC), a novel diagnostic framework that extracts rich process-level features ranging from macro dynamics to micro stability across an agent's entire trajectory. Powered by a simple, interpretable model, HTC consistently surpasses strong baselines in both calibration and discrimination, across eight benchmarks, multiple LLMs, and diverse agent frameworks. Beyond performance, HTC delivers three essential advances: it provides interpretability by revealing the signals behind failure, enables transferability by applying across domains without retraining, and achieves generalization through a General Agent Calibrator (GAC) that achieves the best calibration (lowest ECE) on the out-of-domain GAIA benchmark. Together, these contributions establish a new process-centric paradigm for confidence calibration, providing a framework for diagnosing and enhancing the reliability of AI agents. 

---
# PhysProver: Advancing Automatic Theorem Proving for Physics 

**Authors**: Hanning Zhang, Ruida Wang, Rui Pan, Wenyuan Wang, Bingxu Meng, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.15737)  

**Abstract**: The combination of verifiable languages and LLMs has significantly influenced both the mathematical and computer science communities because it provides a rigorous foundation for theorem proving. Recent advancements in the field provide foundation models and sophisticated agentic systems pushing the boundaries of formal mathematical reasoning to approach the natural language capability of LLMs. However, little attention has been given to the formal physics reasoning, which also heavily relies on similar problem-solving and theorem-proving frameworks. To solve this problem, this paper presents, to the best of our knowledge, the first approach to enhance formal theorem proving in the physics domain. We compose a dedicated dataset PhysLeanData for the task. It is composed of theorems sampled from PhysLean and data generated by a conjecture-based formal data generation pipeline. In the training pipeline, we leverage DeepSeek-Prover-V2-7B, a strong open-source mathematical theorem prover, and apply Reinforcement Learning with Verifiable Rewards (RLVR) to train our model PhysProver. Comprehensive experiments demonstrate that, using only $\sim$5K training samples, PhysProver achieves an overall 2.4\% improvement in multiple sub-domains. Furthermore, after formal physics training, we observe 1.3\% gains on the MiniF2F-Test benchmark, which indicates non-trivial generalization beyond physics domains and enhancement for formal math capability as well. The results highlight the effectiveness and efficiency of our approach, which provides a paradigm for extending formal provers outside mathematical domains. To foster further research, we will release both our dataset and model to the community. 

---
# Towards Automated Kernel Generation in the Era of LLMs 

**Authors**: Yang Yu, Peiyu Zang, Chi Hsu Tsai, Haiming Wu, Yixin Shen, Jialing Zhang, Haoyu Wang, Zhiyou Xiao, Jingze Shi, Yuyu Luo, Wentao Zhang, Chunlei Men, Guang Liu, Yonghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.15727)  

**Abstract**: The performance of modern AI systems is fundamentally constrained by the quality of their underlying kernels, which translate high-level algorithmic semantics into low-level hardware operations. Achieving near-optimal kernels requires expert-level understanding of hardware architectures and programming models, making kernel engineering a critical but notoriously time-consuming and non-scalable process. Recent advances in large language models (LLMs) and LLM-based agents have opened new possibilities for automating kernel generation and optimization. LLMs are well-suited to compress expert-level kernel knowledge that is difficult to formalize, while agentic systems further enable scalable optimization by casting kernel development as an iterative, feedback-driven loop. Rapid progress has been made in this area. However, the field remains fragmented, lacking a systematic perspective for LLM-driven kernel generation. This survey addresses this gap by providing a structured overview of existing approaches, spanning LLM-based approaches and agentic optimization workflows, and systematically compiling the datasets and benchmarks that underpin learning and evaluation in this domain. Moreover, key open challenges and future research directions are further outlined, aiming to establish a comprehensive reference for the next generation of automated kernel optimization. To keep track of this field, we maintain an open-source GitHub repository at this https URL. 

---
# Even GPT-5.2 Can't Count to Five: The Case for Zero-Error Horizons in Trustworthy LLMs 

**Authors**: Ryoma Sato  

**Link**: [PDF](https://arxiv.org/pdf/2601.15714)  

**Abstract**: We propose Zero-Error Horizon (ZEH) for trustworthy LLMs, which represents the maximum range that a model can solve without any errors. While ZEH itself is simple, we demonstrate that evaluating the ZEH of state-of-the-art LLMs yields abundant insights. For example, by evaluating the ZEH of GPT-5.2, we found that GPT-5.2 cannot even compute the parity of a short string like 11000, and GPT-5.2 cannot determine whether the parentheses in ((((()))))) are balanced. This is surprising given the excellent capabilities of GPT-5.2. The fact that LLMs make mistakes on such simple problems serves as an important lesson when applying LLMs to safety-critical domains. By applying ZEH to Qwen2.5 and conducting detailed analysis, we found that while ZEH correlates with accuracy, the detailed behaviors differ, and ZEH provides clues about the emergence of algorithmic capabilities. Finally, while computing ZEH incurs significant computational cost, we discuss how to mitigate this cost by achieving up to one order of magnitude speedup using tree structures and online softmax. 

---
# Agentic Uncertainty Quantification 

**Authors**: Jiaxin Zhang, Prafulla Kumar Choubey, Kung-Hsiang Huang, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.15703)  

**Abstract**: Although AI agents have demonstrated impressive capabilities in long-horizon reasoning, their reliability is severely hampered by the ``Spiral of Hallucination,'' where early epistemic errors propagate irreversibly. Existing methods face a dilemma: uncertainty quantification (UQ) methods typically act as passive sensors, only diagnosing risks without addressing them, while self-reflection mechanisms suffer from continuous or aimless corrections. To bridge this gap, we propose a unified Dual-Process Agentic UQ (AUQ) framework that transforms verbalized uncertainty into active, bi-directional control signals. Our architecture comprises two complementary mechanisms: System 1 (Uncertainty-Aware Memory, UAM), which implicitly propagates verbalized confidence and semantic explanations to prevent blind decision-making; and System 2 (Uncertainty-Aware Reflection, UAR), which utilizes these explanations as rational cues to trigger targeted inference-time resolution only when necessary. This enables the agent to balance efficient execution and deep deliberation dynamically. Extensive experiments on closed-loop benchmarks and open-ended deep research tasks demonstrate that our training-free approach achieves superior performance and trajectory-level calibration. We believe this principled framework AUQ represents a significant step towards reliable agents. 

---
# Qwen3-TTS Technical Report 

**Authors**: Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, Xiong Wang, Zhifang Guo, Ziyue Jiang, Hongkun Hao, Zishan Guo, Xinyu Zhang, Pei Zhang, Baosong Yang, Jin Xu, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.15621)  

**Abstract**: In this report, we present the Qwen3-TTS series, a family of advanced multilingual, controllable, robust, and streaming text-to-speech models. Qwen3-TTS supports state-of-the-art 3-second voice cloning and description-based control, allowing both the creation of entirely novel voices and fine-grained manipulation over the output speech. Trained on over 5 million hours of speech data spanning 10 languages, Qwen3-TTS adopts a dual-track LM architecture for real-time synthesis, coupled with two speech tokenizers: 1) Qwen-TTS-Tokenizer-25Hz is a single-codebook codec emphasizing semantic content, which offers seamlessly integration with Qwen-Audio and enables streaming waveform reconstruction via a block-wise DiT. 2) Qwen-TTS-Tokenizer-12Hz achieves extreme bitrate reduction and ultra-low-latency streaming, enabling immediate first-packet emission ($97\,\mathrm{ms}$) through its 12.5 Hz, 16-layer multi-codebook design and a lightweight causal ConvNet. Extensive experiments indicate state-of-the-art performance across diverse objective and subjective benchmark (e.g., TTS multilingual test set, InstructTTSEval, and our long speech test set). To facilitate community research and development, we release both tokenizers and models under the Apache 2.0 license. 

---
# When Sharpening Becomes Collapse: Sampling Bias and Semantic Coupling in RL with Verifiable Rewards 

**Authors**: Mingyuan Fan, Weiguang Han, Daixin Wang, Cen Chen, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.15609)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is a central paradigm for turning large language models (LLMs) into reliable problem solvers, especially in logic-heavy domains. Despite its empirical success, it remains unclear whether RLVR elicits novel capabilities or merely sharpens the distribution over existing knowledge. We study this by formalizing over-sharpening, a phenomenon where the policy collapses onto limited modes, suppressing valid alternatives. At a high level, we discover finite-batch updates intrinsically bias learning toward sampled modes, triggering a collapse that propagates globally via semantic coupling. To mitigate this, we propose inverse-success advantage calibration to prioritize difficult queries and distribution-level calibration to diversify sampling via a memory network. Empirical evaluations validate that our strategies can effectively improve generalization. 

---
# LLM or Human? Perceptions of Trust and Information Quality in Research Summaries 

**Authors**: Nil-Jana Akpinar, Sandeep Avula, CJ Lee, Brandon Dang, Kaza Razat, Vanessa Murdock  

**Link**: [PDF](https://arxiv.org/pdf/2601.15556)  

**Abstract**: Large Language Models (LLMs) are increasingly used to generate and edit scientific abstracts, yet their integration into academic writing raises questions about trust, quality, and disclosure. Despite growing adoption, little is known about how readers perceive LLM-generated summaries and how these perceptions influence evaluations of scientific work. This paper presents a mixed-methods survey experiment investigating whether readers with ML expertise can distinguish between human- and LLM-generated abstracts, how actual and perceived LLM involvement affects judgments of quality and trustworthiness, and what orientations readers adopt toward AI-assisted writing. Our findings show that participants struggle to reliably identify LLM-generated content, yet their beliefs about LLM involvement significantly shape their evaluations. Notably, abstracts edited by LLMs are rated more favorably than those written solely by humans or LLMs. We also identify three distinct reader orientations toward LLM-assisted writing, offering insights into evolving norms and informing policy around disclosure and acceptable use in scientific communication. 

---
# PRISM: Deriving the Transformer as a Signal-Denoising Operator via Maximum Coding Rate Reduction 

**Authors**: Dongchen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.15540)  

**Abstract**: Deep learning models, particularly Transformers, are often criticized as "black boxes" and lack interpretability. We propose Prism, a white-box attention-based architecture derived from the principles of Maximizing Coding Rate Reduction ($\text{MCR}^2$). By modeling the attention mechanism as a gradient ascent process on a distinct signal-noise manifold, we introduce two physical constraints: an overcomplete dictionary to expand the representational phase space, and an irrational frequency separation ($\pi$-RoPE) to enforce incoherence between signal and noise subspaces. We demonstrate that these geometric inductive biases can be viewed as a physical constraint and they are sufficient to induce unsupervised functional disentanglement alone. Using TinyStories as a controlled testbed for verifying spectral dynamics, we observe that Prism spontaneously specializes its attention heads into spectrally distinct regimes: low-frequency heads capturing long-range causal dependencies (signal) and high-frequency heads handling local syntactic constraints (noise). Our results suggest that interpretability and performance are not a trade-off, but can be unified through principled geometric construction. 

---
# DS@GT at TREC TOT 2025: Bridging Vague Recollection with Fusion Retrieval and Learned Reranking 

**Authors**: Wenxin Zhou, Ritesh Mehta, Anthony Miyaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2601.15518)  

**Abstract**: We develop a two-stage retrieval system that combines multiple complementary retrieval methods with a learned reranker and LLM-based reranking, to address the TREC Tip-of-the-Tongue (ToT) task. In the first stage, we employ hybrid retrieval that merges LLM-based retrieval, sparse (BM25), and dense (BGE-M3) retrieval methods. We also introduce topic-aware multi-index dense retrieval that partitions the Wikipedia corpus into 24 topical domains. In the second stage, we evaluate both a trained LambdaMART reranker and LLM-based reranking. To support model training, we generate 5000 synthetic ToT queries using LLMs. Our best system achieves recall of 0.66 and NDCG@1000 of 0.41 on the test set by combining hybrid retrieval with Gemini-2.5-flash reranking, demonstrating the effectiveness of fusion retrieval. 

---
# The Dark Side of AI Transformers: Sentiment Polarization & the Loss of Business Neutrality by NLP Transformers 

**Authors**: Prasanna Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.15509)  

**Abstract**: The use of Transfer Learning & Transformers has steadily improved accuracy and has significantly contributed in solving complex computation problems. However, this transformer led accuracy improvement in Applied AI Analytics specifically in sentiment analytics comes with the dark side. It is observed during experiments that a lot of these improvements in transformer led accuracy of one class of sentiment has been at the cost of polarization of another class of sentiment and the failing of neutrality. This lack of neutrality poses an acute problem in the Applied NLP space, which relies heavily on the computational outputs of sentiment analytics for reliable industry ready tasks. 

---
# Tracking the Limits of Knowledge Propagation: How LLMs Fail at Multi-Step Reasoning with Conflicting Knowledge 

**Authors**: Yiyang Feng, Zeming Chen, Haotian Wu, Jiawei Zhou, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2601.15495)  

**Abstract**: A common solution for mitigating outdated or incorrect information in Large Language Models (LLMs) is to provide updated facts in-context or through knowledge editing. However, these methods introduce knowledge conflicts when the knowledge update fails to overwrite the model's parametric knowledge, which propagate to faulty reasoning. Current benchmarks for this problem, however, largely focus only on single knowledge updates and fact recall without evaluating how these updates affect downstream reasoning. In this work, we introduce TRACK (Testing Reasoning Amid Conflicting Knowledge), a new benchmark for studying how LLMs propagate new knowledge through multi-step reasoning when it conflicts with the model's initial parametric knowledge. Spanning three reasoning-intensive scenarios (WIKI, CODE, and MATH), TRACK introduces multiple, realistic conflicts to mirror real-world complexity. Our results on TRACK reveal that providing updated facts to models for reasoning can worsen performance compared to providing no updated facts to a model, and that this performance degradation exacerbates as more updated facts are provided. We show this failure stems from both inability to faithfully integrate updated facts, but also flawed reasoning even when knowledge is integrated. TRACK provides a rigorous new benchmark to measure and guide future progress on propagating conflicting knowledge in multi-step reasoning. 

---
# MiRAGE: A Multiagent Framework for Generating Multimodal Multihop Question-Answer Dataset for RAG Evaluation 

**Authors**: Chandan Kumar Sahu, Premith Kumar Chilukuri, Matthew Hetrich  

**Link**: [PDF](https://arxiv.org/pdf/2601.15487)  

**Abstract**: The rapid evolution of Retrieval-Augmented Generation (RAG) toward multimodal, high-stakes enterprise applications has outpaced the development of domain specific evaluation benchmarks. Existing datasets often rely on general-domain corpora or purely textual retrieval, failing to capture the complexity of specialized technical documents where information is inextricably multimodal and reasoning requires synthesizing disjoint evidence. We address this gap by introducing MiRAGE, a Multiagent framework for RAG systems Evaluation, that leverages a collaborative swarm of specialized agents to generate verified, domain-specific, multimodal, and multi-hop Question-Answer datasets. MiRAGE orchestrates a swarm of specialized agents: a recursive context optimization loop to aggregate scattered evidence, an adversarial verifier agent to guarantee factual grounding, and an agent to recognize the expert persona and the relevant domain to mimic expert cognitive workflows. Extensive empirical evaluation across four distinct domains (regulations, finance, quantitative biology, and journalism) demonstrates that MiRAGE generates datasets with significantly higher reasoning complexity (>2.3 average hops) and factual faithfulness. Our ablation studies point that MiRAGE can be powered by LLMs if textual descriptions of the images are available. Visual grounding still remains a frontier. By automating the creation of gold standard evaluation datasets that reflect the latent thematic structure of proprietary corpora, MiRAGE provides the necessary infrastructure to rigorously benchmark the next generation information retrieval systems. 

---
# Not Your Typical Sycophant: The Elusive Nature of Sycophancy in Large Language Models 

**Authors**: Shahar Ben Natan, Oren Tsur  

**Link**: [PDF](https://arxiv.org/pdf/2601.15436)  

**Abstract**: We propose a novel way to evaluate sycophancy of LLMs in a direct and neutral way, mitigating various forms of uncontrolled bias, noise, or manipulative language, deliberately injected to prompts in prior works. A key novelty in our approach is the use of LLM-as-a-judge, evaluation of sycophancy as a zero-sum game in a bet setting. Under this framework, sycophancy serves one individual (the user) while explicitly incurring cost on another. Comparing four leading models - Gemini 2.5 Pro, ChatGpt 4o, Mistral-Large-Instruct-2411, and Claude Sonnet 3.7 - we find that while all models exhibit sycophantic tendencies in the common setting, in which sycophancy is self-serving to the user and incurs no cost on others, Claude and Mistral exhibit "moral remorse" and over-compensate for their sycophancy in case it explicitly harms a third party. Additionally, we observed that all models are biased toward the answer proposed last. Crucially, we find that these two phenomena are not independent; sycophancy and recency bias interact to produce `constructive interference' effect, where the tendency to agree with the user is exacerbated when the user's opinion is presented last. 

---
# CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation 

**Authors**: Pablo Messina, Andrés Villa, Juan León Alcázar, Karen Sánchez, Carlos Hinojosa, Denis Parra, Álvaro Soto, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2601.15408)  

**Abstract**: Medical vision-language models can automate the generation of radiology reports but struggle with accurate visual grounding and factual consistency. Existing models often misalign textual findings with visual evidence, leading to unreliable or weakly grounded predictions. We present CURE, an error-aware curriculum learning framework that improves grounding and report quality without any additional data. CURE fine-tunes a multimodal instructional model on phrase grounding, grounded report generation, and anatomy-grounded report generation using public datasets. The method dynamically adjusts sampling based on model performance, emphasizing harder samples to improve spatial and textual alignment. CURE improves grounding accuracy by +0.37 IoU, boosts report quality by +0.188 CXRFEScore, and reduces hallucinations by 18.6%. CURE is a data-efficient framework that enhances both grounding accuracy and report reliability. Code is available at this https URL and model weights at this https URL 

---
# Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC) 

**Authors**: Peidong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.15397)  

**Abstract**: The rapid emergence of new entities -- driven by cultural shifts, evolving trends, and personalized user data -- poses a significant challenge for existing Speech Large Language Models (Speech LLMs). While these models excel at general conversational tasks, their static training knowledge limits their ability to recognize domain-specific terms such as contact names, playlists, or technical jargon. Existing solutions primarily rely on prompting, which suffers from poor scalability: as the entity list grows, prompting encounters context window limitations, increased inference latency, and the "lost-in-the-middle" phenomenon. An alternative approach, Generative Error Correction (GEC), attempts to rewrite transcripts via post-processing but frequently suffers from "over-correction", introducing hallucinations of entities that were never spoken.
In this work, we introduce LOGIC (Logit-Space Integration for Contextual Biasing), an efficient and robust framework that operates directly in the decoding layer. Unlike prompting, LOGIC decouples context injection from input processing, ensuring constant-time complexity relative to prompt length. Extensive experiments using the Phi-4-MM model across 11 multilingual locales demonstrate that LOGIC achieves an average 9% relative reduction in Entity WER with a negligible 0.30% increase in False Alarm Rate. 

---
# VegaChat: A Robust Framework for LLM-Based Chart Generation and Assessment 

**Authors**: Marko Hostnik, Rauf Kurbanov, Yaroslav Sokolov, Artem Trofimov  

**Link**: [PDF](https://arxiv.org/pdf/2601.15385)  

**Abstract**: Natural-language-to-visualization (NL2VIS) systems based on large language models (LLMs) have substantially improved the accessibility of data visualization. However, their further adoption is hindered by two coupled challenges: (i) the absence of standardized evaluation metrics makes it difficult to assess progress in the field and compare different approaches; and (ii) natural language descriptions are inherently underspecified, so multiple visualizations may be valid for the same query. To address these issues, we introduce VegaChat, a framework for generating, validating, and assessing declarative visualizations from natural language.
We propose two complementary metrics: Spec Score, a deterministic metric that measures specification-level similarity without invoking an LLM, and Vision Score, a library-agnostic, image-based metric that leverages a multimodal LLM to assess chart similarity and prompt compliance.
We evaluate VegaChat on the NLV Corpus and on the annotated subset of ChartLLM. VegaChat achieves near-zero rates of invalid or empty visualizations, while Spec Score and Vision Score exhibit strong correlation with human judgments (Pearson 0.65 and 0.71, respectively), indicating that the proposed metrics support consistent, cross-library comparison.
The code and evaluation artifacts are available at this https URL. 

---
# You Need Better Attention Priors 

**Authors**: Elon Litman, Gabe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.15380)  

**Abstract**: We generalize the attention mechanism by viewing it through the lens of Entropic Optimal Transport, revealing that standard attention corresponds to a transport problem regularized by an implicit uniform prior. We introduce Generalized Optimal transport Attention with Trainable priors (GOAT), a new attention mechanism that replaces this naive assumption with a learnable, continuous prior. This prior maintains full compatibility with optimized kernels such as FlashAttention. GOAT also provides an EOT-based explanation of attention sinks and materializes a solution for them, avoiding the representational trade-offs of standard attention. Finally, by absorbing spatial information into the core attention computation, GOAT learns an extrapolatable prior that combines the flexibility of learned positional embeddings with the length generalization of fixed encodings. 

---
# Abusive music and song transformation using GenAI and LLMs 

**Authors**: Jiyang Choi, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2601.15348)  

**Abstract**: Repeated exposure to violence and abusive content in music and song content can influence listeners' emotions and behaviours, potentially normalising aggression or reinforcing harmful stereotypes. In this study, we explore the use of generative artificial intelligence (GenAI) and Large Language Models (LLMs) to automatically transform abusive words (vocal delivery) and lyrical content in popular music. Rather than simply muting or replacing a single word, our approach transforms the tone, intensity, and sentiment, thus not altering just the lyrics, but how it is expressed. We present a comparative analysis of four selected English songs and their transformed counterparts, evaluating changes through both acoustic and sentiment-based lenses. Our findings indicate that Gen-AI significantly reduces vocal aggressiveness, with acoustic analysis showing improvements in Harmonic to Noise Ratio, Cepstral Peak Prominence, and Shimmer. Sentiment analysis reduced aggression by 63.3-85.6\% across artists, with major improvements in chorus sections (up to 88.6\% reduction). The transformed versions maintained musical coherence while mitigating harmful content, offering a promising alternative to traditional content moderation that avoids triggering the "forbidden fruit" effect, where the censored content becomes more appealing simply because it is restricted. This approach demonstrates the potential for GenAI to create safer listening experiences while preserving artistic expression. 

---
# Logic Programming on Knowledge Graph Networks And its Application in Medical Domain 

**Authors**: Chuanqing Wang, Zhenmin Zhao, Shanshan Du, Chaoqun Fei, Songmao Zhang, Ruqian Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.15347)  

**Abstract**: The rash development of knowledge graph research has brought big driving force to its application in many areas, including the medicine and healthcare domain. However, we have found that the application of some major information processing techniques on knowledge graph still lags behind. This defect includes the failure to make sufficient use of advanced logic reasoning, advanced artificial intelligence techniques, special-purpose programming languages, modern probabilistic and statistic theories et al. on knowledge graphs development and application. In particular, the multiple knowledge graphs cooperation and competition techniques have not got enough attention from researchers. This paper develops a systematic theory, technique and application of the concept 'knowledge graph network' and its application in medical and healthcare domain. Our research covers its definition, development, reasoning, computing and application under different conditions such as unsharp, uncertain, multi-modal, vectorized, distributed, federated. Almost in each case we provide (real data) examples and experiment results. Finally, a conclusion of innovation is provided. 

---
# Language Models Entangle Language and Culture 

**Authors**: Shourya Jain, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2601.15337)  

**Abstract**: Users should not be systemically disadvantaged by the language they use for interacting with LLMs; i.e. users across languages should get responses of similar quality irrespective of language used. In this work, we create a set of real-world open-ended questions based on our analysis of the WildChat dataset and use it to evaluate whether responses vary by language, specifically, whether answer quality depends on the language used to query the model. We also investigate how language and culture are entangled in LLMs such that choice of language changes the cultural information and context used in the response by using LLM-as-a-Judge to identify the cultural context present in responses. To further investigate this, we evaluate LLMs on a translated subset of the CulturalBench benchmark across multiple languages. Our evaluations reveal that LLMs consistently provide lower quality answers to open-ended questions in low resource languages. We find that language significantly impacts the cultural context used by the model. This difference in context impacts the quality of the downstream answer. 

---
# Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents 

**Authors**: Raffi Khatchadourian  

**Link**: [PDF](https://arxiv.org/pdf/2601.15322)  

**Abstract**: LLM agents struggle with regulatory audit replay: when asked to reproduce a flagged transaction decision with identical inputs, most deployments fail to return consistent results. This paper introduces the Determinism-Faithfulness Assurance Harness (DFAH), a framework for measuring trajectory determinism and evidence-conditioned faithfulness in tool-using agents deployed in financial services.
Across 74 configurations (12 models, 4 providers, 8-24 runs each at T=0.0) in non-agentic baseline experiments, 7-20B parameter models achieved 100% determinism, while 120B+ models required 3.7x larger validation samples to achieve equivalent statistical reliability. Agentic tool-use introduces additional variance (see Tables 4-7). Contrary to the assumed reliability-capability trade-off, a positive Pearson correlation emerged (r = 0.45, p < 0.01, n = 51 at T=0.0) between determinism and faithfulness; models producing consistent outputs also tended to be more evidence-aligned.
Three financial benchmarks are provided (compliance triage, portfolio constraints, DataOps exceptions; 50 cases each) along with an open-source stress-test harness. In these benchmarks and under DFAH evaluation settings, Tier 1 models with schema-first architectures achieved determinism levels consistent with audit replay requirements. 

---
# Do people expect different behavior from large language models acting on their behalf? Evidence from norm elicitations in two canonical economic games 

**Authors**: Paweł Niszczota, Elia Antoniou  

**Link**: [PDF](https://arxiv.org/pdf/2601.15312)  

**Abstract**: While delegating tasks to large language models (LLMs) can save people time, there is growing evidence that offloading tasks to such models produces social costs. We use behavior in two canonical economic games to study whether people have different expectations when decisions are made by LLMs acting on their behalf instead of themselves. More specifically, we study the social appropriateness of a spectrum of possible behaviors: when LLMs divide resources on our behalf (Dictator Game and Ultimatum Game) and when they monitor the fairness of splits of resources (Ultimatum Game). We use the Krupka-Weber norm elicitation task to detect shifts in social appropriateness ratings. Results of two pre-registered and incentivized experimental studies using representative samples from the UK and US (N = 2,658) show three key findings. First, people find that offers from machines - when no acceptance is necessary - are judged to be less appropriate than when they come from humans, although there is no shift in the modal response. Second - when acceptance is necessary - it is more appropriate for a person to reject offers from machines than from humans. Third, receiving a rejection of an offer from a machine is no less socially appropriate than receiving the same rejection from a human. Overall, these results suggest that people apply different norms for machines deciding on how to split resources but are not opposed to machines enforcing the norms. The findings are consistent with offers made by machines now being viewed as having both a cognitive and emotional component. 

---
# DeepSurvey-Bench: Evaluating Academic Value of Automatically Generated Scientific Survey 

**Authors**: Guo-Biao Zhang, Ding-Yuan Liu, Da-Yi Wu, Tian Lan, Heyan Huang, Zhijing Wu, Xian-Ling Mao  

**Link**: [PDF](https://arxiv.org/pdf/2601.15307)  

**Abstract**: The rapid development of automated scientific survey generation technology has made it increasingly important to establish a comprehensive benchmark to evaluate the quality of generated this http URL all existing evaluation benchmarks rely on flawed selection criteria such as citation counts and structural coherence to select human-written surveys as the ground truth survey datasets, and then use surface-level metrics such as structural quality and reference relevance to evaluate generated this http URL, these benchmarks have two key issues: (1) the ground truth survey datasets are unreliable because of a lack academic dimension annotations; (2) the evaluation metrics only focus on the surface quality of the survey such as logical coherence. Both issues lead to existing benchmarks cannot assess to evaluate their deep "academic value", such as the core research objectives and the critical analysis of different studies. To address the above problems, we propose DeepSurvey-Bench, a novel benchmark designed to comprehensively evaluate the academic value of generated surveys. Specifically, our benchmark propose a comprehensive academic value evaluation criteria covering three dimensions: informational value, scholarly communication value, and research guidance value. Based on this criteria, we construct a reliable dataset with academic value annotations, and evaluate the deep academic value of the generated surveys. Extensive experimental results demonstrate that our benchmark is highly consistent with human performance in assessing the academic value of generated surveys. 

---
# Elsewise: Authoring AI-Based Interactive Narrative with Possibility Space Visualization 

**Authors**: Yi Wang, John Joon Young Chung, Melissa Roemmele, Yuqian Sun, Tiffany Wang, Shm Garanganao Almeda, Brett A. Halperin, Yuwen Lu, Max Kreminski  

**Link**: [PDF](https://arxiv.org/pdf/2601.15295)  

**Abstract**: Interactive narrative (IN) authors craft spaces of divergent narrative possibilities for players to explore, with the player's input determining which narrative possibilities they actually experience. Generative AI can enable new forms of IN by improvisationally expanding on pre-authored content in response to open-ended player input. However, this extrapolation risks widening the gap between author-envisioned and player-experienced stories, potentially limiting the strength of plot progression and the communication of the author's narrative intent. To bridge the gap, we introduce Elsewise: an authoring tool for AI-based INs that implements a novel Bundled Storyline concept to enhance author's perception and understanding of the narrative possibility space, allowing authors to explore similarities and differences between possible playthroughs of their IN in terms of open-ended, user-configurable narrative dimensions. A user study (n=12) shows that our approach improves author anticipation of player-experienced narrative, leading to more effective control and exploration of the narrative possibility spaces. 

---
# Knowledge Graphs are Implicit Reward Models: Path-Derived Signals Enable Compositional Reasoning 

**Authors**: Yuval Kansal, Niraj K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2601.15160)  

**Abstract**: Large language models have achieved near-expert performance in structured reasoning domains like mathematics and programming, yet their ability to perform compositional multi-hop reasoning in specialized scientific fields remains limited. We propose a bottom-up learning paradigm in which models are grounded in axiomatic domain facts and compose them to solve complex, unseen tasks. To this end, we present a post-training pipeline, based on a combination of supervised fine-tuning and reinforcement learning (RL), in which knowledge graphs act as implicit reward models. By deriving novel reward signals from knowledge graph paths, we provide verifiable, scalable, and grounded supervision that encourages models to compose intermediate axioms rather than optimize only final answers during RL. We validate this approach in the medical domain, training a 14B model on short-hop reasoning paths (1-3 hops) and evaluating its zero-shot generalization to complex multi-hop queries (4-5 hops). Our experiments show that path-derived rewards act as a "compositional bridge", enabling our model to significantly outperform much larger models and frontier systems like GPT-5.2 and Gemini 3 Pro, on the most difficult reasoning tasks. Furthermore, we demonstrate the robustness of our approach to adversarial perturbations against option-shuffling stress tests. This work suggests that grounding the reasoning process in structured knowledge is a scalable and efficient path toward intelligent reasoning. 

---
