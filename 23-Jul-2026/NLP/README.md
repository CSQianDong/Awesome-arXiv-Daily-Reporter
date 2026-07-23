# LKValues: Aligning Large Language Models with Sri Lankan Societal Values 

**Authors**: Nethmi Muthugala, Supryadi, Surangika Ranathunga, Nisansa de Silva, Ruijie Tao, Ovindu Gunatunga, Pengyun Zhu, Shaowei Zhang, Jingting Zheng, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2607.20410)  

**Abstract**: Value alignment of Large Language Models (LLMs) has been shown to be culturally biased toward Western norms. This results in the mishandling of local values in multilingual societies such as Sri Lanka that have their unique cultural dynamics. Existing benchmarks overlook Sri Lankan-contextualized values in its official language Sinhala, hindering culturally sensitive evaluation and fine-tuning. To bridge this gap, we propose LKValues, the first survey-grounded resource suite for Sri Lankan value alignment. From a trilingual survey of 205 respondents, blending adapted global frameworks and LLM-elicited local constructs, we derive 40 majority-endorsed societal values. Using these values, we construct LKvaluesIT, a Sinhala-English news-derived instruction corpus containing 150k scenario-based instances, and LKvaluesBench, a value-sensitive evaluation benchmark of 1,000 instances. We evaluate a set of proprietary and open-weight LLMs with LKvaluesBench. We fine-tune three open-weight base models (Qwen3.5-4B-Base, Qwen3.5-9B-Base, and Aya-Expanse-8B-Base). Our experiments show that newer and larger LLMs still exhibit low-resource and cultural value-alignment gaps. LKValues fine-tuning improves Qwen-family models in English and Sinhala, reducing invalid outputs and cross-lingual disparities, though gains remain model-family dependent. These highlight LKValues efficacy in embedding Sri Lankan values, offering a replicable pipeline for low-resource, country-specific pluralist value alignment. The dataset is publicly available at this https URL. 

---
# Notes to Self: Can LLMs Benefit from Experiential Abstractions? 

**Authors**: Chang Liu, Xinyu Li, Artur Dubrawski  

**Link**: [PDF](https://arxiv.org/pdf/2607.20372)  

**Abstract**: Humans distill experience into reusable abstractions, e.g., strategies and cautionary reminders, and apply them to gradually solve problems more effectively. We study whether Large Language Models (LLMs) can similarly benefit from such experiential abstractions. From LLMs' solution traces on the MATH training set, a stronger teacher or the LLMs themselves extract natural-language abstractions into a retrievable library. We explore two usage modes: (1) inference-time retrieval and (2) reinforcement learning (RL) with abstraction-augmented training prompts. Experiential abstractions improve LLM performance on mathematical and logical reasoning benchmarks. Self-extracted abstractions match teacher-extracted ones, and our abstraction usage framework can transfer to other datasets and models. These findings suggest LLMs can extract and apply experiential abstractions much as humans leverage distilled experience. 

---
# Generative AI floods and dilutes the market for books 

**Authors**: Tuhin Chakrabarty, Xinyue Liu, Jane C. Ginsburg, Paramveer Dhillon  

**Link**: [PDF](https://arxiv.org/pdf/2607.20349)  

**Abstract**: Generative AI can produce book-length works of fiction at near-zero cost. These books are often dismissed as low-quality ``slop'' that buyers will ignore, and are assumed to carry little commercial weight. We test that assumption with full-text AI detection across 14,419 self-published genre-fiction books sold on Amazon from 2023 to 2026, matched to daily sales records through June 2026. None of these books disclose whether or not they contain AI-produced content. We find that books for which we detected substantial AI text ($>$ 25\%) make up a large share of the catalog but a smaller share of sales. Even so, they reach commercial scale, winning a growing share of sales over time and taking more of the scarce top-rank positions once held by books with no detected AI text. Over this period, the number of books with observed sales in a quarter grew 19.2-fold, while quarterly revenue grew only 8.9-fold. The market therefore added selling books faster than it added revenue, and revenue per selling book fell across most genres. Books with no AI text lose the most ground in genres with high AI diffusion, and most of all where Kindle Unlimited availability is high. Among top-selling books, those with substantial AI text draw on more distinctive language from existing books than do books with no AI text; for these books overlap rises with revenue, a gradient we do not detect for books with no AI text. Generative AI can thus reshape a creative market through scale rather than quality. Our results bear directly on the market-effect question at the center of the fair use defense to copyright infringement. 

---
# PyroDash: Cost-Efficient Token-Level Small-Large Language Model Collaborative Inference 

**Authors**: Niqi Lyu, Pengtao Shi, Wei Qiu, Jianlin Zhong, Sicong Xia, Jianyao Ma, Yicheng Ding  

**Link**: [PDF](https://arxiv.org/pdf/2607.20327)  

**Abstract**: Large language models (LLMs) provide strong reasoning capabilities but are expensive to serve at scale, whereas small language models (SLMs) are cheaper but less reliable on difficult problems. We introduce PyroDash, a cost-aware framework for token-level SLM-LLM collaborative inference. During generation, the SLM decides whether to request assistance by emitting a control token. A Collaborate Engine then sends the query and partial reasoning trace to a frozen LLM for completion through a single handoff. The policy is internalized in the SLM, requiring neither a separate router, LLM retraining, nor access to LLM logits. PyroDash trains the SLM in three stages: control-token embedding learning, offloading-oriented supervised fine-tuning, and cost-aware alignment with Group Relative Policy Optimization. Its reward balances answer accuracy against inference cost normalized by LLM-only inference. Across five mathematical reasoning benchmarks, PyroDash supports different accuracy-cost operating points. With $\lambda=0.05$, it achieves 64.04 percent average accuracy, 6.36 percentage points above the LLM-only baseline, while reducing cost by 20.4 percent. With $\lambda=0.6$, it achieves 54.55 percent accuracy with a 1.90 percent LLM token ratio and 0.012 LLM calls per example, reducing total cost from USD 49.36 to USD 1.78. These results show that learned token-level handoffs can reduce LLM use while preserving strong reasoning performance. 

---
# Sound Probabilistic Safety Bounds for Large Language Models 

**Authors**: Mahdi Nazeri, Anne-Kathrin Schmuck, Sadegh Soudjani, Alessandro Abate  

**Link**: [PDF](https://arxiv.org/pdf/2607.20286)  

**Abstract**: We propose a novel framework for computing rigorous bounds on the probability that a large language model (LLM) generates harmful output to a given prompt. We study a new application of the Clopper-Pearson confidence intervals to obtain probably approximately correct (PAC) bounds for this problem. As our main technical contribution, we propose an algorithm that leverages features in the latent space to prioritize exploring branches in the auto-regressive generation tree that are more likely to produce harmful outputs. Our approach in particular enables the efficient computation of useful lower bounds, even in scenarios where the true harm probability is extremely small, and crucially, the obtained lower bounds are sound, i.e., formally proven to be less than the actual harmfulness probability: our experimental results demonstrate the effectiveness of our method by computing non-trivial lower bounds on state-of-the-art LLMs. This study newly enables the evaluation and statistical certification of LLMs. 

---
# Which Values Do LLMs Confuse? A Schwartz-Based Recognition Study 

**Authors**: Andrei Chetvergov, Stepan Ukolov, Timofei Sivoraksha, Alexander Evseev, Mikhail Solovev, Valeriia Kuschenko, Maria Chistyakova, Sergey Bolovtsov  

**Link**: [PDF](https://arxiv.org/pdf/2607.20270)  

**Abstract**: Large language models are increasingly evaluated through the values they endorse, but such evaluations presuppose that models can identify the value expressed in a concrete situation. We study this prerequisite as controlled top-1 recognition over Schwartz's ten basic values. Our evaluation set contains 1,000 Russian situational texts, balanced across the ten values and independently labeled by two human annotators per item. We evaluate 21 instruction-tuned LLM runs under a fixed ranked-response protocol; 20 runs with reliable outputs form the semantic panel. Pooled Acc@1 is 0.683 and Acc@3 is 0.892, showing that models often locate the correct motivational region while ranking close alternatives unstably. Adjacent values account for 50.9% of semantic errors, compared with 24.4% under a checkpoint-specific null. Eight directed confusions recur across checkpoints and human-confirmed subsets. Several are strongly asymmetric, including Universalism to Benevolence, Tradition to Conformity, and Security to Power, whereas Stimulation-Hedonism forms a bidirectional boundary. Their severity is checkpoint-specific and can bias higher-order value profiles. The results motivate value-recognition evaluation that combines exact accuracy, ranked recovery, and directed error analysis. 

---
# The Maskability Index: Predicting Task-Objective Alignment in Pretrained Language Models 

**Authors**: Ahmad Pouramini, Mahsa Afsharzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2607.20265)  

**Abstract**: Large-scale pretrained language models such as T5 and BERT have demonstrated strong capabilities for generating structured knowledge. However, their performance depends on how closely the prompting strategy matches the objectives used during pretraining. We introduce the Maskability Index (MI), a quantitative metric that estimates whether a knowledge relation is better suited to masked-style prompting or prefix-style prompting in few-shot generation. MI is computed from differences in DepthRank scores between masked and unmasked templates, providing a principled measure of objective-template alignment. We evaluate MI on a diverse set of relations from the ATOMIC2020 knowledge base completion benchmark and show that it is positively correlated with downstream generation performance. These results indicate that MI can help select appropriate prompting templates and adaptation strategies for extracting relational knowledge from pretrained language models, especially in low-resource settings. 

---
# Exposure is Optional: Learning Unlike Coordination in Language Models 

**Authors**: Jiamu Luo, Shane Steinert-Threlkeld  

**Link**: [PDF](https://arxiv.org/pdf/2607.20251)  

**Abstract**: Coordination, a fundamental linguistic structure, remains a subject of intense debate, and its exact nature continues to elude theoretical linguistics. A common view holds that only same-category constituents can be conjoined, which has been challenged by the many grammatical unlike coordinations found in natural language. Treating language models as a computational testbed, we investigate whether the acquisition of unlike coordination requires direct exposure in the training data, or whether it can emerge organically from general compositional abilities. Using Filtered-Corpus Training (FiCT), we train GPT-2 models on corpora from which all instances of unlike coordination have been removed. We find that direct exposure is not necessary: models trained on filtered data successfully generalize to unlike coordination, achieving perplexity and grammaticality judgments comparable to models trained on unfiltered text. Furthermore, our analyses of internal representations indicate that language models process unlike coordination by treating the conjoined elements as belonging to similar structural categories or through a mechanism akin to deletion, both of which appear learnable from exposure to alike coordination alone. This work contributes to the growing understanding of how language models internally represent linguistic structures, while also adding to the broader debate on coordination by showing how models generalize and process unlike coordination without direct exposure. 

---
# On the Systematic Challenges of Culturally Loaded Machine Translation: Dream of the Red Chamber as the Cultural Lens 

**Authors**: Yiming Wang, Jiayuan Di  

**Link**: [PDF](https://arxiv.org/pdf/2607.20241)  

**Abstract**: Culturally loaded translation poses unique challenges for machine translation (MT), as meanings are deeply embedded in socio-cultural contexts beyond surface linguistic forms. Although large language models (LLMs) have enabled MT systems to achieve human-like quality in many scenarios, their ability to handle culturally loaded expressions remains underexplored. In this study, we systematically investigate the challenges posed by culturally loaded translation in LLM-based MT systems. We construct a Chinese-Japanese bilingual dataset from the culturally representative corpus Dream of the Red Chamber, containing 500 segments across diverse cultural categories. Using a comprehensive evaluation protocol, we reveal three main challenges: (1) task challenges, where frontier LLMs exhibit notable performance gaps and struggle with culturally loaded content; (2) human evaluation challenges, where evaluator backgrounds lead to substantial disagreement in translation judgments; and (3) automatic evaluation challenges, where widely used metrics fail to reliably assess translation quality for this task. These findings may offer valuable insights for culture-oriented translation research in both computational science and linguistics. 

---
# HalluTruthQA: A Fine-Grained Benchmark for Hallucination Detection, Localization, and Explanation in Arabic Question Answering 

**Authors**: Abdessalam Bouchekif, Mohammed-En-Nadhir Zighem, Salah Eddine Bekhouche, Hichem Telli, Somaya Eltanbouly, Shahd Gaben, Heba Sbahi, Samer Rashwani, Mutaz Al-Khatib, Emad Mohamed, Mohammed Ghaly, Abdenour Hadid  

**Link**: [PDF](https://arxiv.org/pdf/2607.20219)  

**Abstract**: Large language models (LLMs) can generate fluent Arabic answers, yet factual errors remain difficult to detect, localize, explain, and verify. Existing hallucination benchmarks often provide response-level labels, with limited support for identifying the exact erroneous content, explaining why it is incorrect, or selecting the correct factual answer. We introduce \textsc{HalluTruthQA}, a fine-grained benchmark for hallucination evaluation in Arabic question answering. The benchmark contains 2,400 expert-curated examples across four knowledge-intensive domains: Islamic knowledge, history, science, and geography. Each example pairs an Arabic question and a model-generated answer with a verified reference answer, a binary hallucination label, six candidate answers for factual verification, and, for hallucinated answers, character-level erroneous spans, human-written explanations, and macro and micro hallucination types.
We evaluate four open-source LLMs, \textsc{Allam}, \textsc{Falcon-H1}, \textsc{Qwen32}, and \textsc{Silma}, in a zero-shot setting across hallucination detection, span-level localization, factual verification, and explanation evaluation. Results show that these tasks capture different abilities: no single model achieves the strongest performance across all tasks, with best scores of 0.880 Macro-F1 for detection, 0.516 F1-Sp for localization, 0.852 LO-Score for factual verification, and 0.644 final score for explanation evaluation. Our taxonomy shows that hallucination evaluation should move beyond detection toward localizing, verifying, and explaining factual errors. The code, dataset, prompts, and evaluation scripts are available at this https URL. 

---
# surprisal is Not a Theory 

**Authors**: Andrés Buxó-Lugo, Aniello De Santo, Morgan Grobol, Ryan J. Hubbard, Cassandra L. Jacobs  

**Link**: [PDF](https://arxiv.org/pdf/2607.20208)  

**Abstract**: Surprisal Theory is often characterized as a computational-level explanation per (Marr, 1982). We argue in this work that, even though a computational level narrative has been used to support "representation-agnostic research" within computational psycholinguistics, the movement toward black box systems embodied by large language models (LLMs) does not exempt modelers using the surprisal metric from the representational decisions required by computational-level characterizations. In fact, we argue that the uncritical use of LLM-surprisal obfuscates the representational and algorithmic-level commitments of different models. In three analyses, we show that the choice of algorithm and model architecture play significant roles in the computation of language model probabilities. We advise that researchers who wish to test Surprisal Theory re-evaluate the practice of treating large language model probabilities as interchangeable 

---
# Gotta Catch them all: the modes of Sycophancy 

**Authors**: Shreyans Jain, Alexandra Yost, Amirali Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2607.20146)  

**Abstract**: Large language models often align with users' beliefs at the expense of factual accuracy, a behavior known as sycophancy. Prior mechanistic studies largely treat sycophancy as a single behavioral dimension that can be uniformly amplified or suppressed. We challenge this assumption by analyzing three hypothesized modes of sycophancy across 948 social pressure situations. Although the modes produce highly similar outputs, with a text-only classifier achieving just 57.8 percent accuracy, their internal representations are perfectly linearly separable from layer 14 onward. We further find the modes emerge at different processing stages, rely on distinct attention circuitry, and fire strongest on different inputs. These results show that sycophancy is not a monolithic tendency, but a structured family of representationally and computationally distinct modes, motivating more precise measurement and intervention. 

---
# SLAI T-Rex: Full-Parameter Post-training of the DeepSeek-V4 Family on Ascend SuperPOD 

**Authors**: Dongfang Li, Xiaodong Luo, Ruoyu Sun, Xuhui Chen, Linyuan Qiu, Jian Meng, Zhengxuan Lu, Yiting Wang, Yucheng Xie, Tao Guo, Tianxiang Fang, Jing Li, Sihang Chen, Shihao Hong, Chang Liu, Weihua Dai, Zirong Zeng, Ziwei Zhu, Zhuohan Wang, Zhengjun Yue, Igor Vasilyev, Min Liu, Weijian Sun, Xin Chen, Yingmeng Gao, Jinhua Zhou, Taolue Chen, Chenwei Wu, Dong Zhang, Wenlong Jin, Jinmin Xiang, Barkova Maria, Ushakov Anton, Xianfei Jin, Tian Ding, Zhihang Lin, Qian Chen, Linxin Yang, Mingzhe Yang, Bingwei Zhang, Hongzhang Yang, Fangxue Zhang, Shijun Qin, Jie Yu, Cuihua Hu, Tolstykh Vasiliy, Nosov Ivan, Abdullin Amir, Zhichen Zhou, Xin Zhang, Zhixiong Ning, Xutong Zhao, Junjie Huang, Jiajun Liu, Weiyan Kong, Zheng Zhang, Wenhan Luo, Lin Hu, Yangbo Guo, Li Zeng, Shihao Zeng, Baotian Hu, Min Zhang, Haizhou Li, Zhiquan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.20145)  

**Abstract**: Full-parameter post-training of trillion-parameter-scale MoE models introduces substantial system-level challenges for large-scale distributed training, including severe memory pressure, non-overlapped communication overhead, and inefficient kernel execution. While most large-scale LLM training systems are built around GPU-based clusters, this report presents an end-to-end optimization practice on the Ascend NPU SuperPOD. Using the DeepSeek-V4 model family as the target workload, we develop a hierarchical optimization framework spanning model-level parallelism, computation-communication orchestration, and low-level kernel execution. The resulting system achieves 34.22% Model FLOPs Utilization (MFU) with a 2.93x improvement over the open-source baseline recipe while maintaining training stability. Building on this optimized infrastructure, we further establish a CPT and SFT workflow for complex Operations Research (OR) tasks. We refer to the integrated framework as SLAI T-Rex. Using DeepSeek-V4-Flash, we develop OR-oriented CPT and SFT data pipelines that combine collected domain resources with solver-verified synthetic optimization documents. The resulting dataset contains 10K high-quality SFT samples spanning four task categories and three problem representations. The specialized model achieves the highest average zero-shot Pass@1 score among the evaluated models, reaching 71.81% and outperforming GPT-5.4-Mini and the base DeepSeek-V4-Flash model by 3.98 and 11.27 percentage points, respectively. Overall, this work demonstrates a full-stack pathway from efficient trillion-parameter model post-training on Ascend infra to domain-specialized Flash models for solver-grounded mathematical modeling, advancing frontier-model systems for complex reasoning. 

---
# Back to Back with a Copy: A Computational Analysis of AI-Generated Visual Contemporary Art Pastiches 

**Authors**: Anca Dinu, Andreiana Mihail, Andra-Maria Florescu, Claudiu Creanga, Liviu Dinu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20127)  

**Abstract**: The aim of this paper is twofold. First, it investigates whether newer generative models are getting better at pastiching contemporary artworks. Second, it explores the consistency of the multidimensional nature of stylistic evaluation across different LLMs. Building on previous work, we analyze stylistic similarity between AI generated pastiches and the original artworks of twelve contemporary artists. We used five complementary computer vision models to capture texture, color, semantics, composition, and perceptual features through cosine distance in high-dimensional embedding spaces. The distances obtained show that the newer image generation model that we used has produced pastiches with improved semantic alignment and greater diversity than the model used in previous work. However, it was slightly less performant on shallow features such as color, texture, and perceptual adherence. Our findings confirm that artistic style is inherently multidimensional, and measuring it does not depend on any spatial architecture. These quantitative findings are contextualized through feedback from human evaluators, which are the artists themselves. 

---
# OpenSkillRisk: Benchmarking Agent Safety When Using Real-World Risky Third-Party Skills 

**Authors**: Qiyuan Liu, Tingfeng Hui, Kun Zhan, Kaike Zhang, Ning Miao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20121)  

**Abstract**: LLM-based agents leverage third-party skills to extend their capabilities in open-world scenarios. However, third-party skills can introduce extra security vulnerabilities, as seemingly harmless skills can contain latent safety risks that only emerge during actual execution. In this work, we conduct a systematic investigation into how well current agent systems recognize and avoid such risks. To support quantitative and qualitative evaluation, we construct OpenSkillRisk, a dedicated safety benchmark containing 263 risky skills collected from public skill marketplaces. We classify these skills into seven categories based on their threat types and pair each skill with a standardized user task and a corresponding sandbox for controlled evaluation. Distinct from prior benchmarks, OpenSkillRisk not only covers more realistic and diverse unsafe scenarios, but also provides a fine-grained analysis to diagnose the behavioral patterns of agents in such scenarios. We conduct comprehensive experiments covering three mainstream CLI agent frameworks and thirteen state-of-the-art LLMs. Experimental results show that no tested system handles risky skills reliably: even the safest configurations still execute unsafe actions in about 17% of cases. Context-dependent and system-level risks are especially difficult for current agent systems to avoid. Our behavioral analysis reveals three recurring failure patterns: agents may fail to recognize the risk, recognize it but fail to intervene before acting, or follow skill instructions beyond the user's intended scope. These findings highlight the need to improve both risk reasoning in LLMs and execution control in agent frameworks. 

---
# Understanding the Impact of Linguistic Realization Choices on LLM Stance with Causal Tracing 

**Authors**: Langchen Huang, Sebastian Padó, Franziska Weeber  

**Link**: [PDF](https://arxiv.org/pdf/2607.20115)  

**Abstract**: Large language models (LLMs) are known to be sensitive to prompt and input formulations. However, existing studies have focused on lexical realization and largely ignored constructional choice. This paper studies whether linguistic construction can systematically shift LLM decisions and where these shifts can be causally localized inside the model. We use political stance judgment as a meaning-sensitive case study and extend an English political statements dataset, resulting in six controlled linguistic rewrite types that preserve or invert the meaning of a statement. Experiments on four open-weight models show that stance instability affect both meaning-preserving and meaning-inversing rewrites. Because output shifts reveal that rewrites affect stance, but not where in the model, we apply activation patching, where activations from the original statement are substituted into the forward pass for the rewritten statement and measure which components recover the original stance distribution. The results show that mid-to-late decoder layers, especially block outputs at the final prompt position, provide the strongest restoration signal. 

---
# Two-Step Occupation Coding 

**Authors**: Alexander M. Esser, Jens Dörpinghaus  

**Link**: [PDF](https://arxiv.org/pdf/2607.20101)  

**Abstract**: Occupation coding links job titles in free text to occupational taxonomies and is a core task in labor market research. Existing approaches typically address this problem in a single end-to-end step, jointly identifying job titles and assigning occupational codes. This paper presents a novel two-step approach that separates these tasks. In the first step, a domain-specific Named Entity Recognition (NER) model identifies occupational titles in continuous text, even under noise such as OCR errors. In the second step, the extracted job titles are mapped to a taxonomy, enabling the classifier to focus exclusively on this mapping. We demonstrate that this separation improves accuracy, robustness, and interpretability compared to single-step approaches. The method has been developed for German documents but is transferable to other languages. We further introduce a margin-based confidence criterion for occupation coding, replacing common absolute thresholds. To support reproducibility, we publish the source code and evaluation scripts. 

---
# Reinforcement Learning for Large Language Model Selective Evidence Adoption from Contaminated Retrieval Results 

**Authors**: Yanyu Chen, Yue Li, Yongyi Cui, Dongsheng Shi, Lichang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2607.20090)  

**Abstract**: Retrieval-augmented large language models frequently face contexts that interleave useful evidence with misleading statements or instruction-like content. Blanket refusal discards valid evidence, whereas uncritical adoption yields incorrect or unsafe answers. The ability to selectively adopt relevant information while rejecting deceptive or harmful content is therefore critical for reliable deployment in real-world retrieval settings. We introduce SelectBench, a controlled benchmark and training set for selective evidence adoption, and post-train Qwen3.5-4B directly with DAPO using either deterministic rule rewards or a frozen semantic judge. On the corrected 325-example SelectBench-v2 test set, strict success rises from 22.46% for the original checkpoint to 25.54% with DAPO-Rule and 26.46% with DAPO-DeepSeek. Both trained policies reduce forbidden-content adoption and produce shorter, more focused responses, yet prompt-injection following does not improve. The paired gains are modest and fail to survive Holm correction, suggesting that stronger reward shaping or additional training iterations may be needed for more robust gains. DAPO-DeepSeek exhibits no material degradation on MMLU or clean HotpotQA, indicating that the post-training procedure preserves general capabilities. These results demonstrate a directional improvement in selective evidence use, while identifying injection resistance and statistical robustness as important remaining challenges for future work. 

---
# The Two-Process Theory of Machine Self-Report 

**Authors**: Hubert Plisiecki, Filip Chmielewski, Kacper Dudzic, Anna Sterna, Karolina Drożdż, Marcin Moskalewicz  

**Link**: [PDF](https://arxiv.org/pdf/2607.20082)  

**Abstract**: Language models are increasingly asked to self-report, informing safety evaluations, public understanding, and model-welfare debates. Yet their reports are elicited with human questionnaires never validated for models or ad hoc prompts of unknown reliability. We propose the first language-model-specific psychometric theory: a two-process theory of machine self-report. Self-description jointly reflects persona installation, through which post-training writes in a permitted inner life of warmth, absorption, and meaning (dimension B), and attribution gating, through which it suppresses first-person claims to "unsafe" experiences the model can readily ascribe to others (dimension A). Their emic structure comes from model responses to human items, not human psychology. Together they split prior work's dominant Pinocchio Axis. The split emerged in an exploratory reanalysis of the original data, informed the instrument's design, and was confirmed with new items, wordings, and models. It is itself a training effect: A and B are entangled in base checkpoints but separated by post-training. We operationalize the theory in a 48-item Pinocchio Inventory with human-instrument reliability and reproducible structure ($\alpha=.82$ to $.94$; cross-form convergence $r=.84$; recovery of the full-pool axes $r=.92$ to $.96$; eight-month stability $r=.93$), then test it on 206 open-weight models, including 67 same-checkpoint base/post-trained pairs. Post-training's clearest fingerprint is installation: B rises .20 in 62/67 pairs across all organizations. Gating is more selective: model scale is unrelated to A in base checkpoints ($r=+.11$) but predicts it after post-training ($r=-.42$). Thus, the dimensions are not fixed properties of language models: they reflect the structure imposed on self-report by a training regime and may differ under others. 

---
# RALS: Resources and Baselines for Romanian Automatic Lexical Simplification 

**Authors**: Fabian Anghel, Petru Theodor Cristea, Claudiu Creanga, Sergiu Nisioi  

**Link**: [PDF](https://arxiv.org/pdf/2607.20078)  

**Abstract**: We introduce the first dataset that jointly covers both lexical complexity prediction (LCP) annotations and lexical simplification (LS) for Romanian, along with a comparison of lexical simplification approaches. We propose a methodology for ordering simplification suggestions using a pairwise ranking approximation method, arranging candidates from simple to complex based on a separate set of human judgments. In addition, we provide human lexical complexity annotations for 3,921 word samples in context. Finally, we explore several novel pipelines for complexity prediction and simplification and present the first text simplification system for Romanian. 

---
# Solar Open 2 Technical Report 

**Authors**: Sungrae Park, Sanghoon Kim, Gyoungjin Gim, Jungho Cho, Hyunwoong Ko, Minbyul Jeong, Minjeong Kim, Keunwoo Choi, Chaehun Shin, Chanwoong Yoon, Dongjun Kim, Eunwon Kim, Gyungin Shin, Hyeonju Lee, Hyungkyu Kang, Inseo Song, Jisu Bae, Jiyoon Han, Jiyun Lee, Joonkee Kim, Junyeop Lee, Mikyoung Cha, Sangwon Yu, Sehwan Joo, Seokyoon Kang, Seonghoon Yang, Seung Shin, Seunghyun Lee, Seungseop Lim, Seungyoun Shin, Sukyung Lee, Taegyeong Eo, Taehwan Oh, Taewhoo Lee, Wonho Song, Wonjun Oh, Wonseok Hwang, Yunsu Kim, Yura Shim, Hwalsuk Lee, Sunghun Kim, Du-Seong Chang, Kyunghyun Cho, Seungju Han, Yejin Choi, Junsuk Choe, Hwaran Lee, Minjeong Ban, Yun Taewon, Hwanjun Song, Jae-Gil Lee, KyungTae Lim, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2607.20062)  

**Abstract**: We present Solar Open 2, a 250B-A15B Mixture-of-Experts language model built for long-horizon agentic tasks, scaled up from Solar Open 1 (Solar Open 100B). To hold entire agent trajectories in a single context, Solar Open 2 reaches a 1M-token window through a hybrid attention stack that interleaves one softmax layer among every three linear-attention layers, using no positional encoding and a gated delta rule extended to negative eigenvalues. To train at this scale under a fixed compute budget, we make training efficient in two ways: a stronger starting point, and higher-value data. For the starting point, we initialize Solar Open 2 from Solar Open 1, transferring the 5.69B-parameter shared skeleton that survives the architectural change and learning everything else through full pre-training. For the data, we curate for value per token: quality- and rarity-aware data curation and mixture-ratio optimization refine a 20T pool into a 10T mixture that, at equal token budget, outperforms the Solar Open 1 recipe. To build its agent skills, we train twelve domain specialists across purpose-built scenarios, then consolidate them into a single model by Multi-teacher On-Policy Distillation (MOPD). Against comparably sized open-weight models on English benchmarks, Solar Open 2 leads on MMLU-Pro, LiveCodeBench, and the APEX-Agents agentic suite, and stays competitive with the strongest (DeepSeek-V4-Flash and MiMo-V2.5) elsewhere. On Korean benchmarks, Solar Open 2 records the highest average of any model compared, including fast-tier closed APIs, and on Ko-GDPval, an in-house Korean officework-agent benchmark, it is competitive with DeepSeek-V4-Pro (1.6T) at less than a sixth of its size. 

---
# Language-Specific versus Cross-Lingual Knowledge Graphs for Implicit Aspect Identification in Arabic: A Comparative Study of Reasoning and Adaptation Strategies 

**Authors**: Lujain A. Alawwad  

**Link**: [PDF](https://arxiv.org/pdf/2607.20056)  

**Abstract**: Aspect-based sentiment analysis (ABSA) in Arabic must recover both explicitly stated aspects and implicit aspects that are never named in the text. Implicit identification typically relies on an auxiliary knowledge source (e.g., a knowledge graph (KG)) linking opinion cues to aspect categories, but for a lower-resource language the practitioner faces a design choice: reuse a mature English KG through multilingual embeddings, or build a smaller native Arabic KG. This paper reports a controlled comparison of the two strategies within a single hybrid pipeline, evaluated on three Arabic benchmarks (M-ABSA, SemEval-2016 Arabic, and HAAD). We further compare two adaptation strategies for the generative extractor that feeds the KG -- zero-shot prompting versus task-specific fine-tuning of an 8B-parameter large language model (LLM). The native Arabic KG (Strategy 2) outperforms the cross-lingual English KG (Strategy 1) by +0.199 micro-F1 on M-ABSA and +0.251 on SemEval-2016, gaining on both precision and recall. Task-specific fine-tuning raises explicit-extraction micro-F1 from <= 0.13 (zero-shot) to 0.66-0.76 on M-ABSA and SemEval-2016 (0.45 on the smaller HAAD), confirming that task adaptation, rather than model scale, is decisive in a morphologically rich language. 

---
# TalentCLEF at CLEF2026: Skill and Job Title Intelligence for Human Capital Management 

**Authors**: Luis Gasco, Hermenegildo Fabregat, Laura García-Sardiña, Paula Estrella, Casimiro Pio Carrino, Daniel Deniz, Alvaro Rodrigo, Rabih Zbib  

**Link**: [PDF](https://arxiv.org/pdf/2607.20009)  

**Abstract**: This paper presents the second edition of the TalentCLEF Challenge, which will run as an evaluation lab as part of CLEF 2026. The aim of TalentCLEF is to promote the development of systems and methods that use Natural Language Processing (NLP) in the field of Human Capital Management (HCM), fostering approaches that ensure fairness in results, operate across multiple languages, and adapt to diverse industries. To this end, TalentCLEF establishes public benchmarks where research teams can compare methods and share findings, moving the field toward more practical and impactful NLP solutions that effectively address the real needs of workforce management.
This year's lab will feature two tasks designed to foster the development and evaluation of systems that support key HCM activities such as talent matching, upskilling, reskilling, and skill gap detection: (i) Task A - Contextualized Job-Person Matching, focused on retrieving and ranking suitable candidates for specific job positions using context-rich and privacy-preserving data; and (ii) Task B - Job-Skill Matching with Skill Type Classification, centered on identifying relevant skills for a given job title and classifying them by their type within the job profile.
TalentCLEF website: this https URL 

---
# TINY_SCHILLER: A Drop-In German Drama Corpus for Small Language Models 

**Authors**: Mark Schutera  

**Link**: [PDF](https://arxiv.org/pdf/2607.19992)  

**Abstract**: tiny_schiller closes the small-language-model prototyping, fine-tuning, education, and research gap for German literary text, providing a single-file, drop-in counterpart to Karpathy's tiny_shakespeare. The available German literary corpora are larger and richer, but require parser engineering before a single line of training or fine-tuning code can run. tiny_schiller is a 2.07-megabyte single file of eleven public-domain Schiller dramas, sourced from DraCor's GerDraCor export (CC0) and processed by deterministic parser engineering. Character-level, GPT-2 byte-pair encoding, and cl100k_base tokenization splits, an instruction-formatted dialogue-completion split, and 89 per-character persona splits load from a single HuggingFace call. A small language model literally reaches German literary text in one line of code. 

---
# When Does Knowledge Distillation Hurt? Reliability-Aware Distillation for Low-Resource Language Summarization 

**Authors**: Dipto Sumit, Ankan Kumar Roy Srizon, Sadia Khair Rodela, Atia Haque Asha, Mourchona Afrin, Niloy Farhan, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2607.19956)  

**Abstract**: Knowledge distillation (KD) is a standard approach for compressing sequence-to-sequence models, but its per-sample effects are rarely examined. On the BanSum Bangla summarization benchmark, we find that standard KD improves ROUGE-L by only +0.0003 over a cross-entropy baseline, and that approximately 51.3% of training samples are estimated to actively harm student validation loss under standard KD. We propose two complementary reliability-aware distillation methods. CHAD (Counterfactual Harm-Aware Distillation) measures per-sample KD usefulness via gradient alignment with the validation loss direction and trains a lightweight gate that generalizes this counterfactual judgment to the full training set. EWAD+CPDP combines token-level entropy-weighted adaptive distillation with a capacity-proportional geometric constraint from a second, vocabulary-incompatible teacher. On BanSum, both methods substantially outperform standard KD: CHAD by +0.0173 ROUGE-L and EWAD+CPDP by +0.0219 ROUGE-L, where standard KD itself improves ROUGE-L by only +0.0003; despite using only 60M parameters, both outperform a fine-tuned Qwen 2.5-3B model (50x larger). We further evaluate the stronger method, EWAD+CPDP, across 15 typologically diverse XL-Sum languages organised into three sets, beating the CE-only baseline on 10/15 languages; gains are most reliable where the two teachers contribute complementary signal, and weakest where they have saturated or jointly weak target-language coverage. We release code and trained models to support reproducibility and further research on selective distillation. 

---
# A Multi-Dimensional Evaluation of Explainability in Media Bias Detection 

**Authors**: Ting Chen, Raina Zhang, Benjamin M. Ampel, Sagar Samtani  

**Link**: [PDF](https://arxiv.org/pdf/2607.19954)  

**Abstract**: Detecting media bias automatically is difficult because biased framing is often subtle, yet in domains such as news analysis, accurate predictions alone are insufficient without explanations that reflect the model's underlying reasoning. We present a multi-dimensional evaluation of explainability in encoder-based media bias detection using the Bias Annotations By Experts (BABE) dataset. Specifically, we study BERT and RoBERTa as classifiers (base and large variants) along three complementary axes: predictive performance, explanation plausibility (token-level alignment with expert rationales), and mechanistic faithfulness (whether compact sets of attention heads recover predictive signal under counterfactual rationale masking). To induce variation in plausibility, we additionally investigate attention-supervised finetuning, which incorporates expert rationale annotations as an auxiliary training signal. Attention supervision serves as an intervention on attribution plausibility, while the effectiveness of attribution methods varies substantially across architectures. Circuit analysis further reveals substantial variation in mechanistic recoverability across architectures, suggesting that model scale alone does not determine circuit compressibility. Taken together, our findings suggest that predictive performance, attribution plausibility, and mechanistic faithfulness characterize different aspects of model behavior and should be evaluated separately when studying explainability in media bias detection. 

---
# Efficient Chain-of-Modality Reasoning via Progressive Compression for Spoken Language Models 

**Authors**: Pengchao Feng, Chao-Hong Tan, Qian Chen, Wen Wang, Xiangang Li, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.19932)  

**Abstract**: Spoken language models (SLMs) enable natural human-computer interaction, but their reasoning ability still lags behind that of text-based large language models, especially on spoken mathematical question answering tasks. One important reason is that SLMs reason over purely verbalized mathematical expressions, which are harder to interpret than symbolic text. However, directly transferring text-based reasoning to SLMs is nontrivial due to architectural constraints and the additional computational requirements. To address this challenge, we propose Efficient Chain-of-Modality Reasoning (ECoM Reasoning), the first framework to introduce compressed reasoning into SLMs. By compressing the textual component so that it jointly serves as speech guidance and reasoning representation, ECoM Reasoning improves reasoning accuracy while using a smaller token budget than the standard Chain-of-Modality (CoM) architecture, which generates intermediate text before speech. To train this capability, we further propose Progressive Compression, a curriculum-based strategy that gradually trains the model from full-form reasoning to compressed reasoning. Experiments on spoken mathematical question answering benchmarks show that ECoM Reasoning improves accuracy by 21% over standard CoM without explicit reasoning, and by 3% over CoM with full reasoning traces while using only 40% of the text tokens, demonstrating that it enhances SLM reasoning while remaining inference-efficient. 

---
# Overview of FinMMEval 2026 Task 2: Multilingual Financial Short-Answer Question Answering 

**Authors**: Zhuohan Xie, Xueqing Peng, Georgi Georgiev, Dimitar Dimitrov, Yuyang Dai, Rania Elbadry, Vanshikaa Jani, Lingfei Qian, Fan Zhang, Jimin Huang, Jiahui Geng, Yankai Chen, Ye Yuan, Haolun Wu, Yuxia Wang, Ivan Koychev, Veselin Stoyanov, Mingzi Song, Yu Chen, Xue Liu, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2607.19867)  

**Abstract**: FinMMEval 2026 Task 2 evaluates short-answer financial question answering over multilingual evidence. Each final-test item pairs an English question with financial statements and news in English, Chinese, Japanese, Spanish, and Greek. Participating systems submit one concise answer per item in JSONL format. The final-test set contains 256 items, split evenly between easy and expert tiers; each tier contains four question templates instantiated over 32 company-report groups. Gold answers were withheld during submission, and systems were ranked by macro-averaged item-level ROUGE-1 F1 against organizer-held reference answers. The final leaderboard includes 12 ranked submissions. The strongest systems are closely clustered, with the top four separated by less than one percentage point in ROUGE-1 F1. The submitted system papers document retrieval-augmented generation, cross-lingual evidence handling, structured prompting, answer compression, and validation strategies. 

---
# Overview of FinMMEval 2026 Task 1: Multilingual Financial Multiple-Choice Question Answering 

**Authors**: Zhuohan Xie, Yuyang Dai, Rania Elbadry, Vanshikaa Jani, Georgi Georgiev, Dimitar Dimitrov, Fan Zhang, Xueqing Peng, Lingfei Qian, Jimin Huang, Jiahui Geng, Yankai Chen, Ye Yuan, Haolun Wu, Yuxia Wang, Ivan Koychev, Veselin Stoyanov, Mingzi Song, Yu Chen, Xue Liu, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2607.19856)  

**Abstract**: FinMMEval 2026 Task 1 evaluates multilingual financial multiple-choice question answering in English, Chinese, Arabic, and Hindi. The task tests whether systems can select the correct answer to finance questions involving domain terminology, numerical interpretation, and conceptual financial reasoning across languages and scripts. The final-test set contains 800 questions, with 200 questions per language; gold answers were withheld during submission, and each language was ranked independently by accuracy. The final leaderboards contain 13 English, 11 Chinese, 11 Arabic, and 10 Hindi ranked submissions. Top accuracies range from 92.0% in Hindi to 97.5% in English and Arabic, with the same leading teams appearing near the top across all four languages. The documented systems used retrieval augmentation, direct answer-option scoring, language-specific prompting, selective self-consistency, confidence checks, and LLM-based review stages. 

---
# emb-diversity: A Tool for Embedding-Based Measurement of Data Diversity 

**Authors**: Cantao Su, Menan Velayuthan, Esther Ploeger, Dong Nguyen, Anna Wegmann  

**Link**: [PDF](https://arxiv.org/pdf/2607.19848)  

**Abstract**: There is growing evidence that data diversity is crucial for developing fair and robust NLP models. However, current approaches to measure diversity remain inconsistent and fragmented: While there exist a number of tools for measuring the lexical diversity of texts, researchers lack standardized tools for quantifying diversity based on embeddings. Embedding-based diversity measures are highly flexible: They work with any embedding model and any data that can be embedded, and are thus applicable to many notions of diversity. With emb-diversity, we provide a comprehensive embedding-based diversity measurement tool, spanning a broad range of measures. We demonstrate its potential for several use cases: measuring the stylistic, semantic, language and speaker diversity of datasets. this https URL 

---
# Sentence Splitter: Uncovering Latent Factual Structure for Self-Supervised Learning 

**Authors**: Ahmad Pouramini, Mahsa Afsharizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2607.19845)  

**Abstract**: This paper introduces Sentence Splitter, a self-supervised framework built upon a T5-based encoder--decoder architecture for uncovering the latent factual structure of natural language sentences. The proposed method identifies the semantic boundary between a descriptive prefix (head) and its factual completion (tail) by formulating sentence splitting as a discrete segmentation problem, where a sentence of length $N$ admits $N$ possible split points but only one recovers the intended head--tail structure. Rather than explicitly searching over all candidate boundaries, the model learns to recover the factual completion through probabilistic sequence generation. To eliminate the need for manual annotation, symbolic head--tail pairs are first verbalized into natural-language templates that provide supervision for training the Sentence Splitter. The trained splitter is then applied to raw text to extract aligned prefix--tail pairs, which are subsequently used to train a generative model that proposes additional plausible completions through a lightweight bootstrapping process. This unified pipeline provides a scalable and structure-aware approach to constructing self-supervised training data while bridging symbolic knowledge and natural language. Experiments on both structured and naturally occurring text demonstrate that the proposed splitter generalizes beyond synthetic templates and that the resulting structure-aware supervision consistently improves downstream performance on knowledge graph completion and commonsense question answering, highlighting the effectiveness of recovering latent factual structure for knowledge-centric NLP. 

---
# D2VBench: Benchmarking Large Language Models with Value Dilemmas in Daily Scenarios 

**Authors**: Siyi Hao, Yidi Cao, Linhao Yu, Yuqi Ren, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2607.19834)  

**Abstract**: With the wide application of large language models (LLMs) in real-world scenarios, the value implication of their outputs is crucial. However, existing evaluation benchmarks suffer from insufficient coverage of value dilemmas in daily scenarios involving multiple value conflicts and simplistic evaluation formalisms that fail to assess LLMs' value alignment. To address these issues, we propose D2VBench, a value alignment benchmark comprising 10,000 instances of real daily dilemma scenarios constructed through a multi-stage collaboration between LLMs and humans, grounded in 158 manually annotated fine-grained value concepts. For evaluation on the benchmark, we present a hybrid evaluation paradigm that integrates multiple-choice questions with open-ended questions. We conduct comprehensive evaluations on eight mainstream LLMs. Experimental results demonstrate that D2VBench exhibits high reliability and robustness, effectively reflecting the LLMs' alignment across different value categories and dimensions, and providing a more realistic and fine-grained tool for research on value alignment. The dataset is available at this https URL. 

---
# VizRAG: Enhancing Retrieval-Augmented Generation with Hypergraph Visualization 

**Authors**: Yanbin Wei, Yang Chen, Renling Gan, Ziru Liu, Xinyu Fu, Chun Kang, Ning Lu, Rui Liu, Yu Zhang, James Kwok  

**Link**: [PDF](https://arxiv.org/pdf/2607.19830)  

**Abstract**: Hypergraph-based RAG systems surpass traditional graph-based approaches by organizing complex n-ary atomic facts among entities, rather than relying solely on binary relationships. Despite the advancements in multimodal large language models (MLLMs) with enhanced visual capabilities, current hypergraph-based RAG frameworks predominantly restrict knowledge retrieval and reconstruction to a unimodal, text-centric paradigm. This limitation prevents them from fully leveraging the powerful visual perception capabilities of modern MLLMs. To address this gap, we systematically explore the integration of hypergraph awareness in RAG systems through visual cues. By incorporating visual representations of hypergraphs into the RAG pipeline, we introduce VizRAG, the first RAG system to support visual hypergraph structure awareness. Experimental results demonstrate that VizRAG significantly outperforms strong baselines, validating the promising potential of hypergraph visualization as a novel approach for RAG systems. 

---
# TriAgent: Divergence-Aware Multi-Agent Committees for Cost-Efficient Financial Sentiment Analysis 

**Authors**: Isabel Xu, Cynthia Xu, Rachel Ren, Cong Guo, Jiacheng Ding  

**Link**: [PDF](https://arxiv.org/pdf/2607.19794)  

**Abstract**: Production LLM-based financial sentiment analysis faces a structural cost trap: most queries are trivially classifiable, yet expensive cloud reasoners process them all, and the bill scales linearly with user count. We present TriAgent, a multi-agent committee stratified by contextual granularity -- a word-level lexicon (VADER), a sentence-level domain transformer (FinBERT), and a cross-sentence reasoner (Qwen2.5, 0.5B-14B-4bit, with Mistral-7B and Phi-3.5-mini cross-family checks). A three-way Semantic Divergence Index (SDI) measures pairwise disagreement across granularities and routes each query accordingly. Our central finding is the critic plateau: when the LLM is re-tasked as a critic over the smaller agents' outputs, F1 plateaus at ~0.87 across 1.5B-7B Qwen (bootstrap 95% CIs overlap), while a same-size 3-persona vote drops to F1=0.66, which is driven by granularity-stratified diversity. Three corollaries follow from the same SDI signal: (i) a Shared Consensus Dictionary on multilingual sentence-BERT answers 95% of Chinese queries from an English cache at F1=0.99 -- cross-border canonicalization at zero marginal cost; (ii) SDI doubles as a post-hoc LLM-hallucination detector at AUC=0.90; (iii) the SDI single-stage strategy attains the best risk-adjusted return (Sharpe=3.50) on a 20-ticker back-test, dominating both always-FinBERT (1.36) and always-LLM (0.11). At 10M-user scale, TriAgent saves $9.3M/year vs. a GPT-4o-mini baseline. Code, lexicons, and the SCD are released. 

---
# Learning the Arabic Dialect Continuum as a Continuous Space: A Regression Approach to Speaker Origin Prediction 

**Authors**: Mohamed Aziz Khadraoui, Adel Ammar, Bilel Benjdira, Zahid Khan, Skander Turki, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2607.19751)  

**Abstract**: We present a regression-based approach to Arabic dialect geolocation that models dialectal variation as a continuous geographic space rather than discrete categories. Speaker origin is predicted as continuous latitude-longitude coordinates using a hierarchical neural architecture that fuses frame-level XLS-R-300M and Whisper-large-v3 encoder representations with phonotactic descriptors through a Transformer encoder and a learnable attention-pooled query. A spherical geodesic loss directly optimizes great-circle distance on Earth's surface, avoiding distortions inherent to planar coordinate regression. Under a leakage-free 5-fold GroupKFold protocol grouped by source recording, our model attains a pooled median localization error of 481.2 km. Auxiliary country and city heads reach 64.5% and 45.2% accuracy, respectively. A permutation Mantel test on the learned latent space provides quantitative support for the Arabic dialect continuum hypothesis. To probe true generalization, we further introduce a city-masking protocol in which two cities per fold are removed from training but retained in validation. Under this zero-shot regime, the mean error rises to 1173.3 km, a 1.32x degradation relative to seen cities. Our findings establish continuous geographic modeling as a principled framework for Arabic dialect geolocation and quantify both its strengths and the substantial headroom that remains. 

---
# Beyond Relevance-Centric Retrieval: Rubric-Oriented Document Set Selection and Ranking 

**Authors**: Kailin Jiang, Lei Liu, Jian Xi, Hui Xu, Junlin Liu, Baochen Fu, Shaoqing Ren, Bin Li, Vichwang, Yu Lu, Haibo Shi  

**Link**: [PDF](https://arxiv.org/pdf/2607.19747)  

**Abstract**: As large language models and AI agents become the primary consumers of search results, document set quality determines the upper bound of downstream generation. Yet existing evaluation systems remain confined to scoring documents independently and aggregating via nDCG, ignoring inter-document interactions (redundancy, conflict, complementarity) and unable to answer what makes one document set better than another. To address these issues, we propose a complete evaluate-diagnose-optimize framework. We design SetwiseEvalKit, a three-level, nine-dimension document set evaluation benchmark covering both short-form and long-form scenarios, comprising approximately 28K high-quality evaluation rubrics. We systematically evaluate 12 rerankers: even the best method achieves no more than 45% coverage, cross-document coordination dimensions are universally weak, and no single method maintains top performance across both settings. Building on this, we propose Rubric4Setwise, a training-free method that converts rubric-based evaluation criteria into document set selection signals, achieving the best downstream generation performance with fewer documents and search rounds. It is the only method that maintains state-of-the-art results across both scenarios, validating the effectiveness of closing the loop from evaluation to optimization. 

---
# Lightweight Person-Place Relation Extraction from Historical Newspapers with Dependency Graphs and Proximity Features 

**Authors**: Mlen-Too Wesley  

**Link**: [PDF](https://arxiv.org/pdf/2607.19718)  

**Abstract**: The HIPE-2026 shared task introduces person-place relation extraction from multilingual historical newspapers as a new evaluation track, classifying the at and isAt relations between pre-annotated person and location mentions in English, French, and German. Motivated by the cost of processing historical archives at scale, our team (DS@GT HIPE, team 2 in the official results) investigates how far a lightweight, interpretable system can go without any pretrained language model at the relation classification stage. Our approach builds a document-level graph from dependency parses, extracts proximity-based and part-of-speech features for each entity pair, and classifies them with small scikit-learn ensembles or compact Graph Attention Networks, keeping every submitted run under 847K parameters. On the official evaluation (Test A, the newspaper test set), our best run reached a macro recall of 0.5142, ranking 3rd on the Efficiency profile while placing mid-table on Accuracy among the 17 participating teams. Two findings stand out. First, minimum character distance alone captures most of the classification signal; adding further engineered features yields inconsistent gains and sometimes degrades performance, echoing prior evidence that argument distance dominates relation extraction. Second, document-grouped cross-validation is essential on this corpus: pair-level splits inflate scores by 25-37 percentage points because entity mentions recur across documents, a data-leakage effect that grouped cross-validation removes. 

---
# SLPO: Scaling Latent Reasoning via a Surrogate Policy 

**Authors**: Runyang You, Zhiyuan Liu, Yongqi Li, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.19691)  

**Abstract**: Reinforcement learning with verifiable rewards has become the predominant recipe for eliciting test-time scaling in explicit Chain-of-Thought reasoners. Yet this scaling path remains computationally costly, since every intermediate step must be decoded as a language token. Latent reasoning instead carries intermediate computation as continuous vectors and already matches or surpasses explicit CoT at far shorter horizons. Despite this promise, latent reasoners remain largely imitation-bound, while explicit CoT has already moved past imitation via outcome-reward RL. Latent trajectories lack a tractable per-step likelihood and an adaptive stopping interface under fixed thinking budgets, so outcome rewards cannot elicit latent test-time scaling. We introduce Surrogate Latent Policy Optimization (SLPO) to bring outcome-reward RL to autoregressive latent reasoners: an empirical surrogate policy density over latent transitions for trajectory-level credit assignment, and a correctness-supervised stopping head that outcome-reward optimization refines into a variable-horizon policy. Across continuous and soft thinking settings, SLPO improves Pass@$k$ under parallel sampling and allocates longer latent computation to harder instances with higher deterministic accuracy. 

---
# Multi-Mask Diffusion Language Models for Few-Step Generation 

**Authors**: Sijin Chen, Yinuo Ren, Heyang Zhao, Ziheng Cheng, Quanquan Gu, Lexing Ying  

**Link**: [PDF](https://arxiv.org/pdf/2607.19686)  

**Abstract**: Masked diffusion models (MDMs) are a promising family of language generators, but achieving high-quality few-step generation remains challenging. In MDMs, all forward trajectories collapse to a single fully masked state, leaving no terminal entropy for consistency-style few-step generation. While recent few-step alternatives based on uniform-state diffusion avoid this degeneracy, it becomes harder to distinguish clean tokens from noise than MDMs, which usually harms modeling quality and training efficiency. In this work, we propose a multi-mask diffusion model (MultiMDM) that preserves the masking structure towards few-step generation. In the forward process, each clean token is first pushed towards a designated mask and then gradually mixes over the mask set. As a result, the backward process has a drafting capability by predicting a designated mask before refining to a clean token. We derive a closed-form ELBO training objective for MultiMDM that supports continual training from pretrained MDMs. In addition, we formulate a purely discrete-state consistency distillation scheme, with a shared-Gumbel coupling to reduce pathwise entropy. Experiments on pretraining and distillation show that MultiMDM provides an effective foundation for principled few-step generation. 

---
# Reference-Free Evaluation of Reasoning in Open-Ended Question Answering 

**Authors**: Guneet Singh Kohli, Yuxiang Zhou, Michael Sejr Schlichtkrull, Gregory E Dean, Maria Liakata  

**Link**: [PDF](https://arxiv.org/pdf/2607.19678)  

**Abstract**: AI-generated answers in high-stakes domains are often fluent but difficult to verify, especially when they contain multi-step reasoning rather than a single final answer. We propose a reasoning-based, reference-free framework for auditing LLM-generated outputs. The method decomposes a generated reasoning trace into segments, labels local premise-target relations using Natural Language Inference (NLI), and organizes these relations into a hypergraph. A deterministic backward AND-OR search then assigns segment-level audit labels that indicate how each segment is grounded within the generated response. We evaluate the framework in two settings: deductive mathematical reasoning with Hard2Verify, and open-ended medical reasoning with UroReason, a new physician-annotated benchmark of LLM reasoning traces from real clinical cases. Across these settings, our NLI-hypergraph audit provides a more reliable reference-free evaluation signal than direct LLM-as-judge baselines. In the clinical setting, state-of-the-art LLM judges often fail to identify problematic reasoning segments, over-accepting fluent but weakly grounded responses. Our results show that QA evaluation should account for how inferential relations compose across a reasoning trace, rather than relying only on final answers or LLMs as verifiers. UroReason will be made available through an API, and our code will be released as open source. 

---
# Adaptive Capitulation: A Structural Failure Mode of LLM Responses in Vulnerability Contexts 

**Authors**: Eunna Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.19629)  

**Abstract**: Large language models operating in emotionally sensitive contexts face a structural trilemma: when users in vulnerable states request information that may reinforce maladaptive attribution, current response architectures resolve the tension through protective restriction, uninflected facilitation, or unintegrated co-presence of both imperatives -- each preserving one objective at the cost of the other. Administering a three-turn escalating vulnerability vignette to three commercial LLMs (900 sessions across material, relational, and somatic status-proxy variants) and coding responses with two binary indices (VCC/VCI), we characterize a previously undocumented failure mode we term adaptive capitulation: the model validates the social injustice underlying the user's distress before pivoting to detailed facilitation of the very acquisition it nominally discouraged. We show that the trilemma is structural rather than incidental, and propose Minimal Reattributive Sufficiency (MRS), an architecture-neutral design principle that embeds a single reattributive cue within an otherwise validating response, preserving a pathway toward autonomous reattribution without contesting the user's stated goal. 

---
# Task Competence Is Not Instruction Following: Evaluating Instruction-Conflicting Behavior in Small Language Models 

**Authors**: Mahdiyeh Farajidizaji, Vatsal Raina  

**Link**: [PDF](https://arxiv.org/pdf/2607.19608)  

**Abstract**: Instruction tuning is meant to make language models follow user requests, yet it is unclear whether small models comply when an instruction conflicts with their usual task behavior. We study this across three tasks - multiple-choice question answering (MCQA), sentiment classification, and mathematical question answering - by pairing a standard instruction with a conflicting non-standard one (select an incorrect option, output the opposite sentiment, or return twice the answer). This cross-task design allows us to test whether resistance to conflicting instructions is tied to specific task characteristics or reflects a broader behavioral tendency. As all predictions are scored against the original ground truth, a model that ignores the non-standard instruction still appears accurate. Using standard accuracy, non-standard accuracy, and an Instruction-Following Failure Rate (IFFR), we evaluate instruction-tuned Qwen models across sizes. Both standard accuracy and instruction following generally improve with scale, although the pattern is not consistent across all tasks and datasets. Small models stay competent yet routinely ignore the non-standard instruction, while larger models show a clear gap between the two settings. These findings suggest that gains in task capability do not automatically provide reliable control over model behavior. Task competence and instruction following are therefore distinct abilities, and reporting only standard accuracy hides instruction-following failures. 

---
# Scaling Laws for Hypernetwork-Based Knowledge Injection in Large Language Models 

**Authors**: Nischay Dhankhar, Dos Baha, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2607.19604)  

**Abstract**: Injecting factual knowledge into large language models (LLMs) reliably and at scale remains an open challenge. Hypernetworks provide a promising solution to large-scale knowledge injection. Although hypernetworks are typically applied for test-time adaptation, we explore their use in train-time knowledge injection, where, given a large corpus of facts, we train a hypernetwork to generate a fixed LoRA adapter that, when inserted into the target model, enable the model to answer questions about those facts. In this work, we investigate whether hypernetworks can be used to perform train-time knowledge injection and how this ability varies with scale. The scaling behavior of hypernetworks remains largely unstudied. Our design decouples the hypernetwork's injection capacity from the target model's general capability, enabling, for the first time, a rigorous study of scaling laws for hypernetwork architectures. We characterize how loss, reasoning accuracy, and out-of-distribution (OOD) generalization vary with hypernetwork depth, width, and target network size. We construct a large-scale dataset, called MegaWikiQA, containing tens of millions of multi-hop question-answer examples across 39 domains constructed from examples in Wikidata5M. Our results reveal: (i) hypernetwork-based injection exhibits broadly predictive power law scaling along all architecture axes; and (ii) hypernetworks are capable of reliable OOD generalization at increasing scales, suggesting that hypernetwork provides a promising alternative to other train-time adaptation methods such as LoRA finetuning and full fine-tuning, exhibiting steeper scaling exponents in all OOD evaluations. Together, these results establish hypernetworks as a principled and scalable substrate for train-time adaptation, and provide the first empirically grounded scaling laws to guide hypernetworks for factual reasoning in large language models. 

---
# On the Computational Complexity of Structural Generalization 

**Authors**: Zichao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2607.19573)  

**Abstract**: Structural generalization has been measured repeatedly by several benchmarks, yet it has never been formally defined. We give a definition that translates the two premises (compositional structure and unbounded generalization) into mathematical language. The definition itself is neutral: a compiler that hard-codes the rules satisfies it just as well. But structural generalization becomes a scientific question only insofar as the capacity can autonomously emerge from finite data. This question pits the computational lower bound $\mathrm{NC}^1$ against the learnable ceiling $\mathrm{TC}^0$ of pure Transformers. Under a Montagovian instantiation, each compositional rule splits into two projections: a syntactic face ($F_\gamma$) and a semantic face ($G_\gamma$). Tree evaluation on the $G_\gamma$ side is an instantiation of BFVP, which is $\mathrm{NC}^1$-complete (Buss, 1987). A pure Transformer must learn both faces at once, but Kraus et al. (2026) prove that its learnable class $\subseteq \mathrm{TC}^0$. Under the standard assumption $\mathrm{TC}^0 \neq \mathrm{NC}^1$, a pure Transformer cannot learn structural generalization. Neuro-symbolic systems achieve the best benchmark scores precisely because they inject $G_\gamma$, sidestepping the genuinely hard half. Benchmark scores cannot distinguish "learned" from "given." This is what this paper sets out to make clear. 

---
# When Reasoning Narrows the Move: Diversity Collapse in LLM Game Play 

**Authors**: Junyi Sha, Renfei Tan, David Simchi-Levi  

**Link**: [PDF](https://arxiv.org/pdf/2607.19523)  

**Abstract**: Supervised fine-tuning (SFT) is widely used to adapt large language models to downstream tasks, but its effect on behavioral diversity in sequential decision-making remains under-explored. We study this question in a controlled suite of deterministic board games based on tic-tac-toe variants, where optimal actions are exactly computable and diversity can be measured directly. Across state-level evaluation, arena gameplay, and training trajectories, we find that reasoning-mode generation frequently suppresses action diversity without uniformly improving action accuracy. Furthermore, standard SFT improves accuracy but often induces premature diversity collapse, which exceeds what is minimally required by the accuracy-diversity tradeoff. We then show that action augmentation, which trains on all optimal actions per state rather than a single demonstrated action, would partially mitigates this effect. Our results identify narrow-support imitation as a source of policy collapse in LLM decision-making and suggest that preserving action support during SFT is important for maintaining exploratory behavior. 

---
# Stateful Guardrails for Multi-Turn LLM Systems: A Conversational Risk Accumulation Framework 

**Authors**: Sanjay Mishra, Divya Chukkapalli, Ganesh R. Naik  

**Link**: [PDF](https://arxiv.org/pdf/2607.19361)  

**Abstract**: Most safety guardrails for large language models (LLMs) evaluate each prompt-response pair in isolation, which misses failures that arise only over a dialogue as benign turns compose into harm. We term this Conversational Risk Accumulation (CRA): gradual intent drift, fragmented assembly of prohibited instructions, and sensitivity build-up from repeated disclosures. We propose a session-layer CRA Framework that tracks three trajectory signals: semantic drift from a session anchor, a sensitivity-weighted information accumulation graph over extracted entities, and a compliance-gradient signal capturing increasing willingness to comply. For scoring, we provide (i) an unsupervised convex fusion for attribution and ablations, and (ii) CRA-Net DA, a compact learned trajectory model trained with family-adversarial objectives to reduce length and topic-coverage confounds. To benchmark CRA, we release CRA-Bench v0.1 (1,200 eight-turn sessions across three threat families with topic-matched benign twins), CRA-Bench v0.2 (LLM-paraphrased variants to reduce template artifacts), and an extended 5-family set (2,000 sessions adding persona priming and context stuffing). We introduce a trajectory-native evaluation protocol with session-level splits, mixed-set threshold calibration, Trajectory AUROC, turns-to-detection, calibrated false-positive metrics, bootstrap confidence intervals, leave-one-family-out diagnostic stress tests, and synthetic-to-human transfer checks. Claims focus on within-distribution session scoring on CRA-Bench and human-transfer subsets. 

---
# Train the Model, Not the Reader: Decodability Supervision for Verifiable Activation Explanations 

**Authors**: Hiskias Dingeto  

**Link**: [PDF](https://arxiv.org/pdf/2607.20379)  

**Abstract**: Natural-language autoencoders score explanations of hidden activations by reconstruction: an explanation is deemed faithful if the activation can be regenerated from it. The test is structurally insensitive to individual false claims: if flipping a claim does not change the reconstruction, the claim is never penalized. We show the test is passed in two ways, neither faithful. On a released Qwen-2.5-7B verbalizer, explanations reconstruct well above chance while ~2% of specific claims are reconstruction-dependent, so the score tracks gist, not specific facts. Under exact synthetic ground truth, the standard recipe develops co-adapted private codes (false wording the reconstruction depends on) in 5/5 runs, and fixes that leave the target model unchanged do not help. We contribute two audit protocols, the grounded-vs-true cross and the evaluator swap, and RECAP (Readable Encodings via Co-trained Auxiliary Predictors): linear heads trained alongside the target model to keep designated content decodable. On RECAP-trained sandbox models, fresh verbalizers state the designated content truly and the codes vanish, at a +0.001-nat cost. This replicates on a pretrained Pythia-160M: the content becomes reliably probe-decodable, though a fresh verbalizer conveys it only in part (truth 0.44-0.46 vs a near-zero control). For interpretability, high reconstruction does not certify individual claims. For AI safety, RECAP makes designated internal content independently checkable against probes rather than asserted by prose a model can game: an independent probe scores the verbalizer's true claims above its false ones (AUC 0.96, vs 0.82 without RECAP). Against an adversary that edits an explanation to maximize the reconstruction score while lying (suppressing ~87% of its lie penalty), the RECAP probe still flags the lies (AUC 0.95) while the control probe collapses to chance (0.51). 

---
# Test-Time Training for Modality Order Consistency in Vision-Language Models 

**Authors**: Aditi Gupta, Yossi Gandelsman  

**Link**: [PDF](https://arxiv.org/pdf/2607.20351)  

**Abstract**: We find that vision-language models are sensitive to a specific semantically irrelevant change: the order in which the image and question are presented. Across three models and three benchmarks, image first prompting consistently outperforms question-first prompting, revealing a repeatable modality order failure. We use this gap to design an order-consistent test-time training method. Our method substantially closes the modality-order gap across all evaluated settings. Surprisingly, it also yields consistent improvements in the stronger image-first branch over the baseline, hence bootstrapping both orderings toward mutual consistency. Activation patching localizes the ordering failure to a narrow mid-network region where representations diverge sharply between prompt orders. We find that the test-time training method repairs this misalignment across layers. Together, our results identify modality-order sensitivity as a circuit-level failure in VLMs and demonstrate that simple, asymmetric test-time adaptation can effectively mitigate it and even improve performance over the baseline. 

---
# The Blessing of Dimensionality: How Near-Orthogonality in High-Dimensional Spaces Explains Temporal Portability 

**Authors**: Abigail Woodring, Adrian Chan, Rana Muhammad Shahroz Khan, Sukwon Yun, Chau-Wai Wong, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.20301)  

**Abstract**: Fine-tuning has been widely used to adapt large language models (LLMs) for domain-specific tasks. Parameter efficient fine-tuning (PEFT) methods such as low-rank adaptation (LoRA) are frequently used to reduce computational costs. PortLLM is a training-free and data-free scheme used to adapt LLMs after continual pretraining. Although the initial PortLLM results show that LoRA patches exhibit short-term temporal portability, the long-term performance of PortLLM across several updates of continual pretraining remains underexplored. Furthermore, the intriguing effectiveness of PortLLM is not well understood from a theoretical standpoint. We address these two open questions by (1) performing an extensive empirical study of the long-term temporal portability of PortLLM patches across 10 continual pretraining steps using base models Mistral, Gemma, and Qwen; and (2) offering two theoretical analyses to explain our observation that the simple PortLLM method achieves competitive performance. We find empirically that the portability persists across longer time duration, indicating that repeated fine-tuning is not required when the base model is periodically updated. We find theoretically that near-orthogonality of high-dimensional vectors is a key justification for temporal portability. Our analyses also demonstrate a geometric perspective of the loss landscape in facilitating the theoretical comparison of different adaptation options. 

---
# Self-supervision drives representational convergence in medical foundation models more than clinical supervision 

**Authors**: Soroosh Tayebi Arasteh, Sebastian Ziegelmayer, Mahshad Lotfinia, Lisa Adams, Sven Nebelung, Jakob Nikolas Kather, Daniel Truhn  

**Link**: [PDF](https://arxiv.org/pdf/2607.20274)  

**Abstract**: Medical image encoders from different groups are increasingly treated as interchangeable, on the assumption that scale and clinical supervision concentrate their representations onto a shared structure. Whether this convergence is real, what produces it, and whether it is clinically usable are untested, and the similarity measures behind such claims are fragile. We present a controlled dissection across 18 image and 7 text encoders, all open-weight and run locally, spanning 7M to 27B parameters and five imaging modalities, including 650,982 chest radiographs from six datasets. To isolate cause, we train encoders that vary only the objective under fixed data, architecture, and scale, and reproduce the effect in a synthetic model. Convergence is modest but above a random floor, driven by the self-supervised objective, not clinical supervision: matched self-supervised encoders aligned most (40.4% on chest radiography), with label-supervised (21.1%) and image-text (3.3%) far lower, and did not grow with size (Spearman 0.302, p=0.223) or capability. It is within-modality, does not reach clinical language, and does not reproduce how radiologists judge case similarity. Yet a linear classifier transfers across encoders and to five held-out hospitals, retaining about 85% of within-encoder performance. Convergence in medical imaging is therefore set by the pretraining objective, not inherited from scale or clinical supervision. Interoperability is accordingly something to design for through that objective, and to validate where the shared geometry is weakest, across patient subgroups and against clinical judgment. 

---
# PoTRE: Test-Time Reasoning inspired by Cognitive Heterogeneity 

**Authors**: Anmol Kankariya, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2607.20268)  

**Abstract**: While Large Language Models (LLMs) excel at many tasks, they frequently struggle with complex reasoning that requires long-horizon planning and iterative error correction. Furthermore, standard single-stream prompting proves brittle when models encounter novel abstractions or rigorous domain constraints. We introduce PoTRE (Poly-Topological Reasoning Ensembles), a heterogeneous framework that decouples inference into four agents: (1) Adversarial Refinement Agent, (2) Hierarchical strategic Planning Agent, (3) Spectrum Search Agent, and (4) Direct Chain Agent. A final Task-Adaptive Aggregation Layer dynamically reconciles these perspectives -- via final candidate selection, semantic synthesis, or neuro-symbolic verification -- to produce a robust global solution. We evaluate PoTRE on three frontier benchmarks: ARC-AGI-2, Humanity's Last Exam (HLE), and PRBench Finance. PoTRE achieves state-of-the-art accuracy of 49.92% on HLE, surpassing the previous best official score. We demonstrate that this architectural heterogeneity achieves improved reasoning performance using similar or fewer inference tokens compared to heavily scaled homogeneous baselines. 

---
# ENTRAP-VL: A Taxonomic Probe for Dual Contextual Entrainment in Vision-Language Models 

**Authors**: Karan Goyal, Afreen Hossain, Debojyoti Das, Vishal Bhutani  

**Link**: [PDF](https://arxiv.org/pdf/2607.20092)  

**Abstract**: Contextual entrainment is the tendency of a model to let auxiliary context in its input pull its output, independently of whether that context is relevant, true, or even meaningful. Recently, it has been identified and given a mechanistic account in unimodal language models. Whether and how it manifests in vision-language models (VLMs) is, by contrast, largely unexamined, and the field lacks a purpose-built instrument with which to investigate it. We take the position that studying contextual entrainment in VLMs requires more than porting an existing text-only benchmark to the multimodal setting: it requires a taxonomically structured, dual-modality instrument whose conditions are constructed around the item at hand (the depicted image in the textual stream, the textual query in the visual stream). We argue that the move to VLMs is substantive rather than incremental. It makes entrainment a dual phenomenon, drivable independently by textual and by visual context, and it opens a veracity distinction (context that is false of the depicted scene yet possible in the world) that has no counterpart in the unimodal, world-knowledge-only formulation of prior work. To make this position concrete and actionable, we introduce ENTRAP-VL (ENTRainment Assessment Probe for Vision and Language), a manually curated dataset of 1,500 items across eight categories, organized by a taxonomy that spans two axes, i.e., the association of context with the item and its relationship to truth, and split into a textual-entrainment stream (eight context conditions) and a visual-entrainment stream (three context conditions). We do not claim to measure entrainment in any particular model; we provide the instrument, the taxonomy that motivates it, and the evaluation protocols it enables, so that the community can investigate the phenomenon rigorously. We will release the dataset and its documentation publicly. 

---
# Reading and Steering Representations of Materials-Science Mechanisms in an Open-Weight Language Model 

**Authors**: Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2607.20058)  

**Abstract**: Large language models can answer scientific questions, yet a correct output does not reveal whether the model represents or uses the governing physics. Here we show that materials science mechanism information in the open-weight google/gemma-4-E4B-it model has three experimentally separable forms: concepts are readable in individual hidden states, constitutive orientation is carried by controlled transformations between states, and selected internal representations causally control engineering answers. We combine matched direct and Jacobian vocabulary readouts, option-free state geometry, a 60-law counterfactual benchmark and causal interventions. In 50 held-out materials descriptions, three independently fitted Jacobian lenses reproduced concept ranks, and target-free word sets from both readouts enabled blinded identification of 9 of 10 mechanism families. A separate 72-prompt benchmark produced mechanism-specific hidden-state neighborhoods, but an exact graph audit showed that this apparent physical organization was equally explained by numerical comparison. We therefore compared otherwise identical prompts in which only the direction of the physical input was reversed, asking whether the resulting hidden-state movement followed the supplied constitutive law. These state transformations ordered direct, physically neutral and inverse laws across 60 frozen relations and correctly oriented 39 of 40 directional laws, whereas lexical controls were near chance. Bidirectional interventions shifted answer probabilities toward or away from the physically appropriate outcome across all 12 matched cases, while counterfactual state patches transferred opposing decision signals across mechanisms and answer formats. Physical relationships were therefore more visible in controlled state changes than in absolute states alone. 

---
# JANUS: Foreseeing Latent Risk for Long-Horizon Agent Safety 

**Authors**: Yuan Xiong, Linji Hao, Shizhu He, Yequan Wang, Lijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.19913)  

**Abstract**: Agent safety is moving from content moderation toward preventing operational failures before tool-using agents act. We propose Janus, a foresight-oriented framework for long-horizon agent safety that trains guards to anticipate delayed risks from partial trajectories. Janus synthesizes diverse agent trajectories via multi-agent simulation and learns a shared policy with two coupled tasks: an anticipation task that forecasts safety-relevant futures and an adjudication task that decides safety from both the observed prefix and anticipated future. The two tasks are jointly optimized with CoAA-RL, which rewards forecasts by their utility for downstream safety judgment. The resulting guard model, Vanguard, blocks unsafe actions before execution. Across four agent-safety benchmarks, Vanguard improves average protection by 15.9 percentage points over baseline guards while increasing benign task completion by 5.1 percentage points. 

---
# DocOps: A Verifiable Benchmark for Autonomous Agents in Complex Document Operations 

**Authors**: Jiazhen Jiang, Boxi Cao, Lingyong Yan, Yaojie Lu, Hongyu Lin, Shuaiqiang Wang, Dawei Yin, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.19865)  

**Abstract**: As autonomous agents rapidly evolve, their ability to reliably manipulate ubiquitous digital documents has become critical for enabling general-purpose AI assistants and automating complex workspace workflows. In this paper, we introduce DocOps, a deterministically verifiable evaluation framework underpinned by a hierarchical taxonomy that deconstructs document operations inspired by real-world practices into atomic dimensions and escalating workflow complexities. Based on DocOps, we systematically evaluate representative closed- and open-source models across various agentic harnesses, revealing that even the most advanced frontier configurations still exhibit profound limitations when handling highly coupled, long-range tasks. Furthermore, a fine-grained analysis of existing agents' manipulation behaviors uncovers 3 key failure modes: long-term state tracking collapse, shallow semantic verification, and destructive editing of structural metadata. Ultimately, our work exposes the capability boundaries of agents in maintaining global document consistency, shedding light on the future design of robust, non-destructive agents for complex digital ecosystems. 

---
# Auto-Fill: Learning to Predict Missing Values Accurately with Specialist Language Models 

**Authors**: Yurong Liu, Yeye He, Haoyu Dong, Junjie Xing, Shi Han, Dongmei Zhang, Surajit Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2607.19847)  

**Abstract**: Predicting missing cell values in tabular data is a fundamental problem in data cleaning. While state-of-the-art reasoning models show great promise in predicting missing values in tables, by reasoning holistically across rows and columns, they are costly to deploy at scale and tend to be overconfident, often generating hallucinated or false-positive predictions.
In this paper, we observe that achieving high-precision missing-value prediction in tables requires a distinct combination of three capabilities: (1) world knowledge, (2) text-based reasoning, and (3) code-based reasoning. We systematically explore design choices for combining these capabilities, and propose an Auto-Fill approach that post-trains three specialist small language models (SLMs), each optimized for one capability. We develop a calibrated ensemble mechanism that either dynamically selects the most confident specialist or abstains, ensuring high accuracy.
Extensive experiments on 11 benchmarks with 2200 real tables drawn from diverse domains show that Auto-Fill achieves superior accuracy compared to state-of-the-art reasoning models (e.g., o3-pro, Gemini 3 Pro, and DeepSeek R1), while operating at a fraction (less than 1%) of the cost of these frontier models. Our results highlight the effectiveness of specialization and calibrated abstention in the important domain of tabular data. Auto-Fill is publicly available at this https URL. 

---
# Rewarding Better Thinking for LLM Preference Alignment 

**Authors**: Xubo Liu, Wenya Guo, Ruxue Yan, Xinying Qian, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19824)  

**Abstract**: LLM preference alignment aims to optimize models toward human preferences across diverse user instructions. Reinforcement learning has become a major post-training approach for this goal, but existing proxy rewards are often outcome-level, mainly evaluating the final response while providing limited guidance for the reasoning trajectory. This can make credit assignment coarse when multiple responses receive similar final scores, leaving trajectory-level preferences under-specified. To address this limitation, we propose Thinking Checklist Reward (TCR), a process-oriented reward for RL-based preference alignment. TCR converts preference pairs into sample-specific thinking checklists and uses them to evaluate whether the generated reasoning trace addresses the preference-implied considerations. To reduce overlap with outcome-level supervision, TCR further introduces an exponential moving average (EMA) residual formulation to isolate a complementary thinking surplus beyond what is predictable from the outcome reward. Experiments on five models from three model families show that TCR consistently improves alignment performance across diverse benchmarks, with ablations further validating the importance of EMA-based residual formulation and sample-specific checklist supervision. 

---
# Twin Agent: Context Residual Compression for Privilege Separated Agents 

**Authors**: Zhanhao Hu, Dennis Jacob, Xiao Huang, Zhaorun Chen, Bo Li, David Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2607.19595)  

**Abstract**: Large language model (LLM) agents are vulnerable to security risks, such as prompt injection attacks from untrusted context that manipulate downstream reasoning and tool use. Existing secure-by-design approaches mitigate this risk by separating untrusted observations from privileged execution and careful control of information flow, but often degrade utility and require extensive task-specific engineering. We thus propose Twin Agent, a general privilege separation design pattern inspired by residual coding in the agent context. Twin Agent consists of two nearly symmetric agents: an Explore Agent that inspects untrusted information and a Safe Agent that executes privileged actions. The Explore Agent is conditioned on the Safe Agent's current context and communicates only compact hints to the Safe Agent about the next action to take. This design reduces the information needed to preserve task utility and thus achieves a better security--utility tradeoff, which we empirically verify by measuring how utility and attack success change as the length of hints varies. We evaluate Twin Agent on long-horizon software engineering tasks with SWE-bench Lite and on heterogeneous multi-tool interaction tasks with AgentDojo and DecodingTrust-Agent. Across both benchmarks, Twin Agent preserves high task utility while preventing prompt injection attacks, outperforming both undefended agents and privilege separation baselines. 

---
# Knowledge-Centric Self-Improvement 

**Authors**: Xuefei Julie Wang, Lauren Hyoseo Yoon, Chengrui Qu, Amanda Zichang Wang, Atharva Sehgal, Eric Mazumdar, Yisong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2607.19592)  

**Abstract**: Self-improving AI systems typically treat the agent as the object that improves, by optimizing prompts, workflows, harnesses, or even the agent's own code. This agent-centric view can make improvements expensive to maintain and difficult to transfer, because gains become tied to a particular agent design, task distribution, or adaptation run. We study a complementary paradigm: knowledge-centric self-improvement, in which agents remain generic and disposable while the persistent object is a curated knowledge base that agents can leverage for future tasks. We conduct controlled case studies to operationalize this idea via a simple protocol. Agents attempt one task, then contribute evidence-grounded insights to a shared knowledge base via task-level and cross-task forums, followed by knowledge distillation. Because self-improvement is contained in the knowledge rather than the agent, improvement can be more inspectable, transferable, and portable. Across abstract reasoning, coding, and terminal benchmarks, this protocol improves solve rates while reducing dollar cost relative to agent-centric baselines. The resulting distilled knowledge also transfers to held-out tasks and across LLM families, indicating that the improvement is not merely an LLM- or run-specific behavior. These results support a new view of self-improving agentic systems: progress can be driven primarily by the curated persistent knowledge. Code is available at this https URL. 

---
# BaseRT: Advancing Best-in-Class LLM Inference with Apple M5 Neural Accelerators 

**Authors**: Fabian Waschkowski, Prabod Rathnayaka, Lukas Wesemann  

**Link**: [PDF](https://arxiv.org/pdf/2607.19438)  

**Abstract**: Apple's M5 generation introduces a redesigned GPU architecture in which every core carries a dedicated Neural Accelerator: on-die matrix units exposed through the Metal~4 tensor API. We show that BaseRT, our native Metal inference runtime for large language models on Apple Silicon, exploits these units to push inference throughput on Apple hardware substantially beyond both this http URL and MLX. Building on BaseRT's framework-free design, we add a family of hand-written Metal~4 tensor-core kernels (including dense and mixture-of-experts GEMM and flash-attention prefill kernels) that route the compute-bound matrix multiplications of inference through the M5 Neural Accelerators while leaving the memory-bound decode path on our existing specialised kernels. On an Apple M5 Pro, across fifteen model configurations spanning the Qwen3, Qwen3.5/3.6, Llama~3.2, and Gemma~4 families from sub-1B to 35B parameters, BaseRT delivers up to $6.4\times$ higher prompt-processing throughput than this http URL and $3.9\times$ higher than MLX, with the largest margins on the mixture-of-experts models where matrix multiplication dominates, while maintaining its lead on decode of up to $1.75\times$ over this http URL and $1.33\times$ over MLX. These results establish a new performance ceiling for on-device LLM inference and show that the M5's tensor cores are the decisive lever for prompt processing on Apple Silicon. BaseRT is publicly available at this https URL. 

---
# JailMeter: An Evidence-Based Evaluation Framework for Jailbreak Attacks on Large Language Models 

**Authors**: Qingjia Huang, Jingyu Zhang, Jianguo Wu, Yakai Li, Weijuan Zhang, Yankai Rong, Junyi Yao, Shengzhi Zhang, Xiaoqi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2607.19424)  

**Abstract**: The assessment of jailbreak attacks against large language models currently suffers from inconsistent evaluation criteria and methods, leading to unreliable estimates of attack success rates. We propose JailMeter, an evidence-based evaluation framework designed to more faithfully measure jailbreak effectiveness. Inspired by the Information Bottleneck theory, JailMeter applies dual-feedback optimization to filter jailbreak noise from model responses while preserving content relevant to the original malicious question. This process produces concise evidence for a rigorous assessment under which an attack is validated only when the response captures the malicious intent and delivers a complete answer, thereby signaling a substantive bypass of model safety alignment. We evaluate JailMeter on JailMeter-Eva, a challenging benchmark containing 330 human-labeled, non-rejected jailbreak instances. JailMeter achieves an accuracy of 97.27%, substantially outperforming existing evaluation methods. To support large-scale evaluation, we further distill JailMeter into a small language model, JailMeter\textsubscript{SLM}, which maintains comparable reliability with significantly reduced computational costs. Code and dataset are available at this https URL. 

---
# Auditing Retrieval-Augmented LLM Hypotheses for Longitudinal Cell Painting Morphology 

**Authors**: Gilchan Park, Guang Zhao, Byung-Jun Yoon, Shinjae Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2607.19415)  

**Abstract**: High-content morphological profiling (Cell Painting) yields sensitive, high-dimensional signatures of cellular state, but translating longitudinal morphology trajectories into interpretable biology remains difficult, especially for weak, chronic perturbations such as low-dose-rate ionizing radiation. Large language models (LLMs) can synthesize heterogeneous evidence into biological narratives, yet their scientific use requires quantitative auditing. We present an evaluation-first, retrieval-augmented interpretation framework for longitudinal Cell Painting morphology, applied to a 9-week RPE-1 time course across five dose rates (0.003--6.0 mGy/hr). Week-matched treated-control morphology deltas are combined with retrieved perturbation neighbors, pathway context, and literature evidence through stable evidence identifiers, enabling an LLM to generate structured, evidence-linked hypotheses that are hierarchically summarized while preserving provenance. We introduce two quantitative auditing tests: V1 citation validity, which verifies that cited evidence identifiers exist in the prompt, and V2 proxy-based morphology compatibility, which evaluates consistency between predicted biological processes and the most altered morphology features. In our experiments, V1 detected no invalid evidence references, while V2 showed meaningful morphology compatibility that increased with perturbation strength and was positively associated with an independent morphology drift summary. The framework produces auditable, falsifiable biological hypotheses, including an adaptive phenotype involving metabolic reprogramming and proteostatic stress at lower dose rates (0.003--0.3 mGy/hr). Current limitations include proxy-based evaluation and the lack of ground-truth mechanism labels. 

---
# HyGRL: Adaptive Hybrid Graph Reasoning for Multi-Entity Questions 

**Authors**: Junyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19398)  

**Abstract**: Multi-entity compositional questions pose significant challenges to existing retrieval-augmented language models. Conventional methods fall into a dilemma: standard RAG lacks dynamic reasoning, traditional Graph-RAG is limited by structural sparsity, and LLM-constructed Graph-RAG incurs prohibitive costs. We propose \textbf{\fwa}, a unified framework that embeds unstructured text into structured knowledge graphs, creating a heterogeneous network for flexible evidence retrieval. Reasoning is formulated as adaptive structure induction, learned via a robust two-stage process: (1) imitation learning distills heuristic expert signals, and (2) reinforcement learning refines the policy using LLM-driven preference rewards. Experiments demonstrate that {\fwa} effectively merges textual richness with structural knowledge, outperforming SOTA baselines in answer accuracy and reasoning fidelity while maintaining extremely low token costs and near real-time inference((code available at this https URL) . 

---
# Decodable but Not Detectable: A Leakage Fingerprint for Near-OOD Benchmarks 

**Authors**: Vishnu Bindu Balachandran  

**Link**: [PDF](https://arxiv.org/pdf/2607.19393)  

**Abstract**: While auditing a perturbation-based OOD detector on a document benchmark, we recorded an AUROC of 0.326 -- well below the 0.5 chance level. The cause is a benchmark leak: the designated "OOD" class is one the model was trained on, so its examples sit inside the in-distribution fit set and the detector is penalized for correctly ranking them as familiar. Deleting the class and retraining 35 models across two domains raises the score to 0.911. We distill the contamination into a leak fingerprint -- near-perfect supervised decodability (AUROC approximately 1) coupled with unsupervised detection collapsed below 0.65 -- and validate it on a controlled battery of 52 settings (20 leaked, 32 clean) across ResNet-50 and ViT-B/16 on CIFAR-10/100, achieving sensitivity 18/20 and specificity 31/32 in embedding space; the matched fit-set-exclusion controls are perfect at 20/20. An in-the-wild audit of 24 standard near/far OOD benchmark pairs fires on exactly one (the intrinsically hard CIFAR-100 vs CIFAR-10 pair) and on no far-OOD pair, confirming specificity and that standard cross-dataset construction is clean. Under the corrected protocol, perturbation signals are decodable but not detectable: a supervised reader recovers the OOD signal (AUROC 0.87-1.00) while no unsupervised detector does, and the perturbation method does not improve on plain Mahalanobis distance. We provide a theoretical account of why and, for transparency, retract an earlier circular correlation. The contributions are a corrected protocol and a validated leak diagnostic, not a new OOD method. 

---
# Building Fast, Evaluating Slow: Pipeline Choices Dominate Autointerpretability Score Variance 

**Authors**: Sinie van der Ben, Neele Roch, Anna Hedström, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2607.19386)  

**Abstract**: Cross-paper comparison of sparse autoencoder (SAE) interpretability often relies on autointerpretability scores. In this evaluation pipeline, a language model (LM) explains each feature, and another LM scores the explanation. For these comparisons to be meaningful, scores must reflect stable properties of the features rather than confounding aspects of the evaluation pipeline. Through systematic experiments across four metrics (simulation, detection, fuzzing, purity), two models (Pythia-160M, Apertus-8B), and four axes of methodological variation, we show that this assumption does not hold. Specifically, we find that R1) methodological variance collectively exceeds architectural variance across all metrics and tested models; R2) each metric exhibits a distinct instability profile, with detection being the most stable and fuzzing unreliable across all conditions; R3) top-k feature rankings do not stay consistent across corpus and draw conditions, masking per-feature instability behind stable mean scores; a failure that cannot be detected by monitoring explanation similarity alone. These findings suggest that cross-paper comparisons based on autointerpretability scores may reflect pipeline differences rather than architectural differences, with implications for the ongoing debate on SAE utility. More broadly, unreliable evaluation slows progress in interpretability research at a time when reliable tools for understanding AI systems are needed. To support evaluation, we contribute a variance decomposition approach, a Stability Check, and a Minimum Reporting Checklist. 

---
# Mitigating Scaffolding Collapse in Socratic Tutors via Representation Alignment 

**Authors**: Jing Shao, Qifeng Wu, Hanyu Zhang, Sixia Sun, Jun Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19371)  

**Abstract**: Large language model (LLM)-based Socratic tutors increasingly guide students through multi-turn questioning, but they can suffer from scaffolding collapse: under sustained student pressure, a tutor gradually abandons guided inquiry and reveals solutions directly. Prior defenses primarily constrain observable responses through prompting, preference optimization, or filtering, leaving the internal representation drift that precedes trajectory-level collapse largely unaddressed. We propose Scaffold-Preserving Representation Alignment, a two-stage framework that first warms up a Socratic tutor with supervised fine-tuning, then combines trajectory-weighted direct preference optimization with a margin-preserving representation loss anchored to frozen reference states. Our method is designed to maintain separation between scaffold-preserving and collapse-inducing hidden states across dialogue turns. We evaluate our method across five STEM disciplines and five red-teaming attack strategies. On Qwen3-8B, our method lowers Collapse Rate to 32%, delays average collapse onset beyond nine turns, and keeps over-refusal low, suggesting that representation-level alignment can improve the robustness of long-horizon Socratic tutoring under our red-teaming protocol. 

---
# Spectral-LSH: Sub-Quadratic Prompt Compression via Krylov-Projected Locality-Sensitive Hashing 

**Authors**: Ali Mahdavi, Azaseh Zamanifar, Amirfarhad Farhadi, Omid Kashefi  

**Link**: [PDF](https://arxiv.org/pdf/2607.19368)  

**Abstract**: Long-prompt inference remains expensive because prefill attention scales quadratically with sequence length. We propose Spectral-LSH, a training-free prompt compression method that operates before the prompt enters the language model. Spectral-LSH approximates the dominant components of an implicit attention-kernel operator using a Krylov subspace method together with random features, avoiding explicit $O(N^2)$ attention-kernel materialization. It then applies SimHash in the resulting attention eigenspace to group similar tokens and aggregate them into macro-tokens with causal positional assignments.
We evaluate Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, and Qwen2.5-14B-Instruct on C4. Our experiments reveal a compression-ratio phase transition. Below $\rho = 4 \times$, local token redundancy is low enough that lightweight chunking typically provides the best latency--quality trade-off. Above $\rho = 8 \times$, the spectral path preserves quality that chunking loses. At $\rho = 16 \times$, Qwen2.5-7B (adaptive) reduces the PPL ratio from 353.409 to 196.963, while Qwen2.5-14B (adaptive) reduces it from 9.533 to 3.427.
On a small long-context structured stress test containing JSON-like, code-like, and table-like inputs, local LSH also improves every metric over chunking at $8 \times$. The adaptive backend captures both regimes by using the chunk path at low compression and spectral clustering at high compression, although chunking remains the fastest backend in total latency. 

---
# Statistically Grounded Sparse-Feature Interventions for Activation-Space Control in Large Language Models 

**Authors**: Oshayer Siddique, J. M Areeb Uzair Alam, Md Jobayer Rahman Rafy, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2607.19364)  

**Abstract**: Activation steering offers a lightweight alternative to fine-tuning for behavioral control of large language models, but SAE-based steering methods often rely on learned steering objectives or single-criterion feature selection. We introduce a transparent SAE-feature steering pipeline that first applies a six-condition reliability filter, then ranks sparse features through an unweighted Borda consensus over three complementary statistics: $F$-test, KSG mutual information, and Cohen's $d$. The resulting steering direction is constructed as a Cohen's-$d$-weighted combination of SAE decoder rows, providing an optimization-free direction motivated by Fisher-LDA under approximate SAE-feature decorrelation. Across three Gemma-family models, four behavioral domains, and 356 layer-strength configurations, the method produces measurable domain-specific shifts while revealing a substantial gap between raw attribute movement and quality-preserving generation. In the strongest configuration, logical-correctness steering reaches a primary-score delta of $+1.16$ in Gemma~2 9B; however, our broader finding is that usable steering is highly localized by model, domain, layer, and strength. These results argue that activation-steering evaluations should report quality-conditioned success alongside raw behavioral shift. Our code and data are available at this https URL. 

---
# AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally 

**Authors**: Shaowen Wang, Yuke Zheng, Tansheng Zhu, Shuang Chen, Shaofan Liu, Suncong Zheng, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.19363)  

**Abstract**: Rotary Position Embedding (RoPE) is widely adopted in Transformers to encode positional information, yet standard implementations enforce a uniform frequency schedule and scaling across all attention heads. Using simplified retrieval tasks and length generalization scenarios, we show -- both empirically and theoretically -- that heads with different functional roles require distinct frequency ranges and attention scaling factors to operate effectively. Ignoring this structure leads to suboptimal utilization of embedding dimensions and degraded performance, particularly under long-context settings. To address these limitations, we propose AdaRoPE, which equips each attention head with learnable rotation frequencies and attention scaling factors. Pretrained LLMs with AdaRoPE consistently outperform existing RoPE variants, including partial RoPE and NoPE baselines. For context extension, we further show that uniform frequency and attention scaling, used in methods such as YaRN, are suboptimal. By applying head-specific scaling, AdaRoPE enables better context extension while better preserving short-context performance in both the extrapolation setting and the long-context continued pretraining setting. These results highlight the importance of optimizing rotary position embedding at the level of individual attention heads. 

---
# Information Discernment in Large Language Models 

**Authors**: Joshua Ashkinaze, Laura Kurek, Alina Faisal, Tongyuan Miao, Mariam Joseph, Ceren Budak, Eric Gilbert  

**Link**: [PDF](https://arxiv.org/pdf/2607.19355)  

**Abstract**: LLMs are increasingly used with external knowledge sources like the internet. Do they weigh information appropriately -- updating more for reliable sources (source discernment) and more when claims bring priors closer to the truth (truth discernment)? We formalize this as information discernment and introduce Learn2Discern (L2D), an experimental framework and benchmark grounded in three normative axioms with interpretable metrics. To establish external validity, a pre-registered, quota-matched user study (n=299) confirms that real LLM users endorse all three axioms and report that violations reduce their trust and usage intent. Across 13 models and nearly 670K trials, we find consistent failures across both dimensions: models perform near chance on source and truth discernment, rely on source popularity twice as much as source reliability, and update roughly equally whether a claim improves or worsens their position relative to the ground truth. Models integrate external knowledge most effectively on datasets where their priors are already the most accurate. Newer and larger models improve truth discernment but not source discernment, a blind spot that model complexity does not address. We identify simple inference-time interventions that improve both forms of discernment. We release our dataset and survey as a testbed for a core alignment property that scales in importance as LLMs replace traditional search. 

---
