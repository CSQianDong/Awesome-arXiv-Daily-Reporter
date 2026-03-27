# Natural-Language Agent Harnesses 

**Authors**: Linyue Pan, Lexiao Zou, Shuo Guo, Jingchen Ni, Hai-Tao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.25723)  

**Abstract**: Agent performance increasingly depends on \emph{harness engineering}, yet harness design is usually buried in controller code and runtime-specific conventions, making it hard to transfer, compare, and study as a scientific object. We ask whether the high-level control logic of an agent harness can instead be externalized as a portable executable artifact. We introduce \textbf{Natural-Language Agent Harnesses} (NLAHs), which express harness behavior in editable natural language, and \textbf{Intelligent Harness Runtime} (IHR), a shared runtime that executes these harnesses through explicit contracts, durable artifacts, and lightweight adapters. Across coding and computer-use benchmarks, we conduct controlled evaluations of operational viability, module ablation, and code-to-text harness migration. 

---
# S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation 

**Authors**: Ligong Han, Hao Wang, Han Gao, Kai Xu, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2603.25702)  

**Abstract**: Block-diffusion language models offer a promising path toward faster-than-autoregressive generation by combining block-wise autoregressive decoding with within-block parallel denoising. However, in the few-step regime needed for practical acceleration, standard confidence-thresholded decoding is often brittle: aggressive thresholds hurt quality, while conservative thresholds require unnecessary denoising steps. Existing approaches that address this issue either require additional training or incur extra test-time compute. We present S2D2, a training-free self-speculative decoding framework for block-diffusion language models. Our key observation is that a block-diffusion model becomes autoregressive when the block size is reduced to one, allowing the same pretrained model to act as both drafter and verifier. S2D2 inserts a speculative verification step into standard block-diffusion decoding and uses lightweight routing policies to decide when verification is worth its cost. This yields a hybrid decoding trajectory in which diffusion proposes tokens in parallel, while the autoregressive mode acts as a local sequence-level critic. Across three mainstream block-diffusion families, S2D2 consistently improves the accuracy-speed tradeoff over strong confidence-thresholding baselines. On SDAR, we observe up to $4.7\times$ speedup over autoregressive decoding, and up to $1.57\times$ over a tuned dynamic decoding baseline while improving accuracy by up to $4.5$ points. On LLaDA2.1-Mini, S2D2 remains complementary to built-in self-correction, including a conservative setting where it is $4.4\times$ faster than the static baseline with slightly higher accuracy. 

---
# Self-Improvement of Large Language Models: A Technical Overview and Future Outlook 

**Authors**: Haoyan Yang, Mario Xerri, Solha Park, Huajian Zhang, Yiyang Feng, Sai Akhil Kogilathota, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.25681)  

**Abstract**: As large language models (LLMs) continue to advance, improving them solely through human supervision is becoming increasingly costly and limited in scalability. As models approach human-level capabilities in certain domains, human feedback may no longer provide sufficiently informative signals for further improvement. At the same time, the growing ability of models to make autonomous decisions and execute complex actions naturally enables abstractions in which components of the model development process can be progressively automated. Together, these challenges and opportunities have driven increasing interest in self-improvement, where models autonomously generate data, evaluate outputs, and iteratively refine their own capabilities. In this paper, we present a system-level perspective on self-improving language models and introduce a unified framework that organizes existing techniques. We conceptualize the self-improvement system as a closed-loop lifecycle, consisting of four tightly coupled processes: data acquisition, data selection, model optimization, and inference refinement, along with an autonomous evaluation layer. Within this framework, the model itself plays a central role in driving each stage: collecting or generating data, selecting informative signals, updating its parameters, and refining outputs, while the autonomous evaluation layer continuously monitors progress and guides the improvement cycle across stages. Following this lifecycle perspective, we systematically review and analyze representative methods for each component from a technical standpoint. We further discuss current limitations and outline our vision for future research toward fully self-improving LLMs. 

---
# Measuring What Matters -- or What's Convenient?: Robustness of LLM-Based Scoring Systems to Construct-Irrelevant Factors 

**Authors**: Cole Walsh, Rodica Ivan  

**Link**: [PDF](https://arxiv.org/pdf/2603.25674)  

**Abstract**: Automated systems have been widely adopted across the educational testing industry for open-response assessment and essay scoring. These systems commonly achieve performance levels comparable to or superior than trained human raters, but have frequently been demonstrated to be vulnerable to the influence of construct-irrelevant factors (i.e., features of responses that are unrelated to the construct assessed) and adversarial conditions. Given the rising usage of large language models in automated scoring systems, there is a renewed focus on ``hallucinations'' and the robustness of these LLM-based automated scoring approaches to construct-irrelevant factors. This study investigates the effects of construct-irrelevant factors on a dual-architecture LLM-based scoring system designed to score short essay-like open-response items in a situational judgment test. It was found that the scoring system was generally robust to padding responses with meaningless text, spelling errors, and writing sophistication. Duplicating large passages of text resulted in lower scores predicted by the system, on average, contradicting results from previous studies of non-LLM-based scoring systems, while off-topic responses were heavily penalized by the scoring system. These results provide encouraging support for the robustness of future LLM-based scoring systems when designed with construct relevance in mind. 

---
# Beyond Via: Analysis and Estimation of the Impact of Large Language Models in Academic Papers 

**Authors**: Mingmeng Geng, Yuhang Dong, Thierry Poibeau  

**Link**: [PDF](https://arxiv.org/pdf/2603.25638)  

**Abstract**: Through an analysis of arXiv papers, we report several shifts in word usage that are likely driven by large language models (LLMs) but have not previously received sufficient attention, such as the increased frequency of "beyond" and "via" in titles and the decreased frequency of "the" and "of" in abstracts. Due to the similarities among different LLMs, experiments show that current classifiers struggle to accurately determine which specific model generated a given text in multi-class classification tasks. Meanwhile, variations across LLMs also result in evolving patterns of word usage in academic papers. By adopting a direct and highly interpretable linear approach and accounting for differences between models and prompts, we quantitatively assess these effects and show that real-world LLM usage is heterogeneous and dynamic. 

---
# PICon: A Multi-Turn Interrogation Framework for Evaluating Persona Agent Consistency 

**Authors**: Minseo Kim, Sujeong Im, Junseong Choi, Junhee Lee, Chaeeun Shim, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2603.25620)  

**Abstract**: Large language model (LLM)-based persona agents are rapidly being adopted as scalable proxies for human participants across diverse domains. Yet there is no systematic method for verifying whether a persona agent's responses remain free of contradictions and factual inaccuracies throughout an interaction. A principle from interrogation methodology offers a lens: no matter how elaborate a fabricated identity, systematic interrogation will expose its contradictions. We apply this principle to propose PICon, an evaluation framework that probes persona agents through logically chained multi-turn questioning. PICon evaluates consistency along three core dimensions: internal consistency (freedom from self-contradiction), external consistency (alignment with real-world facts), and retest consistency (stability under repetition). Evaluating seven groups of persona agents alongside 63 real human participants, we find that even systems previously reported as highly consistent fail to meet the human baseline across all three dimensions, revealing contradictions and evasive responses under chained questioning. This work provides both a conceptual foundation and a practical methodology for evaluating persona agents before trusting them as substitutes for human participants. We provide the source code and an interactive demo at: this https URL 

---
# Humans vs Vision-Language Models: A Unified Measure of Narrative Coherence 

**Authors**: Nikolai Ilinykh, Hyewon Jang, Shalom Lappin, Asad Sayeed, Sharid Loáiciga  

**Link**: [PDF](https://arxiv.org/pdf/2603.25537)  

**Abstract**: We study narrative coherence in visually grounded stories by comparing human-written narratives with those generated by vision-language models (VLMs) on the Visual Writing Prompts corpus. Using a set of metrics that capture different aspects of narrative coherence, including coreference, discourse relation types, topic continuity, character persistence, and multimodal character grounding, we compute a narrative coherence score. We find that VLMs show broadly similar coherence profiles that differ systematically from those of humans. In addition, differences for individual measures are often subtle, but they become clearer when considered jointly. Overall, our results indicate that, despite human-like surface fluency, model narratives exhibit systematic differences from those of humans in how they organise discourse across a visually grounded story. Our code is available at this https URL. 

---
# An Experimental Comparison of the Most Popular Approaches to Fake News Detection 

**Authors**: Pietro Dell'Oglio, Alessandro Bondielli, Francesco Marcelloni, Lucia C. Passaro  

**Link**: [PDF](https://arxiv.org/pdf/2603.25501)  

**Abstract**: In recent years, fake news detection has received increasing attention in public debate and scientific research. Despite advances in detection techniques, the production and spread of false information have become more sophisticated, driven by Large Language Models (LLMs) and the amplification power of social media. We present a critical assessment of 12 representative fake news detection approaches, spanning traditional machine learning, deep learning, transformers, and specialized cross-domain architectures. We evaluate these methods on 10 publicly available datasets differing in genre, source, topic, and labeling rationale. We address text-only English fake news detection as a binary classification task by harmonizing labels into "Real" and "Fake" to ensure a consistent evaluation protocol. We acknowledge that label semantics vary across datasets and that harmonization inevitably removes such semantic nuances. Each dataset is treated as a distinct domain. We conduct in-domain, multi-domain and cross-domain experiments to simulate real-world scenarios involving domain shift and out-of-distribution data. Fine-tuned models perform well in-domain but struggle to generalize. Cross-domain architectures can reduce this gap but are data-hungry, while LLMs offer a promising alternative through zero- and few-shot learning. Given inherent dataset confounds and possible pre-training exposure, results should be interpreted as robustness evaluations within this English, text-only protocol. 

---
# Translation Asymmetry in LLMs as a Data Augmentation Factor: A Case Study for 6 Romansh Language Varieties 

**Authors**: Jannis Vamvas, Ignacio Pérez Prat, Angela Heldstab, Dominic P. Fischer, Sina Ahmadi, Rico Sennrich  

**Link**: [PDF](https://arxiv.org/pdf/2603.25489)  

**Abstract**: Recent strategies for low-resource machine translation rely on LLMs to generate synthetic data from higher-resource languages. We find that this method fails for Romansh, because LLMs tend to confuse its 6 distinct language varieties. Our experiments show that instead, the direction of data augmentation should be aligned with the resource gradient between source and target language. This approach surpasses Gemini 3 Pro in the lowest-resource variety of Romansh by 23 BLEU. A human evaluation confirms that our experiments yield the first model that generates fluent translations in the individual Romansh varieties. 

---
# Navigating the Prompt Space: Improving LLM Classification of Social Science Texts Through Prompt Engineering 

**Authors**: Erkan Gunes, Christoffer Florczak, Tevfik Murat Yildirim  

**Link**: [PDF](https://arxiv.org/pdf/2603.25422)  

**Abstract**: Recent developments in text classification using Large Language Models (LLMs) in the social sciences suggest that costs can be cut significantly, while performance can sometimes rival existing computational methods. However, with a wide variance in performance in current tests, we move to the question of how to maximize performance. In this paper, we focus on prompt context as a possible avenue for increasing accuracy by systematically varying three aspects of prompt engineering: label descriptions, instructional nudges, and few shot examples. Across two different examples, our tests illustrate that a minimal increase in prompt context yields the highest increase in performance, while further increases in context only tend to yield marginal performance increases thereafter. Alarmingly, increasing prompt context sometimes decreases accuracy. Furthermore, our tests suggest substantial heterogeneity across models, tasks, and batch size, underlining the need for individual validation of each LLM coding task rather than reliance on general rules. 

---
# TAPO: Translation Augmented Policy Optimization for Multilingual Mathematical Reasoning 

**Authors**: Xu Huang, Zhejian Lai, Zixian Huang, Jiajun Chen, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25419)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in English mathematical reasoning, yet a significant performance disparity persists in multilingual contexts, largely attributed to deficiencies in language understanding. To bridge this gap, we introduce Translation-Augmented Policy Optimization (TAPO), a novel reinforcement learning framework built upon GRPO. TAPO enforces an explicit alignment strategy where the model leverages English as a pivot and follows an understand-then-reason paradigm. Crucially, we employ a step-level relative advantage mechanism that decouples understanding from reasoning, allowing the integration of translation quality rewards without introducing optimization conflicts. Extensive experiments reveal that TAPO effectively synergizes language understanding with reasoning capabilities and is compatible with various models. It outperforms baseline methods in both multilingual mathematical reasoning and translation tasks, while generalizing well to unseen languages and out-of-domain tasks. 

---
# Large Language Model as Token Compressor and Decompressor 

**Authors**: Wenbing Li, Zikai Song, Jielei Zhang, Tianhao Zhao, Junkai Lin, Yiran Wang, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25340)  

**Abstract**: In this paper, we establish the novel insight that an off-the-shelf LLM can function as an excellent token compressor and decompressor. To demonstrate, we design a self-expressive autoencoding learning framework fine-tunes a pretrained LLM to translate long texts into a compact internal language of discrete, variable-length latent codes, termed Z-tokens, and to reconstruct the original text exactly from them. The resulting representation is content-adaptive: semantically dense segments receive more Z-tokens, while redundant or predictable regions are aggressively compressed, via lightweight LoRA-based adapter heads. Empirically, our method achieves up to 18 times token reduction on Wikipedia, CNN/DailyMail, HotpotQA, and Qulac-style long-query datasets, while preserving reconstruction fidelity and downstream performance. This simple yet effective design supports applications including prompt compression and autoregressive generation directly in the Z-token space, offering a potential pathway toward token-efficient long-context reasoning. 

---
# Adaptive Chunking: Optimizing Chunking-Method Selection for RAG 

**Authors**: Paulo Roberto de Moura Júnior, Jean Lelong, Annabelle Blangero  

**Link**: [PDF](https://arxiv.org/pdf/2603.25333)  

**Abstract**: The effectiveness of Retrieval-Augmented Generation (RAG) is highly dependent on how documents are chunked, that is, segmented into smaller units for indexing and retrieval. Yet, commonly used "one-size-fits-all" approaches often fail to capture the nuanced structure and semantics of diverse texts. Despite its central role, chunking lacks a dedicated evaluation framework, making it difficult to assess and compare strategies independently of downstream performance. We challenge this paradigm by introducing Adaptive Chunking, a framework that selects the most suitable chunking strategy for each document based on a set of five novel intrinsic, document-based metrics: References Completeness (RC), Intrachunk Cohesion (ICC), Document Contextual Coherence (DCC), Block Integrity (BI), and Size Compliance (SC), which directly assess chunking quality across key dimensions. To support this framework, we also introduce two new chunkers, an LLM-regex splitter and a split-then-merge recursive splitter, alongside targeted post-processing techniques. On a diverse corpus spanning legal, technical, and social science domains, our metric-guided adaptive method significantly improves downstream RAG performance. Without changing models or prompts, our framework increases RAG outcomes, raising answers correctness to 72% (from 62-64%) and increasing the number of successfully answered questions by over 30% (65 vs. 49). These results demonstrate that adaptive, document-aware chunking, guided by a complementary suite of intrinsic metrics, offers a practical and effective path to more robust RAG systems. Code available at this https URL. 

---
# Beyond Detection: Rethinking Education in the Age of AI-writing 

**Authors**: Maria Marina, Alexander Panchenko, Vasily Konovalov  

**Link**: [PDF](https://arxiv.org/pdf/2603.25329)  

**Abstract**: As generative AI tools like ChatGPT enter classrooms, workplaces and everyday thinking, writing is at risk of becoming a formality -- outsourced, automated and stripped of its cognitive value. But writing is not just output; it is how we learn to think. This paper explores what is lost when we let machines write for us, drawing on cognitive psychology, educational theory and real classroom practices. We argue that the process of writing -- messy, slow, often frustrating -- is where a human deep learning happens. The paper also explores the current possibilities of AI-text detection, how educators can adapt through smarter pedagogy rather than bans, and why the ability to recognize machine-generated language may become a critical literacy of the 21st century. In a world where writing can be faked, learning can not. 

---
# Separate Before You Compress: The WWHO Tokenization Architecture 

**Authors**: Kusal Darshana  

**Link**: [PDF](https://arxiv.org/pdf/2603.25309)  

**Abstract**: Current Large Language Models (LLMs) mostly use BPE (Byte Pair Encoding) based tokenizers, which are very effective for simple structured Latin scripts such as English. However, standard BPE tokenizers struggle to process complex Abugida scripts due to their structural complexity. The problem is that these tokenizers break complex conjuncts, which are multi-codepoint grapheme clusters, into meaningless sub-character units. This degrades the LLM's reasoning efficiency by forcing it to learn basic orthographic structures at inference time and raises inference costs, resulting in a significant "Token Tax" for the Global South. We propose a new three-layer architecture, the WWHO (Where-What-How Often), and an algorithm named SGPE (Syllable-aware Grapheme Pair Encoding) that separates the linguistic rules of the script from the statistical compression process while enabling seamless multilingual tokenization. Using Sinhala and Devanagari (Hindi/Sanskrit) as highly complex Abugida scripts, we trained WWHO on a cleaned 30-million-sentence dataset and evaluated on a 1,499,950-sentence test set. For Sinhala, SGPE achieves a Token to Word Ratio (TWR) of 1.274 with 4.83 characters per token, representing a 61.7 percent reduction in tokens compared to OpenAI's o200k base. For Hindi, it achieves a TWR of 1.181 (27.0 percent reduction vs o200k). On the mixed-script (Sinhala, Devanagari, and English) dataset, SGPE achieves an overall TWR of 1.240, representing token reductions of 36.7 percent, 39.6 percent, and 60.2 percent relative to o200k base, Llama 4 Scout, and DeepSeek V3, respectively. This effectively extends the usable context window by up to 4.38 times for these Abugida languages while ensuring a Linguistic Zero-Breakage Guarantee, which ensures that no valid syllable is ever split across multiple tokens. 

---
# When Hate Meets Facts: LLMs-in-the-Loop for Check-worthiness Detection in Hate Speech 

**Authors**: Nicolás Benjamín Ocampo, Tommaso Caselli, Davide Ceolin  

**Link**: [PDF](https://arxiv.org/pdf/2603.25269)  

**Abstract**: Hateful content online is often expressed using fact-like, not necessarily correct information, especially in coordinated online harassment campaigns and extremist propaganda. Failing to jointly address hate speech (HS) and misinformation can deepen prejudice, reinforce harmful stereotypes, and expose bystanders to psychological distress, while polluting public debate. Moreover, these messages require more effort from content moderators because they must assess both harmfulness and veracity, i.e., fact-check them. To address this challenge, we release WSF-ARG+, the first dataset which combines hate speech with check-worthiness information. We also introduce a novel LLM-in-the-loop framework to facilitate the annotation of check-worthy claims. We run our framework, testing it with 12 open-weight LLMs of different sizes and architectures. We validate it through extensive human evaluation, and show that our LLM-in-the-loop framework reduces human effort without compromising the annotation quality of the data. Finally, we show that HS messages with check-worthy claims show significantly higher harassment and hate, and that incorporating check-worthiness labels improves LLM-based HS detection up to 0.213 macro-F1 and to 0.154 macro-F1 on average for large models. 

---
# CRAFT: Grounded Multi-Agent Coordination Under Partial Information 

**Authors**: Abhijnan Nath, Hannah VanderHoeven, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2603.25268)  

**Abstract**: We introduce CRAFT, a multi-agent benchmark for evaluating pragmatic communication in large language models under strict partial information. In this setting, multiple agents with complementary but incomplete views must coordinate through natural language to construct a shared 3D structure that no single agent can fully observe. We formalize this problem as a multi-sender pragmatic reasoning task and provide a diagnostic framework that decomposes failures into spatial grounding, belief modeling and pragmatic communication errors, including a taxonomy of behavioral failure profiles in both frontier and open-weight models. Across a diverse set of models, including 8 open-weight and 7 frontier including reasoning models, we find that stronger reasoning ability does not reliably translate to better coordination: smaller open-weight models often match or outperform frontier systems, and improved individual communication does not guarantee successful collaboration. These results suggest that multi-agent coordination remains a fundamentally unsolved challenge for current language models. Our code can be found at this https URL 

---
# MolQuest: A Benchmark for Agentic Evaluation of Abductive Reasoning in Chemical Structure Elucidation 

**Authors**: Taolin Han, Shuang Wu, Jinghang Wang, Yuhao Zhou, Renquan Lv, Bing Zhao, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.25253)  

**Abstract**: Large language models (LLMs) hold considerable potential for advancing scientific discovery, yet systematic assessment of their dynamic reasoning in real-world research remains limited. Current scientific evaluation benchmarks predominantly rely on static, single-turn Question Answering (QA) formats, which are inadequate for measuring model performance in complex scientific tasks that require multi-step iteration and experimental interaction. To address this gap, we introduce MolQuest, a novel agent-based evaluation framework for molecular structure elucidation built upon authentic chemical experimental data. Unlike existing datasets, MolQuest formalizes molecular structure elucidation as a multi-turn interactive task, requiring models to proactively plan experimental steps, integrate heterogeneous spectral sources (e.g., NMR, MS), and iteratively refine structural hypotheses. This framework systematically evaluates LLMs' abductive reasoning and strategic decision-making abilities within a vast and complex chemical space. Empirical results reveal that contemporary frontier models exhibit significant limitations in authentic scientific scenarios: notably, even state-of-the-art (SOTA) models achieve an accuracy of only approximately 50%, while the performance of most other models remains below the 30% threshold. This work provides a reproducible and extensible framework for science-oriented LLM evaluation, our findings highlight the critical gap in current LLMs' strategic scientific reasoning, setting a clear direction for future research toward AI that can actively participate in the scientific process. 

---
# Comparing Natural and Synthetic Structured Data: A Study of the Passive Verb Alternation in French and Italian 

**Authors**: Giuseppe Samo, Paola Merlo  

**Link**: [PDF](https://arxiv.org/pdf/2603.25227)  

**Abstract**: This study compares the impact of natural and synthetic data on training and evaluating large language models (LLMs), using the case of passive verb alternation in French and Italian. We use Blackbird Language Matrices (BLMs), structured datasets designed to probe linguistic knowledge of underlying patterns across sentence sets. We compare structured templates instantiated with natural sentences extracted from Universal Dependencies to structured templates of synthetic sentences. Experiments show that while models achieve ceiling performance when trained and tested on synthetic datasets, they do not reliably generalize to natural sentences. In contrast, models trained on natural data exhibit robust performance across both natural and synthetic test suites, demonstrating their superior ability to capture abstract linguistic patterns. These results corroborate the value of natural data and of structured set ups in linguistic evaluation for probing LLMs' syntactic and semantic knowledge. 

---
# Translation or Recitation? Calibrating Evaluation Scores for Machine Translation of Extremely Low-Resource Languages 

**Authors**: Danlu Chen, Ka Sing He, Jiahe Tian, Chenghao Xiao, Zhaofeng Wu, Taylor Berg-Kirkpatrick, Freda Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.25222)  

**Abstract**: The landscape of extremely low-resource machine translation (MT) is characterized by perplexing variability in reported performance, often making results across different language pairs difficult to contextualize. For researchers focused on specific language groups -- such as ancient languages -- it is nearly impossible to determine if breakthroughs reported in other contexts (e.g., native African or American languages) result from superior methodologies or are merely artifacts of benchmark collection. To address this problem, we introduce the FRED Difficulty Metrics, which include the Fertility Ratio (F), Retrieval Proxy (R), Pre-training Exposure (E), and Corpus Diversity (D) and serve as dataset-intrinsic metrics to contextualize reported scores. These metrics reveal that a significant portion of result variability is explained by train-test overlap and pre-training exposure rather than model capability. Additionally, we identify that some languages -- particularly extinct and non-Latin indigenous languages -- suffer from poor tokenization coverage (high token fertility), highlighting a fundamental limitation of transferring models from high-resource languages that lack a shared vocabulary. By providing these indices alongside performance scores, we enable more transparent evaluation of cross-lingual transfer and provide a more reliable foundation for the XLR MT community. 

---
# SafeMath: Inference-time Safety improves Math Accuracy 

**Authors**: Sagnik Basu, Subhrajit Mitra, Aman Juneja, Somnath Banerjee, Rima Hazra, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2603.25201)  

**Abstract**: Recent research points toward LLMs being manipulated through adversarial and seemingly benign inputs, resulting in harmful, biased, or policy-violating outputs. In this paper, we study an underexplored issue concerning harmful and toxic mathematical word problems. We show that math questions, particularly those framed as natural language narratives, can serve as a subtle medium for propagating biased, unethical, or psychologically harmful content, with heightened risks in educational settings involving children. To support a systematic study of this phenomenon, we introduce ToxicGSM, a dataset of 1.9k arithmetic problems in which harmful or sensitive context is embedded while preserving mathematically well-defined reasoning tasks. Using this dataset, we audit the behaviour of existing LLMs and analyse the trade-offs between safety enforcement and mathematical correctness. We further propose SafeMath -- a safety alignment technique that reduces harmful outputs while maintaining, and in some cases improving, mathematical reasoning performance. Our results highlight the importance of disentangling linguistic harm from math reasoning and demonstrate that effective safety alignment need not come at the cost of accuracy. We release the source code and dataset at this https URL. 

---
# A Decade-Scale Benchmark Evaluating LLMs' Clinical Practice Guidelines Detection and Adherence in Multi-turn Conversations 

**Authors**: Andong Tan, Shuyu Dai, Jinglu Wang, Fengtao Zhou, Yan Lu, Xi Wang, Yingcong Chen, Can Yang, Shujie Liu, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.25196)  

**Abstract**: Clinical practice guidelines (CPGs) play a pivotal role in ensuring evidence-based decision-making and improving patient outcomes. While Large Language Models (LLMs) are increasingly deployed in healthcare scenarios, it is unclear to which extend LLMs could identify and adhere to CPGs during conversations. To address this gap, we introduce CPGBench, an automated framework benchmarking the clinical guideline detection and adherence capabilities of LLMs in multi-turn conversations. We collect 3,418 CPG documents from 9 countries/regions and 2 international organizations published in the last decade spanning across 24 specialties. From these documents, we extract 32,155 clinical recommendations with corresponding publication institute, date, country, specialty, recommendation strength, evidence level, etc. One multi-turn conversation is generated for each recommendation accordingly to evaluate the detection and adherence capabilities of 8 leading LLMs. We find that the 71.1%-89.6% recommendations can be correctly detected, while only 3.6%-29.7% corresponding titles can be correctly referenced, revealing the gap between knowing the guideline contents and where they come from. The adherence rates range from 21.8% to 63.2% in different models, indicating a large gap between knowing the guidelines and being able to apply them. To confirm the validity of our automatic analysis, we further conduct a comprehensive human evaluation involving 56 clinicians from different specialties. To our knowledge, CPGBench is the first benchmark systematically revealing which clinical recommendations LLMs fail to detect or adhere to during conversations. Given that each clinical recommendation may affect a large population and that clinical applications are inherently safety critical, addressing these gaps is crucial for the safe and responsible deployment of LLMs in real world clinical practice. 

---
# A Catalog of Basque Dialectal Resources: Online Collections and Standard-to-Dialectal Adaptations 

**Authors**: Jaione Bengoetxea, Itziar Gonzalez-Dios, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2603.25189)  

**Abstract**: Recent research on dialectal NLP has identified data scarcity as a primary limitation. To address this limitation, this paper presents a catalog of contemporary Basque dialectal data and resources, offering a systematic and comprehensive compilation of the dialectal data currently available in Basque. Two types of data sources have been distinguished: online data originally written in some dialect, and standard-to-dialect adapted data. The former includes all dialectal data that can be found online, such as news and radio sites, informal tweets, as well as online resources such as dictionaries, atlases, grammar rules, or videos. The latter consists of data that has been adapted from the standard variety to dialectal varieties, either manually or automatically. Regarding the manual adaptation, the test split of the XNLI Natural Language Inference dataset was manually adapted into three Basque dialects: Western, Central, and Navarrese-Lapurdian, yielding a high-quality parallel gold standard evaluation dataset. With respect to the automatic dialectal adaptation, the automatically adapted physical commonsense dataset (BasPhyCowest) underwent additional manual evaluation by native speakers to assess its quality and determine whether it could serve as a viable substitute for full manual adaptation (i.e., silver data creation). 

---
# Probing the Lack of Stable Internal Beliefs in LLMs 

**Authors**: Yifan Luo, Kangping Xu, Yanzhen Lu, Yang Yuan, Andrew Chi-Chih Yao  

**Link**: [PDF](https://arxiv.org/pdf/2603.25187)  

**Abstract**: Persona-driven large language models (LLMs) require consistent behavioral tendencies across interactions to simulate human-like personality traits, such as persistence or reliability. However, current LLMs often lack stable internal representations that anchor their responses over extended dialogues. This work explores whether LLMs can maintain "implicit consistency", defined as persistent adherence to an unstated goal in multi-turn interactions. We designed a 20-question-style riddle game paradigm where an LLM is tasked with secretly selecting a target and responding to users' guesses with "yes/no" answers. Through evaluations, we find that LLMs struggle to preserve latent consistency: their implicit "goals" shift across turns unless explicitly provided their selected target in context. These findings highlight critical limitations in the building of persona-driven LLMs and underscore the need for mechanisms that anchor implicit goals over time, which is a key to realistic personality modeling in interactive applications such as dialogue systems. 

---
# Cross-Preference Learning for Sentence-Level and Context-Aware Machine Translation 

**Authors**: Ying Li, Xinglin Lyu, Junhui Li, Jinlong Yang, Hengchao Shang, Min Zhang, Shimin Tao, Daimeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.25183)  

**Abstract**: Context-aware machine translation (MT) leverages document-level information, yet it does not consistently outperform sentence-level MT, as contextual signals are unevenly beneficial across sentences. Existing training objectives do not explicitly model this variability, limiting a model's ability to adaptively exploit context. In this paper, we propose Cross-Preference Learning (CPL), a preference-based training framework that explicitly captures the complementary benefits of sentence-level and context-aware MT. CPL achieves this by integrating both intra- and cross-condition preferences into the preference optimization objective. The introduction of intra- and cross-condition preferences provides explicit supervision on when and how contextual information improves translation quality. We validate the proposed approach on several public context-aware MT tasks using multiple models, including Qwen3-4B, Qwen3-8B, and Llama-3-8B. Experimental results demonstrate consistent improvements in translation quality and robustness across both input conditions, achieved without any architectural modifications. 

---
# Prompt Attack Detection with LLM-as-a-Judge and Mixture-of-Models 

**Authors**: Hieu Xuan Le, Benjamin Goh, Quy Anh Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25176)  

**Abstract**: Prompt attacks, including jailbreaks and prompt injections, pose a critical security risk to Large Language Model (LLM) systems. In production, guardrails must mitigate these attacks under strict low-latency constraints, resulting in a deployment gap in which lightweight classifiers and rule-based systems struggle to generalize under distribution shift, while high-capacity LLM-based judges remain too slow or costly for live enforcement. In this work, we examine whether lightweight, general-purpose LLMs can reliably serve as security judges under real-world production constraints. Through careful prompt and output design, lightweight LLMs are guided through a structured reasoning process involving explicit intent decomposition, safety-signal verification, harm assessment, and self-reflection. We evaluate our method on a curated dataset combining benign queries from real-world chatbots with adversarial prompts generated via automated red teaming (ART), covering diverse and evolving patterns. Our results show that general-purpose LLMs, such as gemini-2.0-flash-lite-001, can serve as effective low-latency judges for live guardrails. This configuration is currently deployed in production as a centralized guardrail service for public service chatbots in Singapore. We additionally evaluate a Mixture-of-Models (MoM) setting to assess whether aggregating multiple LLM judges improves prompt-attack detection performance relative to single-model judges, with only modest gains observed. 

---
# To Write or to Automate Linguistic Prompts, That Is the Question 

**Authors**: Marina Sánchez-Torrón, Daria Akselrod, Jason Rauchwerk  

**Link**: [PDF](https://arxiv.org/pdf/2603.25169)  

**Abstract**: LLM performance is highly sensitive to prompt design, yet whether automatic prompt optimization can replace expert prompt engineering in linguistic tasks remains unexplored. We present the first systematic comparison of hand-crafted zero-shot expert prompts, base DSPy signatures, and GEPA-optimized DSPy signatures across translation, terminology insertion, and language quality assessment, evaluating five model configurations. Results are task-dependent. In terminology insertion, optimized and manual prompts produce mostly statistically indistinguishable quality. In translation, each approach wins on different models. In LQA, expert prompts achieve stronger error detection while optimization improves characterization. Across all tasks, GEPA elevates minimal DSPy signatures, and the majority of expert-optimized comparisons show no statistically significant difference. We note that the comparison is asymmetric: GEPA optimization searches programmatically over gold-standard splits, whereas expert prompts require in principle no labeled data, relying instead on domain expertise and iterative refinement. 

---
# Goodness-of-pronunciation without phoneme time alignment 

**Authors**: Jeremy H. M. Wong, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.25150)  

**Abstract**: In speech evaluation, an Automatic Speech Recognition (ASR) model often computes time boundaries and phoneme posteriors for input features. However, limited data for ASR training hinders expansion of speech evaluation to low-resource languages. Open-source weakly-supervised models are capable of ASR over many languages, but they are frame-asynchronous and not phonemic, hindering feature extraction for speech evaluation. This paper proposes to overcome incompatibilities for feature extraction with weakly-supervised models, easing expansion of speech evaluation to low-resource languages. Phoneme posteriors are computed by mapping ASR hypotheses to a phoneme confusion network. Word instead of phoneme-level speaking rate and duration are used. Phoneme and frame-level features are combined using a cross-attention architecture, obviating phoneme time alignment. This performs comparably with standard frame-synchronous features on English speechocean762 and low-resource Tamil datasets. 

---
# Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2603.25112)  

**Abstract**: Standard evaluation of LLM confidence relies on calibration metrics (ECE, Brier score) that conflate two distinct capacities: how much a model knows (Type-1 sensitivity) and how well it knows what it knows (Type-2 metacognitive sensitivity). We introduce an evaluation framework based on Type-2 Signal Detection Theory that decomposes these capacities using meta-d' and the metacognitive efficiency ratio M-ratio. Applied to four LLMs (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Llama-3-8B-Base, Gemma-2-9B-Instruct) across 224,000 factual QA trials, we find: (1) metacognitive efficiency varies substantially across models even when Type-1 sensitivity is similar -- Mistral achieves the highest d' but the lowest M-ratio; (2) metacognitive efficiency is domain-specific, with different models showing different weakest domains, invisible to aggregate metrics; (3) temperature manipulation shifts Type-2 criterion while meta-d' remains stable for two of four models, dissociating confidence policy from metacognitive capacity; (4) AUROC_2 and M-ratio produce fully inverted model rankings, demonstrating these metrics answer fundamentally different evaluation questions. The meta-d' framework reveals which models "know what they don't know" versus which merely appear well-calibrated due to criterion placement -- a distinction with direct implications for model selection, deployment, and human-AI collaboration. Pre-registered analysis; code and data publicly available. 

---
# OMIND: Framework for Knowledge Grounded Finetuning and Multi-Turn Dialogue Benchmark for Mental Health LLMs 

**Authors**: Suraj Racha, Prashant Harish Joshi, Utkarsh Maurya, Nitin Yadav, Mridul Sharma, Ananya Kunisetty, Saranya Darisipudi, Nirmal Punjabi, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2603.25105)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities for complex tasks, yet adaptation in medical domain, specifically mental health, poses specific challenges. Mental health is a rising concern globally with LLMs having large potential to help address the same. We highlight three primary challenges for LLMs in mental health - lack of high quality interpretable and knowledge grounded training data; training paradigms restricted to core capabilities, and evaluation of multi turn dialogue settings. Addressing it, we present oMind framework which includes training and aligning LLM agents for diverse capabilities including conversations; high quality ~164k multi-task SFT dataset, as a result of our generation pipeline based on Structured Knowledge retrieval, LLM based pruning, and review actions. We also introduce oMind-Chat - a novel multi turn benchmark dataset with expert annotated turn level and conversation level rubrics. Our diverse experiments on both core capabilities and conversations shows oMind LLMs consistently outperform baselines. oMind-LLM also shows significantly better reasoning with up to 80% win rate. 

---
# Closing the Confidence-Faithfulness Gap in Large Language Models 

**Authors**: Miranda Muqing Miao, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2603.25052)  

**Abstract**: Large language models (LLMs) tend to verbalize confidence scores that are largely detached from their actual accuracy, yet the geometric relationship governing this behavior remain poorly understood. In this work, we present a mechanistic interpretability analysis of verbalized confidence, using linear probes and contrastive activation addition (CAA) steering to show that calibration and verbalized confidence signals are encoded linearly but are orthogonal to one another -- a finding consistent across three open-weight models and four datasets. Interestingly, when models are prompted to simultaneously reason through a problem and verbalize a confidence score, the reasoning process disrupts the verbalized confidence direction, exacerbating miscalibration. We term this the "Reasoning Contamination Effect." Leveraging this insight, we introduce a two-stage adaptive steering pipeline that reads the model's internal accuracy estimate and steers verbalized output to match it, substantially improving calibration alignment across all evaluated models. 

---
# Approaches to Analysing Historical Newspapers Using LLMs 

**Authors**: Filip Dobranić, Tina Munda, Oliver Pejić, Vojko Gorjanc, Uroš Šmajdek, David Bordon, Jakob Lenardič, Tjaša Konovšek, Kristina Pahor de Maiti Tekavčič, Ciril Bohak, Darja Fišer  

**Link**: [PDF](https://arxiv.org/pdf/2603.25051)  

**Abstract**: This study presents a computational analysis of the Slovene historical newspapers \textit{Slovenec} and \textit{Slovenski narod} from the sPeriodika corpus, combining topic modelling, large language model (LLM)-based aspect-level sentiment analysis, entity-graph visualisation, and qualitative discourse analysis to examine how collective identities, political orientations, and national belonging were represented in public discourse at the turn of the twentieth century. Using BERTopic, we identify major thematic patterns and show both shared concerns and clear ideological differences between the two newspapers, reflecting their conservative-Catholic and liberal-progressive orientations. We further evaluate four instruction-following LLMs for targeted sentiment classification in OCR-degraded historical Slovene and select the Slovene-adapted GaMS3-12B-Instruct model as the most suitable for large-scale application, while also documenting important limitations, particularly its stronger performance on neutral sentiment than on positive or negative sentiment. Applied at dataset scale, the model reveals meaningful variation in the portrayal of collective identities, with some groups appearing predominantly in neutral descriptive contexts and others more often in evaluative or conflict-related discourse. We then create NER graphs to explore the relationships between collective identities and places. We apply a mixed methods approach to analyse the named entity graphs, combining quantitative network analysis with critical discourse analysis. The investigation focuses on the emergence and development of intertwined historical political and socionomic identities. Overall, the study demonstrates the value of combining scalable computational methods with critical interpretation to support digital humanities research on noisy historical newspaper data. 

---
# Imperative Interference: Social Register Shapes Instruction Topology in Large Language Models 

**Authors**: Tony Mason  

**Link**: [PDF](https://arxiv.org/pdf/2603.25015)  

**Abstract**: System prompt instructions that cooperate in English compete in Spanish, with the same semantic content, but opposite interaction topology. We present instruction-level ablation experiments across four languages and four models showing that this topology inversion is mediated by social register: the imperative mood carries different obligatory force across speech communities, and models trained on multilingual data have learned these conventions. Declarative rewriting of a single instruction block reduces cross-linguistic variance by 81% (p = 0.029, permutation test). Rewriting three of eleven imperative blocks shifts Spanish instruction topology from competitive to cooperative, with spillover effects on unrewritten blocks. These findings suggest that models process instructions as social acts, not technical specifications: "NEVER do X" is an exercise of authority whose force is language-dependent, while "X: disabled" is a factual description that transfers across languages. If register mediates instruction-following at inference time, it plausibly does so during training. We state this as a testable prediction: constitutional AI principles authored in imperative mood may create language-dependent alignment. Corpus: 22 hand-authored probes against a production system prompt decomposed into 56 blocks. 

---
# Exons-Detect: Identifying and Amplifying Exonic Tokens via Hidden-State Discrepancy for Robust AI-Generated Text Detection 

**Authors**: Xiaowei Zhu, Yubing Ren, Fang Fang, Shi Wang, Yanan Cao, Li Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.24981)  

**Abstract**: The rapid advancement of large language models has increasingly blurred the boundary between human-written and AI-generated text, raising societal risks such as misinformation dissemination, authorship ambiguity, and threats to intellectual property rights. These concerns highlight the urgent need for effective and reliable detection methods. While existing training-free approaches often achieve strong performance by aggregating token-level signals into a global score, they typically assume uniform token contributions, making them less robust under short sequences or localized token modifications. To address these limitations, we propose Exons-Detect, a training-free method for AI-generated text detection based on an exon-aware token reweighting perspective. Exons-Detect identifies and amplifies informative exonic tokens by measuring hidden-state discrepancy under a dual-model setting, and computes an interpretable translation score from the resulting importance-weighted token sequence. Empirical evaluations demonstrate that Exons-Detect achieves state-of-the-art detection performance and exhibits strong robustness to adversarial attacks and varying input lengths. In particular, it attains a 2.2\% relative improvement in average AUROC over the strongest prior baseline on DetectRL. 

---
# LLM-Driven Reasoning for Constraint-Aware Feature Selection in Industrial Systems 

**Authors**: Yuhang Zhou, Zhuokai Zhao, Ke Li, Spilios Evmorfos, Gökalp Demirci, Mingyi Wang, Qiao Liu, Qifei Wang, Serena Li, Weiwei Li, Tingting Wang, Mingze Gao, Gedi Zhou, Abhishek Kumar, Xiangjun Fan, Lizhu Zhang, Jiayi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24979)  

**Abstract**: Feature selection is a crucial step in large-scale industrial machine learning systems, directly affecting model accuracy, efficiency, and maintainability. Traditional feature selection methods rely on labeled data and statistical heuristics, making them difficult to apply in production environments where labeled data are limited and multiple operational constraints must be satisfied. To address this, we propose Model Feature Agent (MoFA), a model-driven framework that performs sequential, reasoning-based feature selection using both semantic and quantitative feature information. MoFA incorporates feature definitions, importance scores, correlations, and metadata (e.g., feature groups or types) into structured prompts and selects features through interpretable, constraint-aware reasoning. We evaluate MoFA in three real-world industrial applications: (1) True Interest and Time-Worthiness Prediction, where it improves accuracy while reducing feature group complexity, (2) Value Model Enhancement, where it discovers high-order interaction terms that yield substantial engagement gains in online experiments, and (3) Notification Behavior Prediction, where it selects compact, high-value feature subsets that improve both model accuracy and inference efficiency. Together, these results demonstrate the practicality and effectiveness of LLM-based reasoning for feature selection in real production systems. 

---
# Toward domain-specific machine translation and quality estimation systems 

**Authors**: Javad Pourmostafa Roshan Sharami  

**Link**: [PDF](https://arxiv.org/pdf/2603.24955)  

**Abstract**: Machine Translation (MT) and Quality Estimation (QE) perform well in general domains but degrade under domain mismatch. This dissertation studies how to adapt MT and QE systems to specialized domains through a set of data-focused contributions. Chapter 2 presents a similarity-based data selection method for MT. Small, targeted in-domain subsets outperform much larger generic datasets and reach strong translation quality at lower computational cost. Chapter 3 introduces a staged QE training pipeline that combines domain adaptation with lightweight data augmentation. The method improves performance across domains, languages, and resource settings, including zero-shot and cross-lingual cases. Chapter 4 studies the role of subword tokenization and vocabulary in fine-tuning. Aligned tokenization-vocabulary setups lead to stable training and better translation quality, while mismatched configurations reduce performance. Chapter 5 proposes a QE-guided in-context learning method for large language models. QE models select examples that improve translation quality without parameter updates and outperform standard retrieval methods. The approach also supports a reference-free setup, reducing reliance on a single reference set. These results show that domain adaptation depends on data selection, representation, and efficient adaptation strategies. The dissertation provides methods for building MT and QE systems that perform reliably in domain-specific settings. 

---
# Estimating near-verbatim extraction risk in language models with decoding-constrained beam search 

**Authors**: A. Feder Cooper, Mark A. Lemley, Christopher De Sa, Lea Duesterwald, Allison Casasola, Jamie Hayes, Katherine Lee, Daniel E. Ho, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24917)  

**Abstract**: Recent work shows that standard greedy-decoding extraction methods for quantifying memorization in LLMs miss how extraction risk varies across sequences. Probabilistic extraction -- computing the probability of generating a target suffix given a prefix under a decoding scheme -- addresses this, but is tractable only for verbatim memorization, missing near-verbatim instances that pose similar privacy and copyright risks. Quantifying near-verbatim extraction risk is expensive: the set of near-verbatim suffixes is combinatorially large, and reliable Monte Carlo (MC) estimation can require ~100,000 samples per sequence. To mitigate this cost, we introduce decoding-constrained beam search, which yields deterministic lower bounds on near-verbatim extraction risk at a cost comparable to ~20 MC samples per sequence. Across experiments, our approach surfaces information invisible to verbatim methods: many more extractable sequences, substantially larger per-sequence extraction mass, and patterns in how near-verbatim extraction risk manifests across model sizes and types of text. 

---
# LogSigma at SemEval-2026 Task 3: Uncertainty-Weighted Multitask Learning for Dimensional Aspect-Based Sentiment Analysis 

**Authors**: Baraa Hikal, Jonas Becker, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2603.24896)  

**Abstract**: This paper describes LogSigma, our system for SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DimABSA). Unlike traditional Aspect-Based Sentiment Analysis (ABSA), which predicts discrete sentiment labels, DimABSA requires predicting continuous Valence and Arousal (VA) scores on a 1-9 scale. A central challenge is that Valence and Arousal differ in prediction difficulty across languages and domains. We address this using learned homoscedastic uncertainty, where the model learns task-specific log-variance parameters to automatically balance each regression objective during training. Combined with language-specific encoders and multi-seed ensembling, LogSigma achieves 1st place on five datasets across both tracks. The learned variance weights vary substantially across languages due to differing Valence-Arousal difficulty profiles-from 0.66x for German to 2.18x for English-demonstrating that optimal task balancing is language-dependent and cannot be determined a priori. 

---
# Prune as You Generate: Online Rollout Pruning for Faster and Better RLVR 

**Authors**: Haobo Xu, Sirui Chen, Ruizhong Qiu, Yuchen Yan, Chen Luo, Monica Cheng, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2603.24840)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capabilities of Large Language Models (LLMs). However, methods such as GRPO and DAPO suffer from substantial computational cost, since they rely on sampling many rollouts for each prompt. Moreover, in RLVR the relative advantage is often sparse: many samples become nearly all-correct or all-incorrect, yielding low within-group reward variance and thus weak learning signals. In this paper, we introduce arrol (Accelerating RLVR via online Rollout Pruning), an online rollout pruning method that prunes rollouts during generation while explicitly steering the surviving ones more correctness-balanced to enhance learning signals. Specifically, arrol trains a lightweight quality head on-the-fly to predict the success probability of partial rollouts and uses it to make early pruning decisions. The learned quality head can further weigh candidates to improve inference accuracy during test-time scaling. To improve efficiency, we present a system design that prunes rollouts inside the inference engine and re-batches the remaining ones for log-probability computation and policy updates. Across GRPO and DAPO on Qwen-3 and LLaMA-3.2 models (1B-8B), arrol improves average accuracy by +2.30 to +2.99 while achieving up to 1.7x training speedup, and yielding up to +8.33 additional gains in average accuracy in test-time scaling. The code is available at this https URL. 

---
# Synthetic Rewriting as a Quality Multiplier: Evidence from Portuguese Continued Pretraining 

**Authors**: Thales Sales Almeida, Rodrigo Nogueira, Hélio Pedrini  

**Link**: [PDF](https://arxiv.org/pdf/2603.24826)  

**Abstract**: Synthetic data generation through document rewriting has emerged as a promising technique for improving language model pretraining, yet most studies focus on English and do not systematically control for the quality of the source data being rewritten. We present a controlled study of how synthetic rewriting interacts with source data quality in the context of Portuguese continued pretraining. Starting from ClassiCC-PT, a Portuguese corpus annotated with STEM and Educational quality scores, we construct two 10B-token subsets at different quality levels and rewrite each into four styles using a 7B instruction-tuned model, producing approximately 40B tokens of synthetic data per condition. We train two English-centric base models (1.1B and 7B parameters) on each condition and evaluate on PoETa V2, a comprehensive 44-task Portuguese benchmark. At the 7B scale, rewriting high-quality data yields a +3.4 NPM gain over the same data unmodified, while rewriting low-quality data provides only +0.5 NPM. At the 1.1B scale, this interaction is weaker, with unmodified low-quality data performing comparably to rewritten high-quality data. Our results demonstrate that synthetic rewriting acts primarily as a quality multiplier rather than a substitute for data curation, and that this effect is scale-dependent. 

---
# Enhancing Structured Meaning Representations with Aspect Classification 

**Authors**: Claire Benét Post, Paul Bontempo, August Milliken, Alvin Po-Chun Chen, Nicholas Derby, Saksham Khatwani, Sumeyye Nabieva, Karthik Sairam, Alexis Palmer  

**Link**: [PDF](https://arxiv.org/pdf/2603.24797)  

**Abstract**: To fully capture the meaning of a sentence, semantic representations should encode aspect, which describes the internal temporal structure of events. In graph-based meaning representation frameworks such as Uniform Meaning Representations (UMR), aspect lets one know how events unfold over time, including distinctions such as states, activities, and completed events. Despite its importance, aspect remains sparsely annotated across semantic meaning representation frameworks. This has, in turn, hindered not only current manual annotation, but also the development of automatic systems capable of predicting aspectual information. In this paper, we introduce a new dataset of English sentences annotated with UMR aspect labels over Abstract Meaning Representation (AMR) graphs that lack the feature. We describe the annotation scheme and guidelines used to label eventive predicates according to the UMR aspect lattice, as well as the annotation pipeline used to ensure consistency and quality across annotators through a multi-step adjudication process. To demonstrate the utility of our dataset for future automation, we present baseline experiments using three modeling approaches. Our results establish initial benchmarks for automatic UMR aspect prediction and provide a foundation for integrating aspect into semantic meaning representations more broadly. 

---
# Evaluating Fine-Tuned LLM Model For Medical Transcription With Small Low-Resource Languages Validated Dataset 

**Authors**: Mohammed Nowshad Ruhani Chowdhury, Mohammed Nowaz Rabbani Chowdhury, Sakari Lukkarinen  

**Link**: [PDF](https://arxiv.org/pdf/2603.24772)  

**Abstract**: Clinical documentation is a critical factor for patient safety, diagnosis, and continuity of care. The administrative burden of EHRs is a significant factor in physician burnout. This is a critical issue for low-resource languages, including Finnish. This study aims to investigate the effectiveness of a domain-aligned natural language processing (NLP); large language model for medical transcription in Finnish by fine-tuning LLaMA 3.1-8B on a small validated corpus of simulated clinical conversations by students at Metropolia University of Applied Sciences. The fine-tuning process for medical transcription used a controlled preprocessing and optimization approach. The fine-tuning effectiveness was evaluated by sevenfold cross-validation. The evaluation metrics for fine-tuned LLaMA 3.1-8B were BLEU = 0.1214, ROUGE-L = 0.4982, and BERTScore F1 = 0.8230. The results showed a low n-gram overlap but a strong semantic similarity with reference transcripts. This study indicate that fine-tuning can be an effective approach for translation of medical discourse in spoken Finnish and support the feasibility of fine-tuning a privacy-oriented domain-specific large language model for clinical documentation in Finnish. Beside that provide directions for future work. 

---
# Fine-Tuning A Large Language Model for Systematic Review Screening 

**Authors**: Kweku Yamoah, Noah Schroeder, Emmanuel Dorley, Neha Rani, Caleb Schutz  

**Link**: [PDF](https://arxiv.org/pdf/2603.24767)  

**Abstract**: Systematic reviews traditionally have taken considerable amounts of human time and energy to complete, in part due to the extensive number of titles and abstracts that must be reviewed for potential inclusion. Recently, researchers have begun to explore how to use large language models (LLMs) to make this process more efficient. However, research to date has shown inconsistent results. We posit this is because prompting alone may not provide sufficient context for the model(s) to perform well. In this study, we fine-tune a small 1.2 billion parameter open-weight LLM specifically for study screening in the context of a systematic review in which humans rated more than 8500 titles and abstracts for potential inclusion. Our results showed strong performance improvements from the fine-tuned model, with the weighted F1 score improving 80.79% compared to the base model. When run on the full dataset of 8,277 studies, the fine-tuned model had 86.40% agreement with the human coder, a 91.18% true positive rate, a 86.38% true negative rate, and perfect agreement across multiple inference runs. Taken together, our results show that there is promise for fine-tuning LLMs for title and abstract screening in large-scale systematic reviews. 

---
# Demystifying When Pruning Works via Representation Hierarchies 

**Authors**: Shwai He, Guoheng Sun, Haichao Zhang, Yun Fu, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.24652)  

**Abstract**: Network pruning, which removes less important parameters or architectures, is often expected to improve efficiency while preserving performance. However, this expectation does not consistently hold across language tasks: pruned models can perform well on non-generative tasks but frequently fail in generative settings. To understand this discrepancy, we analyze network pruning from a representation-hierarchy perspective, decomposing the internal computation of language models into three sequential spaces: embedding (hidden representations), logit (pre-softmax outputs), and probability (post-softmax distributions). We find that representations in the embedding and logit spaces are largely robust to pruning-induced perturbations. However, the nonlinear transformation from logits to probabilities amplifies these deviations, which accumulate across time steps and lead to substantial degradation during generation. In contrast, the stability of the categorical-token probability subspace, together with the robustness of the embedding space, supports the effectiveness of pruning for non-generative tasks such as retrieval and multiple-choice selection. Our analysis disentangles the effects of pruning across tasks and provides practical guidance for its application. Code is available at this https URL 

---
# When Consistency Becomes Bias: Interviewer Effects in Semi-Structured Clinical Interviews 

**Authors**: Hasindri Watawana, Sergio Burdisso, Diego A. Moreno-Galván, Fernando Sánchez-Vega, A. Pastor López-Monroy, Petr Motlicek, Esaú Villatoro-Tello  

**Link**: [PDF](https://arxiv.org/pdf/2603.24651)  

**Abstract**: Automatic depression detection from doctor-patient conversations has gained momentum thanks to the availability of public corpora and advances in language modeling. However, interpretability remains limited: strong performance is often reported without revealing what drives predictions. We analyze three datasets: ANDROIDS, DAIC-WOZ, E-DAIC and identify a systematic bias from interviewer prompts in semi-structured interviews. Models trained on interviewer turns exploit fixed prompts and positions to distinguish depressed from control subjects, often achieving high classification scores without using participant language. Restricting models to participant utterances distributes decision evidence more broadly and reflects genuine linguistic cues. While semi-structured protocols ensure consistency, including interviewer prompts inflates performance by leveraging script artifacts. Our results highlight a cross-dataset, architecture-agnostic bias and emphasize the need for analyses that localize decision evidence by time and speaker to ensure models learn from participants' language. 

---
# Training the Knowledge Base through Evidence Distillation and Write-Back Enrichment 

**Authors**: Yuxing Lu, Xukai Zhao, Wei Wu, Jinzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25737)  

**Abstract**: The knowledge base in a retrieval-augmented generation (RAG) system is typically assembled once and never revised, even though the facts a query requires are often fragmented across documents and buried in irrelevant content. We argue that the knowledge base should be treated as a trainable component and propose WriteBack-RAG, a framework that uses labeled examples to identify where retrieval succeeds, isolate the relevant documents, and distill them into compact knowledge units that are indexed alongside the original corpus. Because the method modifies only the corpus, it can be applied once as an offline preprocessing step and combined with any RAG pipeline. Across four RAG methods, six benchmarks, and two LLM backbones, WriteBack-RAG improves every evaluated setting, with gains averaging +2.14%. Cross-method transfer experiments further show that the distilled knowledge benefits RAG pipelines other than the one used to produce it, confirming that the improvement resides in the corpus itself. 

---
# RenoBench: A Citation Parsing Benchmark 

**Authors**: Parth Sarin, Juan Pablo Alperin, Adam Buttrick, Dione Mentis  

**Link**: [PDF](https://arxiv.org/pdf/2603.25640)  

**Abstract**: Accurate parsing of citations is necessary for machine-readable scholarly infrastructure. But, despite sustained interest in this problem, existing evaluation techniques are often not generalizable, based on synthetic data, or not publicly available. We introduce RenoBench, a public domain benchmark for citation parsing, sourced from PDFs released on four publishing ecosystems: SciELO, Redalyc, the Public Knowledge Project, and Open Research Europe. Starting from 161,000 annotated citations, we apply automated validation and feature-based sampling to produce a dataset of 10,000 citations spanning multiple languages, publication types, and platforms. We then evaluate a variety of citation parsing systems and report field-level precision and recall. Our results show strong performance from language models, particularly when fine-tuned. RenoBench enables reproducible, standardized evaluation of citation parsing systems, and provides a foundation for advancing automated citation parsing and metascientific research. 

---
# Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes 

**Authors**: Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Yuanheng Zhu, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.25562)  

**Abstract**: On-policy distillation (OPD) is appealing for large language model (LLM) post-training because it evaluates teacher feedback on student-generated rollouts rather than fixed teacher traces. In long-horizon settings, however, the common sampled-token variant is fragile: it reduces distribution matching to a one-token signal and becomes increasingly unreliable as rollouts drift away from prefixes the teacher commonly visits. We revisit OPD from the estimator and implementation sides. Theoretically, token-level OPD is biased relative to sequence-level reverse-KL, but it has a much tighter worst-case variance bound; our toy study shows the same tradeoff empirically, with stronger future-reward coupling producing higher gradient variance and less stable learning. Empirically, we identify three failure modes of sampled-token OPD: an imbalanced one-token signal, unreliable teacher guidance on student-generated prefixes, and distortions caused by tokenizer or special-token mismatch. We address these issues with teacher top-K local support matching, implemented as truncated reverse-KL with top-p rollout sampling and special-token masking. Across single-task math reasoning and multi-task agentic-plus-math training, this objective yields more stable optimization and better downstream performance than sampled-token OPD. 

---
# Synchronous Signal Temporal Logic for Decidable Verification of Cyber-Physical Systems 

**Authors**: Partha Roop, Sobhan Chatterjee, Avinash Malik, Nathan Allen, Logan Kenwright  

**Link**: [PDF](https://arxiv.org/pdf/2603.25531)  

**Abstract**: Many Cyber Physical System (CPS) work in a safety-critical environment, where correct execution, reliability and trustworthiness are essential. Signal Temporal Logic (STL) provides a formal framework for checking safety-critical CPS. However, static verification of STL is undecidable in general, except when we want to verify using run-time-based methods, which have limitations. We propose Synchronous Signal Temporal Logic (SSTL), a decidable fragment of STL, which admits static safety and liveness property verification. In SSTL, we assume that a signal is sampled at fixed discrete steps, called ticks, and then propose a hypothesis, called the Signal Invariance Hypothesis (SIH), which is inspired by a similar hypothesis for synchronous programs. We define the syntax and semantics of SSTL and show that SIH is a necessary and sufficient condition for equivalence between an STL formula and its SSTL counterpart. By translating SSTL to LTL_P (LTL defined over predicates), we enable decidable model checking using the SPIN model checker. We demonstrate the approach on a 33-node human heart model and other case studies. 

---
# Supercharging Federated Intelligence Retrieval 

**Authors**: Dimitris Stripelis, Patrick Foley, Mohammad Naseri, William Lindskog-Münzing, Chong Shen Ng, Daniel Janes Beutel, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2603.25374)  

**Abstract**: RAG typically assumes centralized access to documents, which breaks down when knowledge is distributed across private data silos. We propose a secure Federated RAG system built using Flower that performs local silo retrieval, while server-side aggregation and text generation run inside an attested, confidential compute environment, enabling confidential remote LLM inference even in the presence of honest-but-curious or compromised servers. We also propose a cascading inference approach that incorporates a non-confidential third-party model (e.g., Amazon Nova) as auxiliary context without weakening confidentiality. 

---
# DAGverse: Building Document-Grounded Semantic DAGs from Scientific Papers 

**Authors**: Shu Wan, Saketh Vishnubhatla, Iskander Kushbay, Tom Heffernan, Aaron Belikoff, Raha Moraffah, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.25293)  

**Abstract**: Directed Acyclic Graphs (DAGs) are widely used to represent structured knowledge in scientific and technical domains. However, datasets for real-world DAGs remain scarce because constructing them typically requires expert interpretation of domain documents. We study Doc2SemDAG construction: recovering a preferred semantic DAG from a document together with the cited evidence and context that explain it. This problem is challenging because a document may admit multiple plausible abstractions, the intended structure is often implicit, and the supporting evidence is scattered across prose, equations, captions, and figures. To address these challenges, we leverage scientific papers containing explicit DAG figures as a natural source of supervision. In this setting, the DAG figure provides the DAG structure, while the accompanying text provides context and explanation. We introduce DAGverse, a framework for constructing document-grounded semantic DAGs from online scientific papers. Its core component, DAGverse-Pipeline, is a semi-automatic system designed to produce high-precision semantic DAG examples through figure classification, graph reconstruction, semantic grounding, and validation. As a case study, we test the framework for causal DAGs and release DAGverse-1, a dataset of 108 expert-validated semantic DAGs with graph-level, node-level, and edge-level evidence. Experiments show that DAGverse-Pipeline outperforms existing Vision-Language Models on DAG classification and annotation. DAGverse provides a foundation for document-grounded DAG benchmarks and opens new directions for studying structured reasoning grounded in real-world evidence. 

---
# WebTestBench: Evaluating Computer-Use Agents towards End-to-End Automated Web Testing 

**Authors**: Fanheng Kong, Jingyuan Zhang, Yang Yue, Chenxi Sun, Yang Tian, Shi Feng, Xiaocui Yang, Daling Wang, Yu Tian, Jun Du, Wenchong Zeng, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2603.25226)  

**Abstract**: The emergence of Large Language Models (LLMs) has catalyzed a paradigm shift in programming, giving rise to "vibe coding", where users can build complete projects and even control computers using natural language instructions. This paradigm has driven automated webpage development, but it introduces a new requirement about how to automatically verify whether the web functionalities are reliably implemented. Existing works struggle to adapt, relying on static visual similarity or predefined checklists that constrain their utility in open-ended environments. Furthermore, they overlook a vital aspect of software quality, namely latent logical constraints. To address these gaps, we introduce WebTestBench, a benchmark for evaluating end-to-end automated web testing. WebTestBench encompasses comprehensive dimensions across diverse web application categories. We decompose the testing process into two cascaded sub-tasks, checklist generation and defect detection, and propose WebTester, a baseline framework for this task. Evaluating popular LLMs with WebTester reveals severe challenges, including insufficient test completeness, detection bottlenecks, and long-horizon interaction unreliability. These findings expose a substantial gap between current computer-use agent capabilities and industrial-grade deployment demands. We hope that WebTestBench provides valuable insights and guidance for advancing end-to-end automated web testing. Our dataset and code are available at this https URL. 

---
# Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection 

**Authors**: Ruichao Yang, Wei Gao, Xiaobin Zhu, Jing Ma, Hongzhan Lin, Ziyang Luo, Bo-Wen Zhang, Xu-Cheng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2603.25203)  

**Abstract**: Multimodal misinformation poses an escalating challenge that often evades traditional detectors, which are opaque black boxes and fragile against new manipulation tactics. We present Probabilistic Concept Graph Reasoning (PCGR), an interpretable and evolvable framework that reframes multimodal misinformation detection (MMD) as structured and concept-based reasoning. PCGR follows a build-then-infer paradigm, which first constructs a graph of human-understandable concept nodes, including novel high-level concepts automatically discovered and validated by multimodal large language models (MLLMs), and then applies hierarchical attention over this concept graph to infer claim veracity. This design produces interpretable reasoning chains linking evidence to conclusions. Experiments demonstrate that PCGR achieves state-of-the-art MMD accuracy and robustness to emerging manipulation types, outperforming prior methods in both coarse detection and fine-grained manipulation recognition. 

---
# Bilingual Text-to-Motion Generation: A New Benchmark and Baselines 

**Authors**: Wanjiang Weng, Xiaofeng Tan, Xiangbo Shu, Guo-Sen Xie, Pan Zhou, Hongsong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25178)  

**Abstract**: Text-to-motion generation holds significant potential for cross-linguistic applications, yet it is hindered by the lack of bilingual datasets and the poor cross-lingual semantic understanding of existing language models. To address these gaps, we introduce BiHumanML3D, the first bilingual text-to-motion benchmark, constructed via LLM-assisted annotation and rigorous manual correction. Furthermore, we propose a simple yet effective baseline, Bilingual Motion Diffusion (BiMD), featuring Cross-Lingual Alignment (CLA). CLA explicitly aligns semantic representations across languages, creating a robust conditional space that enables high-quality motion generation from bilingual inputs, including zero-shot code-switching scenarios. Extensive experiments demonstrate that BiMD with CLA achieves an FID of 0.045 vs. 0.169 and R@3 of 82.8\% vs. 80.8\%, significantly outperforms monolingual diffusion models and translation baselines on BiHumanML3D, underscoring the critical necessity and reliability of our dataset and the effectiveness of our alignment strategy for cross-lingual motion synthesis. The dataset and code are released at \href{this https URL}{this https URL} 

---
# Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale 

**Authors**: Yicheng Zou, Dongsheng Zhu, Lin Zhu, Tong Zhu, Yunhua Zhou, Peiheng Zhou, Xinyu Zhou, Dongzhan Zhou, Zhiwang Zhou, Yuhao Zhou, Bowen Zhou, Zhanping Zhong, Zhijie Zhong, Haiteng Zhao, Penghao Zhao, Xiaomeng Zhao, Zhiyuan Zhao, Yechen Zhang, Jin Zhang, Wenwei Zhang, Hongjie Zhang, Zhuo Zhang, Wenlong Zhang, Bo Zhang, Chao Zhang, Chen Zhang, Yuhang Zang, Fei Yuan, Jiakang Yuan, Jiashuo Yu, Jinhui Yin, Haochen Ye, Qian Yao, Bowen Yang, Danni Yang, Kaichen Yang, Ziang Yan, Jun Xu, Yicheng Xu, Wanghan Xu, Xuenan Xu, Chao Xu, Ruiliang Xu, Shuhao Xing, Long Xing, Xinchen Xie, Ling-I Wu, Zijian Wu, Zhenyu Wu, Lijun Wu, Yue Wu, Jianyu Wu, Wen Wu, Fan Wu, Xilin Wei, Qi Wei, Bingli Wang, Rui Wang, Ziyi Wang, Zun Wang, Yi Wang, Haomin Wang, Yizhou Wang, Lintao Wang, Yiheng Wang, Longjiang Wang, Bin Wang, Jian Tong, Zhongbo Tian, Huanze Tang, Chen Tang, Shixiang Tang, Yu Sun, Qiushi Sun, Xuerui Su, Qisheng Su, Chenlin Su, Demin Song, Jin Shi, Fukai Shang, Yuchen Ren, Pengli Ren, Xiaoye Qu, Yuan Qu, Jiantao Qiu, Yu Qiao, Runyu Peng, Tianshuo Peng, Jiahui Peng, Qizhi Pei, Zhuoshi Pan, Linke Ouyang, Wenchang Ning, Yichuan Ma, Zerun Ma, Ningsheng Ma, Runyuan Ma, Chengqi Lyu, Haijun Lv, Han Lv  

**Link**: [PDF](https://arxiv.org/pdf/2603.25040)  

**Abstract**: We introduce Intern-S1-Pro, the first one-trillion-parameter scientific multimodal foundation model. Scaling to this unprecedented size, the model delivers a comprehensive enhancement across both general and scientific domains. Beyond stronger reasoning and image-text understanding capabilities, its intelligence is augmented with advanced agent capabilities. Simultaneously, its scientific expertise has been vastly expanded to master over 100 specialized tasks across critical science fields, including chemistry, materials, life sciences, and earth sciences. Achieving this massive scale is made possible by the robust infrastructure support of XTuner and LMDeploy, which facilitates highly efficient Reinforcement Learning (RL) training at the 1-trillion parameter level while ensuring strict precision consistency between training and inference. By seamlessly integrating these advancements, Intern-S1-Pro further fortifies the fusion of general and specialized intelligence, working as a Specializable Generalist, demonstrating its position in the top tier of open-source models for general capabilities, while outperforming proprietary models in the depth of specialized scientific tasks. 

---
# Can MLLMs Read Students' Minds? Unpacking Multimodal Error Analysis in Handwritten Math 

**Authors**: Dingjie Song, Tianlong Xu, Yi-Fan Zhang, Hang Li, Zhiling Yan, Xing Fan, Haoyang Li, Lichao Sun, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2603.24961)  

**Abstract**: Assessing student handwritten scratchwork is crucial for personalized educational feedback but presents unique challenges due to diverse handwriting, complex layouts, and varied problem-solving approaches. Existing educational NLP primarily focuses on textual responses and neglects the complexity and multimodality inherent in authentic handwritten scratchwork. Current multimodal large language models (MLLMs) excel at visual reasoning but typically adopt an "examinee perspective", prioritizing generating correct answers rather than diagnosing student errors. To bridge these gaps, we introduce ScratchMath, a novel benchmark specifically designed for explaining and classifying errors in authentic handwritten mathematics scratchwork. Our dataset comprises 1,720 mathematics samples from Chinese primary and middle school students, supporting two key tasks: Error Cause Explanation (ECE) and Error Cause Classification (ECC), with seven defined error types. The dataset is meticulously annotated through rigorous human-machine collaborative approaches involving multiple stages of expert labeling, review, and verification. We systematically evaluate 16 leading MLLMs on ScratchMath, revealing significant performance gaps relative to human experts, especially in visual recognition and logical reasoning. Proprietary models notably outperform open-source models, with large reasoning models showing strong potential for error explanation. All evaluation data and frameworks are publicly available to facilitate further research. 

---
# FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol 

**Authors**: Jie Zhu, Yimin Tian, Boyang Li, Kehao Wu, Zhongzhi Liang, Junhui Li, Xianyin Zhang, Lifan Guo, Feng Chen, Yong Liu, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24943)  

**Abstract**: This paper introduces \textbf{FinMCP-Bench}, a novel benchmark for evaluating large language models (LLMs) in solving real-world financial problems through tool invocation of financial model context protocols. FinMCP-Bench contains 613 samples spanning 10 main scenarios and 33 sub-scenarios, featuring both real and synthetic user queries to ensure diversity and authenticity. It incorporates 65 real financial MCPs and three types of samples, single tool, multi-tool, and multi-turn, allowing evaluation of models across different levels of task complexity. Using this benchmark, we systematically assess a range of mainstream LLMs and propose metrics that explicitly measure tool invocation accuracy and reasoning capabilities. FinMCP-Bench provides a standardized, practical, and challenging testbed for advancing research on financial LLM agents. 

---
# Beyond Attention Magnitude: Leveraging Inter-layer Rank Consistency for Efficient Vision-Language-Action Models 

**Authors**: Peiju Liu, Jinming Liu, Xipeng Qiu, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24941)  

**Abstract**: Vision-Language-Action (VLA) models excel in robotic manipulation but suffer from significant inference latency due to processing dense visual tokens. Existing token reduction methods predominantly rely on attention magnitude as a static selection. In this work, we challenge this assumption, revealing that high-attention tokens are task-dependent and can even degrade policy performance. To address this, we introduce \textbf{TIES} (\textbf{T}au-guided \textbf{I}nter-layer \textbf{E}fficient \textbf{S}election), a dynamic framework guided by inter-layer token ranking consistency. By adaptively balancing attention magnitude with ranking consistency, TIES ensures robust token selection without requiring additional training. On the CogACT + SIMPLER benchmark, TIES improves average success rates by 6\% while reducing token usage by 78\%, and demonstrate strong generalization across diverse decoders and benchmarks. 

---
# LogitScope: A Framework for Analyzing LLM Uncertainty Through Information Metrics 

**Authors**: Farhan Ahmed, Yuya Jeremy Ong, Chad DeLuca  

**Link**: [PDF](https://arxiv.org/pdf/2603.24929)  

**Abstract**: Understanding and quantifying uncertainty in large language model (LLM) outputs is critical for reliable deployment. However, traditional evaluation approaches provide limited insight into model confidence at individual token positions during generation. To address this issue, we introduce LogitScope, a lightweight framework for analyzing LLM uncertainty through token-level information metrics computed from probability distributions. By measuring metrics such as entropy and varentropy at each generation step, LogitScope reveals patterns in model confidence, identifies potential hallucinations, and exposes decision points where models exhibit high uncertainty, all without requiring labeled data or semantic interpretation. We demonstrate LogitScope's utility across diverse applications including uncertainty quantification, model behavior analysis, and production monitoring. The framework is model-agnostic, computationally efficient through lazy evaluation, and compatible with any HuggingFace model, enabling both researchers and practitioners to inspect LLM behavior during inference. 

---
# GraphER: An Efficient Graph-Based Enrichment and Reranking Method for Retrieval-Augmented Generation 

**Authors**: Ruizhong Miao, Yuying Wang, Rongguang Wang, Chenyang Li, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2603.24925)  

**Abstract**: Semantic search in retrieval-augmented generation (RAG) systems is often insufficient for complex information needs, particularly when relevant evidence is scattered across multiple sources. Prior approaches to this problem include agentic retrieval strategies, which expand the semantic search space by generating additional queries. However, these methods do not fully leverage the organizational structure of the data and instead rely on iterative exploration, which can lead to inefficient retrieval. Another class of approaches employs knowledge graphs to model non-semantic relationships through graph edges. Although effective in capturing richer proximities, such methods incur significant maintenance costs and are often incompatible with the vector stores used in most production systems. To address these limitations, we propose GraphER, a graph-based enrichment and reranking method that captures multiple forms of proximity beyond semantic similarity. GraphER independently enriches data objects during offline indexing and performs graph-based reranking over candidate objects at query time. This design does not require a knowledge graph, allowing GraphER to integrate seamlessly with standard vector stores. In addition, GraphER is retriever-agnostic and introduces negligible latency overhead. Experiments on multiple retrieval benchmarks demonstrate the effectiveness of the proposed approach. 

---
# How Far Are Vision-Language Models from Constructing the Real World? A Benchmark for Physical Generative Reasoning 

**Authors**: Luyu Yang, Yutong Dai, An Yan, Viraj Prabhu, Ran Xu, Zeyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.24866)  

**Abstract**: The physical world is not merely visual; it is governed by rigorous structural and procedural constraints. Yet, the evaluation of vision-language models (VLMs) remains heavily skewed toward perceptual realism, prioritizing the generation of visually plausible 3D layouts, shapes, and appearances. Current benchmarks rarely test whether models grasp the step-by-step processes and physical dependencies required to actually build these artifacts, a capability essential for automating design-to-construction pipelines. To address this, we introduce DreamHouse, a novel benchmark for physical generative reasoning: the capacity to synthesize artifacts that concurrently satisfy geometric, structural, constructability, and code-compliance constraints. We ground this benchmark in residential timber-frame construction, a domain with fully codified engineering standards and objectively verifiable correctness. We curate over 26,000 structures spanning 13 architectural styles, ach verified to construction-document standards (LOD 350) and develop a deterministic 10-test structural validation framework. Unlike static benchmarks that assess only final outputs, DreamHouse supports iterative agentic interaction. Models observe intermediate build states, generate construction actions, and receive structured environmental feedback, enabling a fine-grained evaluation of planning, structural reasoning, and self-correction. Extensive experiments with state-of-the-art VLMs reveal substantial capability gaps that are largely invisible on existing leaderboards. These findings establish physical validity as a critical evaluation axis orthogonal to visual realism, highlighting physical generative reasoning as a distinct and underdeveloped frontier in multimodal intelligence. Available at this https URL 

---
# AI Security in the Foundation Model Era: A Comprehensive Survey from a Unified Perspective 

**Authors**: Zhenyi Wang, Siyu Luan  

**Link**: [PDF](https://arxiv.org/pdf/2603.24857)  

**Abstract**: As machine learning (ML) systems expand in both scale and functionality, the security landscape has become increasingly complex, with a proliferation of attacks and defenses. However, existing studies largely treat these threats in isolation, lacking a coherent framework to expose their shared principles and interdependencies. This fragmented view hinders systematic understanding and limits the design of comprehensive defenses. Crucially, the two foundational assets of ML -- \textbf{data} and \textbf{models} -- are no longer independent; vulnerabilities in one directly compromise the other. The absence of a holistic framework leaves open questions about how these bidirectional risks propagate across the ML pipeline. To address this critical gap, we propose a \emph{unified closed-loop threat taxonomy} that explicitly frames model-data interactions along four directional axes. Our framework offers a principled lens for analyzing and defending foundation models. The resulting four classes of security threats represent distinct but interrelated categories of attacks: (1) Data$\rightarrow$Data (D$\rightarrow$D): including \emph{data decryption attacks and watermark removal attacks}; (2) Data$\rightarrow$Model (D$\rightarrow$M): including \emph{poisoning, harmful fine-tuning attacks, and jailbreak attacks}; (3) Model$\rightarrow$Data (M$\rightarrow$D): including \emph{model inversion, membership inference attacks, and training data extraction attacks}; (4) Model$\rightarrow$Model (M$\rightarrow$M): including \emph{model extraction attacks}. Our unified framework elucidates the underlying connections among these security threats and establishes a foundation for developing scalable, transferable, and cross-modal security strategies, particularly within the landscape of foundation models. 

---
# Reaching Beyond the Mode: RL for Distributional Reasoning in Language Models 

**Authors**: Isha Puri, Mehul Damani, Idan Shenfeld, Marzyeh Ghassemi, Jacob Andreas, Yoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.24844)  

**Abstract**: Given a question, a language model (LM) implicitly encodes a distribution over possible answers. In practice, post-training procedures for LMs often collapse this distribution onto a single dominant mode. While this is generally not a problem for benchmark-style evaluations that assume one correct answer, many real-world tasks inherently involve multiple valid answers or irreducible uncertainty. Examples include medical diagnosis, ambiguous question answering, and settings with incomplete information. In these cases, we would like LMs to generate multiple plausible hypotheses, ideally with confidence estimates for each one, and without computationally intensive repeated sampling to generate non-modal answers. This paper describes a multi-answer reinforcement learning approach for training LMs to perform distributional reasoning over multiple answers during inference. We modify the RL objective to enable models to explicitly generate multiple candidate answers in a single forward pass, internalizing aspects of inference-time search into the model's generative process. Across question-answering, medical diagnostic, and coding benchmarks, we observe improved diversity, coverage, and set-level calibration scores compared to single answer trained baselines. Models trained with our approach require fewer tokens to generate multiple answers than competing approaches. On coding tasks, they are also substantially more accurate. These results position multi-answer RL as a principled and compute-efficient alternative to inference-time scaling procedures such as best-of-k. Code and more information can be found at this https URL. 

---
# SlopCodeBench: Benchmarking How Coding Agents Degrade Over Long-Horizon Iterative Tasks 

**Authors**: Gabriel Orlanski, Devjeet Roy, Alexander Yun, Changho Shin, Alex Gu, Albert Ge, Dyah Adila, Frederic Sala, Aws Albarghouthi  

**Link**: [PDF](https://arxiv.org/pdf/2603.24755)  

**Abstract**: Software development is iterative, yet agentic coding benchmarks overwhelmingly evaluate single-shot solutions against complete specifications. Code can pass the test suite but become progressively harder to extend. Recent iterative benchmarks attempt to close this gap, but constrain the agent's design decisions too tightly to faithfully measure how code quality shapes future extensions. We introduce SlopCodeBench, a language-agnostic benchmark comprising 20 problems and 93 checkpoints, in which agents repeatedly extend their own prior solutions under evolving specifications that force architectural decisions without prescribing internal structure. We track two trajectory-level quality signals: verbosity, the fraction of redundant or duplicated code, and structural erosion, the share of complexity mass concentrated in high-complexity functions. No agent solves any problem end-to-end across 11 models; the highest checkpoint solve rate is 17.2%. Quality degrades steadily: erosion rises in 80% of trajectories and verbosity in 89.8%. Against 48 open-source Python repositories, agent code is 2.2x more verbose and markedly more eroded. Tracking 20 of those repositories over time shows that human code stays flat, while agent code deteriorates with each iteration. A prompt-intervention study shows that initial quality can be improved, but it does not halt degradation. These results demonstrate that pass-rate benchmarks systematically undermeasure extension robustness, and that current agents lack the design discipline iterative software development demands. 

---
# Training LLMs for Multi-Step Tool Orchestration with Constrained Data Synthesis and Graduated Rewards 

**Authors**: Cheng Jiayang, Xin Liu, Zhihan Zhang, Haoyang Wen, Zixuan Zhang, Qingyu Yin, Shiyang Li, Priyanka Nigam, Bing Yin, Chao Zhang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.24709)  

**Abstract**: Multi-step tool orchestration, where LLMs must invoke multiple dependent APIs in the correct order while propagating intermediate outputs, remains challenging. State-of-the-art models frequently fail on full sequence execution, with parameter value errors accounting for a significant portion of failures. Training models to handle such workflows faces two obstacles: existing environments focus on simple per-turn function calls with simulated data, and binary rewards provide no signal for partial correctness.
We present a framework addressing both challenges. First, we construct a reinforcement learning environment backed by a large-scale cache of real API responses, enabling a data synthesis pipeline that samples valid multi-step orchestration traces with controllable complexity and significantly higher generation efficiency than unconstrained methods. Second, we propose a graduated reward design that decomposes correctness into atomic validity (individual function call correctness at increasing granularity) and orchestration (correct tool sequencing with dependency respect). On ComplexFuncBench, our approach demonstrates substantial improvements in turn accuracy. Ablation studies confirm both reward components are essential: using either alone significantly degrades performance. 

---
# X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs 

**Authors**: Di Cao, Dongjie Fu, Hai Yu, Siqi Zheng, Xu Tan, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.24596)  

**Abstract**: While the shift from cascaded dialogue systems to end-to-end (E2E) speech Large Language Models (LLMs) improves latency and paralinguistic modeling, E2E models often exhibit a significant performance degradation compared to their text-based counterparts. The standard Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) training methods fail to close this gap. To address this, we propose X-OPD, a novel Cross-Modal On-Policy Distillation framework designed to systematically align the capabilities of Speech LLMs to their text-based counterparts. X-OPD enables the Speech LLM to explore its own distribution via on-policy rollouts, where a text-based teacher model evaluates these trajectories and provides token-level feedback, effectively distilling teacher's capabilities into student's multi-modal representations. Extensive experiments across multiple benchmarks demonstrate that X-OPD significantly narrows the gap in complex tasks while preserving the model's inherent capabilities. 

---
