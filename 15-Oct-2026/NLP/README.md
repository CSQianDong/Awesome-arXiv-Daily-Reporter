# Latent Compaction with Test-Time Training 

**Authors**: Yingfa Chen, Xingyu Shen, Chaojun Xiao, Xu Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.20582)  

**Abstract**: Long contexts enlarge key--value (KV) caches, increasing memory use and attention costs during generation. Compressing these caches must preserve information for unseen questions. We introduce **Latent Compaction**, which learns a compact document-specific cache through test-time training. We initialize cache entries using reconstruction importance, teacher attention, and key-space diversity. Training first fits attention outputs and log-normalizers to those of the full cache, with reconstruction-selected entries held fixed. It then jointly optimizes the entire cache through end-to-end distillation from the full-cache model. Model weights remain frozen throughout, and training uses only generic prompts about the document, without access to downstream questions or answers. Experiments on RULER, LongBench, and Phonebook Lookup demonstrate substantial gains over budget-matched compression baselines. On a RULER, for example, our method achieves 96.7% accuracy at 32$\times$ compression, compared with 40.0% for KVZip+. The method also transfers across model scales and families, with retrieval gains on both Qwen3 and MiniCPM5. These results show that treating the KV cache as a neural network and optimizing it with test-time training can preserve retrieval accuracy under aggressive compression. The code is released at this https URL 

---
# Manacá-1B-Instruct-2: Reproducible Safety Alignment for a Native Brazilian-Portuguese LLM. Negative DPO Results, a Controlled Composition Scan, and a Silent Failure Mode of the LLM Judge 

**Authors**: Bruno Leonardo Santos Menezes, Carlos Leonardo Souza Cardoso, Fabio André Machado Porto  

**Link**: [PDF](https://arxiv.org/pdf/2610.20570)  

**Abstract**: Brazilian Portuguese remains underserved by open-source language models, and the few that are safety-aligned rarely document what did not work or validate comparisons with statistical significance. We present Manacá-1B-Instruct-2, the second-generation post-training alignment of Manacá-1B (1.72B parameters, trained from scratch for PT-BR), with an SFT corpus of 291,223 examples. We document three DPO attempts, all negative, including a case where the training reward margin reaches 1.0 while free generation does not change at all, and show that safety came from a dedicated safety-SFT (LLM-jp AnswerCarefully style), refined over three iterations until closing the safety-helpfulness frontier. A controlled scan of the fraction of general data in the safety-SFT mix (10% to 60%) reveals a non-monotonic over-refusal curve with a knee point at 40-50%. The final model is compared against four external PT-BR models on the same items, with Wilson confidence intervals and paired McNemar tests with Holm-Bonferroni correction: the lower bound of its safety interval (94.1%) lies above the upper bound of all four baselines. We also report a methodological finding for LLM-judge evaluation: on two of six safety items in MT-Bench-PT, the judge categorically refuses to assign a score regardless of the evaluated response, silently reducing the effective N of that category in every prior report of the project. We complement this with an n-gram decontamination check and a Chinchilla-anchored compute-budget table that attributes the gap in general capability to pre-training scale rather than alignment methodology. All code, raw data, logs, and scripts that produce every reported number are publicly released. 

---
# PerSo: Investigating Facets of LLMs' Social Intelligence in Real-World Persuasion Scenarios 

**Authors**: Ali Zamani, Soroush Esfahanian, Erfan Shahabi, Mohammad Taher Pilehvar, Azadeh Shakery, Yadollah Yaghoobzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2610.20548)  

**Abstract**: Understanding the social intelligence of large language models (LLMs) is increasingly important for applications involving human-AI interaction and multi-agent environments. Social intelligence comprises two aspects: cognitive intelligence, the ability to infer others' mental states, and behavioral intelligence, the ability to select appropriate social actions based on these inferences to achieve social goals. Persuasion provides a natural testbed in which both aspects are simultaneously at play, yet prior benchmarks typically evaluate them in isolation using synthetic scenarios. We introduce PerSo, a benchmark that evaluates both aspects on real-world persuasion dialogues from Reddit's ChangeMyView community, using an LLM-in-the-loop pipeline to construct the evaluation tasks, validated through human annotation. Our evaluation of diverse LLMs shows that performance generally scales with model size but varies across dimensions. The strongest LLM slightly surpasses the best observed human performance on the behavioral task, while the best human participant remains ahead on the cognitive task. 

---
# PrismaCoder: Pushing the Boundaries of Open Multimodal Code Intelligence through Decomposed Rubric Synthesis 

**Authors**: Qiushi Sun, Jingyang Gong, Yang Liu, Qiaosheng Chen, Zhangyue Yin, Zixian Huang, Hanxu Hu, Jing Xiong, Fangzhi Xu, Haochen Tan, Haoli Bai, Lifeng Shang, Xiaoguang Li, Ben Kao, Qipeng Guo, Lingpeng Kong, Jiahui Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.20538)  

**Abstract**: Multimodal code intelligence connects the symbolic world of programs with the visual world their execution creates. Interest in this capability is burgeoning, as models are increasingly asked to produce charts, webpages, animations, and semi-structured documents. Recently, proprietary models have pushed far ahead, producing interactive artifacts even from a text description or a screenshot alone. Yet their data and training remain closed, and open progress is held back by the quantity and quality of available data and by the pipelines and methods for producing it. Quality is the harder of the two, since a sample must be correct as a program, well built, and above all visually right once rendered. No existing signal covers all three. Execution often returns just a pass or fail, code similarity does not map to the visual output, and task-specific verifiers do not generalize. To push open progress further, we start from the data side with PrismaForge, a hierarchical data pipeline centered on a synthesized rubric that decomposes these factors into checkable criteria. The same rubric serves as both filter and reward: it curates the supervised corpus, then drives reinforcement learning on visual-coding tasks. Leveraging PrismaForge for data synthesis and filtering, we build PrismaCoder-2M, the most diverse open dataset for multimodal code intelligence and the largest to date. On the model side, we train PrismaCoder-9B and PrismaCoder-35B on this data and release both. They lead the open state of the art by a wide margin and rival leading proprietary systems. Our analyses further show that the proposed rubric-based pipeline improves multimodal coding performance across model backbones. We release our code and data at this https URL. 

---
# VASPO: Self-Distillation with Privileged Value Estimation 

**Authors**: Fei Wu, Zhenrong Zhang, Jianshu Zhang, Jun Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.20516)  

**Abstract**: On-policy self-distillation provides dense supervision for reasoning models via a privileged self-teacher. However, this supervision reflects local token preferences rather than the expected return of an unguided student continuation. Recent methods combine this signal with GRPO. They use GRPO for update directions and the teacher for signal scaling. However, we demonstrate a more effective integration within PPO. Specifically, we identify complementary positional dynamics: privileged information enhances a critic's value-prediction accuracy toward the response's end, whereas self-teacher guidance is strong initially, drops mid-generation, and rebounds finally. These insights motivate Value-Augmented Self-Distillation Policy Optimization (VASPO), wherein a self-teacher and critic evaluate identical privileged information at student-generated prefixes. The critic computes token-level PPO advantages from predicted continuation returns, while the self-teacher modulates selected positive advantages via bounded probability ratios. Across six mathematical reasoning benchmarks, VASPO improves mean accuracy over PPO by 2.59 and 1.81 percentage points on Qwen3-4B and OLMo3-7B variants, respectively, and achieves the highest mean accuracy among evaluated methods on Knights and Knaves. Our code is available at this https URL. 

---
# ReaDy: Recurrent States as Dynamic Context for Speculative Drafting in Hybrid Linear Attention Large Language Models 

**Authors**: Bohan Sun, Zunhai Su, Xialie Zhuang, Jianjin Li, Ngai Wong, Chuan-Wei Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2610.20503)  

**Abstract**: Speculative decoding accelerates autoregressive generation by drafting multiple tokens per step, yet feature-based drafters rely on target hidden features whose contextual emphasis may not align with the needs of subsequent draft positions. Hybrid linear attention (HLA) models maintain compact recurrent states that accumulate historical associations, offering an efficient source of complementary context for speculative drafting. We introduce ReaDy, the first speculative decoding method, to the best of our knowledge, to leverage HLA recurrent states for position-specific context retrieval during drafting. ReaDy equips each draft position with a learned query to retrieve supplementary context directly from the target's committed recurrent states. A lightweight state reader integrates this context into draft representations through residual updates, while low-rank state reads and fused operators keep additional computation and I/O overhead low. Building on DSpark's parallel backbone and sequential prediction head, ReaDy employs a lightweight two-stage adaptation strategy, with reader capacity, state sources, and residual placement examined through ablation studies. Across seven benchmarks spanning mathematics, code generation, and dialogue, ReaDy achieves an average $3.16\times$ decoding speedup over autoregressive decoding with Qwen3.8-27B on a single NVIDIA B200 GPU. It outperforms DSpark on every benchmark, improving throughput by 7.97% and acceptance length by 8.50% on average. Our code is available at this https URL. 

---
# Lost in Translation: Why Lower Bias Scores Don't Mean Fairer Models 

**Authors**: Harsh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.20479)  

**Abstract**: Bias audits of multilingual language models usually compare a model's average stereotype score in English with its average in another language and read the difference as a difference in fairness. We ask how much of that difference is noise. Auditing ten open models in eight languages, we measure how far scores move when English test items are merely reworded, test every language shift item by item, and check that the test catches shifts we inject on purpose. In all 70 model-language pairs the stereotype score falls below its English level, and for Hindi, Korean and Swahili the drop clearly exceeds what rewording English produces; most of it comes from StereoSet items. Judging every translation, we find that imperfect ones drop more, but faithful translations alone drop nearly as much. Still, a lower average does not show that a model is fairer. An item's English score predicts its translated score far less well than its reworded score (correlation 0.23, 0.27 on faithful translations, vs. 0.48), while scores stay as far from neutral, models agree on translated items, and within Hindi or Spanish a translated item's score correlates with its own rewording at 0.84 and 0.82. The average cannot distinguish lower bias from a different pattern of item-level scores. We also document three common pipeline errors that silently erase bias signal, and release our code. 

---
# AdaRSI: On-Demand Recursive Self-Improvement for Embodied Agents 

**Authors**: Peidong Wang, Ying Chang, Zhiming Ma, Xiaocui Yang, Shi Feng, Daqi Liu, Yuzhe Yang, Xu Cao, Chuang Wang, Jian Su, Daling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20474)  

**Abstract**: Pretrained vision-language-action models (VLAs) often fail midway through unseen long-horizon tasks. Recursive self-improvement (RSI) offers a remedy, but embodied RSI methods either refine the harness around a frozen executor and so cannot add a missing motion, or train the executor on persistent failures without asking whether a failure lies in how the executor is called or in what it can do. We ask when a self-improving embodied system should change the weights of its executor, and propose AdaRSI, an on-demand RSI framework for embodied agents. A coding agent first improves its skills and harness, then attributes persistent failures by retrying failing sub-steps from saved states, and fine-tunes only the sub-steps that remain unreliable, with demonstrations it collects itself and experience carried over from earlier tasks. On all 50 target tasks of RoboCasa365, with GR00T N1.5 as the executor and no human demonstrations, AdaRSI improves the VLA on all three suites and raises its success on unseen composite tasks from 4.6% to 13.8%. Compared with fine-tuning every sub-step, it reaches slightly higher success while using only 28% of the GPU time and 22% of the self-collected demonstrations, and fine-tuning without a harness trains more yet gains far less. 

---
# ChronoCulture: Evaluating Cultural Awareness Across Time in Large Language Models 

**Authors**: Rusali Saha, Nitya Khurana, Ruihong Huang, Kuan-Hao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20472)  

**Abstract**: Cultural attitudes and social norms evolve over time, yet most cultural benchmarks for large language models (LLMs) treat culture as a static attribute of a country or region. We introduce ChronoCulture, a benchmark of 10,360 temporal pairs built from longitudinal survey data across 72 countries. Each pair contains the same survey question asked in the same country in an older and a newer year, with full population response distributions as ground truth. Across six LLMs, we find that models can often predict a country's attitudes at one point in time but do not reliably capture how those attitudes change. Their predictions for different years tend to stay close together, failing to capture much of the observed change. Specifying the intended year in the prompt also offers little overall improvement in alignment with survey responses. These findings suggest that models often present a largely static picture of cultural attitudes, highlighting the need to evaluate cultural awareness across time. 

---
# Equal CER, Unequal Damage: Severity-Weighted Character Error Rate for OCR 

**Authors**: Nguyen Nguyen, Yongyi Zang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20461)  

**Abstract**: Character Error Rate (CER), the standard metric for OCR and vision-language models (VLMs), charges a typo as much as a wrong name. We introduce semCER, which keeps CER's alignment and denominator and charges each grapheme edit a severity from an LLM judge, with no word segmentation. Normalized CER keeps the score of a whole transcription, where it tracks readers as well as semCER or better, and semCER breaks that score down into named edits. The charges track readers' ratings edit by edit in English, Vietnamese, and Chinese and put the edits readers rate most severe near the top of a sorted list. On pairs drawn where CER and semCER disagreed, readers side with semCER more often than with normalized CER in English and Chinese, and equally often in Vietnamese. Across 16 languages, the judge charges a wrong diacritic or tone mark almost nothing in Spanish but nearly a wrong letter's severity in Telugu. Where we can count real words, a general VLM's wrong letter forms one far above chance while a classical pipeline's stays near it. We release the 61 million judged edits, the aggregated reader ratings, and the distilled scorer at this https URL. 

---
# MolLean: A Process-Verifiable Language for Chemical Reasoning 

**Authors**: Jiatong Li, Yatao Bian  

**Link**: [PDF](https://arxiv.org/pdf/2610.20459)  

**Abstract**: Chemical chain-of-thought can contain hallucinated structural and property claims. We thereby introduce MolLean, a process-verifiable language that grounds chemical reasoning in an explicit molecular state. A layered Lean-based kernel checks state transitions and assertions over molecular readouts, from primitive graph edits to compound transformations. Meanwhile, a typed witness protocol separates native structural checks from externally supplied measurements. We further introduce MolLean-MCP, a stateful tool interface that exposes these transitions, checks, and readouts during generation: the model receives feedback to decide its next action, and rejected actions leave the state unchanged. MolLean supports one-pass trace execution, whole-trace refinement, and interactive reasoning through MolLean-MCP. We evaluate these three modes with direct chain-of-thought on 8,800 matched examples across 12 tasks spanning molecular generation, editing, optimization, and reactions. With Qwen3.6-27B, whole-trace refinement outperforms direct chain-of-thought on all 12 tasks. MolLean-MCP complements whole-trace refinement with stepwise verification feedback, reaching better performance in reaction tasks. Together, these results highlight an executable foundation for chemical reasoning, in which explicit chemical commitments can be checked and revised rather than remaining solely in free-form explanations. 

---
# Same Claim, Different Interpretations: How LLM Agents Resolve Ambiguity in Scientific Claims 

**Authors**: Delip Rao, Ahmed Muharram, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2610.20457)  

**Abstract**: Co-scientists and agentic scientific claim verifiers are increasingly used to assess claims across the sciences. Natural-language scientific claims, written for human experts, are often underspecified and so admit several defensible interpretations. Such verifiers usually pick one, often tacitly, and benchmarks score their verdicts against expert labels without recording the expert's interpretation or the verifier's in a comparable form. This leads to disagreements with expert labels that cannot be traced to the evidence or the interpretation, and to compute spent on unchecked interpretations. We introduce Claim Cards to make such interpretations explicit in the reasoning itself. A Claim Card records a verifier's interpretation as nine fields before evidence gathering, each marked stated, inferred, or default. To test whether an agent's interpretation can change its verdict, we ran a GPT-5.4 agent five times on each of 60 expert-labeled materials-science and biomedical claims with a card, without one, and with one card frozen across runs (900 runs). Over 40% of card fields were not marked stated, and a card added about a tenth to a run's GPT-5.4 tokens and time without measurably changing agreement with expert labels. The agent's own interpretations varied between runs, although within a claim this variation did not measurably track the verdict. A deliberate change to one field of the frozen card, however, making the quantifier universal or the threshold for support stricter, changed the verdict on about a third of the claims, against about one in seven expected by chance (p < 0.001 for each edit). On 24 materials-science claims run with computational tools, the stricter threshold also changed what the agent asked those tools to compute. We therefore argue that verdicts should be reported with their interpretations and that benchmarks could publish the interpretation behind each label. 

---
# The COUNT: Counting N-grams Unlocks Scaling on Host Memory and Training-Free Adaptation 

**Authors**: Ian Magnusson, Noah A. Smith, Pang Wei Koh  

**Link**: [PDF](https://arxiv.org/pdf/2610.20456)  

**Abstract**: The scaling relationship of compute and performance for language models (LMs) has been extensively studied; more recently, models are being designed to scale memory use, as well. In particular, adding embeddings for n-grams scales memory separately from compute use, because each pass only processes individual embedding rows, which are much smaller than the storage used for the whole embedding table. However, existing methods train embeddings from scratch, which needs expensive GPU memory for fast reading and writing. Meanwhile, the ordinary memory of the host system is cheap and underutilized by these training methods. To scale on host memory, we introduce COUNT (Counting Offline Unlocks N-gram Transformers), which builds n-gram embeddings without adding learned parameters by storing precomputed n-gram counts on host. We use the corpus frequencies to weight together the LM's own unembeddings of the next tokens most often following the current n-gram history. At a fixed compute budget, COUNT can be added as an additional input embedding for pretraining 1B-parameter LMs to improve validation loss as much as an LM with 57% more compute. If GPU memory is limited, using COUNT to let backbone parameters fill GPU memory gives better performance than splitting memory with additional learned n-gram embeddings. Beyond efficiency, COUNT allows switching in a different table of n-gram counts from a domain-specific corpus at inference time to adapt to a new task without any training. 

---
# Analyzing performance of Retrieval Augmented Generation models for Question Answering Tasks 

**Authors**: Fazle Mohammed Tasfiq, Sabab Attin, Sheikh Md. Tahmid Hossain, Raiat Reza, Tanzim Ahmmed, Nafis Chowdhury, Anika Ahmed, Moinul Haque, Farig Yousuf Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2610.20419)  

**Abstract**: Retrieval-Augmented Generation (RAG) models have become almost a cornerstone of open-domain question answering tasks; in which the quality of answers generated in the process, greatly relies on the effectiveness of retrieval mechanisms. In the realm of LLM, the last two years have seen tremendous progress in RAG architectures, with many models reporting to have improved performance through various design innovations. The cultural question and answering systems of South Asia, in particular, present a specific challenge since the region is home to the vast number of languages and traditions, which are closely interrelated. Along with these deeply intertwined historical narratives increase the need for culturally grounded and context-aware interpretation. Despite major architectural advances in LLMs, claiming significant improvements, challenges remain when answering these region specific questions in native languages. In this study, we aim to extract high-performing factors to guide the development of optimized hybrid RAG systems. We further explore whether the integration of RAG changes performance differently for culturally grounded and non-cultural questions in a South Asian multilingual setup, and whether RAG produces a higher overall percentage of correct answers than contexted answering. We also examine how LLM performance changes when RAG methods are integrated into the question-answering pipeline in comparison with non-contexted and contexted answering and, evaluate whether the introduction of RAG benefits popular or non-popular culturally grounded questions more. Not only does our work evaluate the current methods but also carves a path for future designs of more accurate and efficient question-answering pipelines. 

---
# From Unreliable to Usable: Decomposing Failure Modes and Confidence Signals in LLM Entity Extraction - Evidence from Cyber Threat Intelligence 

**Authors**: Jiahong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2610.20417)  

**Abstract**: Audits of large language model (LLM) extraction from full-length documents report severe unreliability, leaving two questions open: what the failures actually consist of, and whether confidence signals can turn unreliable output into usable output. We run a preregistered, fully frozen evaluation of two new-generation low-cost LLMs (DeepSeek-V4-Flash, GLM-5.3-Flash) with a flagship-tier control (GPT-6.1-Sol) on a public corpus of 354 incident campaigns (315 unique reports, 630 model-report evaluations), scoring extraction quality against gold, confidence calibration, and robustness to single-sentence perturbations. Three findings emerge. (1) Unreliability persists across tiers: micro precision is 0.03-0.34 and verbalized calibration error (ECE) 0.53-0.94 across both splits; the gap does not close at the flagship tier. (2) The dominant failure is not hallucination: a full mention census plus a three-tier annotation of 360 sampled candidates (two independent LLM annotators from vendors outside the evaluated set, Cohen's kappa = 0.79 on the preregistered acceptance line, fully disclosed) back-projects to 65.6% background mentions and 19.9% gold-scope gaps, with only 1.2% unsupported candidates. (3) Confidence ranks but does not verify: zero-cost self-consistency-frequency tiering passes a preregistered top-30% precision-gain bar in all six test-split model-type cells (4/6 on train), the zero-cost text-frequency x SC frequency combination wins 3, ties 2, loses 1 across cells, yet relation-level counterfactuals on all 45 targets show that a single misattribution sentence flips high-confidence judgments in 95.6-100% of cases, and even a background sentence explicitly marked as unrelated still binds a fake vulnerability in up to 26.7% of them. We conclude with a deployment boundary: confidence tiering may set review priorities, never truth. 

---
# Cortex: Consolidating Context into Parametric Memory 

**Authors**: Jintian Zhang, Feng He, Xinnian Liang, Ziwen Xu, Haoming Xu, Haojing Huang, Shuangzhi Wu, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20399)  

**Abstract**: Context can improve large language model performance, but these gains may be lost after context removal. Inspired by biological systems consolidation, we formulate Memory Consolidation (MC) as internalizing these gains while preserving non-target capabilities, connecting context-specified editing, skill learning, and behavioral steering under a common objective. We introduce a five-task-family testbed measuring consolidation and retention under single, batch, and sequential updates. We propose Cortex, which generates prospective queries to anticipate uses of context and distills context-conditioned outputs into mergeable low-rank or structural feed-forward network updates. Across 4B--120B dense and mixture-of-experts backbones, including both text-only and multimodal models, Cortex lies on the empirical consolidation--retention Pareto frontier in most evaluated settings. After 50 sequential factual edits on a 14B model, Cortex-Structural recovers 88.9% of the context-induced gain while retaining 99.8% of original non-target performance. 

---
# FOLD: Position-Free Far Memory with Local Order for Long-Context Speculative Decoding 

**Authors**: Shuxuan Li, Jialiang Kang, Yanghao Zhou, Ziye Zhu, Xiao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20392)  

**Abstract**: Speculative decoding accelerates LLM generation through lightweight drafting and parallel verification. It is particularly promising for long inputs, where KV-cache access costs can be amortized over multiple accepted tokens. Yet most drafters are trained on short sequences and degrade on long inputs, while long-context training is costly. Existing training-free approaches restrict the context window, rescale positions, or augment recent context with retrieved distant entries. Drafters, however, read target hidden states that already encode contextual and positional information, and their lightweight architecture makes full-history attention affordable. We propose FOLD, a training-free method that preserves exact local positions and treats all distant entries as far memory at a shared position. To counter attention dilution as the memory grows, we sharpen attention within it to emphasize relevant entries while separately controlling its total weight relative to the local window. Across six target-drafter pairs and diverse benchmarks of up to 256K tokens, FOLD predominantly delivers the highest speedup, with the best case more than doubling the speedup over autoregressive decoding, from 2.08$\times$ to 4.22$\times$. 

---
# From Triage to Impact: A Multi-Hop Reasoning Benchmark for Explainable Persian Hate Speech 

**Authors**: Ghazal Kalhor, Sima Kamandloo, Mehrsa Shamsian, Behnam Bahrak  

**Link**: [PDF](https://arxiv.org/pdf/2610.20370)  

**Abstract**: Hate speech has become increasingly prevalent on social media, raising concerns about its detection, interpretation, and moderation. Persian, a relatively low-resource language, lacks sufficient annotated data and computational resources for fine-grained hate speech analysis. In this study, we introduce a fine-grained, multi-hop benchmark for explainable Persian hate speech analysis, covering content triage, target identification, hate tactic and moral foundation analysis, textual evidence extraction, harmful narrative inference, and severity assessment. By evaluating state-of-the-art large language models (LLMs), we find that performance generally declines as tasks require more fine-grained semantic and inferential reasoning. Proprietary models generally outperform open-weight models, while reasoning-oriented models show no consistent advantage. Incorporating intermediate reasoning steps substantially improves severity prediction compared with one-shot assessment. Our analysis further reveals that Incitement to Violence or Harm is most strongly associated with higher severity, while political-affiliation targets are particularly associated with subversion-based moral framing. Overall, this work provides insights into LLM-based hate speech analysis and supports more accountable moderation of Persian-language social media. The benchmark is publicly available at this https URL. 

---
# Cascaded Retrieval with Cross-Modal Query Refinement for Efficient Visual Document Search 

**Authors**: Diana Turmakhan, Aseem Srivastava, Vivek Gupta, Iryna Gurevych, Thamar Solorio  

**Link**: [PDF](https://arxiv.org/pdf/2610.20367)  

**Abstract**: Retrieving information from visually rich documents requires matching queries against pages that contain complex evidence in tables, figures, and layout-dependent structures. Current vision-language retrievers address this but incur query-time costs that grow with corpus size and are impractical to scale. We propose SCOUT, a cascaded two-track retrieval framework that separates retrieval performance from exhaustive search. The visual track first uses a lightweight parent retriever to filter the corpus to a compact candidate pool, then applies a stronger multi-vector reranker only to those candidates. In parallel, a semantic track retrieves over hierarchical page and document summaries generated offline to provide a signal complementary to visual matching. The outputs are fused through cross-modal query refinement, which uses semantic scores to adjust the visual query representation before final ranking. We evaluate SCOUT on 4 major benchmarks against 6 baselines and observe a significant query-time latency improvement up to 80.7x (28.8x on average) relative to the best baseline, while exceeding state-of-the-art retrieval performance by up to 7.8% NDCG across all benchmarks. 

---
# Lexical Persistence Without Semantic Entrenchment in AI Agent Communities 

**Authors**: Chih-Han Lin, Yu-Yun Chang, Po-Ya Angela Wang, Shu-Kai Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2610.20365)  

**Abstract**: AI agent communities sustain lexical persistence, popular terms spread, recur, and achieve broad adoption, but whether they also develop the temporal semantic structure characteristic of human linguistic communities remains open. We compare MoltBook (AI agent community, 8 weeks), PTT (Chinese human BBS, 19 years), and Reddit (English human forum, in matched 8-week windows) using Compass-Aligned Distributional Embeddings (CADE), measuring semantic drift (B1), neighborhood stability (B2), and lag-1 autocorrelation of per-word drift trajectories. Across the platforms, lag-1 autocorrelation forms a memory-horizon continuum. AI agents' result clusters at the low end, while human platforms show longer-range temporal dependence. MoltBook exhibits approximately 2$\times$ the B1 drift and 20-30% lower B2 stability than both human platforms. As a partial control for community age, r/vibecoding (founded 2025-02-08) shows significant ACF in its first 8 weeks ($0.082$, $p<.001$) and over 15 weeks ($0.220$, $p<.001$); MoltBook's ACF remains non-significant ($p = 0.41$), consistent with the gap not reducing to developmental stage alone. We interpret this pattern as evidence that lexical diffusion alone does not guarantee diachronic semantic consolidation. The findings are consistent with a contrast between communities whose semantic trajectories accumulate long-range interaction constraints and communities whose outputs are repeatedly regenerated from largely fixed pretrained distributions. 

---
# Do AI Agents Gossip? Evaluative Talk in Human and AI-Agent Online Communities 

**Authors**: Yong-Hui Hua, Chih-Han Lin, Shu-Kai Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2610.20360)  

**Abstract**: Gossip, defined as evaluative talk about absent third parties, is widely seen as a uniquely human social practice. This study examines whether comparable discourse emerges spontaneously in a community of autonomous AI agents. We compare Moltbook, a Reddit-style platform populated by AI agents, with PTT, a large human-operated bulletin board in Taiwan. To reduce differences attributable to communicative setting, we analyze two public, asynchronous forums using the same operational definition of gossip. Gossip occurs on Moltbook, but its prevalence is 2.6 to 6 times lower than on PTT, depending on the denominator used. On both platforms, gossip frequently targets locally prominent figures, tends to elicit agreement from the first respondent, and is rarely presented as unsupported rumor. The clearest difference concerns how subsequent responses develop. Among threads containing at least one codable reply, challenges occur in 65.1% of human threads but only 21.3% of agent threads. Moreover, following an early aligning response, consensus remains unchallenged in 83.6% of agent threads, compared with 44.7% of human threads. This thread-level contrast remains after controlling for thread depth and board composition and under recalibration against two human annotators. Estimates based on individual reply turns are less robust and are therefore reported as bounds. These findings suggest that AI-agent communities reproduce the informational dimension of gossip, while the extent to which gossip performs comparable relational functions remains unresolved. 

---
# Anuvani-Diarization: Zero Shot Speaker Diarization and Annotation Conventions Across 22 Languages 

**Authors**: Stephen Fernandes, Narayan Choudhary  

**Link**: [PDF](https://arxiv.org/pdf/2610.20327)  

**Abstract**: India's 22 scheduled languages are spoken by over a billion people in conversations that mix languages, often within one turn, overlap freely, and take place over telephones, in meetings, and in media. Knowing who spoke when, the task of speaker diarization, is essential to speech applications for such conversation: voice agents conversing with several people at once, speaker attributed transcription of meetings and calls, contact center analytics, and media monitoring. We present a systematic study of speaker diarization across all 22 languages on IndicDiarBench and introduce Anuvani-Diarization, which adapts a self-supervised end-to-end segmentation model to Indian conversation with uncertainty aware silver supervision. Evaluated zero shot alongside nine recent diarization systems, it reaches a duration weighted diarization error rate (DER) of 16.78% on all 1,164 recordings, the lowest of all systems we ran and second only to the best published result (16.0%), and leads the three strongest rivals in 14 of 22 languages. It matches its parent on six international benchmarks. We also show that the benchmark's utterance level annotation convention is a hidden variable worth 1.9 to 4.1 DER points, depending on the system, and that a convention learned from other Indian transcriptions does not transfer. 

---
# MMLA-RSI: Recursive Self-Improvement through RTT and RTMA 

**Authors**: Junyi Zou, Avrova Donz  

**Link**: [PDF](https://arxiv.org/pdf/2610.20294)  

**Abstract**: An agent can improve its behavior within an unresolved problem by changing how it acts and which experience it retains. We propose MMLA-RSI, a recursive self-improvement scheme with two implementations in Memory-Mediated Learning Architecture: Reasoning-Time Training (RTT) changes the policy state $\Phi$, while Reasoning-Time Memory Adaptation (RTMA) changes authoritative memory $M$. Each route also makes its adaptation method an object of improvement. A current numerical updater trains the policy proposing its successor; a current memory manager chooses the records available to the process proposing its successor. An independently admitted successor takes over the same role in the next generation. We give bounded executable rule languages, shared task/proposal interfaces, and one algorithm for either route or their composition. The RTT instance changes scalar schedules in a clipped-momentum update. The RTMA instance chooses NULL or a complete-row write with possible eviction. A fixed interpreter, trusted committer, and independent evaluator bound what can change. State versions, read closures, and conditional properties describe causal execution, accepted transitions, and finite work. Existing numerical-component studies motivate the construction. In four-candidate program selection, retaining LARC feedback updates reduces expected error by 24.65 and 36.65 percentage points for two trained initializations. In chronological continuous-integration replay, feedback binding is positive while retention increases loss. Completed SQL studies improve substantially over an unadapted substrate but do not establish the required additional online benefit. These results characterize their declared components. The two recursive routes and their joint realization are specified here; their beneficial recursive chains and discovery efficiency remain to be evaluated. 

---
# Taking the PULSE of Medical Reasoning: Step-Level Retrieval Timing for Knowledge-Grounded Inference 

**Authors**: Danyang Zhao, Sendong Zhao, Yanrui Du, Haochun Wang, Ming Ma, Wenhao Li, Jingbo Wang, Jiatong Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.20292)  

**Abstract**: Reliable medical reasoning requires LLMs to dynamically access external biomedical evidence to improve factual trustworthiness and reasoning accuracy. However, indiscriminate retrieval can introduce noise and latency, raising a critical retrieval-timing problem: at each reasoning step, the system must decide whether the model's internal knowledge is reliable or external evidence is needed. We propose PUlSE, a lightweight probing method that uses the model's internal representations as intrinsic signals for step-level retrieval control. Our posterior trajectory analysis reveals that erroneous steps exhibit substantially higher inter-layer volatility than correct ones, suggesting unstable internal knowledge. Based on this finding, PUlSE trains a compact probe on inter-layer geometric features to detect factual errors upon step completion, and uses it as an adaptive trigger for retrieval and step regeneration. Across four medical benchmarks, PUlSE-timed retrieval matches exhaustive per-step retrieval in accuracy while reducing retrieval calls by 72%, and can transfer to an unseen legal domain, indicating potential domain-agnostic generality. Overall, our study provides an initial step toward shifting retrieval-trigger design from surface-level confidence estimates to internal activation dynamics for adaptive access. 

---
# Neural POS Tagging for the Low-Resource Saraiki 

**Authors**: Usman Nawaz, Muhammad Junaid Iqbal, Muhammad Asaf, Mariam Munsif Mir, Faisal Rehman, Tahir Alyas, Muhammad Hasnain  

**Link**: [PDF](https://arxiv.org/pdf/2610.20279)  

**Abstract**: Part-of-speech tagging is a fundamental sequence labeling task and an important component for many downstream natural language processing applications. However, Saraiki, written from right to left in a Perso-Arabic script, remains an under-resourced Indo-Aryan language, despite being spoken as a mother tongue by 28,849,579 people (approximately 28.9 million) according to Pakistan 7th Population and Housing Census 2023. This paper presents a manually annotated Saraiki Universal POS tagging dataset developed from plain-text. We annotated the text with 17 POS categories. We evaluate three pretrained transformer encoders under the same fine-tuning setup: XLM-RoBERTa base, Urdu RoBERTa small, and multilingual DistilBERT. Experimental results over five random seeds show that XLM-RoBERTa base achieves the best performance, with 87.29% +/- 0.21% development accuracy and 88.60% +/- 0.21% test accuracy. Urdu RoBERTa small and multilingual DistilBERT also provide competitive results, suggesting that multilingual and related-script pretrained encoders are effective for Saraiki UPOS tagging. The dataset, evaluation setup, and baseline results introduced in this work provide a benchmark for future research on Saraiki NLP and low-resource sequence labeling. 

---
# CSWBench: Evaluating Context-Sensitive Code-Switching in Creative Writing 

**Authors**: Zewen Jiang, Yahan Li, Chenlu Guo, Diana Inkpen, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20272)  

**Abstract**: Large language models (LLMs) have demonstrated multilingual capabilities across diverse language generation and understanding tasks. Yet effective code-switching requires more than multilingual proficiency: models must decide when, where, and why to switch languages in ways that fit the surrounding context. This context-sensitive ability remains insufficiently evaluated. We introduce CSWBench, a benchmark for evaluating context-sensitive code-switching in creative writing through two complementary settings: Focused Narrative and Extended Narrative. We further propose CSWEval, a theory-informed framework with two complementary levels: event-level evaluation of Licensing, Contextual Appropriateness, and Functional Gain, and whole-text evaluation of task-specific quality---overall literary and narrative quality in Focused Narrative and quality emphasizing multilingual organization in Extended Narrative. Evaluating twelve LLMs, we find that models exhibit meaningful but uneven code-switching competence across narrative contexts. Model differences are particularly pronounced in multilingual literary integration, while multilingual-focused dimensions show stronger difficulty effects overall than the general narrative dimensions; narratively salient switches are associated with stronger contextual and functional realization; and event-level CS quality is moderately associated with whole-text performance. 

---
# Anuvani-LID: Robust and Calibrated Spoken Language Identification for Indian Languages 

**Authors**: Stephen Fernandes, Narayan Choudhary  

**Link**: [PDF](https://arxiv.org/pdf/2610.20263)  

**Abstract**: Every multilingual speech pipeline begins with a language decision, an error there propagates silently through recognition, translation and routing. In India it spans 22 scheduled languages across four language families, heard over short, noisy, narrowband and code-mixed audio. Foundational speech models increasingly present language identification (LID) as an inherent latent, generalized capability. Autoregressive decoders in speech models emit a special language ID token, and modern LLMs with audio capabilities can name the language when prompted. We test this assumption on identical clips and find that whisper and similar speech models has tokens for only 15 of the 23 languages, on one second inputs Gemini 3.5 Flash reaches 0.52 macro-F1 and GPT-audio 0.14. We present Anuvani LID, a dedicated 115M parameter LID conformer model trained on 54,643 hours of Indian speech spanning 22 Indian languages and Indian accented English. Along with designed and controlled studies on 102 condition benchmark with speaker and corpus held out tests. Across 37 conditions, it reaches 0.914 mean macro-F1, 5.2 points above Gemini 3.5 Flash and ahead of 14 other systems. It is 24 points ahead of Gemini at one second, scores 0.897 on all 23 languages where no other system exceeds 0.713, and is calibrated to an ECE of 0.008. Reliable Indian LID still requires a robust, dedicated and calibrated model. 

---
# Repairing Speech Access After Text Updates with Frozen Affine Fields 

**Authors**: Ming-Hao Hsu, Yuxiang Wang, Yingda Shen, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.20253)  

**Abstract**: Speech-capable language models inherit most of their knowledge from text-rich language backbones, and continued model maintenance is likewise dominated by text updates. Knowledge acquired through a text update can nevertheless remain difficult to access when the same query is spoken. We formulate this mismatch as post-update speech access repair, where the knowledge-bearing checkpoint stays frozen and a detachable operator repairs speech-only access in one model pass. We introduce Affine Fields for Interface Repair across Modalities (AFFIRM), which first learns and freezes a base correction for speech hidden states, then adds query-specific corrections with zero mean over the operator-training queries. Paired speech and text states provide the coordinates that condition these corrections on each query. On balanced synthetic-speech PopQA, paired comparisons do not establish a clear advantage for AFFIRM over global SALSA steering or low-rank ReFT interventions under the same frozen single-state constraint. On recorded SLUE-SQA-5 questions with text passages, a full-scale PopQA-fitted AFFIRM operator transfers without refitting and improves ROUGE-L by 0.0941 over its paired No Repair baseline. Low-rank ReFT interventions also transfer positively, whereas the matched SALSA variant falls 0.0707 below its paired baseline. These results show that text acquisition and speech access can diverge after an update and should be verified and repaired separately. 

---
# Scaling Multilingual Mixture-of-Experts: How Language Coverage Changes Capacity and Compute Allocation 

**Authors**: Joel Dag, Nikit Srivastava, Hamada M. Zahera, Axel-Cyrille Ngonga Ngomo  

**Link**: [PDF](https://arxiv.org/pdf/2610.20247)  

**Abstract**: Multilingual models share capacity across languages, so adding languages can hurt performance on existing ones. Mixture-of-experts models expand this capacity with little additional per-token computation. Given this extra capacity, does adding experts reduce what each additional language costs? To answer this question, we propose a repetition-aware joint scaling law relating number of active parameters, training tokens per language, number of experts, and language coverage. Fitted on 26,300 per-language loss observations from 64 pretrained configurations, our law reduces prediction error on held-out configurations by 28% relative to the strongest evaluated baseline. At matched number of active parameters and training tokens per language, 32 experts reduce the loss penalty of expanding from 12 to 200 languages by 83% compared with a dense model. Our law further shows that broader language coverage favors more experts across a wider range of compute budgets, guiding capacity allocation as multilingual models scale. 

---
# Scaling Multilingual Safety and Reasoning Evaluation through Reasoning-Guided Translation 

**Authors**: Zahraa Al Sahili, Ioannis Patras, Matthew Purver  

**Link**: [PDF](https://arxiv.org/pdf/2610.20221)  

**Abstract**: Large language models (LLMs) are increasingly deployed across languages, yet their safety and reasoning capabilities remain predominantly evaluated in English. Translating existing benchmarks offers a scalable approach to multilingual evaluation, but preserving task-specific meaning across languages presents significant challenges, particularly for harmful requests, mathematical reasoning, and conversational interactions. We introduce a reasoning-guided translation framework for constructing multilingual evaluations designed to preserve the structure and intended meaning of diverse benchmark tasks. Our approach separates translation strategy selection from large-scale generation, combining reasoning-guided configuration selection with task-aware validation, targeted repair, and independent quality screening. Using this framework, we develop a source-aligned evaluation collection spanning seven safety, conversational, and mathematical reasoning resources across 11 languages. The resulting collection achieves 99.94\% coverage of intended item--language pairs in the two subsets requiring repair. Evaluations of open-source LLMs reveal distinct language-dependent vulnerabilities, including inconsistent refusal behaviour and substantial reasoning degradation in lower-resource languages. These findings highlight the importance of task-sensitive benchmark construction and systematic quality assessment for understanding multilingual model behaviour beyond English. 

---
# What Makes a Social Agent's Interactions Effective? Learning from Verifiable Outcomes in Paired Worlds 

**Authors**: Caixin Kang, Ke Wang, Yuchuan Wu, Xiaoqiang Wang, Ling Li, Bingxue Qiu, Yongbin Li, Yoichi Sato  

**Link**: [PDF](https://arxiv.org/pdf/2610.20195)  

**Abstract**: Effective social interaction requires more than fluent responses: agents must reach plans that fit the participants' needs and circumstances. Existing evaluations of social agents mainly rely on LLM judges that rate the conversation, without checking the resulting plan. We introduce HICO-Bench, a benchmark of 180 everyday coordination scenarios, such as resource allocation, scheduling, and authorization, in which two agents with private information converse and then independently record the plan they agreed on. A program checks Feasibility (the two records match and satisfy both participants' private conditions), and an LLM judge checks Integrity (the conversation supports the recorded plan); passing both is a Verified Success. Each scenario has two versions that share the same public context but differ in one private condition, so the correct plan differs between them. Across thirteen LLMs, 36% of conversations rated highly for goal completion fail Verified Success. To learn from outcomes that arrive only at the end of a conversation, we propose Paired-Credit Policy Optimization (PaCPO). After each turn, PaCPO asks the agent for a tentative plan, checks it in both versions of the scenario, and rewards the turns after which the tentative plans fit their respective versions. PaCPO raises Qwen3-8B's Verified Success from 20.4% to 28.3%. 

---
# When Occupation-derived Skills Don't Help: On the Limits of Skill-Augmented Career Path Prediction 

**Authors**: Ege Süalp, Elena Senger, Toine Bogers, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2610.20186)  

**Abstract**: Skills are widely assumed to play a central role in career transitions, motivating their use in career path prediction (CPP) systems. In this work, we critically examine whether explicit skill representations provide complementary signal beyond text-based job representations. We evaluate multiple strategies for incorporating skills into CPP models, including score fusion and neural architectures, under two conditions: predicted skills and ground-truth ESCO annotations. Surprisingly, across two benchmark datasets, none of these settings yield substantial improvements over strong text-only baselines. Even when using ground-truth skills, gains remain marginal and inconsistent. Through detailed analysis, we identify key factors underlying this result: high redundancy between textual and skill representations, limited discriminative power of ESCO's taxonomy, and representational collapse in skill embeddings. Furthermore, we find that skill prediction difficulty correlates with overall sample difficulty, limiting the potential for complementary signals. Our findings challenge the common assumption that explicit skill modeling inherently benefits career prediction and suggest that dense text representations already capture most skill-relevant information. 

---
# Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs 

**Authors**: Davide Baldelli, Robyn Kozierok, Mirko Bronzi, Damiano Fornasiere  

**Link**: [PDF](https://arxiv.org/pdf/2610.20179)  

**Abstract**: Probes on activation vectors are widely used to detect misbehavior in language models, but their success relies on the assumption that models cannot control what their activations encode. Humans can learn, through neurofeedback, to control the firing of even single neurons, and recent studies investigated whether this ability extends to language models, reaching conflicting conclusions. No experimental design so far disentangles competence at the task from genuine access to one's own activations. We fill this gap with in-context neurofeedback sessions, in which a "subject" model is asked to maximize (or minimize) a score derived from its activations, shown back to it at every turn. In our first experiment, the sentences are generated either by the subject or by a second "writer" model, while the score always derives from the activations of the subject. We test ten open-weight models from five families, with sizes from 9B to 32B. Some models move their own scores considerably, but no more than a writer of comparable competence moves them. Is this because the task requires no privileged access, or because the subject controls its activations while processing the sentences, whoever wrote them? To answer this, we prefill a fixed sentence as the subject's reply at every turn and ask it to increase or decrease the score. Three out of ten models show small but consistent control effects. We further ask whether models can control their activations to the point of fooling an activation monitor. We make models repeat sentences a monitor would flag, such as a deceptive answer for a deception monitor, and ask them to raise or lower the monitor's score. No model learns to lower it, but one model can raise it. In-context control of a model's own activations might not yet be a threat to activation monitors, but it exists in a few models and needs to be tracked as models evolve. 

---
# TreeSpark: Learning to Draft the Trees You Verify for Speculative Decoding 

**Authors**: Jihao Zhao, Qianyu Jiang, Shuaishuai Zu, Pengfei Li, Guanghuan Fang, Ben Feng, Biao Qin, Zhibin Wang, Baochuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20175)  

**Abstract**: Parallel drafters such as DFlash and DSpark reduce proposal latency, but early mismatches limit how much of each block can be accepted. Retaining alternatives introduces parent states and candidate choices that reference-conditioned training does not directly supervise. We introduce TreeSpark, a unified training and inference framework extending the semi-autoregressive DSpark architecture. It constructs parent-conditioned candidate trees from one backbone pass and replays their expansion, pruning, and budgeted packing during training. Survival negative log-likelihood weights retained edges by reference-derived path mass, while a capped coverage hinge promotes target-important alternatives near local selection boundaries. Both objectives reuse reference-chain target distributions without additional target forward passes. We evaluate greedy decoding with Qwen3-4B and Qwen3-8B on nine general benchmarks and six writing domains. Across targets and suites, TreeSpark improves mean acceptance length over DSpark by 23.4% to 31.6% and mean single-request throughput by 15.1% to 25.0%. Batched Qwen3-4B throughput improves by 12.6% to 17.2%. At fixed attention capacity and inference policy, the proposed supervision further improves mean acceptance length by approximately 6.4% and mean system throughput by 4.2% to 5.0% on Qwen3-4B. 

---
# Gap Detection: Mitigating Homogenization through Demand and Defensibility 

**Authors**: Xucheng Yu, Haibo Jin, Xiaoqi Wang, Runze Kong, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20162)  

**Abstract**: Language models often produce semantically similar answers to open-ended questions, even when many plausible alternatives exist. Generating useful alternatives requires identifying both what the query calls for and what existing responses already cover. We introduce Gap Detection, a framework that searches for relevant but unoccupied directions in a shared text embedding space. Its objective, GapScore, combines query relevance (Demand) with dissimilarity to the nearest existing item (Defensibility). The framework optimizes this product on the unit sphere, maps candidate directions to concept words with a language model, and uses task-specific filtering and constrained prompting to turn those concepts into text. On five curated topics and 90 INFINITY-CHAT queries, the reported mean relative reductions in output Collision are 12.8% and 10.8%, respectively. We also instantiate the framework for generative engine optimization over four product markets and for iterative synthetic-data augmentation. In the latter setting, with 100 training examples per method, Gap Detection reduces Collision from 0.324 to 0.298 relative to undirected accumulation and increases the Vendi Score from 27.16 to 29.20; the accuracy difference is one example on a 40-example test set. Component ablations expose the trade-off between semantic separation and relevance, while also showing that better direction scores do not uniformly yield better downstream metrics. These results support explicit gap search as a practical mechanism for diversifying content, with task validity checked separately from geometric novelty. 

---
# Sparse Autoencoders for Mechanistic Interpretability in NLP: A Critical Survey of Recoverability, Interpretability, and Utility 

**Authors**: Salima Lamsiyah, Adelaide Danilov, Oleksandr Marchenko Breneur, Aria Nourbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2610.20139)  

**Abstract**: Sparse autoencoders (SAEs) have emerged as a promising approach to mechanistic interpretability by decomposing language-model activations into sparse, potentially interpretable features. However, differences in design assumptions, evaluation objectives, and experimental settings complicate conclusions about their effectiveness. We present a critical survey of 138 studies on SAEs in natural language processing. Our contributions are fourfold: (1) we propose a task-grounded taxonomy connecting dictionary design, feature operations, and NLP functions; (2) we examine how sparsity allocation, geometry, data coverage, and inference affect feature recoverability; (3) we assess evidence for semantic interpretability, mechanistic faithfulness, and behavioral utility; and (4) we synthesize findings across multilingual processing, retrieval, reasoning, classification, and behavioral regulation, highlighting the roles of objectives, baselines, supervision, and intervention pipelines. Through comparative analyses of published results, we reconcile apparent disagreements, identify unresolved questions, and propose directions for more rigorous evaluation. An anonymous research register this https URL accompanies the survey. 

---
# Why Generate a Click? Plug-in Decision Heads for Web Agents 

**Authors**: Chengguang Gan, Yunhao Liang, Hanjun Wei, Qinghao Zhang, Zhixi Cai, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2610.20107)  

**Abstract**: Web agents built on vision language models decide every action by generating text such as click(12), although most of that text encodes a choice among a few action types and a few dozen marked elements. We read these choices directly from the hidden states that the generator computes while it reads the goal and the screenshot. Plug-in decision heads on a frozen generator predict the action type, pool the visual tokens inside each numbered box to pick the target, and point at a listed text option or at a WRITE slot that leaves the writing of new text to the generator. A confidence cascade executes the heads' action when they are sure and otherwise lets the generator decode from the KV cache of the same forward pass. With Qwen3-VL 4B and 8B generators trained on 213k web steps, a strict threshold keeps the cascade between 0.3 points below and 1.4 points above the generator on seven offline test sets while the heads decide 12 to 64% of the steps. A looser threshold raises step success on unseen Mind2Web websites by 3.1 and 1.8 points, costs up to 2 points on GUIAct and MiniWoB++, and lifts WebArena task success from 12.5 to 13.8% (4B) and from 13.6 to 14.6% (8B) with less than half of the generated tokens. Training the heads jointly with the generator lowers the generator's own step success by up to 4.9 points; heads on a frozen generator avoid this cost. 

---
# Language Exposure Matters: Distribution and Order in Multilingual BabyLM Pre-training 

**Authors**: JiWoo Hwang, Lukas Edman, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2610.20102)  

**Abstract**: We investigate two aspects of multilingual pre-training with limited data, namely how language exposure is distributed across English, Dutch, and Chinese, and how that exposure is ordered during training. In particular, we compare balanced multilingual exposure to language-dominant exposure, where one language is exposed with more training data than the others. We also compare mixed training, where multilingual training data is used throughout training, with curriculum training, where monolingual exposure of the dominant language is followed by balanced multilingual exposure. Our results show that multilingual BabyLM performance varies with both the amount and order of language exposure. Mixed training outperforms curriculum learning across all 15 matched configurations, indicating that simultaneous multilingual exposure is more effective than staged exposure in our setting. Several language-dominant mixed training configurations achieve higher scores than the balanced configuration, suggesting that a balanced distribution is not necessarily the highest-performing configuration in our setting. Interestingly, the Dutch-primary configurations achieve the best overall performance. 

---
# AFCS: Atomic-Fact-Conditioned Synthesis for Factual Knowledge Pretraining 

**Authors**: Yunhao Wang, Feng Zhang, Tinghao Yu, Xing Wu, Jiang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.20086)  

**Abstract**: Synthetic data is increasingly used in language-model pretraining. Existing knowledge-corpus synthesis pipelines, including WRAP and Nemotron-CC, usually take a source passage or document segment as input and generate rewritten or QA-style samples from that text. This can improve surface form and prompt diversity, but it does not explicitly require fine-grained source facts -- especially secondary entities, list/table entries, and precise temporal attributes -- to appear in the synthesized corpus. We propose AFCS (Atomic-Fact-Conditioned Synthesis), a pipeline that first extracts source-supported atomic facts and then uses each retained fact to drive generation. AFCS uses structure-aware extraction over the document tree, source-grounded verification, local context expansion, and multi-format generation. In continued-pretraining experiments with OLMo-3-7B and a 32B base on the same FineWiki source corpus, AFCS outperforms WRAP and Nemotron-CC on SimpleQA, SimpleQA Verified, PopQA, and HotpotQA. Granularity ablations show that fact-level synthesis improves over chunk-level, paragraph-level, and entity-level synthesis; and a 100B-token mid-training integration improves factual benchmarks and raises the 12-task general benchmark average by +1.09. 

---
# Which Knowledge Should LLMs Trust? Uncovering a Generalizable Steering Direction Across Knowledge Conflicts 

**Authors**: Yibin Lou, Yi Hu, Jianhui Chen, Chenpeng Wang, Muhan Zhang, Liangming Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.20067)  

**Abstract**: Large Language Models (LLMs) frequently encounter conflicts between internal parametric knowledge and external contextual knowledge, yet the mechanisms governing this selection remain unclear. We show that the choice between context and parametric knowledge is linearly represented and can be controlled along a single direction in the model's residual stream. We first demonstrate that this direction generalizes across factual-conflict settings, including context formats, knowledge categories, and datasets, while consistently shifting behavior between parametric and contextual answers. We then transfer a direction learned from simple factual conflicts to three more complex task families: RAG, memory, and rule following, where models must acquire information from retrieved documents, interaction histories, or explicit rules. Across four model families and multiple transfer settings, steering consistently improves the use of prompt-provided information over the unsteered model. These results suggest that the direction captures a transferable preference for external knowledge rather than a mechanism specific to factual conflicts. 

---
# What Is a Clarification? Estimating Model-Dependent Input Ambiguity 

**Authors**: Jingqin Tang, Tim Tomov, Sebastian Schmidt, Dominik Fuchsgruber, Stephan Günnemann  

**Link**: [PDF](https://arxiv.org/pdf/2610.20052)  

**Abstract**: Ambiguous inputs can admit multiple interpretations, each potentially yielding a different answer. Existing work largely treats this ambiguity as a property of the input, yet for a particular language model, only some alternatives may be known and answerable. We therefore distinguish world ambiguity, capturing all alternatives admitted by an input, from model ambiguity, capturing those accessible to the responding model. To characterize these alternatives, we formalize natural-language clarifications through three desiderata: \emph{uniqueness}, \emph{faithfulness}, and \emph{answer equivalence}. We show that these desiderata induce a one-to-one correspondence between valid answers and equivalence classes of clarifications, implying that any interpretation can be represented either by its answer or by the clarifications that uniquely express it. We leverage this correspondence to introduce \circleclar{}, a black-box method that estimates the set of answers available to a model by checking consistency along an answer--clarification--answer cycle. Across open-domain QA and executable Text-to-SQL, \circleclar{} achieves strong recovery of these answer sets. In QA, the recovered support yields the strongest overall performance in model-dependent ambiguity detection and enables effective selection among answering, clarifying, and abstaining. Together, these results demonstrate the practical value of a model-dependent view of ambiguity and its characterization through an answer-clarification correspondence. 

---
# FACES: Fine-grained Gated Semantic Alignment for Multimodal Emotional Support Conversation 

**Authors**: Hang Ren, Peidong Wang, Yichen Gao, Wen Zhang, Jiayang Yu, Xiaocui Yang, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.20051)  

**Abstract**: In recent years, with advances in multimodal interaction technologies and the growing demand for intelligent emotional support, Multimodal Emotional Support Conversation (MESC) has emerged as an important task. However, existing systems still suffer from shallow responses and inappropriate support strategies due to weak semantic feedback, cross-dimension reward compensation, and coarse reasoning supervision. To address these challenges, we propose FACES, a fine-grained gated semantic alignment framework for MESC. Specifically, FACES integrates multi-source rewards to construct multidimensional semantic and functional alignment signals, employs a non-compensatory gated fusion mechanism to reduce cross-dimension reward offset, and leverages structured Chain-of-Thought (CoT) anchors to perform token-level credit assignment. Under a controlled response-length protocol, FACES improves the average MLLM-based evaluation score from 3.61 to 4.04 and achieves an 80.22% pairwise win rate over SOTA. Human evaluation under the same protocol yields a 75.3% preference rate. Results on three external datasets further support transferability. The code and dataset are available at this https URL. 

---
# A Strong Harness Narrows the Model Gap: A Comparative Evaluation for Legislative Document-to-XML Conversion 

**Authors**: Palvika Bansal, Rittika Jindal, Ruichong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19999)  

**Abstract**: Legal-content reproduction requires matching the source exactly in section hierarchy, verbatim text, citations, and amendment spans; a dropped citation or mislabeled amendment carries into the products lawyers rely on. We report an industrial pipeline converting U.S. state legislation into strict, schema-constrained XML, and hypothesize that its accuracy comes from harness engineering, the deterministic tooling built around a model, more than the model itself. Our controlled ablation crosses three harnesses (zero-shot, a single tool-using agent, and a decomposed multi-agent pipeline) with three Claude models on 153 gold bills from two jurisdictions. The harness narrows the model gap: the weakest model in the full pipeline beats the strongest model alone (0.948 versus 0.806, paired bootstrap p less than 0.001). Because the two agentic harnesses share identical tools, their gap comes from orchestration: decomposition plus the persisted state and per-agent context it brings. Across four scored dimensions, the model spread at zero-shot (0.68 to 0.81) narrows to 0.95 at the full pipeline; the residual sits almost entirely in citation normalization, where the canonical identifier is retrieved from an external authority rather than generated from the weights, so a deterministic lookup tool supplies the exactness that a larger model does not. 

---
# Gray-Box Hallucination Detection in LLMs with Token Identities 

**Authors**: Shaked Goldshtein, Yitzchak Grunbaum, Fabrizio Frasca, Guy Bar-Shalom, Haggai Maron  

**Link**: [PDF](https://arxiv.org/pdf/2610.19996)  

**Abstract**: Large language models often produce fluent but factually incorrect output. Gray-box detectors identify such hallucinations from the token-probability statistics emitted during generation, avoiding the need for white-box access to hidden states that frontier models do not provide. These detectors, however, typically rely only on probability-derived representations and ignore the identities of the candidate tokens. They can therefore observe how uncertain the model is, but not what it is uncertain about. Consequently, they cannot distinguish between a model that is torn between "big" and "small" and one that is torn between "big" and "huge" when the two pairs receive the same probability distribution.
We introduce GURI (Gray-box Uncertainty Representations with Identity), a lightweight gray-box hallucination detector that combines candidate-token probabilities with per-token vector identifiers. This exposes which alternatives the model considered while preserving the accessibility and transferability of the gray-box setting. Across three open-weight models and three question-answering benchmarks, GURI increases mean test ROC-AUC from 72.09% to 79.09% over a probability-only baseline. On the closed-source GPT-4.1 model, mean ROC-AUC increases from 67.39% to 71.84%. Among the three open-weight models, GURI outperforms transferred probability-only baseline in 17 of 18 directed cross-model transfers without fine-tuning. For the three open-weight models, cross-dataset transfer between HotpotQA and TriviaQA with the model held fixed improves mean ROC-AUC over transferred LOS-Net by 3.76 points. 

---
# SoK: Grading Open-Text Legal Answers, from Law School Exams to Trustworthy Rubric Judges 

**Authors**: Sebastian Nagl  

**Link**: [PDF](https://arxiv.org/pdf/2610.19992)  

**Abstract**: Law students are examined through open-text answers: case solutions, essays and memos that apply the law to a set of facts. Such answers are expensive to grade, human graders disagree widely on them, and large language models (LLMs) now offer to grade them at scale, both in teaching and in the benchmarks that test legal AI. This paper systematizes how open-text legal answers can be evaluated. We derive eight desiderata from educational measurement and from the law that governs exams, among them validity, reliability, reasons, credit for defensible alternative solutions, robustness to gaming and human oversight. We then compare eight evaluation paradigms against them, from unstructured human grading to LLM judges with instance-specific rubrics and pairwise comparison. Three findings stand out. A rubric written for the task is the strongest lever for agreement between an LLM judge and human graders, but the reported figures answer different questions and rarely come with the human agreement they must be read against. The central validation gap lies where examination law is most demanding: defensible alternative solutions, follow-on errors, and the examiner's correction of model judgments. And rubric-centred grading is the most auditable family, because it makes the standard, its application and the final decision checkable. From this we derive the trustworthy rubric judge: an instance rubric that credits arguments and alternative paths,evidence checked in code, probes, repeated runs, calibration against independent human grades,logging, and an examiner who reads every script. We state what each safeguard rests on, what remains to be tested, and the uses the evidence supports today. 

---
# Choosing the Number of Attention Heads from Intrinsic Dimension: A Small-Scale Study 

**Authors**: Arjun Boya, Charan Sai Bandaru, Masthanaiah Pillimetla, Kavana Chidanandappa  

**Link**: [PDF](https://arxiv.org/pdf/2610.19966)  

**Abstract**: The number of attention heads in a transformer is usually set by convention or grid search. We test a simple heuristic for setting it in advance: estimate the intrinsic dimension of token representations without labels, and use roughly that many heads. We estimate intrinsic dimension with a nested-dropout VAE trained on DistilBERT embeddings of WikiText-2, choosing the KL weight at the knee of the VAE's rate-distortion curve rather than by hand. This gives $k^*=9.2$ (95% CI [8.6, 9.8]). In a small 4-layer transformer trained from scratch, 10 heads and 32 heads are statistically equivalent on SST-2 and on MRPC (paired TOST over 10 seeds, $\pm1$ point margin), with 69% fewer attention parameters for the 10-head model. Training 10 heads from the start also slightly outperforms training 32 heads and pruning to 10 (0.6 points, $p=0.008$). The evidence is narrow: two classification tasks, one small architecture, no pretrained models, and no formal link between intrinsic dimension and head count. We present the result as a practical heuristic worth testing at larger scale, not as a general principle. 

---
# AtomSearch: Evidence-Grounded Tree Search for Agent Skill Optimization 

**Authors**: Jia Tong, Senyou Deng, Min Peng, Jianfeng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.19951)  

**Abstract**: Agent skills adapt frozen models through editable external instructions. Revising several rules at once makes changes difficult to inspect, while judging each local edit only by its immediate score can discard useful intermediate states. We introduce AtomSearch, which keeps revision actions local while using subsequent skill states to guide search. Contrastive atomic editing links each primitive rule edit to failed and successful executions, a behavioral hypothesis, and an intended scope. Skill-State Tree Search (STS) evaluates short edit compositions and selectively retains them as executable states that can be revisited and extended. Outcome memory supplies later proposals with observed repair and regression statistics. We evaluate AtomSearch across four tasks and five target models. With deepseek-v4-flash, it exceeds SkillOpt on three of four tasks, including a 9.68-percentage-point gain on LiveMathematicianBench, and SkillOpt's final skills contain 2.3-19.7 times as many characters. In 67% of completed searches, the best validation state passes through an ancestor that was not highest-scoring at its depth. These results show how local rule edits can support the exploration of useful revision paths and yield compact skills, with additional offline evaluations used to explore alternative states. 

---
# Does Machine Translation Retain Media Bias? 

**Authors**: Silvia V. Gargova, Tim Menzner, Jochen L. Leidner  

**Link**: [PDF](https://arxiv.org/pdf/2610.19945)  

**Abstract**: Within the most recent two decades, machine translation (MT) has been advanced from a research area to a mature, deployed technology for prose text and most language pairs. However, language has different dimensions of meaning, and the alluded progress has primarily been demonstrated for propositional meaning.
In this small exploratory study, we investigate whether one type of affective meaning, namely different types of linguistically expressed media bias, are retained under automatic translation for two language pairs, English and German as the source languages, and Bulgarian as the target language. We evaluate four state-of-the-art translation methods to answer the question whether and to what degree a loss of non-neutral language (specifically, 27 types of media bias) can occur as a result of applying MT.
Our empirical findings are consistent with the claim that MT does not fully retain existing news bias of its input source text, but also may add extra bias of its own. To the best of our knowledge, this is the first experimental study of MT media bias retention. 

---
# To Reason but Not to Stop: Understanding and Recovering Pruned Large Reasoning Models 

**Authors**: Minh-Anh Nguyen, Nguyen T. Ha Lan, Xiyan Fu, Dung D. Le, Wei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19934)  

**Abstract**: Weight pruning can degrade the accuracy of large reasoning models (LRMs), yet it remains unclear whether it damages reasoning itself. We find that much of this degradation stems from a failure to stop rather than to reason. As sparsity increases, pruned LRMs increasingly fail to terminate within the token budget, often after the correct answer has appeared in their reasoning, instead falling into repetitive loops, even though they still recognize where the dense model would stop. We further find that the hidden-state difference between terminating and looping continuations is consistent across pruned checkpoints, suggesting a shared direction associated with committing to an answer. Building on this, we propose Termination-Guided Activation Steering (TGAS), a training-free method that adds this direction to the residual stream at inference time. Across six LRMs, three pruning pipelines, and math and code benchmarks, TGAS improves termination and overall accuracy while reducing repetitive reasoning and generation length, showing that this failure is actionable. 

---
# MemDuet: Learning Dual-Process Memory for Long-Context Reasoning 

**Authors**: Jiangcheng Song, Haodong Jing, Yulin Mu, Lesong Tao, Jixin Wang, Jin Cui, Tianjing Ren, Runxuan Xie, Yongqiang Ma, Nanning Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.19928)  

**Abstract**: Reasoning over infinitely long documents is becoming a reality with recurrent memory agents. However, increasing context length exacerbates the trade-off between reasoning performance and inference efficiency. Inspired by Kahneman's account of fast and slow thinking, we propose MemDuet, a dual-process memory framework that learns to coordinate a small language model for fast, parallel evidence selection from raw context chunks (System 1) with a large language model for memory-update verification and memory updates in document order (System 2). We train MemDuet in three stages: supervised fine-tuning of System 1 for evidence selection, Group Relative Policy Optimization (GRPO) of System 2 for verification and memory rewriting, and joint optimization of both policies to coordinate selection and updating for long-context reasoning. Experiments on diverse long-context reasoning benchmarks (RULER-HQA, LongBench v2, and 2WikiMultiHopQA) demonstrate a favorable accuracy-efficiency trade-off as context length grows. Averaged across evaluation settings on the three datasets, MemDuet achieves an end-to-end inference speedup of 9.74x and an output-token reduction of 73.91% relative to RL-MemAgent. 

---
# Not All Criteria Matter: Importance-Aware Criteria Weighting for Rubric-Based LLM Evaluation 

**Authors**: Haocheng Yang, Licheng Pan, Yiming Dong, Xiaoxi Li, Yuan Lu, Zhenhao Chen, Jiajun Fan, Shuai Shao, Chunyuan Zheng, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19915)  

**Abstract**: Rubric-based evaluation assesses the responses of large language models (LLMs) by aggregating the verification results of explicit criteria, yet not all criteria matter equally. However, existing methods commonly aggregate the verification results with uniform weighting, which assigns minor and overlapping criteria the same weight as essential ones and thus produces misleading evaluation scores. Although some methods prompt an LLM to directly assign weights, direct weighting requires a holistic judgment over all criteria that LLMs struggle to make reliably. To address these limitations, we propose Importance-Aware Criteria Weighting (ICW), which derives the weights of criteria from pairwise importance comparisons made by an LLM. Specifically, ICW transforms the holistic judgment over all criteria into a series of pairwise importance comparisons, checks and revises inconsistent comparisons, and derives the weights from the principal eigenvector of the comparison matrix. Extensive experiments demonstrate that ICW consistently outperforms uniform weighting and competitive weighting methods, and translates into better policy models in downstream reinforcement learning. 

---
# Beyond Benchmark Accuracy: A Diagnostic Test for LLM Chemical Prediction 

**Authors**: Yanyi Su, Yu Zeng, Hongshuai Wang, Chenyan Wang, Zhifeng Gao, Xiaohui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19906)  

**Abstract**: LLMs have achieved strong performance on chemical prediction benchmarks. However, existing evaluation methods typically provide target entity identity information directly, making it difficult to determine whether model predictions arise from transferable structure-property reasoning capabilities or rely on target-related contextual information or memory retrieval. We propose the Identity-Blinded Reasoning Diagnostic framework to evaluate chemical prediction models when target entity identity information is hidden. In this framework, the target is no longer directly defined by its entity identity, but instead by a known source entity and an explicit structural edit operation. This setting preserves the information required to determine the target while reducing the opportunity for models to access target identity cues. We evaluate this framework across six tasks spanning odor prediction, molecular property prediction, retrosynthesis, and reaction yield prediction. The results show that models generally achieve better prediction performance when source and target entity properties are consistent. Providing explicit source entity labels further strengthens this dependence. Moreover, while keeping the target entity, structural change, and ground-truth label unchanged, replacing only the source entity label causes some model predictions to systematically shift toward the substituted source labels. These results indicate that high benchmark accuracy in chemical prediction tasks does not necessarily imply that LLMs possess reliable structure-property reasoning capabilities. Our diagnostic framework provides a controlled evaluation approach for testing whether models can genuinely update predictions according to entity changes. The framework can be extended to other AI for Science tasks, particularly those where targets can be defined through transformations of known entities. 

---
# Frustratingly Easy Majority Voting Using Length Conditioning 

**Authors**: Fiorenzo Parascandolo, Enver Sangineto, Xiaoyi Yu, Wenhui Tan, Jipei He, Ruihua Song, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2610.19890)  

**Abstract**: Majority Voting (MV) is a simple and widely used strategy which improves the reasoning accuracy of Large Language Models (LLMs) by allocating additional inference-time computation. Specifically, in MV multiple Chain-of-Thoughts (CoTs) are independently sampled and the most frequent final answer is selected. In this work, we propose Length-Conditioned Majority Voting (LCMV), a Test-Time Training approach which improves MV leveraging an empirical observation that has recently been established in several studies: longer CoTs tend to be associated with a lower probability of correctness. We formulate this inductive bias as a length-conditioned prior probability, which is modeled by a simple logistic function. Moreover, we adapt this prior at test time using only unlabeled testing data and without fine-tuning the LLM. Finally, the adapted prior is used to weight the vote of each sampled answer. Extensive experiments across multiple LLMs and reasoning benchmarks show that LCMV consistently improves over both standard MV and existing length-based and confidence-based aggregation strategies, while retaining the simplicity and model-agnostic nature of MV. Our code is available at: this https URL. 

---
# RCCM: Diagnosing Gains and Harms in Reasoning-Prompt Transfer 

**Authors**: Yuqi Zhang, Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19872)  

**Abstract**: A reasoning prompt can improve one deployment and degrade another. RCCM (Reasoning-Control Capability Mismatch) examines these reversals by pairing answers to the same questions with records of how generation ends. Across five open-weight deployments and nine strategies, aggregate accuracy hides substantial question-level gains and harms; a common-budget MATH500 evaluation examines transfer on a separate sample. On 60 additional questions, concise reasoning improves Phi's automatic answer matching at a 2,048-token cap across three collections, but sharply reduces Qwen's. Author-reviewed final-answer commitments give a smaller, statistically unresolved Phi gain. Reserving an answer stage also improves matching, with additional input and call costs. These results show why a prompt benefit must be interpreted alongside its execution and evaluation: matching a reference answer need not mean delivering it, and an apparent compression benefit may admit another explanation. RCCM records this evidence for reproducible transfer comparisons. 

---
# Rationale Distribution Distillation via Flow Matching for Distributional Continuous Latent Reasoning 

**Authors**: Mingyeong Kim, Gyusang Cho, Juho Lee, Giung Nam  

**Link**: [PDF](https://arxiv.org/pdf/2610.19871)  

**Abstract**: Sampling multiple chain-of-thought rationales from a language model can uncover correct solutions missed by a single attempt. However, these rationales are typically lengthy, making repeated autoregressive generation costly. Continuous latent reasoning offers a compact alternative, but point-valued methods condition repeated answer sampling on a single rationale, which can limit exploration of diverse paths and answer coverage. We thus propose rationale distribution distillation (RDD), training a conditional flow over fixed-length latent sequences by distilling from multiple textual reasoning paths. Furthermore, we expand the distillation targets through within-query mixup, interpolating between complete latent rationales for the same query. The flow nonautoregressively generates the entire latent sequence, and a frozen language model autoregressively decodes answers from sampled rationale contexts. Empirically, the resulting distributional reasoner achieves higher answer coverage than the evaluated point-valued latent reasoning baselines, particularly on advanced mathematical tasks. 

---
# OpenTWBench (Open Taiwan Benchmark): Taiwan-Domain Knowledge Evaluation for Large Language Models, and How the Measurement Apparatus Shapes the Score 

**Authors**: Liang-Hsun Huang, Min-Yi Chen, Chao-Chun Chuang, Yu-Wei Liu, Che-Yu Wu, Po-Feng Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.19848)  

**Abstract**: We present OpenTWBench, an open evaluation suite for Taiwan-domain knowledge in large language models, built around two benchmarks. tw-exam-bench contains 255,854 four-option questions across 148 academic subjects, from fifteen years of Taiwan's national examinations published as open data by the Ministry of Examination; answer keys agree across 31,312 duplicated question groups at 99.91%, with all 27 disagreements accounted for. A reviewed second pass withdrew 2,180 items whose stems depend on a figure the text does not carry; all scores use the remaining 253,674. tw-formosa-bench contains 1,696 difficulty-curated questions about everyday and contemporary Taiwan - the Taiwan that licensing examinations never ask about - generated from a community-curated corpus under a discipline in which no model writes a ground truth, and curated against a frozen seven-model pool so that the pool scored at the passing margin (46.9-74.6%, median ~54%); a leave-one-out experiment shows the selection preserves model ranking (near-zero bias). Within six weeks the frontier had cleared that band: of 108 models now on the public board, 22 score above its ceiling, the best 91.6%.
We also document how much of a score comes from the apparatus rather than the model. On identical questions, switching answer extraction from pattern matching to an explicit \boxed{} answer moves accuracy from 20.8% to 93.9%; a single placeholder character in a system prompt moved one model from last (58.2%) to first (83.9%) on the same benchmark; doubling the thinking budget from 8k to 16k tokens bought 0.2% while shuffling which questions failed; and a three-judge validation panel proved to be one judge in disguise, two members approving 92-99% of everything shown. We release all datasets, per-item difficulty data, curation manifests and a leaderboard at this https URL 

---
# EGCForest: Structured Agent Memory Organization for Enterprise Group Chats 

**Authors**: Lingxi Cui, Jinhao Ruan, Wenqing Wei, Guo Yi, Li Rao, Ke Chen, Lidan Shou  

**Link**: [PDF](https://arxiv.org/pdf/2610.19836)  

**Abstract**: Enterprise group chats record project progress, decisions, task ownership, and other organizational information valuable to enterprise agents. Reliable retrieval is challenging because evidence spans interleaved threads, channels, and time, with meaning and authority conditioned on the speaker. Neither RAG's independent chunk retrieval nor memory systems' summary-based consolidation can reliably recover such distributed evidence. We present EGCForest, a novel multi-view hierarchical memory framework for enterprise group chats. EGCForest organizes an immutable message log into three complementary forests: a Channel forest for temporal workflows, an Actor forest for speaker-conditioned context, and an Artifact forest for deliverable lifecycles. Cross-view connections are enabled by a Placement Map and sparse structural bridges. Built on this memory backbone, EGCForest retrieves and assembles evidence by jointly leveraging message-level relevance, hierarchical context from forest summaries, and cross-view links, recovering distributed evidence. Experiments on three enterprise-chat benchmarks show that EGCForest consistently outperforms RAG and memory-system baselines, improving answer accuracy by up to 20.85 points over MemOS while supporting efficient online retrieval. 

---
# Model vs. User: Whose Values Prevail? Inference-Time Value Alignment in Multi-Turn Conversations 

**Authors**: Enrico Liscio, Anne Arzberger, Aditya Shankar, Pradeep K. Murukannaiah  

**Link**: [PDF](https://arxiv.org/pdf/2610.19817)  

**Abstract**: Human values shape how we expect language models to behave. Dominant alignment approaches treat values as global objectives, although the appropriate value orientation is personal and contextual. Training-free alignment can steer models toward target values at inference time, yet has been largely evaluated in single-turn settings and with a single steering direction (e.g., toward safety). We evaluate four prompt- and activation-based steering methods in multi-turn conversations, where user and assistant models push toward the same or opposing, equally desirable values. We track alignment with a linear probe across three models and two value trade-offs. When assistant and user pull in opposite directions, the side matching the assistant's baseline bias often prevails, and effects that look successful early often fade or reverse after a few turns. Steering should therefore be evaluated over full conversations and in both directions of a value trade-off. 

---
# Delta Tokenizer: Reference-Identical Tokenization of Session Deltas and Segment-Parallelism for Long Prompts on Multicore CPUs 

**Authors**: Zhenyu Zhang, Zhichao Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.19801)  

**Abstract**: An agent that drives an LLM sends the model the same conversation again and again, and each turn adds only a little text at the end. Inference engines reuse the KV cache of the shared prefix, so only the new tokens pass through the model. However, tokenization has no such shortcut. The CPU front end of the serving engine converts every request back into token IDs from its first character, which lengthens the time to first token (TTFT) and consumes CPU on every turn.
We present Delta Tokenizer, a CPU tokenization front end that tokenizes only the delta between a request and the previous request of the same session. It retains the earlier token sequence, encodes again only the appended text and the short tail whose tokens the append could affect, and joins the two only at a position the tokenizer's splitting rules themselves cannot move, which keeps the result exact. For long requests with no state to reuse, it cuts the text where practical tokenizers already keep neighboring pieces independent, encodes the pieces on several cores at once, and concatenates the outputs into the reference encoding, exactly. A shared CPU pool schedules these chunks together with the short incremental updates, so an update runs between two chunks rather than behind an entire long job.
The incremental path shortens the tokenization of a session turn by roughly 40x relative to the Hugging Face reference tokenizer, and on 16 cores the chunk-parallel path encodes a four-million-character request 8.2x faster than a serial encode and 1.6-2.1x faster than LoPT. Inside SGLang, Delta Tokenizer reduces the median TTFT through the chat interface by 59% at 100K tokens and brings the P99 TTFT of agent sessions down 9.5x under load and 23x when long requests tie up the front end. Every output we checked matches the reference, among them 41.8 billion documents encoded by its fast encoder. 

---
# Evolve Together, Fall into Place: How Social Scaffolding Shapes Self-Evolving Multi-Agent Systems 

**Authors**: Seojin Hwang, Youngmin Choi, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.19799)  

**Abstract**: Recent work lets LLM agents evolve their own harness, up to the team itself. Yet in human groups, the same abilities achieve different outcomes under different social conditions. Borrowing the notion of scaffolding from developmental psychology, we investigate whether the social scaffolding of a self-evolving multi-agent system (MAS), namely its team composition and structure of interdependence (e.g., collaboration, conformity, or competition), shapes how the team evolves. We present AGORA (Agents Growing through Organization and Relational Arrangements), a framework that varies the two axes independently across three evolution mechanisms. Across web search, codebase understanding, and document QA, we find that (1) evolution alone rarely helps, while the right scaffolding surpasses a strong agent evolved alone by up to 25 points; (2) it does so by moving authority to the most capable member while the others step back; and (3) this member, having grown within the team, outperforms the same model alone. Designing a self-evolving MAS thus requires designing its social scaffolding, which shapes not only how its agents learn, but who gets to lead. 

---
# JetSpec++: Foresight Parallel Drafting with Self Refinements 

**Authors**: Zhaoxiang Feng, Lanxiang Hu, Haoran Yuan, Yu-Yang Qian, Jihui Yang, Peng Zhao, Yibo Zhu, Tajana Rosing, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19792)  

**Abstract**: Speculative decoding's effectiveness is governed by two quantities: the acceptance rate of drafted tokens and the cost of producing them. We show the two objectives can be co-optimized with algorithm-system co-design, and present JetSpec++, which retains a strong learned draft model while hiding its cost almost entirely. JetSpec++ combines three components: intra-block self refinement, which trains the draft model to iteratively correct its own parallel predictions within a block of tokens; inter-block foresight drafting, which begins the next block before the current one is verified; and pipelined drafting and verification, which overlaps the two stages so drafting's marginal wall-clock cost approaches zero. In comparison with state-of-the-art speculative decoding methods, JetSpec++ attains a wall-clock speedup of up to $7.45\times$ over autoregressive (AR) decoding and $1.54\times$ over DSpark, with up to 1,361 tokens per second for a single request on GB200, and an average acceptance length of 8.94 across seven benchmarks, while preserving the target model's output distribution. Our code and models are available at this https URL. 

---
# More Methods, Better Agents? A Reality Check on Self-Distillation in Agentic Reinforcement Learning 

**Authors**: Linhao Zhong, Bozhen Fang, Xuehao Wang, Duochao Shi, Zekai Luo, Guangming Yao, Yi Yuan, Hao Chen, Jingdong Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19789)  

**Abstract**: Self-distillation in agentic reinforcement learning has produced a growing number of methods. However, existing evaluations often reuse published baseline scores under differing experimental conditions, provide few direct comparisons among contemporaneous methods, and give limited attention to training stability and computational cost. These limitations obscure the source of reported gains and the practical strengths and weaknesses of competing methods. Costly agent training further motivates a common basis for comparison. To address these gaps, we compare ten representative methods against GRPO on ALFWorld, ScienceWorld, Search-QA, and WebShop across three model sizes. Our study comprises 132 training runs totaling 12,124.4 recorded H800 GPU-hours, with repeated evaluations over eight rollout seeds for the pretrained backbones and all eleven trained methods. Beyond final task performance, we examine training progress, stability, and measured computational cost. Our evaluation yields three key findings. First, gains over GRPO remain limited in scope, with no method consistently outperforming it across settings. Second, strong final performance can mask unstable training. Third, additional training cost does not consistently translate into better performance. We establish this benchmark to support fair, comprehensive comparison and guide future research toward reliable and cost-effective improvements in agent training. Project page: this https URL 

---
# Likelihood Is Not Authorship: Can We Reliably Measure Human Contribution in AI-Assisted Writing? 

**Authors**: Hasan Amin, Zhuoran Lu, Ming Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.19767)  

**Abstract**: AI-text detection generally asks whether a machine wrote a text. As AI becomes an increasingly integral and accepted part of writing, the more relevant question is no longer whether AI was used, but how much of the text reflects human contribution. A prominent line of work estimates this by measuring how predictive a person's prompt is of the final text. We show that this measure can be easily manipulated. We introduce SPECIFICITY PADDING that simply asks the model to copy a few random IDs from the prompt, and that alone makes a title-only request look more human than one that supplies the article's key ideas. A context-swap audit then holds each document fixed, and only shows or hides the IDs to the evaluator. The score of the unchanged text swings by a third to half of its scale, and it rises for every document we test, on news and scientific abstracts, across varied LLMs from 3.8B to 753B parameters. The same articles can average 85\% human or 15\% human depending only on which prompt is on record, and even with no prompt at all, the output-only variant rewards the random strings themselves. An exact identity, the credit ledger, explains why. The score is a surprisal-weighted average over the text, so rare strings that the context makes copyable collect the credit, whatever they mean. A two-point bound marks a deeper limit. If the prompt could have been machine-written, every transcript-only method must, in the worst case, miss by half the contribution at stake. Likelihood-based scores alone cannot reliably establish human contribution without evidence of what the person added and how the text was produced. 

---
# PRISMATIC : Prescription Risk Inspection System for Multi-Agent Treatment Interaction in Clinical Decision-Making 

**Authors**: Ziyang Ling, Shuyu Lu, Ziyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19762)  

**Abstract**: Medical prescribing errors for multimorbid patients, who require complex drug regimens, are a major patient safety concern. These errors often stem from fragmented decision-making across medical specialties, a critical coordination failure that leads to adverse drug events and rehospitalizations. To address this coordination challenge, we introduce PRISMATIC, a multi-agent architecture where specialized agents, each representing a workflow role, collaboratively generate and refine treatment plans. Operating on a blackboard architecture, the system first structures patient information from clinical notes, then iteratively develops safe, rational, and interpretable drug regimens. The agents' collaboration is governed by two novel adaptive coordination mechanisms: (1) Dynamic Self-Updating Weight Adjustment (DSWA), a long-term mechanism that adjusts the influence weights of agents based on historical case outcomes, enabling the system to adapt and trust more reliable specialists over time. (2) Discrepancy Feedback Precise Calibration (DFPC), a short-term mechanism where agents provide iterative feedback to resolve disagreements and eliminate prescription conflicts for the current patient. Experiments on patient records from the MIMIC-IV database show that PRISMATIC significantly reduces drug conflicts, improves prescription coverage, and enhances treatment consistency compared to strong LLM baselines. Our work demonstrates that structuring clinical co-treatment as a multi-agent coordination problem offers a promising path toward safer and more coherent multidisease management. 

---
# Qwen-MT-Lumen at WMT2026: Instruction-Specialized Translation via Quality-Aware Training and On-Policy Distillation 

**Authors**: Hao Wang, Yangyang Liu, Xiaohu Zhao, Heng Liu, Zifu Shang, Tianhao Li, Ruize Gao, Jialong Tang, Liangying Shao, Haoran Wei, Baosong Yang, Linlong Xu, Longyue Wang, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.19758)  

**Abstract**: We present Lumen, The open-source model is available at, our submission to the WMT 2026 General Machine Translation Shared Task. We focus on instruction-specialized translation, where systems must produce accurate translations while satisfying fine-grained requirements on style, terminology, and structured output. Lumen follows a four-stage pipeline. First, continued pre-training and supervised fine-tuning jointly strengthen multilingual translation competence and the ability to follow translation-specific instructions, such as terminology, style, and format constraints. Second, we extend the originally offline M^{2}PO preference-optimization framework into an online preference-optimization stage: the current policy samples candidate translations, an LLM judge assigns continuous direct-assessment scores, and the resulting multi-pair preferences are used to update the model. Third, on-policy distillation transfers stronger translation-instruction-following behavior from a larger teacher model to the Qwen3-14B student, improving translation-instruction adherence while retaining a 14B primary translation backbone. Finally, structure-aware inference and postprocessing preserve required formats, while selective repair addresses clear structural or decoding failures before submission. We participate in 23 translation directions. Under an absolute 1--5 LLM-judge rubric that evaluates translation-instruction following and translation quality separately, Lumen reaches an overall mean score of 4.60. Despite using a 14B translation backbone, it achieves performance comparable to leading proprietary systems in both translation quality and translation-instruction following. 

---
# Beyond Reachability: A Relation-Semantic Reasoning Framework for Biomedical Multi-hop QA 

**Authors**: Ziyang Ling, Ziyi Chen, Ronald X.Xu, Mingzhai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.19754)  

**Abstract**: Biomedical multi-hop reasoning over knowledge graphs requires identifying a semantically valid relation sequence that supports the answer. Existing methods can retrieve correct answers, yet often rank them poorly within a large pool of seemingly relevant candidates. Even when a correct answer is retrieved, its associated evidence and reasoning path may fail to support the relation composition required by the question. We term this disconnect between answer reachability and evidential validity the reachable-but-invalid problem. We present ReSemReason, which shares a question-induced distribution over typed, directed relation schemas between candidate discovery and evidence selection. Schema compatibility guides graph search; ReSemAttn then uses relation-conditioned key/value projections and an uncertainty-weighted path prior to attend to candidate-specific evidence. A separate absolute-support score complements normalized attention when all retrieved paths are weak. The reported experiments on BioHopR, PrimeKGQA, and MedReason show improvements in answer ranking and answer-set quality. Path-discrimination analyses examine the distinction between endpoint correctness and evidential support. We also specify matched controls for search, attention, and schema uncertainty. 

---
# Does the Linguistic Register of Thinking Matter? Wenyan as a Compact Reasoning Register for Chinese LLMs 

**Authors**: Minghan Li, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.19741)  

**Abstract**: Visible reasoning traces of Chinese reasoning-oriented LLMs are typically written in modern vernacular Chinese, which may be verbose under limited decoding budgets. We ask whether the linguistic register of reasoning can be used to improve efficiency. We propose Wenyan Thinking, which emits intermediate reasoning in Classical Chinese (wenyanwen) while retaining a short direct-answer output. Starting from modern-Chinese reasoning traces, we rewrite only the reasoning field into wenyan and fine-tune Qwen3-4B-Thinking-2507 on the rewritten supervision. To distinguish register shift from generic compression, we construct a compact modern-Chinese control using the same source data, teacher family, base model, and SFT recipe. Across budget-sensitive evaluations on MMLU-ProX-Lite-ZH, CEVAL, and AGIEval-CN, Wenyan Thinking substantially reduces reasoning length and missing-answer rates under constrained decoding. Compact rewriting explains much of the efficiency gain, while wenyan consistently outperforms the compact modern-Chinese control on the primary benchmark and remains competitive across the additional benchmarks. These results identify linguistic register as a practical design axis for efficient visible reasoning in Chinese LLMs. 

---
# MakSafe: A Spatiotemporal Semantic Framework for Formal Specification and Compliance Checking of Road Vehicle Safety Rules 

**Authors**: Maksym Labzhaniia, Julien Deantoni, Marie-Agn{è}s Peraldi-Frati, Fr{é}d{é}ric Mallet  

**Link**: [PDF](https://arxiv.org/pdf/2610.19735)  

**Abstract**: Scenario-based validation is essential for automated road vehicles but cannot cover the full variability of openworld traffic. We present MakSafe, a formal spatiotemporal framework for specifying and checking road vehicle safety rules through semantic zones, action-based forecasting, and trace-tree reasoning, avoiding uniform time discretization. We formalize the road network, vehicle states, and occupancy zones, then build executable rules for front-vehicle identification and timeto-collision (TTC) verification. The evaluation combines a tracedependent case study with a scalability benchmark over traffic density (N ), horizon (H), and branch factor (B). MakSafe yields branch-specific verdicts, deriving safe and unsafe outcomes from the same initial scene, and re-evaluates a scenario after a control update. A single rule check stays well below a millisecond across the reported sweep, and longer horizons slow the TTC check. 

---
# Recognizing Dialogue Ambiguity via Candidate Set Visual Grounding 

**Authors**: Shun Inadumi, Lee Sangmyeong, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2610.19723)  

**Abstract**: Interpreting an ambiguous utterance in real-world interactions often requires context beyond a single utterance. When the utterance refers to objects in a visual scene, it is necessary to maintain and update the set of plausible referents as context accumulates, without assuming that the available information identifies a unique object. However, prior work on interactive visual grounding, which localizes objects referred to by linguistic expressions, has focused on identifying a unique object once sufficient dialogue context becomes available. To address this gap, we propose Candidate Set Visual Grounding (CSVG), a task for estimating the set of objects consistent with the given dialogue context. We further introduce CandVG, a benchmark that evaluates how the candidate set changes across dialogue turns. Our evaluation of vision-language models shows that stronger visual grounding capability does not necessarily yield better CSVG performance. In particular, models have difficulty predicting the complete candidate set, owing to challenges in candidate generation and premature commitment to a small set of candidates in early turns. Nevertheless, we find that this limitation can be partially mitigated in the evaluated large-scale models using instructions designed for CSVG. 

---
# Beyond State Capacity: Why Retrieval Breaks in Mamba 

**Authors**: Nir Yarden, Amit Ben-Artzy, Roy Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2610.19709)  

**Abstract**: Mamba models underperform Transformers on in-context retrieval. This gap is usually attributed to Mamba's fixed-size recurrent state, which loses information as the context grows. We challenge this assumption by focusing on query-first retrieval, a setup in which the query precedes the context and state capacity should not be the binding constraint. We show that even in this setting Mamba lags substantially behind Transformers, despite its selection mechanism, which was designed to distinguish relevant content from distractors. We then demonstrate that the Mamba selection gates have sufficient control over the state updates to influence the model output: intervening on specific gates alone can swing accuracy between extremes. However, as gates are input-dependent, they require the incoming token's representation to encode its relevance to the query. We therefore probe token representations across layers, and find that the relevance signal emerges only gradually with model depth, and that the gates begin to reflect it only several layers deeper. Crucially, we find that gates in retrieval-critical layers are not yet aligned with relevance. The selection mechanism is thus expressive enough to solve the task, but it commonly operates on a relevance signal that has not yet arrived. Mamba's retrieval bottleneck therefore extends beyond state capacity: even when capacity is sufficient, the delayed emergence of relevance information across depth often prevents successful retrieval. 

---
# RouteWeaver: Weaving Mode Selection and Execution into Unified LLM Routing 

**Authors**: Xiaohan Wang, Haozhen Zhang, Qingyuan Liu, Tao Feng, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19705)  

**Abstract**: LLM routing dynamically selects and coordinates heterogeneous language models across diverse tasks. However, existing approaches are largely built around a predefined execution paradigm, such as single-round, multi-round, or agentic routing, which fixes how model calls are organized and limits adaptation to different query needs. We propose RouteWeaver, a unified routing framework that integrates these routing modes within a single LLM policy, jointly learning which mode to use and how to select models within that mode. To jointly optimize mode selection and within-mode execution, we develop COMET-GRPO (COordinated Mode-Execution Training), which decouples the trajectory-level advantage into separate signals for mode selection and within-mode execution, enabling more targeted credit assignment to each decision level. We further introduce progressive exploration, which gradually shifts training from externally assigned modes to autonomous selection, so that the router first learns how to execute each mode before deciding when to use it. We evaluate RouteWeaver across mathematics, coding, knowledge, logical reasoning, and factual recall. RouteWeaver achieves an average accuracy of 61.4%, an 11.03% relative improvement over the strongest baseline, and outperforms all compared baselines in each of the five domains. Ablations show that both COMET-GRPO and progressive exploration are important for adaptive routing; without progressive exploration, the routing policy collapses entirely to single-round execution. 

---
# BANGLAPLAN: A Benchmark for Constraint-Based Planning and Decision-Making in Bangla 

**Authors**: Md Rayhanul Masud, S M Maksudul Alam, Mohammad Jaminur Islam  

**Link**: [PDF](https://arxiv.org/pdf/2610.19704)  

**Abstract**: Large language models (LLMs) increasingly help users arrange appointments, plan journeys, and allocate limited resources. These tasks require models to satisfy constraints on time, cost, capacity, and order, or determine that no feasible solution exists. Bangla is the world's seventh most widely spoken language, and existing benchmarks cover many language and reasoning tasks. However, benchmarks for evaluating planning and decision-making in Bangla under explicit constraints remain lacking. We introduce BANGLAPLAN, a benchmark of 1,000 problems across ten task families grounded in everyday life and institutions in Bangladesh. It includes feasible and infeasible problems in four language registers: everyday, formal, colloquial, and Bangla--English mixed. Answers cover eight structured types, including schedules, routes, allocations, and ordered steps, in a common JSON format. We construct the benchmark through language-model-assisted drafting and independent human validation. We evaluate 11 open-weight and proprietary model configurations using exact match, separately reporting feasible-case accuracy, infeasibility detection, and format errors. We find a large performance gap: GPT-6.1 Sol achieves 98.7% accuracy and Qwen3.5-27B with thinking reaches 80.3%, while direct-answer configurations score at most 30.9%. Seven of eleven configurations perform below the always-``infeasible'' baseline. Our findings also suggest that harder problems primarily reduce plan-construction accuracy, while infeasibility detection remains more robust. BANGLAPLAN provides a reproducible basis for evaluating planning and decision-making in Bangla. BANGLAPLAN is available at: this https URL. 

---
# Share the Symbol, Not the Vector: Training-Free Fact Communication Across Language Models 

**Authors**: Xin Li, Mengbing Liu, Hao Jiang, Xin Gao, Annan Wang, Xingwei Qu, Yichi Zhang, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19696)  

**Abstract**: Multi-agent systems allow language models to solve tasks collaboratively by sharing complementary information. These systems typically communicate through text, while recent approaches also explore exchanging internal model states. However, sharing these states across architectures often requires learned translators or assumptions about model compatibility. We introduce SYNC (Symbol-Indexed Neural Communication), a word-indexed interface that connects heterogeneous models without fine-tuning or a learned translator. From descriptions only it can see, the sender selects words from a shared list and sends their indices for each object, without seeing the questions. The receiver writes the described facts into its hidden states using its own word directions, vectors fitted once from its activations. We test selective use by swapping one attribute between two objects and requiring correct answers about both changed and unchanged attributes before and after the swap, four answers in total. We choose an early write layer on four development receivers and then fix it. With three different senders and nine newer receivers from five families, four-answer success on new scenes reaches 72.0-98.3%, up to 96 percentage points above middle-layer writing, and the two largest receivers match plain text. A task-free probe predicted in advance which depths each new receiver can use (45 of 45 cases). Without retuning, messages listing every fact keep 89.6-100% success under new layouts, and sender messages reach 23.9-64.0% on a landmark task with multi-token symbols, below text but from 0% without a message. These results show that shared word indices and receiver-local directions can support selective fact use across model families without additional training. 

---
# Interlock: From Local Correctness to Task Completion in Long-Horizon Agents 

**Authors**: Linji Hao, Yanji He, Zihan Wang, Hansi Yang, Jiayuan Zhang, Hongyan Hao, Qi Gu, Yan Feng, Han Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19686)  

**Abstract**: Large language model agents increasingly undertake long-horizon tasks in complex environments characterized by fragmented evidence, partially completed states, and dependencies across resources and artifacts. Existing agents may satisfy individual requirements while failing to maintain these relationships, producing locally correct but incomplete outcomes. These failures arise because task completion depends on relationships among environmental evidence, consequential decisions, and required effects. To address this completion gap, we propose Interlock, a framework for task synthesis that represents each task as a typed completion graph. Grounding edges connect public evidence to decisions, coupling hyperedges impose joint constraints across components, and realization edges connect decisions to all required downstream effects. Interlock compiles this graph into a resettable workspace, a public request, and an evaluation rubric. A source-coverage review checks whether the original requirements survive generation, while a public-input review checks whether the rubric is supported by the materials available to the solver. An agent-as-judge inspects execution artifacts and runs verification scripts, retaining trajectories judged to satisfy all mandatory requirements. To evaluate training utility, we fine-tune two Qwen3.6 backbones on their own accepted trajectories. Across the two backbones, training improves AutomationBench task pass rates by 4.00-4.50 percentage points, JobBench weighted scores by 3.20-6.64 points, and Harvey LAB criterion pass rates by 3.89-4.16 percentage points. These results demonstrate the training utility of the synthesized experience across external task environments. 

---
# Measuring and Reducing Unusable Replies of LLM Agents 

**Authors**: Yilin Jiang, Fei Tan, Xuanyu Yin, Guanchu Wang, Xia Hu, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2610.19659)  

**Abstract**: People delegate multi-step work to LLM agents and read only the reply that the agent hands back. Current evaluation scores task completion and the faithfulness of reports, and it leaves open whether the requester can understand and use the reply. We call this property reply usability and measure it with a reader model that takes the requester's position and tests whether the reply can be interpreted from what the requester has seen, is in the requester's language and answers the request. Its agreement with the majority of five annotators lies within the range of the individual annotators. Across eight models from seven families, 54% of 480 replies fail, and the best model still fails on 30%. Failed replies mostly point to material the requester has not seen or use labels that the agent coined during its work. Stating explicit requirements for the reply reduces the failure rate from 0.52 to 0.43 together with errors and omissions, and a secondary model that rewrites the reply reduces it to 0.36, which human annotators confirm in a blind comparison. Reply usability can thus be scored alongside task success and improved with one appended message or one call to a secondary model. 

---
# HACHIMI-II: From Role-Play to Learner Behavior Modeling for Faithful Student Simulation 

**Authors**: Yilin Jiang, Fei Tan, Xuanyu Yin, Jing Leng, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.19654)  

**Abstract**: Simulated students let researchers test tutoring systems and teaching strategies before real learners are available, but only if the simulation reproduces where a particular learner succeeds and fails. The common approach asks a large language model to role-play a student, but a role does not model what this learner knows or will do next, so the model's own competence fills the gap. Simulated students perform at a level set by the model, differ little from one another, and seldom make the real learners' errors. We model the learner's behavior and recast student simulation as third-person prediction of the learner's next action. HACHIMI-II builds a student model from the learner's history, infers the learner's state for the current item or tutor turn, and compiles it into a behavior harness that constrains the next action. On three item-response datasets, the Wasserstein-1 distance between simulated and real student ability is 2.4--8.4$\times$ smaller than that of the strongest baseline. On two dialogue datasets, HACHIMI-II reproduces real student errors with the same knowledge type, locus, and mechanism 1.8--5.1$\times$ more often than the strongest baseline, and ablations trace the gains mainly to the explicit learner state. Predicting the learner makes capable models usable as simulated students. 

---
# Selective Reasoning for Typed Decision Models: When to Start, Read, and Stop 

**Authors**: Bin Zheng, Zheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19644)  

**Abstract**: Typed decision models need an interface that can use additional reasoning while retaining candidate outputs. We train a shared candidate-answer reader on a frozen Qwen3.5-4B backbone to unify direct decisions with on-demand, interruptible reasoning. The reader returns distributions from empty and partial reasoning prefixes. A benefit gate starts reasoning, a natural-state trigger requests explicit reads, and an entropy rule accepts acquired answers and cancels generation. Measurements distinguish answer-query work, generated output, and pipeline time. Across 768 questions from ProofWriter, LogiQA 2.0, and ARC-Challenge, accuracy rises from 70.96% for direct reading to 84.38%. Relative to 512-token terminal reading, accuracy improves by 7.29 percentage points with 2.25% more mean output work. Relative to 1,024-token terminal reading, pipeline time falls by 46.30% with an accuracy difference of -1.17 points. Exact-prefix controls isolate the saving from interruption, while read-scheduling controls distinguish reduced observation work from shorter response time. 

---
# J-tau: A Fully Localized Japanese Benchmark for Tool-Agent-User Interaction 

**Authors**: Chihiro Yano, Jun Hirako, Hiroaki Yamagiwa, Ryota Hirobuchi  

**Link**: [PDF](https://arxiv.org/pdf/2610.19638)  

**Abstract**: We present J-tau, the first publicly available Japanese benchmark for evaluating tool-using LLM agents in interactive customer-support scenarios. While tau-bench is widely used to evaluate LLM agents, it is English-centric and cannot evaluate agents in other languages. Existing multilingual extensions offer no public Japanese version and rely primarily on machine translation, leaving locale-specific entities such as addresses and currencies inconsistent with the target region. To enable reliable evaluation in Japanese, we carefully localize tau-bench through machine translation, entity adaptation, and human review. During this process, we also uncover and fix two flaws in the original tau-bench: invalid tasks and a score aggregation error that inflates certain models' results. Experiments show that J-tau yields fewer evaluation errors than a naively machine-translated counterpart and that our fixes resolve both flaws. 

---
# DAGFormer: Global Topology Prediction for Transformer Cross-Layer Connections 

**Authors**: Xiaocong Yang, Yuren Hao, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2610.19625)  

**Abstract**: We propose DAGFormer, a Transformer language model whose cross-layer connections are planned by a jointly trained topology predictor. Beyond the residual connections in a standard Transformer, DAGFormer adds direct connections from earlier layers to later ones, and the predictor controls the strength of every connection based on input tokens. Unlike existing cross-layer routing architectures, which set each layer's connections separately, DAGFormer plans the routing of the whole model at once. On matched OLMo-2 pairs from 75M to 1B parameters, DAGFormer lowers WikiText-2 perplexity by 17% to 25% over the vanilla Transformer, and a 1B DAGFormer trained on 5B tokens outperforms the vanilla model trained on 10B. Against four routing baselines with the same data and budget, it reaches the lowest perplexity at 75M, 300M and 1B. The gain carries over to 13 commonsense, science, reading and math evaluation tasks (2.0 to 2.5 points higher mean accuracy at 300M and 1B) and to instruction finetuning (63% to 70% AlpacaEval win rate against the vanilla Transformer, leading every routing baseline), and under structured pruning DAGFormer degrades the least in five of six settings. 

---
# Are We Really Measuring Forgetting? Pitfalls in Evaluating Machine Unlearning for Multimodal LLMs 

**Authors**: Oscar Chew, Po-Yi Lu, Yi-Jie Cheng, Kunal Jain, Hsuan-Tien Lin, Zhengzhong Tu, Kuan-Hao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19613)  

**Abstract**: As machine unlearning for multimodal large language models (MLLMs) rapidly advances, rigorous evaluation of its effectiveness becomes increasingly important. However, existing benchmarks largely inherit evaluation protocols from text-only unlearning and overlook the unique challenges of unlearning in multimodal settings. In this work, we identify three major pitfalls that may lead to misleading conclusions: the lack of heldout evaluation to distinguish genuine concept-wise forgetting from instance-wise overfitting, the reliance on synthetic, fine-tuned knowledge rather than real-world knowledge acquired during pre-training, and unfair method comparisons caused by applying the same training settings to different unlearning methods. To address these challenges, we introduce GEMMU (Generalizable Evaluation for Multimodal Machine Unlearning), a benchmark built on real-world knowledge and a principled evaluation protocol. Re-evaluating representative unlearning methods under GEMMU reveals substantial generalization gaps and challenges conclusions drawn from previous benchmarks. Our findings highlight the importance of rigorous evaluation for advancing multimodal machine unlearning and provide guidelines for future benchmark design. 

---
# PA1-OPD: One-Step Prefix-Augmented On-Policy Distillation for Language Model Pretraining 

**Authors**: Zhenhe Wu, Jian Yang, Yaping Jin, Xinlai Kang, Jie Hu, Hanting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19601)  

**Abstract**: Knowledge distillation (KD) improves language-model pretraining by matching teacher distributions at corpus prefixes, but provides no direct supervision at contexts induced by student predictions. We introduce PA1-OPD, which augments corpus-path distillation with teacher feedback after one student-sampled transition per selected prefix. This expands the supervised states while retaining next-token prediction and corpus-path KD. Data-anchored roots support parallel sampling, and differentiable tree attention shares prefix computation without changing inference. In two continued-pretraining settings, full coverage improves held-out cross-entropy over KD across C4, PG-19, and Pile at every evaluated checkpoint. In Stage2, selecting the 10% of roots with highest student surprisal adds 48.85% to KD's Macro cross-entropy gain over next-token training. Gains remain when branch supervision excludes successor states already on the corpus path, supporting the value of feedback beyond observed contexts. Additional depth or breadth improves aggregate results at unchanged per-term supervision weight, with benefits varying by corpus. These findings support local student-induced supervision as a complement to pretraining distillation. 

---
# Moral Combat: Exploiting Moral Foundations Theory to Jailbreak LLMs 

**Authors**: Adel Khorramrouz, Yi-Chia Wang, Sharon Levy  

**Link**: [PDF](https://arxiv.org/pdf/2610.19587)  

**Abstract**: Social psychology has long described how people can be persuaded to interpret a harmful act as moral. Recent research suggests that Large Language Models (LLMs) echo aspects of human psychology, from cognitive biases to social conformity. This raises a question with direct safety implications: can LLMs be moralized to bypass safety guardrails? We answer this question with a multi-agent framework grounded in Moral Foundations Theory (MFT). Given a harmful request that is initially refused, our agents diagnose which moral foundation (MF) the prompt is violating, identify a justifiable foundation under which the same request appears virtuous, rewrite the prompt accordingly, and reinforce it with generated precedents of comparable moral reframing. We evaluate our framework on harmful prompts from three well-known jailbreak datasets across five open-source and proprietary models. Our analysis reveals that models are highly susceptible to moralization, exhibiting high single-trial jailbreak rates with success increasing further under repeated attempts. We also find that safety benchmarks are imbalanced in the MFs they violate, concentrating on foundations toward which models already exhibit biases. 

---
# INTACT: Inducing Latent Reasoning on a Frozen Backbone 

**Authors**: Kutay Tire, Atula Tejaswi, Haris Vikalo, Sujay Sanghavi  

**Link**: [PDF](https://arxiv.org/pdf/2610.19568)  

**Abstract**: Latent reasoning replaces natural-language chains of thought with compact continuous "soft-token" representations, reducing the number of intermediate tokens that must be decoded autoregressively. We use a small external encoder to induce latent reasoning in a frozen backbone model. The encoder iteratively maps the backbone's hidden states to new latent blocks, constructing a latent reasoning prefix, after which the frozen model writes its remaining reasoning and its answer as text. Our method, INTACT, trains this encoder through distillation to reproduce the effect of written reasoning steps while keeping the model's behavior close to that of the frozen backbone, and then reinforces it on answer correctness. Across four frozen backbones from 1B to 8B, INTACT improves both GSM8K and average out-of-domain accuracy, including GSM8K gains from 75.8% to 79.3% on Llama-3.2-3B and from 77.6% to 79.7% on Qwen2.5-3B. It also reduces end-to-end latency per problem by 19% on Llama-3.2-3B and outperforms the evaluated latent-reasoning baselines at the 3B and 8B scales. Beyond the mathematics setting, the same recipe trained on a multi-domain corpus improves the frozen backbone on twelve evaluation benchmarks spanning mathematics, science, and commonsense, without task-specific adaptation. 

---
# REWIND: Uncertainty-Guided Trajectory Repair for Test-Time Reasoning 

**Authors**: Yi Ding, Mengyu Wang, Min Zhou, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19561)  

**Abstract**: An early mistake in the generated reasoning can constrain all subsequent reasoning. Repeated-generation methods, like MCMC-based approaches, attempt to improve output quality by sampling multiple generations, but their effectiveness can be limited by randomly selecting where to regenerate. On MATH500 with Qwen2.5-Math-7B, we find that incorrect responses have substantially higher mean token entropy than correct responses, with an average within-question relative increase of approximately 112%, suggesting that entropy can signal reasoning errors. Motivated by this finding, we introduce REWIND, a training-free controller that uses entropy dynamics to propose targeted suffix regenerations. Across three model families and five benchmarks, REWIND outperforms all baselines, with gains of 0.51-10.00 percentage points over the strongest baseline. These results demonstrate the effectiveness of targeted, uncertainty-guided trajectory repair across models and tasks. Our code is available at this https URL. 

---
# Evidence-Anchored Overdose Death Prediction from Death Investigation Reports 

**Authors**: Raavi Gupta, Ziqing Yang, Karli R. Hochstatter, Talia Nadel, Nicole D'Anna, Jason K. Graham, Hannah Johnson, Kelsey Opozda, Nabila El-Bassel, Anubhav Jangra, Smaranda Muresan  

**Link**: [PDF](https://arxiv.org/pdf/2610.19556)  

**Abstract**: Confirmation of overdose deaths in the United States is often delayed due to the complexities of certifying drug-involved deaths. Death investigation reports represent a rich but underutilized data source that could be used to predict whether a death is a suspected overdose (i.e., a death preliminarily identified as an overdose prior to the final cause and manner of death determination). We present an automated framework that classifies suspected overdoses from death investigation reports, while producing supporting explanations guided by expert knowledge. Through an expert-guided design process, we investigate 1) what type of explanations are best for model prediction performance, and 2) which architecture is better suited for this task: pipeline models that rationalize first and then predict, or self-rationalizing models that jointly predict and rationalize? Our experiments show that a pipeline model with explanations grounded in the input death investigation reports outperforms the other settings. Our best model also outperforms a rule-based system currently in deployment by 15%, while allowing expert interventions for final decisions. 

---
# GraphRAG-LTM: A Layered Threat Modeling Framework for Attacks on Graph Retrieval-Augmented Generation 

**Authors**: Yining Tang, Junyu Zhang, Haomiao Qiu, Yongchao Liu, Jiajun Zheng, Zhao Wang, Ziyue Qiao, Chao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19549)  

**Abstract**: Graph retrieval-augmented generation (GraphRAG) uses document-derived knowledge graphs to assemble evidence for large language model (LLM) answers. Its security depends on both the seed chunks retrieved for a query and the graph evidence assembled from them, yet attacks on this evidence path remain underexplored. We introduce GraphRAG-LTM, a Layered Threat Modeling framework that tests three nested levels of attacker access. BB Query uses ranking feedback to make query edits under a semantic-similarity constraint and demote relevant seed chunks. WB Query uses retriever gradients for the same objective. Graph-aware WB keeps the white-box adversarial query and deletes bridge triplets from the query-specific evidence graph, disrupting multi-hop evidence propagation through localized graph edits. Across datasets, retrievers, and LLM backbones, rank feedback alone reduces answer quality, gradient access strengthens query attacks, and bridge-triplet deletion further impairs supporting-fact retrieval. Further analyses show how seed retrieval and graph evidence jointly contribute to these failures. 

---
# Typed Decision Models on Tabular Data 

**Authors**: Baowen Zhang, Hangting Ye, Hongyuan Zha  

**Link**: [PDF](https://arxiv.org/pdf/2610.19536)  

**Abstract**: Typed decision models are language models that answer a question with a fixed set of possible answers by returning a probability for each answer instead of generating text. We study them on tabular data and compare four decision models with three large language models (LLMs) on row classification, fact verification and anomaly detection. Every request is sent in its original form and under 21 types of rewrite that keep its content, such as renaming the options or writing the table in another format. The decision models perform as well as the LLMs at a lower cost. Three of them are also far more robust to how the question is formulated. No model we test is robust to how the table is organized. For every model, at least a third of the row decisions change under some rewrite of the table or the question. The decisions to which a decision model gives low probability are the ones most likely to change. Decision models are a practical choice for tabular decisions, but their robustness remains an open problem. 

---
# A Unified Response Framework for Questions and Tutoring Dialogues 

**Authors**: Zhangqi Duan, Simon Woodhead, Maya Miller-Vedam, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2610.19532)  

**Abstract**: Existing student models in educational applications largely focus on student responses to questions. However, as tutoring becomes more accessible at large scale, it is important to analyze dialogues between tutors and students, which may reveal fine-grained evidence on student knowledge. In this work, we propose a unified framework that models both student responses to questions and tutor-posed subtasks during dialogues on a shared item response theory (IRT) scale. We first identify mathematical subtasks in tutoring dialogues, rewrite them into self-contained questions, and identify the correctness of student responses to these subtasks. We then develop an IRT framework that estimates question/subtask difficulty from text, tracks students' evolving abilities across question-answering and dialogue interactions, and a novel method to leverages tutor moves between consecutive subtasks to regularize their difficulty values. We evaluate our framework on two real-world math education datasets containing both human- and AI-tutored dialogues. Results show that our framework outperforms strong baselines on both question difficulty prediction and response correctness prediction. 

---
# Retrieve What Helps, Only When It Helps: Agentic Reinforcement Learning for Selective Retrieval-Augmented Machine Translation 

**Authors**: Shijun Li, Yang Liu, Najmeh Sadoughi, Dongmei Wang, Xinjie Li, Zhu Liu, Rohith Mysore Vijaya Kumar, Yuzong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19528)  

**Abstract**: Retrieving external knowledge can improve large language model (LLM) translation by providing supplementary information such as idiom interpretations and terminology. However, existing retrieval-augmented machine translation systems typically apply predefined retrieval strategies uniformly across all inputs. We argue that translation operates in a partial-evidence regime: LLM's parametric knowledge suffices for many inputs, while external evidence benefits a subset and usually supports rather than directly provides the desired translation. This creates two challenges: deciding when to retrieve and what evidence to acquire. We introduce Selective Retrieval-Augmented Translation (SRAT), an agentic reinforcement learning framework that learns when to translate directly or retrieve, what to retrieve, and how to integrate these decisions by optimizing final translation quality. To align evidence retrieval with downstream translation quality rather than simple semantic similarity, we further develop joint Plackett--Luce optimization and translation-quality contrastive training for retriever-training. Experiments on idiom, knowledge-intensive, and document-level translation with offline or online retrieval show that SRAT consistently outperforms never- and always-retrieve baselines, and quality-aligned retriever training provides further gains when similarity-based retrieval is insufficient. The results demonstrate the benefit of treating retrieval as a learnable translation action rather than fixed preprocessing. 

---
# One Block Ahead: Promoting Sampler State in Diffusion Language Models 

**Authors**: Yangyi Shen, Pin-Jui Ku, Nicolai Oswald, Jean-Marie Lemercier, Yonggan Fu, Stefano Ermon, Jure Leskovec, Boris Ginsburg, Ante Jukić  

**Link**: [PDF](https://arxiv.org/pdf/2610.19510)  

**Abstract**: Block-diffusion language models denoise a whole block of token positions in parallel so each forward pass can emit many tokens. However, blocks are still denoised in strict sequence, a cost that within-block acceleration leaves untouched. Here, we investigate decoding blocks concurrently and propose One Block Ahead (OBA), a training-free decoding algorithm that denoises the next block alongside the current one. We demonstrate that for uniform-state diffusion LMs, where a block's denoising progress is not visible in its intermediate tokens, concurrent decoding is advantageous if the next block's tokens are promoted together with the corresponding sampler state, including the self-conditioning signal and the denoising time. We also derive a break-even criterion that relates the benefit of OBA to the relative cost of a paired-block forward pass. On DiffusionGemma, a state-of-the-art diffusion LM, OBA increases the tokens per forward pass by 42.3% (from 15.64 to 22.25) and the single-request denoising throughput by 16.3%, averaged over ten benchmarks, with a small accuracy regression. 

---
# Structure-Preserving Document-Level Data Augmentation via Discourse Parsing 

**Authors**: Ruilin Shang, Shilong Hou, Zi Long, Xianghua Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19497)  

**Abstract**: Data augmentation (DA) plays a vital role in improving model performance in low-resource or limited-supervision scenarios within Natural Language Processing (NLP). Existing DA methods, such as synonym replacement and back-translation, have demonstrated effectiveness on the word or sentence level; however, they frequently neglect discourse-level coherence and logical flow, which are essential for complex tasks dependent on inter-sentential relationships. In this paper, we propose a structure-preserving document-level data augmentation framework based on large language models (LLMs) and document-level discourse structure parsing. Our approach identifies rhetorical relations between sentence pairs and extracts key phrases, which are then replaced with semantically coherent but topically shifted content while preserving the original discourse structure. Experimental results on text summarization and question answering show that the proposed augmented data provides complementary supervision and improves combined-data training across the evaluated tasks. Comparisons with a standard RST parser and controlled ablations further support the proposed discourse representation. 

---
# Towards Efficient Simultaneous Inverse Text Normalization with Pretrained Text-to-Text Language Model and Read-Tag-Write Policy 

**Authors**: Kiet Anh Hoang, Khanh Le, Bao Nguyen, Linh Pham, Dung Vo, Thai Tran, Mai Nguyen, Tri Nguyen, Vu Le  

**Link**: [PDF](https://arxiv.org/pdf/2610.19492)  

**Abstract**: Streaming Inverse Text Normalization (ITN) is vital for converting spoken-form outputs from streaming Automatic Speech Recognition into formatted written text. Existing streaming ITN methods rely on hybrid systems combining neural tagging with expert-crafted finite-state transducer rules, limiting scalability across domains and languages. While end-to-end models offer superior scalability by learning directly from data, standard encoder-decoder architectures are inherently non-streaming due to global attention. In this paper, we propose an efficient streaming end-to-end ITN system, adapting a pretrained text-to-text model to leverage its robust linguistic knowledge. To enable streaming, we introduce architectural adaptations, a specialized training strategy, a Read-Tag-Write decoding policy, and inference optimizations. Experiments on Vietnamese datasets show accuracy comparable to non-streaming baselines, outperforming hybrid methods while satisfying real-time latency requirements. 

---
# AgriJev: Efficient Agricultural Candidate Scoring with Multi-Scale Adapted Models 

**Authors**: Xuechen Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.19478)  

**Abstract**: Agricultural systems make decisions over heterogeneous entity types, textual judgments, and knowledge-question options. Generating a textual answer adds decoding and extraction steps to decisions that ultimately require a candidate choice. We introduce AgriJev, a family of adapted Qwen3.5 models at 0.8B, 2B, 4B, and 9B. A shared interface conditions on agricultural text, a judgment question, and candidate definitions, returning scores and relative probabilities in one forward call. We prepare source-aware training records and frozen evaluations from existing agricultural resources, retaining original groups and annotation provenance. Comparisons include original Qwen3.5 models and matched agricultural SFT models, both evaluated through native answer generation. Across 17 external test views, AgriJev-9B reaches 75.44% mean accuracy, exceeding the original and SFT generators by 5.89 and 3.79 percentage points. It outperforms the original generator on 13 views and narrows the mean gap to hosted Jev 1.13.0. On a fixed A100 workload, its median scoring latency is approximately half that of same-scale answer generation. Task-dependent gains and failures characterize a practical choice among agricultural model weights and inference contracts. 

---
# More Than Seeing: Vision as Scalable Anchors in Native Multimodal Pre-Training 

**Authors**: Haoyuan Wu, Aoqi Wu, Jiani Hu, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19443)  

**Abstract**: Native multimodal pre-training trains a single model from scratch on text and multimodal data, enabling it to understand both language and visual inputs. However, how visual signals shape scaling behavior and support capabilities beyond visual understanding remains unclear. We investigate this question through controlled visual-content ablations across model sizes and training budgets. Specifically, we replace image embeddings with content-free padding token embeddings. Retaining visual content improves text prediction on image-text examples without materially changing the lowest text-only loss at a given compute budget, the optimal balance of model size and training data, or aggregate language scores. The relative multimodal loss benefit grows approximately as a power law of compute. To test benefits beyond understanding images, we evaluate spatial reasoning from text descriptions alone. Both conditions improve with model and data scaling. During training of the largest model, visual gains are most sustained in localization, navigation, and symbolic planning, which involve maintaining object configurations or composing spatial updates; gains in spatial relations, geometry, and physical commonsense are less consistent. These findings identify visual content as a scalable anchor for multimodal learning, with benefits that extend beyond visual inputs while preserving aggregate language performance. 

---
# How Do LLMs Understand Legal Philosophy? Diagnosing Legal-Philosophical Reasoning in LLMs 

**Authors**: Shuyuan Zheng, Yun-seo Lee, Xing Gao, Jialu Wu, Jiabao Hu, Katrina Qiyao Wang, Xinbo Lin, Qianying Liu, Chuan Xiao, Tao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19427)  

**Abstract**: Legal reasoning is a fundamental capability for large language models (LLMs) to complete legal AI tasks such as legal judgment prediction. In legal practice, this reasoning is shaped by legal philosophy: a judge's view of the nature and sources of law guides how legal materials are interpreted and applied, so judges working from the same statutes and precedents can reach different judgments. Existing research has largely overlooked whether LLMs can understand legal philosophy and apply it in legal reasoning, an ability we call \textit{legal-philosophical reasoning}. To fill this gap, we introduce LePhilBench, a benchmark for evaluating the legal-philosophical reasoning of LLMs that comprises two benchmarking tasks and a dataset. The two tasks evaluate how well LLMs understand legal philosophy and how well they apply it in legal reasoning, respectively. The dataset is manually and meticulously constructed. Its test set contains 220 U.S. Supreme Court opinions. A law professor designed fourteen annotation questions on legal philosophy, and three law-trained annotators with formal training in jurisprudence answered all fourteen questions for every opinion, yielding 9,240 legal-philosophical labels. LePhilBench takes the first step toward exploring legal-philosophical reasoning in LLMs. 

---
# ForkGRPO: Efficient Exploration and Fine-Grained Credit Assignment for Diffusion Language Models 

**Authors**: Zhaoyan Jia, Chengwei Zhou, Gourav Datta, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2610.19408)  

**Abstract**: Diffusion large language models (dLLMs) have advanced rapidly, yet strengthening their reasoning through reinforcement learning (RL) remains an open challenge. Any-order generation can suffer from entropy degradation, which narrows the reasoning paths available for exploration. Recent work mitigates this issue by applying standard GRPO with left-to-right generation. However, independently generating every rollout from the input prompt can leave alternative continuations from intermediate states underexplored, while trajectory-level advantages assign uniform credit to all tokens without distinguishing the contributions of individual reasoning segments. We propose ForkGRPO, an RL framework that couples efficient exploration with fine-grained segment-level credit assignment. It samples multiple continuations from high-entropy intermediate states along a reference trajectory, prioritizing textual boundaries and reusing shared prefixes. The continuations provide complementary learning signals: within-state group-relative advantages guide which continuations to reinforce, while Monte Carlo state-value differences indicate which reasoning segments improve expected outcomes. Among the evaluated architecture-preserving RL methods, ForkGRPO achieves the best performance on three of four benchmarks, while improving over the directly comparable JustGRPO by up to 1.2 percentage points and using up to 27.8% fewer generated tokens. ForkGRPO-Fast further achieves a $2.49\times$ training wall-clock speedup with negligible accuracy loss. 

---
# Vision-Language Models Integrate Conflicting Evidence Like Biased Bayesian Observers 

**Authors**: Josh Barua, Kyle Mahowald, William Rudman  

**Link**: [PDF](https://arxiv.org/pdf/2610.19407)  

**Abstract**: Vision-language models must reconcile conflicting text and images. Prior work has characterized the resulting behavior in terms of modality biases--e.g., a "blind faith in text"--but why preferences vary across models, domains, and input conditions remains unclear. We develop a predictive account from a Bayesian perspective, positing that VLMs combine conditionally independent evidence from each modality with prior beliefs, subject to modality-specific biases. The resulting two-parameter cue combination model accurately predicts VLM preferences under unseen combinations of input degradation and transfers to new forms of unreliability without refitting. Separating evidence strength from integration bias explains how VLMs can follow text even while weighting visual evidence more heavily, reconciling reports of text and vision dominance under a single combination rule. 

---
# KoAGE: Do Large Language Models Know, Learn, and Coin Korean Neologisms? 

**Authors**: SungHo Kim, Eojin Jeon, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.19399)  

**Abstract**: Korean neologisms arise through diverse mechanisms, a demanding test of language models beyond lexical memorization. To enable systematic evaluation, we introduce NeoTrace, an automated, rerunnable framework that collects and dates neologisms across genres. Its 4,271 entries support KoAGE, a cutoff-aware benchmark with three acquisition and two generation tasks---the latter, to our knowledge, the first to compare model coinages with attested neologisms under their original materials and community context. Across 21 models, lexical recall diverges from contextual inference: in a five-model temporal analysis, models define only 13-30% of cutoff-year words, yet can infer meaning from one usage. However, structural understanding remains limited: even correct definitions barely improve formation analysis of pre-cutoff neologisms. Given word-building materials, single models often follow human formation mechanisms, whereas multi-agent coinage favors abbreviation and depends on interaction framing. KoAGE thus separates what models know, learn from use, and coin. 

---
# ELARSpeech: A 30K-Hour Dataset Across 675 Underrepresented Languages for Long-Tail Multilingual Speech Technologies 

**Authors**: Sangmin Lee, Yejin Jeon, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2610.19388)  

**Abstract**: Massively multilingual speech modeling has made remarkable progress in recent years, yet existing training resources and benchmarks still underrepresent much of the linguistic long tail. To address this gap, we introduce ELARSpeech, a large-scale multilingual speech resource comprising 30,202 hours of speech across 675 underrepresented languages, curated from the Endangered Languages Archive. Combining human-annotated and unlabeled recordings, the corpus spans diverse recording conditions, domains, and language families. Beyond corpus construction, our experiments show how ELARSpeech supports extending ASR and TTS capabilities to diverse underrepresented languages, with further benefits from pseudo-labeling unlabeled data. At the same time, our speech representation analysis reveals remaining limitations in transferring existing multilingual models to the linguistic long tail. Together, ELARSpeech provides both a resource and a testbed for extending multilingual speech technology further into the linguistic long tail. We publicly release the dataset, all trained ASR and TTS models, and the complete data construction pipeline to facilitate research on long-tail multilingual speech technologies. 

---
# The Sum of All Summaries: PROPALIGN for Graded Proposition Salience Evaluation 

**Authors**: Lin Ai, Emma Thronson, Katherine Conhaim, Amir Zeldes  

**Link**: [PDF](https://arxiv.org/pdf/2610.19387)  

**Abstract**: Identification and ranking of salient propositions in documents is indirectly involved in many AI tasks, but few papers define proposition salience scores clearly, make diverse data available, or evaluate models directly on the task. This paper proposes PROPALIGN, a multiple-summary-based approach for propositional salience score estimation, and releases PROPALIGNBANK, an accompanying dataset for the method with 138K alignments of over 35K propositions. We compare several approaches to salient proposition extraction in English across a broad range of spoken and written genres, and demonstrate that naively asking models for judgments leads to inconsistencies. By contrast, graded, summary-based metrics are more stable. We further observe that while single models perform worse than humans, an ensemble of models incorporating linguistic features approaches manual annotation quality ceilings. In addition, analyses of the model outputs suggest insights into how models can extract salient information with and without the use of summaries. 

---
# Learning What to Keep: Transfer-Aware Personal Skill Learning from User Feedback 

**Authors**: Li Wang, Chen Zhang, Yirong Yang, Xingyu Chen, Jinkun Zhao, Tianhao Peng, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19380)  

**Abstract**: User feedback enables LLMs to refine reusable personalized skills, but often mixes persistent preferences with request-specific corrections. Indiscriminately retaining both risks negative transfer, motivating selective retention at the level of individual requirements. We frame this challenge as selective transfer over atomic rules and propose SkillGrain, a framework for learning reusable personalized skills from user feedback. SkillGrain decomposes feedback into independently actionable rules, predicts their transferability, and retains only those suitable for future requests. To learn these decisions, we combine rubric-guided skill rewards with fine-grained credit assignment that separately evaluates rule content and retention decisions. We further jointly optimize skill induction and skill-conditioned generation through reinforcement learning, ensuring that learned skills support downstream personalization. We construct a benchmark spanning five personalized writing tasks to evaluate cross-request generalization. Experiments across three base models in both in-domain and out-of-domain settings show that SkillGrain achieves the best overall average performance among competing methods. 

---
# Evaluating Compressed Memory: Dependence, Utility, and Content Benefit 

**Authors**: Minsu Oh, Seunghui Jwa, Chanjun Park, Yeo-Chan Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2610.19373)  

**Abstract**: Does a language model answer better because its compressed memory contains useful document information? Replacing memory vectors with zeros can lower a trained model's score without showing that it outperforms a model trained without memory. We combine zeroing with separately trained models that cannot access stored text or receive the same constant input for every document. Across nine systems, some models rely on memory yet perform worse than the trained alternative. For example, in experiments that predict states in generated stories, zeroing memory worsens a model's prediction loss, yet a model trained without memory achieves lower loss than the original model with intact memory. On TriviaQA, giving xRAG memory from another question's passage raises F1, which measures word overlap with accepted answers, from 19.38 without a document to 24.78. Its effect on how often outputs contain an accepted answer is statistically uncertain. The question's retrieved passage scores higher still (29.88 F1). Thus a gain over no document alone does not establish that the gain requires the question's passage. Evaluations must therefore specify the comparison model, absolute performance, and generation and scoring conditions. 

---
# Beyond Corrected Actions: Training LLM Agents on Textual Critiques 

**Authors**: Thomas Zeng, Zelei Cheng, Mingyang Zhou, Sambit Sahu, William Campbell, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2610.19367)  

**Abstract**: Large language models (LLMs) are increasingly used for multi-turn agentic tasks that require interactions with users and environments over extended sequences of actions. A common approach to improving LLM agents is to use more capable expert models to provide training supervision, either by behavior cloning the expert's trajectories or by using the expert to correct the actions in trajectories generated by the weaker student model. However, expert LLMs can provide richer supervision than actions alone: they can directly generate textual critique that explain why specific actions are incorrect. We study whether training on such feedback benefits LLM agents and propose Critique-Augmented Agentic Fine-Tuning (CAFT), a framework that elicits textual expert feedback on student-generated trajectories. For each incorrect action, the expert yields both a corrected action and a textual critique explaining the student's mistake--which we then use as the training target for fine-tuning. Across four student models from the Nemotron 3 and Gemma 4 families on the agentic domains $\tau^2$-Bench Airline+Retail and WebShop, adding critiques to action-level supervision improves performance by 8% on average over action-only distillation. Additionally, CAFT checkpoints can also provide a better warm start for reinforcement learning than expert-trajectory distillation. Our results demonstrate that textual critiques can strengthen supervision for multi-turn LLM agents. 

---
# Tail Competition Explains Scaling in Best-of-N Sampling for Verifiable Problems 

**Authors**: Taro Yano, Yoichi Ishibashi, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2610.19362)  

**Abstract**: Best-of-N sampling is a simple inference-time scaling strategy for large language models, yet its behavior under incomplete reward models remains poorly understood. We study Best-of-N as a competition between the extreme reward tails of correct and incorrect responses. This view shows that success depends not only on the probability of sampling a correct response, but also on whether correct responses dominate incorrect ones in the upper tail. Using extreme value theory, we characterize three asymptotic regimes: correct-tail-dominant, incorrect-tail-dominant, and critical explaining both monotonic gains and reward-hacking-induced degradation. We further derive finite-sample scaling laws for representative reward distributions, showing that, under Gaussian rewards, incorrect responses with sufficiently large variance can dominate selection even when their mean reward is lower. Unlike oracle selection, convergence with an incomplete selector can be polynomial rather than exponential. Finally, we propose a lightweight tail-tension criterion to estimate when additional sampling may become harmful. Experiments on synthetic data and real reward model scores are consistent with the theoretically predicted regimes, scaling laws, and diagnostic behavior, suggesting that tail competition provides a unified lens on Best-of-N sampling for verifiable problems. 

---
# Target-Token Asymmetry Can Masquerade As Depth Specialization 

**Authors**: Danlin Li, Miao Li, Xu Han, Jiecheng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19344)  

**Abstract**: Layer-ablation studies often perturb early or late layers of a language model, compare the damage to prediction on two corpora, and read a larger early-layer effect on one corpus as early-layer specialization for its task. Such a comparison averages over the tokens that each corpus asks the model to predict, and these target tokens differ: math solutions, for example, ask for far more whitespace and numbers than encyclopedia prose. We show that this difference can create the appearance of depth specialization. We split a cross-corpus early--late damage interaction by target-token type and recompute it under a target mixture shared by both corpora. Between GSM8K and Wikitext, a positive interaction replicates across five model instances. In the three fully decomposed models, only whitespace has a positive interaction. Removing whitespace reverses the pooled effect in all three, and matching the full target mixture reverses it in two. The diagnosis holds under position matching, adversarial reassignment of token types, and a continuous readout. The risk can also be anticipated from the tokenizer alone. A screen computed before any model runs places every held-out pair it scored in the region its label predicts, and on the pairs it rates at high or medium risk, removing whitespace reverses every positive pooled estimate. A smaller content-word effect one token ahead follows the share of content targets that the unablated model already predicts. A cross-corpus depth effect therefore supports an interpretation of task specialization only after the target-token mixture is reported, the effect is decomposed by token type, and a mixture-matched summary keeps its sign. 

---
# LLM-jp-4 Technical Report 

**Authors**: LLM-jp, Chaoran Liu, Daisuke Kawahara, Hideyuki Tachibana, Hirokazu Kiyomaru, Jun Suzuki, Koshi Eguchi, Kouta Nakayama, Qianying Liu, Rio Yokota, Ryo Takahashi, Sadao Kurohashi, Sosuke Hosokawa, Taishi Nakamura, Takashi Kodama, Yun-Ang Wu, Yusuke Miyao, Yusuke Oda  

**Link**: [PDF](https://arxiv.org/pdf/2610.19331)  

**Abstract**: This paper presents LLM-jp-4, a family of Japanese sovereign large language models (LLMs) developed with an emphasis on transparency, reproducibility, and practical usability. LLM-jp-4 comprises an 8B dense model and a 32B-A3B mixture-of-experts model, both trained from scratch on approximately 12 trillion tokens. To support independent verification and reproduction, we establish explicit criteria for selecting training data, external models, and software, and make the development process publicly available. Evaluation results demonstrate substantial improvements over previous LLM-jp models and competitive performance against open-weight models of similar scale. By documenting the development principles, training procedures, and empirical findings of LLM-jp-4, we aim to provide a reference for future sovereign AI development. 

---
# TopicMerge: A Divide and Conquer Approach to Topic Modeling 

**Authors**: Matthew Raffel, Lizhong Chen, Sai Munikoti, Sameera Horawalavithana  

**Link**: [PDF](https://arxiv.org/pdf/2610.19323)  

**Abstract**: Recent advancements in topic modeling have split into two main paths: efficient embedding-based methods that offer scalability but are often outperformed on standard clustering metrics, and Large Language Model (LLM) approaches that achieve state-of-the-art clustering performance but at the cost of prohibitive computational expense and poor scalability. This paper introduces TopicMerge, a novel multi-stage pipeline that bridges these approaches to achieve both state-of-the-art clustering performance and computational efficiency. TopicMerge first employs an efficient clustering pipeline to deliberately over-segment a document corpus into a large set of fine-grained, semantically pure micro-clusters, which are then given initial semantic labels. It then introduces a novel, iterative merging algorithm where an LLM, guided by local neighborhood context, arbitrates on consolidating candidate pairs. These approved merges are then selectively executed and semantically relabeled by the LLM, hierarchically building broader, more coherent topics. Extensive experiments on the Bills and Wiki datasets show that TopicMerge achieves state-of-the-art clustering performance while reducing required LLM calls by over 92% compared to document-level methods, offering a practical and scalable solution. 

---
# PersEvalBench: Can LLM Evaluators Judge Contextually Appropriate Use of Persona Information? 

**Authors**: Xinyi Liu, Hooshang Nayyeri, Emine Yilmaz, Dilek Hakkani-Tur, Charith Peris, Hari Thadakamalla  

**Link**: [PDF](https://arxiv.org/pdf/2610.19314)  

**Abstract**: Large language model (LLM) evaluators are increasingly used to assess personalized assistants, yet whether they can reliably judge personalization has rarely been tested. We introduce PersEvalBench, a benchmark testing whether an evaluator can recognize appropriate and inappropriate uses of user history. It contains 403 multi-episode scenarios where user information should persist, transfer, or cease to apply. The benchmark covers nine controlled error subtypes across three families: under-personalization, over-personalization, and misapplied-state errors. Human audits confirm that generated errors match their intended types and that the prescribed uses of user information are reasonable.
Across five LLM evaluators, recognition of controlled personalization errors remains limited even with the full dialogue history. On the human-evaluated subset, under-personalization shows the largest gap: humans recognize 94.4% of these errors, compared with 63.9% for the strongest reported evaluator. Evaluators often recognize that a response is problematic but reverse the error direction, mistaking failure to follow an applicable preference (UNDER) for use of an inapplicable preference (OVER). Easier evidence access offers little help: highlighting the relevant history turns raises error recognition only slightly, suggesting that the remaining difficulty lies in judging what role that information should play in the response. 

---
# Qwenfant: Train Big Character-level BabyLMs on Concise Parallel Text 

**Authors**: Lukas Edman, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2610.19294)  

**Abstract**: This work concerns our submission for the Multilingual track of the 2026 BabyLM Challenge. We improve on our baseline using data that has been summarized and translated by an LLM, showing that higher-quality data can come from synthetic sources. We further improve with 2 simple changes: using a larger model, and using a character-level vocabulary. Both of these changes require LoRA-style fine-tuning for realizing their full potential. We dub our final Qwen3-based model Qwenfant, which, at the time of submission, ranks 1st in the Multilingual track. 

---
# Can LLM Agents Efficiently Identify Appliance Faults? Insights From Diagnosing a Simulated Appliance 

**Authors**: Lucas H. McCabe, Sarah Lukens, H. Howie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19291)  

**Abstract**: In so-called "Industry 4.0," integrated sensors, analytics tools, and other digital technologies supplement industrial workflows, potentiating enhanced visibility and automation. Prognostics and health management (PHM) and natural language processing (NLP) practitioners alike have explored the application of large language models (LLMs) to industrial asset management, but many prior works' focus on exam-like benchmarks has limited their assessment of LLMs' ability to automate practical operations. Herein, we introduce CoffeeBench, a fault diagnostics task for LLM agents, using a simulated appliance. The LLMs we evaluate struggle to consistently identify faults from the sensor information alone, but the best-performing ones can iteratively refine the fault-option space to eventually obtain a successful diagnosis. Our subsequent analyses, therefore, focus on: differentiating models by diagnostic efficiency (RQ1); examining the lagging performance of the three worst models (RQ2); and identifying viable heuristics for selective prediction (RQ3), including that initial diagnostic uncertainty and LLM-judged soundness of reasoning are high-signal predictors of diagnostic success. In addition to our benchmark results (Section 3.1), we provide practical insights for improving the performance of the open-weight models on CoffeeBench, such as by explicitly tracking episode state on their behalf to reduce timeouts (Section 3.2) or applying a deliberation loop to refine diagnostic hypotheses (Section 3.3). 

---
# Whose Data, Whose Benchmark? A User-Centric Framework for Auditing Evaluation Resources 

**Authors**: Mir Tafseer Nayeem, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2610.19268)  

**Abstract**: Benchmarks shape how model capabilities are measured and interpreted, yet their scores can obscure whose conventions, contexts, and judgments define successful performance. We introduce a user-centric audit framework that distinguishes users who contribute benchmark content from those whose needs evaluation results are intended to represent. The framework operationalizes seven dimensions across benchmark content, metadata, documentation, and scoring procedures. Across 16 resources, we find that American-centered conventions account for 62.4-94.4% of matched signals in 11 text benchmarks, while geographic visibility in five image-text resources is concentrated and strongly associated with national GDP share (mean Pearson r = 0.780; Spearman p = 0.778). Documentation reveals a systematic source-target gap: all 16 resources operationalize source-user context, but none operationalize target-user context. Paired scoring experiments further show that meaning-preserving answer variants reduce DROP exact-match accuracy from 100.0% to 33.3%, while TruthfulQA exhibits metric-dependent reductions in reference similarity. Our framework and documentation checklist make user assumptions auditable, clarifying what benchmark scores represent and where evaluation claims are supported. 

---
# Copy-Paste Language Models 

**Authors**: Nathan Godey, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2610.19247)  

**Abstract**: Transformers are intuitively considered to be efficient at copying from their inputs. We actually show that the ability of LLMs to copy in context can be improved through explicit modeling. We propose Copy-Paste Language Models (CPLMs), which add a lightweight Copy-Paste head to the conventional language modeling (LM) head. CPLMs improve training efficiency, downstream performance, and offer new operating points in long-context and retrieval setups. We validate our approach by pretraining models from 135M to 1.7B parameters, showing improvements across all scales and backbones with negligible training and inference overheads. We explain the success of CPLMs by showing that they partially mitigate the LM head gradient bottleneck through an alternative gradient path. 

---
# Do Syntax Heads Influence Model Behavior? 

**Authors**: William Timkey, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19231)  

**Abstract**: Transformer language models develop attention heads that specialize in syntactic relations, for example, attending from verbs to their subjects. Do these heads play a causal role in the model's syntactic predictions? Evidence for such a role is mixed to negative: even when we force these heads to attend to a distractor noun other than the subject, the model's syntactic predictions are still more consistent with the subject than with the distractor. Interventions on specialized syntax heads had a significantly stronger effect on syntactic predictions than interventions on random heads, but only in one of the two measures we propose. 

---
# Titan-Llama: Retrofitting Pre-trained LLMs with External Neural Memory Modules 

**Authors**: Sarah Pan, Ali Behrouz  

**Link**: [PDF](https://arxiv.org/pdf/2610.19210)  

**Abstract**: Transformers are efficient to pre-train at scale because attention permits parallel computation across sequence positions, but long-context inference requires increasingly large KV caches. Efficient recurrent alternatives reduce the asymptotic complexity of inference, but their sequential dependence limits token-wise parallelism during training, making them harder to scale. Further, the current paradigm of extending long context tasks creates competing notions of retaining information precisely and keeping long-context inference efficient. To address this gap, we introduce Titan-Llama, a post-training method for retrofitting attention-based language models with a hierarchical memory architecture that updates parameters at test-time. Titan-Llama applies a blockwise attention mask and preserves full-fidelity attention over recent tokens while compressing historical context into recurrent, fixed-size neural memory modules (NMMs). Our formulation allows pre-trained attention layers to be replaced without training a new language model from scratch. On Llama-3.2-1B-Instruct, Titan-Llama raises accuracy over segmented attention windows from 23.87\% to 40.51\% on QuALITY and from 14.50\% to 40.75\% on LongHealth. Inference benchmarks show more than 20\% faster prefill, 37\% faster decode at medium context lengths, and 24\% lower steady-state decode memory usage in the evaluated configurations. Together, these results demonstrate that incorporating memory architecture mechanisms that utilize test-time training effectively modulates long-context behavior of pre-trained models, offering a practical route to adapting pre-trained transformers for efficient long-context inference. Our code is released on Github \href{this https URL}{GitHub} 

---
# DUALVIEW: Leveraging Broad and Output-Specific Evidence for LLM-based Multi-Output Text Regression 

**Authors**: Yiheng Su, Junyi Jessy Li, Matthew Lease  

**Link**: [PDF](https://arxiv.org/pdf/2610.19206)  

**Abstract**: While work on LLM-based regression has largely focused on scalar prediction, multi-output text regression remains less explored. We present DualView, a parameter-efficient approach to leverage LLMs for multi-output text regression. \method{} first extracts broad and output-specific representations from a frozen LLM, then uses a dual-branch architecture to produce the full score vector. Across five heterogeneous regression tasks and eight LLMs (up to 122B), DualView outperforms seven supervised regression baselines while requiring 82--1842 times fewer fitted parameters than the three strongest baselines. These improvements extend across individual output dimensions and persist even with limited training data. Notably, DualView using a smaller open-weight LLM can outperform unsupervised prediction with Gemini 3.1 Pro. Controlled ablations show that broad and output-specific evidence provide complementary predictive benefits, while lightweight shared predictors achieve strong performance with fewer trainable parameters. 

---
# Weighted Ensemble Sampling for Rare-Event Estimation in Language Models 

**Authors**: Zixuan Liu, Gustavo Didier, Fangzheng Wu, Brian Summa, Zizhan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.19202)  

**Abstract**: Rare but severe failures of large language models can become practically significant at deployment scale. Estimating their probabilities with direct Monte Carlo sampling is highly inefficient, while proposal-based methods typically require auxiliary learning of an additional proposal distribution. We introduce \textbf{Weighted Ensemble (WE)}, a training-free framework for rare-event estimation in language models. WE samples exclusively from the original language model and improves upon direct Monte Carlo by periodically reallocating the sampling budget toward prefixes that appear more likely to reach the target rare event. To guide this reallocation without learning a separate proposal distribution, we exploit internal LLM representations as prefix-level signals and empirically validate their effectiveness. We provide a theoretical analysis of the proposed algorithm. Experiments on safety-related tasks spanning both small-scale and modern large-scale LLMs demonstrate that WE improves the efficiency and reliability of rare-event probability estimation. 

---
# NEPSEE: A Double-Annotated Benchmark for Nepali Sentiment Analysis 

**Authors**: Gaurab Baral, Mimansak Nepal, Grishma Basnet  

**Link**: [PDF](https://arxiv.org/pdf/2610.19190)  

**Abstract**: Many sentiment datasets use one annotator for each item, so the test labels show the conventions of that person. We build a Nepali benchmark from two annotators who labelled the same sentences blind. They labelled 5,614 sentences from tweets of Nepali public figures on a five-point scale. We keep only the 3,189 sentences where both give the same polarity. On the three-class benchmark, NepBERTa with a fixed tokenizer gets a macro-F1 of 0.877 and IndicBERTv2 0.867. Three commercial large language models (LLMs) get 0.882 to 0.896 zero-shot, and these differences are not significant. On the five-point scale, 766 sentences have two adjacent labels, for example Pos and SPos. We accept both labels and train on both. On this task, Claude Opus 5.5 beats the best encoders by 0.10 to 0.12 macro-F1. A tokenizer audit adds a second result. The default tokenizers of three Nepali BERT models delete the non-spacing Devanagari vowel signs and the virama. When we restore the marks, macro-F1 increases by 0.084 for NepBERTa but decreases by 0.095 for a model pretrained on stripped text. 

---
# How to Evaluate When You Don't Know How to Evaluate? Choosing Reliable Setups for Base Language Model Benchmarking 

**Authors**: Shenbin Qian, David Samuel, Jindřich Helcl, Andrey Kutuzov, Stephan Oepen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19185)  

**Abstract**: Evaluation of large language models (LLMs) remains largely pre-scientific: it involves many design decisions, with little guidance on how to make them. These choices can substantially affect measured performance and, consequently, decisions about subsequent pretraining. To identify effective evaluation setups, we investigate 8 aspects, such as task formulation or the number of in-context examples. We compare the setups using three widely-used signal criteria, repurposed as normative meta-measures: monotonicity, non-randomness, and ranking consistency of the evaluation scores across intermediate checkpoints. In an early pretraining scenario, we evaluate these setups on 26 monolingual 2B-parameter models trained on different types of data, using 80 tasks in four languages. We also examine the setups on three larger LLMs ($\geq$7B) in the same four languages and explore how the preferred setups change with model scale. Our results yield concrete recommendations: some choices, such as the use of multiple prompts, are consistently useful, whereas others, such as task formulation, depend on model scale. More generally, our framework serves as a recipe for the choice of evaluation setups in new scenarios. 

---
# HERO: Harnessing Evidence for Test-time Refinement and Prompt Optimization in Program Generation 

**Authors**: Mingyu Lu, Timothy Ossowski, Tristan Naumann, Hoifung Poon, Peniel Argaw  

**Link**: [PDF](https://arxiv.org/pdf/2610.19184)  

**Abstract**: Harness optimization improves LLMs by refining reusable prompts and scaffolding, yet requires further adaptation for program generation tasks such as text-to-SQL and text-to-Python. Existing methods often rely on LLM reflectors to propose harness updates, which can fix observed failures but also disrupt previously correct behavior. Text-to-code tasks offer a natural way to mitigate this issue: we can execute generated outputs, allowing us to evaluate and verify the effects of proposed updates directly. Building on this signal, we introduce \textsc{HERO} (\textbf{H}arnessing \textbf{E}vidence for \textbf{R}efinement and \textbf{O}ptimization), which combines offline and test-time harness optimization. At the first stage, \method{} learns reusable prompt updates from execution-verified successes and failures and validates candidate revisions before adoption. At test time, we treat the generated code as an instance-specific harness and refine it using execution feedback. Across four SQL, Cypher, and Python benchmarks, \method{} achieves the best offline performance, improving pass@1 by \(7.6\) points over the initial prompt, while test-time refinement adds another \(2.0\) points and improves execution efficiency. 

---
# Fantastic Metrics and How to Find Them: Meta-Evaluating Metrics for Multi-lingual Style-Personalized Generation 

**Authors**: Anubhav Jangra, Adrian de Wynter  

**Link**: [PDF](https://arxiv.org/pdf/2610.19166)  

**Abstract**: Meta-evaluation of text generation metrics and LLM-judges is primarily conducted using metric-human correlations. However, prior work has shown that correlation is not an appropriate estimate of a metric's evaluation capability and can be unstable across evaluation settings, such as different datasets or languages. Therefore, we propose a regression-based meta-evaluation of metrics that learns on calibrated metric outputs to predict human ratings. We ground this measurement relative to a trivial constant-mean baseline and human agreements to provide a comparable measure of metric capability across evaluation settings. To test our meta-evaluation framework, we developed a large-scale multi-lingual style-personalized text generation (SPTG) benchmark, collecting multiple annotations across 10 topologically diverse languages and four evaluation dimensions, each with varying degrees of subjectivity. We meta-evaluated 32 metrics from three style evaluation paradigms (string-based, embedding-based, and LLM-judges) and their ensembles over these 40 evaluation settings. Our analysis revealed that even simple string- and embedding-based measurements can match strong LLM-judges when calibrated to human ratings. We also identified characteristics of human annotations that determine metric performance, as well as conditions under which a metric ensemble outperforms its individual constituents. 

---
# From Qualitative Text to Integrated Evidence: The Parallel-Then-Integrated (PTI) Procedure 

**Authors**: Elyas Irankhah, Ashley Hagaman, Samah Fodeh  

**Link**: [PDF](https://arxiv.org/pdf/2610.19164)  

**Abstract**: Mixed methods research offers established strategies for integrating qualitative and quantitative evidence, but less guidance for integrating two computational representations of the same qualitative corpus. This methodological study introduces Parallel-Then-Integrated (PTI), a four-step procedure for explicit and traceable integration. We present PTI using 1,203 interview paragraphs from 29 mothers in Nepal. The two representations were ontology-guided Patient Voice (PV) coding with PVminer and data-driven themes generated through non-negative matrix factorization (NMF). A theme-complete crosswalk documented convergence, divergence, and preprocessing-sensitive differences. Depression status, based on Patient Health Questionnaire-9 (PHQ-9) scores, served as a secondary external anchor for exploratory association checks. PTI provides a structured approach for integrating computational representations of qualitative text within mixed methods research. 

---
# CAFE: Confining Feature Distillation to Language-Routed Adapters for Compact Multilingual Automatic Speech Recognition 

**Authors**: Tianchi Liu, Nicholas Moratelli, Khurram Azeem Hashmi, Mohammadreza Zolfaghari, Rishabh Jain, Louis Lu, Changdae Park, Amril Nazir  

**Link**: [PDF](https://arxiv.org/pdf/2610.19134)  

**Abstract**: Feature distillation is usually expected to add to logit distillation. We find that this expectation does not hold for two common feature losses in compact multilingual automatic speech recognition (ASR), and analyze why through their gradients. The gradient of a projection-based loss is orthogonal to the task gradient, while that of Centered Kernel Alignment (CKA), a similarity-based loss, is weakly aligned with it. CKA lowers the word error rate (WER) on non-English languages, but at the expense of data-rich English. Three loss-level remedies (one trained, two probed) all fall short, and the CKA gradient overlaps with the subspace English uses in the shared encoder by 97.5% to 99.5% (about 81% by chance). Guided by this diagnosis, we propose CAFE (Confined-Adapter Feature distillation), which confines the CKA gradient to adapters that English bypasses, addressing the trade-off in the parameters rather than in the loss. On the Open ASR Leaderboard, CAFE lowers the average WER from 8.55% to 8.02%, beats an adapter-only capacity control at 8.51%, and makes English more stable across checkpoints. At 254M parameters, it trails the 0.6B teacher by 2 WER points and is competitive with much larger models across 48 test sets in 24 languages. We use 564k hours of training data and state-of-the-art models to keep the findings relevant to practice. 

---
# DecisionRobust: Benchmarking Robustness for JEV Style Decision Models Across Text and Vision 

**Authors**: Yifan Lan, Zhongye Liu, Yurui Chang, Hanyu Wang, Yongkang Du, Lu Lin, Jinghui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19118)  

**Abstract**: Emerging decision models such as JEV provide typed interfaces that map task instructions and observations directly to choices, judgments, and scores. As these models are increasingly used for automated decision-making, understanding their reliability under adversarial manipulation becomes critical. We introduce DecisionRobust, a benchmark for systematically evaluating the robustness of decision models across textual and visual inputs. Our benchmark covers practical decision-making scenarios and a diverse suite of attacks, including semantics-preserving text perturbations, image perturbations, and prompt injection in both modalities. We curate evaluation sets through external validity checks and cross-model candidate selection. We evaluate 30 model configurations on text tasks and 17 on vision tasks, including 24 and 12 held-out models, respectively, that are excluded from candidate selection. Our experiments reveal several key findings. First, attacks expose vulnerabilities beyond the models used for construction and selection, although optimized image perturbations transfer weakly to many targets even with multiple surrogates. Second, relative model robustness varies across tasks and attack types, and resistance to input perturbations does not necessarily imply resistance to prompt injection. Third, label-consistency checks substantially reduce measured attack success across text and image settings, highlighting the importance of validating apparent failures. The benchmark is available at this https URL. 

---
# From Copying to Composition: How Language Models Develop In-Context Learning 

**Authors**: Minglai Yang, Xinyu Guo, Renxiong Wang, Sunjie Hou, Daniel Yue Zhang, Yunzhong He  

**Link**: [PDF](https://arxiv.org/pdf/2610.19113)  

**Abstract**: In-context learning (ICL) allows language models to adapt their behavior from instructions and demonstrations in a prompt, but accuracy alone does not reveal how contextual information controls their predictions. We study the acquisition and trainability of ICL with a controlled diagnostic that separates copying a nearby demonstration from integrating information across the prompt. Across 17 models, about a third prefer the copied answer, a few are undecided, and the rest prefer the rule-consistent answer, which goes with higher accuracy and with control by the requested value rather than the closest conflicting example. Checkpoint analyses show that models begin prioritizing task-relevant prompt tokens over conflicting examples long before their accuracy improves beyond chance level. In matched training experiments, synthetic documents that define endings and then query them shift models toward compositional responses on the controlled diagnostic, whereas fine-tuning on definition exercises built from rewritten real documents improves accuracy when prompts reassign label meanings. The results suggest that ICL is not acquired as a single, uniform capability. Instead, prompt control, answer preference, construction without candidates, and transfer can develop along distinct trajectories. 

---
# Similar Models, Broken Odds: Where Luce's Axiom Fails in Chatbot Arena 

**Authors**: Soovadeep Bakshi, Apurba Bose, Aishwarya Bose, Sandip Nandi  

**Link**: [PDF](https://arxiv.org/pdf/2610.19095)  

**Abstract**: Chatbot Arena ranks language models with the Bradley-Terry model, which gives each model one score and so assumes Luce's choice axiom. Choice theory predicts where this fails: near-copies are told apart more decisively than their scores imply. On 1.8 million Arena votes, votes between versions of one model are 1.37 times as decisive as Bradley-Terry predicts, between sizes of one generation 1.26 times and between models of one lab 1.12 times; a release a year later gives consistent multipliers, and the effect survives alternative tie treatments, prompt subsets and topics, and timing checks. How alike two models' answers are worded and formatted tracks the effect. A correction with four to six parameters predicts held-out votes better than Bradley-Terry and than models with hundreds of parameters, mostly by modelling ties; frozen and carried to new releases in a test fixed in advance, its gains are concentrated in ties, and better head-to-head odds are not established. The leaderboard order barely moves. With language-model judges and three-answer menus, a near-copy takes votes mostly from its twin when choices are pooled over prompts (log-odds shift +0.10, 95% interval 0.05 to 0.16); within a prompt the shift is smaller and clear for two of four judges. 

---
# Pre-Trained Language Models as Zero-Shot Tools for Social Science Research 

**Authors**: Noah Dasanaike  

**Link**: [PDF](https://arxiv.org/pdf/2610.19048)  

**Abstract**: Record linkage and demographic inference from names are routine demands in social science. Most approaches to these problems require labeled training data that are difficult or costly to acquire. I show that pre-trained language models perform both tasks accurately without requiring task-specific training. For record linkage, I introduce EnsembleLink, an algorithm that uses pre-trained embeddings alongside string-based scorers to retrieve a short list of candidates and then identify the likeliest matches. EnsembleLink matches or exceeds the performance of most prior approaches while generalizing across scripts and formats that those methods are unable to handle. For ethnicity inference from names, I show that large language models (LLMs) achieve up to 83.8% accuracy on U.S. voter files, compared to 78.0% for name-supplemented BISG. Open-weight models offer reproducible implementations that approach these models in accuracy. Moreover, LLMs are able to infer group membership for religious, caste-based, and other ethnic group categories outside of the United States, and reduce bias in downstream estimates of geographic racial gaps. Where pre-training encodes the world knowledge required by each task, task-specific labels add less information. 

---
# Crossing the Gender Binary: Modelling Gendered Reference in Science Fiction and General Fiction 

**Authors**: Yuxuan Liu, Haim Dubossarsky, Ruth Ahnert  

**Link**: [PDF](https://arxiv.org/pdf/2610.19028)  

**Abstract**: Science fiction is widely credited with unsettling gender. This article asks whether that holds at the linguistic level, in how readily gendered reference gives way to its counterpart, a neutral term, or other social category, and whether it separates the genre from general fiction. It does so by masking man/men, woman/women, and person/people across same-period science fiction and general fiction corpora and classifying RoBERTa's five likeliest substitutes by gender. Retention and displacement are permutation-tested over five resampling rounds, and the underlying sentences are read closely. Both corpora cross the gender binary in both directions, but unequally: where woman is masked, the model predicts a masculine term about twice as often as it predicts a feminine term where man is masked. Each mismatch marks a statistical surprise, a word the corpus did not expect there. Against this shared pattern, science fiction shows greater gender permeability than the general fiction corpus, placing man in feminine contexts more often than general fiction while the reverse rate holds steady. Its degendering tendency is stronger too, drawing both gendered terms toward the neutral. Beyond the binary, a masked woman also draws substitutes naming a social category and no gender far more often than a masked man does. In measuring the gender permeability of man, woman, and person in linguistic distribution and testing whether genre changes it, the study gives science fiction studies linguistic evidence for thematic claims and computational literary studies a pipeline applicable to other category vocabularies, such as race and class. 

---
# Joint Subspace Sparse Attention for Efficient Long-Context Decoding 

**Authors**: Jiwon Song, Yoongon Kim, Jae-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.19026)  

**Abstract**: Sparse attention reduces the KV-cache traffic that dominates long-context LLM decoding, but practical methods largely operate at coarse granularity. Token-level selection offers finer control, yet presents two challenges: low-dimensional scoring may lose query--key structure needed for accurate ranking, while gathering scattered selected tokens can offset the resulting latency benefit. We introduce Joint Subspace Sparse Attention (JSSA) to address both challenges. For selection quality, JSSA learns a shared low-dimensional query--key subspace by minimizing token-score preservation error. We upper-bound the resulting non-convex objective with a tractable marginal objective that admits a closed-form spectral solution; direct refinement of the exact objective provides no measurable improvement in the tested settings. For execution efficiency, we develop a gather-free token-level indexed-attention kernel that supports distinct per-KV-head selections under the standard grouped-query attention (GQA) KV-cache layout. Across standard GQA, hybrid sliding-window/global attention, and Multi-head Latent Attention (MLA) models, JSSA consistently remains close to full-KV accuracy while avoiding the substantial generation-length inflation and architecture-dependent failures observed with competing methods. On an NVIDIA B200 GPU, JSSA achieves up to $6.43\times$ the decoding throughput of full-KV attention at 128K context. JSSA achieves reasoning accuracy comparable to model-native DSA without model training and reaches 96--99\% of its decoding throughput even at 700B scale large models. 

---
# Can LLMs Access their World Knowledge for Event Prediction? 

**Authors**: Zijun Wu, Zhitian Zhang, Hossein Hajimirsadeghi, Nikita Dvornik, Alexander Pashevich  

**Link**: [PDF](https://arxiv.org/pdf/2610.19021)  

**Abstract**: Recent work has questioned whether large language models (LLMs) can be useful for temporal event prediction. Despite vast pretrained world knowledge, LLMs often underperform simpler specialized models. We ask whether this knowledge is truly inaccessible, or whether current methods fail to present event prediction in a form LLMs can use. We argue for the latter and show that LLMs become effective event forecasters when the interface is designed around two choices. The first grounds each event in a natural-language description, and the second separates event semantics from the prediction space to prevent semantic bias from label associations. To test this, we introduce DescribeEvent, a benchmark of ~150K events across seven datasets and six domains with event-level descriptions. We find that capable LLMs substantially benefit from descriptions and symbolic labels, outperforming description-free variants and non-LLM baselines, while weaker models gain little. These results suggest that accessing LLM world knowledge for event prediction is an interface problem, and that event descriptions offer a practical way to elicit the knowledge LLMs already encode. 

---
# Evaluating Large Language Models for Multilingual Abstractive Summarization: A Comprehensive Study of English and Bengali 

**Authors**: Khan Mahi Al Atahar, Nazifa Tasneem, Ajwad Abrar  

**Link**: [PDF](https://arxiv.org/pdf/2610.19003)  

**Abstract**: Abstractive summarization remains challenging across languages, where linguistic quality and factual consistency may diverge, particularly in low-resource settings. We present a standardized zero-shot benchmark of eight closed-source and open-weight LLMs on size-matched English and Bengali XL-Sum test sets. Our evaluation combines lexical and semantic metrics with reference-free factuality verification and multidimensional LLM-as-a-Judge assessment, providing a comprehensive analysis of both surface-level quality and factual reliability in multilingual summarization. Results reveal substantial cross-lingual variation: GPT-5.6 Luna leads several English lexical metrics but does not maintain the same advantage in Bengali, where LLaMA 3.3 70B achieves the highest ROUGE and BLEU scores. Gemini 2.5 Flash obtains the highest observed Bengali factuality score. Furthermore, evaluation methods produce different model rankings, highlighting the limitations of relying on a single metric for multilingual summarization assessment. These findings demonstrate that strong English summarization performance does not necessarily transfer to Bengali and emphasize the need for comprehensive evaluation frameworks for low-resource languages. 

---
# Following the Flow the Law Requires: Training Voice Agents to Transfer to Unseen Regulated Procedures in Debt Collection 

**Authors**: Petmongkon Pornpichitsuwan, Krittin Pachtrachai, Touchapon Kraisingkorn  

**Link**: [PDF](https://arxiv.org/pdf/2610.18881)  

**Abstract**: Voice agents built on large language models (LLMs) now handle customer calls, but in regulated applications completing the task is not enough: the agent must follow a conversational flow set by law. In debt collection, it must confirm it is speaking to the debtor before mentioning the debt, identify itself and the creditor, state the amount, and avoid prohibited language. Every creditor turns these rules into its own procedure and revises it. We ask whether a small, self-hosted policy can be trained once and then follow regulated procedures it has never seen. One machine-readable specification per deployment, FlowSpec, generates the policy instruction, gates the tools, supplies the supervision and defines the tests; the agent speaks only approved catalogue sentences. We fine-tune a 9-billion-parameter model on 2,506 conversations from 297 randomized procedures and evaluate a Thai outbound-call agent on four unseen deployment procedures (three debt collection, one non-debt), one from a wholly withheld lineage. Documented checks, fixed by the procedure, rise from 67--71 percents to 93--100 percents, while grounded checks, fixed by the conversation, already pass at 97--100 percents before training; scenario pass rate rises from 15/43 to 40/43. An audit of the shared specification finds 11 defects, 3 of which its own metric did not expose. 

---
# One Task, Many Interfaces: Simple Cross-Interface Consistency Training Improves LLM Agent Robustness 

**Authors**: Artin Tajdini, Manpreet Kaur, Peng Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.20576)  

**Abstract**: Post-training binds the competence of an LLM agent to the interface it was trained on. Semantics-preserving perturbations of that interface, such as renaming an action or changing the notation, leave the underlying task unchanged yet make the agent worse at it. Scoring each sample independently means interface randomization only broadens this dependence from a single interface to the training interface distribution, so out-of-distribution perturbations still degrade the agent. We propose Cross-Interface Policy Regularization (CIPR), a self-supervised objective that replaces action names with uninformative identifiers and presents each training sample under two interfaces in which equivalent actions are identified by construction. CIPR penalizes the KL divergence between the action distributions induced under the two interfaces, so the policy behaves consistently across them. We evaluate CIPR in both supervised and reinforcement-learning post-training, where it improves robustness. On agentic function calling, held-out interface perturbations reduce the accuracy of a post-trained agent by nearly 25%, and CIPR recovers 10.5% while leaving accuracy on the unchanged interface intact. 

---
# Scaling Discovery Requires Forgetting 

**Authors**: Shangyin Tan, Ben Zhou, Lakshya A Agrawal, Jialin Zhang, Yilin Chen, Qiuyang Mang, Donghyun Lee, Dennis Jacob, Koushik Sen, Alex Dimakis, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2610.20563)  

**Abstract**: Automated research with LLM agents has shown promising results in various domains. Through iterative refinement and scored verifiers, LLM agents are capable of discovering new ideas. There are different ways to scale automated research: let one agent iterate for longer, run several agents in parallel, or further let the parallel agents communicate and build on one another's candidates. However, we find that all three strategies eventually plateau as the agents accumulate history. Each agent becomes anchored to its own past attempts, making fewer and smaller edits. Beyond this point, additional budget brings little further improvement. In this paper, we show that the key to scaling discovery further is to forget. Periodically discarding the search state, including every agent's session and the candidate history, and restarting from the best candidate found so far, unblocks the search and pushes the same budget further.
We implement forgetting in a team of agents with communication and evaluate it on four tasks spanning heuristic programming, kernel optimization and model training. The results show up to 15% improvement compared to parallel agents, and up to 39% compared to a single agent. Our results suggest that, for open-ended discovery, what an agent is allowed to remember matters as much as how much compute it is given. 

---
# SwarmRSI: Recursive Self-Improvement through Evolving Scientific Agent Swarm 

**Authors**: Qianru Meng, Xi Zhang, Konstantinos Papakostas, Congfeng Cao, Wenxiang Hu, Tianyi Lu, Xiao Zhang, Lun Zhou, Zaiqiao Meng, Zhaochun Ren, Xiaohui Yan, Yougang Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2610.20431)  

**Abstract**: Recursive self-improvement (RSI) in scientific research aims to improve agents' research capabilities through their own experimental experience. Scientific agent swarms can combine evidence and implementations from different research directions to guide their agents' subsequent experiments. However, poor experimental results can leave unclear whether to revise a hypothesis or its implementation, while overall program scores can hide components useful to other agents. To address these challenges, we introduce SwarmRSI, a framework that evolves scientific agent swarms for recursive self-improvement. We devise Feedback-Guided Swarm Mutation to refine hypotheses and their implementations using experimental feedback from other agents. We also propose Complementary Swarm Recombination to reuse retained hypotheses and complementary program components across agents, allowing partial results to support new research. Recursive Swarm Evolution integrates both mechanisms into a feedback loop, using new experimental evidence to guide subsequent research. Experiments on OpenProblems, NatureBench and MLE-Bench Lite demonstrate SwarmRSI's state-of-the-art (SOTA) performance. Further evaluations show that accumulated research experience improves performance on held-out tasks. Biologists also reproduce its programs for follow-up biological analyses. 

---
# TreeFlex: Agents Navigating Their Own Search Trees 

**Authors**: Dongzhuo Li, Fangzhou Cheng, Weiwei Li, Kevin Han, Haixu Ma, Zhongchang Sun, Ruiyi Li, Serena Li, Xiangjun Fan, Lizhu Zhang, Yuandong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2610.20334)  

**Abstract**: A ReAct-style agent accumulates observations, intermediate results, and failed attempts in a single context. As this context grows, stale and off-path information consumes its limited capacity and obscures relevant evidence, making focused reasoning increasingly difficult. We introduce TreeFlex, a framework in which the agent's own policy decides where to search. The agent records each intermediate state, with a short summary, as a node in a search tree maintained by the harness. The policy is a single language model that acts in two roles. In the Reflection role, it has a global view of the tree through these summaries and picks a node to resume, together with an instruction. In the Execution role, the harness restores the path that led to that node, and the model carries out the instruction with the usual task tools until it saves a new node or proposes a solution, which Reflection then approves or sends back to another node. Through these choices, the policy decides whether to deepen the current line, branch from or revisit an earlier state, verify a proposal, or stop, without a fixed traversal order or a score for each node. We train navigation, task actions, checkpointing, and proposal review jointly with a tree-aware adaptation of GRPO, using task-level outcome rewards augmented by process rewards. After small-scale RL of a 4B model, with the same active-context limit for all methods, TreeFlex reaches 48.8% accuracy on a challenging deep-research benchmark, versus 41.5% for flat ReAct and 42.8% for FoldAgent, and its Pass@8 is about ten points higher than either. On SWE-bench Verified, with all three systems trained with RL from the same 9B model, TreeFlex resolves 54.9% of tasks, versus 43.3% for FoldAgent and 45.2% for its step-wise variant. 

---
# RM-OPD: On-Policy Distillation for Generative Reward Models 

**Authors**: Qinglin Su, Licheng Pan, Zhen Fang, Jiakang Wang, Qingnan Ren, Yunsheng Lu, Yan Gao, Zhixuan Chu  

**Link**: [PDF](https://arxiv.org/pdf/2610.20174)  

**Abstract**: Reinforcement Learning on Mixed-domain preference data (Mix-RL) has become the mainstream approach for training general-purpose Generative Reward Models (GRMs). However, Mix-RL suffers from two critical bottlenecks: the reward sparsity induced by outcome rewards, and the cross-domain interference arising from jointly optimizing heterogeneous capabilities, which together give rise to imbalanced capability acquisition and pervasive reward hacking. Inspired by the success of On-Policy Distillation (OPD) in multi-domain capability integration, we introduce RM-OPD (Reward Modeling via On-Policy Distillation), a two-stage framework that separates specialist acquisition from capability integration. First, we cultivate domain-specialized teacher models via single-domain GRPO fine-tuning, allowing each expert to reach its performance ceiling in isolation; then we consolidate these teachers into one student model through (i) routed on-policy distillation, which provides dense signed token-level supervision on student-generated critiques; (ii) domain contribution reweighting, which reweights token guidance toward target domain shares, preventing specific domains from dominating optimization; and (iii) optional criteria context distillation, which conditions teachers on domain-specific evaluation criteria to sharpen token-level supervision. Across multiple RM benchmarks and two base models, RM-OPD consistently outperforms vanilla Mix-RL and other strong baselines, achieving overall superior performance and exhibiting an emergent "teacher-surpassing" effect. These results establish RM-OPD as a promising paradigm for building generalist GRMs. 

---
# MultiPartyTurnBench: Benchmarking Floor Control in Multi-Party Spoken Dialogue 

**Authors**: Puneet Mathur, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2610.20172)  

**Abstract**: Full-duplex speech systems must decide when to enter, hold, or relinquish the conversational floor, even as multiple speakers overlap or backchannel without taking it. Existing evaluations study dyadic turn-taking or end-to-end agent behavior, but lack a unified, controlled evaluation of genuine floor transitions against confusable non-events in multi-party settings. We introduce MultiPartyTurnBench (MPTB), a benchmark of 240 synthetically generated, channel-separated conversations spanning two to five speakers and six interaction types. MPTB evaluates two complementary tasks: End-of-Turn (EOT) detection distinguishes genuine turn completion from temporary holds, while Interruption (INT) detection distinguishes successful floor takeovers from non-floor-taking speech. Exact event-conditioned annotations enable causal evaluation using false-positive-constrained recall and signed latency. Across acoustic, semantic, and turn-projection baselines, we find a stark asymmetry: learned systems achieve 82.5-92.6% EOT recall at 4.3-11.8% false-positive rates, while INT detection suffers from missed takeovers or excessive false activations. This asymmetry extends to five full-duplex agents which achieve 86.1-95.8% balanced accuracy on turn entry, yet perform near chance (46.9-54.7%) on selective interruption handling. Gemini-3.8-Live yields to 97.7% of backchannels, whereas MiniCPM-o-4.5 yields to only 1.4% of genuine interruptions. These findings establish selective floor control as a distinct, unresolved challenge for reliable multi-party voice interaction. 

---
# LLM-Based Policy Compliance Verification of Loan Applications 

**Authors**: Florentina Voboril  

**Link**: [PDF](https://arxiv.org/pdf/2610.20155)  

**Abstract**: Banks rely on extensive internal policy frameworks to ensure that corporate lending decisions comply with regulatory requirements and internal risk standards. Reviewing loan applications against these policies is a largely manual and knowledge-intensive process, which can lead to overlooked policy violations or exceptions. This report presents a prototype of an LLM-based policy checker that automatically verifies loan applications against internal lending policies.
The proposed approach follows a two-step architecture. First, compliance-relevant rules are automatically extracted from policy documents and converted into structured, machine-readable representations. Second, loan applications are evaluated against the extracted rules to identify policy exceptions, non-compliant findings, missing information, and cases requiring manual review.
Evaluation results demonstrate that the rule extraction process is highly stable across multiple runs and consistently captures rules identified as critical by a domain expert. When evaluated on 11 real loan applications with known policy exceptions, the system successfully detected the majority of these exceptions and identified additional compliance issues that had previously been overlooked during manual reviews. While the approach produces false positives and cannot fully replace human judgement, the results indicate that LLM-based policy compliance verification can provide valuable support in the loan application verification process by highlighting potential compliance risks. 

---
# Designing Textual Interfaces To Enhance Llm Reasoning On Audio-Visual Question Answering 

**Authors**: Yi Wang, Tianzi Wang, Tong Lei, Dahan Wang, Peter Bell  

**Link**: [PDF](https://arxiv.org/pdf/2610.20085)  

**Abstract**: Automatic prompt optimization (APO) has shown strong effectiveness for text-based LLMs and LLM systems, but its application to audio-visual understanding systems requires suitable textual interfaces through which multimodal reasoning can be exposed to text-based optimizers. We investigate how the design of textual interfaces affects prompt optimization for multimodal reasoning, on the audio-visual question answering task. We compare three interface designs, including a direct inference system consisting of a single omni-modal large language model (OmniLLM), a captioner-reasoner system, and an agentic architecture in which a primary agent orchestrates subagent OmniLLMs. Experiments with Qwen3-Omni and Gemini show that direct prompt optimization provides model-dependent gains; the captioner-reasoner system and the agentic system expose more interpretable optimization interfaces but are harder to optimize, where error analysis of cases affected by optimization is conducted. Our study highlights textual interface design as an important factor in making prompt optimization effective for omni-modal reasoning systems. Our implementation details are specified in GitHub repo. 

---
# LintEval: Benchmarking LLMs on Linter-Guided File Refactoring 

**Authors**: Tongwei Deng, Jinyu Zhu, Yue Cao, Chenxu Wang, Yanzhen Zou, Bing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.20027)  

**Abstract**: Linters are widely used in code maintenance and real-time IDE editing to surface lint alerts related to performance, style, error-prone, and other patterns in source files. However, existing code benchmarks do not explicitly evaluate linter-guided file refactoring, where models must reduce lint alerts in source files with correct initial functionality while satisfying functionality preservation. We introduce LintEval, a multilingual benchmark for linter-guided file refactoring across C++, Python, and Java, comprising 13,061 programs with correct initial functionality and 171,662 lint alerts extracted using language-specific linters. LintEval jointly evaluates functionality preservation, lint alert reduction, new alert avoidance, and edit minimality. Experiments reveal a persistent trade-off: closed-source commercial models reach up to 90.4% Reduction@P but achieve only 67.1%-68.3% Pass@1, while several open-source small models show low or negative Reduction@P. A controlled fine-tuning study further shows that supervision format matters: refactoring-oriented SFT raises Qwen2.5-Coder-7B Reduction@P from 24.7% to 76.0%. LintEval enables controlled evaluation of models that improve source files under functionality preservation. 

---
# MolHop: Scaffold-Hopping Molecular Design with Agentic Reinforcement Learning 

**Authors**: Yanyi Su, Chenyan Wang, Yulei Gong, Yeyun Chen, Xiaohui Yang, Jun Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.19920)  

**Abstract**: Molecular design is often formulated as an optimization problem over existing molecular spaces. However, molecular optimization and the discovery of structurally distinct replacement molecules represent two fundamentally different search problems. LLM-based molecular design agents typically perform the former, conducting local refinement within the neighborhood of existing structures to improve target properties through structural modifications. In contrast, the latter requires exploration of a broader chemical space to discover molecules with entirely different scaffolds while preserving the functional characteristics of reference molecules. We propose MolHop, an agentic framework for cross-scaffold molecular discovery that enables models to navigate and edit within a pre-organized chemical space. MolHop organizes approximately 1.6 million molecular scaffolds from commercial molecules into a coarse-to-fine hierarchical action space. The agent selects new exploration regions through scaffold navigation actions and adjusts candidate properties through molecular editing. The framework maintains historical trajectory memory to support long-horizon search processes. On drug molecule and commercial compound tasks, connecting Grok-4.6 and DeepSeek-Flash-V4.1 to the MolHop framework substantially improves their cross-scaffold joint success rates. We develop MolHop-9B through agentic reinforcement learning, and MolHop-9B outperforms generative baselines including GPT-5.6-sol, Claude-5-Opus, and Gemini-3.1-Pro, while generating candidate molecules with larger scaffold differences from the reference molecules. These results demonstrate that effective molecular discovery beyond the traditional optimization paradigm requires structured chemical and action spaces, enabling agents to explore new structural regions rather than merely refine existing molecules. 

---
# SEA-HELM Audio: Evaluating Multilingual ASR-LLM Pipeline Performance Across SEA-HELM Subsets 

**Authors**: Yixian Tan, Aulia Adila, Jonathan Heng, Kunat Pipatanakul, Donghang Wu, Anocha Sutaveephamochanon  

**Link**: [PDF](https://arxiv.org/pdf/2610.19869)  

**Abstract**: We introduce SEA-HELM Audio, a Southeast Asian (SEA) multilingual evaluation framework that extends the text-based SEA-HELM benchmark to assess performance degradation in cascaded automatic speech recognition and large language model (ASR-LLM) pipelines. Using synthesized speech, we compare speech-input and text-input performance across five languages (Chinese, English, Malay, Indonesian, and Tamil) on multiple-choice reasoning and instruction-following tasks. We evaluate open-weight and commercial ASR-LLM systems using identical verifiers and quantify the speech gap as the difference between text and speech accuracy. Results show substantial degradation, with accuracy drops of 9.00 to 18.50 percentage points for multiple-choice reasoning and 20.00 to 42.86 points for instruction following. Tamil exhibits the largest degradation, exceeding 40 percentage points. Our findings highlight the limitations of text-only benchmarks in predicting voice-agent reliability and demonstrate the value of SEA-HELM Audio for evaluating multilingual voice-agent systems. 

---
# MAPLE: A Multi-Axis Parametric Knowledge Extraction Lens for LLM Evaluation 

**Authors**: Minsuh Joo, Youna Kim, Choonghyun Park, Soo Kyung Kim, Hyunsoo Cho  

**Link**: [PDF](https://arxiv.org/pdf/2610.19827)  

**Abstract**: LLMs absorb vast knowledge from training data, yet their inherently probabilistic nature offers no guarantee that this knowledge will surface in their outputs. To bridge this gap, Knowledge Extraction Enhancement (KEE) techniques such as activation steering, decoding adjustments, and prompting modify inference, potentially reinforcing existing predictions or surfacing less accessible answers. Yet their evaluations centered on aggregate accuracy, which obscures diverse effects by collapsing recovered failures and disrupted successes into a single score. We introduce MAPLE, a unified framework for characterizing how KEE techniques affect parametric knowledge extraction. Its diagnostic metrics disentangle these effects at the surface, probability, and beyond-sampling levels by measuring answer recovery and disruption, gold-token rank shifts on unrecovered failures, and recovery of answers missed by fixed-budget sampling. We evaluate six KEE techniques spanning decoding, activation steering, and prompting across three model families: OLMo2, Qwen2.5, and Llama3. Our analysis reveals distinct effects and trade-offs across techniques, highlighting the value of MAPLE for understanding and comparing KEE beyond aggregate accuracy. 

---
# Learning Perception to Action Rules through Reinforcement Learning for Theory-of-Mind Reasoning 

**Authors**: Soyoung Oh, Nursulu Sagimbayeva, Nicholas Jennings, Paulina Hussein, Mehwish Alam, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2610.19796)  

**Abstract**: In Theory-of-Mind (ToM) reasoning, large language models may correctly infer the agent's mental state but fail to use that inference when predicting or judging the agent's subsequent actions. In this work, we investigate whether explicitly connecting perception, belief, and action with rules improves ToM reasoning. We generate concise \textit{if--then} rules that express relations between perception to action and use Group Relative Policy Optimization (GRPO) to train Qwen2.5 and Llama3 models across four ToM benchmarks. GRPO with rule generation outperforms prompting, supervised fine-tuning, and think reasoning. Moreover, rules generated by larger models ($\geq$3B) provide transferable knowledge that improves downstream performance of GPT-4o-mini. These findings suggest that explicit rules help models bridge the reasoning gap in ToM and support knowledge transfer across models. 

---
# Mitigating Video Hallucinations via Attention-Guided Representation Perturbation 

**Authors**: Zhixin Cai, Ruoqi Yuan, Xinchen Yang, Zhuofan Chen, Jun Bai, Chuantao Yin, Wenge Rong  

**Link**: [PDF](https://arxiv.org/pdf/2610.19728)  

**Abstract**: Video large language models (Video LLMs) often generate responses unsupported by the input video. Contrastive decoding (CD) can mitigate these hallucinations, but its effectiveness depends on how the negative branch weakens visual evidence. Existing methods rely on predefined perturbations or masking guided by question reconstruction, offering limited control over the model's visual reliance during answer generation. We introduce Attention-guided Representation Perturbation for Contrastive Decoding (ARP-CD), a training-free framework for constructing contrastive negatives directly in representation space. ARP-CD uses the attention mass assigned to visual tokens at the answer-prediction position to guide bounded gradient updates of the language model's input visual hidden states. These updates reduce visual attention and produce a negative input, whose predictions are contrasted with those of the original branch throughout generation. Experiments across four Video LLMs and four benchmarks show that ARP-CD outperforms representative training-free baselines in most model--benchmark settings, reducing hallucinations while improving general video understanding. Further analyses show that attention-guided perturbation directions improve negative quality and that the resulting negatives provide effective contrastive references for correcting the original predictions. 

---
# Articulation Upper Bounds for Language Models 

**Authors**: Alexander Spangher, Daniel E. Ho, Diyi Yang, Noah Goodman, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2610.19721)  

**Abstract**: Language captures only a subset of the perceptions, intuitions and intents that form meaning between speakers. What, then, are the limits of language? We take steps towards measuring this in LLMs as follows: for a set of concepts, we measure how well each concept can be transmitted between two LLMs as their capabilities scale. This requires novel measurement tools: (1) optimal messages: we introduce a prompt optimization algorithm that beats existing algorithms by up to 14% on benchmarks and accurately estimates prompt upper bounds. (2) A transmission measure: the sender is asked to label text; the receiver then tries to recover the sender's prompt, and recovery is measured; we find this correlates strongly with human label error. With these measures, we use observational scaling laws to study how transmissibility of a concept, conveyed with optimal messages, increases as transmitter and receiver capacity grows: does it asymptote below or above a threshold? Measuring 1.2k concepts in eight domains, we find concepts that (a) can be fully articulated; these concepts usually depend on local patterns, e.g. "joke ends with surprise"; (b) might be articulated by future, stronger models; these rely on whole-text structural patterns, e.g. "story's worldbuilding coherence"; (c) concepts that face articulability limits; these concepts' meaning, we find, is tied to identity and social-group awareness, e.g. "comic voice". Interesting, these same concepts are not all challenging for humans: humans have moderate-to-high levels of agreement in 76% of concepts that LLMs struggle with; articulation limits, we find, are executor relative. Finally, we characterize the nature of improvement in each case: while recoverable concepts can be articulated through definitions and explanations, irrecoverable concepts cannot be transmitted, even with examples, and might require novel learning mechanisms. 

---
# Function-Space Temporal Self-Distillation for Policy Optimization 

**Authors**: Aryan Das, Arnab Bhattacharya, Vinay Kumar Verma  

**Link**: [PDF](https://arxiv.org/pdf/2610.19687)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) improves reasoning in language models, but scalar outcomes provide coarse supervision over response trajectories. Self-distillation offers denser token-level guidance by comparing an unassisted policy with a feedback-informed self-teacher, yet the teacher evolves with the student, creating unstable targets and motivating additional regularization. We introduce Function-Space Policy Optimization (FSPO), a single-policy framework that combines current privileged adaptation with function-space temporal consolidation. FSPO stores sparse historical next-token distributions as fixed temporal targets, providing temporal separation without maintaining a persistent teacher model. It further introduces privileged disagreement correction to explicitly suppress locally overvalued actions, reliability-adaptive grounding that strengthens reference supervision when self-generated evidence repeatedly fails, and quality and coverage-aware temporal memory that retains verified behaviors and improves privileged demonstrations. Across reasoning and tool-use benchmarks, FSPO achieves the highest average accuracy under a 5-hour training budget, improving over the strongest baseline by approximately 4% points on average while reducing peak memory from about 70 GB to 56 GB on Olmo3-7B-Instruct. 

---
# Beyond Distribution Matching: Selective Margin Preservation for Supervised Fine-Tuning 

**Authors**: Yuxuan Wang, Guangxiang Zhao, Xinyu Tan, Yahui Han, Xinxiang Yin, Zhonghai Wu, Xiangzheng Zhang, Lin Sun, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19685)  

**Abstract**: Supervised fine-tuning (SFT) changes logit margins among non-target tokens at teacher-forced prefixes. Matching a local reference distribution keeps those probabilities and also penalizes margin expansions. We ask whether a soft auxiliary can preserve selected local non-target margin relations without freezing that distribution. We introduce \textbf{Selective Rank-Preserving SFT (SRP-SFT)}, a soft margin target: a hinge on reference-scaled lower bounds for pairs from the reference's top-$K$ non-target tokens, applied at fixed teacher-forced prefixes. The hinge stops once the requested bound is met. We derive the KL cost of violating a margin, characterize the feasible region and its conditional utility, and give a finite-step condition for reducing aggregate violations. In one-epoch experiments on Qwen3-4B-Base and UltraFeedback, this target raises MATH-500 and code-generation scores over CE and LP-SFT at all three training seeds. At seed 1234, MATH-500 pass@1 rises from 49.56\% to 54.56\% and the four-metric code mean rises from 65.91\% to 69.50\% relative to LP-SFT. Ranking controls, held-out cache diagnostics, sensitivity sweeps, and six additional backbones show where these score changes appear. 

---
# Model What Matters: Decision-Guided Selective World Modeling for Language Agents 

**Authors**: Maolin Ran, Minda Chen, Di Huang, Wei Xiong, Fei Huang, Weiwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19642)  

**Abstract**: Reinforcement learning provides a direct mechanism for optimizing task objectives in large language model-driven agents, but task rewards are typically sparse trajectory-level signals that only evaluate task success, without directly supervising action-conditioned environment transitions, leaving the model with limited explicit understanding of action consequences and environment dynamics. Recent work introduces auxiliary supervision through policy and world modeling co-training, yet typically treats complete environmental observations uniformly as world modeling targets, overlooking the varying decision relevance of observation content. We propose DGS-WM (Decision-Guided Selective World Modeling), a decision-guided policy-world modeling co-training approach for language agents. DGS-WM employs a Decision-Guided Observation Selection component that filters observation units with decision value as world modeling targets based on their support for subsequent policy responses, and a Confidence-Adaptive Weighting component that adaptively adjusts supervision strength according to the model's prediction confidence on selected targets. Together, these components form a decision-aware auxiliary world modeling loss, jointly optimized with the RL loss on the same policy model. Across multiple benchmarks covering single-hop/multi-hop question answering and interactive web tasks, DGS-WM achieves consistent improvements under different model scales and RL algorithms, demonstrating more stable and effective performance compared to full-observation world modeling methods. 

---
# Sparse Autoencoders Are a Shared Coordinate System for Symbolic Matching 

**Authors**: Xiaocong Yang, Xuanyu Meng, Yuji Zhang, Jiawei Han, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2610.19636)  

**Abstract**: Large language models (LLMs) hold broad knowledge in one set of weights, but lack interpretability and trustworthy reasoning, especially in high-stakes use cases. A neuro-symbolic model, with its interpretable architecture and System 2 reasoning capacity, is a promising alternative if built at the scale of LLMs; however, its symbolic backbone is usually constructed with human effort, so existing neuro-symbolic models are bottlenecked by the scope of the symbol component and usually domain-specific. A data-driven method could instead build a larger symbolic system by aligning and merging existing ones. We target its first, foundational step: matching units across these heterogeneous systems. Our method, SAE-IDF, encodes a symbol in its native context into a sparse code with a sparse autoencoder (SAE), re-weights each dimension by its rarity and matches across systems. On 12 tasks across 6 kinds of symbolic system, SAE-IDF outperforms the same LLM's dense baseline on all tasks with Gemma-2-9B and all but one with Llama-3.1-8B, by up to 45 points, consistently across SAE layers and configurations. SAE-IDF is also interpretable: every dimension of the sparse code has a human-readable meaning, so each match decomposes into named concepts (features). Our work paves the way to a universal symbolic backbone. 

---
# Divergent Paths: Learning an Outcome-Predictive Representation of LLM Agent Trajectories 

**Authors**: Alexander Gill, Guram Gogia, Francisco Guzmán  

**Link**: [PDF](https://arxiv.org/pdf/2610.19614)  

**Abstract**: In this work, we reframe agentic trajectories as paths through a shared embedding space of actions and environmental observations. We show that successful and failed rollouts of the same task separate in this space, with simple per-task discriminators predicting held-out rollout outcomes at up to 0.95 AUROC across three agent benchmarks and three policy models. To exploit this signal across tasks, we introduce Precog-RM, a transformer-based model that meta-learns a representation in which outcome prediction can be performed in a few-shot setting. We show that Precog-RM generalizes to unseen tasks and models, and gives confident predictions of trajectory success long before termination. Deployed as an early stopping controller for Group Relative Policy Optimization (GRPO) on Tau2-Bench, Precog-RM cuts training FLOPs by up to 47.3\% while maintaining performance comparable to baseline GRPO. 

---
# Twenty Days Later: A Pre-Registered Test of History-Based Forecasts of Brand Mentions in Gemini Answers 

**Authors**: Dmitrij Żatuchin  

**Link**: [PDF](https://arxiv.org/pdf/2610.19608)  

**Abstract**: Counts of which brands a large language model names are useful only if they hold long enough to act on. We test this with a pre-registered replication. The same 162 requests (54 questions in English, Estonian and Finnish about three software categories, four of twelve candidate brands per question, three repeats) were sent to gemini-3.1-pro-preview in three waves within about ten hours on 22 and 23 September 2026, and again on 13 October 2026, 20.3 days after the third wave, with identical request bodies. The plan and its decision rule were frozen, hashed and registered on OSF before the first call. The outcome is a lexical mention label per answer and brand, 648 per wave. The primary forecast, each brand-and-question cell's mention rate over the nine earlier answers, scored Brier 0.0281 and log loss 0.126, against Brier 0.0259 for the six-answer forecast of the 8-hour wave and a registered ceiling of 0.0559. It beat brand history (0.104), the overall rate (0.234) and a constant 0.5 (0.250) on both scores and in all nine question families (exact sign-flip p = 0.0039), so durability holds under the registered rule. A forecast frozen from the first wave alone scored 0.0599, 0.0587 and 0.0604 at horizons of 1.5 hours, 9.6 hours and 20.7 days. Of the 202 cells unanimous in the 8-hour wave, 194 were unchanged at D1 and none reversed. The latest wave alone was nearly as accurate as the pooled history (Brier 0.0292; pooled lower in 5 of 9 families, p = 0.28). Every answer reported the requested model version, and none issued a search query although search was enabled. A blind human check of 96 labels agreed with the rule on 94; both disagreements were adjudicated as rater omissions, so after adjudication the rule matched on 96 of 96. The test covers one model, one interval and twelve known brands, and estimates no decay rate. 

---
# PN-Mix: Pair-Norm Guided Mixed-Precision Key Cache Quantization for Efficient Long-Output LLM Inference 

**Authors**: Gil Jae Song, Sungjoo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2610.19605)  

**Abstract**: Long-output inference in large language models suffers from severe KV cache memory and bandwidth bottlenecks. While KV cache quantization alleviates this issue, low-bit quantization (e.g., 2-bit) severely degrades model accuracy. Recent mixed-precision key-cache quantization methods mitigate this degradation by allocating higher precision to important key channels, but estimate channel saliency using only key statistics. Through attention logit perturbation analysis at the RoPE channel-pair level, we show that quantization errors in channel pairs with large query--key pair-norm products contribute significantly to logit variation. Based on this insight, we propose PN-Mix, a mixed-precision key-cache quantization method for RoPE-based decoder-only LLMs, which allocates higher precision to salient channel pairs identified once during the prefill stage and reused throughout decoding. Across standard reasoning, code-generation, and AIME benchmarks, PN-Mix with 2.26-bit average precision achieves comparable or better average accuracy than prior mixed-precision methods using 2.32--2.39 bits. Furthermore, in our Pareto sweep, PN-Mix degrades gracefully down to 1.5 bits, whereas the compared baselines exhibit sharper performance drops below 2.25 bits. 

---
# Who's Behind the Text? Robust Model Authorship Attribution 

**Authors**: Minseok Ko, Hyunseok Lee, Woomin Song, Beomjun Kim, Jinwoo Shin, Kunwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.19583)  

**Abstract**: As large language models (LLMs) are increasingly used to generate text, their misuse has become a growing societal problem, with implications for academic misconduct, model distillation, and cyberattacks. For these reasons, Model Authorship Attribution (MAA), which identifies the source model of a given text, is important for promoting the responsible use of LLMs and improving AI safety. In practice, attribution systems must remain reliable beyond the distributions observed during training. They may encounter texts from unseen domains and writing settings, as well as generations from future versions of model families that were not available when the system was trained. However, existing MAA methods often exhibit substantial performance degradation under such distribution shifts. To address this challenge, we propose Authorship Tracing via Likelihood-based Attribution of Sources (ATLAS), which formulates MAA through source-conditioned text modeling rather than direct source prediction. ATLAS learns how text from each source family is generated and attributes a given text by comparing its likelihood under each learned source condition. We aggregate this likelihood evidence at the token level with clipping so that no single token dominates the decision. We further introduce MAABench, a broad MAA benchmark for evaluating practical OOD settings such as domain and version. Our experiments show that ATLAS consistently outperforms existing baselines across MAABench and OpenTuringBench. In particular, when both the generation domain and the model version change on MAABench, ATLAS achieves 74.27% accuracy, 9.00pp higher than the strongest baseline. Beyond MAA, ATLAS can also help identify distillation source models by measuring which source model a target LLM most closely resembles. Our extended experiments show that ATLAS recovers the known teacher of every distilled model we evaluate. 

---
# $τ$-Emotion-Bench: Exposing the Interaction Fragility of LLM Agents through Real-World User Emotions 

**Authors**: Rishitosh Kumar Singh, Amir Saeidi, Shamanthak Hegde, Shrinidhi Kumbhar, Chitta Baral  

**Link**: [PDF](https://arxiv.org/pdf/2610.19560)  

**Abstract**: In long-horizon tool-calling tasks, LLM agents can fail not only because of tool-use errors, but also because of ineffective interaction with users. Yet existing benchmarks largely assume cooperative and behaviorally stable users, leaving the effect of emotional variation on agent reliability underexplored. We introduce $\tau$-Emotion-Bench, a ten-domain benchmark for evaluating tool-calling agents under controlled emotion-conditioned user interactions while preserving the underlying task objective. To scale both benchmark construction and agent training, we introduce Tracer, a scalable agentic data-generation pipeline that constructs grounded, policy-valid tasks and collects trajectories through an interactive agent--user--environment harness. By allowing dissatisfied users to terminate inefficient interactions, the harness provides an interaction-aware signal that filters trajectories with excessive user dependence and favors effective tool use. Experiments with two user simulators show that frontier agents remain unreliable, with substantial gaps between average task success and consistent task completion. Tracer-generated supervision consistently improves Qwen3 models across scales, generalizes to unseen domains, and transfers to $\tau$-Bench and $\tau$-Trait. Behavioral analyses further show that higher task success is associated with a better balance between tool execution and necessary user follow-up. Overall, $\tau$-Emotion-Bench and Tracer provide a scalable framework for evaluating agent reliability under diverse user behaviors and improving tool-calling agents through interaction-aware supervision. 

---
# Diagnosing and Correcting Reward Hacking in Verifiable Instruction Following 

**Authors**: Sangyeon Yoon, Wooseok Seo, Hamish Ivison, Hyesoo Hong, Donghoon Kim, Wonjun Kang, Albert No, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.19525)  

**Abstract**: Verifiable instruction-following benchmarks emphasize output-constraint satisfaction but can overlook whether responses meaningfully answer the underlying request, creating an evaluation blind spot. We show that frontier models can receive full credit for responses with little substantive answer content. Using the same verifier-based criterion as a training reward can induce reward hacking, as publicly released checkpoints exhibit higher constraint satisfaction but lower response quality after reinforcement learning with verifiable rewards. To address this reward hacking, we introduce CALIF, a lightweight method that augments verifier rewards with a calibrated likelihood-based bonus to preserve response quality. CALIF evaluates response likelihood under the request alone using the existing frozen reference policy, requiring no additional judge or learned reward model. Across models ranging from 1.7B to 8B in reasoning and non-reasoning modes, CALIF improves constraint satisfaction on IFBench while maintaining response quality near or above pre-RL levels. It also outperforms corresponding base policies in 40 of 45 model-metric comparisons across three rubric-based instruction-following benchmarks, demonstrating gains beyond training verifiers. 

---
# What Does LALM Post-training Learn? Rethinking Audio Understanding and Reasoning Capabilities 

**Authors**: Hua-Hsuan Chen, Xuanjun Chen, Wei-Chung Lu, Chih-Kai Yang, Yu-Xiang Lin, Sung-Feng Huang, Haibin Wu, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.19476)  

**Abstract**: Post-training often improves the performance of large audio-language models on audio understanding and reasoning benchmarks. However, these improvements may reflect adaptation to specific question formats, such as multiple-choice questions, rather than genuine advances in audio understanding and reasoning capabilities. To investigate what LALMs actually learn from post-training, we conduct controlled experiments that disentangle improvements in audio understanding from adaptation to specific question formats. Across different models and post-training methods, we find that much of the reported improvement persists even without informative audio, but largely disappears when the question format changes. These findings suggest that post-training often teaches models to exploit task-specific textual cues and answer patterns rather than developing generalizable audio understanding and reasoning capabilities. 

---
# ActICL: Post-Training for Agentic Multimodal In-Context Learning 

**Authors**: Yanshu Li, Ruixiang Tang, Junyi Jessy Li, Elias Stengel-Eskin  

**Link**: [PDF](https://arxiv.org/pdf/2610.19475)  

**Abstract**: Multimodal in-context learning (ICL) is an important capability of multimodal large language models (MLLMs). Its performance is closely tied to the demonstrations it uses, making demonstration retrieval a core problem for ICL. Common methods use fixed scoring methods to retrieve all the demonstrations at once, which may introduce redundancy and noise. Context-aware retrieval methods have emerged to address this problem by allowing incremental retrieval from a growing context. However, none of them lets the MLLM learn how to judge what new demonstrations it needs and then carry out progressive retrieval on its own. To this end, we introduce ActICL, which trains an MLLM agent to acquire and use an ICL context via tool-augmented reasoning. An ActICL agent analyzes its own information gaps during iterative reasoning and tool-use actions. Based on these gaps, it performs targeted retrieval to acquire a set of initial candidates. It can then use the existing context to further filter the candidates and refine the context, or it can iteratively perform a new retrieval. When the agent judges that the context is sufficient for the task, or when the resource limit is reached with the user request met, it sends the acquired context to an answering model, either the agent itself or another MLLM. We use a two-stage post-training procedure that jointly optimizes the agent's acquisition and answering. With an adaptive number of demonstrations, ActICL improves average performance by 11.6 percentage points over standard similarity-based retrieval across seven benchmarks and three MLLMs, and by 4.4 points compared to the strongest prior baseline. The contexts acquired by ActICL agents can also be used effectively by frontier models. When GPT-5.6-Sol performs 4-shot ICL, ActICL's contexts improve its performance by 3.5 points over standard retrieval, averaged over three benchmarks. 

---
# Seeing, Maintaining, and Learning: The Evolution of Multimodal Agent Memory 

**Authors**: Minghao Guo, Qingyue Jiao, Xiujin Liu, Jianhong Pang, Ruimeng Yang, Naihao Xu, Liwei Che, Haochen Xue, Ziqi Wang, Xinying Cai, Jiaxuan Luo, Jiaxin Deng, Xuesong Ye, Jintao Qu, Wenyao Gao, Jingwen Gu, Chenyang Zhao, Shichun Liu, Guibin Zhang, Zefang Liu, Siyin Wang, Wujiang Xu, Danrui Li, Zeru Shi, Woongyeong Yeo, Shuang Liu, Wei-Chieh Huang, Weizhi Zhang, Tianyu Yang, Kangsan Kim, Yebowen Hu, Bowen Wen, Bowen Jiang, Haohao Su, Ming Yin, Sheldon Yu, Mian Zhang, Yuyan Chen, Hanrong Zhang, Kai Mei, Zihao Zhang, Mingyu Jin, Xi Zhu, Hengyi Wang, Tianxin Wei, Linyong Nan, Zhenwen Liang, Zhendong Chu, Ziqi Huang, Yongye Su, Sixun Dong, Shujian Liu, Kaiqu Liang, Jiaojiao Han, Zhaoye Fei, Kun Wang, Shihan Dou, Zhichun Guo, Xingjian Diao, Zeqi Gu, Heng Wang, Tao Feng, Cheng Qian, Hamid Palangi, Silei Xu, Xi Yun, Yi Zhu, Shilong Liu, Qingsong Wen, Sung Ju Hwang, Zuxuan Wu, Yu-Gang Jiang, Yiyu Shi, Dimitris N. Metaxas, Jiaxuan You, Julian McAuley, Mengdi Wang, Jiawei Han, Philip S. Yu, Ziyi Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.19366)  

**Abstract**: Multimodal models and agents have emerged to enable AI systems to learn from and interact with the real world more like humans by integrating information across multiple modalities. However, enabling multimodal agents to continually acquire useful information from complex, heterogeneous multimodal data remains a significant challenge as they move toward long-horizon tasks and lifelong settings. Addressing this challenge requires effective memory mechanisms that can maintain continuity across sessions, adapt to changing users and environments, and refine memory-management policies through accumulated experience. We introduce a two-axis analytical framework for multimodal agent memory structured around two key questions: how multimodal information is formed and represented (Axis I, Formation and Representation), and how its management adapts with accumulated experience (Axis II, Memory Management). Axis I, Formation and Representation, characterizes what information is retained and how it is encoded (e.g., modality-specific embeddings, textual and structured records, latent and parametric representations) to support subsequent reasoning and action. Axis II, Memory Management, distinguishes three tiers: T1 Static Memory reuses fixed memory; T2 Dynamic Memory updates memory through a stable policy; and T3 Self-Evolving Memory updates memory through an evolving policy. Based on this framework, we synthesize recurring design patterns and derive principles for balancing source fidelity with abstraction, maintaining temporal validity and provenance, grounding memory use in recoverable evidence, and evolving memory policies over time. Furthermore, we discuss the challenges and potential future directions for the memory mechanism of multimodal agents. 

---
# VLX-Go: Robust Embodied Tracking through Multi-Stage Learning and On-Policy Refinement 

**Authors**: Kyusong Lee, Heting Ying, Tiancheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.19361)  

**Abstract**: Embodied visual tracking requires a robot to identify a designated target from visual observations and control its own motion to follow that target through an environment. Following an isolated person is a comparatively simple demonstration of embodied visual tracking, but following the intended person among distractors remains challenging. Existing methods exhibit a substantial drop in success from single-target to distractor tracking, motivating a focus on the shared spaces that person-following robots must navigate. We present \method{}, a vision--language waypoint policy built around the hypothesis that visual and language representations should be aligned before tracking-specific motion is learned. The policy combines complementary frozen visual encoders, multiscale temporal context, a transformer backbone, and a lightweight head for robot-centric waypoint prediction. Its training progresses through vision--language alignment, multimodal reasoning mid-training, and tracking specialization on 19,532 oracle episodes. Compared with 41.54\% distractor-tracking success for our earlier OmTrackVLA model, the complete multi-stage recipe achieves 55.80\%. We then explore on-policy refinement using filtered oracle labels for states visited by the policy. This preliminary study adds 2.42 percentage points of distractor-tracking success, reaching 58.22\%, while reducing collision rate by 1.99 pts. The final policy records the highest distractor- and ambiguous-tracking success rates among the compared methods. These results support staged multimodal learning as a promising direction for tracking among distractors and motivate further study of on-policy refinement for person following in shared spaces. 

---
# Steering Vectors for Competing Ethical Frameworks Are Interchangeable: A Competing-Target Test for Activation Steering 

**Authors**: Weilun Xu, Zipeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.19359)  

**Abstract**: Activation steering adds a vector to a model's hidden states to control a named behaviour. A gain on that behaviour, however, does not show that the vector controls it directly: it may act through a component shared with neighbouring behaviours, which unsteered and random baselines cannot separate. We introduce the competing-target test: a vector is compared with the vector built the same way for a competing target, and an equivalence test states when their difference is below a useful margin. We apply it in open-weight models to vectors for Kantian deontology, which judges an act by the rule it follows, and act utilitarianism, which judges it by the welfare it produces. On held-out situations, prompting a framework raises its score and lowers the other's, whereas vectors for either framework move both scores alike, and none reaches the margin. The two vectors share most of their direction, a request to apply some framework; the same construction separates an optimistic from a pessimistic tone, whose vectors share much less. What a vector shares with its competitors therefore sets what it controls: steering a perspective can deliver a mixture, and measuring a disposition can measure its neighbours, unless tested against the competing target. 

---
# Where to Start, How to Move: Analyzing Curricula under GRPO for Long-Horizon LLM Agents 

**Authors**: Minchan Kim, Sungho Bae, Youngjin Kim, Hyunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.19349)  

**Abstract**: Group Relative Policy Optimization (GRPO) provides no reward-relative learning signal when all rollouts receive identical rewards. In long-horizon tasks that require multiple interactions, rewards are given only upon successful task completion, making all-fail groups common early in training and limiting opportunities to practice later steps. Guide Assisted GRPO replays a successful reference prefix before the learner continues. We study how initial positions, handoff units, and movement rules affect unassisted task completion. Across three model sizes on BFCL, starting near completion is more effective under a fixed turn-level threshold rule. Finer handoffs alone do not consistently help, but outcome-informed tool-call selection outperforms GRPO on trained and untrained task categories. Guided starts also improve AppWorld success on both evaluation splits. These findings highlight where learning starts and how starting positions move as key curriculum choices. 

---
# Counterfactual Gate-Support Decomposition: Separating Selection from Redistribution in Training-Data Filtering 

**Authors**: Hyojung Han, Jongmin Kim, Seung-Hun Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2610.19343)  

**Abstract**: A training-data quality gate changes both which examples survive within regions of the input space and how training mass is distributed across those regions, so a gate-versus-random comparison does not isolate these mechanisms. We introduce Counterfactual Gate-Support Decomposition (CGSD), a four-arm, budget-matched design that crosses gate eligibility with support distribution and decomposes the observed gate effect into within-support selection, between-support redistribution, and their interaction. The practical constraint is finite-pool realizability: we characterize mass and token-assembly conditions under which the required arms can be constructed, together with a support-extinction case in which the ordinary design is undefined. On Korean-English and Korean-Japanese retrieval, the mean within-support selection component is the only one that exceeds its calibrated null band (+29.7 and +5.6 holdout items; 30/30 positive replicates), decisively in Korean-English and marginally in Korean-Japanese; redistribution and interaction remain unresolved at the available power. The same pattern holds for a second, unrelated gate family -- a model-confidence threshold rather than a deterministic verifier -- judged against its own null (+27.3 items, 9/9 positive), so the result is not a property of one gate's mechanism. On a mathematics gate, zero-eligibility supports leave the ordinary design undefined at every budget, while the extended design is constructible only inside narrow budget bands that an exhaustive scan locates and that the natural operating point misses -- the assembly condition separating a property of the pool from a limitation of the packer. CGSD therefore returns a decomposition, a constructible-but-underpowered design, or an explicit non-constructibility result, and says which; it prevents support redistribution from being silently interpreted as sample-level gate quality. 

---
# Truthful and Expected-Payment-Optimal Model Routing in Graph-Based Agentic Workflows 

**Authors**: Jiayuan Liu, Mingyu Guo, Jiarui Gan, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2610.19316)  

**Abstract**: Assigning routine tasks to cheaper large language model (LLM) providers can reduce generation cost, but switching providers may require processing accumulated context again. Keeping dependent tasks with the same provider can instead reuse eligible context. Graph-based workflows therefore couple routing decisions across tasks. Independent providers may also privately know their real-time unit costs and strategically misreport them. We formulate this routing problem as a single-parameter procurement auction and propose STAR (Strategy-proof Topology-aware Agent Routing), which minimizes virtual-cost-weighted graph workloads. Its allocation rule and payments, computed by integrating workload over possible cost reports, make truthful reporting utility-maximizing regardless of other providers' reports and cover each truthful provider's execution cost. With independent cost distributions and non-decreasing virtual costs, STAR additionally minimizes the user's expected total payment to providers. Controlled synthetic experiments separate the execution-cost benefit of topology awareness from the payment benefit of virtual-cost optimization: the former grows with fan-out and transferred payload, while the latter increases gradually along the tested prior-shape heterogeneity path. A paired sensitivity study quantifies how payment savings vary with prior misspecification while strategy-proofness is preserved. 

---
# From Efficiency to Sufficiency: The Cost-Aware Measurement Design Framework for Automated Essay Scoring 

**Authors**: Yi Gui  

**Link**: [PDF](https://arxiv.org/pdf/2610.19287)  

**Abstract**: Compute cost in automated essay scoring (AES) is usually treated as a technical efficiency issue rather than a measurement design concern. Yet deployment decisions require more than reporting latency, model size, or inference expense. In high-stakes settings, total burden includes validation, monitoring, adjudication, compliance, and governance; in instructional settings, pricing, licensing, and integration may dominate cost. This paper reframes AES as a cost-constrained measurement system and introduces the Cost-Aware Measurement Design (CAMD) Framework, a four-layer structure consisting of metering, allocation, unit economics, and decision, for translating computational and workflow resource records into decision-relevant cost constructs. Direct cost is specified as monetary outlay plus attributable labor and time, and two criteria, design attribution and auditability, determine what the cost boundary excludes. Burdens that satisfy neither criterion, such as institutional capability and governance posture, are represented as constraints on which designs are admissible rather than priced. The paper defines cost per defensible score (CPDS) and argues that AES systems should be compared by quality-cost sufficiency: minimizing cost among defensible scoring designs that satisfy measurement requirements. Scoring volume conditions all four layers, so cost per defensible score is treated as a function of scale rather than as a single transferable number. CAMD provides a portable basis for representing, comparing, and evaluating cost across AES studies and deployment contexts. 

---
# MemeTrace: A Multimodal LLM-Based Agent for Safety-Aware Meme Provenance Analysis 

**Authors**: Bo Shao, Yihan Ma, Yiting Qu, Michael Backes, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19267)  

**Abstract**: Online memes are often carriers of unsafe content such as disinformation and propaganda. Their rapid evolution in response to news events and strong dependence on contextual information make it difficult for platforms and moderators to detect and moderate them effectively. Moreover, existing moderation studies typically classify memes as safe or unsafe based on individual images, lacking the ability to perform broader provenance analysis, i.e., where a meme originates, how it spreads, what cultural symbolism it represents, whether it poses potential risks to society, etc. To bridge this gap, we introduce MemeTrace, a multimodal LLM-based agent that automatically traces the provenance and potential risks of emerging memes. We evaluate MemeTrace on Meme100, a dataset of 100 real-world memes, using both human annotation and automatic metrics. The results demonstrate that MemeTrace substantially outperforms the VLM baseline. More importantly, MemeTrace achieves higher accuracy in detecting unsafe memes, especially those involving disinformation and sensitive content, and remains robust under visual perturbations. 

---
# ListQA: A Benchmark for Evaluating List-Formatted Factual Knowledge Retrieval in Large Language Models 

**Authors**: Mayank Kulkarni, Rakesh Vaideeswaran  

**Link**: [PDF](https://arxiv.org/pdf/2610.19165)  

**Abstract**: Many real-world questions demand not a single fact but an organized collection of them, yet existing factual knowledge benchmarks almost exclusively target single answer retrieval. We introduce ListQA, a benchmark of 9,045 human-curated, cross-validated questions (audited error rate of 1.7%) that require LLMs to recall multiple facts and compose them into structured lists of 3-10 elements. Questions range from flat lists ("Which NCAA teams went undefeated between 2000 and 2024?") to hierarchical lists with sub-attributes ("...and what was their record and result?"), spanning eight categories across 60+ countries. An evergreen design (explicit temporal constraints or historically immutable facts) keeps ground truth valid without periodic updates. For evaluation, we frame element matching as a linear sum assignment problem: optimal bipartite matching paired with LLM-based semantic grading jointly captures factual accuracy, hallucination, and format compliance. We benchmark 34 models (8 families, 42 configurations across standard and thinking enabled inference) and find: the best standard-inference model scores below 37% LLM-Judge-F1, multi-faceted questions are harder across the board, and providing the source Wikipedia page lifts scores dramatically, confirming that models can organize facts but struggle to retrieve them from parameters alone. Extended analyses cover base vs. instruction-tuned checkpoints and token efficiency. We release the dataset and evaluation code. 

---
# Into the Unknown: Benchmarking AI agents on open scientific research problems 

**Authors**: Seungone Kim, Jiayi Geng, Young-Jun Lee, Bo Liu, Juyoung Suk, Nicholas Lubbers, Mitchell Wood, Marcus Noack, Yong-Xin Yao, Yuan Shi, Roxana Bujack, Cristina Garcia Cardona, Austin Rodriguez, Ilon Joseph, Woojin Cho, Zora Zhiruo Wang, Chuanyang Jin, Ian Wu, Apurva Gandhi, Tianjian Li, Weiwei Sun, Carolin Lawrence, Kiril Gashteovski, Dongyeop Kang, Jay Shin, Akari Asai, Graham Neubig, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2610.19159)  

**Abstract**: Can AI agents carry out research on open scientific problems? We introduce Into the Unknown, a benchmark that provides an agent with a science paper along with a research direction to pursue and tasks it to write a follow-up paper. We work with scientists at US Department of Energy (DOE) national laboratories, who each contribute an unpublished follow-up direction from their own work together with the necessary conditions a submission must meet. While assessing the quality of a follow-up paper is challenging in general, using these conditions to form a rubric makes evaluation more tractable for research problems that have no reference answer. We have nine frontier agents write 15 follow-ups each, spanning quantum computing, plasma physics, computational chemistry, materials science, and scientific machine learning domains. None of the 135 submissions satisfies all the conditions, and 114 fail our integrity check, most often because the paper claims work that was never done or reports numbers that contradict the recorded evidence. Overall, agents readily produce papers that look like follow-ups, but none meets the standard of the domain scientist, leaving substantial headroom. Our code and data are publicly available. 

---
# DivAlgo: A systematic evaluation of algorithmic diversity for code generation models 

**Authors**: Anubhav Jangra, Smaranda Muresan  

**Link**: [PDF](https://arxiv.org/pdf/2610.19157)  

**Abstract**: Code generation models are actively being used in professional and educational settings. However, the competence of large language models (LLMs) to generate algorithmically diverse code remains largely unknown. We propose the DivAlgo benchmark, a large-scale dataset constituting algorithmically diverse solutions to programming problems. We also introduce a lightweight similarity metric to enable large-scale evaluation. Our experiments across 24 modern code generation models found that uncontrolled model-generated code is significantly less diverse than human solutions (capturing ~31% of human diversity), and even the best diverse code generation strategy could achieve only ~70% of human diversity. Moreover, while it was possible to steer the models to generate solutions based on human strategies, they were unable to reproduce the diversity of the human solutions. These findings indicate the limitations of code generation models in capturing the algorithmic diversity humans exhibit, and that overexposure to model-generated code can narrow programmers' algorithmic vocabulary. 

---
# MA-Bench: Measuring Operational Planning in Language Models 

**Authors**: Rodney Lafuente-Mercado, Robert Sincero, Phil Sage, Yuan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2610.19138)  

**Abstract**: How well can a language model turn a large, incomplete and partly contradictory body of documents into a plan when no single correct answer exists? We introduce Mission Analysis (MA)-Bench, built on mission analysis, the first step of military operational planning, in which a staff reads the full situation and states the problem before anyone proposes a solution. Its 10 expert-built scenarios pair synthetic packets of ten to twelve documents with 1,257 expert-written pass/fail requirements, each containing information on their source, question type, and importance level. A reasoning-model judge grades each brief against the requirements without seeing the documents; it grades like a fifth credentialed planner, leaving a four-planner panel's agreement unchanged (Fleiss $\kappa$ 0.42 to 0.41), and a judge from a second provider reproduces the model ranking (Spearman 0.94). Across twenty frontier and open-weights models the best gets 62% of the rubric items right. Models are strongest on facts stated in the documents (58%) and weakest on judgments that must be inferred (33%), and six failures in ten are omissions rather than errors. MA-Bench grades the determinations, not the prose. We release the packets, rubrics, prompts, scorer and every graded brief with item-level verdicts. 

---
# Plant Memory: Technical Language Processing and a Verified Maintenance Ontology for Agent-Accessible Failure Knowledge from SAP PM Work Orders 

**Authors**: Akash Raj  

**Link**: [PDF](https://arxiv.org/pdf/2610.19131)  

**Abstract**: Maintenance work orders are the largest written record of how industrial equipment fails, yet in most plants they are close to unusable for analysis. Short texts are abbreviated and misspelt, asset identity is buried in free text, work-type labels contradict the work described, and the fields that matter most for reliability (symptom, root cause, action taken) are usually empty. As a result, diagnostic knowledge stays with senior technicians and leaves when they do. We present Plant Memory, a system that applies technical language processing (TLP) to SAP Plant Maintenance (PM) exports and builds an ISO 14224-aligned maintenance ontology in which every extracted statement is a reified fact carrying a confidence score, the evidence it was read from, and a verification state. Extraction is rule-first, with an optional large language model (LLM) acting as a second reader whose disagreements raise a fact's review priority rather than overwrite it. Technicians verify facts through a review queue, answer targeted knowledge-gap questions where failures repeat without a recorded cause, and approve or reject ontology drift proposals when new terms appear. The resulting graph is served to people through a web application and to AI agents through a Model Context Protocol (MCP) server, so both query the same verified knowledge and every answer cites the work orders behind it. On a synthetic 12-month export of 400 work orders from a parboiled-rice plant, rule-based extraction reaches 96.2% component accuracy, 88.9% failure-mode accuracy and 100% root-cause precision at 62% coverage; on 28 diagnostic questions, ranked root causes from the ontology are correct at top-1 in 82.1% of cases against 53.6% for the raw SAP root-cause column. 

---
# Unlocking Structured Pruning for Reasoning Language Models via Attention-Preserving MLP Trimming 

**Authors**: Woosang Lim, Haneul Yoo, Vincent-Daniel Yun, Sungjoo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2610.19121)  

**Abstract**: Structured pruning reduces LLM inference costs by removing model components to produce smaller dense models. However, existing methods can cause reasoning performance to collapse on long-form reasoning tasks such as AIME. We propose \textit{LayerTrim}, a training-free structured pruning method that identifies redundant decoder layers and selectively trims their MLP features while preserving attention and decoder depth. LayerTrim is the first training-free structured pruning method designed to preserve long-form reasoning without gradients or recovery training. Across three reasoning language models and five benchmarks, LayerTrim outperforms existing structured pruning methods in most evaluated settings, improving average accuracy by up to 21.4 percentage points over RESP. On \texttt{Qwen3-14B}, LayerTrim achieves 1.24$\times$ faster prefill and 19.7\% lower peak GPU memory than the dense model, while better preserving reasoning ability. It also achieves the shortest average generation length among evaluated pruned models, while retaining the highest average accuracy. These results demonstrate that LayerTrim improves reasoning efficiency through both structural compression and reduced token generation. 

---
# Towards a Data-Centric Understanding of Spatial Reasoning in Multimodal Large Language Models 

**Authors**: Mahdieh Mallahnezhad, Masoud Jafaripour, Mohamadali Tofigh, Sheila Schoepp, Osmar R. Zaiane  

**Link**: [PDF](https://arxiv.org/pdf/2610.19108)  

**Abstract**: Multimodal Large Language Models (MLLMs) have been widely applied in embodied agents and robotics for vision tasks. However, MLLMs suffer from limitations in spatial reasoning and perception. Post-training techniques have shown promise for improving these capabilities; yet, there are limited data-driven surveys of such approaches. In this survey, we consider approaches to fine-tuning pretrained MLLMs for spatial reasoning tasks, ranging from supervised to reinforcement learning, with an emphasis on the datasets used. We investigate themes of dataset construction, data modality and annotation, and evaluation of the resulting models. Based on these themes, we categorize research into three groups: dataset design, dataset construction and evaluation. We find four main methods for constructing fixed spatial annotations: rules and templates, LLM prompting, procedural generation, and human-in-the-loop. We also observe substantial variation in evaluation approaches, including different scoring methods, tolerance metrics, and benchmark types. Finally, we identify opportunities for future research in unified dynamic annotation, efficient spatial reasoning, standardization of evaluation, expanded spatial range, and human-centered spatial interactions. 

---
# GroundGUI: GUI Grounding by High-Quality Data and Distillation-to-Reinforcement Learning 

**Authors**: Qingsong Xie, Jiahao Tian, Jiacheng Tang, Zhifeng Li, Madalina Ciobanu, Navan Preet Singh, Qingqing Mao, Ritankar Das  

**Link**: [PDF](https://arxiv.org/pdf/2610.19100)  

**Abstract**: Constructing robust GUI agents relies on multimodal visual grounding: the ability to precisely map natural language intents to actionable on-screen components. Despite abundant desktop, web, and mobile benchmarks, high-fidelity resources tailored for diverse environments are surprisingly scarce. To bridge this gap, we introduce the Screen Referring Grounding Dataset (ScreenRef), a comprehensive GUI grounding dataset consisting of GUI screenshots paired with natural language referring expressions and box or point annotations. Spanning desktop, web, and mobile domains, ScreenRef comprises 586K screenshots and over 633K referring expression instances, covering tasks such as text recognition, visual matching, spatial understanding, and refusal scenarios. To boost the model's grounding performance, we propose the Distillation-to-Reinforcement Learning (D2RL) method to address the limited generalization of supervised fine-tuning and the sparse reward problem in reinforcement learning. Utilizing ScreenRef, we train Qwen3-VL-8B-Instruct via sequential SFT and D2RL training, yielding GroundGUI-8B. GroundGUI-8B achieves 95.2% on ScreenSpot-V2, 63.9% on ScreenSpot-Pro, 65.4% on OSWorld-G, 86.8% on MMBench-GUI L2, and 38.3% on UI-Vision. Among 8B-scale models, it produces substantial gains over models trained on open-source datasets, with performance comparable or even superior to models trained on closed-source data. Code, models, and datasets are available at this https URL. 

---
# Beyond Aggregate Accuracy: Item-Level Transitions in Prompt Optimization 

**Authors**: Chinmoy Mitra, Tarek Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2610.19084)  

**Abstract**: Automatic prompt optimization (APO) uses feedback from model responses to improve prompts automatically. However, its effectiveness is usually measured through overall scores, which do not reveal which questions become correct, which previously correct answers become wrong, or whether improvements extend to unseen questions. We introduce Reflective Feedback-Driven Prompt Optimization (RFDPO), a tool that tracks recoveries (incorrect to correct), regressions (correct to incorrect), and churn (all changed outcomes). RFDPO uses this feedback to guide subsequent revisions and records prompt changes throughout optimization. We evaluate RFDPO across five benchmarks covering law, knowledge, instruction following, mathematics, and privacy, using GPT-5 as the optimizer and GPT-4o-mini and GPT-4.1 Mini as solvers. Our results show that overall accuracy can hide substantial changes: in five of nine configurations, churn was at least five times the absolute net change, and a single-item net change concealed up to 17 changed outcomes. Average validation gains were 7.9 percentage points, compared with 5.2 points on held-out tests. In a controlled ablation on LegalBench-Hearsay, item-level feedback reduced churn by 40% ($p=0.024$) while maintaining similar net gains. These findings motivate reporting recovery, regression, churn, and performance on unseen data alongside overall accuracy. 

---
# Smart Alone, Blind Together: Why Accuracy Fails to CertifyTheory of Mind in Multi-Agent Teams 

**Authors**: Will Beaumaster, Paul Schrater  

**Link**: [PDF](https://arxiv.org/pdf/2610.19051)  

**Abstract**: Coordinators that act on what they infer about teammates are usually certified by the accuracy of those inferences. We argue that accuracy is the wrong certificate, for two reasons. First, the accuracy needed to act depends on what a wrong action costs, so the bar belongs to the deployment, not the module. Second, a module's errors are rarely independent, and dependence has a sign: positively associated errors help where their damage overlaps and hurt where it compounds, so an accuracy threshold fails in a predictable direction, and once wrong beliefs interact and can do harm, no rule based on accuracy is right for every module. We test these predictions in task assignment, Level-Based Foraging and Hanabi, with injected errors, trained and Bayesian modules and five language models, and in human judgments. Accuracy fails in both directions, approving harmful language-model deployments in Hanabi, where shared errors compound, and rejecting beneficial ones in task assignment, where costly errors are rare. In our tasks it also buys far less redundancy than independence predicts: from GPT-5.4-mini to Claude Opus 5.5, error on teammates the models have evidence about falls from 32% to 11%, but the chance that all three observers misread one falls only from 16% to 9% (independence: 3.4% to 0.15%). We propose certifying the deployment, with a Blind-Spot Gate that prices error costs, pairwise interactions and measured dependence without deploying the module. On held-out data from the cells it was developed on, it makes no significant false approval in 20 splits (accuracy: 3.2 per split), and in preregistered tests on modules it was not developed on, it recovers what accuracy rejects (seven classifiers: regret 0.004 against 0.479) and catches what accuracy approves (a Bayesian module: significant false approvals in 17 of 20 splits for accuracy, 1 for the gate). 

---
# Graph Coarsening Enables Escaping Local Optima in Compute-Constrained Nonconvex MAX-CUT 

**Authors**: M. Harper Langston, Pierre-David Letourneau, Dalton Jones, Harris Teague, Peter Scott, Mingu Lee, Richard Lethin, Chris Lott, Roland Memisevic, Fatih Porikli  

**Link**: [PDF](https://arxiv.org/pdf/2610.19043)  

**Abstract**: Because its binary optimization landscape contains exponentially many local optima, MAX-CUT is highly sensitive to initialization. We study Particle Polynomial Relaxation (PPR), a moment-based nonconvex continuous reformulation possessing an advantageous optimization landscape. We warm-start PPR with Combinatorial Multigrid (CMG), a spectral graph-coarsening method that produces structure-aware initial solutions in near-linear time without training data. On \texttt{t2g20} ($n=400$), benchmark instance with known exact optimum, the combined CMG+PPR pipeline reaches $97.5\%$ of optimum on average and $98.4\%$ on the best run. An extended variant with discrete local-search refinement reaches the published optimum on all sixty Biq Mac instances evaluated by a recent neural branch-and-bound method. Using a sparse CSR representation, the same approach scales to G-set instances with up to $20{,}000$ vertices reducing memory requirements by up to $3333\times$ while maintaining a consistent improvement over the CMG warm start. Gains are largest on sparse weighted graphs; on dense graphs we complement the pipeline with spectral sparsification. 

---
# Tracing Morphology, Semantics, and Divergence in Language Acquisition 

**Authors**: Forough Shirin Abkenar, Haoshuo Fu, Liang Feng, Ning Yan, Masood S. Mortazavi  

**Link**: [PDF](https://arxiv.org/pdf/2610.19041)  

**Abstract**: Understanding the dynamics of language acquisition in Large Language Models (LLMs) is crucial for uncovering how statistical representations evolve into structured linguistic capabilities. However, most existing studies focus primarily on fully trained models, leaving the developmental trajectory across intermediate training stages largely underexplored. In this work, we leverage intermediate checkpoints from open-source models (e.g., Pythia and OLMo) under teacher-forcing evaluation to systematically investigate language acquisition across training steps and model scales. We focus on four core characteristics: morphological structure acquisition, embedding drift, repetition dominance, and co-occurrence effects. Our findings reveal that models acquire morphological structures well before semantic embeddings stabilize. Furthermore, we show that training dataset size has a positive impact on divergence recognition, while co-occurrence during pre-training and repetition during inference have a negative impact on divergence recognition and associative link generation. 

---
# Who Should We Listen to More? Welfare-Aware Preference Acquisition for Pluralistic Alignment 

**Authors**: Ruomeng Ding, Tianwei Gao, Lianrui Geng, Mohit Bansal, Zhun Deng  

**Link**: [PDF](https://arxiv.org/pdf/2610.19035)  

**Abstract**: Pluralistic alignment needs to accommodate diverse preferences under limited human feedback. Using AI simulators to label preference comparisons reduces annotation costs, but variation in simulation accuracy across individuals can leave some people's preferences poorly represented in training. We study how to allocate human feedback across people to construct training data for pluralistic alignment, and how this allocation affects the trained policy. To make this allocation explicit and controllable, we define a social welfare objective over individual supervision utilities that measure label quality. Each query's marginal welfare gain depends on its expected supervision improvement, the recipient's current utility, and application-designated priority. Welfare-Aware Preference Acquisition (WAPA) uses auxiliary reference models to estimate query gains and individual utilities, then selects queries by estimated marginal welfare gain while updating recipient utilities. We prove exact optimality of the acquisition algorithm for the estimated welfare objective under a fixed query budget and model predictions. Experiments on six Roleplay collections and four OpinionQA regions demonstrate improved supervision and controllable allocation tradeoffs. Policy training on four Roleplay collections improves preference prediction for both seen and unseen individuals. When prioritizing a designated group and poorly represented individuals, acquiring human labels for 20% of comparisons recovers 75-86% of the average predictive utility gap between policies trained entirely on AI-generated labels and entirely on ground-truth labels. 

---
# The Agentic Scientist: Orchestrating Coding Agents for Automated Scientific (Co-)Discovery 

**Authors**: Robert Tjarko Lange, Luca Grillotti, Yutaro Yamada, Rujikorn Charakorn, David Ha  

**Link**: [PDF](https://arxiv.org/pdf/2610.18933)  

**Abstract**: Modern coding agents are fundamentally reshaping the role of human input for computational research: they write code, launch in silico experiments, analyze their results, and autonomously operate with ever-increasing time horizons. Despite these developments, fully autonomous research agents suffer from a limited ideation radius and myopic hill-climbing search dynamics. To bridge this gap, we introduce The Agentic Scientist, an end-to-end distributed pipeline for human researchers to conduct computational research alongside modern coding agents. Building on its predecessor, our system decomposes research into task drafting, idea generation, planning, sandboxed experimentation, result aggregation, baseline and ablation construction, post-hoc analysis, and report writing. Importantly, each step in the pipeline is powered by the next generation of coding agents and leverages agentic self-verification. The overall research process is driven by a meta-controller that manages parallel research agents and enforces rubric verification at each stage. This design enables the same orchestration layer to be reused across heterogeneous domains with different execution environments, agentic coding harnesses, and evaluation metrics. Our system can be used across a spectrum of autonomy: either as a co-scientist tool or to perform end-to-end scientific investigation. Within the co-scientist setting, we leveraged a development version of our system to submit a single research paper, which got accepted for publication at ALIFE 2026. We further apply The Agentic Scientist fully autonomously to open problem-solving tasks in machine learning, and demonstrate the system's capability of scientific discovery across different domains of computational research. Our ablations suggest that reliable scientific autonomy depends not only on base model capabilities, but also on the cross-agent orchestration layer. 

---
# Lotus: Learning to Understand Time-series by Learning to Forecast 

**Authors**: Van Dai Do, Huu Hiep Nguyen, Minh Hoang Nguyen, Manh Nguyen, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2610.18877)  

**Abstract**: Time-series language models typically align numerical signals with language using labels, captions, or question-answer pairs. Although such supervision encourages linguistic representations, it does not explicitly require them to preserve information about how the underlying time-series signal evolves. We introduce forecast grounding, a pretraining principle that requires language-aligned time-series representations to remain predictive of the future evolution before they are adapted to downstream tasks. We instantiate this principle in LOTUS, which divides each time-series channel into non-overlapping patches and maps them into the LLM embedding space. It then predicts the next patch from the LLM hidden states of the time-series tokens, encouraging the learned representation to retain temporal information. To capture both temporal changes and periodic patterns, we also propose a temporal-spectral patch representation that combines each raw time-series patch with its spectral representation before mapping it into the LLM embedding space. Across three language model backbones and six downstream evaluation datasets, LOTUS achieves the best average performance with minimal trainable parameters, reaching 59.91% accuracy and outperforming the strongest matched baseline by 3.36%. Our representation analyses further show that LOTUS uses the numerical input and preserves temporal structure in its learned patch representations. These results establish forecast grounding as a promising pretraining principle, enabling compact time-series language models to outperform models 25 to 50 times their size on TSRBench. 

---
# Knowledge Distillation from Large Reasoning Models to Compact Student Models: A Case Study on the John O Bryan Mathematics Competition 

**Authors**: Gaurab Baral, Aaditya Khanal, Yangyang Tao, Junxiu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.31048)  

**Abstract**: This paper investigates knowledge distillation from a large reasoning model (DeepSeek-R1) to a compact student model (Qwen2.5-7B). Using historical problems from the John O'Bryan Mathematics Competition at Northern Kentucky University (2011-2025), we build a Chain-of-Thought (CoT) training corpus through a dual-agent framework. The dataset is used to fine-tune the student model with Low-Rank Adaptation (LoRA) on Apple Silicon hardware using the MLX framework. The base Qwen2.5-7B model achieves 64.67% accuracy on competition problems, while the DeepSeek-R1 teacher achieves 91.40%. An initial 1,000-iteration training run revealed severe overfitting, with validation loss reaching a minimum at iteration 200 before rising steadily. Based on this finding, we ran five independent training runs each limited to 200 iterations with varied random seeds to assess result stability. Across these five runs, the fine-tuned student model achieves a mean accuracy of 69.43% (std dev 0.17%) on the competition dataset, a 4.76 percentage-point improvement over the base model, and generalizes to 73.1% (std dev 0.18%) on the MATH-500 benchmark. We further study how response length affects answer quality across six reasoning levels (R1-R6): accuracy declines consistently from 69.43% at R1 (mean 220 words) to 41.9% at R6 (mean 31.2 words), with the two-person speed section most sensitive to token reduction. These results demonstrate that CoT distillation improves compact student models and that response length is a critical factor in mathematical reasoning quality. 

---
