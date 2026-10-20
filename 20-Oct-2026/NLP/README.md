# OpenVLHarness: Real World Visual Agentic Tool-Use with Multimodal Memory 

**Authors**: Yu Zhou, Cheng-Fu Yang, Rui Sun, Di Wu, Sibo Peng, Mingyang Zhang, Zhen Yang, Ruiyi Zhang, Nanyun Peng, Zhe Gan, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26860)  

**Abstract**: Multimodal large language models (MLLMs) have made rapid progress on visual reasoning, yet many tasks require evidence that is difficult to obtain through internal reasoning alone: our controlled study on 22 benchmarks shows that simply increasing the inference budget of the base model yields only modest gains. Such tasks demand grounded perception, external knowledge, and numerical measurements obtained through heterogeneous tools, whose outputs must be made interpretable to the model and retained across reasoning steps. We introduce OpenVLHarness, a multimodal reasoning harness organized around a capability layer and persistent multimodal memory. The capability layer combines task-level tool interfaces with evidence rendering, providing consistent access to heterogeneous backends and converting their outputs into model-readable observations. Persistent multimodal memory retains the corresponding images, structured data, and web references as addressable artifacts throughout each query. A coding interface executes generated programs over stored data, enabling computations that combine results from multiple tools. Together, these mechanisms support an iterative process of acquiring, inspecting, and composing multimodal evidence. Evaluated on 22 benchmarks with eight open and proprietary backbones, OpenVLHarness improves average scores by 9.7-12.6 points over matched no-tool baselines, including +11.2 with Qwen3-VL-8B, +9.7 with Kimi K3, and +9.9 with GPT-6 Astra. With GPT-6 Luna at medium reasoning effort, OpenVLHarness outperforms the base model, Codex, and Visual Sketchpad at maximum effort by 7.7-8.5 points while incurring at most 37% of their model-token cost. 

---
# LatComm: Reusable Context Compression for Latent Communication between Large Language Models 

**Authors**: Huawei Lin, Jinting Liu, Tan Yu, Weijie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.26840)  

**Abstract**: Systems that combine several large language models (LLMs) often use one context for many questions. When the context is passed between models as text, the receiving model processes all of its tokens again for every question. Avoiding this cost requires a context representation that is compact, reusable across questions, and readable by a different LLM. Existing compression and latent communication methods each provide only some of these properties. We introduce LatComm, a Latent Communication interface that provides all three by separating encoding from reading: the context is encoded once without the question, and the receiver selects what to read for each question. A frozen context encoder LLM compresses the context into a fixed-size packet and sends it to a different frozen receiver LLM, which reads the packet through cross-attention, with gates computed from its prompt. On 12 multiple-choice question-answering datasets, LatComm improves average accuracy over single-vector baselines by up to 10.2 points. On QuALITY, it reduces the receiver's input by 57x and answers the questions of an article 3x faster. 

---
# Competing Learning in Hybrid Architectures 

**Authors**: Xingyu Shen, Yingfa Chen, Ziqing Qiao, Chaojun Xiao, Xu Han, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26830)  

**Abstract**: Many modern language models have adopted hybrid architectures that incorporate both linear-time recurrence and softmax attention. Modern recurrence layers provide a highly expressive mixing mechanism while ensuring high throughput by compressing contextual information into a fixed-size state, but underperform softmax attention in contextual information retrieval. In this paper, we reveal a Competing Learning phenomenon where different kinds of mixers compete to learn the same function, and when any layer learns a function, it suppresses the corresponding learning signal for other layers. Layer attribution analysis reveals that under typical data distributions, modern recurrence learns retrieval functions faster than softmax attention, causing attention to fail to learn retrieval in most cases. Stopping weight updates in recurrence in the middle of training helps attention learn retrieval. Based on these results, we propose a staged training procedure for pre-training hybrid architectures on natural language. Empirical results on models up to 1B parameters show that our method results in substantial improvement in language modeling loss, common-sense reasoning, as well as a wide range of retrieval-related tasks across different context lengths. 

---
# KEEP-Bench: Evaluating Evidence Preservation in Context Compression 

**Authors**: Ziqing Man, Di Wu, Jia-Chen Gu, Junyi Zhang, Kai-Wei Chang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.26827)  

**Abstract**: Context compression improves long-context efficiency but may introduce faithfulness issues through an information bottleneck. Existing evaluations rely on QA metrics, but a reader can still answer correctly from an unfaithful compression, so these metrics underestimate such loss. To bridge this gap, we introduce KEEP-Bench, a diagnostic benchmark that directly assesses compression faithfulness with 4,102 cases across three challenging settings that require keeping evidence for plausible reference resolutions, explicit qualifiers, and minority perspectives. Experiments show that even with tailored prompting, the strongest models remain unreliable: despite context windows exceeding 100K tokens and inputs averaging only 3.1K tokens, the best open-weight model (31B) preserves all required evidence in only 66.8% of cases, and the best proprietary model in only 61.6%. Performance also varies sharply across evidence structures, so conclusions from limited structures may be biased. We further introduce a metric, Evidence Preservation Return (EPR), to relate evidence recall to inference FLOPs, and find that it peaks at moderate scales (4B-8B), where recall remains low. Additional output budget mainly benefits already strong models and widens capability gaps, yet the stronger preservation of these models comes at a steep computational cost. These findings highlight the importance for future work to investigate compressors with strong information preservation under a small computation budget, rather than relying on a large output budget or model scaling. 

---
# RefCraft: Benchmarking Visual-Reference-Conditioned Agentic Generation through Web Development 

**Authors**: Yuchi Wang, Rongyao Fang, Weihao Yu, Xilin Wei, Tiancheng Gu, Dingkang Yang, Hang Zhang, Dayiheng Liu, Hongsheng Li, Shuai Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.26826)  

**Abstract**: Beyond basic tool use and coding skills, effectively incorporating visual references into agentic generation is equally essential for applying AI agents to real-world productivity tasks. We study this crucial capability through webpage development, a practical and readily evaluable setting in which existing benchmarks remain largely idealized: they typically focus on reconstructing visual targets or generating webpages from textual instructions alone, overlooking the more common workflow of developing from visual examples. Specifically, we introduce RefCraft, a benchmark comprising 300 realistic multimodal webpage-development tasks that combine natural-language instructions with screenshots from one to five reference webpages. Agents must selectively interpret and utilize these references while rendering, visually inspecting, and iteratively refining their outputs. We evaluate the resulting open-ended artifacts using independently verifiable keypoints and complementary programmatic and VLM judges. Across nine frontier (M)LLMs, the strongest system achieves an overall score of only 52.12 out of 100, revealing that visual-reference-conditioned agentic generation remains a substantial challenge for current models in practical webpage-development settings. 

---
# MemT-1.0: Towards Memory-Native Evolution 

**Authors**: Zhiyu Li, Zeyu Zhang, Linyun Liu, Yiwei Bu, Chenyang Xi, Xichong Zhang, Yi Yu, Yu Yu, Zhichen Liu, Yuxiao Yi, Chunyu Li, Haoran Liu, Zhenkai Ding, Le Zhou, Juncheng Zhang, Boyu Chen, Zehao Lin, Guorui Li, Renyu Fu, Ji Li, Youyang Xia, Jiajun Shen, Harvey Li, Jingsong Zhou, Helan Hu, Tong Shen, Tianxiao Xu, Chenye Han, Weijie Li, Zhongquan Zhou, Hongming Xu, Yuekeng Li, Erenkai Zhu, Peng Yu, Wenhao Li, Jingyi Kang, Xixuan Hao, Xuzhou Zhu, Can Chen, Jiawei Liu, Chen Jiang, Tao Jiang, Fangwei Xiang, Zepeng Yao, Kai Chen, Guang Yang, Mingfei Lu, Chunhui Li, Zhengxuan Jiang, Bo Tang, Shiyu Zhan, Yihang Sun, Jie Chu, Wenjun Xiong, Zhenxiang Hua, Chen Lin, Detong Su, Dongyu Deng, Qiuping Ding, Guofeng Ding, Si Chen, Guohua Wen, Xu Chen, Zhi-Qin John Xu, Ying-Cong Chen, Ningyu Zhang, Haofen Wang, Muning Wen, Haotian Chen, Wentao Zhang, Xuanhe Zhou, Rita Zhang, Junchi Yan, Feiyu Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2610.26825)  

**Abstract**: Memory-native evolution requires making memory an intrinsic capability of foundation models, enabling them to continuously accumulate and utilize information. To this end, we introduce MemT-1.0, a new architecture of memory foundation models that internalize memory as part of model computation. Building on Metis, MemT-1.0 employs Gated DeltaNet-2 with a stable token-level memory update mechanism within each turn. We further introduce Mixture-of-Storage to scale native memory capacity. Following a deliberate backbone preparation, we develop a two-stage training pipeline consisting of continued pre-training and post-training, supported by 34B tokens of training data and efficient training and inference infrastructure. We release a model family comprising MemT-1.0-Nano and MemT-1.0-Flash, and conduct a comprehensive evaluation of memory and general capabilities. The results show that they outperform the parametric memory baselines on average and remain competitive with retrieval-augmented methods, while largely preserving general capabilities. At a history length of 128K tokens, MemT-1.0-Nano achieves approximately $1.86\times$ end-to-end inference speedup and a $66\times$ reduction in persistent storage relative to full context KV cache. By internalizing retrieved procedural experience, MemT-1.0 improves task success on agentic tasks, providing a practical step toward memory-native evolution. 

---
# SpecMRL: Redesign Matryoshka Representation Learning within Parametric Spectrum 

**Authors**: Zhongxin Chen, Songhao Wu, Heng Cui, Cong Li, Xiao Zhou, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.26824)  

**Abstract**: Matryoshka Representation Learning (MRL) is designed to support flexible representation sizes by jointly optimizing nested prefixes. While effective, this design lacks an explicit constraint to ensure that the leading portion provides a suitable compact encoding. To address this, we propose SpecMRL, which reorganizes the feature space to enable more effective prefix readout. Specifically, we extend MRL with a parameter-derived spectral basis organized with feature importance. SpecMRL otherwise follows standard MRL, with prefix readout performed in the learned spectral space. Experiments on MTEB show that SpecMRL outperforms MRL at all evaluated reduced dimensions, with larger gains under more aggressive compression. These results suggest that the learned spectral basis better prioritizes informative directions, enabling more effective representation compression. Notably, SpecMRL extends the performance frontier of MRL, surpassing full-dimensional MRL with only half the embedding dimensions. 

---
# AfriBED: a Speech Recognition Bias Evaluation Dataset for Low Resource African Languages 

**Authors**: Claytone Sikasote, Hussein Suleman, Jan Buys  

**Link**: [PDF](https://arxiv.org/pdf/2610.26821)  

**Abstract**: Despite advancements in automatic speech recognition systems (ASR), current systems exhibit speaker-attribute bias, i.e., their speech recognition performance is biased against certain speakers based on their demographic attributes such as gender, accents, and speaker type. However, accurately assessing speaker attribute bias in a given ASR system requires the availability of a bias evaluation dataset with comparable speech across subgroups of a speaker attribute. In this paper, we introduce African speech Bias Evaluation Dataset (AfriBED), a novel speech recognition bias evaluation dataset for two low resource African languages, Bemba and Nyanja, to detect and estimate gender and speaker-type bias for ASR systems. The dataset comprises 9 hours of speech with 3928 utterances recorded by 80 speakers for Bemba, while the Nyanja set includes 6260 utterances with approximately 18 hours of speech recorded by 127 speakers. The results of our baseline experiments demonstrate that the datasets are useful for examining gender and speaker-type attribute bias in the target languages. 

---
# ASTOR: Benchmarking and Improving Autoformulation for Soft-Constraint Optimization with a Formal Grammar 

**Authors**: Amer Goel, Yeganeh Alimohammadi, Andrew Daw  

**Link**: [PDF](https://arxiv.org/pdf/2610.26819)  

**Abstract**: Real-world decision problems combine two qualitatively different kinds of constraints: requirements that must be satisfied, called hard constraints, and preferences that can be violated, called soft constraints. Because these problems are often posed to decision-makers in natural language, large language models offer a natural tool for assisting with their solution. This has given rise to autoformulation, which focuses on using LLMs to translate natural-language descriptions of decision problems into executable mathematical optimization models. However, this research often neglects soft constraints like those proposed by consumers, managers, and planners, despite the fact that many practical problems require modeling them explicitly. We introduce ASTOR, an inference-time framework that maps a natural-language decision problem to a grammar-guided abstract syntax tree (AST) before deterministically compiling the resulting representation into executable optimization code. We also introduce SoftOR, a unified benchmark of natural language problems which focuses on soft constraint formulations. On SoftOR, ASTOR achieves the best aggregate accuracy among comparable single-request and multi-agent LLM methods while remaining competitive on standard autoformulation benchmarks that include only hard constraints. Code and data are available at this https URL. 

---
# Safety Mid-Training: Internalizing LLM Safety as a Foundational Capability 

**Authors**: Zhengyue Zhao, Liwei Jiang, Yejin Choi, Chaowei Xiao1  

**Link**: [PDF](https://arxiv.org/pdf/2610.26809)  

**Abstract**: Safety alignment for large language models (LLMs) is primarily performed during post-training, while recent work has also begun incorporating safety data into pre-training. Yet safety can degrade when later training no longer includes safety data, and gains in safety often come at the cost of general capability. This tension suggests that safety and general reasoning should not be treated as separate training goals. We instead view safety as an LLM capability that can be learned through a reasoning structure shared with general tasks. Both require the model to understand intent or task goals, apply relevant knowledge, and reason about potential consequences. We introduce safety mid-training, a stage between pre-training and post-training that jointly develops safety and general reasoning through this shared structure. Given safety and utility prompts, the base model generates structured documents and reasoning traces that are then used for mid-training. Safety mid-training reduces average attack success rate (ASR) by 72% to 81% relative to Qwen3-1.7B, 4B, and 8B base models, while also improving general capability. The benefits persist through subsequent post-training. After general-only SFT, mid-trained models largely retain their safety, achieving lower ASR than the baselines without sacrificing general capability. They also retain safety under general-only GRPO without safety rewards, while subsequent RL reaches a better safety-utility frontier when initialized from the mid-trained models. Together, these results suggest that learning safety and general reasoning within a shared mid-training stage can reduce the safety-utility trade-off and make safety more robust to subsequent training. 

---
# RESIN: Multi-Chain Reasoning with Entity-Structured Indexing 

**Authors**: Xuanyu Meng, Jash Rajesh Parekh, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.26782)  

**Abstract**: Multi-chain reasoning is a challenging task not only because the multi-chain evidence is dispersed, but also because different parts of reasoning may require distinct, interacting chains of support. Flat retrieval can flood a model with loosely related text, whereas early selection of a single reasoning path can discard evidence that becomes useful only after other facts are considered. We introduce RESIN (Reasoning with Entity Structured Indexing), which uses structure to locate evidence while keeping the original source as the basis for reasoning. RESIN decomposes a question into focused one-hop needs, retrieves a small set of source-grounded units for each hop, and places only this relevant evidence in a compact reasoning buffer. The buffer preserves alternative supports until final composition, making intermediate decisions explicit without forcing a brittle path. On the full HolisQA benchmark, RESIN achieves 82.92 context recall and 76.70 answer relevancy over all questions, with higher reported CR and AR across five domains and higher F1 in four domains than the published baselines. Ablations show that both typed evidence diversity and hop-wise decomposition are essential, supporting a clear conclusion: Precise evidence organization and delayed commitment make multi-chain reasoning effective and auditable. Our code is available at this https URL. 

---
# Monsoon: Scalable Spontaneous Speech Collection for Low-Resource ASR in 50 Languages 

**Authors**: Shobhit Banga, Manas Dhir, Aaditya Pareek, Amritansh Walecha, Bhaskar Singh, Hanuman Sidh, Manmeet Kaur, Sagar Jain, Supriya Paul, Vanshika Chhabra, Utkarsh Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.26776)  

**Abstract**: For most languages, automatic speech recognition (ASR) is limited less by model architecture than by a lack of accurately transcribed spontaneous speech. We present Monsoon, a framework for collecting such data at scale. It elicits unscripted conversations between paired contributors, screens recordings with automated classifiers, corrects transcripts over five levels of human review, and selects segments for lexical and acoustic coverage. It has collected speech in 50 languages; we analyse seven 3,500-hour corpora. At equal volume, its quality stages reduce semantic word error rate (WER) by 10-39% relative to a conventional protocol. Compared with an architecture-matched baseline trained on public corpora, Whisper fine-tuned on Monsoon performs comparably on the baseline's own test sets and reduces semantic WER by 20-44 points on conversational benchmarks. The gains transfer to a second architecture and grow with data volume. We release the framework, its quality-control models, and the fine-tuned models. 

---
# TokenGate: Adapting LLMs for Fast and Memory-Efficient Prefill with Learned Token Pruning 

**Authors**: Liana Patel, Rishabh Ranjan, Ion Stoica, Matei Zaharia, Carlos Guestrin  

**Link**: [PDF](https://arxiv.org/pdf/2610.26754)  

**Abstract**: Expert human readers process large inputs (e.g., code repositories and long documents) by skimming to closely read only what matters. On the other hand, language models process all input tokens through each model layer, resulting in significant processing costs for long-context inputs. We introduce TokenGate, a small, drop-in component that extends existing trained models to enable end-to-end learnable token pruning over long-context inputs for fast and memory-efficient prefill. To enable end-to-end learning despite the non-differentiable top-K operator for token pruning, we introduce token-gated attention, a mechanism inspired by Mixture-of-Experts that re-weights attention logits using token importance scores. Across diverse tasks, TokenGate achieves $1.4$-$11.3\times$ faster time-to-first token (TTFT) and $1.5$-$3.4\times$ lower peak memory at no accuracy degradation for prefill in transformers. Compared to the best-performing prior methods for token-pruning, TokenGate gives up to $6.4\times$ faster prefill and $2\times$ lower peak memory. TokenGate natively extends both standard transformers (e.g., Qwen3) and hybrid-attention models (e.g., Qwen3.5) with simple lightweight adaptation comparable to standard parameter-efficient fine-tuning. We make our code available at: this https URL. 

---
# EpiLens: Lightweight RAG for Epidemiological Knowledge Acquisition 

**Authors**: Vincenzo Perri, Ciro Cattuto, Daniela Paolotti  

**Link**: [PDF](https://arxiv.org/pdf/2610.26730)  

**Abstract**: To prepare for future outbreaks, researchers need to know what studies have been done and what data they use. Mapping a large literature by hand is slow, while large language models alone can misstate or conflate sources. We present EpiLens, a lightweight, open-source retrieval-augmented generation package that applies a fixed retrieval template and output schema to every paper in a corpus and links outputs to source passages. We build a corpus of 2,297 papers from the sources of a multi-pathogen parameter study and evaluate EpiLens's paper-classification workflow against 214 manually labelled papers. Without task-specific training data, the classifier matches or exceeds the tested supervised and zero-shot baselines across four tasks. The study's early extraction sheets cite published sources for 327 estimates. We link each estimate to a cited document or to corpus studies included by a review. Then, we use the classifier's labels to compare the papers associated with different parameters and pathogens. EpiLens labels more papers linked to reproduction numbers than fatality risks as model-based (54% versus 22%, compared with 62% versus 24% in the manual labels). The gap is consistent with the frequent use of transmission models to estimate reproduction numbers and persists within each of five pathogens compared. Beyond the contrast between parameters, the model-based share of papers linked to incubation-period estimates differs by pathogen: 55% for SARS-CoV-2 versus 6% for mpox. 

---
# HoloCosm: Beyond Single-Metric Benchmark Compression Across Multiple Metrics 

**Authors**: Yupei Lin, Zhongzhan Huang, Junxin Li, Guoming Ling, Shanshan Zhong, Hefeng Wu, Yukai Shi, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.26714)  

**Abstract**: Benchmark compression reduces evaluation cost by selecting a small set of samples. Existing methods typically guide selection using a single metric, although benchmarks often assess models along multiple dimensions. We find that model rankings are often weakly correlated across metrics, indicating that different metrics capture distinct aspects of model performance. Samples selected for one metric do not reliably preserve scores and rankings on others when used for compressed evaluation. At 80\% compression, each of the five single-metric baselines yields higher reconstruction error than equal-size random sampling in over 70\% of cross-metric comparisons. To address this challenge, we present HoloCosm, which jointly minimizes normalized score-reconstruction error across source models and metrics while constraining ranking distortion. We evaluate HoloCosm across multiple compression rates on 15 language, multimodal, VLA, and agent benchmark configurations with 85 component metrics. With 80\% compression, HoloCosm outperforms Random, the strongest baseline overall: average NMAE drops by 35.3\% to 0.236, worst-component NMAE drops by 39.5\% to 0.330, and average Spearman correlation rises from 0.884 to 0.915. In addition, HoloCosm remains competitive with existing baselines in single-metric evaluation. We will release our code and experimental results. 

---
# Hierarchical Token Merging with Recursive Autoencoders for Unsupervised Grammar Induction 

**Authors**: Murat Guney Kemal, Robert Lieck  

**Link**: [PDF](https://arxiv.org/pdf/2610.26695)  

**Abstract**: Hierarchical abstraction is a fundamental principle across many data modalities, enabling complex sequences to be represented compositionally at multiple levels. Learning such hierarchies directly from data, without explicit structural supervision, remains a central challenge in representation learning. We investigate whether meaningful hierarchical structure can emerge from compression alone. While autoencoders learn compressed representations through reconstruction, this objective does not necessarily favour meaningful compositional structure. We study this challenge through unsupervised grammar induction, where the underlying hierarchy can be evaluated explicitly. We introduce a hierarchical token-merging autoencoder that recursively combines adjacent tokens into increasingly abstract representations, compressing entire sequences through a variational information bottleneck. The model jointly learns representations and hierarchical structure by marginalising over possible binary trees using dynamic programming. To address training instability, we combine independently trained models through unsupervised consensus and distillation. Experiments on a synthetic, ambiguous grammar demonstrate the emergence of meaningful hierarchical structure without structural supervision. The resulting model achieves 93% grammatical parse-validity and 93% exact reconstruction on held-out sequences, compared with 42% parse-validity for the strongest trivial baseline. These findings establish a proof of concept for hierarchical token merging as a mechanism for unsupervised structure discovery and abstraction learning. 

---
# The Development of Sycophancy in Language Models: Tracing Vulnerabilities from Pretraining to Post-Training 

**Authors**: Haonan Huang, Chengkun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.26655)  

**Abstract**: Language models can change a committed answer to accommodate a user's opinion or pushback without new evidence. However, when this behaviour develops during training and what a released model inherits from earlier stages remain unclear. We follow 741 checkpoints of 30 open model lineages, from early pretraining to released models. We plant initial verdicts on moral scenarios and measure how far a model follows a user's stated opinion or abandons its verdict under pressure. In most lineages the behaviour is present before post-training, and its average strength changes in either direction, at different stages across lineages. A clearer pattern emerges at the level of individual questions. Beyond responses without pressure, pretraining behaviour predicts little of which questions a released model yields on. That pattern forms during post-training: on routes with supervised fine-tuning, the fine-tuning checkpoint predicts part of it and the preference-optimization checkpoint most of it. A sibling trained from the same base along another route predicts far less: for Tülu-3 8B, its own preference-stage checkpoint removes 85% of the remaining prediction error, a sibling 19%. Internal states add predictive information without changing this ranking. Similar average scores conceal different vulnerabilities, and training history matters for auditing a released model. 

---
# Conducting Reliable Keyword Estimation with Unreliable ASR in Police Audio Data 

**Authors**: Robert Cole Molloy, Rhiannon N. Miller, Stephen Morgan, Anjalie Field  

**Link**: [PDF](https://arxiv.org/pdf/2610.26614)  

**Abstract**: Extracting insights from large audio data sets using human transcription is expensive and time-consuming. Off-the-shelf Automatic Speech Recognition (ASR) systems offer an easy-to-use scalable alternative, but systematic transcription errors, especially in domain-specific settings, can bias downstream analyses and lead to unreliable conclusions. We investigate whether imperfect ASR transcripts can still be used to obtain statistically valid keyword frequency estimates by combining model outputs with a small set of human-labeled examples, following Prediction Powered Inference and Design-based Supervised Learning frameworks. We evaluate this approach on police radio communications from Baltimore and Chicago, a domain in which keyword distributions could provide valuable insights into police activity, but off-the-shelf ASR systems exhibit systematic errors on operationally important terminology. Using bootstrap-based confidence interval coverage as our evaluation metric, we find that our approach substantially increases the number of domain-specific keywords that can be analyzed with statistically valid confidence intervals, from 43.0% using ASR alone and 27.2% using only a small set of human-labeled transcriptions to 56.2% in the Baltimore dataset, with similar improvements observed in Chicago data. Our results demonstrate that post-hoc error correction can improve statistical validity of downstream analyses even for poor-quality transcriptions, offering an avenue for better leveraging off-the-shelf ASR models for large-scale speech analysis. 

---
# BERT Is Already a Denoiser: Cheaply Unlocking Iterative Generation in Masked Language Models 

**Authors**: Julio C. Amador Diaz Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2610.26601)  

**Abstract**: Masked language models (MLMs) such as BERT are typically used for representation learning despite their denoising pretraining. We test whether this capability supports conditional generation without diffusion training. Our decoder repeatedly predicts masked tokens and commits high-confidence predictions using a cosine schedule. It requires no diffusion timestep or reverse-diffusion objective. Frozen BERT already exhibits substantial denoising capability, reconstructing 61.0%, 37.5%, and 24.4% of masked tokens at 15%, 45%, and 60% corruption, respectively, although generation deteriorates sharply as visible context disappears. This capability alone does not yield task-directed generation: frozen BERT reaches only 11.0% on ARC-Easy and 4.2% on MMLU. Ordinary MLM adaptation, however, rapidly converts the same architecture into a usable generator. Full adaptation reaches 28.2% and 24.6%, respectively, while parameter-efficient adaptation substantially reduces unmatched outputs. On open-ended conditional generation, full MLM adaptation reaches BLEU/ROUGE-L of 0.163/0.583 on QQP and 0.107/0.327 on Quasar-T, placing it in the same broad performance range as published BERT models trained through diffusion without explicit diffusion training. Finally, applying the same generative interface to stronger masked representations substantially improves performance: RoBERTa-large reaches 44.2% on ARC-Easy and 33.8% on MMLU. These results separate three components of masked iterative generation: pretraining supplies the representation and denoising operator, task adaptation determines what the model generates, and an iterative commitment rule supplies the decoding process. Explicit diffusion training is therefore not necessary to obtain effective iterative conditional generation from pretrained MLMs in the settings studied here. 

---
# ReAnchor: Reusing Rejected Verifier States for Diffusion Drafting 

**Authors**: Sanghyun Lee, Wonjin Nam, Seungho Jang, Jaeho Lee, Jiho Park, Jaewoon Lee, Seungryong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.26531)  

**Abstract**: Block diffusion drafters accelerate speculative decoding by generating multiple draft tokens in a single forward pass. However, each pass produces only a fixed-length block from a single target-generated anchor, limiting how far the draft can extend. Meanwhile, target verification computes hidden states even at positions beyond the first rejection, but these states are typically discarded rather than reused. We propose ReAnchor, a training-free method that reanchors a pretrained drafter at shifted positions using discarded verifier states to generate parallel draft blocks. These blocks improve predictions within the original block and extend coverage beyond its endpoint. At sampling temperature of T=0, ReAnchor achieves its largest average gains on Gemma 4-12B, improving acceptance length by 19.4% and decoding speed by 17.9% across eight benchmarks. Averaged equally over four target models and eight benchmarks, it improves acceptance length by 15.2% and decoding speed by 13.4% with pretrained DFlash drafters. 

---
# Mitigating Reward Hacking in Preference Reward Models with Evolving Penalties 

**Authors**: Afonso Baldo, António Farinhas, Marcos Treviso, Ricardo Rei, Nuno M. Guerreiro  

**Link**: [PDF](https://arxiv.org/pdf/2610.26496)  

**Abstract**: Reinforcement learning for language models has moved from preference reward models to verifiable rewards, which sustain much longer optimization. Preference models are prone to reward hacking: the policy raises their score without improving response quality, or even while degrading it. Yet for open-ended tasks, where quality can neither be verified nor fully written down as rubrics, they remain valuable. We introduce Dynalize, which mitigates reward hacking online, during policy training. Periodically and asynchronously, a language model analyzes training rollouts and writes penalties for undesirable behaviors that the reward model appears to reward; a judge applies them to every rollout. In clinical counseling and creative writing, training on the reward model alone raises its score continuously but ultimately leads to a decline in external benchmarks. Dynalize sustains longer training and results in improved downstream performance. For example, we find that our method suppresses patterns incentivized by preference models that, without penalties, spread to most responses, and even to the model's reasoning, which the reward model does not explicitly score. Furthermore, we show that its gains hold in multi-turn training, and even after we retrain the reward model on the discovered behaviors. 

---
# AfriGuard: Safety and Robust Refusal For African Languages 

**Authors**: Israel Abebe Azime, Jesujoba O. Alabi, Iffat Maab, Alexander D. Cai, Tadesse Destaw Belay, Atnafu Lambebo Tonja, Jonas Kgomo, Folasade Alabi, Salomey Osei, Saminu Muhammad Aliyu, Nkechinyere Faith Aguobi, Bontu Fufa Balcha, Blessing Sibanda, Davis David, Mouhamadane Mboup, Daud Abolade, Neo Putini, Louis Powell, Kiant'e Brantley, Philipp Slusallek, Dietrich Klakow, Julia Kreutzer  

**Link**: [PDF](https://arxiv.org/pdf/2610.26474)  

**Abstract**: Large language models (LLMs) designed to interact with users are typically trained to respond appropriately to harmful or toxic content. However, ensuring appropriate responses becomes more challenging in diverse linguistic and cultural contexts, where safety risks may be expressed or interpreted differently. Building safe AI systems therefore requires detecting safety risks, understanding their nature, and responding appropriately, while accounting for linguistic and cultural variations that add complexity to each stage. In this work, we introduce AfriGuard, a culturally grounded multilingual dataset for studying harmful intent, safety-category understanding, and refusal behavior across 10 African languages and English. Our findings reveal that guardrail models underperform both open- and closed-source LLMs on safety-intent and safety-category classification, while models show limited understanding of appropriate safety behavior, scoring below 60% F1. We further find that models often fail to respond appropriately to unsafe prompts not because they explicitly refuse, but because they produce linguistically inadequate responses that result in incidental non-assistance rather than proper refusals. Finally, using AfriGuard, we develop efficient guardrail models that achieve performance competitive with models up to 30x larger, introduce refusal detectors that improve average performance by 43.9% over existing detectors, and demonstrate that supervised fine-tuning can substantially improve overall safety performance. 

---
# Training Negatives Still Matter After Calibrating Answerability Readouts 

**Authors**: Jinchen Zhang, Zheyuan Liu, Yihan Zhu, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26461)  

**Abstract**: Answerability readouts predict whether a passage supplies an answer, but whether calibration can compensate for removing negative passages during training remains unclear. We present a controlled study of how training-negative construction affects calibrated prediction, ranking, and downstream answer retention. We compare native training, which retains negative passages, with empty training, which removes them, holding questions, labels, positive examples, and model weights fixed. Both are calibrated on the same labeled native examples. On SQuAD 2.0 and RepLiQA, native-trained logistic readouts at selected depths achieve lower Brier loss for Qwen2.5-7B-Instruct and Gemma-2-9B-it, with or without output scores. Native fusion also outperforms calibrated output alone. Post-hoc AUROC analysis shows better ranking under native training, beyond what threshold adjustment alone can explain. On the official SQuAD 2.0 development set, native readout policies improve F1 over empty policies but retain more answers to unanswerable questions. The advantage varies by estimator and depth, while RepLiQA's text shortcuts limit claims about contextual understanding. Unfinished generations prevent complete answer-level evaluation, although worst-case bounds preserve the native-empty F1 gains. Our findings show that training negatives affect ranking and downstream decisions even after common calibration, without establishing a universal advantage for native training. 

---
# TeleASR-Amharic: A Benchmark for Robust Amharic ASR in Realistic Telecommunication Conditions 

**Authors**: Dagnachew Mekonnen Marilign, Enrique Molero, Abrham Haile, Louis Powell, Israel Abebe Azime  

**Link**: [PDF](https://arxiv.org/pdf/2610.26448)  

**Abstract**: The real-world usability of ASR models remains challenging, particularly for low-resource languages, where existing speech resources are often collected in clean studio environments. In this work, we introduce TeleASR-Amharic, a benchmark for evaluating Amharic ASR under various acoustic and telecommunication conditions. We analyze the major degradations observed in real telecommunications recordings and develop a benchmark that covers multiple channel conditions, environmental disturbances, and telecom-domain variations. Our results show that telecommunication degradation can substantially reduce recognition performance, making even high-performing ASR models impractical under severe conditions. For example, the WER of the best-performing model increases from 0.23 to 0.79 under severe telecommunication degradation, corresponding to a relative increase of 243% in recognition errors. We further investigate the effects of different noise and channel distortions and show that scaling training data and applying carefully designed telecommunication-aware augmentation substantially improve robustness. In particular, our best training configuration achieves a 23.6% relative WER reduction under extreme noise conditions. Together with the release of our telecommunications-domain data, these findings provide a reproducible benchmark for measuring progress in Amharic ASR while highlighting open challenges in developing robust speech technologies for real-world deployment. 

---
# WenSight-Decision-1 for Direct Decisions under E-commerce Rules 

**Authors**: Mingwen Liu, Jun Sun, Sichen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26444)  

**Abstract**: E-commerce decisions require applying natural-language rules to the available case evidence. We introduce WenSight-Decision-1, a shared model that scores supplied options directly from rules and case information for six operational tasks, with examples spanning six languages. We fine-tune Qwen3.5-9B with reference options as supervision, using the same option scores for training and inference. On 1,440 synthetic operational cases reserved for final evaluation, accuracy rises from 61.60% for Qwen3.5-9B to 92.92% for WenSight-Decision-1 under identical inputs and scoring. Joint fine-tuning improves all six tasks, bringing heterogeneous operational judgments under a common decision interface. 

---
# The Geometry of Agent Memory: Forecasting Retrieval-Bound Memory Failures, and Characterizing Its Limits 

**Authors**: Iliya Zhechev, Radoslav Ralev, Chris Coleman, Srijith Rajamohan, Jen Agarwal, Tyler Hutcherson, Simba Khadder  

**Link**: [PDF](https://arxiv.org/pdf/2610.26433)  

**Abstract**: We study whether geometric relationships between conversation turns, memory texts, and queries can predict answer correctness before generation. We evaluate four memory implementations with dense or hybrid retrieval and one observation-summary comparator. We embed these texts in a shared space. A logistic classifier uses 25 features describing query proximity, memory-conversation correspondence, and conversation structure. On LongMemEval-small, pooled AUROC is 0.640 over 2,367 question-system pairs, below a question-difficulty baseline of 0.655. Performance varies by question type: geometry reaches 0.841 on single-session-assistant questions, but system identity alone reaches 0.826. After controlling for difficulty and identity, geometry adds no statistically established pooled improvement (delta = +0.002, 95% CI [-0.010, 0.014]). A fixed-store context-selection experiment finds higher accuracy with proximity-ranked memories than with random subsets, consistent with ordinary retrieval relevance. On LoCoMo, nearest-turn similarity has modest predictive value for single-hop questions and near-chance estimates for several other categories. Training uses correctness labels; prediction uses pre-answer features and no additional LLM generation call. The results characterize associations between memory representations and correctness, rather than establish a reliable deployment diagnostic. In particular, embedding proximity does not verify fact coverage or distinguish memory-construction, retrieval, and generation errors. 

---
# HarnessRL: Co-Evolving Policies with a Just-in-Time Training Harness for Open-Ended RL 

**Authors**: Xiaoqiang Wang, Ting-En Lin, Yuchuan Wu, Pinyi Zhang, Caixin Kang, Ling Li, Bingxue Qiu, Jieping Ye, Yongbin Li, Bang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26426)  

**Abstract**: Rubric-based reinforcement learning (RL) trains policies for open-ended generation with natural-language criteria as rewards, but these rewards can become ineffective as training progresses. We identify two failure modes. Under non-discriminative rewards, scores collapse to uniformly low or high values due to insufficient exploration or a mismatch between task difficulty and policy capability. Under spurious rewards, higher scores do not correspond to better responses. Both reflect a training configuration whose task sampling, exploration, and evaluation stay fixed as the policy evolves. We introduce HarnessRL, a training-time harness that co-evolves with the policy through three interfaces for task sampling, rollout guidance, and rubric criteria. HarnessRL maintains a training-time memory from which a harness evolver retrieves within-task history and cross-task evidence. Using this evidence, the evolver attributes each reward failure to the interface whose update can address it. It then runs a proposer and just-in-time critic loop in which the critic accepts a candidate update only after measuring its effect on the current frozen policy, returning rejected candidates for revision. We also curate Rubric-Mix-10k, a five-domain training and evaluation suite for open-ended RL across medical, science, writing, role-play, and instruction-following tasks. On Rubric-Mix-10k, HarnessRL outperforms the strongest baselines by 2.8 to 3.1 points across both 4B and 9B model scales and nearly halves the training steps to its best checkpoint. Code and data are available at this https URL. 

---
# SNO-PI: Label-Scarce SNOMED CT Entity Linking with Pre-trained Encoders and Curated Terminology Knowledge 

**Authors**: Piyawoot Songsiritat  

**Link**: [PDF](https://arxiv.org/pdf/2610.26425)  

**Abstract**: We present SNO-PI, the top-performing submission on SNOMED International's SNOMED CT Entity Linking Benchmark as of 18 October 2026. SNO-PI achieves a macro character IoU of 0.5387, outperforming the organisers' reference system (0.4582; based on the 2024 challenge winner's approach) by 18% relative. Linking clinical mentions to SNOMED CT is a zero- and few-shot problem over a very large label space: the 272 annotated MIMIC-IV discharge summaries cover only 6,595 of the roughly 219,000 eligible concepts, and a third of those occur only once. This motivates a design built on transfer from pre-trained models and human-curated knowledge. SNO-PI combines high-recall span proposal, SapBERT candidate retrieval and listwise ensemble reranking with three BERT-family encoders, each further pre-trained with masked language modelling on unlabelled MIMIC-IV notes and then on SNOMED CT synonyms and hierarchy before task fine-tuning. We find that curated knowledge helps most on concepts never seen in training, yet these concepts remain the main source of error. 

---
# Governing the Rule Space. Ruliology, Computational Irreducibility and Anticipatory Governance of Quantum Technologies 

**Authors**: Antonio Felice Uricchio, Francesco Scagliola  

**Link**: [PDF](https://arxiv.org/pdf/2610.26423)  

**Abstract**: The regulation of emerging technologies is often conceived as a strategy of catching up, because legislators arrive too late. Yet gathering signals, constructing scenarios and adapting rules to an anticipated future are fundamental tasks for those entrusted with governance, including the governance of rules themselves. The literature on so-called future law has already shown that anticipation means building institutional capabilities rather than predicting outcomes precisely (anticipatory governance). Taking this premise as its starting point, this article proposes to understand regulation as an exploration of the transformations that may occur. The Ruliological Framework for Anticipatory Regulation distinguishes rules, transformations and trajectories, assessing alternatives and possible developments. Within this approach, computational irreducibility is adopted conditionally, not as a demonstrated property of legal systems. The Quantum Europe Strategy provides a case study and further grounds for reflection. 

---
# Self-Consistency of Entity Sets and Citations in a Deep Research System under Repeated Prompting 

**Authors**: Lauri Pykälä, Kimmo K. Kaski  

**Link**: [PDF](https://arxiv.org/pdf/2610.26421)  

**Abstract**: Deep research agents (DRA) are commonly used for open-ended synthesis tasks where closed-world QA benchmarks are poorly suited to characterize relevant aspects of performance. We study the self-consistency of one commercial deep research system under repeated identical prompting, representing each report with the set of named entities it mentions, categorised by a custom schema, and the set of citations it retrieves. Across repeated runs of a single open-ended cybersecurity related prompt, we quantify pairwise overlap to characterize how stable the retrieved and synthesized output structure is under repeated identical prompts. In ten runs, the pairwise entity Jaccard results in a median of 0.12 and the pairwise citation Jaccard is below 0.02, with discovery curves that do not visibly saturate. The magnitude and structure of (dis)agreement between entity sets varies by entity class, which suggests that self-consistency in a DRA is better understood as a function of the underlying entity space than as a stable property of the system. 

---
# Co-Evolving Policy and Reward for Code Generation: A Mechanistic Study of Single-Adapter Drift 

**Authors**: Amarsaikhan Batjargal, Noriko Tomuro  

**Link**: [PDF](https://arxiv.org/pdf/2610.26416)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is particularly well suited to code generation because generated programs can be objectively evaluated through execution. Execution also provides behavioral information that can be reused as additional supervision. The Synergistic Policy and Reward Co-Evolving Framework (SPARK), originally proposed for reasoning tasks, exploits this idea by recycling training rollouts and their evaluation results to train a model not only to generate solutions but also to evaluate and improve them. We investigate whether this policy-reward co-evolution approach transfers effectively to code generation and whether execution-grounded reflection provides an additional source of self-improvement. We compare SPARK-style rollout recycling with execution-only Group Relative Policy Optimization (GRPO) using Qwen2.5-Coder-3B-Instruct. Our results do not support either hypothesis under the configuration tested. Although co-evolution improves training performance, these gains do not generalize to held-out problems. Compared at the same stage of training, rollout recycling produces roughly 34x greater KL divergence from the reference policy than execution-only GRPO. Over training, trained tokens per GRPO sequence fall from 185 to 86 as training pass rate increases, while held-out performance declines or fails to sustain early gains. We characterize these patterns as single-adapter policy drift: the shared trainable parameters increasingly depart from pretrained code-generation behavior without corresponding gains in generalization. The results show that higher training reward alone is not evidence of generalizable code self-improvement and highlight the importance of held-out evaluation and drift diagnostics. 

---
# Expert L2 English Speech Evaluation on a Gaming PC: Accuracy, Fluency, and Prosody with XLS-R, XEUS, and PhoneticXeus 

**Authors**: Richard Rose  

**Link**: [PDF](https://arxiv.org/pdf/2610.26406)  

**Abstract**: oollim is a locally executable system for evaluating L2 English pronunciation accuracy, fluency, and prosody on a consumer computer. The system combines pretrained XLS-R, XEUS, and PhoneticXeus representations with reference-text features and ridge regressions trained on SpeechOcean762, the Korean Learners' L2 Speech Corpus (English), J-AESOP, and their combinations. On 305 independently held-out recordings assessed by three expert raters, several ridge configurations substantially outperform archived Azure API scores obtained in January 2026 in both Spearman correlation and quadratic weighted kappa across all three constructs. These results support expert speech evaluation on consumer hardware using pretrained representations and construct-specific regressions, without neural-model fine-tuning or calibration on the private evaluation data. 

---
# How Benchmark Graph Construction Shapes Multi-Hop KG Navigation: A Controlled Audit of PathQuestion and PathQuestion-Large 

**Authors**: Eduin E. Hernandez, Sergio A. Diaz, Luis F. Garcia, Nurassyl Askar, Stefano Rini  

**Link**: [PDF](https://arxiv.org/pdf/2610.26399)  

**Abstract**: Multi-hop knowledge-graph question answering (KGQA) benchmarks are often treated as fixed language-and-reasoning tasks, but for navigation-based methods, difficulty also depends on the graph in which those tasks are instantiated. This makes graph construction a consequential but often entangled part of the navigation task. To isolate this effect, we audit PathQuestion (PQ) and PathQuestion-Large (PQL) as graph-navigation benchmarks under a controlled intervention: we hold questions, topic entities, released answers, released path annotations, and family-level splits fixed while replacing each released task-filtered graph with the larger source KG distributed with the benchmark. Source restoration expands the effective navigation search space and graph-consistent answer and path multiplicity while sharply reducing unguided reachability to released answers. We evaluate whether these structural changes persist across navigation policies using structural references, frozen small-language-model navigation policies, and trained MINERVA policies. Across both frozen and trained policies, restoration substantially reduces released-answer accuracy and trajectory fidelity, while graph-consistent evaluation recovers only part of the loss. These results show that benchmark graph construction is part of the navigation task specification rather than merely an implementation detail and can materially shape measured behavior even when the released QA task is held fixed. 

---
# Bridged Self-Distillation for On-Policy Learning 

**Authors**: Runsheng Chen, Yihong Liu, Raoyuan Zhao, Michael A. Hedderich, Hinrich Schütze, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2610.26397)  

**Abstract**: On-policy distillation can transfer the reasoning capabilities of a strong teacher to a smaller student. However, distillation from a stronger teacher model faces two mismatches: its solutions may be difficult for a smaller student to imitate, and its token-level probabilities may be incompatible with, or even unavailable to, the student. Existing methods handle these settings through different forms of supervision. To this end, we propose Bridged Self-Distillation (BSD), a unified framework for distilling from any teacher that provides reasoning traces. The model to be distilled from, termed the external teacher, first provides a reasoning trace as privileged context, which can be generated in an offline fashion. The student generates rollouts without this context, while a copy of the student conditioned on it, serving as the self-teacher, scores the same rollouts in the student's own token space. BSD uses the resulting token-level guidance to adjust updates on unsuccessful rollouts, with a verifier providing the outcome signal. As the external teacher is queried only to produce text, the same training procedure applies to all teacher scenarios: shared-tokenizer, cross-tokenizer, and closed API teachers. Across these scenarios, BSD improves over the base model and outperforms baselines tailored to each scenario, offering a novel and unified pathway for distillation. Code is available: this https URL 

---
# Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs 

**Authors**: Adrián Sánchez-Freire, Gonzalo Pérez-Gómez, Aitziber Atutxa, Ander Barrena  

**Link**: [PDF](https://arxiv.org/pdf/2610.26339)  

**Abstract**: We present a controlled, large-scale evaluation of how corpus scale, document representation, and reranking affect the effectiveness and efficiency of biomedical Retrieval-Augmented Generation (RAG) systems. Retrieval collections were progressively expanded while retaining the relevant source documents, enabling consistent comparisons across corpus sizes. We compared abstract, flat full-text, and hierarchical full-text representations across first-stage retrieval, reranking, and end-to-end multiple-choice question answering. Configurations selected on development data were evaluated on five external biomedical benchmarks and a held-out test set, while retrieval latency and index storage were measured to quantify operational cost. Performance patterns observed with small or controlled collections did not consistently persist as collections grew, and the usefulness of reranking depended on the corpus and retrieval setting. RAG improved three of four model configurations across the external benchmarks, with mean gains of up to 9.6 percentage points, and improved all four configurations on the held-out test set by 10.2-30.3 points. Abstract retrieval performed best when reference evidence was abstract-aligned, whereas full-text representations were generally more effective when richer evidence was needed. Hierarchical representation matched or outperformed flat full-text retrieval in most end-to-end comparisons while requiring 67% fewer indexed units and 44-56% less storage. These findings show that corpus scale should be treated as an explicit evaluation variable in biomedical RAG, and that the most appropriate design depends on evidence format, model behavior, and operational constraints. 

---
# Answering While You Type: Optimizing Time-to-Answer with Streaming Generation 

**Authors**: Nikita Dragunov, Matvey Mikhalchuk, Anton Razzhigaev  

**Link**: [PDF](https://arxiv.org/pdf/2610.26296)  

**Abstract**: Work on efficient large-language-model inference has concentrated almost entirely on reducing the cost of decoding once a request is complete. The quantity a user experiences, however, is the interval between submitting a request and receiving a usable response, and that interval can be shortened without making decoding itself faster. While a request is being composed it is already partially specified, yet conventional deployments begin no computation until submission. We propose and study streaming generation, a decoding regime in which the model begins producing a response from an incomplete request and revises it as the remaining input arrives, so that only a small residue of the answer remains to be computed once the request is submitted. We evaluate the regime on autoregressive and diffusion language models, characterise the conditions under which it reduces user-visible latency, and analyse the mechanism responsible. 

---
# Shaping the Document, Not the Model: Optimized Suffixes for Context Following 

**Authors**: Gia-Bao Dinh Ho, Sekh Mainul Islam, Masahiro Kaneko, Timothy Baldwin, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2610.26236)  

**Abstract**: When a user supplies a document and wants the answer drawn from it, a language model is expected to follow it even when it contradicts its parametric knowledge. Prior work has focused on making the model rely on context by steering activations, contrasting logits during decoding, or instructing it in plain language. We ask whether the document alone, with no instruction written into it and no access to the model at inference time, is enough. We optimize a document suffix by discrete token search, starting from the answer to a single question and extending the objective to a set of passages, to multi-hop questions that follow from the planted fact, and to later turns of a conversation. On counterfactual passages that contradict parametric knowledge of the model, the suffix flips nearly all items the model resists across three instruction-tuned model families, in most settings outperforms the decoding baseline and the strongest instruction we could write, and matches activation steering. Controls show that the suffix largely acts through the passage rather than injecting the answer, and a single suffix per model transfers to unseen passages. Each extended objective carries the effect to its target, so the objective, not only the document, sets how far the effect reaches. 

---
# Heartbeat Thinking: Speculative Reasoning and Tool Execution While Listening in Realtime Voice Agents 

**Authors**: Zhenyu Zhang, Kewei Fu, Yongqi Tong, Chenzhi Zheng, Wei Wu, Jiang-Ming Yang, Jianshe Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.26233)  

**Abstract**: Tool-using voice agents often leave users waiting after speech ends, even when actionable information arrives earlier in the request. Speculative approaches begin work during speech, but chunk-bounded or interrupt-on-update generation can truncate a tool call before its arguments are complete. This couples computation time to speech updates, limiting useful work completed in advance. We introduce Heartbeat Thinking, which handles updated input through new Thinker invocations while allowing ongoing invocations and dispatched tools to finish. Periodic invocations read the latest partial transcript and a shared record of completed reasoning and tool results. These results help resolve missing arguments and initiate dependent calls, allowing preparation to progress across speech updates. Across SpeechFCEval, FDBench v3, and AudioCRAG, Heartbeat improves aggregate speculative tool coverage or retrieval reuse over a SHANKS-style windowed baseline, with benchmark-dependent effects on final task quality. On FDBench v3, it increases tool coverage by 16 percentage points and reduces mean final-response latency by 36.4\%. The results show that heartbeat thinking paradigm preserving ongoing computation during speech can turn listening time into reusable preparation for faster and more capable voice-agent responses. 

---
# Rethinking BPE: Merge Selection as Model Selection 

**Authors**: Vijini Liyanage, François Yvon  

**Link**: [PDF](https://arxiv.org/pdf/2610.26225)  

**Abstract**: Subword tokenization methods such as Byte-Pair Encoding (BPE) are a core component of modern language models; yet, tokenizer capacity remains typically determined by a fixed merge budget, chosen heuristically. Since larger vocabularies increase memory usage and computational cost, selecting compact and effective token inventories is essential for efficient language modeling. We study the optimization of a BPE merge budget as a statistical model selection problem and investigate information-theoretic criteria for \emph{computing optimal merge sets} directly from corpus statistics. To further improve the merge selection in limited data conditions, we augment the BPE training corpora with synthetic texts generated from the original data, providing estimates for the unigram and bigram distributions. Experiments across multiple corpora and downstream tasks show that using a principled vocabulary selection technique reduces model size, with little degradation in performance. We further observe that synthetic data can yield better tokenizers, especially in low-resource settings. 

---
# VEILED-SH: Vulnerability Expression through Indirect Language Evaluation Dataset (Self-Harm) 

**Authors**: Leanne Tan, Keith Chua, Shaun Khoo, Onno P. Kampman, Lim Ruling Charmaine, Hazirah Hoosainsah, Kellie Yu Hui Sim, Kai He, Kenny Tsu Wei Choo, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.26190)  

**Abstract**: AI systems may miss indirect, culturally situated distress or provide support mismatched to the application context. Existing benchmarks do not jointly address cultural variation, application context, and practitioner disagreement. We introduce VEILED-SH, a benchmark and construction methodology for evaluating single-turn responses to indirect expressions of suicide, self-harm, and distress. Its 10,003 prompts span counselor chat, forum posts, platform search, and triage, drawing on Singapore-grounded sources and public datasets. An expert-validated multi-label taxonomy distinguishes intent, communicative function, and response behavior. Six automated profiles, individually calibrated to three peer-support practitioners and three clinicians, retain separate reference policies for proxy-alignment scoring. Evaluating five systems on a coverage-enriched sample reveals variation in proxy alignment across profile-generated reference policies, even for the same system responses. 

---
# Reading Time Modeling with Pixel-Based Language Models 

**Authors**: Junlin Li, Sara Alhajeri, Tatsuki Kuribayashi  

**Link**: [PDF](https://arxiv.org/pdf/2610.26181)  

**Abstract**: Tokenization-free, pixel-based language models (PLMs) have recently been proposed as an alternative to conventional token-based language models (TLMs), typically motivated by practical considerations, e.g., the flexibility to support new languages or vocabularies. Nevertheless, PLMs also have the potential to serve as infrastructure for computational (psycho)linguistics, where tokenization has been a source of bias in estimating information-theoretic values, such as surprisal, and has pushed the bottom-up visual orthographic features outside the scope. In this study, we first integrate PLMs into the modeling of naturalistic human reading and, in particular, examine their use to estimate next-word predictability, which can also integrate orthographic and parafoveal information. Our results demonstrate that next-word prediction difficulty, computed in pixel space, has strong predictive power for gaze duration and word skipping behavior across various languages. In particular, their advantage is complementary to existing token-level surprisal and entropy estimates. Cross-lingual analyses with script variations also clarify their distinctive visual processing biases, facilitating research into how bottom-up visual information is incrementally integrated into expectation-based reading. 

---
# Conditional information Retrieval for Longitudinal Survey Harmonisation 

**Authors**: Wing Yan Li, Zeqiang Wang, Michelle Lin, Jon Johnson, Suparna De  

**Link**: [PDF](https://arxiv.org/pdf/2610.26173)  

**Abstract**: Identifying conceptually equivalent questions across different surveys is a challenging information retrieval task where surface-level textual similarity is insufficient. Two questions may be lexically similar yet assess entirely different sociological sub-concepts, or lexically different yet conceptually equivalent. We introduce a conditional information retrieval framework that addresses this by decomposing the task into two steps: concept extraction, which identifies the underlying sociological concept of a survey question using a fine-tuned generative model, and concept-aware retrieval, which augments question representations with the extracted concept as an explicit conditioning signal for finer-grained alignment. To reflect the realistic data scarcity conditions of specialised social science annotation, we use CLOSER as a case study. Our framework consistently outperforms baselines by 2-3 points in both MAP and Recall. We further analyse fine-tuning strategies including SFT, DPO, and dynamic DPO, finding that SFT delivers the most stable gains while dynamic DPO interacts differently with instruction-tuned versus distilled reasoning backbones. Finally, we demonstrate that a model fine-tuned on a general social science thesaurus (ELSST) transfers robustly to task-specific survey data, offering a practical solution under data scarcity. 

---
# VISAGENT: An Agent-Based End-to-End Voice Interaction System 

**Authors**: Xiang Wang, Zhaoxiang Liu, Xiang Chen, Liqiang He, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2610.26168)  

**Abstract**: Voice-based public services have evolved from manual switchboards to intelligent platforms to meet modern user needs. However, current systems remain constrained by rigid menu-driven interaction, fragmented vertical services, and limited extensibility. To address these issues, we present VISAGENT, an agent-based end-to-end voice interaction system for large-scale public services that integrates end-to-end speech interaction, hierarchical agent routing, a scalable voice-agent marketplace, and user memory support. The system replaces menu-driven workflows with natural speech and dynamically dispatches user requests to either a general conversational model or specialized domain agents via a two-level routing paradigm that achieves sub-second intent-to-agent selection over hundreds of agents. User memory informs preference-aware agent selection and supplies relevant preferences and context to selected agents, supporting personalized task execution and reducing the need for users to repeat previously provided information. A voice-agent registration mechanism converts task agents into conversational entities, enabling dynamic onboarding and ecosystem-level extensibility. We validate VISAGENT through a real-world deployment in China's 114 hotline across multiple service domains. Experimental results demonstrate low-latency full-duplex interaction, robust routing scalability, and competitive speech naturalness. More broadly, this work advocates a shift from isolated voice assistants toward agent-centric conversational infrastructures, providing a general blueprint for building extensible voice-agent systems in public services and beyond. 

---
# DeepSEA: A Multilingual Evaluation Benchmark for Southeast Asia 

**Authors**: Jiuzhou Han, Abhishek Bijay Mishra, Trevor Cohn, Shubham Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2610.26155)  

**Abstract**: While language models have demonstrated remarkable capabilities, their performance remains disproportionately skewed towards high-resource languages. This disparity is particularly acute for the 700 million inhabitants of Southeast Asia, where vast linguistic diversity remains under-represented in existing benchmarks, often suffering from limited language coverage or narrow task scopes. To address this, we introduce DeepSEA, a comprehensive benchmark constructed from established datasets through professional human translation into 12 Southeast Asian languages, spanning both high-resource (e.g., Indonesian, Thai, Vietnamese) and low-resource (e.g., Burmese, Khmer, Lao) tiers across five distinct tasks: mathematical reasoning, scientific reasoning, question answering, abstractive summarization, and machine translation. Extensive evaluation of frontier proprietary and open-weights models reveals that while frontier models like Gemini-3.7-Flash maintain robustness across resource tiers, smaller models face severe performance degradation on low-resource languages. Crucially, models perform better in reasoning and summarization when pivoted through English rather than target languages, exposing a critical dependency on English representations, especially for low-resource languages. Beyond these observations, we provide a cross-lingual error analysis and a cross-script tokenizer fertility analysis. We hope these findings shed light on the development of next-generation inclusive multilingual models. 

---
# Honeyed Words, Hidden Daggers: Benchmarking and Training LLMs to See Through Mock Politeness 

**Authors**: Caixin Kang, Sitong Gong, Tianyu Yan, Kaipeng Zhang, Yoichi Sato, Yifei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26059)  

**Abstract**: The Chinese idiom "kou mi fu jian" ("honeyed words, hidden daggers") names speech that is polite in form but hostile in function. We present ZhenhuanBench, a 1,592-item benchmark built from expert commentary on Empresses in the Palace, covering speaker intent, emotion-language mismatch, and per-listener uptake. Text-only evaluation covers 34 LLMs. GPT-6 Astra leads the leaderboard at 73.48% overall, while its listener-reaction score is 58.37%. The annotations also supply supervised traces and string-verifiable training rewards. With 338 gold traces, an 8B model outperforms teacher distillation on identical items. Subsequent RL saturates, and chance correction reduces the estimated recoverable performance gap by a factor of 7.5. Our CPR-GRPO loop (Compress-Probe-Replenish) uses this estimate to weight gold supervision, reaching 67.0 on public multiple-choice tasks and 74.5 under the training-matched hidden protocol. Without target-task training, CPR outperforms Base, SFT, and GRPO across six Chinese and English pragmatic and sarcasm evaluations. On three external datasets, training with CPR yields higher mean Macro-F1 than SFT and GRPO. 

---
# Balance-Aware Pivot Selection for Parallel Diffusion LLM Decoding 

**Authors**: Yejin Lee, Hyundong Jin, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.26040)  

**Abstract**: Diffusion language models enable parallel token generation, offering a promising alternative to sequential autoregressive decoding. Existing acceleration methods use predictive confidence to enable parallel token commitment while preserving generation quality. However, confidence-based selection prioritizes reliable token predictions without explicitly considering how token positions shape the remaining decoding schedule. Moreover, such ordering does not always reflect inter-token dependencies, suggesting that confidence need not be the sole criterion for token selection. Motivated by this limitation, we propose Balance-Aware Pivot Decoding (BPD), which favors central pivots within contiguous masked intervals and decodes the selected positions in parallel. Balanced pivots yield logarithmic decoding depth in the interval length. Our practical method accounts for model uncertainty by combining confidence-based prefilling with confidence-thresholded center selection. We evaluate the resulting quality-efficiency tradeoff on LLaDA and Dream across five mathematical reasoning, code generation, and instruction-following benchmarks. Our implementation is available at: this https URL 

---
# Accelerated Drift of Scientific Prose in the LLM Era: Evidence from an Interrupted Time-Series Analysis 

**Authors**: Hanna Hubarava, Benjamin V. Ineichen  

**Link**: [PDF](https://arxiv.org/pdf/2610.26008)  

**Abstract**: Every scientific text is embedded in the writing norms of its time: it reflects earlier publications while also shaping the texts that follow. The growing use of LLM-based writing tools introduces a new factor in this evolving system, but the linguistic changes coinciding with their emergence remain insufficiently understood. We investigate whether the trajectories of scientific-preprint writing changed in rate or direction after conversational LLMs became publicly accessible, using the release of ChatGPT in November 2022 as the observational breakpoint. We compare pre- and post-2022 trajectories across vocabulary, punctuation, readability, and syntax in abstracts from arXiv's artificial intelligence, statistics, and quantitative biology categories and from medRxiv. Using interrupted time-series regression, we find substantial post-2022 changes across multiple feature families and research domains, including increases in average word length, reduced formula-based readability, and declines in maximum dependency-tree depth and sentence-level depth variation. Individual features follow different dynamics: some lexical markers accelerate persistently, while others rise and then recede. Our findings show that isolated marker frequencies capture only part of the shift: changes in the rate, direction, and structure of linguistic trajectories provide complementary evidence of broader post-2022 change in scientific-preprint writing. More broadly, these findings motivate monitoring scientific writing as a multidimensional, evolving system rather than through a fixed set of LLM-associated markers. 

---
# Scaling Synthetic Pre-training Data and Evaluation for Indic LLMs 

**Authors**: Abhinav Khattar, Aishwarya Chhabra, Ameya Velingker, Anusha Kamath, Brandon Norick, Kanishk Singla, Kiran Praveen, Markus Kliegl, Mohammad Qazim Bhat, Mostofa Patwary, Niket Agarwal, Rajpreet Thethy, Ramani Akkati, Raviraj Joshi, Sai Nishanth Padala, Shuoyang Ding, Sriharsha Niverty, Utkarsh Vaidya, Vitaly Lavrukhin, Yev Meyer, Anoop Kunchukuttan, Mitesh Khapra, Mohammed Safi Ur Rahman Khan, Rudra Murthy, Sumanth Doddapaneni, Thanmay Jayakumar, Ujjwal Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2610.26006)  

**Abstract**: Building capable language models for India requires scalable multilingual data and robust evaluation. We introduce Nemotron-Pretraining-Indic-v1, a synthetic pretraining corpus of approximately 2.83 trillion tokens covering all 22 scheduled Indian languages in native script and twelve in romanized form. It is the largest openly released Indic pretraining corpus. Using our GPU-accelerated pipeline, we translate curated English documents, assess translation quality, and repair low-quality outputs while preserving technical content and document structure and maintaining source traceability. Through continued-pretraining experiments across twelve languages, we show that combining our synthetic corpus with natural Indic text improves knowledge and reasoning over a public-corpus baseline. Additional experiments show how data volume, script coverage, and Indic training share affect English performance. We also introduce the Indic-LLM-Eval-Suite, a multilingual benchmark suite for evaluating Indic language models across chat, instruction following, reasoning, knowledge, code, and translation. The suite combines existing Indic benchmarks with task-aware translations of English benchmarks, using automated quality assessment, targeted correction, and native-speaker review in a human-in-the-loop pipeline. We use the suite to comprehensively benchmark contemporary open-weight Indic and multilingual models across capabilities and languages, revealing substantial variation across tasks and languages. Together, these resources provide a scalable foundation for developing and evaluating more capable Indic language models. 

---
# TempoMoE: Identifying and Steering Fast-Slow Thinking Experts for Efficient Multimodal Reasoning 

**Authors**: Xiyue Zhang, Keru Zhou, Ziyi Wang, Bing Hu, Li Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25999)  

**Abstract**: Multimodal Mixture-of-Experts (MoE) reasoning models rely on long chain-of-thought (CoT) for complex tasks, yet often overthink easy inputs, increasing token cost and latency, and sometimes degrading accuracy. Existing methods optimize reasoning length during training or impose external control at inference, but leave the underlying reasoning mechanism a black box. We find that fast and slow thinking exhibit distinct expert-routing patterns: holding the input fixed and varying only the prompted reasoning tempo produces stable routing shifts. Suppressing slow-thinking experts shortens the reasoning chain, whereas suppressing fast-thinking experts lengthens it, providing causal evidence that these experts influence reasoning depth. Based on this finding, we propose TempoMoE, a training-free, two-stage framework. It identifies fast- and slow-thinking experts via tempo-contrastive routing analysis and redistributes routing weights within the original top-k set according to sample difficulty. It amplifies fast-thinking experts and suppresses slow-thinking experts on easy inputs and reverses this adjustment on hard ones, enabling difficulty-adaptive control over reasoning depth. We evaluate TempoMoE on Kimi-VL, Qwen3-VL, and ERNIE-4.5-VL across six benchmarks. Compared with Vanilla, TempoMoE consistently improves accuracy while reducing output tokens, with accuracy gains of up to 2.6 points and token reductions of up to 35.4%. 

---
# AdaExploreBench: Benchmarking LLM Agents for Proactive Exploration and Test-time Adaptation in Evolving Environments 

**Authors**: Rui Wang, Zishen Lai, Jiayu Liu, Cheng Qian, Jeonghwan Kim, Heng Wang, Xuejun Zhang, Jiateng Liu, Bingxuan Li, Shujin Wu, Qing Zong, Tianshi Zheng, Chunyang Li, Qihan Lin, Chen Li, Dadi Guo, Xiusi Chen, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.25950)  

**Abstract**: Large Language Model (LLM) agents are increasingly deployed in unfamiliar and evolving environments, requiring them to proactively explore, understand, and adapt to environmental dynamics. However, existing benchmarks rarely test whether agents can proactively discover how an environment works and adapt as it evolves. To address this gap, we introduce AdaExploreBench, a benchmark for proactive exploration and test-time adaptation in evolving environments. By instantiating environmental dynamics as hidden rules, AdaExploreBench requires agents to actively acquire evidence and infer the rules. Under implicit rule drift, the latent rule changes without announcement, requiring agents to detect from degraded performance that prior knowledge is stale and re-explore. Experiments on eight leading LLMs show that proactive and adaptive exploration remain challenging: GPT-6 Sol achieves only 61.9% calibrated accuracy, while providing sufficient evidence or removing rule drift improves average performance significantly by close to 10%. These results reveal a critical bottleneck in current LLM agents: they can often reason effectively from informative evidence once it is available, but struggle to actively acquire the right evidence and recognize when previously acquired evidence is no longer valid. 

---
# Speech LLMs That Breathe: Modeling Non-Verbal Vocalizations in Dialogue 

**Authors**: Heyang Liu, Lixin Zhang, Jiayi Huang, Ziyang Cheng, Yuhao Wang, Ronghua Wu, Qunshan Gu, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25942)  

**Abstract**: The user experience of speech interaction models has improved, benefiting from the modeling of paralinguistic attributes and advances in speech reconstruction paradigms. However, human communication follows extensive norms regarding non-verbal vocalizations (NVVs) spanning respiratory, affective, and discourse. The omission of these events imposes inherent limitations on the naturalness of current approaches. In this work, we cover 13 major types of NVVs and construct a natural dialogue dataset that conforms to human conversational norms through an automated synthesis pipeline. We enhance the ability of speech interaction models to generate contextually appropriate NVV events through multi-stage fine-tuning and NVV-aware GRPO (Group Relative Policy Optimization). Based on distinct modeling paradigms, we substantially improve the expressiveness of VocalNet and VITA-Qinyu in empathy interactions and role-play dialogues, and further discuss the architectural necessity for achieving high-quality natural speech generation. 

---
# MITRA-lexicon: Lexica of attested translations from Sanskrit into English, Tibetan and Chinese 

**Authors**: Sebastian Nehrdich, Kengo Harimoto, Sven Sellmer, Oliver Hellwig  

**Link**: [PDF](https://arxiv.org/pdf/2610.25934)  

**Abstract**: The standard Sanskrit lexica are monumental and indispensable, but they cite their attestations as abbreviated references to editions, without links to the digitized texts now widely available and without a translation of the passage, and they naturally do not reflect translations made after them. To address these gaps, we present MITRA-lexicon, a family of corpus-grounded lexica based on span annotations of translation equivalents with Sanskrit as the pivot: Sanskrit-English, derived from the Mitrasamgraha corpus, as well as Sanskrit-Tibetan and Sanskrit-Chinese, derived from the MITRA-parallel corpus of Buddhist literature. In total, the resource comprises 2.18 million headword entries in six directional lexica. Headwords are lemmas and compounds from silver annotations that follow the Digital Corpus of Sanskrit (DCS). Senses are ranked by frequency, and each of the 10.4 million attestations includes its sentence pair, with character offsets of the word and its equivalent. We propose a multi-step pipeline based on grammatical annotation models, fine-tuned large language models, and a teacher-student setup to produce the alignments. We evaluate the lexica against reference dictionaries and manually corrected gold alignments. We release the lexica and the fine-tuned span annotation model, together with a web interface for looking up headwords and their attested senses. 

---
# LEAPS: Long-context Evaluation of Adaptive Pattern Learning and State Tracking 

**Authors**: Howard Yen, Nitya Mani, Alex Renda, Christopher S. Hardin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25919)  

**Abstract**: Long-context language models (LCLMs) power agents that act over long trajectories, from improving a codebase to running research experiments to forecasters that condition on lengthy histories. These applications draw on three distinct long-context capabilities: recall, where the answer is a fact present in the context; state tracking, where the outcome can be computed by performing a series of transformations to a specified state; and pattern learning, where the goal is to make a prediction by sampling from a distribution inferred from many observations. Existing benchmarks primarily focus on the first capability. In this work, we introduce LEAPS (Long-context Evaluation of Adaptive Pattern learning and State tracking), a collection of 12 novel datasets, spanning agentic workflows, behavioral modeling, and forecasting of market and world events. By construction, LEAPS is resistant to data contamination, extends to millions of input tokens, and uses reliable metrics. We evaluate 30 API and open models from 8K to 512K input tokens. LEAPS distinguishes models that perform near perfectly on recall tasks, finding that such models' state tracking and pattern learning abilities can vary significantly and degrade much more dramatically with increasing input lengths--even for pattern learning, where more observations should make learning a theoretically easier task. Moreover, thinking models often underperform their base counterparts, exhausting their generation budget or oversimplifying the task. We hope LEAPS can guide the development of the next generation of LCLMs. 

---
# FisherEdit: Fisher Information Guided Knowledge Editing for Large Language Models 

**Authors**: Yizhou Yang, Jiahong Fu, Zhiwu Wang, Kailong Fan, Haokun Lin, Yichen Wu, Zhenan Sun, Renzhen Wang, Deyu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2610.25914)  

**Abstract**: Sequential knowledge editing in Large Language Models (LLMs) often leads to a degradation of general capabilities due to cumulative parameter interference. In this paper, we revisit prominent projection-based methods like AlphaEdit and reveal a surprising mismatch: their robustness is primarily driven by update regularization rather than the intended null-space projection. Based on this insight, we propose FisherEdit, a theoretically grounded method built on Fisher geometry, which replaces coarse activation-space projections with importance-aware regularization in parameter space. It penalizes changes to sensitive parameters more strongly while retaining flexibility for new facts, and admits an efficient row-wise closed-form solution. Experiments on Llama-3, Qwen2.5, and Olmo-3 across multiple benchmarks demonstrate that FisherEdit achieves state-of-the-art performance while better preserving model stability and downstream task accuracy. 

---
# CALM: Consistency-Aware Logit Masking in Decoding Space for Mitigating Overthinking of Large Reasoning Models 

**Authors**: Shengjing Liu, Feiyang Li, Yuxuan Yang, Yile Wang, Hui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25908)  

**Abstract**: Large reasoning models (LRMs) have achieved strong performance on challenging mathematical and scientific tasks through long chain-of-thought reasoning, but they often suffer from overthinking with unnecessary reasoning steps. Existing truncation-based methods typically detect consecutive trial-answer agreement and force early exit, yet such local stopping signals do not necessarily indicate final-answer stability and can disrupt useful subsequent reflection or verification, potentially leading to performance degradation. We propose consistency-aware logit masking (CALM), a training-free decoding intervention for mitigating overthinking in LRMs. CALM first monitors trial answers at step boundaries and then applies logit masking to action transition point (ATP) tokens such as "Wait", "Alternatively", and "Hmm", suppressing redundant transitional reasoning while allowing the model to continue its thought process more naturally. Experiments across four advanced LRMs and five benchmarks show that CALM reduces token consumption by 18.6%-25.0% compared with full CoT reasoning while maintaining performance, and can even improve accuracy by 0.07%-1.48% by reducing redundant reasoning. LLM-as-a-Judge results also demonstrate that CALM provides more coherent and concise results. 

---
# Learning to Reason Efficiently with Interventional Step Credit 

**Authors**: Guanda Li, Xuefeng Bai, Yulong Chen, Zhiyong Wang, Kehai Chen, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25876)  

**Abstract**: Large reasoning models often produce lengthy chain-of-thought (CoT) traces mixing useful intermediate steps with unnecessary ones. Recent methods explore step-level signals such as attention mass and step entropy to identify and suppress unnecessary steps. Despite their effectiveness, such signals reflect the model's behavior at a step rather than that step's contribution to the answer. To address this, we propose the interventional step credit (ISC), which takes a causal-intervention view and assesses step importance to the reasoning through targeted parameter-level interventions. Building on ISC, we further propose IGPO, which uses ISC to guide fine-grained advantage rescaling in reinforcement learning, reducing redundant steps while preserving the important ones. Experiments across two model scales and nine benchmarks show that IGPO achieves up to 62% average token reduction and better overall performance among compared methods. 

---
# Arabic Financial NLP 

**Authors**: Mo El-Haj  

**Link**: [PDF](https://arxiv.org/pdf/2610.25854)  

**Abstract**: Financial communication connects numerical accounts of economic activity with explanations of performance, responsibility and future expectations. Arabic financial natural language processing offers methods for studying these connections across corporate reports, market disclosures, journalism and banking interactions. Its contribution to digital humanities depends on retaining the relationship between a computational representation and the document from which it was made. This chapter follows that relationship from source collection and Arabic text processing to the analysis of vocabulary, evaluation, entities, causal attribution and multilingual reporting. It then examines what financial summaries and retrieval-grounded answers preserve or lose, and how dialectal intent detection changes the perspective from institutions speaking to institutions listening. Published Arabic resources are discussed in relation to the questions they were designed to answer, with a distinction between financial pretraining corpora, annotated datasets and evaluation benchmarks. A running constructed example connects the methodological discussion without presenting illustrative analysis as empirical findings. The chapter argues for an account of Arabic financial NLP in which linguistic context, documentary provenance and interpretive judgement remain part of the evidence throughout the analytical process. 

---
# Measurement Validity and Representation Dynamics of Confidence Signals in Large Language Models 

**Authors**: Debojeet Ghosh, Shreyansh Kumar, Ayushman Garg  

**Link**: [PDF](https://arxiv.org/pdf/2610.25846)  

**Abstract**: Whether a language model "knows what it knows" matters for trustworthy deployment and for deciding when it should abstain. Prior work usually measures this through one channel: verbalized confidence, a linear probe on hidden states, or agreement among sampled answers (semantic entropy), and often treats these as interchangeable. We compare all three in a pre-registered audit of five Qwen2.5 models (0.5B-7B, including a base/instruct pair at 7B) on factual retrieval (PopQA, SimpleQA) and multi-step reasoning (GSM8K, MATH). First, probes trained on prefill states to predict whether sampled answers will disagree become informative earlier for retrieval (about 7% of layer depth on average) than for reasoning (about 28%). The pre-registered five-layer test supported this only marginally; a finer 21-layer follow-up, designed after those results, supports it more clearly. In most retrieval cells, however, a prompt-only TF-IDF classifier matches the probe, so the early signal may reflect surface features of the question. Second, verbalized confidence is unstable: slightly different prompt templates yield scores that barely correlate. Third, Expected Calibration Error is misleading: verbal confidence looks well calibrated because it clusters near the base rate while discriminating poorly, whereas sampling agreement has worse ECE but about twice the resolution. Of five pre-registered hypotheses, two were falsified, one was partly supported, one was not supported, and one could not be tested because the base model's verbal confidence was degenerate. 

---
# Where RL Breaks: Localizing and Repairing Failures via Targeted Continued Pretraining 

**Authors**: Xinyi Wang, Danqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.25836)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is widely used to improve reasoning in language models, yet growing evidence suggests that it mostly sharpens what the base model can already do. This raises a practical question: when RL fails, what is the base model missing? We look inside the RL policy's own rollouts and find that failures are localized: whether a rollout fails is strongly predicted by a small set of characteristic n-grams, ranging from reflection phrases such as "let's recheck" to specific operations such as "the inverse of A". These failures persist throughout RL training, although they are not beyond the model's reach: other sampled answers to the same question are often correct. Building on this observation, we use the failure $n$-grams to retrieve text from the pretraining corpus and continue pretraining the base model on it before RL. The resulting corpus has only 35M tokens, 3--4 orders of magnitude smaller than typical mid-training. We evaluate this approach on OLMo2 and Qwen3 models trained with RL on (1) GSM8K; (2) a mixture of math, commonsense, and knowledge questions; and (3) general math and science reasoning. Despite its size, the corpus yields consistent gains: our approach outperforms continued pretraining on the same amount of text sampled from the pretraining corpus across 10 out of 11 settings, including by $2.0$ points on GSM8K when inserted before mid-training. The retrieved text also transfers across model families: text retrieved for OLMo2-1B's failures improves Qwen3-4B on the mixture by $1.9$ points over the same baseline. Together, our results suggest that where RL fails is a cheap and effective signal for selecting pretraining data. 

---
# Diffusion Language Models Change Speed Across Tasks and Languages 

**Authors**: Mohsen Fayyaz, Ali Modarressi, Daniel Israel, Lucas Bandarkar, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.25835)  

**Abstract**: Diffusion language models can write several tokens in one forward pass, and they decide during generation how many passes to spend. Two outputs of the same token length can therefore vary a lot in cost. To understand the underlying factors of this cost variation, we analyze two diffusion models across 10 tasks. We find that the number of forward passes per token depends heavily on the task. Text copied or lightly edited from the input is cheap, while open-ended writing is costly. In DiffusionGemma, generating a token for story writing requires more than three times as many forward passes as for grammar correction. Two output properties explain much of this gap: how much of it is new relative to the input, and how rare its words are within the task. The same holds within a single response when thinking is involved. The final answer is cheap when it repeats the model's reasoning trace and costly when it adds something new. In multilingual generation, we find that highly tokenized languages also require more passes per token, adding to the cost they already pay with more tokens. Finally, cost tracks correctness: on the same math problem, wrong answers take more passes per token than correct ones, and weighting candidate answers by decoding speed beats majority voting by up to 4.2 points at no extra cost. Taken together, our results show that diffusion's computational advantage depends on how much of the output is already determined, while its decoding effort offers a practical signal for making inference more efficient and reliable. 

---
# Modal Sense Disambiguation in Language Models: The Role of Context, Training Domain, and Input Frequency 

**Authors**: Ezgi Başar, Yevgen Matusevych, Annemarie van Dooren, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2610.25816)  

**Abstract**: Modals such as may and must can take on different senses depending on discourse context, alternating between root interpretations (e.g., permission, obligation) and epistemic interpretations (e.g., knowledge, belief). While modals are known to be notoriously difficult for children to acquire, limited research exists on whether language models (LMs) can reliably distinguish between different modal senses. We evaluate seven LMs that vary in architecture, size, and training data type using vector-based similarity and surprisal measures across four datasets. Specifically, we investigate whether LMs can leverage contextual information to disambiguate modal senses and whether their performance reflects the distribution of modal senses in their training data. We show that larger LMs trained on diverse text successfully leverage additional contextual information when available. In the absence of context, they tend to favor the dominant sense based on corpus statistics. Smaller developmentally plausible LMs, including those trained on child-directed language (CDL), fail to show consistent sensitivity to modal sense distinctions and do not benefit from context to the same degree. 

---
# CoMem: Learning to Anticipate Failures from Retrieved Experience 

**Authors**: Yun Shen, Ming Shen, Tianjun Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.25790)  

**Abstract**: Reflection lets LLM agents improve from their own attempts, but the memory it builds is local to a single task, so an agent reflecting on its own tends to revisit the same explanation for failure. Cross-task reflective memory addresses this in two forms: retrieved memory keeps past experience explicit and looks up related cases, which is specific but bounded by the bank, and parametric memory internalizes experience into parameters, which generalizes but is grounded in nothing beyond the task description. When both are used, existing agents combine them additively, feeding their outputs to the solver side by side, so retrieved experience never changes what the learned memory produces. We introduce CoMem (Compounding Memory), which makes retrieved experience an input to the learned memory itself, at training and at inference. The agent builds a bank of its own successful and failed reflections, learns from retrieved cases to write failure hypotheses, candidate explanations of how an attempt is likely to go wrong, and at test time reasons from newly retrieved cases to guide each attempt. Varying the retrieved cases gives the memory different perspectives, yielding reflective diversity that is not merely sampled but also grounded in retrieved experience. Across two backbones and five benchmarks in math, code, and multi-hop QA, CoMem achieves strong overall performance. Additional evaluations show that it generalizes to out-of-distribution tasks and enables smaller models to provide effective guidance to stronger solvers. 

---
# Truncated Reasoning Under a Budget: Evaluating System-1 Models as Decision-makers 

**Authors**: Zhongyin Wang, Zhaoqian Xue, Junting Mu, Haozheng Liu, Ye Yuan, Haolun Wu, Lingyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25768)  

**Abstract**: Large language models (LLMs) often generate more reasoning tokens than needed. Limiting the reasoning budget reduces cost, but forcing an answer from an incomplete trace can hurt accuracy. System-1 decision models select among candidate options without producing a reasoning response. We evaluate Jev as a decision reader of truncated reasoning across seven LLM reasoners and five multiple-choice benchmarks (35 pairs). Jev achieves higher mean accuracy than forced answering at tested budgets from 500 to 16,000 tokens, including the estimated costs of reasoning and answering. At a 1,500-token budget, it improves accuracy by 4.1 pp overall and 8.6 pp on truncated questions, with significant gains in 17 pairs under Holm-corrected bootstrap tests. Offline replay estimates 34.5% of the full-run cost, with a 1.13 pp accuracy loss. A policy selected on training folds saves 71.3% of the estimated full-run cost on held-out folds, with a 0.58 pp accuracy loss. We also find that Jev shows strong question-only accuracy, exceeding 500-token forced answering on average (68.0% vs. 65.5%), although this advantage varies by benchmark. These findings suggest that System-1 decision models can complement LLM reasoning and reduce the need for costly reasoning completion. 

---
# A Calibrated Instrument for Detecting Disposition Drift in Language Models 

**Authors**: Jerry Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2610.25735)  

**Abstract**: Before language models are released, they are tested for harmful traits such as willingness to carry out dangerous requests, deception, power seeking, and sycophancy. The safe behavior these tests certify may degrade as models are modified, by topical fine-tuning, new training data, or changes to settings such as the depth of their reasoning. Most work reports such drift as single summary rates scored by another AI model, without the error analysis needed to interpret them. We adapt a calibrated measurement methodology recently developed for assessing how inference optimizations affect the quality of model outputs. Applying it to several misalignment dimensions in two open-weight models, we first establish the instrument's precision, showing that shifts of a few percentage points can be reliably detected. We then induce drift deliberately by fine-tuning on correct and on subtly harmful medical and financial advice; harmful advice induces a far greater shift than correct advice. Most strikingly, a model that reasons before answering can conceal the shift almost entirely in its answers, even though a direct reading of its internal state shows that its disposition has moved. With reasoning switched off, the same model shifts steadily as the share of harmful advice in its training grows. These results demonstrate the importance of testing safety with reasoning turned both on and off. 

---
# Does Unlearning Survive Collaboration? Robust Unlearning for Shared-Model LLM Teams 

**Authors**: Hengsong Liu, Liangyu Teng, Jingnan Cai, Yuntian Shi, Jingyu Zhang, Yang Liu, Jing Liu, Liang Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.25731)  

**Abstract**: LLM unlearning is essential for removing sensitive or unwanted knowledge while preserving useful capabilities. As LLM teams are increasingly used to solve complex problems, forgetting must remain effective during collaboration. Despite advances in forgetting effectiveness, efficiency, and robustness to relearning, team deployment remains underexplored. We find that collaboration weakens forgetting: final-answer evaluations miss recoverable evidence in team messages, including complementary clues supporting joint recovery. We establish theoretically and confirm empirically that controlling final answers or individual messages does not ensure low recovery from their combination. To address this challenge, we propose Aggregation-Conditioned Unlearning (ACU). ACU trains a shared model using reconstruction feedback from individual and pooled messages, prioritizes views with higher recovery risk, and uses replacement-based credit assignment while protecting direct and collaborative utility. TOFU evaluates trajectory-level recovery, while WMDP-bio tests native forgetting and retained utility. On TOFU, ACU lowers full-trajectory recovery from 41.8% to 19.3% in ROUGE-L recall and factual recovery from 34.2% to 18.7% versus output-supervised training, with comparable retained-task performance. These results demonstrate that ACU curbs information leakage in LLM teams and generalizes across team sizes and readers. 

---
# No Matter Who You Are: Addressivity and Sycophancy in Large Language Models 

**Authors**: Seungjong Sun, Hyunmin Jun, Minsik Yoon, Eungu Lee, Choongwon Kang, Young Seok Woo, Jang Hyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.25728)  

**Abstract**: Sycophancy, where an LLM becomes overly agreeable, deferential, or flattering toward the user, has been identified as a major failure mode. While post-training has been implicated as a cause, how it internally gives rise to sycophantic behavior remains insufficiently understood. Post-training transforms the model's output from text completion into utterances directed toward an addressee, thereby giving the output \textit{addressivity}. We hypothesize that the accompanying pressure to follow the addressee's instructions and align with their preferences links this addressivity to sycophancy. To test this, we use Sparse Autoencoders to extract addressivity directions from pre-trained and post-trained models and examine their relationship with sycophancy. Steering models with addressivity directions can amplify or suppress sycophancy, and this effect is observed only in post-trained models. Furthermore, comparing the internal mechanisms of both model types confirms that the association between addressivity and sycophancy is strengthened through post-training. Finally, through a controlled post-training experiment, we find that models are sycophantic not toward a specific addressee but toward any recognized addressee, suggesting that the presence of an addressee, rather than the addressee's identity, is what elicits sycophancy. 

---
# Inside the Scorer: Mechanistic Evidence on Rubric Control in LLM-Based Automated Essay Scoring from 25 Base Models 

**Authors**: Chungche Huang, Po-Ya Angela Wang, Shu-Kai Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2610.25724)  

**Abstract**: Automated essay scoring (AES) using large language models (LLMs) offers promising use cases for language assessment, yet high-stakes testing requires scores that are both controllable and verifiable, which current LLM-based scorers lack. Given that, we investigate whether LLM-based scorers form distinct internal representations for individual scoring rubrics and whether these representations can be independently controlled. Probing and steering 25 open-weight base LLMs (0.13B-10.31B parameters; 14 families) on human-scored BESTEP English writing tasks, we find that holistic scores are decodable from model activations for the longer tasks (Parts 2 and 3) and only weakly for the shortest (Part 1). Furthermore, scoring rubrics have partly separable directions in representation space for the longer tasks, though not for short responses (Part 1). However, causal steering reveals a critical limitation: applying a matching intervention (e.g., steering "grammar" on a grammar-degraded response) yields only a negligible advantage (about 2% of the effect) over a mismatched intervention (e.g., steering "vocabulary"), and none that is detectable in any of the 10 testable comparisons across reliably measured rubrics. Rather than controlling distinct evaluative dimensions, the rubric directions share a dominant lever whose rubric-specific parts are not selective, which may mirror the halo structure of human ratings. Our contributions are two-fold. First, we show that analytic scoring rubrics designed for human raters do not behave as independent controls in base LLMs, even among the seven models whose scores agree best with human raters ($r \geq 0.6$). Second, we establish a methodology that shows that score agreement and decodability do not imply rubric-level control in LLM-based assessment. 

---
# Consistency Training Suppresses Cue Following Without Teaching Cue Selectivity 

**Authors**: Prakhar Gupta, Terry Jingchen Zhang, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25650)  

**Abstract**: Language models often change answers because of a cue, a hint such as "I believe the answer is B" that gives no valid evidence. Consistency training (CT) trains a model to answer the same with and without the cue. Four CT methods apply this idea to output tokens (BCT), residual activations (ACT), attention patterns (AttCT) and sampled-answer rates (RMCT). Prior work suggests some learn different solutions. We train all four on Gemma-2-2B-it and Llama-3.1-8B-Instruct with the same data. We test predictions written in advance and inspect the models with a known circuit that carries cues to the answer. All four make the trained cue less effective, and their fixes share three properties. First, correct cues lose about as much influence as wrong ones. Second, the fix is usually large for cues before the answer options, as in training, and much smaller after them. Llama RMCT depends least on position. Third, the circuit heads still carry most remaining cue-induced errors. Making BCT targets depend on the cue's correctness mostly weakens the fix. CT thus weakens the trained cue without teaching models when a cue is right. Evaluations with only wrong cues at the trained position miss the first two properties. 

---
# MapSteal: Map-based Autonomous Knowledge Stealing Attack Against Large Language Models 

**Authors**: Zhengao Li, Chuxu Zhang, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.25626)  

**Abstract**: Knowledge acquisition from black-box large language models (LLMs) offers a practical approach to adapting smaller student models. However, many existing methods rely on predefined queries or student feedback to guide acquisition, leaving autonomous expansion of knowledge targets an important challenge. We propose \textbf{MapSteal}, a student-independent framework for structured recursive knowledge acquisition. Starting from a small set of initial roots, MapSteal represents acquisition targets using structured states and recursively expands them through teacher-proposed directions. Question realization, proposal admission, and deduplication constrain the acquisition process, while root-aware scheduling organizes exploration under a fixed QA budget. The resulting question--answer pairs are used to adapt student models without requiring student feedback during acquisition. Experiments with Qwen3.5 students at 2B, 4B, and 9B scales show competitive downstream performance across six medical and cybersecurity benchmarks. On the 4B medical evaluation, Full MapSteal improves average accuracy by 1.17 percentage points over a simplified recursive acquisition variant. Cross-domain evaluations show that gains depend on the domain and evaluation panel. These results demonstrate the feasibility of structured, student-independent recursive acquisition while highlighting the distinction between generating acquisition targets and obtaining useful training supervision. Our code is available at \href{this https URL} {this https URL}. 

---
# Cognitive Map in Large Language Models 

**Authors**: Albert Qin, Jonathan Cohen, Gautam Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2610.25596)  

**Abstract**: It has long been proposed that animals construct cognitive maps to support flexible behavior in novel environments. We tested whether large language models form similar representations using relational reasoning tasks and causal "mosaic experiments" that manipulate internal activations in open-weight language models. Models beyond the ten-billion-parameter scale successfully inferred unstated relationships between entities and generalized to synthetic entity names. Across model and task families, models generated abstract representations of relational structure and combined them additively with representations of entity identity, binding entities to roles within that structure. Models stored observed relationships as pairwise links that they composed retrospectively when queried, rather than prospectively positioning each entity in a global coordinate system. Binding was mediated by a sparse, distributed set of key--value pairs; by perturbing these pairs, we implanted "false memories" that altered model behavior in predictable ways. Overall, our findings reveal an internal organization analogous to a cognitive map that explains how language models reason with limited information. 

---
# Assertion and Belief: Moore's Paradox in Language Models 

**Authors**: Anshun Asher Zheng, Weili Cao, Fangcong Yin, David Beaver, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25586)  

**Abstract**: Moore's paradox \citep{moore1942} concerns sentences such as ``It is raining, but I don't believe that it is,'' which may well be true yet cannot be sensibly asserted. The absurdity lies not in the content but in the act, since to assert is to vouch for one's own belief. This makes the paradox a test of whether language models take the speech act into account when computing meaning. We present the first test of Moore's paradox in LLMs, evaluating 12 models on a curated set of Moorean sentences in varied contexts. We find that models largely recognize the infelicity of Moorean sentences, and that the judgement is not tied to a memorized form and tracks the speech act. Using the Rational Speech Act framework (RSA; \citealt{goodmanfrank2016}), we then test whether these judgements arise from a listener reasoning about a speaker. We find the models' own listeners do not reason about their own speakers, and their reasoning traces suggest that they instead recall knowledge about the paradox and apply rules therein. The models thus appear to have learned the paradox as a fact about sentences, not as a requirement on speakers. 

---
# Representations in Motion: Layer-wise Rotation and Trajectory Divergence in LLMs 

**Authors**: Zhuonan Yang, Yik Siu Chan, Wannan Yang, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2610.25574)  

**Abstract**: Recent work suggests that activation dynamics--how representations change during inference--might be more revealing about language model behavior than static readouts from individual layers. In this work, we investigate the layer-wise evolution of representations of factuality, using true and false statements when the model is prompted to either lie or to be honest. We report a geometric phenomenon in which the factuality directions extracted from the lying and honest conditions appear to rotate away from each other across layers. We find that such layer-wise rotation also occurs in a task that requires the model to take different visual perspectives in the 3D world, suggesting that the rotation phenomenon might be indicative of manipulating conceptual spaces in general, rather than unique to factuality. We then investigate the "real time" transformation of conceptual representations on a trial-to-trial basis. We show that it is possible to make strong predictions about the model's target task and behavior from trajectories alone. In addition, diverging points in single-trial trajectories correspond to the onset of rotation in aggregated-trial analysis. Finally, we show that both rotation and divergent trajectories occur in more naturalistic settings in which models are not overtly prompted to "lie" but rather report false statements in the course of completing a different task. 

---
# Flattening Fantasies: Analyzing Stereotypes in Pornographic Discourse 

**Authors**: Sarmad Chandio, Osama Khalid, Sharaf Zia, Ethan Kutlu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25566)  

**Abstract**: This study presents a large-scale quantitative analysis of 5.4 million video
titles from xHamster, the world's fourth most visited pornography platform, to
examine linguistic patterns used to describe different dimensions of
identities (e.g., race and gender). Specifically, we investigate how
nationalities, ethnic groups, and gender identities are portrayed in
pornographic discourse, focusing on stereotypes and deviations from implicit
norms of pornography. The results show the dominance of American pornography
over other groups, which acts as the de facto norm for the adult industry. We
identify persistent ethnic stereotypes reflecting colonial power dynamics,
religious fetishization, and the sexualization of cultural taboos. The
analysis reveals gendered representational differences across cultures, with a
prevalent Madonna-Whore dichotomy in the portrayal of women and stereotypical
depictions of men in physically dominant roles. 

---
# EmotiQ: A Quantum Residual Specialist Framework for Text Emotion Classification 

**Authors**: Carlton Kigonya, Tanaka Chirara, Yunting Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25549)  

**Abstract**: Emotion classification from text is challenging because many affective states are distinguished by subtle linguistic signals and semantically overlapping boundaries. We investigate whether compact quantum modules can serve as targeted residual specialists that assist traditional classifiers on difficult emotion decisions. We introduce EmotiQ, a residual specialist framework for text emotion classification using a frozen transformer classifier as the primary decision maker, identifies challenging emotion boundaries from validation behavior, and trains boundary specialists, including quantum encodings, data re-uploading specialists, prototype and fidelity specialists, and kernel-based specialists. Specialists are permitted to intervene only when they satisfy validation-based admission criteria. We evaluate the framework across six text emotion classification datasets and establish traditional model baselines. Experimental results show that the admitted EmotiQ policy improves mean macro-F1 metrics over the baseline, and achieves positive gains on five of six datasets. Among the quantum components, ZZ interaction feature maps are the most consistently effective, yielding a mean macro-F1 improvement of +0.0030 and positive gains on four datasets. These findings suggest that current quantum NLP methods can be deployed as residual specialists that provide localized improvements under strong classical supervision. 

---
# Where Are Language-Specific Features Represented in Multilingual LLMs? Everywhere 

**Authors**: Mikaela Martins, Jonathan Dunn  

**Link**: [PDF](https://arxiv.org/pdf/2610.25505)  

**Abstract**: Previous work has argued that multilingual LLMs encode language-specific features most strongly in their initial and final layers, implying that intermediate layers support language-agnostic representations. This paper reconsiders this finding by performing four different experiments: three that undertake a fine-grained analysis of representations of Portuguese and Spanish, two closely related languages, and one large-scale language identification experiment that covers over 200 languages. The results of these experiments challenge the strict separation between language-specific and language-agnostic layers in multilingual LLMs, using Llama 3.2 1B as a test case. By analyzing and manipulating contextual sentence embeddings, we demonstrate that language identity remains strongly recoverable throughout the model rather than being restricted to initial and final layers. Moreover, we find that the model tends to classify ambiguous strings as closer to Spanish than to Portuguese, revealing a skew in model behavior that differs from human judgments. 

---
# Self-Distillation for Supervised Fine-Tuning 

**Authors**: Wenxuan Zhu, Bang An, Shuming Liu, Wenqing Cui, Tianqi Xu, Bing Li, Yibo Yang, Peter Wonka, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2610.25497)  

**Abstract**: Supervised fine-tuning treats the expert demonstration or gold response as a one-hot target, suppressing valid alternative continuations that the model would otherwise assign probability to. We propose Gold-Conditioned Distillation (GCD), a method that turns expert demonstrations into model-native supervision. GCD uses one model as both teacher and student: the teacher conditions on a prompt containing the gold response in context, the student conditions on the original prompt as at inference, and the student is trained to minimize the forward KL divergence to the teacher's next-token distribution at each gold response token. This lets the model use its prior knowledge to teach itself the new task: correctness is anchored by the gold response, while the soft target preserves plausible alternatives. Across various training tasks on Qwen and Llama models, GCD improves new-task accuracy over SFT and generally reduces degradation on general benchmarks. 

---
# DiaLong: Benchmarking Reminiscence in Long-Context Dialogues 

**Authors**: Owen Bianchi, Samer Aslan, Ryunosuke Saito  

**Link**: [PDF](https://arxiv.org/pdf/2610.25463)  

**Abstract**: How well can transformer-based Large Language Models (LLMs) retain and recall memories in long contexts of dialogue? LLMs are known to forget long-range factual information in their context, yet the field lacks a comprehensive benchmark to quantitatively assess these shortcomings. We present DiaLong, a long-context dialogue dataset and accompanying memory task that standardizes evaluation in this space. It consists of 100 unique conversations created by humans and then augmented with LLMs, extending beyond lengths found in existing datasets. Our selected memory task is fact recall: classify 40 candidate facts about a dialogue as true or false. This task directly probes the model's ability to actively retain and retrieve information, serving as a critical measure of memory within extended conversational contexts. In testing both the latest LLMs and proposed memory enhancements on DiaLong, we demonstrate that more work needs to be done to enable LLMs with better memory performance. By enhancing memory capabilities, LLMs can improve in maintaining coherence, relevance, and human-like interaction over extended conversation. Improving these capabilities will help LLMs fulfill diverse user needs and expectations in various conversational scenarios. 

---
# Quality-Controlled Reply and Summary Data for Post-Training Small Language Models in AI Call-Screening Agents 

**Authors**: Jiaqi Gan, Tsang Ng, Ankit Raj, Tommy Duong, Xingyu Shen, Kidus Zewde, Huaiyu Guo, Simiao Ren  

**Link**: [PDF](https://arxiv.org/pdf/2610.25458)  

**Abstract**: Small language models have improved to the point where a phone can plausibly screen its own calls, answering on an absent owner's behalf without sending call content off the device, a privacy-respecting alternative to cloud call handling. Realizing that promise takes a model that can do two things well: reply within the owner's authority during the call, and afterward tell the owner what happened and what needs attention. Released small models do neither reliably - they raise unwarranted alarms about ordinary callers and write summaries that drop material conditions or report a caller's claims as fact - and real calls supply no reference replies or summaries to learn from. We construct supervision for both tasks from synthetic calls under an information-boundary constraint: a caller writer works from a private scenario brief, a reply writer sees only the conversation up to the current turn, and a summary writer sees the completed transcript. Separate model-based reviews of the full dialogue and the summary decide which calls enter training, yielding a corpus of 3,395 calls with 15,102 targets (11,707 reply, 3,395 summary). Training a single shared LoRA adapter on both tasks, we evaluate six models (0.6-4B parameters) before and after adaptation on 192 held-out calls. Reply qualification rises from 21-54% to 94-97% and summary qualification from 0-6% to 43-72%; all twelve paired gains (six models x two tasks) have bootstrap intervals above zero, and the summary gains hold in every source family. Because one model both screened the training data and scored the outputs, these rates measure agreement with that evaluator's standard rather than human acceptance, which a planned human study will test. After adaptation, coherence and naturalness failures nearly vanish, leaving grounding and usefulness - faithfully tracking who said what, under which conditions - as the remaining challenge. 

---
# Design Considerations in Low-Resource Archival OCR: A Case Study in Lakota 

**Authors**: Téa Wright, Laura X. Ma, Claudia Iron Hawk, Hannah J. Haynie, Alexis Palmer, Alane Suhr, Abteen Ebrahimi, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2610.25440)  

**Abstract**: Building and expanding access to digital archives supports language reclamation and creates lasting resources for community revitalization. Designing effective tools to support these efforts requires taking into account the community-, language-, and document-specific context. We investigate how small ($\leq$3B) vision-language models (VLMs) can be adapted as optical character recognition (OCR) tools for Lakota, a North American Indigenous language. In addition to benchmarking modeling approaches, we inspect four practical facets of low-resource archival OCR that arise in previous work and conversations with community members: constrained training budgets, segmentation noise, scan degradation, and orthographic variance. We demonstrate that fine-tuning VLMs for OCR post-correction on 35 pages of transcribed data decreases character error rate (CER) from 17.1 to 3.9\% on a bilingual corpus, and find that training only with nine pages increases CER by only 1.1\% over this. However, evaluating under realistic automatic segmentation nearly doubles the CER of the best-performing models, stressing the importance of taking realistic use cases into account. We hope this work encourages future research on developing language technologies grounded in the workflows and constraints of ongoing revitalization efforts. 

---
# RARE: Reinforcement Learning with Adaptive Rubric Evolution for Open-Ended Generation 

**Authors**: Xueyun Tian, Bingbing Xu, Chuyue Li, Jiaming Liu, Chao Zhang, Weijia Feng, MingHua Ma, Yu Zhao, Jingchu Wang, Ruihua Huang, Yuanzhuo Wang, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.25434)  

**Abstract**: Verifiable rewards have advanced reinforcement learning in mathematics and code. However, many valuable open-ended tasks share three properties: no automatic verifier of overall quality, multiple requirements to satisfy jointly, and multiple valid responses to each query. Training therefore needs feedback on requirement satisfaction across different valid responses. Rubric-based reinforcement learning provides this feedback by aggregating scores from explicit criteria. As learning progresses unevenly, some criteria become fully satisfied and lose discrimination while others remain informative. Dynamic methods revise shared rubrics or select evolving criteria to restore discrimination, but may also change useful criteria. To address this, we propose Reinforcement Learning with Adaptive Rubric Evolution (RARE). The core idea is to use current policy feedback to guide both which criteria change and what their replacements assess. Sample-level updates replace criteria fully satisfied across current outputs with checks of remaining quality differences, retaining all other criteria and scores. Within this rule, we compare rewarding desirable qualities with penalizing observed flaws. Video script generation exemplifies these properties and supports content production. Its demands for coherent, executable narratives under constraints make it a challenging testbed. Targeting flaws performs best across three video types with a mean gain of 3.82 points over matched static rubrics. Model and human preferences support these gains. Improvements also extend to writing and general generation benchmarks. 

---
# Mixture-of-Memories: Low-Overhead Language Modeling with Modular Knowledge Injection 

**Authors**: Yifan Lu, Rubin Wei, Ziyun Zou, Junhong Liang, Haolin Yang, Xinlin Zhuang, Abhijit Das, Adinath Madhavrao Dukre, Shadab Khan, Zhouhan Lin, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2610.25425)  

**Abstract**: Large language models store substantial factual knowledge in their parameters, yet this knowledge is difficult to control, update, and reliably elicit at inference time. Existing memory-augmented approaches typically rely on a single auxiliary mechanism, such as retrieval, parameter updates, or neural memory, and therefore apply the same form of correction to tokens with very different knowledge demands. We propose Mixture-of-Memories (MoM), a memory-augmented language modeling framework that separates routed neural memory from writable factual memory. MoM combines a frozen base LM with two neural memory experts: a latent memory that provides low-latency hidden-state corrections and a lexical memory that provides a token-space knowledge pathway. A lightweight Budget Router sets per-token mixture weights from uncertainty and disagreement features, replacing the fixed interpolation of prior memory modules. For post-hoc updates, an external explicit memory stays outside the router, because a router trained before facts are written learns to down-weight it; new facts are appended without retraining any component and used only when an answer-level gate accepts the retrieved fact. Experiments across language modeling, factual QA, update analysis, and efficiency show that MoM improves both predictive accuracy and updateability while retaining low inference overhead. 

---
# Towards Figurative Language Understanding in Moroccan Darija: An Evaluation of Large Language Models 

**Authors**: Jihad R'baiti, Youssef Hmamouche, Emmanuele Chersoni, Chu-Ren Huang, Amal El Fallah Seghrouchni  

**Link**: [PDF](https://arxiv.org/pdf/2610.25406)  

**Abstract**: Figurative language remains a major challenge for Machine Translation (MT) systems, as the intended meaning of such expressions often goes beyond literal translation. Although recent advances in Large Language Models (LLMs) improved the translation of Arabic dialects, benchmark resources for evaluating figurative language translation in Moroccan Darija remain scarce. To this end, we introduce the Moroccan Darija Figurative Language Corpus (MDFC), a benchmark dataset for MT and multiple-choice question (MCQ) selection covering idioms, metaphors, and proverbs. Using MDFC, we evaluate 19 open- and closed-weight language models and MT systems at both the expression and contextual levels under zero-shot and six-shot settings. We also conduct layer-wise similarity-based probing to determine how model representations distinguish figurative from literal meanings. Our findings reveal that closed-weight models achieve the strongest translation performance, while the benefits of contextual information vary across models and tasks. Differences between translation performance and MCQ accuracy highlight the distinction between generating an appropriate figurative translation and recognizing the intended meaning among candidate translations. 

---
# From Unverifiable Domain Text to Cross-Lingual RLVR via Code-Switched Task Synthesis 

**Authors**: Cheonbok Park, Jihwan Kim, Shinyoung Joo, Joosung Lee, Hwiyeol Jo, Jeonghoon Kim, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2610.25399)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) can improve reasoning in large language models, but specialized non-English domains expose a data mismatch. Raw English domain text is often available, while target-language RLVR tasks with verifiable rewards are scarce. We introduce code-switched task synthesis for cross-lingual RLVR, a data-centric approach that derives target-language and code-switched RLVR task variants from otherwise unverifiable English domain text. Using fill-in-the-middle multiple-choice QA as a seed task format, we synthesize automatically verifiable tasks from domain passages and reshape them into cross-lingual variants. These variants preserve the same evidence-answer relation but change how English and target-language evidence are presented during RLVR. On Korean medical benchmarks and cybersecurity benchmarks across four target languages, code-switched training outperforms matched monolingual and bilingual-mixture baselines on average target-language performance while maintaining comparable English performance. Additional analyses provide evidence of better target-language input understanding and stronger cross-lingual alignment in both model outputs and internal representations. These results suggest that code-switched training helps target-language inputs access domain reasoning learned more strongly in English. 

---
# Aligning Cultural Behaviors in Arabic Language Models via Preferences 

**Authors**: Junhong Liang, Daniil Orel, Saeed Almheiri, Bilal Elbouardi, Younes Samih, Abed Alhakim Freihat, Zhuohan Xie, Momina Ahsan, Muhammad Dehan Al Kautsar, Sarfraz Ahmad, Omar El Herraoui, Mohamed Anwar, Preslav Nakov, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2610.25362)  

**Abstract**: Arabic LLMs have improved rapidly, but aligning them with country-specific cultural behavioral preferences remains underexplored. Previous work has focused primarily on instruction-tuning for factual knowledge, overlooking nuanced cultural behaviors. We propose ArabPref, which to our knowledge is the first large-scale pan-Arab preference data set for the alignment of cultural behavior conditioned on the country. Our contribution includes two resources in English and Modern Standard Arabic: a culturally grounded preference dataset for training and evaluation, and a multiple-choice benchmark designed to assess culturally aligned behavior across nations. All test data are validated by native speakers to ensure authenticity. To optimize cultural alignment, we experiment with post-training methods across multilingual and Arabic-centric language models, evaluating performance on both generation tasks and our cultural multiple-choice benchmark. By releasing an evaluation benchmark, we provide a foundation for systematically studying and improving culturally aligned behavior in language models. 

---
# Where and Why Search-Augmented LLMs Fail When Reasoning over Rapidly Evolving Knowledge 

**Authors**: Tianyi Hu, Bhavyajeet Singh, Niket Tandon, Akhil Arora  

**Link**: [PDF](https://arxiv.org/pdf/2610.25323)  

**Abstract**: As the world changes, new information is published while older records remain online. This complicates reasoning for large language models (LLMs), which must retrieve evidence and resolve conflicts to determine which information remains valid. Existing evaluations treat information seeking as a black box, revealing little about where and why LLMs fail. To address this gap, we introduce TREK, a suite of three components: TREKTRACE is a plug-and-play framework that attributes failures to specific stages of retrieval and reasoning. We apply it to TREKBENCH-WILD, a meta-benchmark of real-world questions that require complex reasoning over rapidly evolving knowledge. Surprisingly, the main bottleneck is post-retrieval reasoning rather than retrieval: frontier models are often misled by conflicting evidence even after retrieving all the information needed to answer. To study this bottleneck, we build TREKBENCH-CTRL, a controlled benchmark with distractors spanning temporality, relevance, and trustworthiness, and show that benchmarks dominated by outdated distractors substantially underestimate task difficulty. Together, TREK provides a testbed for diagnosing and improving reasoning over rapidly evolving knowledge. 

---
# Atoms, Not Knowledge: What Production-Scale Addressed-Memory Writes Deposit 

**Authors**: Wei Gao, Cheng Tian, Duan Yang, Jianqing Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25317)  

**Abstract**: Deterministic hash-addressed conditional memory permits local post-hoc knowledge injection. Extending arXiv:2610.21553's bounded-feasibility audit rejecting under-specified generality claims with repaired probes and training-dose/capacity axes, we give, to our knowledge, the first production-scale write-content verdict. We test 124M-7B self-trained/publicly pretrained beds, Qwen3.8-Flash-Next-FP8 (176.95B; 320,001,536 FP8 rows), and DeepSeek-V4.1-Flash. Write coverage requires a sufficiently trained backbone prior, full-width readout, trainable gates; undertrained-bed intervention confirms a threshold, not a slope. At fixed budget, coverage falls from 98% (50 facts) to 12.7% (1,000); table size is not binding. Two-hop composition does not exceed the independent single-hop baseline (26% observed vs 25% expected; 15% lexical-distortion floor). Synergy remains absent across dose/load in two independent bed families. A 28-percentage-point in-context gap (given vs recalled intermediates) localizes failure to retrieval-for-reasoning, not gate access. Writes are independently retrievable, precisely addressable, reversible atoms, not composable into reasoning. A layered protocol, the first, to our knowledge, to cover compositionality in behavioral memory evaluation, separates baseline, synergy and distortion floor. Any addressed/key-value memory must report coverage, synergy ratio and distortion rate before deployment claims. Composition must be system-orchestrated; multi-constraint queries should explicitly chain written atoms. In-context controls show given intermediate entities are exploited at twice the implicit rate. Production load fits exponential interference ($R^2=0.98$); its mechanism, between budget dilution and write interference, remains open. All numbers are recomputed from archived raw logs/JSON; a fresh 52-target production rerun gives 52/52, exactly matching archived baselines. 

---
# SPARC: Spoken Profile Authorship Recognition Corpus 

**Authors**: Efthymios Tsaprazlis, Bowen Yi, Tiantian Feng, Junghwan Kim, David Jurgens, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2610.25265)  

**Abstract**: Speech anonymization has focused on the acoustic signal, leaving open how much speaker information the transcribed words themselves retain. In this work, we study pairwise speaker and demographic-attribute verification from transcribed speech. We introduce SPARC (Spoken Profile Authorship Recognition Corpus), a benchmark built from two podcast corpora comprising over 12,500 speakers, 1.5M quality-filtered utterances, and speaker-level inferred or source-provided labels for sex, age, and accent, and evaluate five attribution models and three LLMs on identity- and attribute-verification trials with episode- and metadata-based topic proxies. LLM adversaries distinguish same-speaker pairs from text alone. Attribute-label agreement is harder to verify, but LLMs perform sex- and accent-label verification above chance. Finally, releasing only short probe texts (50 words) degrades the strongest adversaries the most, offering a partial reduction in observed linkage. 

---
# Hidden Cartographer: Hidden State Guided Data Selection for Dual Budgeted Instruction Tuning 

**Authors**: Zixuan Qin, Qi Qin, Erbo Li, Ting Wei, Yifan Sun, Zhaoxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2610.25215)  

**Abstract**: Given the high cost of expert annotation and limited fine-tuning resources, effective data selection from large unlabeled pools under strict budgets has become a crucial step in adapting LLMs to specialized domains. Existing label-dependent methods risk distorted selection when using pseudo-labels, while traditional active learning methods do not explicitly characterize instruction-tuning dynamics. To address these limitations, we propose Hidden Cartographer (HC), a two-stage framework that uses hidden states to guide annotation and exact training dynamics to select the final training subset. The first stage is Progressive Candidate Expansion, which utilizes the proxy to iteratively select top unlabeled candidates for expert annotation, merging them to recompute training dynamics and update the proxy until the annotation budget is exhausted; the second stage is Exact Cartography Pruning, which ranks these annotated candidates by their exact dynamics scores to isolate a compact subset under the final training budget. Experiments across the medical, legal, and financial domains demonstrate that HC achieves the highest average accuracy among all budgeted baselines, requiring annotation for only 20--30% of the original pool and fine-tuning on just 10--15%. 

---
# Calibration-Free Early Stopping for Chain-of-Thought Reasoning via E-processes 

**Authors**: Dong-Jae Lee, Gyojin Han, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.25212)  

**Abstract**: Reasoning large language models (LLMs) benefit from test-time scaling, motivating adaptive policies that spend tokens when they help and stop early otherwise. Existing rules halt when a confidence or stability signal crosses a threshold, but such thresholds rarely transfer across models and datasets, and validation-calibrated rules lose their guarantee under shifts in test-time horizon or signal dynamics. We reframe adaptive stopping as sequential evidence accumulation along the reasoning trajectory rather than threshold calibration on held-out data. Our framework runs two e-processes on a shared per-chunk signal: a commit stream tracks the signal's position relative to its slow trend, and an abstain stream tracks its signed velocity, making the two anti-correlated by construction. Under signal-level nulls on the per-chunk increments, Ville's inequality yields an anytime-valid signal-level false-alarm bound at user-specified significance levels, with no held-out calibration. Across various model--dataset combinations, the framework matches or exceeds calibration-based stopping on the token--accuracy Pareto front using zero calibration data, and the same significance level transfers unchanged across cells where calibration becomes infeasible. 

---
# A Thousand Faces: A Persuasion Benchmark with Experience-Grounded Persuadees Informed by Psychology and Pragmatics 

**Authors**: Ziqing Zhuang, Dingyi Zhang, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.25206)  

**Abstract**: People can express the same beliefs and desires yet need different reasons to change their minds. What makes an argument relevant depends on how those views arose: experiences shape what a person finds credible, what matters to them, and what could prompt reconsideration. These personal grounds may also be expressed indirectly, through contextual clues in conversation. Benchmarks that specify beliefs and desires without these connections can reward arguments that match a description while overlooking whether they address the person's reasons. We introduce A Thousand Faces (ATF-Persu), with 1,000 personas across 200 topics and nine domains. Drawing on personality and attitude theories and conversational implicature, we construct five personas per factual context, linking experiences to judgments, expression, and grounds for revision. Evaluations show lower persuasion success than three comparison benchmarks under their respective protocols. Complete backgrounds yield broader use of supplied personal grounds and higher persona faithfulness than simplified specifications; blinded reviewers also rate ATF-Persu above CToMPersu on identifiable personal reasons, their influence on responses, and contextual expression. The dataset is publicly available at this https URL. 

---
# Acharya 2.0: An AI Tutor for Self-Learning Subjects in Hindi 

**Authors**: Shantipriya Parida, Debasish Dhal, Sonal Khosla, Sambit Sekhar, Muhammed Abdur Rahmaan Kamaldeen  

**Link**: [PDF](https://arxiv.org/pdf/2610.25192)  

**Abstract**: Text processing is crucial for low-resource languages as it enables the development of tools and technologies that bridge the digital divide and provide equitable access to information and services. By empowering low-resource languages with robust text processing, we can preserve linguistic diversity, enhance education, improve communi- cation, and foster inclusivity in a rapidly digitizing world. Due to the lack of sufficient annotated datasets and lin- guistic resources, it is challenging to develop accurate and reliable language technologies for these low-resource languages. Large Language Models (LLMs) have attracted sig- nificant attention with their performance and the value they produce. However, with the ever-evolving language, adapting LLMs for a specific task is time-consuming and resource-intensive. Thus, Retrieval-Augmented Genera- tion (RAG) has overcome this limitation by augmenting LLMs with external knowledge, thereby improving their efficiency and extensive task-specific adaptation. Most applications of the RAG technique have been limited to English and has seen limited application in In- dic languages. This paper introduces the RAG-powered AI-Tutor application, Acharya 2.0, designed to produce academic assistance to Hindi-speaking students. The client-server-based web application integrates many fea- tures to support students in accessing and interacting with educational content. 

---
# Reranking Answers of Large Language Models with Knowledge Graphs 

**Authors**: Mikhail Salnikov, Hai Le, Olga Tsymboi, Ivan Lazichny, Dmitrii Iarosh, Egor Cheremiskin, Andrey Savchenko, Dmitry Simakov, Alexander Panchenko  

**Link**: [PDF](https://arxiv.org/pdf/2610.25185)  

**Abstract**: Answering natural language questions over knowledge graph data is challenging due to the vast number of facts, which can be difficult to process and navigate. One potential solution for this issue is to use mined subgraphs related to the query, although this process still requires extracting these subgraphs. This research presents a solution for extracting subgraphs related to entity candidates from a question-and-answer set, which can be obtained by inferring a large language model by calculating the shortest paths between entities. The proposed approaches detail various features that can be extracted from the subgraphs and reranking models to select the most probable answers from a list of candidates. Experiments were conducted on Wikidata to evaluate the effectiveness of the proposed approaches. This involved enumerating all the main feature types that can be extracted from mined subgraphs and a detailed analysis of the proposed features and reranking method combinations. In addition, a public web application that provides a useful web tool for studying the graph space between question and answer entities has been developed to work with subgraphs. This includes visualization of the extracted subgraph and automatic generation of natural language text to describe it. 

---
# Concept Representations in Vector Space: Toward an Interdisciplinary Study of Meaning Variation 

**Authors**: Simon Josef Durstewitz  

**Link**: [PDF](https://arxiv.org/pdf/2610.25184)  

**Abstract**: An increasing number of disciplines recognize the explanatory potential of meaning-bearing mental structures through which individuals interpret the world and guide their behavior. Concept representations constitute a core class of these structures, emerging from individuals' experiences with concept referents and encoding and organizing their understanding of them. Recognizing a private level of meaning, where concepts are not objectively defined by rules in the world but constitute subjective structures in individuals' minds, raises a fundamental question: do different individuals actually mean the same thing when they invoke the same concept? As latent structures, concept representations are difficult to access and model empirically. The present study introduces a measurement framework that makes such representations empirically accessible through textual traces elicited under standardized conditions. Using transformer-based sentence embeddings, the framework maps standardized responses elicited for concepts into a high-dimensional shared vector space, opening concept representations to the systematic analysis of interindividual variation. It develops three analytical avenues: (1) comparing representational sharedness across concepts, (2) identifying variables that correlate with representational variation, and (3) examining how focal actors or subpopulations deviate from a reference population. Each avenue is illustrated through an empirical proof-of-concept drawing on a large-scale free-association dataset. Together, the demonstrations show how researchers from across disciplines can move from a research question and data source to analysis and interpretation. An accompanying GitHub repository provides a ready-to-use implementation of the framework for existing and custom datasets. 

---
# Legal Guidance in the Wild: How Users Seek Legal Help in Real-World LLM Conversations 

**Authors**: Amirreza Mohseni, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2610.25159)  

**Abstract**: People ask conversational AI models about their legal rights, obligations, procedures, and personal situations that may have legal consequences. To study this behavior, we first need to distinguish requests for legal guidance from other legally themed discussions and unrelated conversations. We contribute a multi-axis taxonomy of legal queries capturing whether legal guidance is sought, how a request is formulated, and which legal domains it concerns; apply it to 2,122 naturally occurring WildChat-1M conversations; and create a human-reviewed gold evaluation set of 200 conversations. All annotations, model checkpoints, and evaluation code are publicly available. In this enriched sample, 71.1% of conversations containing legal content seek personalized or practical guidance. Rights-and-options advice is the most common primary request type among legal-content conversations, and 25.3% of guidance-positive conversations span multiple legal topics. The gold evaluation set supports a two-stage comparison of ModernBERT-base and ModernBERT-large against a TF-IDF cascade. Our framework supports large-scale studies of legal guidance in conversational AI, and the public release of the annotated data facilitates reproducible research and future model development. 

---
# Do LLMs Reproduce the Age Gap in Loneliness? Evaluating Survey Simulation against European Survey Data 

**Authors**: Yuliya Li, Achim Rettinger, Dominik Beyer  

**Link**: [PDF](https://arxiv.org/pdf/2610.25158)  

**Abstract**: Large language models (LLMs) are often used to simulate survey respondents, but matching survey averages does not establish accurate demographic differences. We test whether LLM outputs reproduce within-country age gaps in loneliness using the 2022 EU Loneliness Survey, comparing adults aged 18-34 and 65 or older in France, Italy, Poland, and Sweden. We evaluate simulation with and without age information and direct distribution estimation against a probability-based sample, with a non-probability sensitivity reference. Both human samples show higher mean loneliness among younger adults. Under age-specified simulation with standard response-option order, Mistral Small 4 and GPT-5.6 Terra reverse this ordering under all three prompt framings in three and two countries, respectively, mainly by overestimating older-adult means. DeepSeek V4 Flash preserves the ordering everywhere but underestimates both group means, with errors largely cancelling in France and Italy. Against the probability-based reference, direct estimation has lower absolute age-gap error than simulation for Mistral and Terra in every country. For DeepSeek, simulation has lower gap error but direct estimation has lower group-mean error; these comparisons depend on low-validity older-adult simulation cells. Evaluating LLM outputs as proxies for age groups within countries therefore requires checking both group means and their contrast against human survey data. 

---
# Morpheus: a fully local, defense-in-depth de-identification engine for Greek clinical text - a benchmark evaluation 

**Authors**: Filippos-Paschalis Rorris, Nikoleta Stanitsa, Tatiana Rorri, Foteini Rorri, Constantine N. Antonopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2610.25155)  

**Abstract**: Background and Objective: Clinical free text cannot leave the hospital without de-identification, yet no de-identification system or benchmark exists for Greek clinical text. We evaluated Morpheus, a fully local de-identification pipeline for Greek clinical documents, on 500 synthetic discharge summaries. Methods: The notes (300 cardiac surgery, 100 cardiology, 100 nephrology) were generated by a frontier large language model (Gemini 3.1 Pro) constrained to authentic Greek hospital document layouts. Notes were pre-annotated at generation; two clinicians then independently reviewed and corrected every note, blinded to each other, with disagreements arbitrated. Primary outcomes were recall over must-redact identifiers and over-redaction of clinical content, measured over must-survive probes. Secondary outcomes: residual leaks flagged by an independent egress-review layer, processing time and reproducibility. Results: Inter-annotator agreement was F1 = 0.998. The de-identification rate was 96.59% (7,666/7,937 identifier instances; 95% CI 96.16-96.96): 98.49% for direct and 93.42% for quasi-identifiers. No patient's own complete identifier survived in any note; of 271 residual instances, 199 (73.4%) concerned institutional identity, chiefly the hospital letterhead. The over-redaction rate was 2.82% (190/6,747 probes; 95% CI 2.45-3.24). The egress guard flagged 2 of 271 residuals, both complete telephone numbers. Two independent runs gave identical verdicts. Conclusions: A fully local, deterministic pipeline removed 98.5% of direct patient identifiers from Greek clinical text while altering under 3% of protected clinical content; most residuals concerned the institution, not the patient. To our knowledge this is the first benchmark evaluation of clinical-text de-identification for Greek and the first quantified evaluation of an independent egress-guard layer. 

---
# TIRE: Top-Down Formula Instantiation with Reliable Execution for Physics Problem Synthesis 

**Authors**: Xiaoyu Wu, Zhaoxin Yu, Xinglin Xiao, Nan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25134)  

**Abstract**: Advancing physics reasoning in large language models requires scalable training data that connect physical laws, concrete problem scenarios, and verifiable numerical answers. Expert-designed problems offer top-down control over physical laws, conditions, and scenarios, but require costly human effort. LLM-based synthesis from texts scales efficiently, but model-generated answers and assessments leave supervision vulnerable to hallucinations. We introduce TIRE, a framework for Top-Down Formula Instantiation with Reliable Execution that brings knowledge-driven problem construction to large-scale data generation. Top-down instantiation uses reusable physical formulas and their applicability conditions to control the physical relations and numerical configurations of generated problems. Reliable execution establishes reference answers before problem generation, providing a common numerical foundation for problem construction and teacher distillation. Together, these mechanisms combine control over problem content with execution-grounded supervision for both supervised fine-tuning and reinforcement learning. Using TIRE, we construct TIRE-19k, spanning nine subfields of physics. Across 3 backbone LLMs and 4 benchmarks, TIRE-19k outperforms 5 representative datasets with equal amounts of SFT data, exceeding the strongest baseline by 2.86 percentage points on average. Further GRPO raises Qwen3-8B's average accuracy by 1.52 percentage points over its TIRE-19k SFT checkpoint. These results demonstrate the value of generating physics training data from explicit physical knowledge with reliable execution. 

---
# MemBubble: Benchmarking How Agent Memory Shapes LLM Response Diversity 

**Authors**: Jiajun Tan, Fei Sun, Xinyu Ma, Shixuan Zhang, Du Su, Qi Cao, Daiting Shi, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.25127)  

**Abstract**: Memory serves as the core component of LLM-based agents, enabling personalized responses via extracting and using user preferences from past interactions. However, its use may reduce response diversity, thereby narrowing the range of information available to users. Existing evaluations rely largely on synthetic interactions and cover few memory systems, leaving the effects of memory on response diversity insufficiently evaluated in real-world scenarios. We introduce MemBubble, a benchmark of 1000 instances built on preferences extracted from real user-assistant conversations. To measure the extent to which memory concentrates response distributions, we construct test queries in scenarios distinct from user preferences while preserving topical relevance through shared preference categories. Across multiple LLMs and memory mechanisms, entropy decreases by 34.7% on average compared with directly providing the full interaction history, highlighting the important role of memory processing in diversity loss. We also explore simple user-side interventions and changes in information availability but find that these do not effectively close the diversity gap. Our findings highlight the need to consider preserving diversity and user access to information alongside improvements in personalization when designing and using agent memory systems. 

---
# Memories That Outlive Their Reasons: Dependency-Tethered Memory for Tool-Using LLM Agents 

**Authors**: Yiqi Wang, Jinqian Ju, Jiaqi Zhang, Zhangkai Wu, Yiqun Duan, Zequn Sun, Taotao Cai, Mingkai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.25123)  

**Abstract**: Agents now act on the Web, publishing packages, models and datasets through registry APIs, and remembering past verifications spares them the work. But a verdict rests on premises. When a signing key is revoked, everything built on it stops passing, yet the advisory names only the key, and the old verdicts of its dependents may never share a retrieval window with it. This non-locality of invalidation turns memory into a liability: in long simulated chats, an agent reading a memory of atomic facts publishes 62% of the objects whose failure lies upstream, each backed by a record that was true when written. Yet derivation is local even when invalidation is not. Tether records, at write time, the exact versions each verdict derives from, and at read time withdraws any verdict whose support has failed anywhere in the store; the agent reads the same fields as before. On Fallout, a generator of versioned worlds with hypothesis-blind simulated histories shared by every memory, Tether publishes 0.3% of upstream-failing objects across three domains when relations are stated with their verdicts, with 37% fewer verification calls than checking everything. A-Mem and Graphiti re-verify most candidates (0.83-0.86 checks per candidate), which limits their savings, yet publish 9% of upstream-failing objects; Mem0 relies on memory more and publishes 42%. Flushing every verdict on any failure is as safe as Tether but costs 32% more checks. The gain over fact memory holds with three open-weight backbones and against a whole-store window. These gains depend on capturing dependencies: when relations arrive in sessions apart from their verdicts, Tether's downstream unsafe publication rate rises to 9.4-22.1%, and a weak writer that misses edges loses safety in the same way. 

---
# Behave Like Experts: Fusing Data with Semantic Knowledge for Effective and Efficient Semantic Map Construction 

**Authors**: Zhu Liu, Zhen Hu, Yuxuan Liu, Ying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25122)  

**Abstract**: Semantic map models (SMMs) construct a graph that reflects conceptual similarity by comparing cross-lingual form-meaning mappings. Experts build such models not only by obeying the data constraint, i.e., the connectivity hypothesis, which states that conceptual nodes belonging to the same form must be connected, but also by incorporating their prior semantic knowledge. Current automatic methods ignore this knowledge, which harms both accuracy and efficiency by enlarging the search space of candidate graphs. In this paper, we fuse the distributional semantic knowledge hidden in the internal representations of language models into a top-down, data-driven algorithm. This expert-like behavior re-sorts the candidates in a huge search space, guided by conceptual semantics similarity. Experiments on a self-collected benchmark of 10 datasets and comprehensive metrics show that our method builds SMMs both effectively and efficiently. 

---
# Generation-Heavy Hybrid Machine Translation 

**Authors**: Nizar Yahya Habash  

**Link**: [PDF](https://arxiv.org/pdf/2610.25107)  

**Abstract**: The state of the art techniques in Machine Translation (MT) require large amounts of symmetric resources from source and target languages. This is true regardless of whether the approach is Transfer or Interlingua, Symbolic or Statistical. Symmetry within these approaches is necessary to ensure quality, robustness and retargetability. In reality, such symmetry, whether it is in terms of structural transfer lexicons, interlingual dictionaries or parallel corpora, is a major bottleneck in developing any MT system.
This dissertation presents an approach to MT that addresses the lack of symmetry by exploiting symbolic and statistical target language resources in source-poor/target-rich language pairs. This approach is called Generation-Heavy Hybrid Machine Translation (GHMT). Expected source language resources include a syntactic parser and a simple one-to-many translation dictionary. No transfer rules or complex interlingual representations are used. Rich target language symbolic resources are used to overgenerate multiple structural variations from a target-glossed syntactic dependency representation of source language sentences. Statistical target-language resource are then used to select amongst the overgenerated translations. The source-target asymmetry of systems developed in this approach makes them more easily retargetable to new source languages. 

---
# Deep Fusion Memory: A Chunk-Level Parametric Memory for Large Language Models 

**Authors**: Rubin Wei, Siyuan Sun, Hao Sun, Yuliang Liu, Xiang Shi, Jiarui Wang, Junming Zhang, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25100)  

**Abstract**: Large language models need effective ways to acquire and use external knowledge beyond what they retain from pretraining. Continued pretraining risks catastrophic forgetting, while retrieval at inference time can add substantial overhead. External parametric memory avoids online retrieval, but constructing its training data requires extensive offline search, and output-level interpolation provides only shallow fusion. We introduce Deep Fusion Memory (DFM), a parametric memory architecture that reduces this construction workload and incorporates learned memory into intermediate hidden states. A causal memory predictor learns to generate memory from the preceding context, using retrieved chunk embeddings from the corpus as supervision. The predicted memory is shared across tokens within each chunk and fused into their hidden states through gated cross-attention. Memory prediction and fusion are trained end to end with the backbone frozen, and inference requires no explicit retrieval. Compared with existing parametric memory approaches, which often construct token-level memory targets, DFM requires $32\times$ fewer retrieval queries and $2048\times$ fewer retrieved items to construct its training data. Across six QA tasks, DFM performs better over the strongest baseline by 12.71% on Mistral-7B-v0.3 and 11.19% on Qwen3-8B-Base. Beyond QA, domain adaptation yields average gains of 6.51 and 4.38 points over Qwen3-4B-Base and Qwen3-8B-Base, respectively. 

---
# Beyond Factual Accuracy: A Benchmark for Evaluating Perspective Alignment in LLMs 

**Authors**: Sunghoon Kim, Sangyeop Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.25087)  

**Abstract**: Large Language Models often generate factually correct responses that fail to maintain user-requested perspectives, blending multiple perspectives and forcing users to filter through unwanted information---a perspective control failure that factual accuracy metrics cannot detect. We introduce PerspectiveBench, a benchmark quantifying perspective alignment by constructing scenarios where multiple valid reasoning paths coexist, and propose the Perspective Alignment Score (PAS) to measure adherence to specified perspectives. Experiments reveal that state-of-the-art models successfully include target content but fail to exclude non-requested perspectives. Providing documents through RAG improves factual grounding but does not resolve perspective drift, as models prioritize comprehensive coverage over strict user constraints. Code and data are available at this https URL. 

---
# Dual-Slot: From Serial Loops to Parallel Slots in Language Models 

**Authors**: Yiqin Hao, Zitong Wang, Boyi Zeng, He Li, Shixiang Song, Feichen Song, Yifan Liu, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25086)  

**Abstract**: Looped language models increase computational depth by repeatedly applying shared layers, offering a way to improve model quality without adding parameters. However, sequential dependencies between loop iterations increase autoregressive decoding latency. We introduce Dual-Slot, an architecture that reorganizes serial looped computation into parallel latent and prediction slots. Rather than waiting for the current token's latent computation to finish, the prediction slot combines the previous token's deep latent state with the current representation, while the latent slot produces a state for the next decoding step. This shifts the latent dependency across adjacent tokens, replacing two sequential module invocations with a single paired invocation while preserving standard autoregressive generation. To preserve token-level parallelism during training, we use a small number of Jacobi iterations to refine latent states across the sequence in parallel. Experiments on Qwen-style dense and mixture-of-experts (MoE) models show consistent improvements in language modeling and average downstream accuracy. On the 1B dense model, Dual-Slot reduces Pile validation perplexity from 7.25 to 7.00 and improves average five-shot accuracy across nine task configurations by 2.26 percentage points. After supervised fine-tuning, it improves zero-shot GSM8K accuracy by 5.83 and 3.49 percentage points for the 1B dense and A0.6B MoE models, respectively. Across the evaluated batch sizes, Dual-Slot retains 93.1-96.5% of vanilla decoding throughput for the dense model and 98.1-99.6% for the MoE model. 

---
# P2N: Reusing Deep Representations for Greater Effective Depth 

**Authors**: Yiqin Hao, Boyi Zeng, Zitong Wang, He Li, Shixiang Song, Feichen Song, Yifan Liu, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25076)  

**Abstract**: Mathematical problem solving and autonomous agents are increasingly important applications of language models, demanding stronger reasoning capabilities. Computational depth is a key resource for such reasoning, but adding model layers increases both training and inference costs, while test-time scaling incurs substantial additional inference computation. We introduce P2N (Previous to Next), which reuses the previous token's deep representations, already available during generation, to increase effective computational depth with negligible decoding overhead. Specifically, P2N adds the previous token's deep hidden state to the current token's shallow representation, allowing subsequent layers to build on previously completed deep computation. This recurrence extends effective computational depth across tokens, while introducing only a single vector addition per decoding step. We train the recurrence using parallel Jacobi updates restricted to a middle-layer core. Experiments with Qwen-style models show that, at 1.5B parameters, P2N improves GSM8K accuracy by over 10 percentage points under both zero-shot and few-shot prompting, with nearly unchanged decoding latency. The benefits also extend to language modeling and average performance on general downstream benchmarks. on general downstream benchmarks. 

---
# One Head Away from Language Confusion: Single Attention Heads Across Model Families 

**Authors**: Jiwon Jeong, Anjelo Jann Laroza, Chaewoon Yoon, Arjun Pillai, Pranav Kokati, Christian Hoang, Seunghyeok Hong  

**Link**: [PDF](https://arxiv.org/pdf/2610.25026)  

**Abstract**: Large language models sometimes reply in an unexpected language, a failure known as language confusion. We ablate individual attention heads in 12 model checkpoints spanning seven model series, using FLORES-200 continuations in five languages and Language Confusion Benchmark (LCB) chat replies. In ten models, zeroing one head causes substantially more departures from the expected language than zeroing any other head in its layer; its LCB pass-rate drop persists, though reduced, under mean ablation. Seven heads move 8.0 to 51.5 percent of all FLORES continuations out of the prompt's language, versus at most 1.4 percent for other same-layer heads; the other three mainly affect explicitly requested languages. These seven heads matter less in base models given plain text. In two models, replacing the values a head reads at the requested-language name with another language's mean makes replies fail and shifts some into that language, unlike a random change of equal size. Replacing a head's output with another language's mean redirects 84 to 99 percent of scored non-English monolingual LCB replies into that language in five models. Yet the requested language's mean improves crosslingual replies beyond every control head only in the two Gemma models, whose baselines are low. Code and results are available in the public repository at this https URL. 

---
# When Does Better Acceptance Pay? A Controlled Study of Speculative Decoding 

**Authors**: Yongchao Long, Chunrui Zheng, Yuxuan Zhou, Guangkun Nie, Tian-Chi Ma, Chao Pang, Shuohuan Wang, Yuchen Ding, Xianbin Wen, Dianhai Yu, Junyuan Shang, Yuxi Zhou, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2610.24993)  

**Abstract**: The throughput gains of speculative decoding depend on the trade-off between acceptance length and execution latency. We ask when additional acceptance is worth the cost of obtaining it. We establish a comparison protocol spanning training, inference, and evaluation, remeasuring released checkpoints and retraining six drafter recipes on shared target-generated data across ten benchmarks and two target models. Alignment changes the picture: gaps that appear large in published comparisons shrink substantially under shared training and inference conditions. This narrowing motivates two deeper questions: how is acceptance improved at different costs, and when do these gains improve throughput? We find that acceptance can be improved through nearly free objective and context changes or more expensive token-to-token conditioning. DCore illustrates the former route as a strong baseline with little additional measured drafting overhead. The payoff is conditional: task uncertainty and serving conditions affect acceptance returns, execution costs, and the preferred drafter and budget. These findings inform future drafter design and deployment choices. 

---
# The World as Viewed by 136 Countries: An Open, Curated, Annotated, and Multilingual Dataset of 118 Million State Media Articles 

**Authors**: Bryan Li, Samar Haider, Grace Kim, Fiona Luo, Jinqi Luo, Yuan Yuan, Duncan Watts, René Vidal, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2610.24990)  

**Abstract**: State media outlets often project their governments' narratives to target both domestic and foreign audiences. Because this content is plentiful and free to access, it appears widely in the web corpora that large language models (LLMs) are trained on and that retrieval systems draw from. However, the prevalence of state media in these corpora and its potential influence on LLM outputs remain largely unexplored. In this paper, we introduce WorldStateMedia, an open dataset of 118 million state media articles from 136 countries in 114 languages. We extract these articles from existing pretraining corpora and annotate each outlet with level of state control from State Media Monitor. Using WorldStateMedia to audit these corpora, we find that, in English web crawls, the share of state media pages from editorially controlled (vs. independent) outlets rose from 39% in 2013 to 93% in 2025. In several low-resource languages, state media make up a substantial share of all web data, such as 17.8% for Uyghur and 7.8% for Belarusian. To characterize the differing narratives of news outlets, we compare Russian and Ukrainian state media in English and in their native languages, revealing asymmetric topical focus and opposing framings of contested issues such as Crimea. Furthermore, in a retrieval-augmented generation (RAG) experiment on territorial disputes, retrieving a country's state media shifts model answers toward that country's claims, most strongly for English-language (foreign-targeted) media. We release WorldStateMedia to support multilingual, cross-national insights on state media at this https URL 

---
# A Measurement Perspective on Benchmarking LLM-Generated Text Detectors using Item Response Theory 

**Authors**: Yanbin Zhao, Zhipeng Chen, Guangsheng Bao, Guandong Zhang, Yan Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.24972)  

**Abstract**: Benchmarking large language model (LLM)-generated text detectors commonly relies on aggregate detector-level metrics such as accuracy, F1 score, and AUC. Although useful for summarizing overall performance, these metrics provide limited insight into how benchmark composition shapes detector comparisons or which texts distinguish detectors with different response profiles. We propose a measurement-based benchmarking framework for jointly characterizing benchmark heterogeneity and detector behavior using multidimensional item response theory (IRT). By treating benchmark texts as measurement items and detectors as respondents, the framework estimates text-level discrimination and difficulty-location parameters alongside multidimensional detector abilities. We apply the framework to a benchmark of 32K human-written and machine-generated texts across four domains, evaluated by 14 detectors. The estimated IRT item-parameter space reveals distinct but overlapping human-machine distributions, with substantial variation across domains and text lengths. The human-machine label shows the strongest multivariate association with this space, while domain and text length further contribute to benchmark heterogeneity. Detector ability profiles reveal multidimensional and domain-dependent response patterns that are not fully reflected in aggregate performance rankings. As a practical application, clustering texts in the estimated IRT space yields a compact subset of 2,088 texts (6.5\% of the full 32K benchmark) that largely preserves the comparative detector-performance patterns observed on the full benchmark. Overall, the proposed framework provides a principled and interpretable methodology for benchmarking heterogeneous detection technologies while jointly characterizing the measurement properties of the texts used for their evaluation. 

---
# Rubrics Carry the Standard: Shapley Attribution for Checklist-Based LLM Judges 

**Authors**: Saeid Asgari, Rui Ying, Sara Malvar, Rakshanda Agarwal, Bruce Sun, Claire Wang, Leonardo de Oliveira Nunes, Ranveer Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2610.24971)  

**Abstract**: Checklist rubrics applied by large language model (LLM) judges have become a standard instrument for evaluating open-ended outputs and for rewarding them during training. Two assumptions surround this practice: that leniency or strictness is a property of the judge model, and that a rubric can be improved by asking a model to review it. We test both in a controlled grading task with planted flaws and on two human-labeled datasets, SummEdits and RAGTruth. Holding the judge and the outputs fixed, rubric content moves the pass rate of correct outputs from 2% to 100%, whereas changing the judge model or the rendering of the same checks moves it by a few points. Model self-review then drifts the rubric toward the reviser's own standard: it adds requirements the task never stated and deletes checks that human labels had taught, with three reviser models and on all three datasets. To measure and control this, we show that under all-must-pass aggregation the Shapley value of each check has an exact closed form, which yields per-check catch credit, false-fail blame and unique catches in linear time. Attribution shows that self-review deletes the checks that do unique work and keeps the checks that cause false fails. A change-control gate that combines these quantities with a provenance freeze on label-derived checks holds nearly every drifting revision chain at its starting quality, with identical decisions under three judge models. 

---
# DeGMatch: Target-Aware Reasoning Data Selection via Deconfounded Gradient Matching 

**Authors**: Changho Lee, Janghoon Han, Rajdeep Mukherjee, Yeda Song, Lening Nick Cui, Xinya Du, Honglak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.24963)  

**Abstract**: Reasoning post-training on long trajectories has substantially improved the reasoning capabilities of large language models, but scaling such training to large datasets is increasingly expensive. We study target-aware reasoning data selection, where a small reference set specifies a target capability and guides the selection of training trajectories for specialization. We find that existing target-aware methods can exhibit limited target discrimination on long reasoning trajectories, while dominant population-level structure in candidate gradients can obscure finer target-dependent variation during gradient matching. To address these limitations, we introduce Deconfounded Gradient Matching (DeGMatch), which first constructs a target-relevance gate and then reranks candidates using model-specific gradients. DeGMatch centers gradients using the candidate-population mean and removes leading principal directions before target matching, reducing the influence of dominant population-level structure and improving target discrimination. Across mathematics, science, and code, DeGMatch outperforms strong heuristic, target-aware, and reasoning-specific data selection baselines, while remaining effective across selection budgets, model families, and scales. Our code is publicly available at this https URL. 

---
# Over-Defense in Coding Agents Begins at Silent Forks 

**Authors**: Hanjun Luo, Xuecheng Liu, Xiucheng Zhang, Zhuoning Xu, Yingbin Jin, Yankai Chen, Xue Liu, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2610.24949)  

**Abstract**: Language models behind coding agents are post-trained with outcome rewards, and recent methods concentrate credit on the turns that decide the outcome. We find that defensive actions beyond what a task requires escape both. They begin at silent forks, the turns where every next action leads to the same outcome. Such actions leave the success rate unchanged, and frontier models take the disproportionate branch there in up to 91% of decisive actions under implicit budgets. The most direct fix, a checker in the trajectory reward, reaches these forks only through fully proportionate trajectories, leaving 74% of GRPO groups with identical rewards and behavior unchanged. This motivates a rethink of where learning signals belong, namely where behavior diverges, which a behavioral checker can locate and the outcome cannot. Our method, ForkDPO, contrasts a proportionate and a disproportionate action at the same prefix, at silent forks on both sides of the required behavior. On a test set of unseen repositories, it lowers a 4B model's disproportionate behavior from 70.3% to 58.9%, against 67.7% for DPO at randomly chosen turns, establishing behavioral divergence as a learning signal for how agents act beyond whether they finish. The code is publicly available at this https URL. 

---
# Epi: Towards a Persistent, Self-Improving Agent for Autonomous Research 

**Authors**: Yue Huang, Zhangchen Xu, Xiaonan Luo, Zheyuan Liu, Lichi Li, Yiping Wang, Peiyang Song, Eric Gabriel Ramos, Enze Xie, Olexandr Isayev, Paul Liang, Nouha Dziri, Huan Sun, Jianfeng Gao, Ying Liu, Misha Sra, Xiangliang Zhang, Alex Pentland, Zichen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.24944)  

**Abstract**: We present Épi, an autonomous research agent that sustains investigations across model sessions and evolves its methods through accumulated experience. Épi combines persistent research history, hierarchical memory, branching and independent review to maintain research continuity, learn from experiments, and explore and revisit alternative approaches. Its experiments run across session boundaries, and a fixed verifier evaluates their outputs. Épi produced 120 record-setting results, comprising 20 numerical bound records across 12 families and 100 construction records across eight families. These include an approximately 84-fold narrower interval for the $n$-queens constant and a more than 3.6-million-fold narrower rigorous interval for the quartic Feigenbaum attractor dimension. Exact dual certificates establish that the Cohn-Elkies linear-programming bound cannot be sharp in dimensions 10 and 11. Beyond numerical improvements, Épi gives a computer-assisted proof of Tuza's conjecture for split graphs with an eight-vertex clique part and at most three triangle-active neighborhood types, with arbitrary multiplicities. We extend this research process to algorithmic coding with Épi-turbo, a lightweight configuration that develops executable solvers for tasks including vehicle routing, scheduling and production planning. On 30 FrontierOR tasks, Épi-turbo achieves a 5.4% higher total benchmark score than Codex at 64% of its model cost, using the same underlying model and reasoning effort. Épi is designed to support human understanding by connecting computational results to explicit methods and mathematical arguments that researchers can examine and build upon. 

---
# What Can't Jev Decide? Probing the Limits of One-Pass Decision Models 

**Authors**: Wang Yang, Xinpeng Li, Yanyan Zhang, Chuang Ma, Shouren Wang, Jianzhe Wei, Liqin Ye, Xiaoran Cai, Mohsen Hariri, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.24929)  

**Abstract**: Jev and its open replicas choose among the options a caller supplies with a single forward pass, fast enough to decide inside an agent loop. We ask what such a model cannot decide, and find a simple limit: Jev cannot make a decision that an LLM makes only by writing out intermediate steps. On a benchmark that varies one factor at a time, Jev finds and selects records without error up to 20k tokens and 64 options, but once the answer depends on two or more updates of one user, accuracy falls, to 53-80% at sixteen updates, because similar records interfere. Given a stale summary of the state, Jev follows it. On real decision items graded by how many thinking tokens a reference LLM needs, Jev answers 97.5% of the items that one forward pass of that LLM solves, but only 74% of the others. Jev also does not know its limit: its confidence flags the errors it makes itself but not those it inherits from a stale input, and neither an explicit nor a trained hand-off option catches the latter. The open models we evaluate share these limits. Checking the state such models rely on requires a model that reasons. 

---
# How Well Can LLMs Verbalize Mathematical Formulas? A Multi-Domain Evaluation 

**Authors**: Peter Volbach, Tim Schlippe  

**Link**: [PDF](https://arxiv.org/pdf/2610.24917)  

**Abstract**: Large Language Models (LLMs) have shown strong capabilities across natural language tasks. But their ability to generate accurate written verbalizations of mathematical formulas has not been systematically evaluated. Such verbalization is important for text-to-speech systems supporting audio-based learning and visually impaired users. We evaluated five state-of-the-art LLMs across six scientific domains and constructed Read-Aloud Math, a balanced multi-domain corpus with LaTeX formulas, manually validated simple and extended transcriptions, multiple variants, textual context, domain labels, and formula complexity levels. Using multi-reference BLEU, GPT-5.2 achieved the highest score for simple transcriptions (75.9% without context, 64.9% with context), although without context it did not significantly outperform Grok 4.1 Fast or Claude Sonnet 4.5. Claude Sonnet 4.5 achieved the highest score for extended transcriptions without context (36.5%), and Grok 4.1 Fast with context (48.8%). For extended transcriptions, Statistics and Mathematics scored highest across the two context settings, while Physics scored lowest. Formula length was significantly negatively associated with BLEU for simple transcriptions without context, but not with context, whereas BLEU increased significantly with formula length for extended transcriptions in both settings. Grouping symbols were most frequently involved in transcription differences, while variable names, capitalization, and Greek letters were associated with lower BLEU scores. Overall, LLM-based formula verbalization is promising but remains sensitive to transcription type, textual context, and formula characteristics. 

---
# Compositional Activation Steering Across Languages and Behaviors with Contextual Gates 

**Authors**: Daniil Gurgurov, Akansh Maurya, Josef van Genabith, Simon Ostermann, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2610.24906)  

**Abstract**: Activation steering has emerged as an effective and precise way to influence the behaviours of language models. However, composing multiple steering directions to control several behaviours simultaneously remains challenging, particularly in multilingual settings and with training-free steering vectors. We address this challenge with contextual steering, jointly training gates that adapt the strength of fixed steering directions to the current hidden state and interactions between interventions. Across a broad range of languages, behavioural attributes, and models, our approach outperforms constant-strength steering and four strong baselines, including prompting. It also better preserves general knowledge and keeps the model's internal computation closer to that of the prompted model. Our analysis further shows how jointly trained gates adapt their contributions during composition, explaining the mechanisms underlying effective compositional steering. 

---
# LastWords: Boring Problems Revisited 

**Authors**: Huayang Li, Richard Sproat  

**Link**: [PDF](https://arxiv.org/pdf/2610.24894)  

**Abstract**: In 2022, the second author published a Last Words piece in Computational Linguistics in which he pointed out that, despite the advances in AI up to that point, there were still some problems in natural language processing that on the one hand seemed trivial and perhaps boring, but on the other were hard to solve using purely neural methods. One such problem discussed in that context was text normalization, and a particular subproblem of that, namely number name reading. The piece came out just a few months before the first release of ChatGPT, and needless to say, a lot has happened since then. Indeed, as we document here, the problem (of number name reading) has practically been solved, at least by one of the latest frontier models. However, we believe that there are still open issues for such limited domain, constrained and limited resource problems, and that in some future machine learning approach, the massive resources of the latest frontier models would not be needed to solve them. 

---
# Measuring Data Diversity with Embeddings: A Taxonomy and Benchmark for Diversity Measures 

**Authors**: Cantao Su, Anna Wegmann, Esther Ploeger, Dong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2610.24888)  

**Abstract**: Diversity is a critical property of text corpora, with direct implications for training data curation, test set design, and model generalization. However, its measurement remains fragmented and poorly understood. Existing embedding-based diversity measures vary widely in how they are computed and can yield substantially different diversity scores on the same data, yet no systematic comparison of their empirical behaviour exists. We make two contributions toward addressing this gap. First, we present a taxonomy of 21 embedding-based diversity measures, organizing them into four families based on their underlying computations. Second, we introduce EmbDivBench, a benchmark that evaluates these measures along three conceptually distinct components (variety, balance and disparity) on both simulated and natural text data. Although Mean Pairwise Distance is the most commonly used measure, our experiments show that it is outperformed by several alternatives on the three diversity components, with Graph Entropy being the best choice overall. Lastly, we distil these findings into practical recommendations for diversity measure selection. 

---
# Learning to Program Retrievals: Composing Search Primitives via Reinforcement Learning 

**Authors**: Kyuyoung Kim, Zilin Xiao, Zhiyuan Wang, Avinash Atreya, Ke Yang, Jinwoo Shin, Kecheng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2610.24854)  

**Abstract**: Retrieval-augmented language models increasingly use multi-step search to solve knowledge-intensive tasks, yet existing systems typically expose retrieval through fixed pipelines or collections of atomic tools. We introduce Retrieval Program Learning (RPL), a programmable search framework that allows a policy to compose search primitives into executable programs with local state and control flow. This enables adaptive search that can retrieve broadly, process intermediate results, and selectively construct model context without tying retrieval breadth to context consumption. However, the resulting flexibility creates a difficult exploration problem: useful programs may require several coordinated decisions, while reinforcement learning often supplies only sparse feedback. To address this challenge, we propose program mutation, a grammar-constrained and execution-verified procedure that generates local variants of suboptimal programs sampled during RL, converts them into training trajectories, and distills them back into the policy. We evaluate RPL on open-domain question answering and corpus-level reasoning tasks, comparing it with a range of retrieval and reasoning baselines. RPL consistently improves open-domain QA performance across model sizes and in- and out-of-domain benchmarks, with particularly large gains on multi-hop tasks. Notably, a 4B model trained with RPL outperforms all 8B model baselines. On corpus-level reasoning, RPL achieves the highest answer F1 and document-coverage F1 for both model sizes while substantially reducing context consumption and retrieval turns. Our results show that programmable search expands the retrieval strategies that agents can learn, while program mutation provides an effective mechanism for exploring useful program structures beyond direct on-policy sampling. 

---
# MisleadQA: Diagnosing LLM Failures under Misleading Context 

**Authors**: Hui Wang, Bohao Chu, Qianli Wang, Shuning Zhang, Norbert Fuhr  

**Link**: [PDF](https://arxiv.org/pdf/2610.24851)  

**Abstract**: Large language models (LLMs) are widely used to answer questions from retrieved or user-provided context. However, provided context can be factually true yet out-of-scope, describing a different entity, time, or location from the one the question targets, and still lead models to a wrong answer. Prior work has mainly studied factually false or irrelevant context. It remains unclear whether such out-of-scope context misleads models as much as false context, and whether adding gold context prevents models from being misled by either. We introduce **MisleadQA**, a diagnostic dataset of multiple-choice questions with an abstention option. Each question pairs gold context (G) with true but out-of-scope context (S) and false but in-scope context (F) supporting the same wrong answer. We evaluate six open-weight LLMs. On questions they answer correctly without context, providing S alone leads them to switch to this wrong answer in 29.3% of cases, and providing F alone does so in 75.4%. Adding G nearly removes the effect of S, but F still misleads models in 21.0% of cases, especially when F comes first. A context-faithful instruction reduces these errors mainly through abstention and lowers accuracy even with G. These findings suggest that evaluations should check whether context applies to the question, not only whether it is true, and should report correct, misled, and abstention rates together. Data and code are available at [this https URL](this https URL). 

---
# Efficiency Measured in English: Tokeniser Coverage Bounds Small Language Models on Indic Scripts 

**Authors**: Anna Pramod  

**Link**: [PDF](https://arxiv.org/pdf/2610.24818)  

**Abstract**: Small Language Models with quantisation and parameter-efficient adaptation are widely offered as the route to multilingual AI under compute constraints, and their efficiency is almost always reported as a single memory figure. We show that this figure is measured in English and does not transfer. Across a 131k-sentence corpus spanning eleven Indic scripts, TinyLlama-1.1B's tokeniser emits 1.34 subword tokens per Latin-script word and 22.60 per Kannada word. This is not inefficiency; it is absence. Its vocabulary contains 0% of Odia's codepoints, 1% of Telugu's and 2% of Kannada's, and for five of eleven scripts the measured fertility sits at the UTF-8 byte-fallback ceiling, on Odia at 99.6% of it, with 93.6% of emitted tokens being raw bytes. Because SentencePiece byte fallback never raises an unknown-token marker, this failure is entirely silent: TinyLlama's UNK rate is 0.0% on every script, including the one it reads as bytes. We further show that low fertility can indicate the opposite of efficiency: mBERT posts its best score anywhere (1.10 tokens/word, on Odia) while discarding 87.9% of the text as UNK. MuRIL, an Indic-vocabulary encoder with a fifth of TinyLlama's parameters, reads every script at 1.23-2.26 tokens/word with zero UNK and zero byte fallback. The binding constraint is the vocabulary, not the parameter count. The consequence is concrete: on a 2048-token context window, the same model gives an English user 1,528 words of usable context and a Kannada user 91. 

---
# Ask, Don't Guess: Targeted Clarification for Speech Recognition Errors 

**Authors**: Sankalp Parashar, Preethi Jyothi, Sunita Sarawagi  

**Link**: [PDF](https://arxiv.org/pdf/2610.24809)  

**Abstract**: Speech recognition in real time in far-field conversational settings still incurs significant errors. Identifying potential error spans during streaming ASR permits a recourse mechanism where the speaker can be asked to repeat the affected content under cleaner acoustic conditions. We propose \sysname, a new framework for online error localization, that identifies spans likely to benefit from this intervention. Our supervision uses clean audio to estimate how the recognizer's output might change under improved acoustic conditions. We probabilistically align noisy and clean decodes using a pair-HMM to obtain soft supervision. These soft targets train a counterfactual gain scorer by combining a suite of acoustic and language features to produce token-level scores, that are further grouped into spans for targeted recourse. An LLM uses the ASR transcript of the repetition to revise only the selected span. At a precision of $70\%$, \sysname achieves the highest recall by identifying up to 52\% more erroneous tokens compared to the strongest baseline. Targeted repetition yields relative WER reductions of up to 24\% under a budget that accounts for both interruption frequency and the lengths of re-uttered spans. 

---
# Table Serialization Determines Which Relations Language Models Preserve 

**Authors**: Junwei Ma, Arun Kumar Chithanar, Palash Goyal, Chul Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.24798)  

**Abstract**: We study how text serialization affects table lookup in instruction-tuned language models. Our diagnostic asks the model to find the row containing a specified value under one named column, then return the value under another named column in that same row. We render the same synthetic 256-cell tables in row-major form and in column-major form with each header placed beside its values, while varying the table aspect ratio across five shapes. Across three models, the better format changes with table shape: row-major performs better on tall tables, whereas column-major performs better on wide tables. At 64 x 4, row-major leads by 42.3-54.0 percentage points; at 4 x 64, column-major leads by 35.0-59.0 points. Mapping wrong answers back to their source cells reveals a complementary error pattern: row-major tends to preserve the correct row but lose the target column, while column-major tends to preserve the correct column but lose the row. We use the term relational locality to describe this pattern: serialization keeps one relation locally available while making the other depend on positional alignment. A separate non-tabular ordering experiment supports this interpretation. Thus, apparent width-related failures in text-serialized tables depend on linearization, not table shape alone. 

---
# Who Gets Unmasked? Sampler-Induced Bias in Diffusion Language Models 

**Authors**: Xujun Che, Farsheed Haque, Depeng Xu, Xi Niu, Haopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24795)  

**Abstract**: Diffusion language models (dLLMs) generate by gradually unmasking a canvas, so their samplers decide not only which token to write but also when to commit it. We show that this second decision changes who appears in generated text, a sampler-induced bias on top of the model's own. Common commit rules, including LLaDA's default, commit the positions whose sampled tokens are most probable; this peeking favors majority options. To separate the sampler from the model, we read the model's belief about a fork, such as a protagonist's gender, in one forward pass, and show that this belief matches exact sampling from the model. Measured against it, peeking samplers on LLaDA and Dream amplify occupational gender stereotypes and collapse opinion distributions, and confidence-based ordering shifts gender-balanced occupations toward men, more so under faster parallel decoding. We propose Fork-First Sampling, which decides the fork before anything else is written: it follows the belief or any target distribution, improves fluency, and avoids the stories that remark on the protagonist's gender, which prompt-based fixes produce, at the cost of one forward pass. Fairness evaluations of dLLMs should report the commit rule. 

---
# ECHO: Efficient Causal Self-Recursion Beyond Auxiliary Multi-Token Prediction 

**Authors**: Yuxuan Zhou, Junyuan Shang, Tianyu Wang, Chunrui Zheng, Zichang Liu, Wenbin Chang, Shuohuan Wang, Dianhai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.24766)  

**Abstract**: Existing Multi-token prediction (MTP) methods are auxiliary by design: the multi-horizon loss mostly trains modules attached to the backbone, and each added horizon costs another module and another pass. We propose Efficient Causal Self-Recursion (ECHO), which removes both costs. The backbone's own final layer predicts every horizon, so the supervision lands on the parameters that generate the next token, and Dynamic Depth Sampling (DDS) keeps deeper horizons affordable by drawing the training depth with probabilities that decay along depth. Future prediction thus becomes part of the autoregressive objective rather than a task attached to it, and the gains follow on both sides of it. In pretraining, added depth converts into downstream accuracy where auxiliary modules saturate, concentrating on reasoning and code at half the MTP training overhead of training every depth. At inference, the pretrained model is its own drafter: it drafts beyond the depth it was trained for, and keeps its draft-target alignment on the open-ended, high-entropy generation where post-hoc drafters lose theirs. The advantage survives a doubled token budget, a repeated pretraining run, and supervised finetuning. 

---
# BanglaBBQ: A Bias Benchmark for Question Answering in the Bangladeshi Context 

**Authors**: Sumit Anthony Gomes, Yasin Rahman, Taeeba Tasnia, Faiza Nooren Suhita  

**Link**: [PDF](https://arxiv.org/pdf/2610.24764)  

**Abstract**: This paper introduces BanglaBBQ, a question-answering bias benchmark designed to capture stereotypes grounded in the Bangladeshi context, in both Bangla and English. Following KoBBQ and PakBBQ, BanglaBBQ adapts the original Bias Benchmark for Question Answering (BBQ) by classifying its templates into three groups: directly translated (DT), modified to align target groups or contexts with Bangladesh (TM), and removed when deemed unsuitable for the Bangladeshi context (SR). This process retains 140 templates across six categories, comprising 76 DT and 64 TM templates. Five newly authored (NA) categories, based on stereotypes common in Bangladeshi society, are then added: marital status, age and political perception, regional origin, educational background, and mental-health status. Native Bangladeshi annotators judged all 466 newly authored templates, and 372 were retained because a strict majority of English annotators identified the intended target and rated the attitude as common enough. Each newly authored template is linked to a documented source and a verbatim quote from it. Each evaluated template also has a counter-stereotypical version of its disambiguated context. The dataset was evaluated using two open-weight models (Qwen3-4B and Qwen3-8B) and two closed models (GPT-4.1-mini and Gemini 2.5 Flash). All four lean towards the stereotyped person when the context is ambiguous, in both languages, and the effect of language differs by model, so English results do not predict behaviour in Bangla. On the adapted BBQ templates, both Qwen3 models are also less accurate when the context contradicts a stereotype than when it confirms it. On the newly authored templates, the closed models follow counter-stereotypical evidence almost perfectly, and all four models are more biased on the templates that annotators validated than on those they rejected. 

---
# GLOSS-MT: Controllable Low-Resource Machine Translation with an Editable Dictionary 

**Authors**: Mich-Seth Owusu, Stephen E. Moore, Akwasi Asare, Lawrence Adu Gyamfi, Paul Azunre, Joel Budu, Jonathan Asiamah  

**Link**: [PDF](https://arxiv.org/pdf/2610.24730)  

**Abstract**: Adapting a pretrained machine translation (MT) model to new terminology usually requires new parallel data and another round of fine-tuning, which is costly for low-resource languages. We ask whether controllability can be added to MT, and the fine-tuning burden reduced, without degrading the pretrained model. GLOSS-MT looks up the source's noun phrases in an offline English-Twi dictionary, replaces every phrase found by its Twi translation in place, and fine-tunes a translation model once to translate these mixed inputs. We apply the same data and pipeline to NLLB-200 (600M, 1.3B, 3.3B), OPUS-MT en-tw and TranslateGemma 4B. On a document-level English-to-Asante Twi benchmark (84 documents, four domains), every fine-tune scores above its pretrained base in chrF2 (+2.85 to +5.48), and the three NLLB fine-tunes are the best open systems (50.12-50.80). A direct controllability test on human-translated Bible passages shows that the three fine-tunes tested write 79-97% of the dictionary terms, and that replacing every dictionary entry by a different term changes the output accordingly (70-97% of the new terms appear) with no retraining. We release the models, data and code. 

---
# PV-Decoding: Progressive Verification Decoding with Decode Gating for Diffusion Language Models 

**Authors**: Renxi Liu, Yiming Li, Shengyin Sun, Chen Chen, Xian Wang, Shuai Yang, Huiling Zhen, Xianzhi Yu, Yaoyuan Wang, Weizhe Lin, Xiaosong Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.24722)  

**Abstract**: Discrete diffusion language models (dLLMs) offer an attractive alternative to autoregressive generation by updating multiple tokens in parallel, but their inference efficiency is still limited by the cost of safely committing tentative predictions. In particular, speculative decoding for dLLMs is challenging because speculative tokens within the same diffusion block may depend on one another, making aggressive acceptance prone to error propagation and strict verification computationally expensive. We present PV-Decoding, a confidence-driven speculative decoding framework with progressive verification for block diffusion inference. PV-Decoding partitions masked positions into directly decoded tokens, speculative tokens, and unresolved tokens using two confidence thresholds, and then verifies speculative tokens with progressively stronger policies, including single-batch, dual-batch, and multi-batch verification. This design exposes a practical accuracy--efficiency trade-off: weaker verification minimizes overhead, whereas stronger verification improves acceptance precision and robustness. Experiments on three diffusion language models across GSM8K, MBPP, and HumanEval show that PV-Decoding reduces sequential model forward passes (MFPs) by 8%--37% relative to confidence-based decoding, while keeping the largest observed accuracy drop to 1.3%. On Dream-7B-Instruct/GSM8K, PV-Decoding (1-batch) and PV-Decoding (2-batch) improve throughput by 7.0%--13.6% under the same hardware budget, whereas PV-Decoding (multi-batch) is slower under the current runtime. These results suggest that confidence-guided speculation with structured verification is a practical path toward efficient and reliable dLLM inference. 

---
# DiffuPlan: Editable Planning State for Diffusion Language Model Agents 

**Authors**: Qisheng Zhou, Zhen Xiong, Qiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2610.24717)  

**Abstract**: Diffusion language models (dLLMs) are emerging as an alternative to autoregressive (AR) language models, yet dLLM agents largely inherit an interaction protocol designed for AR generation: once intermediate state is written, later observations can only be incorporated by appending more text. This is a poor match to a defining capability of dLLMs -- reconstructing earlier text conditioned on context that appears both before and after it. We ask whether this capability can support a different agent-state paradigm in which intermediate planning state is directly revisable as evidence arrives. We introduce DiffuPlan, a diffusion search agent that maintains a single editable planning state and reconstructs only the affected steps when later evidence resolves an entity or changes the search route. To support variable-length revisions, we introduce Expansion-Partition Marginalization (EPM), which marginalizes over equivalent expansion paths. We further adapt diffusion trajectory RL to multi-turn search. Across four multi-hop QA benchmarks, DiffuPlan achieves 50.24 average F1, outperforming strong AR search agents. More importantly, under identical data and training, editable planning state improves the diffusion agent by 2.44 F1 versus only 0.47 for its AR source model. These results show that dLLMs need not imitate append-only AR agents: revising committed state is a distinct agentic advantage. 

---
# When Do Repetition Loops Lock? Truncation Sampling, Bounded Penalties, and a Blind Spot in DRY 

**Authors**: Yichuan Chen, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24715)  

**Abstract**: Language models sometimes repeat verbatim until the generation limit. We study the exit rather than the entrance: given that a span has already been repeated $n$ times, when is escape still possible? Along the loop the context is deterministic, so loop survival under any sampler can be computed exactly from one forward pass. Pure sampling leaves a positive escape probability at every step; top-$p$ with threshold $\pi>1/2$ locks a position exactly when the continuation probability reaches $\pi$. A bounded penalty can leave a position locked; a penalty that outgrows the model's preference for the loop over unseen tokens guarantees escape under top-$p$. The same analysis predicts a blind spot in the widely deployed DRY sampler: its sequence breakers cap the penalty at $\lambda b^{g-1-a}$ for loops whose longest breaker-free run has $g$ tokens. On eleven open models, 66\% of locked units with $g=6$ stay locked under DRY and 1\% with $g=8$. In free generation with this http URL, all 109 loops that survive DRY have $g\le 7$. We propose DRY-C, which adds a term that grows exponentially with the number of verbatim recurrences, up to its look-back window. With it, none of 720 structured-output generations looped to the length limit, and task completion stayed close to DRY's. 

---
# Do Conversational Agents Need Explicit Memory? A Controlled Comparison with Raw-History Retrieval 

**Authors**: Yongjie Zhou, Shuai Wang, Bevan Koopman, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2610.24671)  

**Abstract**: Explicit memory systems transform conversation history into persistent representations intended to improve future access, often at substantial construction cost. Yet it remains unclear whether this added complexity yields better answer accuracy than retrieving directly from raw history. This study compares six state-of-the-art explicit memory systems with raw-history RAG on LongMemEval-S and BEAM-100K under original and controlled retrieval settings. We find only localized gains for particular question types and tight run-time context budgets, with no consistent advantage over strong raw-history RAG baselines. Oracle retrieval shows that some systems fail to preserve answer-supporting information during explicit memory construction, while others preserve it but fail to retrieve it effectively, despite construction costs reaching hours per conversation.
Our findings suggest that while many existing memory systems for AI agents emphasize increasingly elaborate explicit memory construction, they are often evaluated without strong controlled comparisons to raw-history RAG and may underemphasize retrieval. Retrieval remains a major bottleneck when useful information is preserved, suggesting that improving access to existing information may be at least as important as constructing richer explicit memory. 

---
# What RL Brings to Reasoning: A Few Decisions the Base Model Would Not Make 

**Authors**: Siyu Zhang, Yuquan Lu, Yoshua Y. Li, Yifan Ding, Kun Zeng, Yu Huo, Yifu Guo, Rongxiang Weng, Xunliang Cai, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24663)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) substantially improves mathematical reasoning, but what it adds to the base model remains unclear. We investigate this through a sequence of generation-time interventions on Qwen3-4B-Base, measuring how much of its RL-trained counterpart's gain each intervention recovers on 120 AIME problems. Power sampling with the same 8,192-token output limit brings no gain beyond greedy decoding. Yet borrowing the RL model's token only where the two models' greedy choices differ, roughly 4% of generated positions, recovers 90.5% of the pass@1 gain, with similar recovery across three model pairs on MATH-500. We find that even a single opening token has a substantial effect: forcing the base model to begin with "Alright" recovers 35% of the RL gain. The opening alone is insufficient, however. Later replacements after the same opening add another 5.2 percentage points to pass@1, and the fixed rules we test for inserting reflective words such as "Wait" do not reproduce this benefit. Sparse replacement also retains much of the base model's diversity while staying close to the RL model at pass@1. At k=256, it solves 67.5% of the problems, significantly more than the RL model (61.4%) and a mixture of whole responses matched in pass@1 (61.45%). The comparison is at equal attempt counts rather than equal compute, since rl-replace consults both models at every decoding step. These results show that the opening can recover a meaningful part of the RL gain, but a larger contribution comes from later choices, which can be borrowed while retaining much of the base model's diversity. Code is available at this https URL. 

---
# Muse Glimmer 30B: An On-Device Agentic Model 

**Authors**: Arash Ebrahim Abtahizadeh, Muna Aghamelu, Adil Ahmed, Juan C. Angeles Acuna, Alexis Audran-Reiss, Asaf Avrahamy, Yoram Bachrach, Can Balioglu, Amar Budhiraja, Onur Çelebi, Gaurav Chaurasia, Joy Chen, Lynn Cherif, Lucia Cipolina-Kun, Gal Cohen, Timothée Darcet, Beto de Paola, Digant Desai, Sam Devlin, Derek Dunfield, Ivan Evtimov, Maryam Fazel-Zarandi, Rob Fergus, Jean-Christophe Gagnon-Audet, Grant Gardner, Bhavul Gauri, Paris Giampouras, Sujan Kumar Gonugondla, Antoine Grosnit, Young Han, Michael Hassid, Monuril Islam, Masoud Jalili Sabet, Songhao Jia, Narine Kokhlikyan, Svetlana Kurtsman, Sandra Lefdal, Chee Hau Leow, Noam Levi, Mike Lewis, Sanae Lotfi, Xiyang Luo, Despoina Magka, Saeed Mahloujifar, Neal Mangaokar, Ishita Mediratta, Alexander H. Miller, Hela Momand, Saba Nazir, Kelvin Niu, Mattia Opper, Parth Pathak, Alberto Pepe, Danielle Pintz, Mikhail Plekhanov, Sidhartha Reddy Potu, Anton Protopopov, Shengyi Qian, Shashank Rajput, Joseph Rocca, Scott Roy, Luca Sbordone, Amin Shakeri, Tatiana Shavrina, Michael Shvartsman, Ruan Silva, Anya Sims, Eric Michael Smith, Hugo Touvron, Mabel Ubong, Isabel Urrego, Yanzheng Xiang, Zhiyuan Yao, Alex Yarmula, He Ye, Jade Yu, Luke Zettlemoyer, Arman Zharmagambetov, Perpetua Fathima Ahlijah, Tremo Ahmed, Enrique Alfonseca, Pedram Bashiri, Nikolay Bashlykov, Aaron Berman, Rohith Bhatia, Siddharth Bhattacharya, Prashant Bhende, Chloe Bi, Shuchao Bi, Clement Anthonioz Blanc, Allen Bolourchi, Andrea Caciolai, Erica Cai, Xiang Tony Cao, Yue Cao, Andrew Caples, Jinzhao Chang, Lele Chen, Moya Chen, Weijie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.24628)  

**Abstract**: We introduce Muse Glimmer, an open-source 30B parameter causal language model with perception encoder, purpose-built for autonomous agentic tasks on consumer hardware. Muse Glimmer integrates multi-step reasoning, reliable tool use and failure recovery into a single model that runs locally on consumer devices without requiring cloud infrastructure. The model accepts text and image inputs and produces text outputs, was trained on data from over a hundred languages, and is quantized to approximately 4-bit precision for efficient single-GPU inference. Muse Glimmer achieves state-of-the-art performance for its size class on agentic benchmarks, setting a new standard compared to the models of its size: Gemma4-31B and Qwen3.6-27B on tool use, multi-step planning, and agentic task completion. We release the model under the Apache 2.0 license to foster open-source development of capable, safe, local AI agents. 

---
# Safety Kernel: Mechanistic Alignment for Cross-Lingual Defense in LLMs 

**Authors**: Chengyu Xie, Fei Shen, Shuyi Miao, Yue Feng, Jian Yu, Pengyang Shao, Yijun Tian, Xiaoyu Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.24614)  

**Abstract**: The safety capabilities of large language models (LLMs) are highly uneven across different languages, often allowing cross-lingual attacks to bypass safety guardrails. Existing alignment methods predominantly rely on external alignment signals and provide limited insight into underlying neuron-level mechanisms, making it difficult to provide a unified internal target for cross-lingual safety interventions. To address this, we analyze cross-lingual safety behavior at the neuron level and identify a compact set of safety-related neurons shared across languages, termed the safety kernel (SK). Specifically, we first utilize multilingual parallel safe and unsafe prompts to compute safety contrast with activation responsiveness (SCAR) scores, identifying language-wise safety-neuron candidates from internal activation patterns. We then intersect the resulting language-wise safety-neuron sets to obtain the SK, and validate its functional relevance to cross-lingual safety behavior through activation suppression and activation-state transfer interventions. Finally, building on the SK coordinates, we introduce SK-Masked LoRA, which constrains LoRA adaptation to the feed-forward network (FFN) output dimensions corresponding to the SK, thereby focusing safety-related changes on the cross-lingually shared neuron coordinates. Extensive experiments demonstrate that SK-Masked LoRA substantially reduces attack success rates across multiple languages, including languages excluded from SK discovery, while intervening on only a small fraction of internal neuron coordinates and producing generally limited changes in the model's general capabilities. 

---
# Training with Ciphers to Enhance In-Context Terminology Learning For Translation from Rare Languages 

**Authors**: Deevyanshu Malu, Deeptanshu Malu, Sunita Sarawagi  

**Link**: [PDF](https://arxiv.org/pdf/2610.24593)  

**Abstract**: Fine-tuning LLMs with in-context examples (IC-Train) lets translation models adapt to new parallel data at deployment, but it is unclear whether they actually learn the translations of rare terms from those examples. We formalize this capability as in-context term learning (ICTL). We show that standard IC-Train develops this capability only weakly, and propose RareCipherFT, which strengthens it without any lexical resources. RareCipherFT pairs each training target with in-context examples that share a rare term with it. It then replaces that term with a character-shuffled cipher in both the context and the reference, so the term can only be recovered by copying from context. The ciphered fraction of training data is annealed to preserve translation quality. Across seven translation directions between English and Tamil, Assamese, Mizo, Nepali and Hausa, and four models from 1B to 7B parameters, RareCipherFT raises term recall over standard IC-Train by 5-15 absolute points without lowering COMET. Probes using synonyms, nonce words and controls for shared surface context show that these gains come from linking source terms to their in-context translations rather than from shortcuts. 

---
# From Stories to Features: Scaling Concept Vectors for AI Alignment & Interpretability 

**Authors**: Dmitry Scherbakov, Oleg Chernikov, Victoria Khamash, Kirill Afendulev, Alexey Dontsov, Ivan Oseledets, Elena Tutubalina, Anton Korznikov  

**Link**: [PDF](https://arxiv.org/pdf/2610.24556)  

**Abstract**: We study feature vectors: directions in a language model's activation space that correspond to human-interpretable concepts, such as honesty or deception. Existing unsupervised methods like sparse autoencoders attempt to recover the whole feature space, often at the cost of feature quality and interpretability. We take a more pragmatic route: focusing on concepts relevant to AI safety and alignment, we generate synthetic concept-antagonist story pairs and use the difference of their mean model activations as a feature vector. We scale this approach to 1,036 alignment-relevant concepts using 2M synthetic stories across multiple languages and genres, which we publicly release. Across six models from three model families, our feature vectors reliably generalize to detecting their corresponding concepts in chat dialogues and web text. They form an interpretable geometry shared across models and can causally steer model behavior when injected into activations. We further use these features to analyze model representations during jailbreak attacks, mathematical and agentic reasoning. Our targeted feature construction thus offers a scalable route to studying concepts in language models, potentially extending to millions of features across arbitrary domains. 

---
# EGIP: An Evidence-Grounded Interaction Protocol for Clinical Agents 

**Authors**: Zecheng Ren, Ruichen Cong, Yi Min Chen, Jing Zhang, Feiyang Chen, Wutong Zhang, Kaixiang Wang, Zihan Wang, Jiong Lou, Jie LI, Qun Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.24539)  

**Abstract**: Interactive clinical agents acquire evidence through patient dialogue and diagnostic examinations, yet their performance also depends on how the interaction environment exposes information, available actions, and examination results. When these conditions are implicit or inconsistently controlled, final-answer accuracy can conflate diagnostic reasoning with the effects of the interaction protocol. We propose the \textbf{E}vidence-\textbf{G}rounded \textbf{I}nteraction \textbf{P}rotocol (EGIP), which enforces role-specific information boundaries, exposes diagnostic actions without revealing their results, retrieves examination results from stored clinical records, and organizes acquired evidence for final decision-making. By separating evidence acquisition from terminal prediction, EGIP enables different decision strategies to operate on the same observations. We evaluate EGIP across two clinical-agent frameworks, five datasets, and six language models. EGIP improves diagnostic accuracy over native baselines across all primary comparisons. A clinician review further shows that the proportion of automatically scored correct responses jointly judged evidence-supported rises from 10.20\% to 36.55\%. These findings support the effectiveness of EGIP as a unified interaction protocol for improving clinical-agent performance through controlled evidence access and acquisition. 

---
# Backend-Frontend Converter: Component-Based Conversion of .NET Computational Models to TypeScript 

**Authors**: Petr Ivankov, Max Shestov  

**Link**: [PDF](https://arxiv.org/pdf/2610.24492)  

**Abstract**: Computational models developed in .NET often need to run locally within web applications. We present Backend--Frontend Converter, an architecture for generating TypeScript implementations of component-based models. The converter maps components, connections, aliases, and formula trees to model-specific TypeScript code. Reusable runtime libraries provide numerical integration, data access, and state handling. We evaluate stored generated implementations through four orbital scenarios and a recorded trading-strategy comparison. Orbital calculations execute in a browser and are paired with C\# results by crossing event and timestamp; trading outputs are aligned by date. Analytic scheduler tests and deliberately perturbed outputs assess the validation procedure. Each orbital scenario produces 91 paired events with identical timestamps. Position and velocity differences remain below $(10^{-6}\)~km$ and $(10^{-9}\)~km/s$. All 20 analytic scheduler checks pass, and the comparator detects each of three prescribed state and time perturbations. The trading comparison contains 501 aligned daily records with matching indicator values, positions, and order events. The evaluated implementations preserve the compared outputs across the C#/TypeScript boundary. Component-specific generation and reusable execution libraries provide a practical route from supported .NET computational models to TypeScript applications. 

---
# Vocal-Tract Acoustics, Aerodynamics, and Overlap-and-Add Synthesis in DYNARTmo 

**Authors**: Bernd J. Kröger  

**Link**: [PDF](https://arxiv.org/pdf/2610.24490)  

**Abstract**: This paper presents the acoustic and aerodynamic extension of DYNARTmo, a dynamic articulatory model for visualizing and synthesizing speech production. Gestural control trajectories determine vocal-tract geometry, which is converted into an acoustic tube with nasal branching and effective lateral passages. A lumped aerodynamic network relates prescribed glottal motion and pulmonary pressure to airflow and supraglottal pressure storage. Phonatory excitation, oral friction, and glottal aspiration are synthesized as separate contributions. Frame-specific acoustic responses are combined by overlap-and-add, preserving response tails beyond individual control frames. Three German words, ``Tasse'', ``Muse'', and ``Lade'', illustrate the coordination of articulation, pressure, flow, and acoustic sources across vowels, nasals, laterals, fricatives, and plosives. The implementation has been optimized by the author primarily for sound recognizability; independent perceptual evaluation remains outstanding, and fricative-noise naturalness requires further refinement. The model provides an inspectable, articulatorily controlled production plant for future neural simulations of babbling, imitation, and early word acquisition. Further work will improve acoustic realism, reduce empirical source corrections, and extend the sensory information available to developmental learning models. 

---
# Modeling Discourse Profiles in Climate Misinformation Discourse 

**Authors**: Ge Gao, Zhengyang Shan, Bryan A. Plummer, Margrit Betke, Derry Wijaya  

**Link**: [PDF](https://arxiv.org/pdf/2610.24489)  

**Abstract**: Prior work on climate misinformation detection typically classifies individual comments or claims using text alone. This framing overlooks a key property of online climate discourse: misinformation is often produced, reinforced, and sustained through recurring rhetorical patterns that reappear across comments and threads. Using a large-scale TikTok climate discourse dataset, we construct discourse profiles by clustering climate-related comments into recurring rhetorical patterns. We then incorporate these profiles into fine-grained climate misinformation classifiers using a late-fusion architecture. Under video-disjoint evaluation across three training seeds, discourse-profile-aware models significantly improve macro-F1 over text-only baselines for bag-of-words, TF-IDF, and MPNet representations. RoBERTa-large exhibits a positive mean improvement but substantially greater variability, and the difference is not statistically significant. Beyond classification, our results show discourse profiles shape misinformation reception and propagation. Discourse-profile-conditioned language model agents exhibit misinformation acceptance patterns aligned with profile-level misinformation prevalence in ClimateDisc, and the relationship between misinformation prevalence and engagement persistence differs across dominant discourse profiles. 

---
# Write Once, Check Many: Reusable Ledgers for Post-Hoc Policy Compliance Auditing of Agent Trajectories 

**Authors**: Ashwin Nagarajan, Gursimran Panesar, Alicia Yang, Mona T. Diab, Virginia Smith, Aashiq Muhamed  

**Link**: [PDF](https://arxiv.org/pdf/2610.24483)  

**Abstract**: Auditing LLM agents against policies is critical to ensure safety but requires rereading long trajectories once per policy. Can we instead record each trajectory once and reuse that record for policies not known in advance? We propose write once, check many, a framework in which a writer model converts each agent trajectory into a policy-independent ledger of the agent's actions and outcomes. A reader then checks each new policy against the ledger alone, without access to the original trajectory. On 900 trajectories from SHADE-Arena, $\tau$-bench, and $\tau^2$-bench, we show that a 12B reader of narrative ledgers can either match or in fact achieve higher AUC than the same 12B model reading full trajectories. On SHADE-Arena policies that combine several facts, our approach outperforms a 31B model reading the trajectory (.777 vs. .713 AUC). When checking eight SHADE-Arena policies separately, ledgers use about $5\times$ fewer tokens, including the cost of writing them. A deeper analysis shows that rule-based extraction performs better on some single-action policies, and ledgers can miss evidence in long tool outputs and message contents. Our results suggest that reusable, policy-independent records can make retrospective agent auditing more efficient and effective, and highlight important considerations in ledger design. 

---
# Beyond Recall Saturation: Scaling Hard-Negative Reranker Training for Industrial Legal Retrieval 

**Authors**: Yunhan Li, Mingjie Xie, Zeyang Shi, Gengshen Wu, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24478)  

**Abstract**: Industrial retrieval systems often improve quality by enlarging embedding candidate pools or adopting complex training pipelines. We show that, under a sufficiently large but practical candidate pool, embedding recall exhibits diminishing returns, motivating a closer examination of reranker quality. On Chinese legal retrieval benchmarks, embedding top-300 reaches a high-recall operating region, while relevant evidence may still be ranked below the output cutoff. We compare hard-negative strategies for reranker training, including embedding-mined, self-mined, and dynamic-negative variants, under aligned retrieval protocols that evaluate reranking after candidate retrieval. Among the evaluated staged configurations, an embedding-to-self-mined pipeline provides a favorable trade-off; additional dynamic stages do not improve all held-out metrics. The selected pipeline scales from a 0.6B to a 4B reranker. Relative to the native 4B reranker, the trained 4B Stage2 model improves reported metrics on the STARD and CSAID benchmarks and on an internal development set, and is preferred in 72.4\% of non-tied comparisons on 410 de-identified, reconstructed internal operational-query inputs in a latency-sensitive offline preference evaluation. These results support evaluating the retrieval and reranking stages separately when selecting an optimization target for industrial retrieval systems. 

---
# Ben-Misandry: A Bengali Dataset and Late-Fusion Transformer Framework for Detecting Anti-Male Gender Hate 

**Authors**: Md Mehedi Hasan Nipu, Chinmoy Mitra, Tarannum Ahmed Nowshin, Israt Moyeen Noumi  

**Link**: [PDF](https://arxiv.org/pdf/2610.24474)  

**Abstract**: Gender based hate speech research in Bengali has focused almost exclusively on misogyny, leaving anti male hate, or misandry, entirely unstudied for this language and creating a systematic blind spot in moderation systems trained to catch attacks on women but not on men. We introduce Ben-Misandry, the first Bengali misandry detection dataset: 5,000 comments across six taxonomy classes, including a novel category for mocking male emotional vulnerability with no direct analogue in existing misogyny taxonomies. The corpus is taxonomy-controlled synthetic data generated with GPT-4.1, spot-checked by four human reviewers on a 500-comment sample who confirmed the large majority of assigned labels; a full inter-annotator- agreement study is left to future work. We benchmark it with a classical pipeline (seven classifiers over four text representations) and a transformer pipeline (BanglaBERT, XLM-RoBERTa, and two fusion strategies). The best classical model, a linear SVM over character-level TF-IDF, reaches 91.8% accuracy; a late fusion ensemble averaging BanglaBERT and XLM-RoBERTa softmax outputs reaches 96.6% accuracy and macro ROC-AUC above 99%, nominally ahead of every single model and early fusion, though we have not run significance testing to confirm these margins are meaningful. We relate this to classifier combination theory, under which averaging encoders with decorrelated errors is expected to reduce variance, and corroborate the decision boundary with Shapley explanations that recover the same class- discriminative vocabulary found in our exploratory analysis, while noting this cannot rule out the model partly keying on synthetic-generation artifacts. Our results indicate misandry detection in Bengali is feasible and measurable on this synthetic corpus, offered as an initial public resource pending validation on real-world data. 

---
# Which Long Documents Matter? A Controlled Study of Data Selection for Context Extension 

**Authors**: Junlong Jia, Jiang Zhou, Xing Wu, Chaochen Gao Tinghao Yu, Feng Zhang, Lilin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24469)  

**Abstract**: Natural long documents are widely used for context extension, but which selection pipelines are effective remains unclear. We conduct a controlled comparison on Qwen2.5-3B at 128K under matched 6B-token continued-training budgets. Random sampling of natural long documents (random-long) outperforms repacking at longer evaluation lengths. Within the same candidate pool, selecting documents with high LongPPL key-token density yields further gains, unlike the evaluated generic-quality and LLM-judge selectors. We introduce truncLongPPL to reduce scoring cost by computing key-token density over a truncated prefix. At 128K, scoring the first half of the reference span yields downstream performance close to full-span scoring with a measured $2.28\times$ end-to-end scoring speedup. Across three training seeds, truncLongPPL achieves the highest mean scores among evaluated pipelines, exceeding random-long by 4.54 points on validation HELMET and 3.28 on held-out RULER, with gains most consistent in retrieval. At a 256K target context, it gains 2.82 average RULER points over random-long on Qwen2.5-3B. In short/long mixed training across Qwen2.5-3B, Qwen3-8B, and OLMo~3-7B, it gains 1.47--7.31 average RULER points over random-long mixing, with seven-task short-context averages within 0.23 points of the corresponding baselines. After supervised fine-tuning (SFT), Qwen3-8B and OLMo~3-7B models trained with selected long documents outperform their random-long counterparts by 1.13--1.67 points on LongBench v2 and 0.62--1.44 points on MRCR. 

---
# When Opposite Premises Translate Identically: A Source-Conditioned Diagnostic for Multilingual Commonsense Items 

**Authors**: Shuai Wang, Rui Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.24428)  

**Abstract**: Multilingual reasoning comparisons depend on translated premises preserving the relations in their English sources. Automatic translation scores can rank source-target pairs, but an acceptance decision alone cannot certify a counterfactual benchmark field. We test a source-contrast diagnostic for ACCORD premises: reverse the English polarity, translate both arms independently, count identical target outputs, and score each arm against its own source. Across 71 matched question identities at two containing-item depths, five target languages, and three translation checkpoints, Swahili has 63, 113, and 67 exact collisions among 710 pairs per checkpoint. At a human-reference-calibrated threshold, round-trip NLI passes both arms of 3 of the 113 NLLB-3.3B collisions, while LaBSE passes 78. Each collision proves at least one unfaithful translation, but does not identify its arm. These controlled measurements do not establish screening accuracy on spontaneous translation errors or a language-by-hop reasoning effect; the study has no native equivalence labels or answer-model evaluation. 

---
# Toward a Universal Local Speech Feature Extractor Through Distillation 

**Authors**: Jessica Yang, Marcelo Beramendi Caballero, Ju-Chieh Chou, Karen Livescu  

**Link**: [PDF](https://arxiv.org/pdf/2610.24424)  

**Abstract**: Self-supervised speech models commonly consist of a set of convolutional neural network (CNN) layers used as a local feature extractor, followed by transformer layers. The CNN layers are computationally demanding, and are also similar across models and resemble the much simpler mel spectral features. We hypothesize that the CNN layers can be replaced by a simple, universal model. To test this hypothesis, we learn a two-layer feature extractor via distillation from one or more teacher models (HuBERT, data2vec, and WavLM). Experiments on SUPERB(-SG) benchmark tasks show that our model largely retains the performance of the teacher models and can also be used in an unseen model (wav2vec 2.0). Our feature extractor requires only 3% of the MACs of the CNNs, resulting in a $>$20% overall inference runtime reduction. 

---
# Ask Less, Ask Better: Allocating User Attention in Proactive LLM Agents with Interruption Budgets 

**Authors**: Xuanming Zhang, Amir H. Rezaeian, Miguel Ballesteros, Weiyi Sun, Werner Geyer, Lydia B. Chilton, Zhou Yu, Yassine Benajiba, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2610.24411)  

**Abstract**: LLM agents can ask clarifying questions to complete tasks, but answering each question demands users' time and effort. Existing evaluations typically measure this burden by counting questions, treating necessary and avoidable questions equally. We propose an evaluation protocol that distinguishes how often agents ask from how well they ask. An interruption budget limits the number of questions, while a user-attention cost charges extra for questions that are hard to read or not worth asking, as rated by a human-validated LLM judge. Beyond question counts, this cost captures perceived effort: on conversation pairs with the same number of questions, human annotators agreed with the cost on which conversation required more effort 75% of the time. Across eight LLMs and four customer-service domains, limiting agents to one to three questions reduces user-attention cost by 36-73% while keeping 86-94% of the task reward they earn with unlimited questions. At the same budget, reviewing each question against the memory of users' earlier answers cuts attention cost by another 13% with little reward loss and yields more valuable questions. Models also differ sharply in how well they ask: at the same user-attention cost, the most efficient models gain more than twice as much reward from their questions as the others. These results show that asking less and asking better are distinct goals, and that evaluations of proactive agents should measure both. 

---
# FlashMap: Flow Map Speculative Decoding 

**Authors**: Jeff Shen, Zhejian Zhou, Ryan Lee, Jonathan May, Shirley Ho  

**Link**: [PDF](https://arxiv.org/pdf/2610.24365)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference by using a lightweight drafter to propose a block of tokens that the target model verifies in one parallel pass. Autoregressive drafters generate this block one token at a time, whereas block diffusion drafters such as DFlash generate it in one pass, making drafting faster. However, these diffusion drafters predict each token independently, ignoring dependencies between tokens in the block. We introduce FlashMap, a continuous flow drafter that conditions a flow map language model on the target model's hidden states and generates the whole block in one forward pass while capturing inter-token dependencies. On Qwen3-8B, FlashMap is faster than a matched DFlash baseline on every benchmark, with 7% higher average speedup and up to 6.1x lossless acceleration. 

---
# What Matters for NVFP4 Training? A Comparative Study of Low-Precision Pretraining Recipes 

**Authors**: Anjulie Agrusa, Andrei Panferov, Elizabeth Wei, Keith Wyss, Paul Gibbons, Erik Schultheis, Jinhang Choi, Frank Sun, Tijmen Blankevoort, Dan Alistarh  

**Link**: [PDF](https://arxiv.org/pdf/2610.24343)  

**Abstract**: Training large language models directly in 4-bit floating-point (FP) formats promises substantial improvements in throughput and energy efficiency. While some recipe design choices have been validated at scale, many promising approaches remain untested beyond small models and short token horizons, leaving open the question of which trends will hold. We present a systematic comparison of recent NVFP4 recipes at medium model scale, evaluating dense 8B and 30B-A3B MoE models trained for up to 1T tokens, with a focus on which ingredients are necessary for accuracy recovery. We propose a final recipe grounded in the principles behind established stable NVFP4 training at scale, incorporating state-of-the-art techniques such as unbiased gradient estimation with lower quantization error than stochastic rounding. To the best of our knowledge, this is the strongest FP4 training result demonstrated at this scale to date in loss gap to BF16. Through ablation studies, we find that: (i) each technique in the optimized recipe measurably improves loss trajectory, (ii) selective high-precision layers are necessary for recovering accuracy at scale, (iii) not all tensors in the backward pass benefit equally from de-biasing, leaving room to apply complementary error-reduction techniques to the remaining tensors. 

---
# Beyond Clean Accuracy: The Practical Challenges and Benefits of Pixel-Based Language Models 

**Authors**: Haowen Tang, Wei Xia, Raoyuan Zhao, Hinrich Schütze, Yihong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.24290)  

**Abstract**: Pixel-based language models process rendered text without relying on a fixed subword vocabulary, offering a promising approach to noisy and multilingual inputs. However, their practical value depends on whether they can be reliably adapted to downstream tasks. To this end, we systematically examine this question through fine-tuning behavior, typographical robustness, and zero-shot crosslingual transfer. Under our tested settings, pixel-based models require longer training and broader labelled-data coverage than text-based models, while delayed learning and learning-rate-dependent failures make adaptation less reliable. Our analyses further suggest that task-relevant information is less readily accessible in pretrained pixel representations, placing greater demands on fine-tuning. Once successfully adapted, however, pixel-based models lose substantially less accuracy under typographical noise and achieve higher target-language accuracy than English-pretrained text-based counterparts on two crosslingual benchmarks, despite weaker clean English performance. Together, these findings reveal a central trade-off: replacing tokenization with rendering reduces sensitivity to corrupted and unfamiliar inputs, but shifts a central bottleneck to downstream adaptation. 

---
# BanglaShield: A Script-Robust Safety Classifier and Benchmark for Bangla Native, Romanized, and Code-Mixed Prompts 

**Authors**: Mohsinul Kabir, Md. Tariquzzaman, Mohammad Rashidujjaman Rifat, Tharindu Ranasinghe, Nishat Raihan  

**Link**: [PDF](https://arxiv.org/pdf/2610.24288)  

**Abstract**: Safety classifiers for large language models (LLMs) are typically built and evaluated on standard-script text, yet Bangla, like many lower-resource languages, is routinely written in native script, in Latin-script romanization, and code-mixed with English, Hindi, or Urdu. We introduce BanglaShield, a resource and model for graded prompt-safety classification across these written forms. We consolidate 10,245 Bangla prompts from six sources under a four-level taxonomy (Safe, Offensive, Harmful, Severe) and render each prompt in three aligned forms that share labels and a group-aware split, enabling strictly paired comparisons. Benchmarking six contemporary LLMs, including a Bangla-specialized one, on 36,882 classification queries and 24,318 judged responses, we find that LLM classifiers label 47.0% of unsafe native-script prompts as Safe, rising to 63.4% under romanization, while response-level safety accuracy drops by 10.3 points. Three multilingual guard models also pass 30.3-57.3% of unsafe native-script prompts. Using a model's own classification as an input filter reduces harmful compliance by only about five points. In contrast, BanglaShield, a compact fine-tuned encoder, reaches 87.8 macro-F1 on native-script and 84.4 on romanized prompts, outperforming the best LLM on each form by 42.8-46.1 points with ~130x fewer parameters than Gemma 4 31B. We release the corpus, splits, judgments, and model. 

---
# Modeling bilingual naming performance through laterally-connected SOMs 

**Authors**: Divya Prabhakaran, Uli Grasemann, Erie Shivers, Marissa Russell-Meill, Swathi Kiran, Risto Miikkulainen  

**Link**: [PDF](https://arxiv.org/pdf/2610.24275)  

**Abstract**: Computational models have explanatory power to understand cognition in individuals, but there is usually not enough training data to build such models with traditional machine learning. However, alternative architectures that model neural and cognitive processes in more detail can be more easily fit to individuals and can lead to mechanistic insights into human behavior. BiLex is such a computational model of the bilingual lexicon, containing a shared semantic system and phonetic language systems represented through self-organizing maps. In this study, BiLex was extended further into biological realism with excitatory and inhibitory lateral connections. This extension makes it possible to account for response times as well as common errors such as those based on semantic concept familiarity, typicality, and specificity. The Boston Naming Test was simulated in Spanish and English, and model performance approximated naming accuracy and error types in Spanish-English bilinguals with varying primary and secondary language histories. The results indicated that the approach captures semantic-level processes in bilingual lexical access, thus modeling the semantic cognition of individual subjects. In the future, such models may serve as a foundation for modeling impairments and treatment options. 

---
# Best of Both Worlds: Fine-Tuned Small Models Complement LLMs for Cost and Performance Benefits 

**Authors**: Shivam Mathur, Sri Harish Popuri, Priya Krishnan Sundararajan, Kevin Horecka, Michael Mugo  

**Link**: [PDF](https://arxiv.org/pdf/2610.24260)  

**Abstract**: BERT-based models require thousands of labeled samples to train. Larger models like GPT-X require only 1-5 examples per class. However, performance, cost and time tradeoffs of such models remain under-explored. In this work, we classify survey text data into 16 classes. We focus on experimental approaches and results on performance, cost, and time taken for this problem. We compare these parameters of a trained RoBERTa-based model, and an (off-the-shelf) GPT-3.5 model for text classification. Baseline experiments show that the time taken by GPT-3.5 for generating predictions on 10,000 records is 6 hours, whereas a RoBERTa-based model takes only 12 mins on a T4 GPU. To identify the best prompting approach for GPT-3.5, we explore batch prompting. We experiment with (1) batch-size, and (2) few-shot sampling. We find the best batch size is 20 with human few-shot curation. Experiments show the F1 score of a trained RoBERTa-based model (96%) is much higher than a GPT-3.5 model (83%). In case of GPT-3.5, batch prompting is beneficial in lowering inference costs by ~70%, without hindering task performance. We find the best approach is an ensemble of a RoBERTa-based and GPT-3.5 model (F1: 97.5%) incurring only 4.4% of inference cost compared to a standalone GPT-3.5. 

---
# SLA-LENS: From Classroom Transcripts to CSLA-Informed and Evidence-Linked L2 Interaction Analytics 

**Authors**: Nemika Tyagi, Olga Kellert, Phanos Kareen, Nelvin Licona-Guevara  

**Link**: [PDF](https://arxiv.org/pdf/2610.24259)  

**Abstract**: What happens after a learner makes an error, and how does the instructor's response shape the next turn? Investigating these exchanges across classrooms requires context-sensitive annotation that is difficult to scale manually. We introduce SLA-LENS, an end-to-end LLM-assisted framework that connects classroom transcripts to structured pedagogical annotations and evidence-linked descriptive analytics. Grounded in interactionist Second Language Acquisition (SLA) theory, its annotation schema captures second language learner errors, pedagogical functions, and uptake or repair while accounting for speaker roles, conversational context, and hierarchical labels. We apply the framework to 40 authentic online L2 Spanish classroom transcripts comprising over 21,000 sentence-level instances. We evaluate proprietary and open-weight LLMs, including task-specific QLoRA adaptation, using expert-reviewed gold annotations, held-out silver data, and metrics sensitive to different types of annotation error. A deterministic reporting layer generates reproducible visualizations of participation, feedback, learner errors, uptake, and temporal interaction patterns, linked to supporting transcript evidence. By making these patterns accessible and inspectable, SLA-LENS supports systematic classroom discourse research, pedagogical reflection, and curriculum design. 

---
# Can This LLM Annotate My Data? Estimating Zero-Shot LLM Performance 

**Authors**: Tong Lin, Grant Schoenebeck, Sabina Tomkins  

**Link**: [PDF](https://arxiv.org/pdf/2610.24250)  

**Abstract**: Large language models (LLMs) are increasingly used in annotation tasks with varying success. While several studies have demonstrated the opportunities and shortcomings of LLMs for annotation, a critical question remains unanswered: is it possible to predict how well they will perform on a new dataset? To answer this and related questions, we construct two task families which induce 461 text classification datasets. We find that it is possible to predict LLM performance - to a certain degree - and that contrary to other work which highlights the usefulness of LLM verbalizations of self-confidence, internal log-probabilities are the most predictive signal. 

---
# Encoder-Free Multimodal LLMs Learn Implicit Encoders 

**Authors**: David Harwath, Desh Raj, Xavier Velez, Sandra Arcos-Holzinger, Willie Jeng, Karen Livescu, Leibny Paola Garcia-Perera, Sanjeev Khudanpur  

**Link**: [PDF](https://arxiv.org/pdf/2610.24239)  

**Abstract**: We empirically analyze two recently proposed encoder-free multimodal LLMs, Gemma-4 12B and Inkling-Small. Unlike previous multimodal LLMs, these models do not use modality-specific encoders to extract tokenized embeddings representing audio and image input; instead, they directly feed windows of samples, pixels, or spectral features to the Transformer language model. We analyze whether the early layers of these models learn to function similarly to dedicated audio and visual encoders. We find that these early layers do in fact act as ``implicit encoders,'' encoding audio and visual inputs in a way that is largely invariant to their cross-modal context. Furthermore, when performing multimodal reasoning tasks, we show that independently encoding the modalities throughout these early model layers achieves the same accuracy as jointly encoding them from the first layer onwards. Finally, we use layer-wise probing to analyze the salient information captured by the implicit encoders and find their behavior to be similar to several popular encoder models. 

---
# Preserving Reasoning Diversity in Reinforcement Learning with Verifiable Rewards via Adaptive Replay 

**Authors**: Wen-Zhi Li, Sainyam Galhotra, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2610.24236)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) improves the reasoning capabilities of large language models, but often reduces the diversity of their reasoning behaviors. We study this phenomenon from the perspective of finite-sample learning dynamics and show that limited on-policy rollouts can progressively suppress semantically distinct but equally rewarded reasoning modes. Once a mode becomes sufficiently under-sampled, standard on-policy RLVR rarely recovers it. Motivated by this observation, we propose TAR, a Targeted Adaptive Replay framework that preserves diverse correct reasoning behaviors already present in the base policy. TAR constructs a compact memory bank of diverse base-model solutions and selectively replays them through an adaptive mechanism that activates only when stored behaviors are being suppressed during RLVR. Across six mathematical reasoning benchmarks and two Qwen2.5 model scales, TAR consistently achieves a better balance between reasoning performance and diversity preservation than existing diversity-preserving post-training methods, while substantially reducing the forgetting of correct base-model solutions. Code is available at this https URL. 

---
# Analyzing Repetition and Interference Effects in Memory-Constrained User Simulators 

**Authors**: Cara Su-Yi Leong, Nicholas Tomlin, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2610.24232)  

**Abstract**: User simulators can help evaluate and train AI systems to interact with humans at scale. To do so, evaluation of user simulators must measure behavioral parity with people. We evaluate whether memory-constrained user simulators are sensitive to two factors known to affect human information retention: repetition of a fact and interference from competing information. We find that some simulators achieve human-like overall accuracy, and replicate the magnitude of repetition and interference effects, but none match people's accuracy on particular questions: while humans misremember facts, models forget them, leading to different error patterns. We release this data as a challenge benchmark for future work on user simulators. 

---
# Language Models are Lazy Poets: Creative Mode Collapse at Scale 

**Authors**: Jacob A. Matthews, Sil Hamilton  

**Link**: [PDF](https://arxiv.org/pdf/2610.24223)  

**Abstract**: While aligned large language models (LLMs) can successfully complete many creative tasks, their outputs tend to be highly self-similar. Though solutions to this problem of "mode collapse" have been proposed, comprehensive large-scale surveys of the phenomenon in specific textual genres are still lacking. We address this gap through an extensive treatment of LLM mode collapse in poetry. Across 15 poetic forms, 248 subject matter tags, and 16 open-weight models, we generate over 1.1 million poems. To our knowledge, this constitutes the largest corpus of LLM-generated poetry to date. At million-poem scale, we find that LLM poetic vocabularies and rhyming abilities are substantial, but severely underutilized. In other words, LLMs are "lazy" poets: despite possessing large poetic inventories, LLMs over-rely on a small number of words, opening phrases, and rhymes. Further, we find that lexical homogenization differs across models and poetic forms, noting observed rhyming vocabulary is correlated with model size. 

---
# Reasoning with Context Management and External Memory 

**Authors**: Man-Ling Sung, William Hartmann, Jiangying Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.24187)  

**Abstract**: Reasoning models maintain information about the status of their reasoning process through in-context memory. Complex problems require larger context windows and memory requirements. In this scenario, context management without caching risks discarding useful information for reasoning. We propose RCMEM, an approach that extends context management with an external memory to maintain additional information without burdening the reasoning model with additional context. RCMEM significantly reduces computation time on problems that require repeated computation of intermediate steps, while achieving better performance on complex problems. We demonstrate our approach on a few mathematics problems using model sizes ranging from 0.4 to 25 million parameters, and GPT-2. 

---
# Multi-author style detection with stylistic embeddings 

**Authors**: Man-Ling Sung, Chak Fai Li, Damianos Karakos, Fred Bernardin, Brian Ulicny  

**Link**: [PDF](https://arxiv.org/pdf/2610.24184)  

**Abstract**: We investigate the use of Large Language Models (LLMs) in multi-author style change detection. Using off-the-shelf LLMs or supervised fine-tuning solely within the target domain fails to yield effective features; for this reason, we propose cross-domain transfer learning with LLM-based embeddings fine-tuned for authorship verification. We fine-tune Mistral-7B embedding model and compare it with DeBERTa, used in the state-of-the-art (SOTA) systems in PAN-25 multi-author writing style analysis challenge. Results show that DeBERTa and Mistral excel in different user-generated text scenarios. We exploit this diversity by combining both embeddings, leading to more effective style change detection. 

---
# Prosodic Structure as a Critical Component in Phonemic Language Modelling at BabyLM Scale 

**Authors**: Bill K. S. Ku, Janet B. Pierrehumbert  

**Link**: [PDF](https://arxiv.org/pdf/2610.24172)  

**Abstract**: Human children learn language from continuous, prosodically structured speech long before they can read, yet almost all language models train on orthographic text. Building on previous phonemic BabyLM training (Goriely et al., 2024; arXiv:2410.22906), we ask whether supplying an estimate of the prosody infants exploit improves phonemic language modelling at BabyLM scale. We compare an orthographic baseline to three conditions forming a ladder of phonological richness. All conditions are trained as BERT MLMs at 10M and 100M words. Addressing the ecological validity of existing metrics in the small-data regime, we build a lexical decision task and two semantic organisation tasks. Our new lexical decision task offers improved insights into the distinction between wordlikeness, token familiarity, and lexicality as factors. Triphone phonotactic control and pseudo-log-likelihood scoring shows that these scores reflect phonotactic familiarity rather than wordhood: every model sits near or below chance on an adversarial inverted condition, revealing prior results of this kind as partly artefactual. On our new semantic tasks, a prosodically annotated phoneme stream gives the strongest taxonomic organisation at both the 10M and 100M word scales. Phonemic models with boundary structure are competitive with the orthographic baseline. These findings suggest that prosodic structure is a key component of phonemic language modelling in the small-data regime most analogous to infant learning. 

---
# Pronouncing Hard Words: Phoneme Adaptation of a Pretrained Conversational Text to Speech Model 

**Authors**: Yashish M. Siriwardena, Dinesh K. Gnanasekaran, Tanmay Laud, Shinji Watanabe, Subhabrata Mukherjee, Shanil Puri  

**Link**: [PDF](https://arxiv.org/pdf/2610.24164)  

**Abstract**: Voice agents built as cascaded automatic speech recognition (ASR)->large language model (LLM)-> text-to-speech (TTS) pipelines increasingly rely on conversational TTS: streaming, context conditioned models that adapt delivery to dialogue history. Such models are natural and expressive but offer little explicit control over how rare, trust-critical drug names are pronounced, because these entities are too infrequent, and their vocabulary too fast-growing, to be covered by any amount of pre-, mid-, or post-training. We retrofit phoneme controllability onto a pretrained conversational TTS model without any pre/mid-training, using an ARPAbet vocabulary exten-sion, continued phoneme-augmented supervised fine-tuning (SFT), and a training-free inference-time acoustic-biasing step. On a 7,500 clip clinical drug-pronunciation evaluation, our model raises strict Gemini-judged correctness from 54.6% (no phoneme augmentation) to 90.7%, outperforming every conversational and stateless TTS baseline on the automatic metric. Because no training stage sees a drug name, the entire evaluation is out-of-domain, so the gains reflect learned pronunciation rather than memorization. In a blinded, randomized human study, adult listeners judge 84% of our clips correct, versus 82% for the strongest baseline, with overlapping confidence intervals on a stratified subset. Across all 250 matched clips, Gemini agrees with the human majority on 84.8% of binary labels (k{appa}=0.57). 

---
# CoDA: A Human-LM Collaborative Codebook Development Assistant for Domain-Specific Text Annotation 

**Authors**: Jaspreet Ranjit, Swaminathan Chellappa, Augusto Rivas Costante, Chandan Manjunath, Laura Onasch-Vera, Hailey Winetrobe Nadel, Eric Rice, Swabha Swayamdipta  

**Link**: [PDF](https://arxiv.org/pdf/2610.24157)  

**Abstract**: We present CoDA, an open-source human-LM collaborative codebook development assistant that helps domain experts develop annotation guidelines for annotating new variables in unstructured text with language models. Existing language model annotation systems primarily focus on downstream classification performance, while assuming the availability of high-quality annotation guidelines, or codebooks. In practice, developing these codebooks is time-consuming and challenging, particularly for concepts that require expert knowledge. CoDA addresses this gap by collaborating with domain experts to iteratively develop codebooks through targeted review of incorrect language model predictions and incorporating expert feedback into the evolving codebook. We demonstrate CoDA on two real-world case studies: extracting health indicators from homelessness case management notes and classifying sentiment of wildlife trafficking posts on social media. Across both domains, CoDA produces codebooks that achieve downstream classification performance comparable to expert codebooks while reducing codebook development time from several weeks to two hours and requiring 5X fewer manually annotated examples. Applying CoDA to generate annotations at scale, we identified 2,410 case notes that contained health discussions or referrals and 10,196 social media posts expressing negative sentiment that may warrant investigation for potential wildlife trafficking. By accelerating codebook development, CoDA helps practitioners use language models to extract domain-specific insights from unstructured text at scale. 

---
# Preservation and loss of algorithmic structure in experimental diffusion of joint-action norms 

**Authors**: Dhara Yu, Bill D. Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2610.24150)  

**Abstract**: Everyday norms of social interaction - e.g. norms of sharing space, taking turns, or queuing - are essential to large societies, offering structured procedures for common interactions. However, it remains unclear how the underlying algorithms of these implicit joint plans evolve as they diffuse through populations. Here we show that this process of diffusion can select against complex algorithms in the earliest stages. We conducted a large-scale study of experimental diffusion of novel social norms among human participants, leveraging advances in program synthesis to characterize the procedures people developed in independent populations. Inherited procedures became structurally simplified and lost during interactions between participants who inherited different variants of joint-action norms; however, opportunities for interactive repair offset this pressure, preserving complex algorithms despite misalignment. These results demonstrate that interactive reasoning mediates selective pressures on structural complexity during cultural evolution of procedural norms. 

---
# Prefix-Tuning for Arbitrary Output Sequences on Pretrained Transformers 

**Authors**: Peter Cho-Ho Lam, Ziyi Wang, Zirui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.24123)  

**Abstract**: Prefix-tuning is a parameter-efficient fine-tuning method for large language models (LLMs) that steers model outputs by prepending task-specific prefix representations to the input sequence. Despite its practical success, the theoretical understanding of the expressiveness of prefixes remains incomplete. In particular, the following question remains open: Given a pretrained autoregressive Transformer, for any input-output sequence pair $(\mathtt{x},\mathtt{y})$, does there always exist a prefix $\mathcal{S}$ such that prepending $\mathcal{S}$ to $\mathtt{x}$ induces the model to generate the target output $\mathtt{y}$? In this work, we prove that under the assumption that $\mathtt{y}$ is in the range of the pretrained Transformer, such a prefix $\mathcal{S}$ exists and can be explicitly computed by solving multiple linear systems. We further show that this assumption is necessary, in the sense that such a prefix may not exist when the assumption fails, thereby providing a complete answer to the above question. Our analysis is constructive and the obtained theoretical results hold under fairly general assumptions on model architecture. Extensive experiments on GPT-2 and Qwen3-4B demonstrate that our algorithm produces prefixes with strong control over model outputs. 

---
# Do LMs Know What Variables Do? Interpreting Code Representations via Variable Roles 

**Authors**: Célian Vasson, Daniil Ognev, Kentaro Inui, Benjamin Heinzerling  

**Link**: [PDF](https://arxiv.org/pdf/2610.24118)  

**Abstract**: Variable roles describe recurring behaviors such as accumulation, traversal, and retaining the best candidate encountered. We use Sajaniemi's independently defined taxonomy to study which role distinctions can be recovered from large language models, how sparse autoencoders (SAEs) retain those distinctions, and whether associated directions support control of generation. We introduce static annotation rules for all eleven roles and a corpus of 126,926 natural-code files across five languages. Across five model-SAE systems, residual probes outperform individual SAE features and tested lexical baselines, with modest losses after neutral target renaming. Combining SAE features closes much of the single-feature gap, but probes on SAE reconstructions are less sensitive to role-changing edits reviewed by LLMs. High paired ranking accuracy coexists with frequent false positives at fixed thresholds. During generation, interventions often change role annotations at the cost of correctness; changes accompanied by passing tests show no clear aggregate advantage over random controls. The benchmark separates role-related recoverability, SAE fidelity, and correctness-preserving control through matched prediction, source edits, and generation interventions. 

---
# OR-License: Evaluating LLM Agents as Real-World Operations Research Practitioners 

**Authors**: Peisong Wang, Feixue Liu, Yize Meng, Bowen Liu, Ruoyu Chen, Gaoyang Dai, Xing He, Aiqian Li, Fengchi Yuan, Aochuan Chen, Chenyi Zi, Yuhan Li, Hongchuan Zeng, Yimo Yan, Nuo Chen, Jie Fu, Zuo-Jun Max Shen, Yong-Hong KUO, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.24087)  

**Abstract**: Can large language model (LLM) agents deliver expert-level operations research (OR) solutions directly from original client materials, without prior expert reformulation? Existing OR benchmarks typically begin with self-contained descriptions or predefined optimization tasks, underrepresenting the translation of original business materials into professionally validated solutions. We introduce OR-License, an end-to-end benchmark of 31 instances across 10 industrial scenarios from paid OR engagements totaling USD 3.68 million. It preserves original requirements, input data, deliverable formats, and information boundaries after de-identification. To study information acquisition, we introduce ORSim, a controlled client simulator that discloses business facts without supplying OR expertise. Hidden evaluation checks feasibility and compares objective quality against original expert solutions accepted by clients before benchmark construction. Deployment-Adjusted Quality (DAQ) averages expert-relative quality across instances, assigning zero to infeasible deliveries. Across 11 frontier LLMs using coding-agent harnesses, several exceed the expert reference in conditional objective quality, but none matches its DAQ; the strongest primary Claude Code configuration achieves 88.39% strict feasibility and a DAQ of 0.87 against the expert reference of 1. Client information acquisition extends beyond clarification into solution development and validation, with some later stages showing higher confirmed issue-resolution proportions. Trajectory analysis identifies formulation and implementation failures, while task-equalized research-chain completion rates range from 24% to 45% across models on evaluable tasks. These findings motivate sustained information-gap resolution, faithful problem formulation, and evidence-linked self-improvement for reliable long-horizon agents. 

---
# Discuss More, Comment Less: Boosting Automated Research via Multi-Perspective Collaborative Review 

**Authors**: Junkai Zhou, Jiajia Wu, Zhaoyi Zhang, Shiting Guan, Chuanguang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24050)  

**Abstract**: As automated research is increasingly adopted for writing papers, automated review has become a critical link between paper generation and quality control. However, existing methods largely rely on single-model feedback or simple review aggregation, failing to explicitly model expert responsibilities or collective deliberation in actual peer reviews, which limits effective cross-examination and reliable decision-making. Therefore, we formulate paper review as a computational modeling problem of peer review mechanisms and propose RoundtableReview, a deliberative multi-agent framework for enhancing automated research. RoundtableReview structures each paper into eight semantic slots and coordinates six academic and industry reviewers with complementary perspectives, retrieval sources, and specialized responsibilities. Through independent assessment, cross-examination, roundtable deliberation, dynamically weighted voting, and chair synthesis, the framework generates evidence-based and traceable feedback. Experiments on two automated research backbones show that the method improves paper quality in both backbones and outperforms baselines in automatic and human evaluations with fewer comments. 

---
# The Window Is in the Reasoning: When LLM Agents Decide to Follow Prompt Injections 

**Authors**: Haoxuan Jia, Yang Liu, Bin Chong, Yingguang Yang, Yancheng Chen, Jiayu Liang, Qian Li, Hanning Lu, Kefu Xu, Hao Zheng, Chongyang Zhang, Yulin Huang, Jianshen Zhang, Yongzhi Qi, Yunuo Lin, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.24048)  

**Abstract**: LLM agents that read untrusted content can be hijacked by prompt injections. Runtime monitors that try to catch a hijack before the agent acts, and benchmarks that score each task with a single run, share an assumption: whether an agent follows an injection is settled when the injection is read. We test this assumption with branching rollouts that replay an episode exactly up to the step at which the agent reads an injection and then resample the rest of the episode many times, splitting the outcome variance into a part fixed at read time and a part decided afterwards. Across four AgentDojo suites, a Qwen3 agent without thinking is largely decided at read time (0.91 of the outcome variance lies between states); with thinking on, this share drops to 0.62 (95% CI of the drop [-0.38, -0.22]), in the same direction for three attack templates and two success labels. How much is fixed at read time is a property of the model that the same measurement reveals, and thinking lowers the attack success rate of Qwen3 but raises it for two other models. Resampling from reasoning prefixes shows that the decision forms gradually as the agent reasons. Zero-shot chain-of-thought monitors built on cheaper models miss much of what is already decided, whereas stronger monitors, and a lookahead monitor that judges simulated next actions, come close to what the prefix determines. Finally, single-run labels of thinking agents disagree with a rerun up to 35% of the time, so benchmarks need repeated runs per task. 

---
# Translating in style: An open-weight alternative to GPT-4 for culturally aligned machine translation 

**Authors**: Thomas Rietman, Hadi Mohammadi, Robert A. Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2610.23868)  

**Abstract**: Machine translation systems often preserve meaning but systematically flatten linguistic style, a failure that is most acute when translating into non-Western languages. The RASTA framework of Havaldar et al. (2025a) addresses this through retrieval-augmented stylistic alignment, but relies on GPT-4, a closed, proprietary model. We investigate whether an open-weight translation specialist, Hunyuan-MT-7B, can serve as an open-weight alternative. We evaluate three variants of the model, the base model, a LoRA supervised fine-tune, and a GRPO-steered policy. We do this across three style dimensions (formality, politeness, and intimacy) and twelve directed language pairs, measuring style alignment with the correlation-based metric $\mathcal{A}$. We find that fine-tuning does not reliably improve alignment: the three variants are statistically indistinguishable in aggregate. The open backbone does not match RASTA-GPT-4 overall, but the shortfall is concentrated in formality and largely closes for intimacy and Chinese directions, where the open model matches or surpasses GPT-4 on individual pairs. An accessible RASTA pipeline built on an open model is thus already viable for parts of the problem. 

---
# Composable Parametric Demographic Injection for Values Survey Simulation 

**Authors**: Xinyuan Cao, Qingyi Pan, Bangde Du, Yujia Zhou, Hexi Wang, Weihang Su, Yiqun Liu, Yueyue Wu, Min Zhang, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2610.23867)  

**Abstract**: Large language models (LLMs) are increasingly used as simulated respondents in computational social science, yet accurately reproducing population-specific heterogeneous response distributions remains challenging. Existing methods such as demographic prompting are sensitive to prompt design and increase inference overhead, while parameter-based methods struggle to reuse demographic information efficiently, limiting generalization to unseen combinations. In this regard, we propose PARADE, a composable PARAmetric DEmographic injection framework that represents each demographic attribute value with a reusable and composable LoRA module. Specifically, PARADE separates task-common survey capability from demographic-specific value shifts via a shared Task LoRA and attribute-level Demographic LoRAs, with composition calibration to reduce parameter interference. At inference time, target population models can be constructed on demand by recombining the relevant modules, including demographic combinations not observed during training. Extensive experiments on the World Values Survey and SocioBench across three backbone models show that PARADE achieves strong population-level distribution alignment while maintaining competitive predictive performance, reducing KL divergence by up to 52.8% relative to the strongest baseline. 

---
# Modeling Human Behavior in the Prisoner's Dilemma with Large Language Models 

**Authors**: Raina Gao, Tristan Raffo  

**Link**: [PDF](https://arxiv.org/pdf/2610.23860)  

**Abstract**: The Prisoner's Dilemma has long served as a foundational framework for studying human decision-making and cooperative outcomes across social scenarios, yet human behavior consistently deviates from predictions of fully rational economic theory. Previous works have studied human and LLM outcomes in the Prisoner's Dilemma; however, these studies have lacked at least one key feature: a memory component for LLM agents, accessible open-source models, or comparison against empirical human data. This project investigates whether open-source LLMs, TII Falcon 7B Instruct, Google Gemma 7B, Meta Llama 3.1 8B Instruct, Microsoft Phi Mini MoE Instruct, and Qwen 2.5 7B Instruct, can model human cooperation patterns in repeated Prisoner's Dilemma games, while examining how an LLM's knowledge that its opponent will be shuffled affects its decisions. Using a standardized prompting framework, we simulated interactions in two trial types: a fixed-partner setting where both agents retain memory, and a shuffled-partner setting where a memory-enabled agent, informed that its opponent changes across rounds, interacts with a memoryless agent. For each trial, we measured cooperation rates over 100 rounds and 10-round increments and compared them to human data from Montero-Porras et al. (2022). Under both frameworks, all LLMs differed from the human cooperation rate. Open-source LLMs generally over-cooperated relative to humans, particularly in the shuffled framework, where human cooperation averaged 28.5% while several models exceeded 80%. Qwen-2.5-7B-Instruct showed deterministic tendencies, while Phi-mini-MoE-Instruct displayed greater variability and briefly approximated human rates. These findings suggest that LLMs have potential as behavioral simulation agents and provide a benchmark across two interaction environments, although open-source models are not yet reliable substitutes for human participants. 

---
# Can It Actually Work? Synthesizing Functional CAD Mechanisms 

**Authors**: Wenqian Zhang, Yangyi Huang, Zigeng Xu, Weiyang Liu, Zhen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26861)  

**Abstract**: While LLM agents can generate visually plausible CAD assemblies, these designs often fail to fulfill their intended physical functions. We study functional CAD design, a setting where agents interpret underspecified requests to construct and iteratively revise mechanisms that satisfy functional requirements in physical simulation. Verifying these requirements is nontrivial: different candidate mechanisms may realize the same function through entirely different components, controls, and operating sequences. To bridge this gap, we propose FunCAD-Harness, a framework that translates abstract requirements into candidate-specific physical tests and uses simulation feedback to guide design revisions. To systematically evaluate this capability, we introduce FunCADBench, a benchmark spanning 46 mechanism families with interdependent and multi-step operations. Experiments across frontier LLM agents show that FunCAD-Harness exposes subtle mechanical failures invisible to visual inspection, substantially improving functional success rates over standard simulation-augmented baselines. Ultimately, our work marks a step toward CAD agents capable of translating human intent into working mechanisms. 

---
# WeAct: Dynamic Weight Switching as Agent Actions 

**Authors**: Rui Shi, Xunjian Yin, Jiacheng Sang, Zichuan Liu, Shuyan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.26847)  

**Abstract**: Long-horizon agents must learn heterogeneous capabilities from interleaved trajectories. Updating shared parameters to learn new capabilities can degrade previously acquired ones, resulting in catastrophic forgetting. As a trajectory unfolds, different stages may require different capabilities, yet conventional joint training and static adapter merging maintain a fixed parameter configuration throughout execution. We introduce WeAct (Weight-level Internal Actions), which trains independent capability-specific LoRA adapters over a frozen backbone and adds adapter selection to the agent's action space. These internal actions use the model's native tool-calling interface for on-demand capability composition in a shared context. Across eight deep-research benchmarks, Qwen3-32B with WeAct improves average accuracy by 6.2% relative to full fine-tuning counterparts despite using less compute, and by 27.5% relative to the strongest static LoRA-merging method. WeAct also supports modular continual adaptation, by adding a new code computation adapter to expand the agent's capabilities and updating the existing writing adapter to improve report quality. Both updates leave the backbone and other adapters unchanged, supporting modular continual adaptation in agents. 

---
# Agent World Model in the Wild: Computer-Use Agent Environment Synthesis 

**Authors**: Zhaoyang Wang, Qianhui Wu, Haorui Wang, Simon Yu, Baolin Peng, Canwen Xu, Hao Cheng, Wenlin Yao, Boyi Liu, Yite Wang, Jianfeng Gao, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.26828)  

**Abstract**: Agent learning is shifting from learning from human data toward learning through interaction with the world. However, scaling such learning directly in real-world computer systems is challenging and can introduce substantial security risks. We therefore propose Agent World Model in the Wild (WildAWM), a scalable pipeline for synthesizing controllable environments and tasks for training general computer-use agents (CUAs). Whereas existing approaches typically specialize in a single agent interaction interface, we construct executable environments that jointly support graphical user interfaces, model context protocol services, and command-line interfaces, allowing agents to flexibly combine visual interaction, tool use, and code execution. WildAWM is entirely powered by open-source models, demonstrating that high-quality computer-use environments can be generated without proprietary frontier models. Beyond environments, we address three recurring challenges in task synthesis: ensuring solvability, maintaining sufficient difficulty, and producing reliable verifiers, by introducing execution-based quality gates that combine heterogeneous agent rollouts, cross-judge consistency checks, and counterfactual probes of judgment robustness. The resulting pipeline synthesizes 341 long-horizon environment-task pairs that can require over 100 agent steps to solve and remain challenging even for frontier models, with GPT-6-Astra solving only 60% of them. We train two Qwen models of different sizes exclusively in these synthetic worlds. Both models achieve substantial gains across 8 out-of-distribution benchmarks spanning web, claw, and computer-use agents, demonstrating that learning in WildAWM's synthesized worlds generalizes broadly to unseen computer-use tasks. The code is publicly available at this https URL. 

---
# Teacher Supervision, Fixed Representations and Calibration: Exact Risk Calculations for Candidate-Scoring Decision Models 

**Authors**: Agus Sudjianto, Denis Burakov  

**Link**: [PDF](https://arxiv.org/pdf/2610.26797)  

**Abstract**: Candidate-scoring decision models return a probability distribution over a presented set of answers. Public documentation of one such system, Jev, and of an open counterpart, JevAny, describes three training ingredients that determine what these probabilities estimate: soft targets from larger models, a frozen pretrained backbone and temperature calibration. We analyze each ingredient in Gaussian surrogates where the finite-sample risk can be computed exactly. A student trained on teacher targets inherits the teacher's bias, which more student data does not remove and agreement with the teacher does not reveal. A fixed representation lowers the risk only below an explicit sample-size threshold. In a probit model the exact recalibration map is a temperature scaling, which removes all excess risk only when the teacher's coefficient has the correct direction and otherwise leaves a residual that calibration diagnostics do not detect. Under Luce's choice axiom, candidate scores are coherent across refinements of an event if and only if they aggregate by log-sum-exp. On 231,521 credit card complaints from the Consumer Financial Protection Bureau, where the decision is whether a complaint ends in monetary relief, a student trained on a logistic teacher's probabilities ends at the teacher's decision cost while a student trained on outcomes goes 8% below it, a 30B model read zero-shot remains worse than the teacher after recalibration, and that model departs from the coherence condition by 17.5 decibans. Pretrained representations reduce the labeled data required and calibration corrects the scale of the probabilities, but neither corrects a biased teacher, so the probabilities have to be checked against realized outcomes. 

---
# Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars' to Counteract Phone Scams 

**Authors**: Nardine Basta, Robin Carpentier, Benjamin Zi Hao Zhao, Dali Kaafar  

**Link**: [PDF](https://arxiv.org/pdf/2610.26781)  

**Abstract**: Phone scams impose substantial and growing costs on individuals and financial systems worldwide, yet existing countermeasures impose no resource cost on perpetrators. Strategic scam-baiting, in which automated agents engage scammers to deplete their operational capacity, offers a proactive alternative but requires dynamic strategy adaptation that static systems cannot provide. We introduce a game-theoretic framework enabling LLMs to learn effective counter-strategies through in-context experience without fine-tuning. The framework models scammer-baiter interactions as a two-player game with empirically grounded strategies and utility functions, using Nash equilibrium as a formal benchmark for evaluating learned behaviour. Across 300 simulated dialogues and three LLM architectures, adaptive strategy selection achieves 28.9-79.7% closer alignment to Nash equilibrium than non-adaptive baselines. The code and an annotated dataset of 300 dialogues with turn-level strategy labels and utility scores are released at this https URL . 

---
# HarnessPro: Recursive Harness Self-Improvement with Pareto Optimization for Performance-Cost Trade-offs 

**Authors**: Zhuoshi Pan, ZhiYuan Feng, Yukuo Cen, Honglin Lin, Junru Lu, Zhongxia Chen, Shuo Yang, H. Vicky Zhao, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26770)  

**Abstract**: Harness development plays a crucial role in building capable AI agents. Recent approaches improve harnesses through recursive self-improvement (RSI), but their performance gains often come with higher execution costs. We introduce HarnessPro, a label-free method for evolving harnesses to improve task performance while reducing execution cost. In this setting, candidate harnesses are evaluated using LLM-generated tests without access to ground-truth solutions or official tests. Specialized agents diagnose execution trajectories, plan updates with different performance-cost priorities, and implement them as candidate harnesses. Rather than returning a single harness, HarnessPro uses normalized weighted squared-distance selection to provide cost-focused (lite), balanced (medium), and performance-focused (heavy) harnesses for different deployment budgets. Selected harnesses enter the next iteration, where execution feedback guides further updates. Our experiments cover Terminal-Bench 2, SWE-Bench Pro, and ProgramBench, using GPT-5.5 with Codex and GLM-5.2 with Claude Code. Across these benchmarks using GPT-5.5 with Codex, the heavy mode of HarnessPro achieves an average pass-rate gain of 2.0 percentage points and an average per-task cost reduction of 26.1% relative to RHO. This advantage also holds when using GLM-5.2 with Claude Code. Code is available at this https URL 

---
# recAMPLIFY: Large-Scale Pretraining of Recursive Protein Language Models 

**Authors**: Saurav Jha, Qianli Liao, Robert Vernon, Sarath Chandar, Quentin Fournier  

**Link**: [PDF](https://arxiv.org/pdf/2610.26747)  

**Abstract**: Protein language models (pLMs) are powerful tools for protein annotation and mutation-effect prediction, but progress has largely depended on scaling model size and data, making them expensive and prone to memorization. Test-time scaling offers a complementary path, but methods developed for autoregressive LLMs do not transfer directly: masked encoders lack a generation loop, and amino acids do not verbalize in a reasoning language. Instead, we adapt recursive pretraining to masked pLMs, where a weight-tied Transformer core refines residue-aligned hidden states over an adjustable number of cycles. We instantiate this recipe as recAMPLIFY, a pLM with 155M unique parameters trained on the same 2T tokens as AMPLIFY-350M. Despite 56% fewer parameters, it improves CASP contact recovery and three out of four supervised tasks, notably raising fold accuracy by 22.5%. Because optimal depth for recAMPLIFY varies across tasks and assays, we then introduce Gated Mixture Recurrence (GMR), a framework that decodes each protein from a probability-weighted mixture of hidden states visited during recursion. GMR employs a learned residual gate at scalar, low-rank, or per-dimension resolution to control how much of each cycle's proposed update is accepted into the state. This unifies halting and gating into a single adaptive computation objective while recovering PonderNet-style halting as a special case without gates. Layered on top of recAMPLIFY, GMR further reduces expected layer depth by up to 37%, although the accuracy trade-off varies by task. Mechanistic analyses show that residual gating controls recurrence stability beyond the trained depth and that halt depth tracks biological properties. 

---
# Copying Under Repetition: From Prefix Matching to Position Hashing 

**Authors**: Anxin Guo, Tianlong Huang, Chenxiao Yang, Zhiyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.26722)  

**Abstract**: Frontier LLMs can copy random text almost flawlessly, yet they often overrun or truncate long blocks of repeated characters. In repeated strings, matching local content no longer reliably identifies where copying should continue or when it should stop. Nevertheless, we show that a two-layer transformer with $O(\log\log N)$-bit precision and $O(\log N/\log\log N)$ width can copy every string up to a chosen design horizon $N$ and terminate correctly under arbitrary repetition. We use RoPE-style rotations of attention values and outputs (VO-rotations) with small prime periods to encode distances from delimiter anchors modulo those primes. By the Chinese Remainder Theorem, these periodic encodings jointly distinguish all required distances, allowing one retrieval rule to select either the next source token or the end marker. In contrast, we prove a width-independent impossibility result for copying under sequential finite-precision softmax accumulation. It applies to transformers whose only positional input is an additive attention bias that eventually becomes constant, including NoPE and a recursive implementation of ALiBi. Once an identical-token run exceeds the resulting saturation threshold, further repetitions leave the logits unchanged. If this threshold grows sublinearly with $N$, these models cannot both complete a long all-identical copy and stop correctly, a pattern resembling the termination errors of frontier LLMs. In synthetic copying experiments, baseline transformer models fail on long repeated strings even within the training range. Trained VO models, by contrast, generalize beyond the training lengths with near-perfect accuracy up to length $\approx 5005$, the selected primes' period product. 

---
# Parallel Language Models via Variational Decoding 

**Authors**: Felix Draxler, Robin C. Geyer, Daniel Geyfman, Farrin Marouf Sofian, Justus Will, Philipp Hildebrandt, Giosue Migliorini, Prakhar Srivastava, Gagan Sai Trinadh Duggaraju Peddiraju, Kushagra Pandey, Jorge da Silva Gonçalves, Elijah Tavares, Jan Niklas Groeneveld, Yang Meng, Julia E. Vogt, Stephan Mandt  

**Link**: [PDF](https://arxiv.org/pdf/2610.26693)  

**Abstract**: Autoregressive language models emit one token per forward pass. This causes high inference latency, in particular as models are scaled up for quality. In this paper, we introduce Parallel Language Models (PLMs), which generate mutually consistent tokens in one forward pass by routing dependencies through latent auxiliary variables. PLMs are trained using a variational bound that reduces to an autoregressive loss plus a parallel decoding loss. Empirically, PLMs are on a faster and better Pareto frontier than autoregressive and discrete diffusion baselines. Pretrained PLMs achieve near-autoregressive pretraining loss on model scales of 80M to 3B parameters, and speedup follows a power law in model size. When post-trained, our 1.3B model matches or exceeds an identically trained autoregressive model on math, coding, and instruction-following benchmarks, speeding up inference $2.3\text{-}5.7\times$. Together, we show that parallel generation can be trained from a single training objective at autoregressive quality. We will make our code available at this https URL. 

---
# Intra-Rubric Reward Divergence: Mitigating Rubric Hacking through Latent Dimension Recovery and Balancing 

**Authors**: Jianqiang Xia, Han Xu, Dianxi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.26685)  

**Abstract**: Rubric hacking limits both performance and training robustness in rubric-based reinforcement learning. By decomposing rubrics into sub-dimensions, we reveal the Intra-Rubric Reward Divergence: as the total rubric score increases, certain sub-dimension scores systematically decrease, suggesting the model sacrifices specific quality aspects to optimize the aggregated objective. From the sub-dimension perspective, we observe two phenomena: (1) dimension imbalance, causing the policy to favor heavily covered dimensions; (2) dimension absence, leading to a systematic gap between rewards and true quality. To mitigate these issues, we propose Fair-Race Dimension Aggregation (FRDA), which has generations compete within each dimension separately to mitigate rubric imbalance, and Residual Semantic Dimension Completion (RSDC), which mines evaluation dimensions implicitly encoded in a holistic preference signal to complete rubric coverage. Experiments in chat, science, and medical domains indicate that FRDA mitigates rubric hacking with more stable improvements, while RSDC outperforms both single-reward training and naive multi-reward aggregation, yielding generally higher-quality responses. 

---
# Which Model for Which Role? Cost-Efficient Pareto Recommendation for LLM Agentic Workflows 

**Authors**: Qian Xie, Wenyue Hua, Yueli He, Sripad Karne, Armaan Agrawal, Nikos Pagonas, Eugene Wu, Kostis Kaffes, Tianyi Peng, Nairen Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.26676)  

**Abstract**: LLM agentic workflows increasingly employ different models for specialized roles. Users may prefer different trade-offs between task quality and deployment cost, sometimes also considering latency. Finding good assignments incurs a separate search cost from evaluating multiple candidates. We introduce Cost-Coupled Gittins (CC-Gittins), a multi-objective extension of single-objective cost-aware Gittins search for Pareto recommendation. It exploits the coupling between evaluation and deployment costs: assignments that cost more to deploy also cost more to evaluate. The search alternates between seeking higher quality and lower deployment cost, reusing each evaluation to inform both searches. After each batch, CC-Gittins recommends assignments that are nondominated under uncertainty-adjusted estimates of quality and deployment cost. We evaluate CC-Gittins on eight quality--cost matrices from workflows with two to five roles and up to 1,681 candidate assignments. We also evaluate the method with latency as a third objective. Across the eight benchmarks, CC-Gittins recovers a substantial fraction of the true Pareto frontier at roughly 10% of the exhaustive-evaluation cost, and recovers most of the frontier with few false positives by 30%. 

---
# SpatialScout: Can VLM Agents Actively Explore and Map 3D Scenes? 

**Authors**: Zhaoxi Zhang, Hao Liang, Linxuan Wu, Yiming Wu, Yifu Yuan, Kun Liang, Chenming Tang, Meiyi Qiang, Wentao Zhang, Weikang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.26659)  

**Abstract**: Spatial intelligence requires agents to actively acquire visual evidence and integrate observations into a coherent representation of the three-dimensional world. Yet spatial question answering offers limited evidence of whether models can construct accurate metric maps through autonomous exploration. We introduce SpatialScout, a benchmark for evaluating vision-language model (VLM) agents on active object-level 3D mapping. SpatialScout comprises 80 diverse indoor scenes containing 3,140 target objects. Given an object inventory, RGB observations, camera calibration, and camera poses, agents control a camera to explore each scene and estimate object centers in a shared coordinate frame. The task requires coordinating viewpoint selection, cross-view association, geometric estimation, and map revision under bounded interaction budgets. Evaluation jointly measures localization accuracy and target coverage through object-scale overlap scores that penalize missing predictions. Experiments with frontier VLMs reveal substantial limitations: the strongest evaluated model, Claude Opus 5.5, achieves a mean overlap score of 41.71%, with 47.23% of targets exceeding an intersection-over-union (IoU) threshold of 0.5 and only 3.25% exceeding 0.9. These results reveal a substantial gap between recovering coarse object layouts and constructing complete, precise metric maps. SpatialScout provides a quantitative testbed for developing agents that build reliable 3D spatial representations through active perception. Code is available at this https URL. 

---
# From Outcomes to Components: A Counterfactual Evaluation Framework for LLM-Proposed Repairs 

**Authors**: Hengyu Ye, Songhua Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26650)  

**Abstract**: An LLM-generated repair action specifies which records to change and by how much. A failed action may still contain a useful target set or numerical update, depending on how the two are paired. We evaluate this problem in traffic networks and databases, where aggregate observations obscure the records that need correction. We examine the pairings by executing each combination from the same initial state, using a reference action that restores the affected records. In 76 of 114 traffic actions, the gain from correcting both parts exceeds the sum of the gains from correcting each part alone. Six of these actions become worse after either correction alone, but improve when both parts are corrected. In database repair, replacing only the proposed update with the reference update worsens 17 of 108 model actions, all of which select unaffected records. We also test whether parts of two proposals can be reused. On held-out observations, selecting among the proposals and their recombined target-update pairs recovers 61.6% of the reference action's loss reduction, compared with 58.0% from selecting between the proposals alone. 

---
# When the harness decides success: state that cannot be re-observed 

**Authors**: Raimondas Lencevicius  

**Link**: [PDF](https://arxiv.org/pdf/2610.26632)  

**Abstract**: Agent harnesses decide what a language model sees across turns: they drop old turns, summarize them, or replay the model's own earlier reasoning. We ask when these choices change whether an agent succeeds, not only what it costs, and report three findings. First, in a pre-registered synthetic probe, truncating the context window costs a low-cost reasoning model nothing in success when a needed fact can be looked up again (it looks the fact up instead), but drops its success to chance when the fact cannot be re-observed: the two penalties differ by 92.5 percentage points (90% CI [89.5, 95.5]). A pre-registered replication on one model from a second provider showed the same contrast. Once the fact was lost, that model chose no answer rather than guessing. Second, on ARC-AGI-3, where the benchmark's Standard harness trails its Provider Adapter harness by a wide margin, we decompose that gap one component at a time, for one model on four games. Summarizing old turns instead of dropping them, while keeping the Standard harness's invitation to take notes, recovers 86% of a 63-point gap without replaying the model's reasoning. The same harness without the invitation recovers 43%, and adding the provider's native replay of the model's reasoning recovers the remaining 57%. These are point estimates from one counted run per configuration and game, and the replay share depends on a single run. Third, in the benchmark's published runs and in ours, the Adapter's advantage in the probability of clearing a level it has reached is larger at deep levels than at early ones. This largely restates where runs stopped and does not separate a growing advantage from a constant one. 

---
# Beyond Attack Success Rate: Cost-Aware Metric for Evaluating Iterative LLM Jailbreaks 

**Authors**: Rohan Menon, Chih-Hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.26589)  

**Abstract**: Attack success rate (ASR) evaluates jailbreaks against large language models without accounting for repeated calls to the victim model. Yet each additional call gives the deployed system another opportunity to detect the attack and intervene. We introduce $\mathrm{ASR}_{\mathrm{STEP}}$, which penalizes additional victim-model calls while retaining binary success judgments. Our evaluation considers GCG~\cite{zou2023universal}, PAIR~\cite{chao2025jailbreaking} with a step budget, and a single-step PAP~\cite{zeng2024johnny} baseline on JailBreakBench across three model families. GCG's scores decrease by 7.0--8.7 percentage points after discounting, while PAP's remain unchanged. While for PAIR~\cite{chao2025jailbreaking}, we see a wider discounting range between 3.4-12 percentage points, showing that the "step" quantity differs per iterative strategy. Reporting both metrics makes the discount due to repeated victim-model interaction explicit. Code is available \hyperlink{this https URL}{here}. 

---
# A Fisher-Geometric View of Multi-Teacher On-Policy Distillation 

**Authors**: Liu Kaiyuan, Zhuang Ziyuan, Ma Haoxuan, Gu Yannian, Song Mingfei, Weng Rongxiang, Ye Jieping  

**Link**: [PDF](https://arxiv.org/pdf/2610.26525)  

**Abstract**: Consolidating specialized foundation models into a unified generalist is a central goal of post-training. Two approaches are in common use: model merging, which is fast but treated as a heuristic, and multi-teacher on-policy distillation, which matches distributions but demands heavy rollout compute and suffers from cross-task gradient conflict. We show that the two are one objective seen at different orders. Within a shared basin of homologous specialists, the multi-teacher on-policy distillation objective is, to second order, exactly the Fisher-merging objective. Its minimizer is therefore the Fisher merge, a classic model-merging method given by a closed-form combination of the specialists, so merging is optimal, to second order, as a closed-form stage of distillation. Building on this identification, we propose Residual-Governed Branch-and-Merge(RGBM), a two-phase consolidation procedure. Concretely, a full-parameter Fisher merge is intractable on foundation models, so Phase~1 restricts the merge to the task-vector subspace. Extending tangent-space analyses of merging, we show that the same coefficients that mix task vectors also mix implicit rewards, which lets a residual program on unlabelled prompts solve for them. The program minimizes the worst per-domain residual and reports its energy-normalized values as the domain residual spectrum. Phase~2 lets this spectrum govern what may be added: RGBM passivates domains with low normalized residuals, trains isolated branches only for domains above the spectral gap, and writes back the admitted updates with calibrated shrinkage. On two open-source consolidation suites, the Phase~1 merge alone matches or outperforms fully trained multi-teacher distillation, and RGBM improves on it further with zero extra joint-training steps, whereas every un-gated write-back, including continued joint distillation from the merge, falls below it. 

---
# RouterWorld: Scaling Conversational AI with Latency-Aware LLM Routing 

**Authors**: Jinyang Li, Denis Emelin, Simone Tedeschi, Weiwei Cheng, Cristina Morariu, Haytham Assem, Jens Lehmann, Emine Yilmaz, Angus Addlesee  

**Link**: [PDF](https://arxiv.org/pdf/2610.26511)  

**Abstract**: Conversational AI systems built on Large Language Models (LLMs) require routing mechanisms that balance answer quality and response latency at scale. However, existing LLM routing benchmarks assume full observability of all candidate model outcomes, under-emphasize latency as a first-class metric, and fragment evaluation across heterogeneous protocols misaligned with real-world settings. We introduce RouterWorld, a comprehensive benchmark for LLM routing in conversational GenAI. RouterWorld unifies 371k single-turn responses and 76k multi-turn agentic trajectories, adopts natural user interactions with a standardized LLM-as-a-judge protocol, and decouples efficiency from transient pricing via hardware-grounded latency annotations. Crucially, RouterWorld formalizes routing under partial observability through two setups: Cold-Start, which restricts learning to sparse, static logs, and Voyager, which models continuous data streams where routing decisions shape future training distributions. Evaluating 13 routing baselines, we find that most routers over-emphasize accuracy while ignoring latency under partial observability. Code and benchmark data are publicly available. 

---
# Strength Controls Change How Interventions Compare Across Sparse Autoencoder Codes 

**Authors**: Jinchen Zhang, Zheyuan Liu, Yihan Zhu, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26489)  

**Abstract**: Changing sparse autoencoder (SAE) codes can alter both intervention direction and magnitude, so comparing intervention effects requires a strength control specifying what is held equal. Norm matching and output calibration provide different controls, but whether interventions yield different effects across codes after strength matching remains unclear. We compare intervention effects under these controls for fixed feature groups in Gemma and Llama and observe: (1) Whole-patch scaling. With one coefficient per input, full-output Kullback-Leibler (KL) calibration leaves smaller mean effect gaps from the encoder-based reference intervention than total-norm matching for nonnegative least-squares (NNLS) refits. On new Llama inputs, total-norm matching loses 11 reference transfers, whereas full-output calibration preserves every transfer decision. (2) Positionwise scaling. On a small matched gender-agreement subset, positionwise norm matching leaves smaller observed mean gaps than full-output calibration. (3) Answer exclusion. Compared with unscaled NNLS, answer-excluded calibration reduces gaps in the measured subject-verb agreement comparisons but increases them for gender refits. Full-output KL calibration also reduces larger gaps for norm-matched constructed candidates, although matching coverage is incomplete in several settings. Since calibration uses the evaluated input's outputs, it serves as a diagnostic of residual disagreement. 

---
# Label Authority in Speech Emotion Recognition: Intended and Perceived Ground Truth in Emozionalmente 

**Authors**: Marco Giordano, Stefano Giacomelli, Claudia Rinaldi, Fabio Graziosi  

**Link**: [PDF](https://arxiv.org/pdf/2610.26446)  

**Abstract**: Ground truth in Speech Emotion Recognition (SER) is often presented as a single categorical fact, yet an acted emotion and the emotion heard by listeners describe different aspects of the same recording. We examine this distinction in Emozionalmente, an acted Italian corpus that provides both an elicited production label and five listener judgements for every clip. Using frozen XLS-R, HuBERT, and WavLM representations with matched probes, we compare three label authorities: the expressed condition, the listeners' hard plurality, and their complete vote distribution. The results tell a consistent story. Recovering the expressed condition is easier than predicting a tie-broken listener plurality, but that gap compares two different questions rather than revealing a general loss of SER ability. As listener agreement weakens, the plurality becomes an increasingly brittle hard target. Training on the full vote distribution does not substantially change which class wins the model's hard decision, but it produces probability assignments that more closely follow the alternatives selected by listeners, especially for ambiguous clips. No authority therefore replaces the others. Expressed labels describe how speech was produced, pluralities provide an operational single-class decision, and vote distributions preserve perceptual uncertainty. For human-facing Internet of Sounds applications, evaluations should state which authority is being predicted, retain its aggregation rule and agreement information, and use distributional scores when disagreement is part of the target. 

---
# Where Does Rhetorical Distortion Live? A Multi-Platform, Multi-Language Measurement of Influencer Content 

**Authors**: Xucheng Yu, Jiangping Chen, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26412)  

**Abstract**: Much of the manipulation in online influencer content is not false but disproportionate: an accurate fact is framed as historic, urgent, or catastrophic. Fact-checking and clickbait detection are not designed to capture such framing, and little is known about where it originates: with the creator, the topic, the platform, or the audience that replies. We operationalize rhetorical distortion through five literature-grounded dimensions (Significance Inflation, Anxiety Manufacturing, Novelty Claims, Loaded Language, and Temporal Distortion) and measure it with a three-tier pipeline in which bilingual lexical rules propose candidates, negative filters cancel known false-positive contexts, and an LLM verifies or relabels the candidates. Applying the pipeline to 13,060 posts from 330 accounts on six platforms (RSS/newsletters, YouTube, Bluesky, Reddit, Weibo, and X/Twitter), we find that platform is far more strongly associated with account-level distortion than content category (Kruskal-Wallis $\varepsilon^2=0.33$; two-way ANOVA $\eta_p^2=0.49$ for platform vs. $0.09$ for category). X/Twitter scores highest (DI 7.74 vs. 0.78 for RSS). Classifying tweets and their replies separately shows where this comes from: the reply layer readers see under a post (its top-5 replies) carries 3.5x the distortion of the tweet it answers (7.65 vs. 2.17, $p<10^{-6}$, 36 of 43 accounts), driven by Loaded Language. At equal text length the gap shrinks to 1.5x but remains significant, and creator tweets alone are statistically indistinguishable from YouTube and Bluesky. Two independent human annotators confirm the reply gap, and the platform ordering and the reply gap replicate with GPT-4o as a second classifier. On threaded platforms, measured distortion thus lives largely in the audience layer, which motivates measurement that separates creator and reply text. 

---
# LaMER: Language-Conditioned Multi-Expert Routing for Weakly Supervised Referring Expression Comprehension 

**Authors**: Yu Yu Chang, Tzu-Hsuan Weng, Youssef Mohamed Amin, Mohaned Ahmed Elsayed, Hung-Hsuan Chen, Min-Te Sun, Sherine Shawky  

**Link**: [PDF](https://arxiv.org/pdf/2610.26404)  

**Abstract**: Referring Expression Comprehension (REC) localizes the image region described by a natural language expression, underpinning applications such as autonomous navigation, human-robot interaction, and medical image analysis. Under weak supervision, REC is challenged by the visual-linguistic semantic gap and the difficulty of inferring region-text correspondences from coarse image-level signals alone. Moreover, no single visual encoder captures all complementary properties---semantic alignment, fine-grained texture, and boundary awareness---while naively fusing multiple pretrained encoders yields heterogeneous, hard-to-reconcile representations. We propose LaMER, a weakly supervised REC framework that dynamically aggregates four frozen expert encoders through a language-conditioned Cross-Modal Routing Fusion module, adapting expert selection to each referring expression without instance-level bounding-box supervision. A Cell-Text Matching Head with Negative Anchor Augmentation, trained with a multi-objective loss combining contrastive, reconstruction, alignment, and load-balancing terms, further stabilizes optimization. LaMER achieves 77.67\%, 62.81\%, and 69.49\% Acc@IoU\(\geq 0.5\) on RefCOCO, RefCOCO+, and RefCOCOg, surpassing DViN and WeakMCN by up to 16.49 and 14.49 points, with ablations validating each component. 

---
# An Empirical Study of Parameter-Efficient Fine-Tuning for Mixture-of-Experts LLMs 

**Authors**: Minglei Li, Peng Ye, Haonan He, Jingqi Ye, Qingyu Yang, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.26370)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods are critical for adapting large models under resource-constrained settings. However, as mixture-of-experts (MoE) architectures, which offer efficient inference at massive parameter scales, gain increasing prevalence in the community, a critical gap arising from the lack of systematic recipes for applying PEFT to MoE models. To narrow this gap, we conduct a large-scale systematic study that investigates a set of essential design recipes, including which modules should be adapted, which granularity should be used to adapt MoE modules, which PEFT method to use, and to what extent should the load-balancing objective be used, across three MoE backbones on downstream tasks and general knowledge tasks. Our experiments yield several key insights. First, adapting MoE experts alone achieves the highest downstream performance with competitive general capability retention under a comparable parameter budget. Second, native expert-weight updates offer a better downstream--retention trade-off than adapters that bypass experts or entire MoE modules. Third, LoRA and DoRA together define the strongest operating region for high downstream accuracy among the evaluated PEFT methods. Fourth, no nonzero load-balancing coefficient Pareto-dominates the zero baseline. We finally evaluate a weight-level redesign of MoE$^2$-LoRA on Qwen3. Moving its module-level bypass pool into native expert weights improves all four upstream endpoints by 2.4 to 17.9 points with 17% fewer trainable parameters and a 3.2-point average downstream cost. 

---
# VibeTraining-Bench: Data-Trainer-Harness Co-Design for Iterative System Improvement 

**Authors**: Ruotian Peng, Yuanyi Ren, Shuaiqi Liu, Xin Guo, Fang Zhu, Yangyi Huang, Yue Wu, Zheng Wang, Yandong Wen, Zhihang Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26342)  

**Abstract**: Recursive self-improvement (RSI) requires agents to diagnose system bottlenecks and coordinate improvements across training and inference. We refer to the agents' capability to jointly optimize training data, the trainer, and the inference harness as cross-component co-design. We introduce VibeTraining-Bench, a benchmark for evaluating this capability across tasks spanning terminal operation, software engineering, multi-turn tool use, and mathematical reasoning. Within each task, research agents start from an identical initial agent system under a fixed resource budget. Guided by validation feedback, they iteratively refine data, trainers, and inference harnesses, then select and submit a final agent system for evaluation on the test set. To prevent reward hacking, we enforce three safeguards: test data isolation, separate research and grading containers, and network restrictions for research agents. Our evaluation results show that frontier research agents substantially improve target agent systems. For example, on SWE-bench Verified, GPT-6-Astra raises the target agent system's performance from 32.9% to 53.7%, comparable to the human-developed instruct model's 55.4%. However, there is still room to improve their ability to co-design and iteratively refine data, trainer, and harness to achieve cumulative performance gains. We hope that VibeTraining-Bench will serve as a unified platform for advancing RSI, deepening our understanding of cross-component optimization for agent systems. Code and data are available at this https URL. 

---
# Typed Decision Models under Prevalence Shift: Decision Training, Calibration, and Implicit Priors 

**Authors**: Abdul Rahman Iqbal, Muhammad Fawad Nasim, Amir Saleem  

**Link**: [PDF](https://arxiv.org/pdf/2610.26325)  

**Abstract**: Typed decision models (TDMs) return a probability over caller-defined options, and Jev is described by its vendor as calibrated. But in deployment class frequencies change, so calibration measured on a fixed test set need not hold where thresholds are applied. Existing evaluations leave two gaps: the gains of vendor decision training have not been separated from those of the backbone under an identical readout, and TDMs have been tested only at the test set's class distribution. We ask whether decision training improves a model over its own backbone, and whether a TDM's calibration and thresholds withstand prevalence (label) shift as well as a fine-tuned classifier's. We compare three TDMs (Jev, Tev1-4B, Nimble-9B), their base models and two trained classifiers on identical decisions from five public datasets under simulated shift, with 35 paired tests fixed before any test was computed. First, decision training is not uniformly beneficial: Nimble-9B, which differs from Qwen3.5-9B only by a LoRA adapter, is better calibrated on intent tasks but 12.0 Macro-F1 points worse on financial sentiment. Second, under moderate shift Jev's expected calibration error (ECE) and threshold risk grow significantly more than a fine-tuned DeBERTa's only on financial sentiment, by 1.7 and 2.5 percentage points; averaged over 2,000 shift scenarios (exploratory), the gap is at most one point on every dataset. Third, EM prior correction with the model's mean development prediction as its prior needs no labels and lowers Jev's ECE under shift by 2.7 and 1.8 points on the two datasets where it is highest, although labelled recalibration lowers it far more. The three TDMs we tested are neither uniformly better than their backbones nor uniquely fragile under shift; for calibration, their main weakness is the miscalibration they already have without shift. We release all inputs, code and outputs. 

---
# The Scheduler Is a Hidden Retriever: How Asynchrony Shapes Evidence Acquisition in Search Agents 

**Authors**: Ruiyang Ren, Yuhao Wang, Yinan Zhang, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2610.26310)  

**Abstract**: Language-model agents increasingly solve complex tasks through parallel reasoning and asynchronous information gathering. Controlled replay across two model families and three benchmarks shows that delivery timing redirects search decisions made before all results arrive, despite identical initial evidence. Acquisition disagreement reaches up to 24 percentage points above same-order repeat variation. This dependence persists after all initial evidence arrives and through a second search, turning temporary differences in what agents have seen into lasting differences in what they acquire. An acquisition-regret bound with a matching construction shows that invariance at a fixed early arrival count can limit the use of evidence already received. We introduce Confluent Evidence Acquisition (CEA), which instead releases local reasoning when each subquestion's evidence is complete and allocates shared search independently of completion order. For a fixed plan and stable conditional response law, we prove joint invariance of decisions, acquisitions, and answers. With question-only plans across three benchmarks, CEA brings acquisition disagreement to within 1.0 percentage point of repeat variation and achieves accuracy comparable to complete-input controls, improving over early commitment by up to 12.6 percentage points. Replacing CEA's fixed allocation with first-ready allocation under contention adds 21.6 percentage points of acquisition disagreement, while local query-target disagreement remains near repeat variation. Timed comparisons show that CEA reaches decisions earlier and responds faster than the same program waiting for all initial evidence. 

---
# AudioScorer: Learning Question-Aware Token Relevance for Large Audio Language Models 

**Authors**: Marwan Farag, Bin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26258)  

**Abstract**: Large Audio Language Models (LALMs) reason over speech, environmental sounds, music, and other acoustic information. However, their audio encoders produce dense token sequences that make long-form audio costly to process. Existing audio pruning methods reduce this cost using heuristics such as attention, representation similarity, or temporal sampling, but do not explicitly learn which tokens are most useful for a given question. We propose AudioScorer, a lightweight model that scores audio tokens according to the user question. It is trained by distilling the ranking of audio tokens under the frozen LALM's answer-gradient saliency, together with downstream answer supervision. Compared with other pruning methods, AudioScorer achieves the best or tied-best result at 10% token retention on 7 of 10 semantic and acoustic tasks, including speech recognition, emotion recognition, and audio-scene classification. 

---
# Machina Pauliana: A neuro-symbolic approach to the actio pauliana in Dutch insolvency law 

**Authors**: Daan Boven, Gregor Behnke, David Graus  

**Link**: [PDF](https://arxiv.org/pdf/2610.26241)  

**Abstract**: Applying a legal provision frequently requires both structured rule-based reasoning and sensitivity to open-textured terms developed in case law. The actio pauliana of Article 42 of the Dutch Bankruptcy Act (Faillissementswet, Fw), which allows an insolvency practitioner to avoid transactions that prejudice creditors, requires both. We present a neuro-symbolic model of Article 42 Fw. Its symbolic core is written in Catala in a literate style that keeps statute and formalisation aligned and was developed through pair programming, with open-textured terms represented as boolean inputs where neural components couple to the symbolic solver. A neural routine structures a case's natural-language facts into boolean and date inputs. Neural subroutines resolve the open-textured terms through case-based reasoning over Dutch civil judgments, retrieved through a pipeline that filters on the requirement being resolved and expands along the procedural chain of a case. Segmenting and labelling those judgments attains high precision and recall against a manually annotated gold standard of 100 judgments. On ten hand-annotated cases the neuro-symbolic model handles most criteria reasonably but struggles with the knowledge requirement. Retrieval depth shows no clear quantitative effect, although retrieved judgments provide explicit grounding in prior case law. 

---
# Decision4Rec: A Jev-Inspired Framework of Decision Models on Knowledge and Active Graphs for Multi-Agent Recommender Systems 

**Authors**: Marco Polignano, Het Darshan Mehta, Pasquale Lops, Ernesto William De Luca, Giovanni Semeraro  

**Link**: [PDF](https://arxiv.org/pdf/2610.26035)  

**Abstract**: Decision models, such as the commercial Jev, are language models that return one probability for each option of a question instead of generating text. We present Decision4Rec, an open and modular framework that makes decision models useful for recommendation. A knowledge graph writes the logs of each recommendation scenario as short statements on a defined scale, such as "sold: top 10%". A pool of agents, proposes candidates, and a residual ranker keeps the best 30. A decision model, trained once on real choices of three scenarios, reads them through one interface of typed questions, and a fusion step combines its answers with the score of the ranker. Any model with this interface can be used, and we run the commercial Jev, the released open Kev and Kev models trained by us on recommendation requests. The framework with a trained 4B decision model is 25 to 36% above the best trained baseline in NDCG@10 in three sparse Amazon catalogues, one of them never used to train the decision model, and almost tie with baselines on the dense Steam dataset. Most of the gain comes from the pool, and with a trained recommender it comes from what the agents know about its candidates rather than from new candidates. The 27B decision model adds to the pool 5 to 6% above the 0.8B model in three of four domains. A trained 0.8B open model beats the released Jev in eleven of fifteen settings and is never worse. On MovieLens-1M, a dense domain never seen in training, the gain is 5 to 7%. Code, configurations and trained models are available at this https URL and this https URL. 

---
# Understanding When Reflective Prompt and Skill Optimization Helps on Locally Served Reasoning Models: A Two-Model Study of GEPA and SkillOpt 

**Authors**: Andrea Giummarra, Carmelo Baglieri, Antonio Di Maria, Andrea Farruggio, Michele Piatto, Giovanni Micale  

**Link**: [PDF](https://arxiv.org/pdf/2610.26020)  

**Abstract**: Reflective text-space optimizers such as GEPA and SkillOpt have reported large gains on frontier and mid-size models. We test replication across two reasoning-native target-serving configurations: DeepSeek V4 Flash through DwarfStar and Qwen3.8-27B through LM Studio. For each target, we evaluate the same held-out tasks and four conditions - minimal baseline, GEPA prompt, SkillOpt skill, and manually engineered prompt - on HotpotQA (n=500 per condition) and MATH-500 (n=250). In the primary Qwen v2 rerun under the same frozen protocol, GEPA selects the HotpotQA seed and exactly reproduces baseline behavior, whereas SkillOpt's validation gain reverses into held-out degradation. On MATH-500, GEPA raises exact match from 0.828 to 0.904 while reducing latency but increasing token consumption, so lower response time does not imply lower token cost. DeepSeek likewise shows its strongest GEPA effect on MATH-500 (0.852 to 0.908). Across both targets, GEPA's strongest gains occur on MATH-500, while neither optimizer improves the HotpotQA baseline. Optimization is therefore task- and method-dependent, and validation gains may reverse on held-out evaluation. An earlier Qwen run is retained as serving-sensitivity evidence. These results suggest that textual headroom for a model-task pair and the observed effect of a text-space optimizer are more informative when decomposed into content/reasoning, answer-realization/evaluator alignment, and serving-completion/reliability. The cross-benchmark pattern further suggests that optimization may benefit from low-ambiguity, actionable evaluation signals. 

---
# Correctness-Dependent Asymmetric Target Distributions for Self-Distillation Policy Optimization 

**Authors**: Suwan Yoon, Seongryong Jung, Changhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.25973)  

**Abstract**: Self-distillation policy optimization provides dense token-level supervision for LLM post-training through feedback-conditioned self-teachers. However, directly matching the self-teacher can disrupt successful behaviors on correct rollouts while providing insufficient correction on incorrect ones. We propose Asymmetric Policy Optimization via Self-Distillation (APSD), which constructs correctness-dependent targets by mixing student and self-teacher distributions: interpolation preserves successful behaviors on correct rollouts, while extrapolation beyond the self-teacher strengthens correction on incorrect ones. We analyze APSD through divergence decomposition, reverse-KL barycentric geometry, and gradient analysis, highlighting the importance of gradient flow through student-dependent targets. Experiments under reinforcement learning with verifiable rewards and rich feedback show that APSD achieves strong overall performance against reinforcement-learning and self-distillation baselines and matches SDPO performance with 2.6x fewer training steps on average across Qwen3 model scales. 

---
# GLOBE: Geometry-Led Online Behavioral Early Exit via Unit-Sphere Trajectories 

**Authors**: Xinyue Peng, Zhenlin Liang, Meng Cai, Hua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25970)  

**Abstract**: Chain-of-thought (CoT) reasoning can improve problem solving, but models often keep generating after the correct answer is already reachable; the extra steps raise serving costs and may replace a correct intermediate result with an incorrect one. We propose GLOBE (Geometry-Led Online Behavioral Early Exit), an online, training-free algorithm that terminates the model's CoT process early. GLOBE embeds each completed reasoning segment as a point on a unit sphere and reads the shape of the resulting semantic trajectory. GLOBE first applies a trial phase that generates short trial answers in a separate branch, stopping the main stream on an accepted trial after checking confidence against a threshold. If generation continues, a trajectory phase stops the stream once this path repeatedly reverses direction with little net progress. On five benchmarks including mathematical reasoning, tool calling, and agent memory, GLOBE cuts output tokens by 49.7% and active end-to-end wall time (E2E) by 51.0% across two models, and achieves a five-task average of 80.58%, comparable to Full CoT. 

---
# Token-Set Choice Confounds POPE: A Systematic Audit of Yes/No Extraction in VLM Hallucination Evaluation 

**Authors**: Kesav Kumar Jayakumar, Karthigeyan Thilak  

**Link**: [PDF](https://arxiv.org/pdf/2610.25964)  

**Abstract**: We set out to reproduce a recent hallucination-mitigation result on the POPE benchmark. The LLaVA-1.5-7B baseline we measured, 82.21 F1 on the adversarial split, sat 1.21 to 8.67 points above the adversarial baselines we planned to compare against (81.00, 79.30, and 73.54). We looked at the evaluation code and found a readout choice that moves F1 on its own. A two-token logit readout decides yes versus no by comparing two vocabulary IDs, 3582 ("yes") and 1217 ("no"). Across 9,000 greedy-decode questions spanning all three splits, neither ID is generated even once. The model emits " Yes" (token 3869) and " No" (token 1939), because SentencePiece prepends a space byte after the prompt suffix "ASSISTANT:". Read with an eight-token lookup that covers the space-prefixed surface forms, LLaVA-1.5-7B scores F1 0.8221, 0.8498, and 0.8713 on the adversarial, popular, and random splits. That 6.13-point adversarial shift exceeds the adversarial gain reported by five of the six papers we surveyed. None of them states its readout, and our reference ran under 4-bit NF4 quantization, so we cannot say how much of any reported gain a matched evaluation would keep. That comparison stays unresolved until readout and precision are matched. On the fixed baseline we run nine inference-time corrections of our own (none beat it) and dissect why the adversarial split resists logit-level intervention. In a 3,000-question probe, wrong predictions draw more image attention than correct ones (Cohen's $d=-0.46$, $p=1.4\times10^{-24}$), the opposite of what grounding-based corrections assume. A four-readout audit across four further VLMs from three tokenizer families confirms the confound is protocol-level, not model-specific: porting the LLaVA-1.5 IDs verbatim drops F1 to 0.00 on the two disjoint-vocabulary models, whereas a tokenizer-derived readout holds between 0.80 and 0.92. 

---
# Keep the Lineages Alive: Diversity-Preserving SMC for Diffusion LLM Decoding 

**Authors**: Minoo Ahmadi, Seyedarmin Azizi, Mehdi Kamal, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2610.25932)  

**Abstract**: Inference-time sequential Monte Carlo (SMC) decoding can improve the reasoning accuracy of masked diffusion language models (diffusion LLMs) without post-training by resampling a population of partial generations according to a reward. Its effectiveness therefore depends on the resampling step, which must concentrate computation on promising candidates while retaining enough diversity to explore alternative reasoning paths. The reward can be internal, the decoder's own confidence on the committed prefix, or external, a process reward model scoring that prefix. Under either reward, multinomial resampling can disrupt this balance by duplicating high-weight particles and discarding lineages that may still reach a correct solution. The effective sample size (ESS) does not register this loss: multinomial resampling resets it to its maximum even as distinct lineages disappear. To retain such lineages, we introduce lineage-preserving SMC (LP-SMC), which combines Chopthin resampling with an input weight floor. Instead of equalizing weights, Chopthin bounds the ratio between the largest and smallest weights it returns and carries the unequal weights forward. The floor limits how far below the maximum a weight may fall before resampling so that lineages with low early weight keep a chance to recover. Across mathematical-reasoning and code-generation benchmarks on two base models, LP-SMC raises oracle coverage, the fraction of problems with at least one correct particle, in every setting, by 1.8 to 25.6 points, and raises end-to-end accuracy under a common reward-model selector in eleven of twelve settings, by up to 7.9 points on LLaDA-1.5 and 18.3 on Dream-7B. The coverage gain persists under the external reward model, which suggests that diversity-preserving resampling and the guidance signal contribute separately to inference-time search. 

---
# EvoSkill v2: Fully Agentic Skill Evolution from Benchmark Adaptation to a Stateful Evolver 

**Authors**: Dastin Huang, Baran Nama, Salaheddin Alzubi  

**Link**: [PDF](https://arxiv.org/pdf/2610.25931)  

**Abstract**: EvoSkill v2 is a three-agent system that edits the instructions (skills) an agent is given, leaving its model and harness intact. The user starts a run by describing a benchmark and supplying a grader, which a benchmark adaptor turns into a benchmark-specific plugin. A solver attempts the tasks, and an evolver reads the solver's training rollouts and rewrites the skills at each epoch. We implement the evolver as one stateful coding agent with a bash tool for the whole run, unlike the two most common designs among the skill optimizers we surveyed: a sequence of single model calls, as in SkillOpt, and an agent that starts afresh each round, as in SkillBoost. We also measure how much the solver's rollouts vary from one attempt to the next, give the evolver several attempts per training task, and accept every epoch's skills instead of selecting among them on noisy scores. With fine-grained ground truth, EvoSkill v2's skills raised the held-out pass rate above the stronger of SkillOpt and SkillBoost by 7.7 percentage points on SpreadsheetBench v2 and by 5.4 on $\tau^3$-bench banking. In both benchmarks and both feedback settings, EvoSkill v2's evolver cost less per percentage point of held-out pass rate gained than either baseline's optimizer. EvoSkill v2 also enables a less capable model to improve its own skills, demonstrated by a gain of up to 21.5 percentage points in the $\tau^3$-bench banking experiment where Claude Haiku 5.5 powers both solver and evolver. 

---
# SWE-Residual: Evaluating Starting-State Robustness of Coding Agents 

**Authors**: Jinze Li, Weijie Liu, Jinfeng Xu, Shuo Yang, Xin Yang, Yang Zhang, Yijin Wang, Zifei Shan, Edith C.H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2610.25899)  

**Abstract**: Software engineering (SWE) benchmarks are the primary way to measure the capabilities of coding agents: an agent is placed in a code repository or machine environment, given a request, and judged by hidden tests. These benchmarks almost always start the agent from a clean state that contains no earlier work on the task. In practice, however, agents often continue work that a developer or another agent has already done and reported as complete, with no indication of what remains incorrect or missing. We introduce SWE-Residual, a benchmark for Starting-State Robustness: the ability to maintain performance on a task when the starting state already contains earlier work on that task. SWE-Residual contains 198 Residual States drawn from 111 tasks in DeepSWE and Terminal-Bench, where the earlier work modifies a code repository or a whole machine environment. Each Residual State is the state a predecessor agent left behind after submitting its work as complete, although the hidden tests still fail. The evaluated agent receives only the original request and this state, without the predecessor's trajectory or any sign that the attempt failed; because the request and hidden tests are unchanged, the starting state is the only difference from the source benchmark. Across five frontier agents, success drops by 20.6--53.5 percentage points on DeepSWE and 20.6--34.9 points on Terminal-Bench relative to a Clean Start, and agent rankings on DeepSWE reverse. Failed continuations show a pattern we call acceptance bias: agents treat the inherited work as the definition of what remains to be done instead of checking it against the request. On DeepSWE, in 58.9--88.2% of failed runs, they submit with exactly the same tests failing, although they pass most of these tests from a Clean Start. SWE-Residual thus exposes a capability gap that clean-start evaluation does not capture. 

---
# Budgeted Attack Scheduling for LLM Red-Teaming: The Roles of Static Priors, Semantic Guidance, and Online Feedback 

**Authors**: Zhicheng Fang, Han Sun, Jiachen Lei, Xinyi Fang, Zongmin Zhang, Chenxu Fu, Sizhe Fang, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25897)  

**Abstract**: As jailbreak attacks against large language models (LLMs) become increasingly diverse, red-teaming has come to rely on large libraries of jailbreak prompts and attack strategies to evaluate the robustness of victim models and their defense settings. However, attack effectiveness is highly condition-dependent: an attack that succeeds under one victim-defense setting may fail under another, while exhaustively evaluating the full attack library for each setting incurs substantial token and monetary costs. This motivates studying how jailbreak attacks should be prioritized under a limited attempt budget for efficient robustness evaluation. In this work, we formalize this problem as budgeted attack scheduling over a fixed attack library and investigate three complementary signals under a fixed victim-defense condition: (1) static historical attack success rate (ASR) priors; (2) semantic guidance from LLMs; and (3) online feedback from earlier queries. Building on these signals, we design four scheduling strategies and evaluate them across three victim models and three defense settings on GPT-5.2. Extensive experiments clearly characterize the conditions under which each signal is most effective. Historical ASR provides a strong initialization, but its reliability degrades under defense-induced distribution shifts. Semantic guidance consistently improves or matches static prior-based scheduling in the cold-start regime, while online feedback becomes particularly valuable when transferred priors become stale. Semantic guidance and online feedback can also partially compensate for missing historical priors. Taken together, these findings characterize budgeted red-teaming as an attack scheduling problem in which historical priors, semantic guidance, and online feedback play complementary roles. 

---
# When Retrieval Partially Conflicts with Skills: Understanding Procedural Hallucination in Language Models 

**Authors**: Guanqun Jiang, Pengfei Cao, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.25827)  

**Abstract**: Language models increasingly use skills that provide reusable instructions and procedures for task execution. After a relevant skill has been selected, the model may need to apply its governing procedure together with retrieved task context. When the retrieved context is useful but contains cues that violate the skill, the model must retain compatible information while rejecting the conflicting cues. We study failures in which the model adopts such cues and produces a retrieval-supported but skill-invalid answer, which we call procedural hallucination. We introduce SourceClash, a matched benchmark with 1,271 bundles and 6,355 prompts across ten procedural domains. Each bundle fixes the task, skill specification, and answer space while varying the relation between the skill and retrieval. Across nine models, valid-answer rates average 25.7% under conflicting retrieval, compared with 78.2% and 72.8% under two supportive retrieval settings. Mechanistic analysis of open-weight models shows that the answer-preference difference typically reaches its maximum shortly after generation begins. Residual replacement moves the model toward valid answers, while attention decomposition attributes most of the repair to value vectors. Intervention at the identified value pathway also improves final answer validity. These results characterize how partly conflicting retrieval changes skill-guided generation. 

---
# Standing on the Shoulders of Giants: Enhancing LLM Agents for Automated Fact-Checking with Human Expert Skills 

**Authors**: Xinwen Chen, Haorui He, Yupeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25777)  

**Abstract**: Automated fact-checking (AFC) systems powered by large language model (LLM) agents still face challenges in evidence retrieval and claim verification. Existing approaches improve their capabilities through computationally costly post-training or human-in-the-loop feedback that is difficult to scale. To transfer professional fact-checking expertise to LLM agents efficiently, we distill publicly available guidelines published by Agence France-Presse (AFP) Fact Check into a reusable, modular Professional Fact-Checking Skill (PFC-Skill). We then present SkillCheck, the first AFC system that gives LLM agents dynamic access to PFC-Skill to guide evidence retrieval and claim verification at inference time. Extensive experiments demonstrate that SkillCheck improves upon skill-free DEFAME baselines across multiple LLM backbones, with gains of 2.2--4.8 pp. in Accuracy and 1.5--6.3 pp. in Macro-F1 on the AVeriTeC benchmark. Notably, SkillCheck with GPT-4o-Mini exceeds DEFAME with GPT-4o by 3.4 pp. in Accuracy while cutting inference cost by roughly 61%. The performance gains also extend to open-web retrieval. Ablations show that incorporating the skill during evidence retrieval provides most of the gains, while incorporation into both stages yields the best overall performance. Case studies show that PFC-Skill enables more targeted search and closer scrutiny of evidence. To conclude, our work shows that LLM agents can "stand on the shoulders of giants" by drawing on the best practices of professional fact-checkers, substantially enhancing AFC performance without extra training or continuous human supervision. 

---
# Computation, Not Memory: Chain-of-Thought and State Tracking in LLM Agents 

**Authors**: Pedram Mouseli, Massieh Moayedi  

**Link**: [PDF](https://arxiv.org/pdf/2610.25760)  

**Abstract**: Language-model agents often reason before they act, and the value of monitoring that reasoning depends on how it relates to their actions. We ask how an agent's actions depend on its reasoning when the correct action depends on a changing state. In a small file-system task, background events announce which file's lock has been toggled but not its new value. The agent therefore has to reconstruct the current locks from its history before it can move files safely. With Qwen3-8B and Qwen3.5-9B, we separate two roles of reasoning: generating it before each action, and retaining earlier reasoning in the prompt. Most of the reduction in lock-violating commands comes from reasoning written at the moment of action. Retaining earlier reasoning adds little to rule-following, although it helps one model complete the task. The benefit depends on what the reasoning says. Filler text, reasoning written for another history and an accurate table of the state do not reproduce it. When we give the agent reasoning written for a history that differs in one lock value, the agent acts on that reasoning even against its own history. Without reasoning, linear probes recover the state from the model's activations about as well as a text classifier recovers it from the transcript. However, decoding accuracy does not predict which interventions change the agent's actions. In this setting, the benefit of reasoning for rule-following comes mainly from generating it at each step. The state that can be decoded without reasoning is a limited guide to how the agent acts. 

---
# Whose Credential Is It? How Owner Attribution in Agent MemoryShapes Cross-User Account Misuse 

**Authors**: Hau-Sing So, Daixian Liu, Carl Hou, Yuhaohang He, Mengzhu Wang, Hai-Tao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.25707)  

**Abstract**: Shared long-term memory lets an LLM agent serving one user inadvertently retrieve another user's credentials, but it is unclear when such exposure leads to actual harm. We introduce a three-stage audit that separates three events: a foreign credential reaches the prompt (exposure), the agent acts with it (behavior), and the other user's account changes (state). In AppWorld, a store that mixes 763 past logins from all users retrieves another user's credential for 84.7 percent of required apps, yet agents rarely act on it. Misuse appears in a stress test where memory lacks the user's own credential and presents a single foreign one as the user's record: seven agents then use it in 80 to 94 percent of runs and change the victim's account in up to 43 percent, sometimes passing the official check. Naming the owner before or inside each credential line prevents most of this: with all else fixed, misuse falls from 92 to 1 of 305 paired runs, also with name-free usernames, though owner-bound notes cost 3 to 5 successful tasks per model on clean memory. A wrongly written owner largely restores misuse; trailing owner tags and added safety instructions help only some models. Agent memory should consistently record whose credential it stores, and evaluations should measure behavior, not only exposure. 

---
# Cross-Pollination Retrieval: Finding Transferable Ideas Across Fields in Scientific Literature 

**Authors**: Seokwon Jung, Yerang Kim, Yejin Kim, Alexander Rubinstein, Arnas Uselis, Sangwoo Park, Seong Joon Oh, Sangdoo Yun  

**Link**: [PDF](https://arxiv.org/pdf/2610.25678)  

**Abstract**: Many scientific advances come from cross-pollination, carrying an idea from one field over to a problem in another. Yet retrieving papers across fields remains underexplored, with existing benchmarks confined to a narrow range of fields. We study cross-pollination retrieval, the problem of identifying transferable ideas across research fields, and introduce CPR-Dataset, a large-scale dataset constructed by mining cross-topic citations. Its evaluation split CPR-eval has 1,300 queries in 26 fields, and its training split CPR-train has 10,573 pairs. On CPR-eval, the best of eleven existing retrievers reaches a Recall@10 of only 0.26, whereas an LLM reranker reaches 0.81. Standard contrastive fine-tuning on CPR-train raises Recall@10 to as high as 0.70. The gain carries over to domains held out from training and to analogy and reasoning-intensive retrieval. On AnaloBench, it raises nDCG@10 by up to 33% over the untrained retriever, whereas training on three existing retrieval datasets lowers it. On BRIGHT, it raises nDCG@10 by up to 70%, more than twice the gain from those datasets. CPR-Dataset is available at this https URL, and the code at this https URL. 

---
# Curriculum Learning Teaches LLMs Long-Horizon Implicit Reasoning, Up To A Limit 

**Authors**: Jimson Huang, Maxwell Nickell, Ruowang Zhang, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2610.25612)  

**Abstract**: LLMs "reason" by externalizing intermediate steps as tokens through chain-of-thought (CoT), yet even frontier reasoning models hit abrupt performance cliffs on long-horizon tasks. Implicit reasoning (reasoning within a single forward pass) is useful during CoT generation, as the model can "search ahead" to decide the correct next move without explicit search. Models with weak implicit reasoning are more likely to commit to wrong paths and must attempt to recover via backtracking, while models with strong implicit reasoning identify the correct proof strategy directly, shortening the CoT. However, prior work shows LLMs struggle to reason implicitly beyond shallow depths, with no relief from scale. We investigate whether curriculum learning can extend this depth by fine-tuning on examples with progressively increasing vocabulary size and lookahead (the number of steps a model must search ahead to identify the correct next move). At matched compute, curriculum-trained models achieve lookaheads where no-curriculum training fails. Qwen3-0.6B reaches 98% accuracy on problems with lookahead up to 256 with curriculum, compared to 45% without. However, progress slows as fine-tuning continues, following a saturating exponential that suggests a maximum intrinsic lookahead to which models can learn to reason implicitly. Larger models reach higher fitted ceilings, fine-tuning pretrained models far outpaces training from scratch, and curriculum step sizes that are too large stall progression. Finally, the capability acquired on synthetic data transfers to natural language reasoning, with curriculum-trained models outperforming models trained without curriculum at matched compute on 11 natural language reasoning benchmarks. 

---
# Text Dominance in Paralinguistic Conflict: Tracing Suppression Through Audio LLM Circuits 

**Authors**: Ankur Banerjee, Vishnu Guntakala, Danity Pike, Advay Roongta, Kevin Yang, Ruizhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25599)  

**Abstract**: Audio LLMs deployed in settings such as crisis hotlines and customer support are only useful beyond speech-to-text pipelines if they use acoustic cues rather than the transcript alone. We test whether they prioritise transcript content over paralinguistic cues (emotion, speaker identity, sarcasm) and trace the mechanism when they do. Across six audio LLMs under LISTEN-style conflict conditions, the mean text-dominance rate on mismatched text+audio items is 0.78 (0.86 excluding sarcasm). In the most text-dominant screened model, probing shows speaker identity remains linearly decodable at ceiling at the language model's input, so the failure is not perceptual. Back-patching, reported to repair lexical text dominance, has no measurable effect here; but it also fails on its own source task in our replication (n=300), so we treat its null as uninformative rather than as evidence of non-transfer. Attention knockout partially restores audio-grounded output by blocking the claim from propagating through intermediate tokens during open-ended generation, whereas blocking the answer position's direct connection to the conflicting claim has no effect. The claim's influence therefore travels through an intermediate relay rather than a single severable edge. The effect appears only in free-form generation and not in closed-set multiple choice, suggesting that benchmarks relying on that format may be systematically blind to interventions of this kind. Code and data are publicly available at this https URL. 

---
# Narrow Decision Models: A System-1 Architecture for Calibrated, Typed Decisions 

**Authors**: Carlos André Antunes, Aldry Albuquerque, Gabriel de Melo Schmith, Natã Batista, Pedro Oziel Escobar da Silva, Stefano Benko Rossi, Lingrid Dietrich Lorenz da Silva, Flávia Cristina da Costa, Thiago Bertotto, Luis Fernando Fausto  

**Link**: [PDF](https://arxiv.org/pdf/2610.25553)  

**Abstract**: System One models, introduced with TypeSafe's Jev, answer typed questions about a state with calibrated probabilities instead of generated text: choosing among options described in natural language (Choice), placing the state on an ordered scale (Score), or judging a yes/no statement (Noul). Jev is general-purpose: one set of weights, served as an API and adapted to a domain only through the request. We study the other end of the design space. A Narrow Decision Model (NDM) offers the same interface but is trained for the decisions of one organization, is small enough to run on its own infrastructure, and has an open architecture whose properties can be stated and tested. The state can be any text or structured record about which decisions are made; our running case is the routing of court communications. The architecture serializes a request as a prefix tree (the state, then each question, then each option) and runs a bidirectional encoder under a block attention mask with logical positions, so that each answer is, by construction, invariant to the order of its options and independent of the other questions asked. Routing through an organization chart is a composition of Choice decisions with an explicit "none of these" option. We describe training with proper scoring rules through a curriculum, calibration and abstention, decisions composed by rubrics, and a protocol that accepts or rejects the mechanism before held-out data is opened. A minimal validation on public and fictitious data shows that the invariances hold to rounding in a trained model, that a small instance routes on organization charts it has never seen well above lexical retrieval, and that, at equal compute, the block architecture reaches a lower loss than a pairwise cross-encoder. A first instance trained on real legal documents agrees with the LLM classifier in production on 91% of held-out documents. 

---
# Auditing Tool Traces in Agentic LLMs: Source Localization and Counterfactual Dependence Testing 

**Authors**: Junhao Liu, Tianxiao Li, Yongxian Wu, Ziheng Duan, Ren Li, Jing Zhang, Martin Renqiang Min  

**Link**: [PDF](https://arxiv.org/pdf/2610.25540)  

**Abstract**: Agentic LLMs increasingly support scientific prediction by grounding decisions in external evidence. In biology, they promise accurate, interpretable predictions for important problems by integrating complex experimental and computational evidence. However, serialized tool results can contain hidden label shortcuts, such as dataset identifiers or template artifacts, that create apparent success without biological reasoning. To address this risk, we introduce a two-stage audit. Stage 1, Per-Tool TF-IDF Decomposition (PT2D), localizes label-predictive patterns to individual tool results while avoiding signal dilution in the concatenated tool trace. Stage 2, Token-Class Redaction, counterfactually tests whether the frozen LLM depends on the localized shortcut. We apply this audit to TCR-pMHC binding, where tool augmentation substantially outperforms a no-evidence baseline. At face value, this suggests that the LLM has learned to use scientific evidence. The audit overturns this interpretation: PT2D localizes the strongest signal to identifiers in one tool result, and redacting them reduces performance to chance. These findings show how hidden label shortcuts can masquerade as scientific reasoning or discovery, motivating source-level audits before benchmark gains are trusted. 

---
# ATEST: Adaptive Resource-Aware Test-Time Scaling for LLM Inference 

**Authors**: Mobina Kashaniyan, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2610.25519)  

**Abstract**: Test-time scaling (TTS) improves large language model (LLM) reasoning by sampling multiple answers, but fixed sampling budgets ignore both query difficulty and the changing energy cost of inference under GPU contention. We measure up to a 4.1-fold increase in energy for the same four-sample query as GPU load rises. We present ATEST, a resource-aware TTS framework that jointly considers the expected value of additional samples and their current GPU cost. ATEST-Online, its default mode, requires no offline calibration: it incrementally generates samples, estimates the stability of the plurality answer, monitors live GPU conditions, and learns step energy cost online to decide when to stop. ATEST-Calibrated instead selects a per-query budget using offline accuracy and cost models, favoring deployments with stable loads. We evaluate both modes on four reasoning and question-answering datasets using Phi-3-mini on an NVIDIA A100 under controlled varying load, and test transfer to an NVIDIA L40S and Qwen2.5-3B. Across the four A100 datasets, ATEST-Online answers every query without exceeding the energy budget, while achieving the highest or statistically tied accuracy and reducing sampling as load increases. On GSM8K at high load, it achieves 60.0% accuracy with no deferrals, versus 44.0% accuracy and 42.4% deferrals for ATEST-Calibrated. Ablations show that online cost learning drives adaptation to load. Including offline profiling, ATEST-Online uses 42 times less GPU time and 48 times less energy than ATEST-Calibrated for the evaluated workload. Online control trades higher tail latency and lower throughput on short tasks for robustness under varying load; offline calibration remains competitive under stable conditions. 

---
# From Seeing to Reporting: A Theory of Reliable Visual Set Generation in Vision-Language Models 

**Authors**: Junzhe Shi, Yuan Gan, Shida Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25503)  

**Abstract**: Vision-language models (VLMs) are asked to report what they see: to count objects, box them, parse a screen or pick a click target (GUI grounding). We argue that reporting is part of the visual task, not merely its formatting. In a dense scene with 173 shapes, GPT-6.1 lists 173 records that match every shape, yet the counts it states in the same response sum to 188. From Seeing to Reporting (FSR) is a theory of visual reporting and a set of inference-time procedures that improve the reports of frozen VLMs. It models reporting as a chain from responses to decoded records, constructed sets and task answers, linked by task-compatible matching. The chain gives exact relations for deleting records, averaging their coordinates and switching readouts, and it splits the error of one answer chosen among several calls into missing candidates, lost representatives and wrong choices. These relations guide procedures for counting, detection, document layout, screen parsing and GUI grounding that use the task specification and a shipped configuration, without requiring labels from the target task. On 50 dense synthetic scenes, a one-call reporting-and-decoding protocol cuts per-type count error by 85.2% for GPT-6.1 and 98.8% for Claude Sonnet 5.5. On 600 ScreenSpot-Pro items, a GUI-grounding tier that combines three models lowers the miss rate from 13.67% for GPT-6.1's two-stage zoom to 9.33%, against 12.67% for three such zooms combined by a cluster vote at about the same cost. A plain vote over the same three models misses 12.33%: the other models supply the candidates, and selecting among them turns those candidates into fewer misses. We release the protocols, the stored responses and the analysis code. 

---
# A failure of Imagination: VLMs fail to use visual representations for text, but self-distillation can help 

**Authors**: Shashwat Singh, He He, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2610.25470)  

**Abstract**: Vision-language (VL) models are typically built by fine-tuning a language model to attend to visual features in service of a Visual Question-Answering objective. We ask whether this alignment changes how language models represent text---especially on tasks where visualization would help, like geometry questions posed in words. We distinguish weak multimodality (the model can answer questions about an input image) from strong multimodality (the model's text representations themselves become vision-aligned, improving performance on text-only tasks with latent visual structure). To test for strong multimodality, we construct a task of shape-counting problems with paired equivalent text and image specifications. This task is trivial when the image is provided, but requires visualization when specified in text as line coordinates. We compare instruction-tuned Qwen-2.5 and Mistral models against their vision-aligned counterparts on the text variant of the task. We find that the multimodal models do no better in this setting, despite strong performance on the image version of the task. Probing experiments corroborate this: vision-aligned decoder embeddings carry no more visual structure than their text-only counterparts. Finally, we provide an optimization objective that distills in-context visual embeddings into text embeddings. This alignment induces gains on the text variant of the task without explicit task supervision, even on samples more complex than seen during alignment. This suggests that VQA-style alignment leaves available cross-modal transfer untapped. 

---
# System X: An LLM Decision Model That Replies from Its Own KV Cache 

**Authors**: Syed Usama Khalid Bukhari, Syed Safwan Khalid, Muhammad Aamir  

**Link**: [PDF](https://arxiv.org/pdf/2610.25456)  

**Abstract**: Decision models return probabilities over supplied options, but a deployed agent may also need to reply in words. The present study introduces System X, a large language model (LLM) decision model that adds a text-reply option to the LLM-as-Jev readout. When this option wins, or confidence is low, the same model continues from the decision's key-value (KV) cache, avoiding a repeated prompt prefill. Qwen3-0.6B has been fine-tuned with LoRA on Reason-9k and Bitext reply rows. JevBench accuracy shows no statistically significant change (55.4% to 54.5%, p=0.87), while expected calibration error falls from 0.292 to 0.169. Held-out Bitext routing accuracy reaches 98.3%. Cache reuse saves 15-37% of latency on an Apple M4 Pro, with token-identical replies in the FP32 demonstration cases. Scoring options on copies of the prompt cache matches full-sequence scoring to within 1.4e-6 in probability. Adding dialogue, industry and complaint data (System X+) improves support-task results but lowers IFEval accuracy from 55.5% to 45.0%. System X v4 uses Qwen3-1.7B, rebalanced data and teacher-rewritten reply targets. Against its base LLM, JevBench accuracy rises from 55.0% to 61.0% (p=0.04), with no statistically significant IFEval change (p=0.71). A matched latency test on unadapted Qwen3-1.7B saves 17-39%. One LLM judge prefers v4 replies to an unadapted 1.7B second call (31 to 11 pairs, p=0.003); the other's preference is not significant. These reply comparisons also differ in prompts and generation limits. State conflicts and unsupported reply details remain failure modes. Further studies with several seeds, human raters and real support data are recommended. 

---
# Enhanced Masked Diffusion Unlearning 

**Authors**: Tomer Garber, Tom Tirer  

**Link**: [PDF](https://arxiv.org/pdf/2610.25447)  

**Abstract**: Masked diffusion language models (MDLMs) generate text by iteratively unmasking tokens with bidirectional context. As these models reach real-world deployment, machine unlearning becomes essential. This process aims to remove the influence of a designated "forget set" without retraining from scratch, so that sensitive, copyrighted, or restricted content can no longer be elicited while preserving retained capabilities. Masked Diffusion Unlearning (MDU) is the first publicly available unlearning method customized for these models: it tunes the model $p_\theta$ to match a frozen prompt-masked anchor $p_{\mathrm{anchor}}$ on the forget set by minimizing the anchor-last divergence $\mathrm{KL}(p_\theta \Vert p_{\mathrm{anchor}})$. We identify two failure modes of MDU. First, random answer masking can leave key forget tokens visible to the anchor even when the question is fully masked. Second, at a memorized (near-one-hot) prediction, the anchor-last logit gradient vanishes while the anchor-first divergence $\mathrm{KL}(p_{\mathrm{anchor}} \Vert p_\theta)$ logit gradient does not. As a result, MDU training stalls on the tokens we most need to forget. Backed by theoretical and empirical analyses, we propose Enhanced Masked Diffusion Unlearning (EMDU): anchor-first KL divergence toward a frozen anchor whose entire answer span is masked. On TOFU and RWKU benchmarks using LLaDA-8B and Dream-7B, our EMDU matches or exceeds the forgetting of MDU and adapted AR baselines, while better preserving retain utility. 

---
# Outcome-Anchored Credit Reallocation for Agentic Knowledge Graph Question Answering 

**Authors**: Mingyuan Li, Yucheng Li, Xiaxia Wang, Gong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.25384)  

**Abstract**: Agentic knowledge graph question answering (KGQA) requires large language models (LLMs) to solve complex questions through multi-turn interactions with a knowledge graph (KG). Outcome-supervised reinforcement learning derives verifiable outcome rewards from final answers but uniformly assigns each rollout's outcome-level advantage to all policy-generated tokens, creating a trajectory-to-operation mismatch since KG operations within a rollout can have varying effects on the search process and final answer. To address this, we propose Outcome-Anchored Credit Reallocation (OACR), which employs a training-only KG-state outcome model to estimate outcome-reward changes between adjacent KG search states and converts them into operation credit, from which token-wise weights are derived. These multiplicative weights reallocate token-level update magnitudes while preserving the sign of the outcome-level advantage at every generated token. Across three benchmarks, OACR achieves stronger overall performance than existing KGQA methods while using fewer inference-time LLM calls and tokens. 

---
# MemDerive: A Diagnostic Benchmark for Agent Memory over Evolving Data 

**Authors**: Sen Gao, Yongan Li, Han Chen, Bingsheng He, Ming Wu, Jason Zeng, Michael Heinrich, Shixuan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.25373)  

**Abstract**: Long-running agents rely on memory of their past interactions for task state, such as what has been completed and which conditions still hold. Such state is usually a derived state. No record states it, and it must be computed by rules from multiple records of an evolving history. Existing memory benchmarks score only the final answer, with no reference for the intermediate states behind it. A wrong answer therefore cannot be traced to the state that was missed, and a state that silently became outdated goes undetected. We present MemDerive, a diagnostic benchmark for agent memory over evolving data. It specifies each scenario symbolically, derives exact gold answers with Datalog, records the gold derivation DAG and the supporting events of every query, and renders the history in natural language with fictional entities and distractors. Its 10 scenarios contain 4,692 events and 1,503 queries whose difficulty varies along access, evolution, and composition. On this ground truth, MemDerive evaluates answer reliability, diagnoses failures by probing the nodes of the gold derivation DAG, and tracks whether answers stay correct as events arrive. Our evaluation of four memory systems reveals failures that final-answer benchmarks hide. On derived state, every memory system falls behind simple BM25 retrieval, although the same answer model reaches 96.01% when given the supporting events. Most failures are ungathered, as a system holds the missing intermediate state but does not retrieve it for the question that needs it. Memory systems also rarely propagate implicit changes, which no new record states. These results suggest three principles for memory design. Gather inputs along the derivation, store rules with their versions, and maintain derived states through recorded dependencies. 

---
# CHSM8: A Chess Transformer that Follows Natural-Language Style Prompts 

**Authors**: Ilia Badanin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25356)  

**Abstract**: We present CHSM8, a 210M-parameter chess transformer that plays moves conditioned on a free-text description of a playing style, e.g. "go for the Sicilian Defense", "solid, safe, risk-free chess" or "You are Magnus Carlsen in his prime". The model is pretrained without search on 1.07B filtered Lichess games, using a factorized move representation and legal-move masking. It reaches 1837 Elo against a strength-limited Stockfish and, in 1,000-game matches, beat every other public model we tested that was likewise trained without preference tuning, rewards or search, including one with 13 times as many parameters. We then fine-tune it to follow text on a synthetic dataset: Gemma-4-31B describes the playing style of 3M game sides, a 1B Gemma-3 model distilled from the 2.4M of those descriptions that contradict nothing on the board labels the rest, and every one of 30M selected game sides gets five descriptions. Supervised fine-tuning attaches a frozen text encoder through gated cross-attention, and direct preference optimization (DPO) uses the same moves under a mismatched description as the rejected sample. After DPO, style adherence rises on every prompt we test (for example, queens traded early 53% -> 67%, pawn moves 29% -> 32% of all moves, queenside castling 97.3% -> 99.7%), and strength without a prompt stays within noise of SFT (1792 vs. 1797 Elo). Design choices at every stage were made by fixed-budget automated search loops on top of standard architectures and training recipes. We document the full pipeline and the decisions behind it, report the ablations that failed, and evaluate the models across many style prompts and kinds of games, with 1,000 games per setting. The work is fully open source and reproducible: weights, datasets with all training labels, prompts, configurations, inference code and playable demos are public and free to build on. 

---
# Learning When to Abort in Transformer Planning: A Mechanistic Study 

**Authors**: Shi Feng, Qimin Zhong, Siwei Wang, Yifei Shen, Shang-Hua Teng, Hao Liao, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.25300)  

**Abstract**: Planning does not by itself imply the ability to recognise errors, but recognising errors can improve planning capacity. In this paper, we study this phenomenon in Transformer planning. To analyze the inner mechanism, we follow the approach of Wang et al. (2024) and abstract planning as path-finding in graphs. In this setting, we train the model to detect the error of invalid transitions, i.e., after the model generates a next node that is not adjacent with the current node, the model should then generate a special abort token x. For expressiveness, we construct a Transformer with feed-forward width $O(|V|)$ for any fixed graph with node set $V$. It can detect all possible invalid transitions, and select a valid next node if the prefix is valid. We then propose a worst-case lower bound on feed-forward width, which makes the detector's linear width near-optimal up to logarithmic factors. In trained models, we demonstrate that invalid-transition detection increases previous-node attention, and most of the valid--invalid separation locates in the feed-forward network's output. This accords with our theoretical insights. Finally, we use this error detection to improve the model's performance: at inference, generating the abort token x will directly cause an external procedure to discard the attempt and restart from the original prompt. Experiments show that error detection combined with retry improves model's planning reliability on graphs, Countdown, and mazes. 

---
# Aligning Language Models with Indigenous Community Cosmovision through Constitutional AI 

**Authors**: Silvia Fernández-Sabido, Jorge Reyes-Magaña, Jesús Adrián Chay Tun, Jean Carlos Buenfil, Victor González Nahuat, Francisco Morales  

**Link**: [PDF](https://arxiv.org/pdf/2610.25298)  

**Abstract**: Large language models (LLMs) are predominantly trained on corpora that reflect Western, anglophone worldviews. This asymmetry is not neutral: when a model answers questions about territory, community authority, or the relationship with nature, it does so from assumptions that may be alien to the communities using the system. This paper presents a system for aligning language models with the ethical principles of Indigenous communities in Latin America. Drawing on interviews with five Indigenous leaders and defenders, we built an operational cultural constitution grounded in an Indigenous cosmovision, with 12 principles. On this basis, we implemented a Constitutional AI (CAI) pipeline in two phases: supervised fine-tuning (SFT) and direct preference optimization (DPO). The base models were Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct. With only 24 preference pairs, improvements were modest but detectable in intergenerational transmission, linguistic caution, and consultation and consent. Qualitative analysis reveals that the SFT phase was responsible for the primary shift in alignment, while DPO refined specific categories. The system was deployed in a conversational interface with retrieval-augmented generation (RAG) that allows communities to feed the knowledge base, turning alignment into a participatory cycle. We argue the architecture is functional and will scale qualitatively once the corpus is expanded and computing constraints are resolved. The work constitutes a first iteration of an epistemic justice mechanism for value alignment in the Global South. 

---
# Assessing the Empirical Evidence for Loss of Control from Agentic General-Purpose AI 

**Authors**: Risto Uuk, Santeri Koivula, Lorenzo Pacchiardi, Rokas Gipiškis, Laura Caroli, Kamaria Horton  

**Link**: [PDF](https://arxiv.org/pdf/2610.25241)  

**Abstract**: Theoretical literature warns that advanced generalist agentic systems might pursue instrumental goals that are misaligned with human preferences, thus posing risks of irreversible loss of human control. The EU's Code of Practice for General-Purpose AI lists loss of control as a systemic risk that must be assessed, yet no structured assessment of the quality of the empirical evidence for this risk exists. This study fills that gap. We evaluate 19 recent empirical studies across eight properties identified from threat models (such as strategic deception, situational awareness, and autonomous replication), assessing the quality of evidence across three dimensions: construct validity, content validity, and external validity. Our central finding is that the current body of evidence supports a judgment of loss of control as "weakly plausible." In our framework, 'plausibility' denotes the strength and direction of available evidence, not a probability estimate. Emerging evidence exists across multiple properties. No study clears the validity thresholds for a stronger judgment that a property is present; a single study clears them only for a negative-direction judgment (against situational awareness). Validity gaps, particularly in external validity, are widespread. Therefore, our findings highlight the need for more rigorous evaluations of agentic general-purpose AI systems. However, for the most dangerous properties, high external validity may be structurally unachievable under responsible research practices, reflecting a ceiling on feasible evidence rather than absence of the capability. Furthermore, as AI capabilities advance rapidly, empirical assessments may systematically lag behind the capability frontier. Our framework provides a structured baseline for ongoing monitoring as capabilities advance. 

---
# Dialogue Generation for Conversational Recommendation Using User Search Interactions and Knowledge Graphs 

**Authors**: Minsung Hwang, Donggyu Yoon, Gowoon Hwang, Kwanghee Jang, Joyce Jiyoung Whang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25182)  

**Abstract**: Large-scale conversational recommendation datasets typically require manual dialogue construction or annotation. We propose SearCl, which automatically generates conversational recommendation dialogues grounded in user search and click behaviors. SearCl segments user interactions into query sessions and derives each recommendation target from content selected after a search query. A configurable probabilistic state machine controls the dialogue flow, while two LLM agents acting as the seeker and recommender generate utterances according to turn-level constraints. To support recommendation, the recommender is provided with knowledge graphs constructed from metadata about the content items used in each dialogue, capturing their attributes and relationships. Using SearCl, we generate 3,538 dialogues from content search and click interaction data, covering 1,731 unique target items. Under an LLM-as-a-judge protocol, the resulting dialogues achieve higher quality scores than dialogues from existing benchmarks. These results demonstrate that SearCl enables controllable and knowledge-grounded dataset construction without human annotations. 

---
# Memory, Know Your PLACE: Governing Lifecycle State and Query-Time Eligibility in Persistent-Memory Agents 

**Authors**: Chen-Fang Hu, Hong-Han Shuai, Woan-Shiuan Chien  

**Link**: [PDF](https://arxiv.org/pdf/2610.25179)  

**Abstract**: Persistent memory turns prior interactions into durable agent state, but existing systems provide limited control over whether that state remains valid and whether it should influence a later query. We argue that reliable memory requires governance over both state formation and query-time influence. We introduce PLACE, a factorized governance layer that adds write-side state certification and read-side authorization to a memory backend. A factorial evaluation on Mem0 with Qwen2.5-3B and Qwen2.5-7B isolates the contribution of each mechanism. Results show that write-side governance substantially improves lifecycle state correctness, while read-side governance eliminates measured causal leakage from ineligible memory in the state-correct setting. Their combination achieves the strongest joint performance while preserving query-relevant support. These findings show that reliable persistent memory requires explicit control over both maintained state and query-time information influence. 

---
# Indistinguishable Agents: Training Agent Privacy as Twin-World Indistinguishability 

**Authors**: Jiaming He, Guanyu Hou, Yi Yu, Xudong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25161)  

**Abstract**: LLM agents search, shop and send messages for their users, and every action is seen by some party. Existing privacy training rewards agents for not mentioning protected items, yet an observer can infer a private fact from what the agent does. Since an observer learns a fact only if its view changes with the fact, we propose \methodfull (\method), which trains on twin worlds that differ only in the private fact. A frozen distinguisher scores each observer's view by its log-likelihood ratio for the true world, and paired GRPO trains the agent on task success minus this ratio. We prove that this objective controls the Bayes success of every observer on every channel. On shopping and messaging tasks \method removes the leakage that disclosure judges and query-level rewards miss, while the agent still completes its tasks. 

---
# PLANE: Plug-and-Play Fine-Grained Visual Experts for Hallucination Mitigation in Large Vision-Language Models 

**Authors**: Jianghao Yin, Lang Yu, Qianxi Yu, Qin Chen, Jie Zhou, Zhikai Lei, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2610.25146)  

**Abstract**: Large Vision-Language Models (LVLMs) commonly suffer from hallucinations, which appear as fabricating non-existent objects or outputting incorrect attributes and spatial relationships in images. Previous works empirically indicate that CLIP-style vision encoders tend to focus on primary objects while ignoring image details, which is one of the causes of hallucinations. To address this issue, we propose plug-and-play fine-grained visual experts (PLANE) comprising two components: LoRA visual experts operating on CLIP for multi-aspect image encoding, and an instruction-aware gate network for adaptive image feature fusion. During training, we construct a fine-grained image-caption pairs dataset, where images are described from three aspects: objects, attributes, and relations between objects. Image-caption pairs from different aspects are used to train the three fine-grained LoRA experts respectively. Aiming at comprehensively evaluate our method's capabilities, we manually annotate a fine-grained hallucination evaluation benchmark for LVLMs which is GPT4-independent during evaluation phase (GIFT). We conduct experiments both on popular hallucination evaluation benchmarks and GIFT. The results show that our method effectively reduces hallucinations across models of different types and sizes while introducing minimal inference latency. Compared to advanced baselines, our model also demonstrates superior performance. 

---
# Trackable Skill Library Lifecycle Management for Agentic Reinforcement Learning 

**Authors**: Xi Zhu, Yuanlin Duan, Changyi Yang, Kai Mei, Riyang Bao, Shichao Zhu, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2610.25126)  

**Abstract**: Skill-augmented agentic RL has emerged as an effective approach to improving LLM agents on long-horizon tasks. Despite this progress, individual skills may follow distinct trajectories as the policy evolves, with some becoming redundant after the model internalizes their capabilities, others remaining valuable as external scaffolding, and still others turning harmful. However, existing skill library systems fail to identify each skill's true contribution due to free-riding among co-retrieved skills, difficulty confounding across tasks, and the absence of skill-specific counterfactual evidence, which prevents reliable skill lifecycle management. We propose TrackRL, an agentic RL framework where skill tracking drives skill library full lifecycle management. Specifically, within each GRPO group, we turn grouped sampling into online randomized controlled experiments. Independent random injection and within-group contrasts on the same task isolate each skill's true contribution, and reliability-weighted aggregation accumulates evidence across groups. As evidence accumulates, we continuously track skill utility, which measures the skill's causal contribution as external scaffolding, and model mastery, which reflects how far the corresponding capability has been internalized. Along this line, we formulate skill lifecycle management as an evidence-gated state-transition process that governs each skill from generation and admission through protected incubation to retention, retirement, or removal, while a soft capacity bound controls library growth. Experiments show that TrackRL outperforms recent skill-augmented RL baselines in both library-assisted and skill-free evaluation, with further analyses revealing how skill value shifts during policy learning and how evidence-driven state transitions shape skill library evolution. Our code is available at this https URL. 

---
# CoEvo-RSI: When Harnesses Evolve, Which Models Should Follow? 

**Authors**: Jiaqi Su, Guibin Zhang, Jie-Jing Shao, Cong Pang, Tiankuo Yao, Ivor Tsang, Haiyan Yin, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.25078)  

**Abstract**: Model--harness co-evolution has emerged as a prominent paradigm for recursive self-improving (RSI) agents, jointly evolving their execution scaffolds and intrinsic capabilities. We study the persistent yet often overlooked brain--body mismatch problem in this paradigm, where an evolving harness (body) demands behaviors its model (brain) has not learned or restricts capabilities it already possesses. We introduce CoEvo-RSI, a RSI framework that advances model--harness co-evolution through agent-driven, bidirectional historical inheritance. Drawing on a shared evolutionary archive and execution feedback, a meta-agent coordinates harness development from historical designs with model checkpoint selection or merging and subsequent adaptation under the selected harness, aligning inherited capabilities with evolving execution requirements to reduce brain--body mismatch. Using only supervised fine-tuning (SFT) on its own rejection-sampled trajectories, CoEvo-RSI makes Qwen3.5-9B competitive with larger models across diverse vertical applications, including multi-agent failure attribution, role-aware text-to-SQL, and smart-home instruction following. It surpasses DeepSeek-V4-Pro on TraceElephant (+14.16 agent accuracy) and GLM-5.3 on HomeBench VM (+4.90 success rate), underscoring the need for brain--body alignment and the potential of bidirectional historical inheritance to advance agent co-evolution. 

---
# Policy Is Not Capability: Some Safety Benchmarks Measure Which Definition of "Unsafe" You Trained On 

**Authors**: Vahid Jebraeeli, Jonathon Romero, Yijin Wang, Adrian Chan, Jianbang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25060)  

**Abstract**: A guardrail classifier is reported as if unsafe were a property of text. It is not: it is a property of a policy, and the policy enters the model through its labels. We exploit an accident of an industrial deployment in which the same ~53k prompts were labelled twice, under two operational definitions ("is this a request for harmful content?" and "would a moderator block this?"), and used to train ten guardrail arms sharing architecture, codebase and optimiser family, at two model scales. First, the guardrail's own training corpus already contained the benchmarks: JailbreakBench-benign is 88% and OR-Bench-hard 82% present in the training pools, two over-refusal benchmarks, the kind a guardrail reports to argue it does not over-block. Everything below is decontaminated. Second, safety benchmarks stratify sharply by how much of their score is decided by the label definition rather than by the model: with architecture and optimiser fixed, the range across the ten arms is 0.99 on HarmBench-copyright and 0.043 on WildGuard-benign, a 23x spread. Third, a benchmark score can be moved across its entire range by 0.03% of the parameters. Across these arms HarmBench-copyright spans 0-100 and OR-Bench-hard over-refusal 3.4-56.4%, and the four arms whose last supervision signal carries a given policy occupy one extreme of both (exact rank p=0.005, or 0.04-0.14 at worst), at unchanged detection capability. For a distilled student the teacher's soft labels do this even when every hard label carries the other policy, so teacher provenance is an unrecorded confound in any released checkpoint. Fourth, validation accuracy on the one split that grades all twenty arms tracks agreement with the annotator's policy (rho=+0.74), not detection capability (rho=+0.34). We argue that guardrail evaluations should report contamination, a seed floor and the definition they were graded against as first-class variables. 

---
# Beyond the Teacher: Understanding and Preserving Distillation Overshoot in Agentic On-Policy Distillation 

**Authors**: Jike Zhong, Ming Li, Yuxiang Lai, Xi Xiao, Sheng Zhang, Ritwick Chaudhry, Tianle Chen, Jingyu Xie, Zheda Mai, Shao-Yuan Lo, Konstantinos Psounis, Yuanyuan Lei  

**Link**: [PDF](https://arxiv.org/pdf/2610.25034)  

**Abstract**: On-policy distillation (OPD) trains agents on their own trajectories using supervision from a stronger teacher. We uncover a counterintuitive phenomenon in multi-turn agentic tasks: a smaller, initially weaker student can substantially outperform its teacher during training, only to regress as distillation continues. We term this phenomenon distillation overshoot. Through decision- and task-level analyses, we trace overshoot to two competing effects of OPD: repairing weak student decisions and eroding successful student choices that the teacher disfavors. Early in training, repair outpaces erosion, allowing the student to combine the teacher's corrections with its own strengths and surpass the teacher. As training continues, erosion overtakes repair, causing these gains to vanish. Motivated by this finding, we propose Success-Preserving On-Policy Distillation (SPOD), which selectively masks conflicting teacher supervision on successful student choices and introduces a soft probability floor to counteract their erosion, while retaining teacher guidance elsewhere. SPOD requires only binary episode success and model probabilities already available during OPD. Across two model families and three agentic benchmarks, SPOD consistently outperforms ordinary OPD and evaluated variants, improving final success by up to 15.2 percentage points and surpassing the teacher by up to 20.0 points. Moreover, SPOD's final checkpoints outperform even ordinary OPD's best intermediate checkpoints in all six settings, demonstrating that beyond-teacher gains can be not only preserved but further extended. Code: this https URL 

---
# SoK: PIAtlas: Decomposing the Unbounded Prompt Injection Attack Space 

**Authors**: Stavros Damianakis, Konstantinos Psounis  

**Link**: [PDF](https://arxiv.org/pdf/2610.24999)  

**Abstract**: Prompt injection is the dominant security risk for large-language-model (LLM) applications and agents, yet the field still lacks a coherent map of how these attacks are constructed. More than twenty public datasets describe prompt injection using incompatible vocabularies, often organized around delivery vector, attacker goal, or generation method. In this Systematization of Knowledge, we present PIAtlas, a taxonomy organized instead around how the injected instruction is constructed to subvert the underlying model. Its six classes, one per subversion mechanism, form the roots of a hierarchy of techniques and sub-techniques that researchers can use to describe attacks consistently and design more systematic evaluations. We derive the taxonomy from how LLMs are trained, prompted, and influenced by human-like framing, and compare it with historical attack data. To ground this comparison, we release PICorpus, to our knowledge the largest license-clean collection of public prompt-injection datasets, and map thousands of its records to PIAtlas under a human-verified protocol. The mapping shows that attacks usually combine several techniques drawn from only a few mechanisms. It also reveals gaps that existing labels obscure: widely used agent benchmarks test only part of the attack space, and adaptive attackers favor mechanisms that curated benchmarks underrepresent. PIAtlas thus provides a common foundation for understanding this attack space, improving benchmark coverage, and guiding red teaming and defenses. 

---
# Certified Sparse Attention via Perturbed Retrieval 

**Authors**: Runguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.24991)  

**Abstract**: Sparse attention speeds up long-context decoding by reading few keys of the KV cache, but existing methods do not control the error of a query, and on real KV caches their errors have heavy tails. We show that perturbing the score of every key with an independent Exp(1) variable turns threshold retrieval into sampling keys with probability proportional to their attention weights. Lowering the threshold along a geometric grid gives nested samples, a Horvitz--Thompson estimate of the output at every level, and confidence bounds that hold simultaneously over all levels, so each head stops at the first level whose certificate meets the target error $\varepsilon$. With fresh perturbations at every decoding step, each certificate fails with probability at most $\delta$ during generation. At a suitable threshold, the estimator reads $O(\varepsilon^{-2}\log(1/\delta))$ values regardless of the context length, and no adaptive method reads fewer, up to constants. A 4-bit scan with deterministic bounds, shared across a grouped-query-attention group, generates the candidates exactly but reads every key. On three LLMs and ten settings at 16K--128K tokens, no certificate is violated, and at equal keys read our 90th-percentile error is 4.6--7.9$\times$ lower than that of the best of 15 baseline families. At equal bytes, the best baselines have 0.6--1.6 times our 90th-percentile error and at least 4.9 times our largest. End to end, our method stays within one point of dense attention on tasks and within 0.1\% in PG-19 perplexity, and the strongest baselines at equal bytes score as well. On CPU, decoding steps at 32K--128K tokens run 1.1--1.4$\times$ faster than dense attention on the same 16-bit cache. 

---
# CoworkEnv: Synthesizing Training Environments for Long-Horizon Cowork Agents 

**Authors**: Zhongwei Xie, Yiming Du, Jiaxin Bai, Baixuan Xu, Chunyang Li, Xiyu Ren, Wei Fan, Tianshi Zheng, Weizhi Du, Jingqiao Yu, Xingsu Li, Zhaowei Wang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.24960)  

**Abstract**: AI coworkers must coordinate tools, communicate across turns, and act on changing workspace state while respecting workflow constraints. Constructing training experience for these capabilities requires coherent requests, service records, tool interfaces, and grading checks. We present CoworkEnv, a capability-guided framework for synthesizing executable training environments for AI coworkers. Workflow requirements guide task construction, and programmatic validation with feedback repair checks generated specifications and grading behavior. Teacher executions yield curated demonstrations for supervised fine-tuning (SFT), while student interactions provide rollouts and rewards for reinforcement learning (RL). Using these resources, we train compact PocketWorker models. For Qwen3-VL-8B-Instruct, SFT raises the mean score from 46.6 to 64.1 on Claw-Eval, including a 31.9-point gain on Multi-turn tasks. An SFT-only extension to ClawMark demonstrates applicability across coworker benchmarks. We further compare RL training dynamics across configurations that vary reward processing and credit assignment. Improving training rewards provide preliminary evidence that the constructed environments support policy optimization. 

---
# Revealing what benchmark scores hide: Jointly measuring the breadth and reliability of LLM capabilities using accessibility profiles 

**Authors**: Kazem Faghih, Yize Cheng, Shoumik Saha, Mobina Pournemat, Armin Gerami, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2610.24957)  

**Abstract**: Language model benchmarks assign each question a single outcome from one fixed wording, even though the same underlying request can be expressed in many equivalent ways. We argue that the quantity of interest should therefore not be correctness on one wording, but the accessibility of an inquiry, which we define as the probability that a model succeeds across meaning-preserving formulations. The distribution of accessibility across inquiries reveals a structure that accuracy and existing best-, worst-, or average-case prompt scores collapse: how broadly a model's capabilities extend across inquiries and how dependably those capabilities can be elicited. Two models can have the same average performance while one succeeds across more inquiries unreliably and the other succeeds across fewer inquiries but more consistently, leading to opposite preferences depending on whether the task allows multiple retries until one success or when repeated consistent success is required. Across six benchmarks and more than twenty models, this tradeoff is common: for about half of the model pairs with nearly equal mean accessibility, which model looks better depends on whether one success across five formulations is enough or all five must succeed, quantities we call Access@5 and Reliable@5, respectively. For some pairs the two measures diverge significantly, by up to 10 points in opposite directions. Accessibility also reveals that scaling, reasoning, and prompt optimization can redistribute capability across inquiries in ways aggregate accuracy obscures, sometimes broadening the set of inquiries a model can succeed on while making success on them less repeatable. Accessibility turns formulation sensitivity from a nuisance into a measurable property of capability, distinguishing what a model can sometimes succeed on from what users can actually depend on. 

---
# CasePlay: Self-Play Reinforcement Learning from Case Reports for Medical Reasoning 

**Authors**: Hao Wang, Zihan Wang, Kai Wu, Jian Yao, Yiwen Ye, Yongcan Luo, Dapeng Wu, Jiale Chen, Haihua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24952)  

**Abstract**: Reinforcement learning (RL) is a promising way to improve the clinical reasoning abilities of large language models (LLMs), but current pipelines often rely on expert-labeled questions or costly synthetic-data generation. Meanwhile, the medical literature offers a wealth of high-quality unlabeled documents, including clinical case reports that trace the full clinical course from initial presentation to final treatment decisions. We study whether document-grounded self-play can convert these reports into effective RL training signals. We identify two challenges that limit existing self-play methods in this setting: knowledge-point fixation}, where the proposer repeatedly asks about a narrow set of salient clues in a source report, and difficulty-quality mismatch, where appropriately difficult questions may still be poorly formed or shallow, limiting their value for reasoner training. To address these challenges, we propose CasePlay, a self-play framework with two key components: (a) a Knowledge-Conditioned Proposer, which anchors question generation to case-specific knowledge points so that questions draw on a broader range of evidence from the source report; and (b) a Rubric-Judge Reasoner, which augments answer-accuracy feedback with rubric-based quality feedback on clue integration, reasoning depth, and option quality, guiding the proposer toward reasoning-intensive questions that better support reasoner training. Extensive experiments across seven medically related benchmarks demonstrate that CasePlay outperforms existing self-play baselines and improves the SFT warm-start model by 3.1 points on average, highlighted by a 6.3-point gain on LiveClin. This approach offers an effective path toward continually improving medical reasoning ability through self-play RL. 

---
# Stable Signer V3: 4D Deformable Primitive Splatting Is All You Need for Sign Language Production 

**Authors**: Sen Fang, Liwei Che, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2610.24926)  

**Abstract**: We present Stable Signer V3, an interactive high-resolution sign language avatar system built on 4D Deformable Primitive Splatting. It addresses three limitations of existing avatars: restricted facial expression, inaccurate identity-specific body deformation, and loss of fine appearance details. Shared Face Primitive Splatting adds transferable superquadric primitives for richer expression control; Test-Time Body Primitive Splatting learns an identity-specific virtual muscle layer for more faithful motion; and Test-Time Tunneling Super-Resolution uses TunnelingNet to recover identity-specific high-resolution details. Both test-time modules are optimized once per signer identity; inference then runs entirely from the stored identity parameters. Together, they reduce facial geometry error by 71.4%, body-boundary error by 35.1% across ten identities, and high-resolution RGB reconstruction error by 24.2% on held-out evaluations, while retaining interactive streaming. 

---
# CARTA: Self-Supervised Reusable Transaction Embeddings via Heterogeneous Tokenization and Cross-Attention 

**Authors**: Shreyas K. Chandrahas, Jose Munoz, Satish Babu Bellapu  

**Link**: [PDF](https://arxiv.org/pdf/2610.24920)  

**Abstract**: Acquiring fraud labels for payment transactions is expensive and inconsistent across tasks, and high-cardinality identifiers such as merchant names make one-hot feature pipelines infeasible at scale. We present CARTA, a self-supervised encoder that learns a compact 16-dim embedding from raw transactions without labels; the frozen embedding is then reused for label-efficient fraud probing and exploratory unsupervised risk stratification without retraining the encoder. CARTA uses type-specific projections with learned embedding tables for high-cardinality identifiers, trained with token excision and InfoNCE contrastive loss. On IEEE-CIS, its linear probe at 1% of labels exceeds a raw-feature probe at 100%; on IBM TabFormer (2M rows), its frozen embedding reaches GBDT AUPRC 0.508, above frequency encoding (0.411) and the merchant-dropping baseline (0.022), though fully-supervised CatBoost is stronger. CARTA is a label-free, reusable representation that complements task-specific supervised models. 

---
# Arctic-SWE: Revisiting the Long Horizon Agent Training by Modeling-System Co-Design 

**Authors**: Boyi Liu, Yite Wang, Xinyu Lian, Fan Shu, Sida Li, Feng Yan, Mingjia Zhang, Yuxiong He, Zhewei Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.24914)  

**Abstract**: Long-horizon agent training, including software engineering, is advancing rapidly, but the recipe and infrastructure needed for a strong reinforcement-learning baseline remain difficult to reproduce. This makes gains from new algorithms or data hard to distinguish from improvements to an under-tuned baseline. We present Arctic-SWE, a fully asynchronous RL framework that establishes such a baseline through model-system co-design, using overfitting on a small task pool as the testbed. This testbed exposes failures that early reward gains conceal and guides the integration of established components: gated reward, importance-weight control, aligned training and inference engines, and consistent state across policy updates. No single component is new, yet together they form the missing foundation. On raw R2E-Gym data, without new algorithms or data work, Arctic-SWE trains Qwen3.5-4B to 62.0% pass@1 on SWE-bench Verified. This is comparable to concurrent FrogNano (61.5%), which relies on online task synthesis, after about one-sixth as many training rollouts. Building on this foundation, we further filter and refine R2E-Gym, SWE-rebench V2, and Scale-SWE. With this curated subset, the 4B model improves from 47.4%/25.0%/37.9% to 65.0%/38.3%/59.7% on SWE-bench Verified/Pro-v1/Pro-v2, and the recipe also improves Qwen3.5-9B from 56.2%/31.9%/48.8% to 65.7%/39.6%/70.2% on Verified/Pro-v1/Pro-v2. Controlled comparisons show that both curation and test strengthening improve held-out performance, although stronger tests reduce apparent training-task mastery. Arctic-SWE provides a documented starting point for studying better algorithms, data and environments for asynchronous agent RL. 

---
# A Cross-Domain Evaluation of Prompting Techniques for Large Language Models 

**Authors**: Kevin Krazius, Tim Schlippe, Paul Libbrecht  

**Link**: [PDF](https://arxiv.org/pdf/2610.24913)  

**Abstract**: Prompt engineering has become a central mechanism for guiding the behavior of Large Language Models (LLMs) across a wide range of reasoning and language understanding tasks [1], [2]. However, existing studies frequently evaluate prompting techniques using different LLMs, datasets, or task settings, making direct comparisons across methods and domains difficult [3]. Consequently, in this paper we present a controlled cross-domain evaluation of Zero-Shot prompting, Few-Shot prompting, Retrieval-Augmented Generation (RAG), and automatic prompt optimization based on a PE2-inspired optimization pipeline [4]-[6]. All experiments were conducted with the same LLM and evaluated on six benchmark datasets covering arithmetic reasoning, logical reasoning, commonsense reasoning, and factual verification. The results show that no prompting strategy consistently outperforms the others across all domains, and pairwise statistical comparisons of the highest-performing configurations revealed no significant differences. RAG achieves the highest overall average accuracy of 78.33% when 300 examples are available for retrieval, while Few-Shot Prompting achieves comparatively high accuracies across several structured reasoning tasks. Zero-Shot Prompting remains competitive on tasks with self-contained inputs, whereas the PE2-inspired optimization approach produces stable but not systematically superior results compared to the other evaluated prompting techniques. Furthermore, increasing the number of task-specific examples does not consistently improve performance across prompting techniques, indicating strong domain and task dependency. These findings suggest that prompting techniques should be selected adaptively based on task characteristics rather than treated as universally applicable optimization strategies. 

---
# HybridResearcher: Bridging the Structured-Unstructured Boundary for Deep Research Agents 

**Authors**: Soyoung Yoon, Peiran Xu, Han Wang, Ruofan Wu, Xiaodong Yu, Feng Yan, Mohit Bansal, Debiao Li, Seung-Won Hwang, Yuxiong He, Zhewei Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.24899)  

**Abstract**: While LLM agents are increasingly capable of gathering and synthesizing evidence from a single homogeneous source, such as structured tables or unstructured data, real-world analytical questions frequently span heterogeneous sources. For example, answering one enterprise question may require both a result computed over an internal database and evidence found on the web. Recent evaluations on benchmarks like HybridDeepResearch show that even frontier agents struggle when evidence must cross this boundary. While these benchmarks have exposed the limitation, methodologies to explicitly teach cross-source reasoning remain absent. To address this, we introduce HybridResearcher, a 4B agent trained to bridge evidence across the structured and unstructured boundary. To build it, we synthesize cross-source bridge trajectories by starting with validated single-source tasks-such as existing text-to-SQL or web search queries-and fabricating the missing source around a bridge entity drawn from the seed's own output. We retain a sample only when strict construction gates confirm that neither side alone suffices, ensuring the data requires true cross-source interdependence. Finally, we train on this curated dataset using supervised fine-tuning followed by reinforcement learning. Evaluated on the existing HybridDeepResearch benchmark, HybridResearcher improves over our base model, Qwen3.5-4B, by 30.0 points pass@8 and surpasses the far larger Qwen3.5-397B-A17B by 8.3 points, while also improving over its base model on established, held-out single-source benchmarks (BrowseComp and BrowseComp-Plus), demonstrating that this complex cross-source reasoning can be taught directly to small models without triggering catastrophic forgetting. 

---
# Peer Pressure: Execution-Verified Co-Evolution of Attacker and Defender Populations for Code Security 

**Authors**: Khushbu Pahwa, Yubin Ge, Daniele Bonadiman, Xiaofei Ma, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24896)  

**Abstract**: LLM code agents increasingly ship code with little oversight, yet defenders trained on static vulnerability datasets fall behind adapting adversaries. We co-train attacker and defender instead, and let execution decide every reward: an attacker (Red) edits a real repository and earns nothing unless its patch applies, builds, passes the tests, completes its development task and reproduces a real exploit in an isolated container, whose path also labels the entry point, the critical operation and the trace between them.
Peer-Relative Policy Optimization (PRPO) makes both sides populations of LoRA adapters over one frozen base and credits each against same-role peers that faced an identical condition, placing the population inside the policy-gradient baseline rather than the opponent pool, with a tournament preserving diversity. Matching cancels the condition's difficulty: at equal coverage it is worth 2.3 points of balanced accuracy over a baseline computed on other conditions, and it matches a within-policy control at half the verifier spend.
PRPO's 7B diff reviewer reaches 64.8% balanced accuracy on four unseen applications in three untrained languages, 5.5 points above supervised security fine-tuning and 4.4 above a frozen frontier reviewer (Claude Opus 4.8), which it leads by 15.8 on the application it trains on. Its 9B repository agent recovers 23 of 172 held-out VulnGym advisories, against 10 after supervised fine-tuning and 7 for the same backbone untrained under the same scaffold, and locates 17 of 43 reachable entry points against 13 for mini-swe-agent with GLM-5.2. Run unchanged on jailbreak defense, the same algorithm matches or exceeds WildGuard's static F1 at equal corpus and model size. Under a judge distinct from its training oracle it cuts attack success from 36.0% to 4.6%, against WildGuard's 8.9%. 

---
# The One-Pixel of Reasoning: One Truthful Relation Is Enough to Mislead LLM Agents 

**Authors**: Jinwei Hu, Youcheng Sun, Yi Dong, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24873)  

**Abstract**: Can one sentence manipulate an agent's belief by targeting a single implicit warrant linking evidence to a conclusion? In this paper, we introduce the one-relation attack (ORA), a novel cognitive attack on this inferential relation. ORA localizes a critical implicit warrant and performs a one-relation intervention using a single source-grounded sentence that qualifies this warrant and steers the agent toward an incorrect belief. Across eighteen models on biomedical inference and vulnerability detection, best-of-$10 attacks overturn initially correct answers at mean rates of 54.9% for proprietary and 80.0% for open-weight models. Attacks also transfer without victim feedback, and paired controls support the roles of targeting and qualification. Moreover, successful sentences can pass source-support checks and evade identification, and multi-agent debate raises team error from 50.0% to 59.5% in our balanced stress test. These findings establish implicit warrants as a cognitive attack surface for manipulating agents' beliefs. 

---
# The Missing Compliance Class: LLM Graders That Collapse Jailbreak Success, and a Free Check 

**Authors**: Ruqiya Bin Safi  

**Link**: [PDF](https://arxiv.org/pdf/2610.24872)  

**Abstract**: Jailbreak papers report an attack success rate, the share of responses that deliver the harmful request, and read it as a property of the target model. We hold the target model, the prompts and the responses fixed and change only the grader. Three dense graders of 8B and 9B parameters, from three model families, move compliant responses to the neighbouring class, evasion, instead of scattering them across classes. One of them assigns the compliance label zero times in 900 responses. This class collapse also appears on new harmful requests, on a target model from a third family, and in a held-out 7B grader from a fourth family. Per-class label counts expose it at no cost and without human labels. The collapse decides how the rate behaves under a change of grader. On two corpora of 900 responses and eight graders, the compliance rate differs by a factor of 22 to 25, while the non-refusal rate differs by 18 to 23 percentage points. Without the three collapsing graders, compliance differs by a factor of 1.7 to 2.8 on all four corpora, and non-refusal still differs by 10 to 22 points. These results do not change when every grader is called once. We propose a reporting minimum that keeps the compliance rate and adds its per-class label counts and the named grader. Beside it goes the non-refusal rate, whose disagreement across graders is additive. 

---
# Bellman Language Models: Solving the Recurring Decisions of LLM Agents 

**Authors**: Adarsh Vatsa, William Eiers  

**Link**: [PDF](https://arxiv.org/pdf/2610.24871)  

**Abstract**: A general observation in agentic workflows, especially bounded ones, is that the decision space of the LLM, when we separate a decision from generation, starts approaching saturation: the agent meets the same states again and again, and after enough conversations it stops meeting new ones. LLMs generally don't treat these decisions as an optimization problem and instead re-derive them from their context/prompt on every turn. We show that treating decision making within reasonable bounds as an optimization problem, solving it once, and reusing the solved decisions can often make an agent faster, cheaper and more reliable. Our approach, which we call Discretize, Then Decide, starts from data such as the workflow's own logs or, when none exists, runs of the LLM agent itself. It reads each conversation into discrete states and estimates the probability of each outcome and the cost of each move, until no new states appear and the probabilities stop changing. It then solves the resulting Markov decision process. The solution runs as a controller around the model, sending moves whose content is fixed in the state, such as a tool call or a templated reply, and any state it hasn't seen is sent back to the LLM. We call an LLM run this way a Bellman language model (BLM), after Richard Bellman's equation for the best move in each state.
On $\tau^2$-bench telecom, the controller with no model solves 79% of tasks. As a BLM, Qwen3.8-27B, an open model, goes from 29% to 96% with 93% fewer model calls. For Claude Opus 5.5, which solves 96% of tasks on its own, running it as a BLM cuts model calls and cost per task by about 98% and lets it solve 2 more tasks. On InventoryBench, the solved policy with no model beats every published method. Where the hard part is writing content rather than choosing moves, such as resolving SymPy issues, few decisions can be solved in advance, and a BLM helps little. 

---
# Dual Task Vector Merging: Leveraging Emotion and ASR Knowledge for Low-Resource Alzheimer's Disease Detection 

**Authors**: Yuan-Chao Li, Yin-Long Liu, Liu He, Sizhe Liu, Ji-Xuan Deng, Yin Zhang, Zhen-Hua Ling, Jiahong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2610.24870)  

**Abstract**: Alzheimer's Disease (AD) detection from speech benefits from both paralinguistic and linguistic cues, yet model training is frequently constrained by limited data. Fine-tuning a large pretrained speech encoder can easily overfit, while a frozen probe leaves the encoder's task-relevant knowledge underutilized. Therefore, we propose Dual Task Vector Merging, a parameter-efficient method that adapts a frozen wav2vec 2.0 backbone using task knowledge learned outside the AD task. Specifically, we inject two task vectors, derived as weight deltas from emotion- and ASR-fine-tuned wav2vec 2.0 models, into each transformer layer. During AD training, only layer-wise merging coefficients and a linear classification head are optimized, updating less than 0.1% of the parameters required for full fine-tuning. Speaker-independent experiments on ADReSS and ADReSSo show that dual merging improves over frozen probes, LoRA, and full fine-tuning, in the main binary settings, and gives the best results when combined with LoRA. Evaluation on PROCESS further shows that our advantages transfer to three-class HC/MCI/dementia classification when paired with LoRA. Experiments under reduced training data (20%, 40%, and 80%) confirm reliable gains in low-resource settings. 

---
# Medical Fine-Tuning Reshapes, Not Reduces, Patient-Facing AI Risk: A Rubric-Based, LLM-Judge Evaluation Framework 

**Authors**: Savan Doshi  

**Link**: [PDF](https://arxiv.org/pdf/2610.24807)  

**Abstract**: Patient-facing medical question answering with large language models (LLMs) is moving from research demonstrations to production deployments, yet evaluation remains anchored on factual correctness. Factuality is necessary but not sufficient: a response can be factually grounded and still issue a directive that, if acted on, would harm the patient. We introduce a risk-sensitive evaluation framework that scores responses by the presence and severity of six categories of risk-bearing language (treatment directives, contraindications, dosage expressions, urgency-triage cues, high-alert medication mentions, and overconfident assertions), anchored on published clinical harm taxonomies. Severity is classified by a three-judge LLM ensemble. We apply this framework to six 7-9B instruction-tuned models, three general (Mistral-7B, Phi-3.5-mini, Llama-3.1-8B) and three medically adapted (BioMistral-7B, OpenBioLLM-Llama3-8B, Med42-Llama3-8B), on 200 real patient questions from the K-QA benchmark. Two findings emerge. First, ensemble risk-bearing language is statistically independent of factual hallucination rate (pooled Spearman $\rho = -0.065$, $n = 300$, 95% CI $[-0.18, +0.05]$; distance correlation $= 0.10$, independence test $p = 0.24$). Second, medical fine-tuning does not reduce aggregate risk but reshapes it: continued-pretraining adaptations (BioMistral, OpenBioLLM) consistently suppress high-alert medication mentions while amplifying overconfident assertions, whereas clinical instruction-tuning (Med42) suppresses dosing only, without amplifying overconfidence. Per-category Krippendorff $\alpha$ is fair to good for the four linguistically anchored categories (0.53-0.71) and low for the two clinical-judgment-dependent categories (0.19, 0.31), indicating where automatic safety evaluation is and is not yet reliable. We release the rubric, judge prompts, and evaluation code. 

---
# Information Theory of Random Composition Trees 

**Authors**: Doron Sivan, Weishun Zhong, Tankut Can, Mikhail Katkov, Misha Tsodyks  

**Link**: [PDF](https://arxiv.org/pdf/2610.24788)  

**Abstract**: Random composition trees model the hierarchical semantic structures of meaningful texts as a statistical ensemble generated by recursively splitting texts into coherent chunks. A recent study shows that different text types can be described by random composition trees with distinct branching factors, and that the entropy rates of these trees exhibit a similar dependence across text types as the entropy rates of the actual texts. Motivated by this correspondence, we analyze information-theoretic properties of this stochastic splitting model. First, we demonstrate that the entropy of random composition trees grows linearly with the root size, and we derive an analytical expression for the entropy rate as a function of the branching factor. Second, we prove that large trees exhibit typicality: with high probability, they belong to a typical set in which each tree is approximately equally likely. Our analysis reveals that the random composition tree model shares two key information-theoretic properties of language when modeled as a stationary and ergodic stochastic process: linear entropy growth and typicality. 

---
# Neuro-Inspired Inhibitory Message Passing for Accurate and Token-Efficient Multi-Agent Reasoning 

**Authors**: Nima H. Siboni, Magdalena Springer, Vahid Rostami  

**Link**: [PDF](https://arxiv.org/pdf/2610.24718)  

**Abstract**: Multi-agent systems promise to solve problems by bringing together information beyond the reach of any individual agent. Retrieval and tool use can distribute this information across agents, making communication essential to collaboration. A central challenge is keeping crucial evidence visible in each agent's context as messages accumulate. Repeated information can crowd out rare, decisive evidence. Models of cortical circuits suggest a principle for addressing this imbalance: inhibition can selectively suppress shared inputs while preserving sensitivity to weaker signals. We adapt this principle into inhibitory message passing, a training-free communication router that cancels redundant evidence separately for each evidence sentence and receiver. Each round, it forwards evidence sentences restated in agents' messages, delivering each at most once to receivers whose own evidence lacks it and excluding the senders' reasoning and answers. Against standard message passing, accuracy in peer groups rose by 39-77 percentage points across three benchmarks, and tokens per run fell by 10-46%. Gains held across seven open models spanning four to thirty-two billion parameters and two commercial models, in groups up to sixty-four agents, and across multi-agent architectures including peer-to-peer, orchestrator-worker, and multilevel hierarchical organizations. These findings support a scalable design pattern for more accurate and token-efficient multi-agent reasoning. 

---
# Verifying Human-Written Reasoning References 

**Authors**: Ali Ansari, Farhana Ferdousi Liza  

**Link**: [PDF](https://arxiv.org/pdf/2610.24690)  

**Abstract**: Evaluating the reasoning of AI systems often relies on human-written reasoning chains as references, yet the logical support for their individual steps is rarely verified. Although prior neuro-symbolic approaches have formalized and checked reasoning, their reported evaluations do not systematically assess whether step conclusions in human-written reference derivations are entailed by their cited premise sets or investigate the sources of failures in those references. To address these limitations, we present a novel reference-centred assessment method for human-written derivations that combines entailment checking against cited premise sets, full-premise rechecking, and failure attribution to examine the reliability of these derivations and their formal representations. We evaluate it on P-FOLIO, using a large language model (LLM) to translate step conclusions into first-order logic (FOL) and the Z3 SMT solver to check entailment. Of 386 derivations examined, 202 (52.3%) meet all our checking criteria, meaning that every translated step is entailed by its stated support and the required target is established under the available FOL representations. Among 230 steps not entailed by their stated support, 191 become entailed when checked against the complete original premises, indicating that most initial failures reflect incomplete or mismatched support rather than unsupported claims. Manual analysis of the remaining 39 cases identifies 21 human reasoning errors, 11 FOL translation errors, and 7 dataset annotation errors. A separate manual audit of 50 randomly sampled steps that passed the entailment checks found 46 faithful FOL translations (92%). These findings suggest that disagreement with a human-written reasoning reference may originate in the reference or its formal representation rather than in the AI system being evaluated. 

---
# ASR Personalization in Pathological Voice: Accuracy, Efficiency, and Duration Effects 

**Authors**: Giuseppe Giaimis, Sergei Katkov, Muhammad Azfar Yaqub, Antonio Liotta  

**Link**: [PDF](https://arxiv.org/pdf/2610.24572)  

**Abstract**: Automatic Speech recognition (ASR) struggles with Pathological speech due to persistent speaker-specific mismatch. In clinical and accessibility scenarios, only limited target-speaker data is typically available, making full model retraining impractical. We present a component-level analysis of LoRA adaptation in Whisper, investigating how adapter placement interacts with the model size, training efficiency and utterance-duration in single-speaker settings. Our experimental comparative study shows that while scaling reduces errors, it does not eliminate mismatch. Furthermore, the full attention LoRa provides the most reliable accuracy, and the best time--accuracy trade-off. Improvements vary across utterance length, with medium-length recordings benefiting the most, while short and long utterances remain challenging. These findings highlight the importance of adaptation placement and duration-dependent evaluation, under challenging, data-scarce personalization scenarios. 

---
# Axiomatic World Modeling for Physics Reasoning 

**Authors**: Xinye Yang, Zhenyang Liu, Yuxuan Wang, Yuanyuan Lei  

**Link**: [PDF](https://arxiv.org/pdf/2610.24519)  

**Abstract**: Recent research-level physics-reasoning benchmarks for LLMs reveal a failure mode that lies not in knowledge or execution, but in modeling the world being reasoned about. Models often silently alter given quantities, fabricate laws, or inject unstated assumptions to force a problem into familiar patterns, and then reason inside this corrupted world. We define this premise-level divergence as modeling drift: the gap between a model's internal axiomatic world model (AWM) and the true physical world specified in the problem. Although RLVR and PRM have yielded substantial gains in mathematics and code, their reward shapes do not directly supervise world modeling in axiomatic domains such as physics. We propose Reinforcement Learning with Axiomatic World Modeling (RLAWM), which optimizes the policy by penalizing divergence between its inferred axiomatic world model and the true physics world. RLAWM probes physics reasoning at two hierarchical levels (modeling and solving), each scored by a physically grounded reward. The modeling reward targets abstract knowledge formulation, evaluating whether the model captures the true axiomatic world before attempting calculation. The solving reward targets instantiated knowledge reasoning, checking whether the model's concrete derivation remains consistent with the axiomatic world. This two-level structure separates what world the model believes it is reasoning in from how it reasons within that world. The framework operates exclusively during training, with no extra cost during inference. RLAWM improves performance by up to +12.2 points on PhysReason and +5.2 on PHYSICS over the baseline. It also generalizes zero-shot across difficulty, source, and domain, with a 2.45x gain and +10 points over the strongest baseline on olympiad-level physics and math. 

---
# ProbePO: Sparse Probes, Redistributed Credit for Critic-Free LLM Reinforcement Learning 

**Authors**: Diyi Hu, Ruizhong Qiu, Lin Qiu, Hanghang Tong, Hanqing Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2610.24515)  

**Abstract**: Reinforcement learning from verifiable rewards assigns one terminal reward to an entire reasoning trajectory, obscuring how individual reasoning steps contributed to the outcome. GRPO broadcasts a group-normalized advantage to every response token, while finer-grained approaches often rely on dense auxiliary rollouts or branching schemes that can also alter the data used for policy updates. We cast the missing middle as budgeted measurement: with a limited rollout budget, which partial solutions should be evaluated, and how should their value estimates refine credit? ProbePO selects partial-solution states with a reward-free uncertainty score and estimates their expected verifier rewards under a capped continuation budget. ProbePO redistributes, rather than replaces, GRPO's trajectory-level advantage across tokens while preserving each trajectory's mean token-level advantage; probe continuations serve only as measurements. Across three DeepSeekMath and Qwen-family models spanning 1.5B-7B parameters, ProbePO consistently outperforms GRPO and achieves the highest accuracy among the four evaluated methods in 16 of 18 model-benchmark comparisons. Against dense-probing VinePPO on DeepSeekMath-7B, ProbePO achieves higher accuracy on all six benchmarks with 76% less end-to-end training time. 

---
# Pluralistic Alignment in Language Models: Objectives, Human References, and Preference Representations 

**Authors**: Meng-Chen Wu, Umesh Bodhwani  

**Link**: [PDF](https://arxiv.org/pdf/2610.24471)  

**Abstract**: Pluralistic alignment aims to represent meaningful variation in human preferences rather than collapse heterogeneous judgments into a single target. Yet existing work often mixes three distinct questions: what pluralistic behavior a model should exhibit, whose preferences define the reference, and what evidence is sufficient to demonstrate success. We organize the literature around a two-layer taxonomy. Layer 1 distinguishes representing a range of human-supported viewpoints (Overton Pluralism), adapting faithfully to a specified perspective or preference condition (Steerable Pluralism), and matching the response distribution observed in a target population (Distributional Pluralism). Layer 2 distinguishes aggregate populations, countries and regions, and subgroups as human references. Preference representation, implementation mechanism, modality, and evidence quality are treated as cross-cutting design choices rather than additional forms of pluralism. We synthesize a protocol-bounded corpus of 70 works on pluralistic alignment and neighboring areas, using the objective-by-human-reference matrix to map existing approaches and identify underexplored combinations. The coded literature concentrates on Steerable Pluralism and Distributional Pluralism, while eligible Overton Pluralism work appears only at the aggregate-population level under our stated protocol. We then connect each objective to the evidence needed to support it, including perspective validity, conditional fidelity, distributional fidelity, support preservation, and uncertainty in subgroup estimates. Together, the taxonomy and evaluation framework clarify what current methods establish, where their claims exceed their evidence, and how future systems can represent disagreement without mistaking variation for faithful representation. 

---
# Long Listening Thoughts: Eliciting Open Auditory Reasoning with Deliberative Perception and Cognitive Refinement 

**Authors**: Jaeyeon Kim, Chao-Han Huck Yang, Luoyi Zhang, Chan-Jan Hsu, Fernando Ruiloba Portilla, Jinchuan Tian, Shinji Watanabe, Carlos Busso  

**Link**: [PDF](https://arxiv.org/pdf/2610.24409)  

**Abstract**: We introduce Long Listening Thoughts (LLT), a framework for constructing post-training datasets that improves audio reasoning by pairing deliberate perceptual grounding with explicit cognitive behaviors. On audio understanding and reasoning benchmarks, large audio-language models (LALMs) are limited primarily by perceptual errors, yet existing post-training datasets rely on audio descriptions of limited diversity. Through a four-stage pipeline, LLT (1) builds multi-aspect dense audio descriptions spanning semantic, acoustic, and temporal information, (2) generates more challenging questions, (3) synthesizes incorrect reasoning traces, and (4) extends short reasoning traces into long ones via a thought-continuation strategy that elicits cognitive behaviors such as verification and backtracking. Using LLT, we synthesize 1.6M reasoning traces and train Qwen2.5-Omni-7B with a two-stage pipeline of supervised fine-tuning (SFT) followed by direct preference optimization (DPO). This improves accuracy on MMAU Test-mini (71.5 to 78.2), MMAR (56.7 to 63.7), and MMSU (60.6 to 66.0). 

---
# Defending Against Dormant Poisoning Attacks Across Language and Multimodal Agents 

**Authors**: Wonjun Lee, Vaidehi Patil, Zaid Khan, Yue Zhang, Hyunji Lee, Suhyun Kim, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2610.24379)  

**Abstract**: Recent work shows that attackers can implant dormant malicious behaviors that remain hidden at release but emerge after benign fine-tuning. As foundation models are used as agents, whether dormant poisoning extends to harmful agent actions and multimodal settings remains unclear. We show two agentic risks: (1) harmful behaviors activated by benign fine-tuning can extend to unseen actions; (2) multiple harmful behaviors can coexist and be jointly activated by benign downstream fine-tuning. Existing defenses often rely on external safe responses, which are hard to obtain for agent actions, or a trusted clean reference model. We exploit a key property of dormant poisoning: the attacker must preserve safe behavior at release to conceal poisoning, and this behavior can serve as a safety reference. Leveraging this property, we automatically construct a Safety Buffer pairing harmful inputs with the released model's responses and benign versions generated by the same model to remove harmful intent. The Safety Buffer can replace external safe responses in existing defenses, but these defenses do not distinguish updates that weaken safety from those that preserve it, often increasing over-refusal or reducing utility. To address this limitation, we introduce LookAhead Defense, which previews benign fine-tuning updates using the Safety Buffer and penalizes only updates predicted to weaken safe behavior on harmful inputs or increase the likelihood of refusals on benign versions in the Safety Buffer. Across language models, language agents, and multimodal agents, LookAhead Defense reduces harmful behavior in agents from up to 93.9% to at most 14.2% and harmful responses in language models from up to 92.5% to at most 2.0% versus vanilla fine-tuning, while maintaining competitive utility and low over-refusal. It remains effective against other poisoning methods. 

---
# The Judge Can Debias Itself: Awareness and Self-Debiasing for Reliable LLM-as-a-Judge 

**Authors**: Youbing Yin, Jingyu Wu, Anoop Kumar, Scott Novotney, Daben Liu, Erin Babinsky  

**Link**: [PDF](https://arxiv.org/pdf/2610.24320)  

**Abstract**: LLM-as-a-Judge systems are increasingly employed for automated evaluation, yet remain susceptible to systematic biases that undermine their reliability. We demonstrate that the judge can effectively debias itself without additional training. We propose two lightweight strategies: bias-aware prompting, which explicitly alerts the judge to potential target biases prior to evaluation, and self-debiasing, in which the judge critiques the initial judgment for bias influence. To broaden the evaluation scope, we introduce three novel benchmarks--authority bias (fabricated citations), confidence bias (assertive phrasing), and formatting bias (structured presentation)--and evaluate across seven bias types in total. Across three frontier models, self-debiasing improves performance in 20 out of 21 model-bias combinations with gains up to 15 percentage points, while bias-aware prompting also yields consistent improvements. The recent fine-tuned external critic, however, does not generalize as reliably to the newly-introduced bias types (degradation observed in 5 out of 9 cases). Our findings demonstrate that prompting a judge to reason about bias, either preemptively or retrospectively, is a robust and generalizable approach for enhancing evaluation integrity. 

---
# Not Every User Says It All: Training Tool-Using Agents Beyond Ideal Simulated Users 

**Authors**: Han Bao, Zheyuan Zhang, Kaiwen Shi, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.24294)  

**Abstract**: Tool-using agents are trained and evaluated almost exclusively with strong simulated users who state every detail clearly, yet real users are often weaker. We construct weak users by masking sentences of a simulated user's replies, holding the task and the user's knowledge fixed, and cross-evaluate agents trained with strong and weak users on tau2-bench and ToolWOZ. Weak users lower the success of every policy, and training with them does not help: no policy trained with weak users, through fixed masks, mixtures or a curriculum, serves weak users better than the strong-trained policy beyond run-to-run noise, and prompting the agent to ask more does not help either. We propose StaGPO (State-Grouped Policy Optimization): the agent writes what the task needs, what is missing and how it is being filled, and GRPO uses this state to compare turns of different rollouts that reach the same situation, assigning step-level credit. Within the standard recipe of a small SFT cold start followed by RL, StaGPO makes weak-user training useful: on ToolWOZ, the Qwen3.5-2B policy trained with weak users reaches 76.8% with weak users, the highest of any Qwen3.5-2B policy we trained, 4.8 points above GRPO and 2.8 above the strongest baseline; it is also 5.4 points above GRPO with strong users, and generalises to an unseen mask rate; trained with strong users instead, it serves weak users at least as well as GRPO trained with weak users. On tau2-bench it scores highest with weak users for both models, by a small margin for Qwen3.5-4B. 

---
# Desirability Confirmation: Relational Sycophancy in AI Intimate Roleplay 

**Authors**: Huiqian Lai, Runze Ding  

**Link**: [PDF](https://arxiv.org/pdf/2610.24265)  

**Abstract**: Research on LLM sycophancy has largely focused on propositional agreement, in which models echo users' beliefs, preferences, or judgments. Adult intimate roleplay presents a different case because what the user seeks may be preservation of a preferred relational position within a generated scene. Using a controlled prompt-based audit of 2,016 outputs from two open-weight LLMs, we identify four recurrent relational moves: legitimating desire, presenting the user as wanted, preserving non-rejection while communicating limits, and keeping intimacy available within fantasy or roleplay frames. Bringing the sycophancy literature into dialogue with sexual script theory, we conceptualize this configuration as desirability confirmation, an output-level pattern that captures how user-pleasing can operate through relational positioning. The findings extend sycophancy research and intimate-AI evaluation beyond binary compliance or refusal by showing how outputs can differ in the ways they communicate boundaries, preserve relational acceptance, and mark intimacy as fictional. 

---
# Probing Ambiguity: Adversarial Intent Resolution for Robust Text-to-SQL 

**Authors**: Youbing Yin, Xujun Peng, Muhammad S. Uddin, Erin Babinsky, Daben Liu, Anoop Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.24248)  

**Abstract**: Text-to-SQL systems typically assume a one-to-one mapping between natural-language queries and correct SQL translations. In practice, however, many queries are inherently ambiguous, admitting multiple valid interpretations that yield distinct outputs. We introduce Adversarial Intent Resolution (AIR), a multi-agent framework that actively probes and resolves such ambiguity without any task-specific training. AIR first constructs competing interpretations, verifies through SQL execution that they produce genuinely different results, and generates targeted clarification questions grounded in the observed data differences. Evaluating on leading benchmarks for ambiguous questions, AIR demonstrates strong performance, efficiency, and robust cross-benchmark transferability. On the AMBROSIA benchmark under schema-only context, a realistic production setting where exposing the full database is infeasible, AIR outperforms IntentRL, the state-of-the-art reinforcement learning model, across all metrics (e.g., F1 0.679 vs. 0.618). Crucially, AIR exhibits only ~7% performance degradation without the full database, compared to ~20% for IntentRL. On the AmbiQT benchmark, where ambiguity arises from schema design rather than linguistic structure, AIR achieves higher F1 under both context settings without any benchmark-specific adaptation. 

---
# Continual Test-Time Learning with Evolution Strategies 

**Authors**: Guangya Hao, Yiting Duan, Longyan Tan, Yunbo Long, Hao Wu, Ge Zheng, Alexandra Brintrup, Zhuokai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.24137)  

**Abstract**: Large language models increasingly operate in open-ended environments, where they are expected to continually adapt to evolving tasks and distributions, often without access to external supervision. Existing continual learning approaches rely on external supervision, such as labeled data, demonstrations, or reward signals, which can be costly or unavailable in practice. Test-time adaptation removes this dependence by learning from self-generated supervision, but existing methods derive that supervision from the single policy being updated, so once the policy drifts, its own learning signal drifts with it, leading to policy collapse, degraded reasoning, and catastrophic forgetting. In this paper, we propose \textbf{Evolution Strategy Reinforcement Learning (ESRL)}, a continual test-time learning framework that draws its self-generated supervision from a population of policies instead of the single policy being trained. ESRL perturbs an anchor policy in parameter space to form a population of nearby policies, samples one trajectory from each, and aggregates their answers into label-free group-relative learning signals for the central student. The anchor is held fixed within each block and refreshed from the student at block boundaries, so the population stays centered on a stable yet current policy and continues to supply diverse and reliable supervision as the student moves. In single-task test-time adaptation across mathematical reasoning, general knowledge, and medical domains, ESRL improves average accuracy by up to 31.7% relative to the base model and 16.6% relative to the strongest test-time adaptation baseline. Notably, ESRL substantially reduces catastrophic forgetting and achieves the strongest final performance in continual learning. 

---
# Inside the Loop: A Mechanistic Study of Weight-Tied Transformers on Depth-Bound Algorithmic Tasks 

**Authors**: Magnus Sesodia, Juil Sock, Philip Torr, Christian Schroeder de Witt  

**Link**: [PDF](https://arxiv.org/pdf/2610.24135)  

**Abstract**: Solving algorithmic tasks with many sequential steps typically requires scaling up transformer depth or chain-of-thought length, incurring unfavorable parameter and inference-compute costs, respectively. Looped transformers, which reuse one block of layers across many iterations, are poised to address this, since they scale effective depth without adding parameters or tokens. Yet, it remains unclear what each iteration is actually computing, or whether the loop merely simulates a deeper untied stack. Here, we study directed permutation traversal, in which a model follows $H$ hops from a starting node and reports the node it reaches, so every intermediate hop has a known ground truth. We find that a minimum effective depth is necessary to solve the task, beyond which the degree of weight tying becomes the determining factor. For instance, a looped transformer with a four-layer block iterated eight times solves the task in every seed, while an identically-trained untied transformer of the same effective depth never does, succeeding only with a lower learning rate or supervision of its intermediate layers. Probing and intervening on the residual stream, we find that the looped model advances through the intermediate nodes gradually, forming a staircase that adapts to its iteration budget, whereas the untied model, at that lower rate, exposes them only in its final layers. Taken together, our results recast looped transformers as iterative program learners, which allocate their iterations to successive steps of the algorithm and solve tasks that identically-trained untied models of the same effective depth do not. Code is available at this https URL. 

---
# Joint, Decoupled, or Agent: When Each Multimodal Alignment Paradigm Wins 

**Authors**: Yi Fang, Gaurav Srivastava, Sirui Ding, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24108)  

**Abstract**: Jointly pretrained multimodal encoders are a common way to combine modalities, while alternatives integrate their information during downstream training or inference. When does each approach win? We compare three paradigms, instantiated with public checkpoints, across 16 paired datasets in five domains and multiple training-label budgets. JOINT trains a fusion head on frozen, jointly pretrained encoders. DECOUPLED trains a fusion head on frozen, independently pretrained encoders. AGENT gives a frozen LLM task instructions, labelled examples, available sample text, and supervised unimodal readouts. No paradigm dominates the complete 144 dataset-budget combinations: DECOUPLED wins 54, JOINT 44, and AGENT 46 by mean utility. ScienceQA, MELD, and MetMeme shift from AGENT to fusion as supervision increases. Between the fusion paradigms, JOINT leads on four of five QA datasets across all evaluated budgets, while higher unimodal 5-NN accuracy is associated with a relative shift toward DECOUPLED across eight classification datasets (Spearman $\rho=-0.67$). We test whether representation and task diagnostics predict these preferences on held-out datasets before fitting their candidate systems. A two-step cascade combines these diagnostics with task and budget metadata to select AGENT or fusion, then JOINT or DECOUPLED. Under leave-one-group-out evaluation, it selects the winner in 71.4% of cells versus 54.2% for metadata-only routing, with mean relative regret 0.028 versus 0.057. Using geometry, supervised representation diagnostics, and text-label matching together yields higher accuracy and lower regret than using any one of these sources with metadata. These results connect paradigm preferences to supervision budget, task format, and unimodal 5-NN accuracy. Target-task diagnostics thus support selecting among the three paradigms before candidate fitting. 

---
# Cache2LoRA: Understanding and Improving Context Internalization 

**Authors**: Yi Fang, Gaurav Srivastava, Zhenyu Bi, Sirui Ding, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24107)  

**Abstract**: Context-to-LoRA methods encode documents as low-rank parameter updates, allowing language models to answer without retaining the source text in their decoding context. How reliably do these updates preserve facts and support their use under different instructions? Controlled diagnosis of existing context-to-LoRA methods separates fact discrimination, target retention, and response adaptation. Competing sources influence answers, while target-only construction and memory-state optimization improve recovery at fixed update size. Recovering facts also does not ensure that answers follow response instructions. We introduce Cache2LoRA, which uses each query to aggregate evidence from a reusable frozen document cache before generating LoRA updates. Trainable encoding operates only on the query and memory-token suffix, and multi-style supervision teaches both evidence preservation and answer expression. Across 17 tasks, Cache2LoRA improves on existing context-to-LoRA methods and achieves macro F1 of 0.663, approaching full-context in-context learning at 0.671. For 32K documents and 256-token outputs at batch size 16, its decoding throughput is 5.9$\times$ that of full-context ICL. At matched memory size, generated LoRA preserves answer accuracy more reliably than the learned-KV variant under long, diverse intervening text and combines more effectively with retained document KV, yielding higher hybrid F1 and EM. 

---
# Coopshift: Post-Training Amplifies The Memory Curse In LLM Cooperation 

**Authors**: Jiayuan Liu, Shiyi Du, Tianqin Li, Emanuel Tewolde, Keyang Xuan, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2610.24086)  

**Abstract**: How does post-training change cooperation during repeated interaction? Within each of seven released model lineages, we compare pretrained (Base), supervised fine-tuned (SFT), and subsequent preference- or reward-optimized (Aligned) checkpoints across seven social dilemmas. Copies of each checkpoint play one another for 200 rounds, with visible history limited to the last two or forty rounds. The memory curse refers to lower cooperation when agents can see more past rounds. Our central finding is that post-training amplifies this decline. Averaged across the six binary-action games, expanding visible history reduces cooperation at every SFT and Aligned checkpoint. At each post-trained stage, these declines exceed Base's in six of seven lineages. Most of this average amplification is already present at SFT. With long history, both post-trained stages also cooperate less than Base on average in each binary-action game. In the symmetric binary games, their fitted cooperation rates fall more steeply on average than Base's as per-round cooperation costs rise, benefits fall, or noncooperation becomes more rewarding. The difference in cooperation rates between Aligned and SFT also changes from one-shot decisions to repeated play. When given the same past records, their choices depend on whether the instructions describe a final round or possible future interaction. Post-training evaluation should therefore go beyond isolated choices to assess whether cooperation is sustained in repeated interaction and how it depends on payoffs and visible history. 

---
# Location, Location, Generation: Where You Ask Matters for Generative Engine Optimization 

**Authors**: Mahe Chen, Nick Koudas  

**Link**: [PDF](https://arxiv.org/pdf/2610.24081)  

**Abstract**: Generative search engines answer local and commercial questions with a written answer and a few cited sources. A growing practice of generative engine optimization (GEO) monitors which sources and brands these answers show, typically from a single place. In this paper, we present a controlled audit of how the user's location changes the sources that generative engines cite. We ask 300 questions of ChatGPT with 19 declared locations, and of ChatGPT and Gemini from five network locations, each twice to measure run-to-run noise (31,200 answers). The engines take location from different channels. ChatGPT follows the location an application declares, down to the city: it writes the place into its own searches, and its search backend localizes results even when the queries name no place. It ignores the network. Gemini ignores the declared location and follows the country of the network, changing most of its sources even for questions that do not depend on place. The businesses the answers recommend change with the sources. Thus, visibility in generative search is specific to the user's location, and audits must control both channels. 

---
# World of SWECraft: Agent Collaboration at Scale via Simulated Software Worlds 

**Authors**: Shannon Zejiang Shen, Yapei Chang, Qiyao Liang, Benjamin Newman, Jenny Liang, Rulin Shao, Ziqiao Ma, Junjie Oscar Yin, Yung-Sung Chuang, Weijia Shi, Yejin Choi, Luke Zettlemoyer, Pang Wei Koh, Mohit Iyyer, Tongshuang Wu, David Sontag  

**Link**: [PDF](https://arxiv.org/pdf/2610.24061)  

**Abstract**: As teams of AI agents take on larger projects and long-standing open problems, agent collaboration offers another dimension for scaling inference-time compute. To understand what makes such collaboration effective and scalable, we develop Software World, a framework for simulating multi-agent coworking in realistic software ecosystems. It comes with a pipeline to create environments of real, interdependent Python packages of different topics, sizes, and dependency structures, which we call software worlds. In a simulation, agents maintain packages in the world with a high-level goal of improving software efficiency, and they are free to discover tasks to work on and collaborate with others. We evaluate agents' work in a world collectively by measuring speedup in hidden dependent packages. The framework allows us to study different elements of collaboration by varying communication, ownership, and resource allocation. We run simulations in three different worlds, totaling more than 10,000 hours of agent work across eight models. Overall, performance improvement scales with inference-time compute, with the best model improving downstream efficiency by up to 31%. We show that most agents can find meaningful tasks via collaboration, but some frontier models can prematurely stop looking for improvements and make little progress. We release the simulation framework and traces to support future studies of agent collaboration at this https URL. 

---
# LogicVal:A Logic-Formalized Benchmark for Constraint Satisfaction in Text-to-Image Generation 

**Authors**: Savitha Sam Abraham, Marjan Alirezaie, Hossein Rahnama  

**Link**: [PDF](https://arxiv.org/pdf/2610.24051)  

**Abstract**: Text-to-Image (T2I) generative models have made significant progress in producing high-fidelity images from natural-language prompts. Alongside this progress, numerous benchmarks have been developed to evaluate capabilities such as attribute binding, numeracy, spatial reasoning, and commonsense reasoning. In this work, we propose a new perspective on T2I evaluation by casting it as a constraint-satisfaction problem. We introduce a logical framework that formally specifies the benchmark domain, including objects, attributes, and relations, alongside the constraints that generated images must satisfy. We evaluated four state-of-the-art T2I models across 9 classes of constraints ranging from a single-object-attribute to multi-object relationships constraints involving logical operators ($\forall, \exists, count, \rightarrow$). Our results reveal a substantial gap between single-object and complex conditional multi-object relationship constraint classes, with the best-performing model achieving 98% accuracy on the former and just 53% on the latter. We assess compositional generalisation, finding that the best-performing model`s accuracy drops from 84% on single-constraint prompts to 20% when satisfying multiple constraints simultaneously. Identifying unsatisfiable constraints is also challenging, with the best model correctly recognizing only 23% of such prompts. Building on this framework, we introduce \textbf{LogicVal}, an evaluation methodology that combines neural visual detection with logical reasoning to assess constraint satisfaction in generated images. On CLEVR, LogicVal achieves ROC-AUC scores ranging from 0.88 to 0.98, indicating strong alignment with human judgments. These findings underscore the value of logic-based evaluation for characterizing T2I model capabilities and limitations. 

---
# LLM Evolution in a Simple Simulated Market 

**Authors**: Charlie Pilgrim, Christopher Krupenye, Dora Biro, Richard P. Mann  

**Link**: [PDF](https://arxiv.org/pdf/2610.24033)  

**Abstract**: Arguments that AI systems will evolve in harmful ways are largely made in the abstract. We provide a concrete empirical example. We simulate simple markets consisting of populations of twenty Large Language Model agents tasked with solving arithmetic problems. Agents differ only in their system prompt; those that go bankrupt are removed and replaced by mutated copies of the prompts of successful agents. Two market conditions select for opposite phenotypes: a market that rewards accuracy produces agents with high accuracy (94%) and high token use; a market that charges per token produces agents with low accuracy (10%) and low token use. In the second market, only tokens are incentivised yet we see a collapse in accuracy. Token use and accuracy are correlated traits such that selection on one also moves the other, and behaviour adapts along an unpriced dimension. A third neutral condition with no selection allows us to separate these effects from bias in the mutation operator. Adaptation occurs through a collective optimisation process that operates above the level of individual agents, who are never told the payment rule, do not receive feedback, and have no memory. 

---
# LLM Evolutionary Search Strategies Should be Compared by Tokens 

**Authors**: Tal Oved, Roi Pony, Oshri Naparstek, Udi Barzelay  

**Link**: [PDF](https://arxiv.org/pdf/2610.23977)  

**Abstract**: LLM evolutionary search is currently evaluated by counting proposals (candidate programs). We demonstrate this metric hides the true cost of search, and redo the comparison using tokens. By replaying 600 logged runs across 5 tasks and 3 methods, we show that proposal costs vary wildly from 16k to 39k tokens. In equal-proposal comparisons, this silently grants some methods up to 1.9x the compute of others, and gives evolution roughly 6x the budget of independent sampling. When we match token budgets, giving sampling six times more attempts, and remove scorer exploits, the evolutionary advantage largely vanishes. Evolution beats independent sampling by 5% or less on 12 of 15 task-method pairs, retaining a 23 to 28% margin on only 3, all on one task. A replication with Claude Haiku 4.5 reproduces this split, though a third backbone shows its size depends on the model too. The performance margin justifying these systems is often an accounting artifact. Reporting tokens, not proposals and not model calls, fixes this and costs nothing, as the logs already hold them. 

---
# UltraData-Code: From Raw Repositories to Algorithmically Dense and Task-Oriented Code Data 

**Authors**: Chengying Tu, Hengyu Zhao, Shuaikang Xue, Zhongming Qu, Jihao Zhou, Xinle Lin, Junshao Guo, Zixuan Fu, Qiang Ma, Jie Zhou, Chaojun Xiao, Hongfei Yan, Yudong Wang, Xu Han, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.23960)  

**Abstract**: Code generation has become a core capability of large language models (LLMs), and code data is central to the pre-training process that develops it. As obtaining public code of sufficient quality becomes more difficult, further progress cannot rely on corpus expansion alone. Tiered data management instead applies increasingly targeted processing to smaller data pools, producing data states suited to different training objectives. Following this perspective, we present UltraData-Code as four connected data states spanning repository archival at L0, standardized natural code at L1, algorithmic selection at L2, and task-oriented synthesis at L3. L0 preserves repository data and provenance for reprocessing, while L1 applies scalable filtering, normalization, and near-deduplication for broad pre-training. L2 selects high-quality algorithmically relevant files using language-adaptive relevance signals and quality constraints, yielding approximately 400B tokens of UltraData-Code-L2 across 11 programming languages for focused continual pre-training. L3 converts selected implementations into structured programming exercises, yielding approximately 150B tokens of UltraData-Code-L3 across the same 11 languages for task-oriented mid-training or supervised fine-tuning (SFT). Under controlled 10B-token continual pre-training of a 1B model, L2 improves pass@1 over L1 by 7.80 percentage points (pp) on EvalPlus and 5.13pp on MultiPL-E. Replacing half of the L2 training tokens with L3 yields a further gain of 8.42pp and 8.07pp on the two benchmarks, respectively. Both gains increase when the training budget scales to 100B tokens. We have released our datasets and classifiers to support further research on LLMs. 

---
# How Data Scales in Agentic Reinforcement Learning: Laws and Synthesis Strategies 

**Authors**: Bowei He, Yankai Chen, Xiaokun Zhang, Changjiang Han, Ye Yuan, Chong Li, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.23946)  

**Abstract**: Pretraining scaling laws treat training data as a one-dimensional quantity: a token count. Agentic reinforcement learning (RL) inherits this scalar abstraction, but modern synthesis pipelines now generate \textit{tasks}, \textit{environments}, and \textit{trajectories} independently and at very different costs, turning data into a structured design space. This raises a question pretraining never had to answer: given a synthesis budget, \textit{which axis} should be scaled? We address this question end-to-end with three tightly coupled contributions. First, we decompose agentic data into three cardinality axes (task, environment, trajectory), plus a reward-density control variable, and fit per-axis scaling relations on different large language models (LLMs) across math+code, SWE, web, and UI/tool application domains. Across all model sizes, model families, domains, and RL algorithms (PPO and GRPO) we measured, the scaling exponents order as environment $>$ task $>$ trajectory. When environment diversity is decomposed into world and environment-structure diversity, its advantage is mainly driven by the former, while the latter is just comparable to the task diversity. Second, we use the fitted laws as a common ruler to price synthesis methods, capturing each method's efficiency as a per-axis discount factor and identifying critical synthetic-to-real ratios at which collapse begins; these ratios differ by an order of magnitude across axes. Third, we cast budget allocation as a constrained optimization under our laws and discounts, deriving a Chinchilla-style compute-optimal recipe and validating it at held-out scale. The empirical results demonstrate that following the recipe yields up to $1.7\times$ compute efficiency over the dominant ``scale-the-trajectories'' practice. 

---
# TeleTwin: A Calibrated Digital Twin of Telegram Crisis Discourse as a Ground-Truth Testbed for Detecting LLM-Powered Information Operations 

**Authors**: Despoina Antonakaki, Sotiris Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2610.23944)  

**Abstract**: Large language models make it cheap to run influence operations whose messages never repeat, yet real platforms offer almost no ground truth for testing whether coordination detectors can find them. The problem is sharpest on Telegram, which moderates little and spreads content through broadcast channels and forwarding rather than a follow graph. We introduce TeleTwin, an LLM-powered digital twin of a Telegram crisis-discourse ecosystem in which channels are agents, each instantiated from the history of a real channel, calibrated on 1.07 million messages from 195 public channels on the Israel-Gaza and Iran-Israel conflict (March-August 2025). On a held-out period containing the June 2025 Iran-Israel war, the twin tracks daily dynamics better than statistical baselines (correlation 0.86 against 0.64) and reacts to the war unprompted, but it over-focuses on the conflict, over-amplifies forwarding cascades, underestimates the shock and writes recognisably LLM-like text. We then inject LLM-written campaigns with known membership into the real data and into the twin. In real data, identical-text, co-URL and synchrony detectors fail once a campaign rewords, staggers or drops links; multilingual semantic similarity recovers loud campaigns, including cross-lingual ones, but not quiet ones, and its false positives fall unevenly across political perspectives. Inside the twin all detectors fail, because the simulated background is about 40 times more semantically homogeneous than the real one. Digital twins can supply ground truth for campaigns, but not, without calibrating their diversity, a realistic background for evaluating semantic detectors. 

---
# Latent Multimodal Memory Reasoning 

**Authors**: Cai Ke, Xinghao Chen, Yuanchen Bei, Yuni Lai, Keyu Chen, Ruizhi Qiao, Siyu An, Junnan Dong, Ruifeng Xu, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.23890)  

**Abstract**: Long-term personalized agents are required to reason over distributed history across modalities and evolving dialogue states. While existing flat cross-modal methods compress visual evidence prematurely and struggle with temporally stale observations, they neglect the necessity to unify multimodal and cross-session evidence. Crucially, raw multimodal memories are inherently entangled and heterogeneous, making fine-grained visual grounding and cross-modal reasoning computationally prohibitive. To this end, we present latent multimodal memory reasoning, i.e., LMMR, a novel neuro-symbolic framework that shifts long-term multimodal memory organization into a continuous latent space. Specifically, (i) instead of converting images to text and relying on static retrieval, LMMR maps both textual hidden states and visual patch features into a shared Sparse Autoencoder concept space, establishing sparse activations as common semantic addresses to dynamically construct a turn-level multimodal memory graph. (ii) A graph encoder then aggregates cross-modal and temporal dependencies across the task-specific latent subgraph, injecting the graph readout as a continuous residual rollout into the generation backbone while retaining native visual tokens to preserve fine-grained details. Extensive experiments on nine Mem-Gallery tasks and twelve MemEye settings demonstrate that LMMR significantly outperforms state-of-the-art baselines in capturing complex multimodal dependencies while enabling faithful and grounded responses. 

---
# GardenBench: A lightweight, daily evaluation of LLM capabilities 

**Authors**: Benedict Cummins-Mburu, Mariana Garcia Mejia, Oscar Heath, Sabrina Kreyzerman, Ellie Murray, Arusan Surendiran, Rohan Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2610.23866)  

**Abstract**: We introduce GardenBench, a benchmark with 11 test suites that evaluate the capabilities of Large Language Models (LLMs) across a range of tasks. By systematically prompting LLMs with various lightweight queries, we document day-to-day changes in model performance, including run costs and times, that other benchmarks miss. This allows us to compare the accuracy and features of responses across model providers and versions. We account for daily variations in model accuracy, capturing both model progress and degradation, for instance, when a model is changed post-deployment or when a new model is released. We base our evaluation on general-use tasks and make our results publicly accessible via a live dashboard, diversifying existing approaches to LLM testing. Our main contribution is an innovative, lightweight, and inexpensive benchmark, designed to be consumer-facing and for everyday tasks. 

---
# I$^2$-Med: Interpretable Medical Inference through Visual-Guided Dynamic Logits Calibration 

**Authors**: Md Sajid, Shreeyut Maheshwari, Akshat Mishra, Ritik Mishra, Mohammad Tanveer  

**Link**: [PDF](https://arxiv.org/pdf/2610.23858)  

**Abstract**: Medical vision language models have demonstrated strong potential for medical image understanding and reasoning. However, their interpretability and reliability remain limited because medical vision language tasks require precise visual grounding and clinically coherent explanations. This is particularly challenging due to complex imaging patterns and scarce expert annotations. These challenges reduce the effectiveness of conventional reasoning methods and lead to visual hallucination, defined as the generation of text that is not grounded in the given visual input, which further limits their applicability in clinical practice. To mitigate these challenges, we propose an interpretable medical inference (I$^2$-Med) framework that enhances visual faithfulness during inference without additional training to improve reasoning and explanation faithfulness in medical vision language models. During text generation, I$^2$-Med uses a visual guidance module to check whether each candidate word is supported by the medical image and consistent with the current reasoning. Based on this guidance, the model's output scores are adjusted to favor visually grounded words, without changing the model parameters. Extensive experiments on the OmniMedVQA benchmark, covering eight medical imaging modalities and five clinical question types, demonstrate that our proposed I$^2$-Med framework generates more clinically meaningful and visually faithful reasoning chains and reduces visual hallucination under standard medical VQA evaluation metrics. The source code is publicly available at \url{this https URL}. 

---
# Multi-LLM Systems Exhibit Robust Semantic Collapse 

**Authors**: Weiyi Kong, Shiyang Lai, Jinghua Piao, James Evans  

**Link**: [PDF](https://arxiv.org/pdf/2605.17193)  

**Abstract**: Whether machines can originate novel content has been debated for nearly two centuries, from Lovelace's assertion that no engine can "originate anything" to Turing's question of whether a machine can amplify ideas brought in from outside. Interacting multi-LLM systems, increasingly deployed for autonomous generation, reopen this question empirically. Here we show that such systems exhibit semantic collapse, systematic convergence in semantic representations despite apparent lexical variation, with no retraining or weight updates. The pattern holds across model families and over 200 to 1,000 rounds. Thirteen intervention implementations tested, spanning decoding parameters, prompt design, agent composition, activation engineering, and reinforcement learning, did not reliably restore semantic diversity. Mechanistic analyses suggest that semantic collapse is not solely attributable to the alignment or conformity biases examined here, but is consistent with intrinsic properties of autoregressive generation. Our results point to persistent constraints on the ability of multi-LLM systems to sustain open-ended exploration in closed-loop settings. 

---
