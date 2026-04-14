# Saar-Voice: A Multi-Speaker Saarbrücken Dialect Speech Corpus 

**Authors**: Lena S. Oberkircher, Jesujoba O. Alabi, Dietrich Klakow, Jürgen Trouvain  

**Link**: [PDF](https://arxiv.org/pdf/2604.11803)  

**Abstract**: Natural language processing (NLP) and speech technologies have made significant progress in recent years; however, they remain largely focused on standardized language varieties. Dialects, despite their cultural significance and widespread use, are underrepresented in linguistic resources and computational models, resulting in performance disparities. To address this gap, we introduce Saar-Voice, a six-hour speech corpus for the Saarbrücken dialect of German. The dataset was created by first collecting text through digitized books and locally sourced materials. A subset of this text was recorded by nine speakers, and we conducted analyses on both the textual and speech components to assess the dataset's characteristics and quality. We discuss methodological challenges related to orthographic and speaker variation, and explore grapheme-to-phoneme (G2P) conversion. The resulting corpus provides aligned textual and audio representations. This serves as a foundation for future research on dialect-aware text-to-speech (TTS), particularly in low-resource scenarios, including zero-shot and few-shot model adaptation. 

---
# Psychological Concept Neurons: Can Neural Control Bias Probing and Shift Generation in LLMs? 

**Authors**: Yuto Harada, Hiro Taiyo Hamada  

**Link**: [PDF](https://arxiv.org/pdf/2604.11802)  

**Abstract**: Using psychological constructs such as the Big Five, large language models (LLMs) can imitate specific personality profiles and predict a user's personality. While LLMs can exhibit behaviors consistent with these constructs, it remains unclear where and how they are represented inside the model and how they relate to behavioral outputs. To address this gap, we focus on questionnaire-operationalized Big Five concepts, analyze the formation and localization of their internal representations, and use interventions to examine how these representations relate to behavioral outputs. In our experiment, we first use probing to examine where Big Five information emerges across model depth. We then identify neurons that respond selectively to each Big Five concept and test whether enhancing or suppressing their activations can bias latent representations and label generation in intended directions. We find that Big Five information becomes rapidly decodable in early layers and remains detectable through the final layers, while concept-selective neurons are most prevalent in mid layers and exhibit limited overlap across domains. Interventions on these neurons consistently shift probe readouts toward targeted concepts, with targeted success rates exceeding 0.8 for some concepts, indicating that the model's internal separation of Big Five personality traits can be causally steered. At the label-generation level, the same interventions often bias generated label distributions in the intended directions, but the effects are weaker, more concept-dependent, and often accompanied by cross-trait spillover, indicating that comparable control over generated labels is difficult even with interventions on a large fraction of concept-selective neurons. Overall, our findings reveal a gap between representational control and behavioral control in LLMs. 

---
# CLSGen: A Dual-Head Fine-Tuning Framework for Joint Probabilistic Classification and Verbalized Explanation 

**Authors**: WonJin Yoon, Kangyu Zhu, Ian Bulovic, Autumn Sehy, Yanjun Gao, Dmitriy Dligach, Majid Afshar, Timothy A. Miller  

**Link**: [PDF](https://arxiv.org/pdf/2604.11801)  

**Abstract**: With the recent progress of Large Language Models (LLMs), there is a growing interest in applying these models to solve complex and challenging problems. Modern LLMs, capable of processing long contexts and generating verbalized explanations, offer significant potential in addressing real-world applications. However, a critical hurdle in deploying LLMs for practical decision-making is their inability to provide reliable, quantitative probabilities. While task-specific fine-tuning of LLMs using traditional discriminative objectives (similar to encoder-only models) can yield probability estimates, this often leads to catastrophic forgetting and linguistic collapse. Consequently, the model loses its ability to generate explanations, severely undermining its interpretability and usability. To address this challenge, we propose CLSGen, a novel LLM fine-tuning framework designed for binary classification tasks. The CLSGen framework encompasses a new model architecture, training methodology, and data construction strategy to enable robust probability estimation without sacrificing the model's inherent explanation-generation capabilities. Experimental results across multiple benchmark datasets demonstrate that models fine-tuned with CLSGen outperform existing baselines in classification metrics (AUROC and F1-score). Regarding explanation, the results showed strong alignment between predicted labels and generated justifications, as well as high readability. 

---
# C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts 

**Authors**: Chenxi Qing, Junxi Wu, Zheng Liu, Yixiang Qiu, Hongyao Yu, Bin Chen, Hao Wu, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2604.11796)  

**Abstract**: Recently, large language models (LLMs) are capable of generating highly fluent textual content. While they offer significant convenience to humans, they also introduce various risks, like phishing and academic dishonesty. Numerous research efforts have been dedicated to developing algorithms for detecting AI-generated text and constructing relevant datasets. However, in the domain of Chinese corpora, challenges remain, including limited model diversity and data homogeneity. To address these issues, we propose C-ReD: a comprehensive Chinese Real-prompt AI-generated Detection benchmark. Experiments demonstrate that C-ReD not only enables reliable in-domain detection but also supports strong generalization to unseen LLMs and external Chinese datasets-addressing critical gaps in model diversity, domain coverage, and prompt realism that have limited prior Chinese detection benchmarks. We release our resources at this https URL. 

---
# General365: Benchmarking General Reasoning in Large Language Models Across Diverse and Challenging Tasks 

**Authors**: Junlin Liu, Shengnan An, Shuang Zhou, Dan Ma, Shixiong Luo, Ying Xie, Yuan Zhang, Wenling Yuan, Yifan Zhou, Xiaoyu Li, Ziwen Wang, Xuezhi Cao, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.11778)  

**Abstract**: Contemporary large language models (LLMs) have demonstrated remarkable reasoning capabilities, particularly in specialized domains like mathematics and physics. However, their ability to generalize these reasoning skills to more general and broader contexts--often termed general reasoning--remains under-explored. Unlike domain-specific reasoning, general reasoning relies less on expert knowledge but still presents formidable reasoning challenges, such as complex constraints, nested logical branches, and semantic interference. To address this gap, we introduce General365, a benchmark specifically designed to assess general reasoning in LLMs. By restricting background knowledge to a K-12 level, General365 explicitly decouples reasoning from specialized expertise. The benchmark comprises 365 seed problems and 1,095 variant problems across eight categories, ensuring both high difficulty and diversity. Evaluations across 26 leading LLMs reveal that even the top-performing model achieves only 62.8% accuracy, in stark contrast to the near-perfect performances of LLMs in math and physics benchmarks. These results suggest that the reasoning abilities of current LLMs are heavily domain-dependent, leaving significant room for improvement in broader applications. We envision General365 as a catalyst for advancing LLM reasoning beyond domain-specific tasks toward robust, general-purpose real-world scenarios. Code, Dataset, and Leaderboard: this https URL 

---
# Agentic Aggregation for Parallel Scaling of Long-Horizon Agentic Tasks 

**Authors**: Yoonsang Lee, Howard Yen, Xi Ye, Danqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11753)  

**Abstract**: We study parallel test-time scaling for long-horizon agentic tasks such as agentic search and deep research, where multiple rollouts are generated in parallel and aggregated into a final response. While such scaling has proven effective for chain-of-thought reasoning, agentic tasks pose unique challenges: trajectories are long, multi-turn, and tool-augmented, and outputs are often open-ended. Aggregating only final answers discards rich information from trajectories, while concatenating all trajectories exceeds the model's context window. To address this, we propose AggAgent, an aggregation agent that treats parallel trajectories as an environment. We equip it with lightweight tools to inspect candidate solutions and search across trajectories, enabling it to navigate and synthesize information on demand. Across six benchmarks and three model families (GLM-4.7, Qwen3.5, MiniMax-M2.5), AggAgent outperforms all existing aggregation methods-by up to 5.3% absolute on average and 10.3% on two deep research tasks-while adding minimal overhead, as the aggregation cost remains bounded by a single agentic rollout. Our findings establish agentic aggregation as an effective and cost-efficient approach to parallel test-time scaling. 

---
# HistLens: Mapping Idea Change across Concepts and Corpora 

**Authors**: Yi Jing, Weiyun Qiu, Yihang Peng, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2604.11749)  

**Abstract**: Language change both reflects and shapes social processes, and the semantic evolution of foundational concepts provides a measurable trace of historical and social transformation. Despite recent advances in diachronic semantics and discourse analysis, existing computational approaches often (i) concentrate on a single concept or a single corpus, making findings difficult to compare across heterogeneous sources, and (ii) remain confined to surface lexical evidence, offering insufficient computational and interpretive granularity when concepts are expressed implicitly. We propose HistLens, a unified, SAE-based framework for multi-concept, multi-corpus conceptual-history analysis. The framework decomposes concept representations into interpretable features and tracks their activation dynamics over time and across sources, yielding comparable conceptual trajectories within a shared coordinate system. Experiments on long-span press corpora show that HistLens supports cross-concept, cross-corpus computation of patterns of idea evolution and enables implicit concept computation. By bridging conceptual modeling with interpretive needs, HistLens broadens the analytical perspectives and methodological repertoire available to social science and the humanities for diachronic text analysis. 

---
# LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling 

**Authors**: Yuxin Chen, Chumeng Liang, Hangke Sui, Ruihan Guo, Chaoran Cheng, Jiaxuan You, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11748)  

**Abstract**: Continuous diffusion models have achieved strong performance across domains such as images. However, in language modeling, prior continuous diffusion language models (DLMs) lag behind discrete counterparts. In this work, we close this gap with LangFlow, the first continuous DLM to rival discrete diffusion. Our approach connects embedding-space DLMs to Flow Matching via Bregman divergence and introduces three key innovations: (1) a novel ODE-based NLL bound for principled evaluation of continuous flow-based language models; (2) an information-uniform principle for noise scheduling, motivating a learnable scheduler based on a Gumbel distribution; and (3) an improved training protocol incorporating self-conditioning, which enhances both likelihood and sample this http URL achieves strong performance across benchmarks, reaching a perplexity (PPL) of 30.0 on LM1B and 24.6 on OpenWebText. It matches top discrete DLMs at comparable scale and surpasses autoregressive baselines in zero-shot transfer across multiple benchmarks. LangFlow provides clear evidence that continuous diffusion is a competitive and promising paradigm for language modeling.
this https URL 

---
# Discourse Diversity in Multi-Turn Empathic Dialogue 

**Authors**: Hongli Zhan, Emma S. Gueorguieva, Javier Hernandez, Jina Suh, Desmond C. Ong, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11742)  

**Abstract**: Large language models (LLMs) produce responses rated as highly empathic in single-turn settings (Ayers et al., 2023; Lee et al., 2024), yet they are also known to be formulaic generators that reuse the same lexical patterns, syntactic templates, and discourse structures across tasks (Jiang et al., 2025; Shaib et al., 2024; Namuduri et al., 2025). Less attention has been paid to whether this formulaicity extends to the level of discourse moves, i.e., what a response does for the person it is addressing. This question is especially consequential for empathic dialogue, where effective support demands not just a kind response at one moment but varied strategies as a conversation unfolds (Stiles et al., 1998). Indeed, prior work shows that LLMs reuse the same tactic sequences more than human supporters in single-turn settings (Gueorguieva et al., 2026). We extend this analysis to multi-turn conversations and find that the rigidity compounds: once a tactic appears in a supporter turn, LLMs reuse it in the next at nearly double the rate of humans (0.50-0.56 vs. 0.27). This pattern holds across LLMs serving as supporters in real emotional support conversations, and is invisible to standard similarity metrics. To address this gap, we introduce MINT (Multi-turn Inter-tactic Novelty Training), the first reinforcement learning framework to optimize discourse move diversity across multi-turn empathic dialogue. The best MINT variant combines an empathy quality reward with a cross-turn tactic novelty signal, improving aggregate empathy by 25.3% over vanilla across 1.7B and 4B models while reducing cross-turn discourse move repetition by 26.3% on the 4B model, surpassing all baselines including quality-only and token-level diversity methods on both measures. These results suggest that what current models lack is not empathy itself, but the ability to vary their discourse moves across a conversation. 

---
# Evaluating Cooperation in LLM Social Groups through Elected Leadership 

**Authors**: Ryan Faulkner, Anushka Deshpande, David Guzman Piedrahita, Joel Z. Leibo, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.11721)  

**Abstract**: Governing common-pool resources requires agents to develop enduring strategies through cooperation and self-governance to avoid collective failure. While foundation models have shown potential for cooperation in these settings, existing multi-agent research provides little insight into whether structured leadership and election mechanisms can improve collective decision making. The lack of such a critical organizational feature ubiquitous in human society presents a significant shortcoming of the current methods. In this work we aim to directly address whether leadership and elections can support improved social welfare and cooperation through multi-agent simulation with LLMs. We present our open-source framework that simulates leadership through elected personas and candidate-driven agendas and carry out an empirical study of LLMs under controlled governance conditions. Our experiments demonstrate that having elected leadership improves social welfare scores by 55.4% and survival time by 128.6% across a range of high performing LLMs. Through the construction of an agent social graph we compute centrality metrics to assess the social influence of leader personas and also analyze rhetorical and cooperative tendencies revealed through a sentiment analysis on leader utterances. This work lays the foundation for further study of election mechanisms in multi-agent systems toward navigating complex social dilemmas. 

---
# Legal2LogicICL: Improving Generalization in Transforming Legal Cases to Logical Formulas via Diverse Few-Shot Learning 

**Authors**: Jieying Xue, Phuong Minh Nguyen, Ha Thanh Nguyen, May Myo Zin, Ken Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2604.11699)  

**Abstract**: This work aims to improve the generalization of logic-based legal reasoning systems by integrating recent advances in NLP with legal-domain adaptive few-shot learning techniques using LLMs. Existing logic-based legal reasoning pipelines typically rely on fine-tuned models to map natural-language legal cases into logical formulas before forwarding them to a symbolic reasoner. However, such approaches are heavily constrained by the scarcity of high-quality annotated training data. To address this limitation, we propose a novel LLM-based legal reasoning framework that enables effective in-context learning through retrieval-augmented generation. Specifically, we introduce Legal2LogicICL, a few-shot retrieval framework that balances diversity and similarity of exemplars at both the latent semantic representation level and the legal text structure level. In addition, our method explicitly accounts for legal structure by mitigating entity-induced retrieval bias in legal texts, where lengthy and highly specific entity mentions often dominate semantic representations and obscure legally meaningful reasoning patterns. Our Legal2LogicICL constructs informative and robust few-shot demonstrations, leading to accurate and stable logical rule generation without requiring additional training. In addition, we construct a new dataset, named Legal2Proleg, which is annotated with alignments between legal cases and PROLEG logical formulas to support the evaluation of legal semantic parsing. Experimental results on both open-source and proprietary LLMs demonstrate that our approach significantly improves accuracy, stability, and generalization in transforming natural-language legal case descriptions into logical representations, highlighting its effectiveness for interpretable and reliable legal reasoning. Our code is available at this https URL. 

---
# Please Make it Sound like Human: Encoder-Decoder vs. Decoder-Only Transformers for AI-to-Human Text Style Transfer 

**Authors**: Utsav Paneru  

**Link**: [PDF](https://arxiv.org/pdf/2604.11687)  

**Abstract**: AI-generated text has become common in academic and professional writing, prompting research into detection methods. Less studied is the reverse: systematically rewriting AI-generated prose to read as genuinely human-authored. We build a parallel corpus of 25,140 paired AI-input and human-reference text chunks, identify 11 measurable stylistic markers separating the two registers, and fine-tune three models: BART-base, BART-large, and Mistral-7B-Instruct with QLoRA. BART-large achieves the highest reference similarity -- BERTScore F1 of 0.924, ROUGE-L of 0.566, and chrF++ of 55.92 -- with 17x fewer parameters than Mistral-7B. We show that Mistral-7B's higher marker shift score reflects overshoot rather than accuracy, and argue that shift accuracy is a meaningful blind spot in current style transfer evaluation. 

---
# Playing Along: Learning a Double-Agent Defender for Belief Steering via Theory of Mind 

**Authors**: Hanqi Xiao, Vaidehi Patil, Zaid Khan, Hyunji Lee, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2604.11666)  

**Abstract**: As large language models (LLMs) become the engine behind conversational systems, their ability to reason about the intentions and states of their dialogue partners (i.e., form and use a theory-of-mind, or ToM) becomes increasingly critical for safe interaction with potentially adversarial partners. We propose a novel privacy-themed ToM challenge, ToM for Steering Beliefs (ToM-SB), in which a defender must act as a Double Agent to steer the beliefs of an attacker with partial prior knowledge within a shared universe. To succeed on ToM-SB, the defender must engage with and form a ToM of the attacker, with a goal of fooling the attacker into believing they have succeeded in extracting sensitive information. We find that strong frontier models like Gemini3-Pro and GPT-5.4 struggle on ToM-SB, often failing to fool attackers in hard scenarios with partial attacker prior knowledge, even when prompted to reason about the attacker's beliefs (ToM prompting). To close this gap, we train models on ToM-SB to act as AI Double Agents using reinforcement learning, testing both fooling and ToM rewards. Notably, we find a bidirectionally emergent relationship between ToM and attacker-fooling: rewarding fooling success alone improves ToM, and rewarding ToM alone improves fooling. Across four attackers with different strengths, six defender methods, and both in-distribution and out-of-distribution (OOD) evaluation, we find that gains in ToM and attacker-fooling are well-correlated, highlighting belief modeling as a key driver of success on ToM-SB. AI Double Agents that combine both ToM and fooling rewards yield the strongest fooling and ToM performance, outperforming Gemini3-Pro and GPT-5.4 with ToM prompting on hard scenarios. We also show that ToM-SB and AI Double Agents can be extended to stronger attackers, demonstrating generalization to OOD settings and the upgradability of our task. 

---
# Hidden Failures in Robustness: Why Supervised Uncertainty Quantification Needs Better Evaluation 

**Authors**: Joe Stacey, Hadas Orgad, Kentaro Inui, Benjamin Heinzerling, Nafise Sadat Moosavi  

**Link**: [PDF](https://arxiv.org/pdf/2604.11662)  

**Abstract**: Recent work has shown that the hidden states of large language models contain signals useful for uncertainty estimation and hallucination detection, motivating a growing interest in efficient probe-based approaches. Yet it remains unclear how robust existing methods are, and which probe designs provide uncertainty estimates that are reliable under distribution shift. We present a systematic study of supervised uncertainty probes across models, tasks, and OOD settings, training over 2,000 probes while varying the representation layer, feature type, and token aggregation strategy. Our evaluation highlights poor robustness in current methods, particularly in the case of long-form generations. We also find that probe robustness is driven less by architecture and more by the probe inputs. Middle-layer representations generalise more reliably than final-layer hidden states, and aggregating across response tokens is consistently more robust than relying on single-token features. These differences are often largely invisible in-distribution but become more important under distribution shift. Informed by our evaluation, we explore a simple hybrid back-off strategy for improving robustness, arguing that better evaluation is a prerequisite for building more robust probes. 

---
# RPA-Check: A Multi-Stage Automated Framework for Evaluating Dynamic LLM-based Role-Playing Agents 

**Authors**: Riccardo Rosati, Edoardo Colucci, Massimiliano Bolognini, Adriano Mancini, Paolo Sernani  

**Link**: [PDF](https://arxiv.org/pdf/2604.11655)  

**Abstract**: The rapid adoption of Large Language Models (LLMs) in interactive systems has enabled the creation of dynamic, open-ended Role-Playing Agents (RPAs). However, evaluating these agents remains a significant challenge, as standard NLP metrics fail to capture the nuances of role adherence, logical consistency, and long-term narrative stability. This paper introduces RPA-Check, a multi-stage automated evaluation framework designed to objectively assess the performance of LLM-based RPAs in complex, constraints-heavy environments. Our methodology is based on a four-step pipeline: (1) Dimension Definition, establishing high-level qualitative behavioral criteria; (2) Augmentation, where these requirements are expanded into granular boolean checklist indicators; (3) Semantic Filtering, to ensure indicator objectivity, no redundancy and agent isolation; and (4) LLM-as-a-Judge Evaluation, which employs chain-of-thought verification to score agent fidelity. We validate this framework by applying it to LLM Court, a serious game for forensic training involving several quantized local models. Experimental results across five distinct legal scenarios demonstrate the framework's ability to identify subtle trade-offs between model size, reasoning depth, and operational stability. Notably, the findings reveal an inverse relationship between parametric scale and procedural consistency, showing that smaller, adequately instruction-tuned models (8-9B) can outperform larger architectures prone to user-alignment bias or sycophancy. RPA-Check thus provides a standardized and reproducible metric for future research in generative agent evaluation within specialized domains. 

---
# CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity 

**Authors**: Xuefeng Wei, Zhixuan Wang, Xuan Zhou, Zhi Qu, Hongyao Li, Yusuke Sakai, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2604.11632)  

**Abstract**: We introduce CARTBENCH, a museum-grounded benchmark for evaluating vision-language models (VLMs) on Chinese artworks beyond short-form recognition and QA. CARTBENCH comprises four subtasks: CURATORQA for evidence-grounded recognition and reasoning, CATALOGCAPTION for structured four-section expert-style appreciation, REINTERPRET for defensible reinterpretation with expert ratings, and CONNOISSEURPAIRS for diagnostic authenticity discrimination under visually similar confounds. CARTBENCH is built by aligning image-bearing Palace Museum objects from Wikidata with authoritative catalog pages, spanning five art categories across multiple dynasties. Across nine representative VLMs, we find that high overall CURATORQA accuracy can mask sharp drops on hard evidence linking and style-to-period inference; long-form appreciation remains far from expert references; and authenticity-oriented diagnostic discrimination stays near chance, underscoring the difficulty of connoisseur-level reasoning for current models. 

---
# Back to Basics: Let Conversational Agents Remember with Just Retrieval and Generation 

**Authors**: Yuqian Wu, Wei Chen, Zhengjun Huang, Junle Chen, Qingxiang Liu, Kai Wang, Xiaofang Zhou, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11628)  

**Abstract**: Existing conversational memory systems rely on complex hierarchical summarization or reinforcement learning to manage long-term dialogue history, yet remain vulnerable to context dilution as conversations grow. In this work, we offer a different perspective: the primary bottleneck may lie not in memory architecture, but in the \textit{Signal Sparsity Effect} within the latent knowledge manifold. Through controlled experiments, we identify two key phenomena: \textit{Decisive Evidence Sparsity}, where relevant signals become increasingly isolated with longer sessions, leading to sharp degradation in aggregation-based methods; and \textit{Dual-Level Redundancy}, where both inter-session interference and intra-session conversational filler introduce large amounts of non-informative content, hindering effective generation. Motivated by these insights, we propose \method, a minimalist framework that brings conversational memory back to basics, relying solely on retrieval and generation via Turn Isolation Retrieval (TIR) and Query-Driven Pruning (QDP). TIR replaces global aggregation with a max-activation strategy to capture turn-level signals, while QDP removes redundant sessions and conversational filler to construct a compact, high-density evidence set. Extensive experiments on multiple benchmarks demonstrate that \method achieves robust performance across diverse settings, consistently outperforming strong baselines while maintaining high efficiency in tokens and latency, establishing a new minimalist baseline for conversational memory. 

---
# Utilizing and Calibrating Hindsight Process Rewards via Reinforcement with Mutual Information Self-Evaluation 

**Authors**: Jiashu Yao, Heyan Huang, Zeming Liu, Yuhang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.11611)  

**Abstract**: To overcome the sparse reward challenge in reinforcement learning (RL) for agents based on large language models (LLMs), we propose Mutual Information Self-Evaluation (MISE), an RL paradigm that utilizes hindsight generative self-evaluation as dense reward signals while simultaneously calibrating them against the environmental feedbacks. Empirically, MISE enables an agent to learn autonomously from dense internal rewards supplementing sparse extrinsic signals. Theoretically, our work provides the first formal foundation for the paradigm of generative self-rewarding. We prove that utilizing hindsight self-evaluation rewards is equivalent to minimizing an objective that combines mutual information with a KL divergence term between the policy and a proxy reward policy. This theoretical insight then informs and justifies our calibration step, which actively aligns these rewards with the optimal policy. Extensive experiments show that MISE outperforms strong baselines, enabling open-source LLMs about 7B parameters to achieve performance comparable to GPT-4o on validation without expert supervision. 

---
# Self-Evolving LLM Memory Extraction Across Heterogeneous Tasks 

**Authors**: Yuqing Yang, Tengxiao Liu, Wang Bill Zhu, Taiwei Shi, Linxin Song, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.11610)  

**Abstract**: As LLM-based assistants become persistent and personalized, they must extract and retain useful information from past conversations as memory. However, the types of information worth remembering vary considerably across tasks. We formalize the \textit{heterogeneous memory extraction} task and introduce \textbf{BEHEMOTH}, a benchmark that repurposes 18 existing datasets spanning personalization, problem-solving, and agentic tasks, using a downstream utility-driven metric for systematic evaluation. Our empirical analysis confirms that no single static extraction prompt dominates across all task categories, and that existing self-evolving prompt optimization frameworks, originally designed for homogeneous distributions, degrade when training tasks are heterogeneous. To address this, we propose \textbf{CluE}, a cluster-based self-evolving strategy that groups training examples into clusters by extraction scenarios, analyzes each cluster independently, and synthesizes cross-cluster insights to update the extraction prompt. Experiments on BEHEMOTH show that CluE generalizes effectively across heterogeneous tasks ($+$9.04\% relative gain), consistently outperforming prior self-evolving frameworks. 

---
# A Triadic Suffix Tokenization Scheme for Numerical Reasoning 

**Authors**: Olga Chetverina  

**Link**: [PDF](https://arxiv.org/pdf/2604.11582)  

**Abstract**: Standard subword tokenization methods fragment numbers inconsistently, causing large language models (LLMs) to lose positional and decimal structure - a primary driver of errors in arithmetic and scientific reasoning. We introduce Triadic Suffix Tokenization (TST), a deterministic scheme that partitions digits into three-digit triads and annotates each triad with an explicit magnitude marker. Critically, the scheme defines a fixed, one-to-one mapping between suffixes and orders of magnitude for the integer part (thousands, millions, billions, etc.) and a parallel system of replicated markers for fractional depth (tenths, thousandths, millionths, etc.). Unlike approaches that rely on positional inference, this method provides a consistent gradient signal, which should ensure stable convergence. Two implementation variants are proposed: (1) a vocabulary-based approach that adds at most 10,000 fixed tokens to an existing vocabulary, covering 33 orders of magnitude ($10^{-15}$ to $10^{18}$); and (2) a suffix-marker approach that uses a small set of special tokens to denote magnitude dynamically. Both variants preserve exact digits while making order-of-magnitude relationships transparent at the token level. The framework is inherently scalable, allowing for linear vocabulary expansion to accommodate arbitrary precision and range. TST is architecture-agnostic and can be integrated as a drop-in preprocessing step. Experimental validation is deferred to future work. 

---
# Decomposing and Reducing Hidden Measurement Error in LLM Evaluation Pipelines 

**Authors**: Solomon Messing  

**Link**: [PDF](https://arxiv.org/pdf/2604.11581)  

**Abstract**: LLM evaluations drive which models get deployed, which safety standards get adopted, and which research conclusions get published. Yet these scores carry hidden uncertainty: rephrasing the prompt, switching the judge model, or changing the temperature can shift results enough to flip rankings and reverse conclusions. Standard confidence intervals ignore this variance, producing under-coverage that worsens with more data. The unmeasured variance also creates an exploitable surface: model developers can optimize against measurement noise rather than genuine capability. This paper decomposes LLM pipeline uncertainty into its sources, distinguishes variance that shrinks with more data from sensitivity to researcher design choices, and projects the most efficient path to reducing total error. For benchmark builders, the same decomposition identifies which design choices contribute exploitable surface for gaming and prescribes designs that minimize it. Across ideology annotation, safety classification, MMLU benchmarking, and a human-validated propaganda audit, projection-optimized pipelines outperform 73\% of possible naive pipelines against a human baseline. On MMLU, optimized budget allocation halves estimation error compared to standard single-prompt evaluation at equivalent cost. A small-sample variance estimation exercise is sufficient to derive confidence intervals that approach nominal coverage when the model includes the relevant pipeline facets, and to generate recommendations for reducing measurement error and improving benchmark robustness. 

---
# MIXAR: Scaling Autoregressive Pixel-based Language Models to Multiple Languages and Scripts 

**Authors**: Chen Hu, Yintao Tai, Antonio Vergari, Frank Keller, Alessandro Suglia  

**Link**: [PDF](https://arxiv.org/pdf/2604.11575)  

**Abstract**: Pixel-based language models are gaining momentum as alternatives to traditional token-based approaches, promising to circumvent tokenization challenges. However, the inherent perceptual diversity across languages poses a significant hurdle for multilingual generalization in pixel space. This paper introduces MIXAR, the first generative pixel-based language model trained on eight different languages utilizing a range of different scripts. We empirically evaluate MIXAR against previous pixel-based models as well as comparable tokenizer-based models, demonstrating substantial performance improvement on discriminative and generative multilingual tasks. Additionally, we show how MIXAR is robust to languages never seen during the training. These results are further strengthened when scaling the model to 0.5B parameters which not only improves its capabilities in generative tasks like LAMBADA but also its robustness when challenged with input perturbations such as orthographic attacks. 

---
# Phonological distances for linguistic typology and the origin of Indo-European languages 

**Authors**: Marius Mavridis, Juan De Gregorio, Raul Toral, David Sanchez  

**Link**: [PDF](https://arxiv.org/pdf/2604.11565)  

**Abstract**: We show that short-range phoneme dependencies encode large-scale patterns of linguistic relatedness, with direct implications for quantitative typology and evolutionary linguistics. Specifically, using an information-theoretic framework, we argue that phoneme sequences modeled as second-order Markov chains essentially capture the statistical correlations of a phonological system. This finding enables us to quantify distances among 67 modern languages from a multilingual parallel corpus employing a distance metric that incorporates articulatory features of phonemes. The resulting phonological distance matrix recovers major language families and reveals signatures of contact-induced convergence. Remarkably, we obtain a clear correlation with geographic distance, allowing us to constrain a plausible homeland region for the Indo-European family, consistent with the Steppe hypothesis. 

---
# Synthius-Mem: Brain-Inspired Hallucination-Resistant Persona Memory Achieving 94.4% Memory Accuracy and 99.6% Adversarial Robustness on LoCoMo 

**Authors**: Artem Gadzhiev, Andrew Kislov  

**Link**: [PDF](https://arxiv.org/pdf/2604.11563)  

**Abstract**: Providing AI agents with reliable long-term memory that does not hallucinate remains an open problem. Current approaches to memory for LLM agents -- sliding windows, summarization, embedding-based RAG, and flat fact extraction -- each reduce token cost but introduce catastrophic information loss, semantic drift, or uncontrolled hallucination about the user. The structural reason is architectural: every published memory system on the LoCoMo benchmark treats conversation as a retrieval problem over raw or lightly summarized dialogue segments, and none reports adversarial robustness, the ability to refuse questions about facts the user never disclosed. We present Synthius-Mem, a brain-inspired structured persona memory system that takes a fundamentally different approach. Instead of retrieving what was said, Synthius-Mem extracts what is known about the person: a full persona extraction pipeline decomposes conversations into six cognitive domains (biography, experiences, preferences, social circle, work, psychometrics), consolidates and deduplicates per domain, and retrieves structured facts via CategoryRAG at 21.79 ms latency. On the LoCoMo benchmark (ACL 2024, 10 conversations, 1,813 questions), Synthius-Mem achieves 94.37% accuracy, exceeding all published systems including MemMachine (91.69%, adversarial score is not reported) and human performance (87.9 F1). Core memory fact accuracy reaches 98.64%. Adversarial robustness, the hallucination resistance metric that no competing system reports, reaches 99.55%. Synthius-Mem reduces token consumption by ~5x compared to full-context replay while achieving higher accuracy. Synthius-Mem achieves state-of-the-art results on LoCoMo and is, to our knowledge, the only persona memory system that both exceeds human-level performance and reports adversarial robustness. 

---
# Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale 

**Authors**: Liujie Zhang, Benzhe Ning, Rui Yang, Xiaoyan Yu, Jiaxing Li, Lumeng Wu, Jia Liu, Minghao Li, Weihang Chen, Weiqi Hu, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11554)  

**Abstract**: Reinforcement learning (RL) post-training has proven effective at unlocking reasoning, self-reflection, and tool-use capabilities in large language models. As models extend to omni-modal inputs and agentic multi-turn workflows, RL training systems face three interdependent challenges: heterogeneous data flows, operational robustness at scale, and the staleness -- throughput tradeoff. We present \textbf{Relax} (Reinforcement Engine Leveraging Agentic X-modality), an open-source RL training engine that addresses these challenges through three co-designed architectural layers. First, an \emph{omni-native architecture} builds multimodal support into the full stack -- from data preprocessing and modality-aware parallelism to inference generation -- rather than retrofitting it onto a text-centric pipeline. Second, each RL role runs as an independent, fault-isolated service that can be scaled, recovered, and upgraded without global coordination. Third, service-level decoupling enables asynchronous training via the TransferQueue data bus, where a single staleness parameter smoothly interpolates among on-policy, near-on-policy, and fully asynchronous execution. Relax achieves a 1.20$\times$ end-to-end speedup over veRL on Qwen3-4B on-policy training. Its fully async mode delivers a 1.76$\times$ speedup over colocate on Qwen3-4B and a 2.00$\times$ speedup on Qwen3-Omni-30B, while all modes converge to the same reward level. Relax supports R3 (Rollout Routing Replay)~\cite{ma2025r3} for MoE models with only 1.9\% overhead, compared to 32\% degradation in veRL under the same configuration. It further demonstrates stable omni-modal RL convergence on Qwen3-Omni across image, text, and audio, sustaining over 2{,}000 steps on video without degradation. Relax is available at this https URL. 

---
# Time is Not a Label: Continuous Phase Rotation for Temporal Knowledge Graphs and Agentic Memory 

**Authors**: Weixian Waylon Li, Jiaxin Zhang, Xianan Jim Yang, Tiejun Ma, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.11544)  

**Abstract**: Structured memory representations such as knowledge graphs are central to autonomous agents and other long-lived systems. However, most existing approaches model time as discrete metadata, either sorting by recency (burying old-yet-permanent knowledge), simply overwriting outdated facts, or requiring an expensive LLM call at every ingestion step, leaving them unable to distinguish persistent facts from evolving ones. To address this, we introduce RoMem, a drop-in temporal knowledge graph module for structured memory systems, applicable to agentic memory and beyond. A pretrained Semantic Speed Gate maps each relation's text embedding to a volatility score, learning from data that evolving relations (e.g., "president of") should rotate fast while persistent ones (e.g., "born in") should remain stable. Combined with continuous phase rotation, this enables geometric shadowing: obsolete facts are rotated out of phase in complex vector space, so temporally correct facts naturally outrank contradictions without deletion. On temporal knowledge graph completion, RoMem achieves state-of-the-art results on ICEWS05-15 (72.6 MRR). Applied to agentic memory, it delivers 2-3x MRR and answer accuracy on temporal reasoning (MultiTQ), dominates hybrid benchmark (LoCoMo), preserves static memory with zero degradation (DMR-MSC), and generalises zero-shot to unseen financial domains (FinTMMBench). 

---
# NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment 

**Authors**: Wenqing Wu, Yi Zhao, Yuzhuo Wang, Siyou Li, Juexi Shao, Yunfei Long, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11543)  

**Abstract**: Novelty is a core requirement in academic publishing and a central focus of peer review, yet the growing volume of submissions has placed increasing pressure on human reviewers. While large language models (LLMs), including those fine-tuned on peer review data, have shown promise in generating review comments, the absence of a dedicated benchmark has limited systematic evaluation of their ability to assess research novelty. To address this gap, we introduce NovBench, the first large-scale benchmark designed to evaluate LLMs' capability to generate novelty evaluations in support of human peer review. NovBench comprises 1,684 paper-review pairs from a leading NLP conference, including novelty descriptions extracted from paper introductions and corresponding expert-written novelty evaluations. We focus on both sources because the introduction provides a standardized and explicit articulation of novelty claims, while expert-written novelty evaluations constitute one of the current gold standards of human judgment. Furthermore, we propose a four-dimensional evaluation framework (including Relevance, Correctness, Coverage, and Clarity) to assess the quality of LLM-generated novelty evaluations. Extensive experiments on both general and specialized LLMs under different prompting strategies reveal that current models exhibit limited understanding of scientific novelty, and that fine--tuned models often suffer from instruction-following deficiencies. These findings underscore the need for targeted fine-tuning strategies that jointly improve novelty comprehension and instruction adherence. 

---
# Triviality Corrected Endogenous Reward 

**Authors**: Xinda Wang, Zhengxu Hou, Yangshijie Zhang, Bingren Yan, Jialin Liu, Chenzhuo Zhao, Zhibo Yang, Bin-Bin Yang, Feng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.11522)  

**Abstract**: Reinforcement learning for open-ended text generation is constrained by the lack of verifiable rewards, necessitating reliance on judge models that require either annotated data or powerful closed-source models. Inspired by recent work on unsupervised reinforcement learning for mathematical reasoning using confidence-based endogenous rewards, we investigate whether this principle can be adapted to open-ended writing tasks. We find that directly applying confidence rewards leads to Triviality Bias: the policy collapses toward high-probability outputs, reducing diversity and meaningful content. We propose TCER (Triviality Corrected Endogenous Reward), which addresses this bias by rewarding the relative information gain between a specialist policy and a generalist reference policy, modulated by a probability-dependent correction mechanism. Across multiple writing benchmarks and model architectures, TCER achieves consistent improvements without external supervision. Furthermore, TCER also transfers effectively to mathematical reasoning, validating the generality of our approach across different generation tasks. 

---
# Policy Split: Incentivizing Dual-Mode Exploration in LLM Reinforcement with Dual-Mode Entropy Regularization 

**Authors**: Jiashu Yao, Heyan Huang, Chuwei Luo, Daiqing Wu, Zeming Liu, Yuhang Guo, Yangyang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11510)  

**Abstract**: To encourage diverse exploration in reinforcement learning (RL) for large language models (LLMs) without compromising accuracy, we propose Policy Split, a novel paradigm that bifurcates the policy into normal and high-entropy modes with a high-entropy prompt. While sharing model parameters, the two modes undergo collaborative dual-mode entropy regularization tailored to distinct objectives. Specifically, the normal mode optimizes for task correctness, while the high-entropy mode incorporates a preference for exploration, and the two modes learn collaboratively. Extensive experiments demonstrate that our approach consistently outperforms established entropy-guided RL baselines across various model sizes in general and creative tasks. Further analysis reveals that Policy Split facilitates dual-mode exploration, where the high-entropy mode generates distinct behavioral patterns to the normal mode, providing unique learning signals. 

---
# METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models 

**Authors**: Pengfeng Li, Chen Huang, Chaoqun Hao, Hongyao Chen, Xiao-Yong Wei, Wenqiang Lei, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.11502)  

**Abstract**: Contextual causal reasoning is a critical yet challenging capability for Large Language Models (LLMs). Existing benchmarks, however, often evaluate this skill in fragmented settings, failing to ensure context consistency or cover the full causal hierarchy. To address this, we pioneer METER to systematically benchmark LLMs across all three levels of the causal ladder under a unified context setting. Our extensive evaluation of various LLMs reveals a significant decline in proficiency as tasks ascend the causal hierarchy. To diagnose this degradation, we conduct a deep mechanistic analysis via both error pattern identification and internal information flow tracing. Our analysis reveals two primary failure modes: (1) LLMs are susceptible to distraction by causally irrelevant but factually correct information at lower level of causality; and (2) as tasks ascend the causal hierarchy, faithfulness to the provided context degrades, leading to a reduced performance. We belive our work advances our understanding of the mechanisms behind LLM contextual causal reasoning and establishes a critical foundation for future research. Our code and dataset are available at this https URL . 

---
# Think Before you Write: QA-Guided Reasoning for Character Descriptions in Books 

**Authors**: Argyrios Papoudakis, Mirella Lapata, Frank Keller  

**Link**: [PDF](https://arxiv.org/pdf/2604.11435)  

**Abstract**: Character description generation is an important capability for narrative-focused applications such as summarization, story analysis, and character-driven simulations. However, generating accurate character descriptions from long-form narratives (e.g., novels) is challenging: models must track evolving attributes (e.g., relationships and events), integrate evidence scattered across the text, and infer implicit details. Despite the success of reasoning-enabled LLMs on many benchmarks, we find that for character description generation their performance improves when built-in reasoning is disabled (i.e., an empty reasoning trace). Motivated by this, we propose a training framework that decouples reasoning from generation. Our approach, which can be applied on top of long-context LLMs or chunk-based methods, consists of a reasoning model that produces a structured QA reasoning trace and a generation model that conditions on this trace to produce the final character description. Experiments on two datasets (BookWorm and CroSS) show that QA-guided reasoning improves faithfulness, informativeness, and grounding over strong long-context baselines. 

---
# METRO: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues 

**Authors**: Haofu Yang, Jiaji Liu, Chen Huang, Faguo Wu, Wenqiang Lei, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.11427)  

**Abstract**: Developing non-collaborative dialogue agents traditionally requires the manual, unscalable codification of expert strategies. We propose \ours, a method that leverages large language models to autonomously induce both strategy actions and planning logic directly from raw transcripts. METRO formalizes expert knowledge into a Strategy Forest, a hierarchical structure that captures both short-term responses (nodes) and long-term strategic foresight (branches). Experimental results across two benchmarks show that METRO demonstrates promising performance, outperforming existing methods by an average of 9%-10%. Our further analysis not only reveals the success behind METRO (strategic behavioral diversity and foresight), but also demonstrates its robust cross-task transferability. This offers new insights into building non-collaborative agents in a cost-effective and scalable way. Our code is available at this https URL. 

---
# Bridging What the Model Thinks and How It Speaks: Self-Aware Speech Language Models for Expressive Speech Generation 

**Authors**: Kuang Wang, Lai Wei, Qibing Bai, Ping Lin, Wenkai Fang, Feng Jiang, Zhongjie Jiang, Jun Huang, Yannan Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11424)  

**Abstract**: Speech Language Models (SLMs) exhibit strong semantic understanding, yet their generated speech often sounds flat and fails to convey expressive intent, undermining user engagement. We term this mismatch the semantic understanding-acoustic realization gap. We attribute this gap to two key deficiencies: (1) intent transmission failure, where SLMs fail to provide the stable utterance-level intent needed for expressive delivery; and (2) realization-unaware training, where no feedback signal verifies whether acoustic outputs faithfully reflect intended expression. To address these issues, we propose SA-SLM (Self-Aware Speech Language Model), built on the principle that the model should be aware of what it thinks during generation and how it speaks during training. SA-SLM addresses this gap through two core contributions: (1) Intent-Aware Bridging, which uses a Variational Information Bottleneck (VIB) objective to translate the model's internal semantics into temporally smooth expressive intent, making speech generation aware of what the model intends to express; and (2) Realization-Aware Alignment, which repurposes the model as its own critic to verify and align acoustic realization with intended expressive intent via rubric-based feedback. Trained on only 800 hours of expressive speech data, our 3B parameter SA-SLM surpasses all open-source baselines and comes within 0.08 points of GPT-4o-Audio in overall expressiveness on the EchoMind benchmark. 

---
# Retrieval as Generation: A Unified Framework with Self-Triggered Information Planning 

**Authors**: Bo Li, Mingda Wang, Gexiang Fang, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.11407)  

**Abstract**: We revisit retrieval-augmented generation (RAG) by embedding retrieval control directly into generation. Instead of treating retrieval as an external intervention, we express retrieval decisions within token-level decoding, enabling end-to-end coordination without additional controllers or classifiers. Under the paradigm of Retrieval as Generation, we propose \textbf{GRIP} (\textbf{G}eneration-guided \textbf{R}etrieval with \textbf{I}nformation \textbf{P}lanning), a unified framework in which the model regulates retrieval behavior through control-token emission. Central to GRIP is \textit{Self-Triggered Information Planning}, which allows the model to decide when to retrieve, how to reformulate queries, and when to terminate, all within a single autoregressive trajectory. This design tightly couples retrieval and reasoning and supports dynamic multi-step inference with on-the-fly evidence integration. To supervise these behaviors, we construct a structured training set covering answerable, partially answerable, and multi-hop queries, each aligned with specific token patterns. Experiments on five QA benchmarks show that GRIP surpasses strong RAG baselines and is competitive with GPT-4o while using substantially fewer parameters. 

---
# Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations 

**Authors**: Yilong Liu, Xixun Lin, Pengfei Cao, Ge Zhang, Fang Fang, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.11322)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in utilizing external tools. In practice, however, LLMs are often exposed to tools that are irrelevant to the user's query, in which case the desired behavior is to refrain from invocations. In this work, we identify a widespread yet overlooked mechanistic flaw in tool refusal, which we term structural alignment bias: Even when a tool fails to serve the user's goal, LLMs still tend to invoke it whenever query attributes can be validly assigned to tool parameters. To systematically study this bias, we introduce SABEval, a new dataset that decouples structural alignment from semantic relevance. Our analysis shows that structural alignment bias induces severe tool-invocation errors in LLMs, yet remains largely unaccounted for in existing evaluations. To investigate the internal mechanisms underlying this bias, we propose Contrastive Attention Attribution, which reveals two competing pathways for semantic checking and structural matching. The relative strength of these pathways drives LLMs' tool invocation decisions. Based on these findings, we further introduce a rebalancing strategy that effectively mitigates structural alignment bias, as demonstrated by extensive experiments, without degrading general tool-use capabilities. 

---
# Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning 

**Authors**: Rui Song, Lida Shi, Ruihua Qi, Yingji Li, Hao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11299)  

**Abstract**: In recent years, rapid advances in Multimodal Large Language Models (MLLMs) have increasingly stimulated research on ancient Chinese scripts. As the evolution of written characters constitutes a fundamental pathway for understanding cultural transformation and historical continuity, how MLLMs can be systematically leveraged to support and advance text evolution analysis remains an open and largely underexplored problem. To bridge this gap, we construct a comprehensive benchmark comprising 11 tasks and over 130,000 instances, specifically designed to evaluate the capability of MLLMs in analyzing the evolution of ancient Chinese scripts. We conduct extensive evaluations across multiple widely used MLLMs and observe that, while existing models demonstrate a limited ability in glyph-level comparison, their performance on core tasks-such as character recognition and evolutionary reasoning-remains substantially constrained. Motivated by these findings, we propose a glyph-driven fine-tuning framework (GEVO) that explicitly encourages models to capture evolutionary consistency in glyph transformations and enhances their understanding of text evolution. Experimental results show that even models at the 2B scale achieve consistent and comprehensive performance improvements across all evaluated tasks. To facilitate future research, we publicly release both the benchmark and the trained models\footnote{this https URL}. 

---
# Polyglot Teachers: Evaluating Language Models for Multilingual Synthetic Data Generation 

**Authors**: Lester James V. Miranda, Ivan Vulić, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11290)  

**Abstract**: Synthesizing supervised finetuning (SFT) data from language models (LMs) to teach smaller models multilingual tasks has become increasingly common. However, teacher model selection is often ad hoc, typically defaulting to the largest available option, even though such models may have significant capability gaps in non-English languages. This practice can result in poor-quality synthetic data and suboptimal student downstream performance. In this work, we systematically characterize what makes an effective multilingual teacher. We measure intrinsic measures of data quality with extrinsic student model performance in a metric we call Polyglot Score; evaluating 10 LMs across 6 typologically diverse languages, generating over 1.4M SFT examples and training 240 student models. Among the models tested, Gemma 3 27B and Aya Expanse 32B emerge as consistently effective teachers across different student base model families. Further analyses reveal that model scale alone does not significantly predict teacher effectiveness; instead, data qualities such as prompt diversity, length, and response fluency capture over 93.3% of variance in intrinsic data quality and predict student performance. Finally, we provide practical recommendations, including matching the model families of teacher-student pairs and translating from or responding to existing prompts, which can yield improvements for less-resourced languages. We hope that our work advances data-centric research in multilingual synthetic data and LM development. 

---
# Transactional Attention: Semantic Sponsorship for KV-Cache Retention 

**Authors**: Abhinaba Basu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11288)  

**Abstract**: At K=16 tokens (0.4% of a 4K context), every existing KV-cache compression method achieves 0% on credential retrieval. The failure mode is dormant tokens: credentials, API keys, and configuration values that receive near-zero attention but become essential at generation time. Because these tokens lack the statistical signals that eviction policies rely on, no method based on attention scores, reconstruction loss, or learned retention gates retains them. We introduce Transactional Attention (TA), a sponsorship mechanism in which structural anchor patterns (e.g., "key:", "password:") protect adjacent value-bearing tokens from eviction. TA achieves 100% credential retrieval at K=16 where six baselines (H2O, TOVA, SnapKV, StreamingLLM, PyramidKV, DynamicKV) achieve 0%, and sustains 100% accuracy across 200 function-calling trials. TA-Fast, an attention-free variant, reduces memory overhead by 52% and is compatible with SDPA and FlashAttention. TA is orthogonal to existing compression methods and adds less than 1% latency overhead. 

---
# Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate 

**Authors**: Zhixiang Lu, Jionglong Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.11258)  

**Abstract**: Multimodal Large Language Models (MLLMs) in healthcare suffer from severe confirmation bias, often hallucinating visual details to support initial, potentially erroneous diagnostic hypotheses. Existing Chain-of-Thought (CoT) approaches lack intrinsic correction mechanisms, rendering them vulnerable to error propagation. To bridge this gap, we propose Dialectic-Med, a multi-agent framework that enforces diagnostic rigor through adversarial dialectics. Unlike static consensus models, Dialectic-Med orchestrates a dynamic interplay between three role-specialized agents: a proponent that formulates diagnostic hypotheses; an opponent equipped with a novel visual falsification module that actively retrieves contradictory visual evidence to challenge the Proponent; and a mediator that resolves conflicts via a weighted consensus graph. By explicitly modeling the cognitive process of falsification, our framework guarantees that diagnostic reasoning is tightly grounded in verified visual regions. Empirical evaluations on MIMIC-CXR-VQA, VQA-RAD, and PathVQA demonstrate that Dialectic-Med not only achieves state-of-the-art performance but also fundamentally enhances the trustworthiness of the reasoning process. Beyond accuracy, our approach significantly enhances explanation faithfulness and decisively mitigates hallucinations, establishing a new standard over single-agent baselines. 

---
# Judge Like Human Examiners: A Weighted Importance Multi-Point Evaluation Framework for Generative Tasks with Long-form Answers 

**Authors**: Guoxin Yu, Chulun Zhou, Lemao Liu, Qi Wang, Mo Yu, Jialong Tang, Baosong Yang, Xiang Ao, Wao Lam, Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11246)  

**Abstract**: Evaluating the quality of model responses remains challenging in generative tasks with long-form answers, as the expected answers usually contain multiple semantically distinct yet complementary factors that should be factorized for fine-grained assessment. Recent evaluation methods resort to relying on either task-level rubrics or question-aware checklists. However, they still 1) struggle to assess whether a response is genuinely grounded in provided contexts; 2) fail to capture the heterogeneous importance of different aspects of reference answers. Inspired by human examiners, we propose a Weighted Importance Multi-Point Evaluation (WIMPE) framework, which factorizes each reference answer into weighted context-bound scoring points. Two complementary metrics, namely Weighted Point-wise Alignment (WPA) and Point-wise Conflict Penalty (PCP), are designed to measure the alignment and contradiction between model responses and reference answers. Extensive experiments on 10 generative tasks demonstrate that WIMPE achieves higher correlations with human annotations. 

---
# RUMLEM: A Dictionary-Based Lemmatizer for Romansh 

**Authors**: Dominic P. Fischer, Zachary Hopton, Jannis Vamvas  

**Link**: [PDF](https://arxiv.org/pdf/2604.11233)  

**Abstract**: Lemmatization -- the task of mapping an inflected word form to its dictionary form -- is a crucial component of many NLP applications. In this paper, we present RUMLEM, a lemmatizer that covers the five main varieties of Romansh as well as the supra-regional standard variety Rumantsch Grischun. It is based on comprehensive, community-driven morphological databases for Romansh, enabling RUMLEM to cover 77-84% of the words in a typical Romansh text. Since there is a dedicated database for each Romansh variety, an additional application of RUMLEM is variety-aware language classification. Evaluation on 30'000 Romansh texts of varying lengths shows that RUMLEM correctly identifies the variety in 95% of cases. In addition, a proof of concept demonstrates the feasibility of Romansh vs. non-Romansh language classification based on the lemmatizer. 

---
# HiEdit: Lifelong Model Editing with Hierarchical Reinforcement Learning 

**Authors**: Yangfan Wang, Tianyang Sun, Chen Tang, Jie Liu, Wei Cai, Jingchi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11214)  

**Abstract**: Lifelong model editing (LME) aims to sequentially rectify outdated or inaccurate knowledge in deployed LLMs while minimizing side effects on unrelated inputs. However, existing approaches typically apply parameter perturbations to a static and dense set of LLM layers for all editing instances. This practice is counter-intuitive, as we hypothesize that different pieces of knowledge are stored in distinct layers of the model. Neglecting this layer-wise specificity can impede adaptability in integrating new knowledge and result in catastrophic forgetting for both general and previously edited knowledge. To address this, we propose HiEdit, a hierarchical reinforcement learning framework that adaptively identifies the most knowledge-relevant layers for each editing instance. By enabling dynamic, instance-aware layer selection and incorporating an intrinsic reward for sparsity, HiEdit achieves precise, localized updates. Experiments on various LLMs show that HiEdit boosts the performance of the competitive RLEdit by an average of 8.48% with perturbing only half of the layers per edit. Our code is available at: this https URL. 

---
# Exploring Knowledge Conflicts for Faithful LLM Reasoning: Benchmark and Method 

**Authors**: Tianzhe Zhao, Jiaoyan Chen, Shuxiu Zhang, Haiping Zhu, Qika Lin, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11209)  

**Abstract**: Large language models (LLMs) have achieved remarkable success across a wide range of applications especially when augmented by external knowledge through retrieval-augmented generation (RAG). Despite their widespread adoption, recent studies have shown that LLMs often struggle to perform faithful reasoning when conflicting knowledge is retrieved. However, existing work primarily focuses on conflicts between external knowledge and the parametric knowledge of LLMs, leaving conflicts across external knowledge largely unexplored. Meanwhile, modern RAG systems increasingly emphasize the integration of unstructured text and (semi-)structured data like knowledge graphs (KGs) to improve knowledge completeness and reasoning faithfulness. To address this gap, we introduce ConflictQA, a novel benchmark that systematically instantiates conflicts between textual evidence and KG evidence. Extensive evaluations across representative LLMs reveal that, facing such cross-source conflicts, LLMs often fail to identify reliable evidence for correct reasoning. Instead, LLMs become more sensitive to prompting choices and tend to rely exclusively on either KG or textual evidence, resulting in incorrect responses. Based on these findings, we further propose XoT, a two-stage explanation-based thinking framework tailored for reasoning over heterogeneous conflicting evidence, and verify its effectiveness with extensive experiments. 

---
# CocoaBench: Evaluating Unified Digital Agents in the Wild 

**Authors**: CocoaBench Team, Shibo Hao, Zhining Zhang, Zhiqi Liang, Tianyang Liu, Yuheng Zha, Qiyue Gao, Jixuan Chen, Zilong Wang, Zhoujun Cheng, Haoxiang Zhang, Junli Wang, Hexi Jin, Boyuan Zheng, Kun Zhou, Yu Wang, Feng Yao, Licheng Liu, Yijiang Li, Zhifei Li, Zhengtao Han, Pracha Promthaw, Tommaso Cerruti, Xiaohan Fu, Ziqiao Ma, Jingbo Shang, Lianhui Qin, Julian McAuley, Eric P. Xing, Zhengzhong Liu, Rupesh Kumar Srivastava, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11201)  

**Abstract**: LLM agents now perform strongly in software engineering, deep research, GUI automation, and various other applications, while recent agent scaffolds and models are increasingly integrating these capabilities into unified systems. Yet, most evaluations still test these capabilities in isolation, which leaves a gap for more diverse use cases that require agents to combine different capabilities. We introduce CocoaBench, a benchmark for unified digital agents built from human-designed, long-horizon tasks that require flexible composition of vision, search, and coding. Tasks are specified only by an instruction and an automatic evaluation function over the final output, enabling reliable and scalable evaluation across diverse agent infrastructures. We also present CocoaAgent, a lightweight shared scaffold for controlled comparison across model backbones. Experiments show that current agents remain far from reliable on CocoaBench, with the best evaluated system achieving only 45.1% success rate. Our analysis further points to substantial room for improvement in reasoning and planning, tool use and execution, and visual grounding. 

---
# TRACE: An Experiential Framework for Coherent Multi-hop Knowledge Graph Question Answering 

**Authors**: Yingxu Wang, Jiaxin Huang, Mengzhu Wang, Nan Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.11193)  

**Abstract**: Multi-hop Knowledge Graph Question Answering (KGQA) requires coherent reasoning across relational paths, yet existing methods often treat each reasoning step independently and fail to effectively leverage experience from prior explorations, leading to fragmented reasoning and redundant exploration. To address these challenges, we propose Trajectoryaware Reasoning with Adaptive Context and Exploration priors (TRACE), an experiential framework that unifies LLM-driven contextual reasoning with exploration prior integration to enhance the coherence and robustness of multihop KGQA. Specifically, TRACE dynamically translates evolving reasoning paths into natural language narratives to maintain semantic continuity, while abstracting prior exploration trajectories into reusable experiential priors that capture recurring exploration patterns. A dualfeedback re-ranking mechanism further integrates contextual narratives with exploration priors to guide relation selection during reasoning. Extensive experiments on multiple KGQA benchmarks demonstrate that TRACE consistently outperforms state-of-the-art baselines. 

---
# MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis 

**Authors**: Zixiong Yu, Jun Rao, Guhan Chen, Songtao Tian, Bohan Li, Jiansheng Wei, Min Zhang, Xiaojun Meng  

**Link**: [PDF](https://arxiv.org/pdf/2604.11188)  

**Abstract**: Synthesizing high-quality mathematical reasoning data without human priors remains a significant challenge. Current approaches typically rely on seed data mutation or simple prompt engineering, often suffering from mode collapse and limited logical complexity. This paper proposes a hierarchical synthesis framework that formulates data synthesis as an unsupervised optimization problem over a constraint graph followed by semantic instantiation, rather than treating it as a direct text generation task. We introduce a Legislator-Executor paradigm: The Legislator adversarially evolves structured generation blueprints encoding the constraints of the problem, while the Executor instantiates these specifications into diverse natural language scenarios. This decoupling of skeleton design from linguistic realization enables a prioritized focus on constructing complex and diverse logical structures, thereby guiding high-quality data synthesis. Experiments conducted on a total of 10 models across the Qwen, Llama, Mistral, and Gemma series demonstrate that our method achieves notable results: models fine-tuned on 1K synthesized samples outperform widely-used datasets of comparable scale (LIMO, s1K) across eight mathematical benchmarks, exhibiting superior out-of-distribution generalization. 

---
# Evaluating Memory Capability in Continuous Lifelog Scenario 

**Authors**: Jianjie Zheng, Zhichen Liu, Zhanyu Shen, Jingxiang Qu, Guanhua Chen, Yile Wang, Yang Xu, Yang Liu, Sijie Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.11182)  

**Abstract**: Nowadays, wearable devices can continuously lifelog ambient conversations, creating substantial opportunities for memory systems. However, existing benchmarks primarily focus on online one-on-one chatting or human-AI interactions, thus neglecting the unique demands of real-world scenarios. Given the scarcity of public lifelogging audio datasets, we propose a hierarchical synthesis framework to curate \textbf{\textsc{LifeDialBench}}, a novel benchmark comprising two complementary subsets: \textbf{EgoMem}, built on real-world egocentric videos, and \textbf{LifeMem}, constructed using simulated virtual community. Crucially, to address the issue of temporal leakage in traditional offline settings, we propose an \textbf{Online Evaluation} protocol that strictly adheres to temporal causality, ensuring systems are evaluated in a realistic streaming fashion. Our experimental results reveal a counterintuitive finding: current sophisticated memory systems fail to outperform a simple RAG-based baseline. This highlights the detrimental impact of over-designed structures and lossy compression in current approaches, emphasizing the necessity of high-fidelity context preservation for lifelog scenarios. We release our code and data at this https URL. 

---
# SHARE: Social-Humanities AI for Research and Education 

**Authors**: João Gonçalves, Sonia de Jager, Petr Knoth, David Pride, Nick Jelicic  

**Link**: [PDF](https://arxiv.org/pdf/2604.11152)  

**Abstract**: This intermediate technical report introduces the SHARE family of base models and the MIRROR user interface. The SHARE models are the first causal language models fully pretrained by and for the social sciences and humanities (SSH). Their performance in modelling SSH texts is close to that of general purpose models (Phi-4) which use 100 times more tokens, as shown by our custom SSH Cloze benchmark. The MIRROR user interface is designed for reviewing text inputs from the SSH disciplines while preserving critical engagement. By prototyping a generative AI interface that does not generate any text, we propose a way to harness the capabilities of the SHARE models without compromising the integrity of SSH principles and norms. 

---
# How Robust Are Large Language Models for Clinical Numeracy? An Empirical Study on Numerical Reasoning Abilities in Clinical Contexts 

**Authors**: Minh-Vuong Nguyen, Fatemeh Shiri, Zhuang Li, Karin Verspoor  

**Link**: [PDF](https://arxiv.org/pdf/2604.11133)  

**Abstract**: Large Language Models (LLMs) are increasingly being explored for clinical question answering and decision support, yet safe deployment critically requires reliable handling of patient measurements in heterogeneous clinical notes. Existing evaluations of LLMs for clinical numerical reasoning provide limited operation-level coverage, restricted primarily to arithmetic computation, and rarely assess the robustness of numerical understanding across clinical note formats. We introduce ClinicNumRobBench, a benchmark of 1,624 context-question instances with ground-truth answers that evaluates four main types of clinical numeracy: value retrieval, arithmetic computation, relational comparison, and aggregation. To stress-test robustness, ClinicNumRobBench presents longitudinal MIMIC-IV vital-sign records in three semantically equivalent representations, including a real-world note-style variant derived from the Open Patients dataset, and instantiates queries using 42 question templates. Experiments on 14 LLMs show that value retrieval is generally strong, with most models exceeding 85% accuracy, while relational comparison and aggregation remain challenging, with some models scoring below 15%. Fine-tuning on medical data can reduce numeracy relative to base models by over 30%, and performance drops under note-style variation indicate LLM sensitivity to format. ClinicNumRobBench offers a rigorous testbed for clinically reliable numerical reasoning. Code and data URL are available on this https URL. 

---
# DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning 

**Authors**: Feiyang Li, Yile Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11129)  

**Abstract**: Task vectors, representing directions in model or activation spaces that encode task-specific behaviors, have emerged as a promising tool for steering large language models (LLMs). However, existing approaches typically require fine-tuning or invasive manipulation of internal states, limiting their flexibility and scalability. We propose \textsc{DeCoVec} (Decoding Space based Task Vector), a training-free and non-invasive framework that constructs task vectors directly in the \textit{decoding space} by leveraging in-context learning (ICL). Specifically, \textsc{DeCoVec} captures the task essence as the difference between the output logit distributions of few-shot and zero-shot prompts, then steers generation by injecting this vector into the decoding process. Experiments across seven LLMs (0.5B--9B) on TruthfulQA, Math-500, and AQUA-RAT show that \textsc{DeCoVec} consistently outperforms standard few-shot baselines, with gains up to +5.50 average accuracy. Further analysis demonstrates that \textsc{DeCoVec} effectively suppresses generation degeneration and logical flaws while exhibiting strong robustness to demonstration ordering, all without incurring additional input token costs. Our method offers a training-free and non-invasive solution for LLM steering without requiring weight updates or auxiliary models. 

---
# BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection 

**Authors**: Atharva Gupta, Dhruv Kumar, Yash Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2604.11121)  

**Abstract**: The POLAR SemEval-2026 Shared Task aims to detect online polarization and focuses on the classification and identification of multilingual, multicultural, and multi-event polarization.
Accurate computational detection of online polarization is challenging due to nuanced rhetoric, implicit framing, and the high cost of human-in-the-loop annotation. Building on recent findings that contextual prompting enables large language models to function as strong polarization detectors, we present a two-stage approach for detecting political polarization in social media text that combines structured supervised fine-tuning with Direct Preference Optimization (DPO) refinement.
We fine-tune Qwen 2.5-7B-Instruct with LoRA using an interpretable slot-filling template (target, claim type, manifestation checklist, and justification). We then apply DPO with automatically generated preference pairs to reduce costly false negatives. Experiments on the SemEval 2026 POLAR shared task dataset show that preference-based refinement improves both accuracy and decreases false negatives without extra annotation. On the English development set, DPO increases recall from 0.5085 to 0.7797 and improves macro-F1 by ~5 points. 

---
# Efficient Training for Cross-lingual Speech Language Models 

**Authors**: Yan Zhou, Qingkai Fang, Yun Hong, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.11096)  

**Abstract**: Currently, large language models (LLMs) predominantly focus on the text modality. To enable more natural human-AI interaction, speech LLMs are emerging, but building effective end-to-end speech LLMs remains challenging due to limited data and the difficulty in expanding to more languages. In this paper, we introduce Cross-lingual Speech Language Model (CSLM), an efficient training method for cross-lingual speech LLMs based on discrete speech tokens. We propose a novel alignment strategy that achieves cross-modal and cross-lingual alignment through continual pre-training. By conducting instruction fine-tuning following a speech-text interleaved chain-of-modality generation process, we enhance modal alignment at a finer granularity, thereby improving generation quality and reducing latency. CSLM aligns different modalities and languages simultaneously without the need for massive speech data, thus exhibiting good language scalability. Evaluations on cross-modal tasks, mono-lingual conversational tasks, and cross-lingual conversational tasks demonstrate CSLM's strong cross-modal alignment capabilities and general task abilities. (Code is available at: this https URL) 

---
# ks-pret-5m: a 5 million word, 12 million token kashmiri pretraining dataset 

**Authors**: Haq Nawaz Malik, Nahfid Nissar  

**Link**: [PDF](https://arxiv.org/pdf/2604.11066)  

**Abstract**: We present KS-PRET-5M, the largest publicly available pretraining dataset for the Kashmiri language, comprising 5,090,244 (5.09M) words, 27,692,959 (27.6M) characters, and a vocabulary of 295,433 (295.4K) unique word types. We assembled the dataset from two source classes: digitized archival and literary material, encompassing literature, news, biographies, novels, poetry, religious scholarship, and academic writing, recovered from the proprietary InPage desktop-publishing format using the converter of Malik~\cite{malik2024inpage}, and Unicode-native text collected from Kashmiri-language web sources. All text was processed through an eleven-stage cleaning pipeline that achieves a mean Kashmiri script ratio of 0.9965, reducing Devanagari contamination to 146 characters across the full dataset. We tokenized the dataset empirically using google/muril-base-cased, yielding a subword ratio of 2.383 tokens per word and a total of approximately 12.13 million subword tokens, substantially higher than prior estimates derived from non-Kashmiri Perso-Arabic analogues. KS-PRET-5M is released as a single continuous text stream under CC~BY~4.0 to support language model pretraining, tokenizer training, and computational linguistic research for Kashmiri. 

---
# Shared Emotion Geometry Across Small Language Models: A Cross-Architecture Study of Representation, Behavior, and Methodological Confounds 

**Authors**: Jihoon Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2604.11050)  

**Abstract**: We extract 21-emotion vector sets from twelve small language models (six architectures x base/instruct, 1B-8B parameters) under a unified comprehension-mode pipeline at fp16 precision, and compare the resulting geometries via representational similarity analysis on raw cosine RDMs. The five mature architectures (Qwen 2.5 1.5B, SmolLM2 1.7B, Llama 3.2 3B, Mistral 7B v0.3, Llama 3.1 8B) share nearly identical 21-emotion geometry, with pairwise RDM Spearman correlations of 0.74-0.92. This universality persists across diametrically opposed behavioral profiles: Qwen 2.5 and Llama 3.2 occupy opposite poles of MTI Compliance facets yet produce nearly identical emotion RDMs (rho = 0.81), so behavioral facet differences arise above the shared emotion representation. Gemma-3 1B base, the one immature case in our dataset, exhibits extreme residual-stream anisotropy (0.997) and is restructured by RLHF across all geometric descriptors, whereas the five already-mature families show within-family base x instruct RDM correlations of rho >= 0.92 (Mistral 7B v0.3 at rho = 0.985), suggesting RLHF restructures only representations that are not yet organized. Methodologically, we show that what prior work has read as a single comprehension-vs-generation method effect in fact decomposes into four distinct layers -- a coarse method-dependent dissociation, robust sub-parameter sensitivity within generation, a true precision (fp16 vs INT8) effect, and a conflated cross-experiment bias that distorts in opposite directions for different models -- so that a single rho between two prior emotion-vector studies is not a safe basis for interpretation without the layered decomposition. 

---
# A Systematic Analysis of the Impact of Persona Steering on LLM Capabilities 

**Authors**: Jiaqi Chen, Ming Wang, Tingna Xie, Shi Feng, Yongkang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11048)  

**Abstract**: Imbuing Large Language Models (LLMs) with specific personas is prevalent for tailoring interaction styles, yet the impact on underlying cognitive capabilities remains unexplored. We employ the Neuron-based Personality Trait Induction (NPTI) framework to induce Big Five personality traits in LLMs and evaluate performance across six cognitive benchmarks. Our findings reveal that persona induction produces stable, reproducible shifts in cognitive task performance beyond surface-level stylistic changes. These effects exhibit strong task dependence: certain personalities yield consistent gains on instruction-following, while others impair complex reasoning. Effect magnitude varies systematically by trait dimension, with Openness and Extraversion exerting the most robust influence. Furthermore, LLM effects show 73.68% directional consistency with human personality-cognition relationships. Capitalizing on these regularities, we propose Dynamic Persona Routing (DPR), a lightweight query-adaptive strategy that outperforms the best static persona without additional training. 

---
# Uncertainty-Aware Web-Conditioned Scientific Fact-Checking 

**Authors**: Ashwin Vinod, Katrin Erk  

**Link**: [PDF](https://arxiv.org/pdf/2604.11036)  

**Abstract**: Scientific fact-checking is vital for assessing claims in specialized domains such as biomedicine and materials science, yet existing systems often hallucinate or apply inconsistent reasoning, especially when verifying technical, compositional claims against an evidence snippet under source and cost/latency constraints. We present a pipeline centered on atomic predicate-argument decomposition and calibrated, uncertainty-gated corroboration: atomic facts are aligned to local snippets via embeddings, verified by a compact evidence-grounded checker, and only facts with uncertain support trigger domain-restricted web search over authoritative sources. The system supports both binary and tri-valued classification where it predicts labels from Supported, Refuted, NEI for three-way tasks. We evaluate under two regimes, Context-Only (no web) and Context+Web (uncertainty-gated web corroboration); when retrieved evidence conflicts with the provided context, we abstain with NEI rather than overriding the context. On multiple benchmarks, our framework surpasses the strongest benchmarks. In our experiments, web corroboration was invoked for only a minority of atomic facts on average, indicating that external evidence is consulted selectively under calibrated uncertainty rather than routinely. Overall, coupling atomic granularity with calibrated, uncertainty-gated corroboration yields more interpretable and context-conditioned verification, making the approach well-suited to high-stakes, single-document settings that demand traceable rationales, predictable cost/latency, and conservative. 

---
# When Valid Signals Fail: Regime Boundaries Between LLM Features and RL Trading Policies 

**Authors**: Zhengzhe Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10996)  

**Abstract**: Can large language models (LLMs) generate continuous numerical features that improve reinforcement learning (RL) trading agents? We build a modular pipeline where a frozen LLM serves as a stateless feature extractor, transforming unstructured daily news and filings into a fixed-dimensional vector consumed by a downstream PPO agent. We introduce an automated prompt-optimization loop that treats the extraction prompt as a discrete hyperparameter and tunes it directly against the Information Coefficient - the Spearman rank correlation between predicted and realized returns - rather than NLP losses. The optimized prompt discovers genuinely predictive features (IC above 0.15 on held-out data). However, these valid intermediate representations do not automatically translate into downstream task performance: during a distribution shift caused by a macroeconomic shock, LLM-derived features add noise, and the augmented agent under-performs a price-only baseline. In a calmer test regime the agent recovers, yet macroeconomic state variables remain the most robust driver of policy improvement. Our findings highlight a gap between feature-level validity and policy-level robustness that parallels known challenges in transfer learning under distribution shift. 

---
# When Verification Fails: How Compositionally Infeasible Claims Escape Rejection 

**Authors**: Muxin Liu, Delip Rao, Grace Kim, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2604.10990)  

**Abstract**: Scientific claim verification, the task of determining whether claims are entailed by scientific evidence, is fundamental to establishing discoveries in evidence while preventing misinformation. This process involves evaluating each asserted constraint against validated evidence. Under the Closed-World Assumption (CWA), a claim is accepted if and only if all asserted constraints are positively supported. We show that existing verification benchmarks cannot distinguish models enforcing this standard from models applying a simpler shortcut called salient-constraint checking, which applies CWA's rejection criterion only to the most salient constraint and accepts when that constraint is supported. Because existing benchmarks construct infeasible claims by perturbing a single salient element they are insufficient at distinguishing between rigorous claim verification and simple salient-constraint reliance. To separate the two, we construct compositionally infeasible claims where the salient constraint is supported but a non-salient constraint is contradicted. Across model families and modalities, models that otherwise saturate existing benchmarks consistently over-accept these claims, confirming the prevalence of such shortcut reasoning. Via model context interventions, we show that different models and prompting strategies occupy distinct positions on a shared ROC curve, indicating that the gap between model families reflects differences in verification threshold rather than underlying reasoning ability, and that the compositional inference bottleneck is a structural property of current verification behavior that strategy guidance alone cannot overcome. 

---
# YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents 

**Authors**: Victor De Lima, Grace Hui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10968)  

**Abstract**: Most conversational agents (CAs) are designed to satisfy user needs through user-driven interactions. However, many real-world settings, such as academic interviewing, judicial proceedings, and journalistic investigations, involve broader institutional decision-making processes and require agents that can elicit information from users. In this paper, we introduce Information Elicitation Agents (IEAs) in which the agent's goal is to elicit information from users to support the agent's institutional or task-oriented objectives. To enable systematic research on this setting, we present YIELD, a 26M-token dataset of 2,281 ethically sourced, human-to-human dialogues. Moreover, we formalize information elicitation as a finite-horizon POMDP and propose novel metrics tailored to IEAs. Pilot experiments on multiple foundation LLMs show that training on YIELD improves their alignment with real elicitation behavior and findings are corroborated by human evaluation. We release YIELD under CC BY 4.0. The dataset, project code, evaluation tools, and fine-tuned model adapters are available at: this https URL. 

---
# Mem$^2$Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation 

**Authors**: Zihao Cheng, Zeming Liu, Yingyu Shan, Xinyi Wang, Xiangrong Zhu, Yunpu Ma, Hongru Wang, Yuhang Guo, Wei Lin, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10923)  

**Abstract**: While large language model--powered agents can self-evolve by accumulating experience or by dynamically creating new assets (i.e., tools or expert agents), existing frameworks typically treat these two evolutionary processes in isolation. This separation overlooks their intrinsic interdependence: the former is inherently bounded by a manually predefined static toolset, while the latter generates new assets from scratch without experiential guidance, leading to limited capability growth and unstable evolution. To address this limitation, we introduce a novel paradigm of co-evolutionary Capability Expansion and Experience Distillation. Guided by this paradigm, we propose the \textbf{Mem$^{\textbf{2}}$Evolve}, which integrates two core components: \textbf{Experience Memory} and \textbf{Asset Memory}. Specifically, Mem$^{2}$Evolve leverages accumulated experience to guide the dynamic creation of assets, thereby expanding the agent's capability space while simultaneously acquiring new experience to achieve co-evolution. Extensive experiments across 6 task categories and 8 benchmarks demonstrate that Mem$^{2}$Evolve achieves improvement of 18.53\% over standard LLMs, 11.80\% over agents evolving solely through experience, and 6.46\% over those evolving solely through asset creation, establishing it as a substantially more effective and stable self-evolving agent framework. Code is available at: this https URL. 

---
# HTAA: Enhancing LLM Planning via Hybrid Toolset Agentization & Adaptation 

**Authors**: Chengrui Huang, Junshuo Zhang, Zhiyuan Ma, Xikun Wang, Ximeng Wang, Menghua Jiang, Gang Zeng, Zhaobing Han, Shen Gao, Shuo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10917)  

**Abstract**: Enabling large language models to scale and reliably use hundreds of tools is critical for real-world applications, yet challenging due to the inefficiency and error accumulation inherent in flat tool-calling architectures. To address this, we propose Hybrid Toolset Agentization & Adaptation (HTAA), a hierarchical framework for scalable tool-use planning. We propose a novel toolset agentization paradigm, which encapsulates frequently co-used tools into specialized agent tools, thereby reducing the planner's action space and mitigating redundancy. To ensure effective coordination, we design Asymmetric Planner Adaptation, a trajectory-based training paradigm that aligns the high-level planner with agent tools via backward reconstruction and forward refinement. To validate the performance of HTAA, we conduct experiments on a real-world internal dataset, InfoVerify, based on the POI validation workflow of China's largest online large-scale ride-hailing platform, featuring long-horizon executable tool trajectories. Experiments on InfoVerify and widely-used benchmarks show that HTAA consistently achieves higher task success rates, requires short tool calling trajectories, and significantly reduces context overhead compared to strong baselines. Furthermore, in a production deployment, HTAA substantially reduces manual validation effort and operational cost, demonstrating its practical efficacy. 

---
# AOP-Smart: A RAG-Enhanced Large Language Model Framework for Adverse Outcome Pathway Analysis 

**Authors**: Qinjiang Niu, Lu Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.10874)  

**Abstract**: Adverse Outcome Pathways (AOPs) are an important knowledge framework in toxicological research and risk assessment. In recent years, large language models (LLMs) have gradually been applied to AOP-related question answering and mechanistic reasoning tasks. However, due to the existence of the hallucination problem, that is, the model may generate content that is inconsistent with facts or lacks evidence, their reliability is still limited. To address this issue, this study proposes an AOP-oriented Retrieval-Augmented Generation (RAG) framework, AOP-Smart. Based on the official XML data from AOP-Wiki, this method uses Key Events (KEs), Key Event Relationships (KERs), and specific AOP information to retrieve relevant knowledge for user questions, thereby improving the reliability of the generated results of large language models. To evaluate the effectiveness of the proposed method, this study constructed a test set containing 20 AOP-related question answering tasks, covering KE identification, upstream and downstream KE retrieval, and complex AOP retrieval tasks. Experiments were conducted on three mainstream large language models, Gemini, DeepSeek, and ChatGPT, and comparative tests were performed under two settings: without RAG and with RAG. The experimental results show that, without using RAG, the accuracies of GPT, DeepSeek, and Gemini were 15.0\%, 35.0\%, and 20.0\%, respectively; after using RAG, their accuracies increased to 95.0\%, 100.0\%, and 95.0\%, respectively. The results indicate that AOP-Smart can significantly alleviate the hallucination problem of large language models in AOP knowledge tasks, and greatly improve the accuracy and consistency of their answers. 

---
# OccuBench: Evaluating AI Agents on Real-World Professional Tasks via Language World Models 

**Authors**: Xiaomeng Hu, Yinger Zhang, Fei Huang, Jianhong Tu, Yang Su, Lianghao Deng, Yuxuan Liu, Yantao Liu, Dayiheng Liu, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2604.10866)  

**Abstract**: AI agents are expected to perform professional work across hundreds of occupational domains (from emergency department triage to nuclear reactor safety monitoring to customs import processing), yet existing benchmarks can only evaluate agents in the few domains where public environments exist. We introduce OccuBench, a benchmark covering 100 real-world professional task scenarios across 10 industry categories and 65 specialized domains, enabled by Language World Models (LWMs) that simulate domain-specific environments through LLM-driven tool response generation. Our multi-agent synthesis pipeline automatically produces evaluation instances with guaranteed solvability, calibrated difficulty, and document-grounded diversity. OccuBench evaluates agents along two complementary dimensions: task completion across professional domains and environmental robustness under controlled fault injection (explicit errors, implicit data degradation, and mixed faults). We evaluate 15 frontier models across 8 model families and find that: (1) no single model dominates all industries, as each has a distinct occupational capability profile; (2) implicit faults (truncated data, missing fields) are harder than both explicit errors (timeouts, 500s) and mixed faults, because they lack overt error signals and require the agent to independently detect data degradation; (3) larger models, newer generations, and higher reasoning effort consistently improve performance. GPT-5.2 improves by 27.5 points from minimal to maximum reasoning effort; and (4) strong agents are not necessarily strong environment simulators. Simulator quality is critical for LWM-based evaluation reliability. OccuBench provides the first systematic cross-industry evaluation of AI agents on professional occupational tasks. 

---
# Advancing Polish Language Modeling through Tokenizer Optimization in the Bielik v3 7B and 11B Series 

**Authors**: Krzysztof Ociepa, Łukasz Flis, Remigiusz Kinas, Krzysztof Wróbel, Adrian Gwoździej  

**Link**: [PDF](https://arxiv.org/pdf/2604.10799)  

**Abstract**: The development of the Bielik v3 PL series, encompassing both the 7B and 11B parameter variants, represents a significant milestone in the field of language-specific large language model (LLM) optimization. While general-purpose models often demonstrate impressive multilingual capabilities, they frequently suffer from a fundamental architectural inefficiency: the use of universal tokenizers. These tokenizers, typically designed to cover a broad spectrum of languages, often fail to capture the morphological nuances of specific languages like Polish, leading to higher fertility ratios, increased inference costs, and restricted effective context windows. This report details the transition from the universal Mistral-based tokenization to a dedicated Polish-optimized vocabulary for the Bielik v3 models, exploring the FOCUS-based embedding initialization, the multi-stage pretraining curriculum, and the subsequent post-training alignment involving Supervised Fine-Tuning, Direct Preference Optimization, and Reinforcement Learning through Group Relative Policy Optimization with verifiable rewards. 

---
# Position-Agnostic Pre-Projection for Transformer Attention: Nonlinear Feature Construction and Content Skip Before Q/K/V 

**Authors**: Chirag Shinde  

**Link**: [PDF](https://arxiv.org/pdf/2604.10791)  

**Abstract**: We propose two complementary modifications to transformer attention blocks. First, a non-linear pre-projection MLP is inserted between layer norm and Q/K/V projections, constructing richer features in a position-agnostic manner before any positional encoding is applied. Second, a content skip connection routes the pre-projection's features around the attention mechanism, allowing content information to bypass position-aware attention where beneficial. In frozen-probe experiments on Pythia-160M and 410M, the combined approach achieves the strongest results across methods: +40.6% LAMBADA accuracy and -39% perplexity at 160M scale. Learned skip connection weights reveal a consistent pattern across model sizes: later transformer layers activate the content bypass more strongly than earlier layers, suggesting that deeper layers benefit from content information that does not pass through positional attention. All modifications add no K/V cache overhead. 

---
# TInR: Exploring Tool-Internalized Reasoning in Large Language Models 

**Authors**: Qiancheng Xu, Yongqi Li, Fan Liu, Hongru Wang, Min Yang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.10788)  

**Abstract**: Tool-Integrated Reasoning (TIR) has emerged as a promising direction by extending Large Language Models' (LLMs) capabilities with external tools during reasoning. Existing TIR methods typically rely on external tool documentation during reasoning. However, this leads to tool mastery difficulty, tool size constraints, and inference inefficiency. To mitigate these issues, we explore Tool-Internalized Reasoning (TInR), aiming at facilitating reasoning with tool knowledge internalized into LLMs. Achieving this goal presents notable requirements, including tool internalization and tool-reasoning coordination. To address them, we propose TInR-U, a tool-internalized reasoning framework for unified reasoning and tool usage. TInR-U is trained through a three-phase pipeline: 1) tool internalization with a bidirectional knowledge alignment strategy; 2) supervised fine-tuning warm-up using high-quality reasoning annotations, and 3) reinforcement learning with TInR-specific rewards. We comprehensively evaluate our method across in-domain and out-of-domain settings. Experiment results show that TInR-U achieves superior performance in both settings, highlighting its effectiveness and efficiency. 

---
# When Meaning Isn't Literal: Exploring Idiomatic Meaning Across Languages and Modalities 

**Authors**: Sarmistha Das, Shreyas Guha, Suvrayan Bandyopadhyay, Salisa Phosit, Kitsuchart Pasupa, Sriparna Saha  

**Link**: [PDF](https://arxiv.org/pdf/2604.10787)  

**Abstract**: Idiomatic reasoning, deeply intertwined with metaphor and culture, remains a blind spot for contemporary language models, whose progress skews toward surface-level lexical and semantic cues. For instance, the Bengali idiom \textit{\foreignlanguage{bengali}{\char"0986\char"0999\char"09CD\char"0997\char"09C1 \char"09B0 \char"09AB\char"09B2 \char"099F\char"0995}} (angur fol tok, ``grapes are sour''): it encodes denial-driven rationalization, yet naive models latch onto the literal fox-and-grape imagery. Addressing this oversight, we present ``Mediom,'' a multilingual, multimodal idiom corpus of 3,533 Hindi, Bengali, and Thai idioms, each paired with gold-standard explanations, cross-lingual translations, and carefully aligned text--image representations. We benchmark both large language models (textual reasoning) and vision-language models (figurative disambiguation) on Mediom, exposing systematic failures in metaphor comprehension. To mitigate these gaps, we propose ``HIDE,'' a Hinting-based Idiom Explanation framework that leverages error-feedback retrieval and targeted diagnostic cues for iterative reasoning refinement. Collectively, Mediom and HIDE establish a rigorous test bed and methodology for culturally grounded, multimodal idiom understanding embedded with reasoning hints in next-generation AI systems. 

---
# Do BERT Embeddings Encode Narrative Dimensions? A Token-Level Probing Analysis of Time, Space, Causality, and Character in Fiction 

**Authors**: Beicheng Bei, Hannah Hyesun Chun, Chen Guo, Arwa Saghiri  

**Link**: [PDF](https://arxiv.org/pdf/2604.10786)  

**Abstract**: Narrative understanding requires multidimensional semantic structures. This study investigates whether BERT embeddings encode dimensions of fictional narrative semantics -- time, space, causality, and character. Using an LLM to accelerate annotation, we construct a token-level dataset labeled with these four narrative categories plus "others." A linear probe on BERT embeddings (94% accuracy) significantly outperforms a control probe on variance-matched random embeddings (47%), confirming that BERT encodes meaningful narrative information. With balanced class weighting, the probe achieves a macro-average recall of 0.83, with moderate success on rare categories such as causality (recall = 0.75) and space (recall = 0.66). However, confusion matrix analysis reveals "Boundary Leakage," where rare dimensions are systematically misclassified as "others." Clustering analysis shows that unsupervised clustering aligns near-randomly with predefined categories (ARI = 0.081), suggesting that narrative dimensions are encoded but not as discretely separable clusters. Future work includes a POS-only baseline to disentangle syntactic patterns from narrative encoding, expanded datasets, and layer-wise probing. 

---
# Generating Multiple-Choice Knowledge Questions with Interpretable Difficulty Estimation using Knowledge Graphs and Large Language Models 

**Authors**: Mehmet Can Şakiroğlu, H. Altay Güvenir, Kamer Kaya  

**Link**: [PDF](https://arxiv.org/pdf/2604.10748)  

**Abstract**: Generating multiple-choice questions (MCQs) with difficulty estimation remains challenging in automated MCQ-generation systems used in adaptive, AI-assisted education. This study proposes a novel methodology for generating MCQs with difficulty estimation from the input documents by utilizing knowledge graphs (KGs) and large language models (LLMs). Our approach uses an LLM to construct a KG from input documents, from which MCQs are then systematically generated. Each MCQ is generated by selecting a node from the KG as the key, sampling a related triple or quintuple -- optionally augmented with an extra triple -- and prompting an LLM to generate a corresponding stem from these graph components. Distractors are then selected from the KG. For each MCQ, nine difficulty signals are computed and combined into a unified difficulty score using a data-driven approach. Experimental results demonstrate that our method generates high-quality MCQs whose difficulty estimation is interpretable and aligns with human perceptions. Our approach improves automated MCQ generation by integrating structured knowledge representations with LLMs and a data-driven difficulty estimation model. 

---
# How You Ask Matters! Adaptive RAG Robustness to Query Variations 

**Authors**: Yunah Jang, Megha Sundriyal, Kyomin Jung, Meeyoung Cha  

**Link**: [PDF](https://arxiv.org/pdf/2604.10745)  

**Abstract**: Adaptive Retrieval-Augmented Generation (RAG) promises accuracy and efficiency by dynamically triggering retrieval only when needed and is widely used in practice. However, real-world queries vary in surface form even with the same intent, and their impact on Adaptive RAG remains under-explored. We introduce the first large-scale benchmark of diverse yet semantically identical query variations, combining human-written and model-generated rewrites. Our benchmark facilitates a systematic evaluation of Adaptive RAG robustness by examining its key components across three dimensions: answer quality, computational cost, and retrieval decisions. We discover a critical robustness gap, where small surface-level changes in queries dramatically alter retrieval behavior and accuracy. Although larger models show better performance, robustness does not improve accordingly. These findings reveal that Adaptive RAG methods are highly vulnerable to query variations that preserve identical semantics, exposing a critical robustness challenge. 

---
# Deep-Reporter: Deep Research for Grounded Multimodal Long-Form Generation 

**Authors**: Fangda Ye, Zhifei Xie, Yuxin Hu, Yihang Yin, Shurui Huang, Shikai Dong, Jianzhu Bao, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.10741)  

**Abstract**: Recent agentic search frameworks enable deep research via iterative planning and retrieval, reducing hallucinations and enhancing factual grounding. However, they remain text-centric, overlooking the multimodal evidence that characterizes real-world expert reports. We introduce a pressing task: multimodal long-form generation. Accordingly, we propose Deep-Reporter, a unified agentic framework for grounded multimodal long-form generation. It orchestrates: (i) Agentic Multimodal Search and Filtering to retrieve and filter textual passages and information-dense visuals; (ii) Checklist-Guided Incremental Synthesis to ensure coherent image-text integration and optimal citation placement; and (iii) Recurrent Context Management to balance long-range coherence with local fluency. We develop a rigorous curation pipeline producing 8K high-quality agentic traces for model optimization. We further introduce M2LongBench, a comprehensive testbed comprising 247 research tasks across 9 domains and a stable multimodal sandbox. Extensive experiments demonstrate that long-form multimodal generation is a challenging task, especially in multimodal selection and integration, and effective post-training can bridge the gap. 

---
# RCBSF: A Multi-Agent Framework for Automated Contract Revision via Stackelberg Game 

**Authors**: Shijia Xu, Yu Wang, Xiaolong Jia, Zhou Wu, Kai Liu, April Xiaowen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.10740)  

**Abstract**: Despite the widespread adoption of Large Language Models (LLMs) in Legal AI, their utility for automated contract revision remains impeded by hallucinated safety and a lack of rigorous behavioral constraints. To address these limitations, we propose the Risk-Constrained Bilevel Stackelberg Framework (RCBSF), which formulates revision as a non-cooperative Stackelberg game. RCBSF establishes a hierarchical Leader Follower structure where a Global Prescriptive Agent (GPA) imposes risk budgets upon a follower system constituted by a Constrained Revision Agent (CRA) and a Local Verification Agent (LVA) to iteratively optimize output. We provide theoretical guarantees that this bilevel formulation converges to an equilibrium yielding strictly superior utility over unguided configurations. Empirical validation on a unified benchmark demonstrates that RCBSF achieves state-of-the-art performance, surpassing iterative baselines with an average Risk Resolution Rate (RRR) of 84.21\% while enhancing token efficiency. Our code is available at this https URL . 

---
# BlasBench: An Open Benchmark for Irish Speech Recognition 

**Authors**: Jyoutir Raj, John Conway  

**Link**: [PDF](https://arxiv.org/pdf/2604.10736)  

**Abstract**: No open Irish-specific benchmark compares end-user ASR systems under a shared Irish-aware evaluation protocol. To solve this, we release BlasBench, an open evaluation harness with Irish-aware text normalisation that preserves fadas, lenition, and eclipsis. We benchmark 12 systems across four architecture families on Common Voice ga-IE and FLEURS ga-IE. All Whisper variants exceed 100% WER. The best open model (omniASR LLM 7B) achieves 30.65% WER on Common Voice and 39.09% on FLEURS. We noticed models fine-tuned on Common Voice lose 33-43 WER points on FLEURS, revealing a generalisation gap that is invisible to single-dataset evaluation. 

---
# Self-Correcting RAG: Enhancing Faithfulness via MMKP Context Selection and NLI-Guided MCTS 

**Authors**: Shijia Xu, Zhou Wu, Xiaolong Jia, Yu Wang, Kai Liu, April Xiaowen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.10734)  

**Abstract**: Retrieval-augmented generation (RAG) substantially extends the knowledge boundary of large language models. However, it still faces two major challenges when handling complex reasoning tasks: low context utilization and frequent hallucinations. To address these issues, we propose Self-Correcting RAG, a unified framework that reformulates retrieval and generation as constrained optimization and path planning. On the input side, we move beyond traditional greedy retrieval and, for the first time, formalize context selection as a multi-dimensional multiple-choice knapsack problem (MMKP), thereby maximizing information density and removing redundancy under a strict token budget. On the output side, we introduce a natural language inference (NLI)-guided Monte Carlo Tree Search (MCTS) mechanism, which leverages test-time compute to dynamically explore reasoning trajectories and validate the faithfulness of generated answers. Experiments on six multi-hop question answering and fact-checking datasets demonstrate that our method significantly improves reasoning accuracy on complex queries while effectively reducing hallucinations, outperforming strong existing this http URL code is available at this https URL . 

---
# Too Nice to Tell the Truth: Quantifying Agreeableness-Driven Sycophancy in Role-Playing Language Models 

**Authors**: Arya Shah, Deepali Mishra, Chaklam Silpasuwanchai  

**Link**: [PDF](https://arxiv.org/pdf/2604.10733)  

**Abstract**: Large language models increasingly serve as conversational agents that adopt personas and role-play characters at user request. This capability, while valuable, raises concerns about sycophancy: the tendency to provide responses that validate users rather than prioritize factual accuracy. While prior work has established that sycophancy poses risks to AI safety and alignment, the relationship between specific personality traits of adopted personas and the degree of sycophantic behavior remains unexplored. We present a systematic investigation of how persona agreeableness influences sycophancy across 13 small, open-weight language models ranging from 0.6B to 20B parameters. We develop a benchmark comprising 275 personas evaluated on NEO-IPIP agreeableness subscales and expose each persona to 4,950 sycophancy-eliciting prompts spanning 33 topic categories. Our analysis reveals that 9 of 13 models exhibit statistically significant positive correlations between persona agreeableness and sycophancy rates, with Pearson correlations reaching $r = 0.87$ and effect sizes as large as Cohen's $d = 2.33$. These findings demonstrate that agreeableness functions as a reliable predictor of persona-induced sycophancy, with direct implications for the deployment of role-playing AI systems and the development of alignment strategies that account for personality-mediated deceptive behaviors. 

---
# Expect the Unexpected? Testing the Surprisal of Salient Entities 

**Authors**: Jessica Lin, Amir Zeldes  

**Link**: [PDF](https://arxiv.org/pdf/2604.10724)  

**Abstract**: Previous work examining the Uniform Information Density (UID) hypothesis has shown that while information as measured by surprisal metrics is distributed more or less evenly across documents overall, local discrepancies can arise due to functional pressures corresponding to syntactic and discourse structural constraints. However, work thus far has largely disregarded the relative salience of discourse participants. We fill this gap by studying how overall salience of entities in discourse relates to surprisal using 70K manually annotated mentions across 16 genres of English and a novel minimal-pair prompting method. Our results show that globally salient entities exhibit significantly higher surprisal than non-salient ones, even controlling for position, length, and nesting confounds. Moreover, salient entities systematically reduce surprisal for surrounding content when used as prompts, enhancing document-level predictability. This effect varies by genre, appearing strongest in topic-coherent texts and weakest in conversational contexts. Our findings refine the UID competing pressures framework by identifying global entity salience as a mechanism shaping information distribution in discourse. 

---
# Attention Sinks as Internal Signals for Hallucination Detection in Large Language Models 

**Authors**: Jakub Binkowski, Kamil Adamczewski, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2604.10697)  

**Abstract**: Large language models frequently exhibit hallucinations: fluent and confident outputs that are factually incorrect or unsupported by the input context. While recent hallucination detection methods have explored various features derived from attention maps, the underlying mechanisms they exploit remain poorly understood. In this work, we propose SinkProbe, a hallucination detection method grounded in the observation that hallucinations are deeply entangled with attention sinks - tokens that accumulate disproportionate attention mass during generation - indicating a transition from distributed, input-grounded attention to compressed, prior-dominated computation. Importantly, although sink scores are computed solely from attention maps, we find that the classifier preferentially relies on sinks whose associated value vectors have large norms. Moreover, we show that previous methods implicitly depend on attention sinks by establishing their mathematical relationship to sink scores. Our findings yield a novel hallucination detection method grounded in theory that produces state-of-the-art results across popular datasets and LLMs. 

---
# QFS-Composer: Query-focused summarization pipeline for less resourced languages 

**Authors**: Vuk Đuranović, Marko Robnik Šikonja  

**Link**: [PDF](https://arxiv.org/pdf/2604.10687)  

**Abstract**: Large language models (LLMs) demonstrate strong performance in text summarization, yet their effectiveness drops significantly across languages with restricted training resources. This work addresses the challenge of query-focused summarization (QFS) in less-resourced languages, where labeled datasets and evaluation tools are limited. We present a novel QFS framework, QFS-Composer, that integrates query decomposition, question generation (QG), question answering (QA), and abstractive summarization to improve the factual alignment of a summary with user intent. We test our approach on the Slovenian language. To enable high-quality supervision and evaluation, we develop the Slovenian QA and QG models based on a Slovene LLM and adapt evaluation approaches for reference-free summary evaluation. Empirical evaluation shows that the QA-guided summarization pipeline yields improved consistency and relevance over baseline LLMs. Our work establishes an extensible methodology for advancing QFS in less-resourced languages. 

---
# Learning and Enforcing Context-Sensitive Control for LLMs 

**Authors**: Mohammad Albinhassan, Pranava Madhyastha, Mark Law, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2604.10667)  

**Abstract**: Controlling the output of Large Language Models (LLMs) through context-sensitive constraints has emerged as a promising approach to overcome the limitations of Context-Free Grammars (CFGs) in guaranteeing generation validity. However, such constraints typically require manual specification -- a significant barrier demanding specialized expertise. We introduce a framework that automatically learns context-sensitive constraints from LLM interactions through a two-phase process: syntactic exploration to gather diverse outputs for constraint learning, followed by constraint exploitation to enforce these learned rules during generation. Experiments demonstrate that our method enables even small LLMs (1B parameters) to learn and generate with perfect constraint adherence, outperforming larger counterparts and state-of-the-art reasoning models. This work represents the first integration of context-sensitive grammar learning with LLM generation, eliminating manual specification while maintaining generation validity. 

---
# HeceTokenizer: A Syllable-Based Tokenization Approach for Turkish Retrieval 

**Authors**: Senol Gulgonul  

**Link**: [PDF](https://arxiv.org/pdf/2604.10665)  

**Abstract**: HeceTokenizer is a syllable-based tokenizer for Turkish that exploits the deterministic six-pattern phonological structure of the language to construct a closed, out-of-vocabulary (OOV)-free vocabulary of approximately 8,000 unique syllable types. A BERT-tiny encoder (1.5M parameters) is trained from scratch on a subset of Turkish Wikipedia using a masked language modeling objective and evaluated on the TQuAD retrieval benchmark using Recall@5. Combined with a fine-grained chunk-based retrieval strategy, HeceTokenizer achieves 50.3% Recall@5, surpassing the 46.92% reported by a morphology-driven baseline that uses a 200 times larger model. These results suggest that the phonological regularity of Turkish syllables provides a strong and resource-light inductive bias for retrieval tasks. 

---
# Efficient Process Reward Modeling via Contrastive Mutual Information 

**Authors**: Nakyung Lee, Sangwoo Hong, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.10660)  

**Abstract**: Recent research has devoted considerable effort to verifying the intermediate reasoning steps of chain-of-thought (CoT) trajectories using process reward models (PRMs) and other verifier models. However, training a PRM typically requires human annotators to assign reward scores to each reasoning step, which is both costly and time-consuming. Existing automated approaches, such as Monte Carlo (MC) estimation, also demand substantial computational resources due to repeated LLM rollouts. To overcome these limitations, we propose contrastive pointwise mutual information (CPMI), a novel automatic reward labeling method that leverages the model's internal probability to infer step-level supervision while significantly reducing the computational burden of annotating dataset. CPMI quantifies how much a reasoning step increases the mutual information between the step and the correct target answer relative to hard-negative alternatives. This contrastive signal serves as a proxy for the step's contribution to the final solution and yields a reliable reward. The experimental results show that CPMI-based labeling reduces dataset construction time by 84% and token generation by 98% compared to MC estimation, while achieving higher accuracy on process-level evaluations and mathematical reasoning benchmarks. 

---
# ProUIE: A Macro-to-Micro Progressive Learning Method for LLM-based Universal Information Extraction 

**Authors**: Wenda Liu, Zhigang Song, Shuai Nie, Guangyao Liu, Lisung Chen, Binyu Yang, Yaran Chen, Peng Zhou, Hongzhen Wang, Yuchen Liu, Wenyue Hu, Jiaming Xu, Runyu Shi, Ying Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10633)  

**Abstract**: LLM-based universal information extraction (UIE) methods often rely on additional information beyond the original training data, which increases training complexity yet often yields limited gains. To address this, we propose ProUIE, a Macro-to-Micro progressive learning approach that improves UIE without introducing any external information. ProUIE consists of three stages: (i) macro-level Complete Modeling (CM), which learns NER, RE, and EE along their intrinsic difficulty order on the full training data to build a unified extraction foundation, (ii) meso-level Streamlined Alignment (SA), which operates on sampled data with simplified target formats, streamlining and regularizing structured outputs to make them more concise and controllable, and (iii) micro-level Deep Exploration (DE), which applies GRPO with stepwise fine-grained rewards (SFR) over structural units to guide exploration and improve performance. Experiments on 36 public datasets show that ProUIE consistently improves unified extraction, outperforming strong instruction-tuned baselines on average for NER and RE while using a smaller backbone, and it further demonstrates clear gains in large-scale production-oriented information extraction. 

---
# Computational Lesions in Multilingual Language Models Separate Shared and Language-specific Brain Alignment 

**Authors**: Yang Cui, Jingyuan Sun, Yizheng Sun, Yifan Wang, Yunhao Zhang, Jixing Li, Shaonan Wang, Hongpeng Zhou, John Hale, Chengqing Zong, Goran Nenadic  

**Link**: [PDF](https://arxiv.org/pdf/2604.10627)  

**Abstract**: How the brain supports language across different languages is a basic question in neuroscience and a useful test for multilingual artificial intelligence. Neuroimaging has identified language-responsive brain regions across languages, but it cannot by itself show whether the underlying processing is shared or language-specific. Here we use six multilingual large language models (LLMs) as controllable systems and create targeted ``computational lesions'' by zeroing small parameter sets that are important across languages or especially important for one language. We then compare intact and lesioned models in predicting functional magnetic resonance imaging (fMRI) responses during 100 minutes of naturalistic story listening in native English, Chinese and French (112 participants). Lesioning a compact shared core reduces whole-brain encoding correlation by 60.32% relative to intact models, whereas language-specific lesions preserve cross-language separation in embedding space but selectively weaken brain predictivity for the matched native language. These results support a shared backbone with embedded specializations and provide a causal framework for studying multilingual brain-model alignment. 

---
# Bridging Linguistic Gaps: Cross-Lingual Mapping in Pre-Training and Dataset for Enhanced Multilingual LLM Performance 

**Authors**: Weihua Zheng, Chang Liu, Zhengyuan Liu, Xin Huang, Kui Wu, Muhammad Huzaifah Md Shahrin, Aiti Aw, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.10590)  

**Abstract**: Multilingual Large Language Models (LLMs) struggle with cross-lingual tasks due to data imbalances between high-resource and low-resource languages, as well as monolingual bias in pre-training. Existing methods, such as bilingual fine-tuning and contrastive alignment, can improve cross-lingual performance, but they often require extensive parallel data or suffer from instability. To address these challenges, we introduce a Cross-Lingual Mapping Task during the pre-training phase, which enhances cross-lingual alignment without compromising monolingual fluency. Our approach bi-directionally maps languages within the LLM embedding space, improving both language generation and comprehension. We further propose a Language Alignment Coefficient to robustly quantify cross-lingual consistency, even in limited-data scenarios. Experimental results on machine translation (MT), cross-lingual natural language understanding (CLNLU), and cross-lingual question answering (CLQA) show that our model achieves gains of up to 11.9 BLEU points in MT, 6.72 points in CLQA BERTScore-Precision, and more than 5% in CLNLU accuracy over strong multilingual baselines. These findings highlight the potential of incorporating cross-lingual objectives into pre-training to improve multilingual LLMs. 

---
# Knowing What to Stress: A Discourse-Conditioned Text-to-Speech Benchmark 

**Authors**: Arnon Turetzky, Avihu Dekel, Hagai Aronowitz, Ron Hoory, Yossi Adi  

**Link**: [PDF](https://arxiv.org/pdf/2604.10580)  

**Abstract**: Spoken meaning often depends not only on what is said, but also on which word is emphasized. The same sentence can convey correction, contrast, or clarification depending on where emphasis falls. Although modern text-to-speech (TTS) systems generate expressive speech, it remains unclear whether they infer contextually appropriate stress from discourse alone. To address this gap, we present Context-Aware Stress TTS (CAST), a benchmark for evaluating context-conditioned word-level stress in TTS. Items are defined as contrastive context pairs: identical sentences paired with distinct contexts requiring different stressed words. We evaluate state-of-the-art systems and find a consistent gap: text-only language models reliably recover the intended stress from context, yet TTS systems frequently fail to realize it in speech. We release the benchmark, evaluation framework, construction pipeline and a synthetic corpus to support future work on context-aware speech synthesis. 

---
# Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models 

**Authors**: Jiyeon Kim, Sungik Choi, Yongrae Jo, Moontae Lee, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2604.10567)  

**Abstract**: Diffusion-based language models (dLLMs) have emerged as a promising alternative to autoregressive language models, offering the potential for parallel token generation and bidirectional context modeling. However, harnessing this flexibility for fully non-autoregressive decoding remains an open question, particularly for reasoning and planning tasks. In this work, we investigate non-autoregressive decoding in dLLMs by systematically analyzing its inference dynamics along the temporal axis. Specifically, we uncover an inherent failure mode in confidence-based non-autoregressive generation stemming from a strong proximity bias-the tendency for the denoising order to concentrate on spatially adjacent tokens. This local dependency leads to spatial error propagation, rendering the entire trajectory critically contingent on the initial unmasking position. Leveraging this insight, we present a minimal-intervention approach that guides early token selection, employing a lightweight planner and end-of-sequence temperature annealing. We thoroughly evaluate our method on various reasoning and planning tasks and observe substantial overall improvement over existing heuristic baselines without significant computational overhead. 

---
# LLMs Should Incorporate Explicit Mechanisms for Human Empathy 

**Authors**: Xiaoxing You, Qiang Huang, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10557)  

**Abstract**: This paper argues that Large Language Models (LLMs) should incorporate explicit mechanisms for human empathy. As LLMs become increasingly deployed in high-stakes human-centered settings, their success depends not only on correctness or fluency but on faithful preservation of human perspectives. Yet, current LLMs systematically fail at this requirement: even when well-aligned and policy-compliant, they often attenuate affect, misrepresent contextual salience, and rigidify relational stance in ways that distort meaning. We formalize empathy as an observable behavioral property: the capacity to model and respond to human perspectives while preserving intention, affect, and context. Under this framing, we identify four recurring mechanisms of empathic failure in contemporary LLMs--sentiment attenuation, empathic granularity mismatch, conflict avoidance, and linguistic distancing--arising as structural consequences of prevailing training and alignment practices. We further organize these failures along three dimensions: cognitive, cultural, and relational empathy, to explain their manifestation across tasks. Empirical analyses show that strong benchmark performance can mask systematic empathic distortions, motivating empathy-aware objectives, benchmarks, and training signals as first-class components of LLM development. 

---
# Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models 

**Authors**: Zhengnan Guo, Fei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.10556)  

**Abstract**: While Diffusion Large Language Models (dLLMs) have emerged as a promising non-autoregressive paradigm comparable to autoregressive (AR) models, their faithfulness, specifically regarding hallucination, remains largely underexplored. To bridge this gap, we present the first controlled comparative study to evaluate hallucination patterns in dLLMs. Our results demonstrate that current dLLMs exhibit a higher propensity for hallucination than AR counterparts controlled for architecture, scale, and pre-training weights. Furthermore, an analysis of inference-time compute reveals divergent dynamics: while quasi-autoregressive generation suffers from early saturation, non-sequential decoding unlocks potential for continuous refinement. Finally, we identify distinct failure modes unique to the diffusion process, including premature termination, incomplete denoising, and context intrusion. Our findings underscore that although dLLMs have narrowed the performance gap on general tasks, their distinct hallucination mechanisms pose a critical challenge to model reliability. Our code is available at this https URL 

---
# ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization 

**Authors**: Suyoung Bae, CheolWon Na, Jaehoon Lee, Yumin Lee, YunSeok Choi, Jee-Hyong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.10520)  

**Abstract**: As Large Language Models (LLMs) have become capable of generating long and descriptive code summaries, accurate and reliable evaluation of factual consistency has become a critical challenge. However, previous evaluation methods are primarily designed for short summaries of isolated code snippets. Consequently, they struggle to provide fine-grained evaluation of multi-sentence functionalities and fail to accurately assess dependency context commonly found in real-world code summaries. To address this, we propose ReFEree, a reference-free and fine-grained method for evaluating factual consistency in real-world code summaries. We define factual inconsistency criteria specific to code summaries and evaluate them at the segment level using these criteria along with dependency information. These segment-level results are then aggregated into a fine-grained score. We construct a code summarization benchmark with human-annotated factual consistency labels. The evaluation results demonstrate that ReFEree achieves the highest correlation with human judgment among 13 baselines, improving 15-18% over the previous state-of-the-art. Our code and data are available at this https URL. 

---
# Structure-Grounded Knowledge Retrieval via Code Dependencies for Multi-Step Data Reasoning 

**Authors**: Xinyi Huang, Mingzhe Lu, Haoyu Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.10516)  

**Abstract**: Selecting the right knowledge is critical when using large language models (LLMs) to solve domain-specific data analysis tasks. However, most retrieval-augmented approaches rely primarily on lexical or embedding similarity, which is often a weak proxy for the task-critical knowledge needed for multi-step reasoning. In many such tasks, the relevant knowledge is not merely textually related to the query, but is instead grounded in executable code and the dependency structure through which computations are carried out. To address this mismatch, we propose SGKR (Structure-Grounded Knowledge Retrieval), a retrieval framework that organizes domain knowledge with a graph induced by function-call dependencies. Given a question, SGKR extracts semantic input and output tags, identifies dependency paths connecting them, and constructs a task-relevant subgraph. The associated knowledge and corresponding function implementations are then assembled as a structured context for LLM-based code generation. Experiments on multi-step data analysis benchmarks show that SGKR consistently improves solution correctness over no-retrieval and similarity-based retrieval baselines for both vanilla LLMs and coding agents. 

---
# Why Don't You Know? Evaluating the Impact of Uncertainty Sources on Uncertainty Quantification in LLMs 

**Authors**: Maiya Goloburda, Roman Vashurin, Fedor Chernogorsky, Nurkhan Laiyk, Daniil Orel, Preslav Nakov, Maxim Panov  

**Link**: [PDF](https://arxiv.org/pdf/2604.10495)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in real-world applications, reliable uncertainty quantification (UQ) becomes critical for safe and effective use. Most existing UQ approaches for language models aim to produce a single confidence score -- for example, estimating the probability that a model's answer is correct. However, uncertainty in natural language tasks arises from multiple distinct sources, including model knowledge gaps, output variability, and input ambiguity, which have different implications for system behavior and user interaction. In this work, we study how the source of uncertainty impacts the behavior and effectiveness of existing UQ methods. To enable controlled analysis, we introduce a new dataset that explicitly categorizes uncertainty sources, allowing systematic evaluation of UQ performance under each condition. Our experiments reveal that while many UQ methods perform well when uncertainty stems solely from model knowledge limitations, their performance degrades or becomes misleading when other sources are introduced. These findings highlight the need for uncertainty-aware methods that explicitly account for the source of uncertainty in large language models. 

---
# From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation 

**Authors**: Mingfei Lu, Yi Zhang, Mengjia Wu, Yue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.10470)  

**Abstract**: Legal consultation question answering (Legal CQA) presents unique challenges compared to traditional legal QA tasks, including the scarcity of high-quality training data, complex task composition, and strong contextual dependencies. To address these, we construct JurisCQAD, a large-scale dataset of over 43,000 real-world Chinese legal queries annotated with expert-validated positive and negative responses, and design a structured task decomposition that converts each query into a legal element graph integrating entities, events, intents, and legal issues. We further propose JurisMA, a modular multi-agent framework supporting dynamic routing, statutory grounding, and stylistic optimization. Combined with the element graph, the framework enables strong context-aware reasoning, effectively capturing dependencies across legal facts, norms, and procedural logic. Trained on JurisCQAD and evaluated on a refined LawBench, our system significantly outperforms both general-purpose and legal-domain LLMs across multiple lexical and semantic metrics, demonstrating the benefits of interpretable decomposition and modular collaboration in Legal CQA. 

---
# Dynamic Adaptive Attention and Supervised Contrastive Learning: A Novel Hybrid Framework for Text Sentiment Classification 

**Authors**: Qingyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.10459)  

**Abstract**: The exponential growth of user-generated movie reviews on digital platforms has made accurate text sentiment classification a cornerstone task in natural language processing. Traditional models, including standard BERT and recurrent architectures, frequently struggle to capture long-distance semantic dependencies and resolve ambiguous emotional expressions in lengthy review texts. This paper proposes a novel hybrid framework that seamlessly integrates dynamic adaptive multi-head attention with supervised contrastive learning into a BERT-based Transformer encoder. The dynamic adaptive attention module employs a global context pooling vector to dynamically regulate the contribution of each attention head, thereby focusing on critical sentiment-bearing tokens while suppressing noise. Simultaneously, the supervised contrastive learning branch enforces tighter intra-class compactness and larger inter-class separation in the embedding space. Extensive experiments on the IMDB dataset demonstrate that the proposed model achieves competitive performance with an accuracy of 94.67\%, outperforming strong baselines by 1.5--2.5 percentage points. The framework is lightweight, efficient, and readily extensible to other text classification tasks. 

---
# EviCare: Enhancing Diagnosis Prediction with Deep Model-Guided Evidence for In-Context Reasoning 

**Authors**: Hengyu Zhang, Xuyun Zhang, Pengxiang Zhan, Linhao Luo, Hang Lv, Yanchao Tan, Shirui Pan, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10455)  

**Abstract**: Recent advances in large language models (LLMs) have enabled promising progress in diagnosis prediction from electronic health records (EHRs). However, existing LLM-based approaches tend to overfit to historically observed diagnoses, often overlooking novel yet clinically important conditions that are critical for early intervention. To address this, we propose EviCare, an in-context reasoning framework that integrates deep model guidance into LLM-based diagnosis prediction. Rather than prompting LLMs directly with raw EHR inputs, EviCare performs (1) deep model inference for candidate selection, (2) evidential prioritization for set-based EHRs, and (3) relational evidence construction for novel diagnosis prediction. These signals are then composed into an adaptive in-context prompt to guide LLM reasoning in an accurate and interpretable manner. Extensive experiments on two real-world EHR benchmarks (MIMIC-III and MIMIC-IV) demonstrate that EviCare achieves significant performance gains, which consistently outperforms both LLM-only and deep model-only baselines by an average of 20.65\% across precision and accuracy metrics. The improvements are particularly notable in challenging novel diagnosis prediction, yielding average improvements of 30.97\%. 

---
# NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning 

**Authors**: Yanyi Su, Hongshuai Wang, Zhifeng Gao, Jun Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.10452)  

**Abstract**: Olfaction lies at the intersection of chemical structure, neural encoding, and linguistic perception, yet existing representation methods fail to fully capture this pathway. Current approaches typically model only isolated segments of the olfactory pathway, overlooking the complete chain from molecule to receptors to linguistic descriptions. Such fragmentation yields learned embeddings that lack both biological grounding and semantic interpretability. We propose NOSE (Neural Olfactory-Semantic Embedding), a representation learning framework that aligns three modalities along the olfactory pathway: molecular structure, receptor sequence, and natural language description. Rather than simply fusing these signals, we decouple their contributions via orthogonal constraints, preserving the unique encoded information of each modality. To address the sparsity of olfactory language, we introduce a weak positive sample strategy to calibrate semantic similarity, preventing erroneous repulsion of similar odors in the feature space. Extensive experiments demonstrate that NOSE achieves state-of-the-art (SOTA) performance and excellent zero-shot generalization, confirming the strong alignment between its representation space and human olfactory intuition. 

---
# Instruction Data Selection via Answer Divergence 

**Authors**: Bo Li, Mingda Wang, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.10448)  

**Abstract**: Instruction tuning relies on large instruction-response corpora whose quality and composition strongly affect downstream performance. We propose Answer Divergence-Guided Selection (ADG), which selects instruction data based on the geometric structure of multi-sample outputs. ADG draws several high-temperature generations per instruction, maps responses into an embedding space, and computes an output divergence score that jointly encodes dispersion magnitude and shape anisotropy. High scores correspond to instructions whose answers are both far apart and multi-modal, rather than clustered paraphrases along a single direction. Across two backbones and three public instruction pools, fine-tuning on only 10K ADG-selected examples consistently outperforms strong selectors on six benchmarks spanning reasoning, knowledge, and coding. Analyses further show that both dispersion magnitude and shape anisotropy are necessary, supporting answer divergence as a practical signal for instruction data selection. Code and appendix are included in the supplementary materials. 

---
# CodaRAG: Connecting the Dots with Associativity Inspired by Complementary Learning 

**Authors**: Cheng-Yen Li, Xuanjun Chen, Claire Lin, Wei-Yu Chen, Wenhua Nie, Hung-Yi Lee, Jyh-Shing Roger Jang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10426)  

**Abstract**: Large Language Models (LLMs) struggle with knowledge-intensive tasks due to hallucinations and fragmented reasoning over dispersed information. While Retrieval-Augmented Generation (RAG) grounds generation in external sources, existing methods often treat evidence as isolated units, failing to reconstruct the logical chains that connect these dots. Inspired by Complementary Learning Systems (CLS), we propose CodaRAG, a framework that evolves retrieval from passive lookup into active associative discovery. CodaRAG operates via a three-stage pipeline: (1) Knowledge Consolidation to unify fragmented extractions into a stable memory substrate; (2) Associative Navigation to traverse the graph via multi-dimensional pathways-semantic, contextualized, and functional-explicitly recovering dispersed evidence chains; and (3) Interference Elimination to prune hyper-associative noise, ensuring a coherent, high-precision reasoning context. On GraphRAG-Bench, CodaRAG achieves absolute gains of 7-10% in retrieval recall and 3-11% in generation accuracy. These results demonstrate CodaRAG's superior ability to systematically robustify associative evidence retrieval for factual, reasoning, and creative tasks. 

---
# Turing or Cantor: That is the Question 

**Authors**: Eugene Eberbach  

**Link**: [PDF](https://arxiv.org/pdf/2604.10418)  

**Abstract**: Alan Turing is considered as a founder of current computer science together with Kurt Godel, Alonzo Church and John von Neumann. In this paper multiple new research results are presented. It is demonstrated that there would not be Alan Turing's achievements without earlier seminal contributions by Georg Cantor in the set theory and foundations of mathematics. It is proposed to introduce the measure of undecidability of problems unsolvable by Turing machines based on probability distribution of its input data, i.e., to provide the degree of unsolvabilty based on the number of undecidable instances of input data versus decidable ones. It is proposed as well to extend the Turing's work on infinite logics and Oracle machines to a whole class of super-Turing models of computation. Next, the three new complexity classes for TM undecidable problems have been defined: U-complete (Universal complete), D-complete (Diagonalization complete) and H-complete (Hypercomputation complete) classes. The above has never been defined explicitly before by other scientists, and has been inspired by Cook/Levin NP-complete class for intractable problems. Finally, an equivalent to famous P is not equal to NP unanswered question for NP-complete class, has been answered negatively for U-complete class of complexity for undecidable problems. 

---
# LASQ: A Low-resource Aspect-based Sentiment Quadruple Extraction Dataset 

**Authors**: Aizihaierjiang Yusufu, Jiang Liu, Kamran Aziz, Abidan Ainiwaer, Bobo Li, Fei Li, Donghong Ji, Aizierguli Yusufu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10417)  

**Abstract**: In recent years, aspect-based sentiment analysis (ABSA) has made rapid progress and shown strong practical value. However, existing research and benchmarks are largely concentrated on high-resource languages, leaving fine-grained sentiment extraction in low-resource languages under-explored. To address this gap, we constructed the first Low-resource languages Aspect-based Sentiment Quadruple dataset, named LASQ, which includes two low-resource languages: Uzbek and Uyghur. Secondly, it includes a fine-grained target-aspect-opinion-sentiment quadruple extraction task. To facilitate future research, we designed a grid-tagging model that integrates syntactic knowledge. This model incorporates part-of-speech (POS) and dependency knowledge into the model through our designed Syntax Knowledge Embedding Module (SKEM), thereby alleviating the lexical sparsity problem caused by agglutinative languages. Experiments on LASQ demonstrate consistent gains over competitive baselines, validating both the dataset's utility and the effectiveness of the proposed modeling approach. 

---
# NameBERT: Scaling Name-Based Nationality Classification with LLM-Augmented Open Academic Data 

**Authors**: Cong Ming, Ruixin Shi, Yifan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10401)  

**Abstract**: Inferring nationality from personal names is a critical capability for equity and bias monitoring, personalization, and a valuable tool in biomedical and sociological research. However, existing name-based nationality classifiers are typically trained on relatively small or source-specific labeled datasets, which can introduce coverage gaps and limit performance for underrepresented countries. While large language models (LLMs) demonstrate strong zero-shot performance for name-based nationality prediction, their computational cost and latency make them impractical for real-time, large-scale deployment. In this work, we created a large-scale name-nationality dataset from the Open Academic Graph (OAG) and introduce a framework that leverages LLMs as dataset enrichers rather than inference engines. We augment low-resource countries with LLM-generated names and evaluate on real and synthetic-tail test sets. We find that augmentation produces large gains when evaluation includes synthetic tail names and still offers a modest lift on tail-country metrics otherwise. Overall, NameBERT models achieve significantly higher accuracy than state-of-the-art baselines across both in- and out-of-domain tasks, while remaining efficient for large-scale inference compared to LLMs. 

---
# BLUEmed: Retrieval-Augmented Multi-Agent Debate for Clinical Error Detection 

**Authors**: Saukun Thika You, Nguyen Anh Khoa Tran, Wesley K. Marizane, Hanshu Rao, Qiunan Zhang, Xiaolei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10389)  

**Abstract**: Terminology substitution errors in clinical notes, where one medical term is replaced by a linguistically valid but clinically different term, pose a persistent challenge for automated error detection in healthcare. We introduce BLUEmed, a multi-agent debate framework augmented with hybrid Retrieval-Augmented Generation (RAG) that combines evidence-grounded reasoning with multi-perspective verification for clinical error detection. BLUEmed decomposes each clinical note into focused sub-queries, retrieves source-partitioned evidence through dense, sparse, and online retrieval, and assigns two domain expert agents distinct knowledge bases to produce independent analyses; when the experts disagree, a structured counter-argumentation round and cross-source adjudication resolve the conflict, followed by a cascading safety layer that filters common false-positive patterns. We evaluate BLUEmed on a clinical terminology substitution detection benchmark under both zero-shot and few-shot prompting with multiple backbone models spanning proprietary and open-source families. Experimental results show that BLUEmed achieves the best accuracy (69.13%), ROC-AUC (74.45%), and PR-AUC (72.44%) under few-shot prompting, outperforming both single-agent RAG and debate-only baselines. Further analyses across six backbone models and two prompting strategies confirm that retrieval augmentation and structured debate are complementary, and that the framework benefits most from models with sufficient instruction-following and clinical language understanding. 

---
# A Structured Clustering Approach for Inducing Media Narratives 

**Authors**: Rohan Das, Advait Deshmukh, Alexandria Leto, Zohar Naaman, I-Ta Lee, Maria Leonor Pacheco  

**Link**: [PDF](https://arxiv.org/pdf/2604.10368)  

**Abstract**: Media narratives wield tremendous power in shaping public opinion, yet computational approaches struggle to capture the nuanced storytelling structures that communication theory emphasizes as central to how meaning is constructed. Existing approaches either miss subtle narrative patterns through coarse-grained analysis or require domain-specific taxonomies that limit scalability. To bridge this gap, we present a framework for inducing rich narrative schemas by jointly modeling events and characters via structured clustering. Our approach produces explainable narrative schemas that align with established framing theory while scaling to large corpora without exhaustive manual annotation. 

---
# Adaptive Multi-Expert Reasoning via Difficulty-Aware Routing and Uncertainty-Guided Aggregation 

**Authors**: Mohamed Ehab, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2604.10335)  

**Abstract**: Large language models (LLMs) demonstrate strong performance in math reasoning benchmarks, but their performance varies inconsistently across problems with varying levels of difficulty. This paper describes Adaptive Multi-Expert Reasoning (AMR), a framework that focuses on problem complexity by reasoning with dynamically adapted strategies. An agile routing system that focuses on problem text predicts problems' difficulty and uncertainty and guides a reconfigurable sampling mechanism to manage the breadth of generation. Three specialized experts create candidate responses, which are modified during multiple correction and finalization phases. A neural verifier assesses the correctness of responses, while a clustering-based aggregation technique identifies the final candidate answer based on a combination of consensus and answer quality. When evaluated on the GSM8K dataset, AMR achieved 75.28% accuracy while only using the original training data. This result outperformed the majority of comparable 7B models that were trained on synthetic data. This showcases that models using difficulty-based routing and uncertainty-driven aggregation are efficient and effective in improving math reasoning models' robustness. 

---
# Comparative Analysis of Large Language Models in Healthcare 

**Authors**: Subin Santhosh, Farwa Abbas, Hussain Ahmad, Claudia Szabo  

**Link**: [PDF](https://arxiv.org/pdf/2604.10316)  

**Abstract**: Background: Large Language Models (LLMs) are transforming artificial intelligence applications in healthcare due to their ability to understand, generate, and summarize complex medical text. They offer valuable support to clinicians, researchers, and patients, yet their deployment in high-stakes clinical environments raises critical concerns regarding accuracy, reliability, and patient safety. Despite substantial attention in recent years, standardized benchmarking of LLMs for medical applications has been limited. Objective: This study addresses the need for a standardized comparative evaluation of LLMs in medical settings. Method: We evaluate multiple models, including ChatGPT, LLaMA, Grok, Gemini, and ChatDoctor, on core medical tasks such as patient note summarization and medical question answering, using the open-access datasets, MedMCQA, PubMedQA, and Asclepius, and assess performance through a combination of linguistic and task-specific metrics. Results: The results indicate that domain-specific models, such as ChatDoctor, excel in contextual reliability, producing medically accurate and semantically aligned text, whereas general-purpose models like Grok and LLaMA perform better in structured question-answering tasks, demonstrating higher quantitative accuracy. This highlights the complementary strengths of domain-specific and general-purpose LLMs depending on the medical task. Conclusion: Our findings suggest that LLMs can meaningfully support medical professionals and enhance clinical decision-making; however, their safe and effective deployment requires adherence to ethical standards, contextual accuracy, and human oversight in relevant cases. These results underscore the importance of task-specific evaluation and cautious integration of LLMs into healthcare workflows. 

---
# CodeComp: Structural KV Cache Compression for Agentic Coding 

**Authors**: Qiujiang Chen, Jing Xiong, Chenyang Zhao, Sidi Yang, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2604.10235)  

**Abstract**: Agentic code tasks such as fault localization and patch generation require processing long codebases under tight memory constraints, where the Key-Value (KV) cache becomes the primary inference bottleneck. Existing compression methods rely exclusively on attention signals to estimate token importance, systematically discarding structurally critical tokens such as call sites, branch conditions, and assignments that are essential for code understanding. We present CodeComp, a training-free KV cache compression framework that incorporates static program analysis into LLM inference via Code Property Graph priors extracted by Joern. Across bug localization and code generation benchmarks, CodeComp consistently outperforms attention-only compression baselines under equal memory budgets, recovering the majority of full-context accuracy under aggressive KV cache compression, while matching the patch generation quality of uncompressed full-context inference and integrating seamlessly into SGLang-based agentic coding pipelines without model modification. 

---
# Relational Probing: LM-to-Graph Adaptation for Financial Prediction 

**Authors**: Yingjie Niu, Changhong Jin, Rian Dolphin, Ruihai Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.10212)  

**Abstract**: Language models can be used to identify relationships between financial entities in text. However, while structured output mechanisms exist, prompting-based pipelines still incur autoregressive decoding costs and decouple graph construction from downstream optimization. We propose \emph{Relational Probing}, which replaces the standard language-model head with a relation head that induces a relational graph directly from language-model hidden states and is trained jointly with the downstream task model for stock-trend prediction. This approach both learns semantic representations and preserves the strict structure of the induced relational graph. It enables language-model outputs to go beyond text, allowing them to be reshaped into task-specific formats for downstream models. To enhance reproducibility, we provide an operational definition of small language models (SLMs): models that can be fine-tuned end-to-end on a single 24GB GPU under specified batch-size and sequence-length settings. Experiments use Qwen3 backbones (0.6B/1.7B/4B) as upstream SLMs and compare against a co-occurrence baseline. Relational Probing yields consistent performance improvements at competitive inference cost. 

---
# FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness 

**Authors**: Xiaoning Dong, Chengyan Wu, Yajie Wen, Yu Chen, Yun Xue, Jing Zhang, Wei Xu, Bolei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.10189)  

**Abstract**: Large Language Models (LLMs) can generate factually inaccurate content even if they have corresponding knowledge, which critically undermines their reliability. Existing approaches attempt to mitigate this by incorporating uncertainty in QA prompt during training, but these numerical scores lack the semantic richness for LLM to properly understand its internal states of trustworthiness and honestness, leading to insufficient factuality alignment. We introduce FAITH (Factuality Alignment through Integrating Trustworthiness and Honestness), a post-training framework for factuality alignment that integrates natural-language uncertainty signals with external knowledge. Specifically, we augment training datasets by computing confidence scores and semantic entropy from LLM outputs and mapping them into a knowledge state quadrant that describes the model's internal knowledge possession (trustworthiness) and answering behaviors (honestness) in natural language. Based on this enhanced data, we design a reward function that considers both correctness and uncertainty signals, and fine-tune the LLM using the Proximal Policy Optimization (PPO) algorithm. To further mitigate weakly grounded responses, we design a retrieval-augmented module that retrieves relevant external passages, improving the consistency between internal and external knowledge representations. Extensive experiments on four knowledge-intensive benchmarks demonstrate that FAITH enhances the factual accuracy and truthfulness of LLMs. 

---
# ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification 

**Authors**: Zhensheng Wang, ZhanTeng Lin, Wenmian Yang, Kun Zhou, Yiquan Zhang, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.10159)  

**Abstract**: The advancement of large language models (LLMs) has enhanced tabular question answering (Tabular QA), yet they struggle with open-domain queries exhibiting underspecified or uncertain expressions. To address this, we introduce the ODUTQA-MDC task and the first comprehensive benchmark to tackle it. This benchmark includes: (1) a large-scale ODUTQA dataset with 209 tables and 25,105 QA pairs; (2) a fine-grained labeling scheme for detailed evaluation; and (3) a dynamic clarification interface that simulates user feedback for interactive assessment. We also propose MAIC-TQA, a multi-agent framework that excels at detecting ambiguities, clarifying them through dialogue, and refining answers. Experiments validate our benchmark and framework, establishing them as a key resource for advancing conversational, underspecification-aware Tabular QA research. 

---
# Nationality encoding in language model hidden states: Probing culturally differentiated representations in persona-conditioned academic text 

**Authors**: Paul Jackson, Ruizhe Li, Elspeth Edelstein  

**Link**: [PDF](https://arxiv.org/pdf/2604.10151)  

**Abstract**: Large language models are increasingly used as writing tools and pedagogical resources in English for Academic Purposes, but it remains unclear whether they encode culturally differentiated representations when generating academic text. This study tests whether Gemma-3-4b-it encodes nationality-discriminative information in hidden states when generating research article introductions conditioned by British and Chinese academic personas. A corpus of 270 texts was generated from 45 prompt templates crossed with six persona conditions in a 2 x 3 design. Logistic regression probes were trained on hidden-state activations across all 35 layers, with shuffled-label baselines, a surface-text skyline classifier, cross-family tests, and sentence-level baselines used as controls. Probe-selected token positions were annotated for structural, lexical, and stance features using the Stanza NLP pipeline. The nationality probe reached 0.968 cross-validated accuracy at Layer 18, with perfect held-out classification. Nationality encoding followed a non-monotonic trajectory across layers, with structural effects strongest in the middle to upper network and lexical-domain effects peaking earlier. At high-signal token positions, British-associated patterns showed more postmodification, hedging, boosting, passive voice, and evaluative or process-oriented vocabulary, while Chinese-associated patterns showed more premodification, nominal predicates, and sociocultural or internationalisation vocabulary. However, sentence-level analysis found no significant nationality differences in the full generated surface text. The findings extend probing methodology to a sociolinguistic attribute and have practical implications for EAP and language pedagogy. 

---
# Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities 

**Authors**: Zhichen Liu, Yongyuan Li, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10135)  

**Abstract**: Researchers have explored different ways to improve large language models (LLMs)' capabilities via dummy token insertion in contexts. However, existing works focus solely on the dummy tokens themselves, but fail to leverage the inherent sentence-level structure of natural language. This is a critical oversight, as LLMs acquire linguistic capabilities through exposure to human-generated texts, which are inherently structured at the sentence level. Motivated by this gap, we propose an approach that inserts delimiters at sentence boundaries in LLM inputs, which not only integrates dummy tokens into the context, but also facilitates LLMs with sentence-by-sentence processing behavior during reasoning. Two concrete methods: (1). In-context learning and (2). Supervised fine-tuning are experimented using 7B models to 600B Deepseek-V3. Our results demonstrate consistent improvements across various tasks, with notable gains of up to 7.7\% on GSM8k and 12.5\% on DROP. Furthermore, the fine-tuned LLMs can incorporate sentence awareness evidenced by their internal representations. Our work establishes a simple yet effective technique for enhancing LLM's capabilities, offering promising directions for cognitive-inspired LLM enhancement paradigm. 

---
# Training-Free Cross-Lingual Dysarthria Severity Assessment via Phonological Subspace Analysis in Self-Supervised Speech Representations 

**Authors**: Bernard Muller, Antonio Armando Ortiz Barrañón, LaVonne Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2604.10123)  

**Abstract**: Dysarthric speech severity assessment typically requires trained clinicians or supervised models built from labelled pathological speech, limiting scalability across languages and clinical settings. We present a training-free method that quantifies dysarthria severity by measuring degradation in phonological feature subspaces within frozen HuBERT representations. No supervised severity model is trained; feature directions are estimated from healthy control speech using a pretrained forced aligner. For each speaker, we extract phone-level embeddings via Montreal Forced Aligner, compute d-prime scores along phonological contrast directions (nasality, voicing, stridency, sonorance, manner, and four vowel features) derived exclusively from healthy controls, and construct a 12-dimensional phonological this http URL 890 speakers across 10 corpora, 5 languages (English, Spanish, Dutch, Mandarin, French), and 3 primary aetiologies (Parkinson's disease, cerebral palsy, ALS), we find that all five consonant d-prime features correlate significantly with clinical severity (random-effects meta-analysis rho = -0.50 to -0.56, p < 2e-4; pooled Spearman rho = -0.47 to -0.55 with bootstrap 95% CIs not crossing zero). The effect replicates within individual corpora, survives FDR correction, and remains robust to leave-one-corpus-out removal and alignment quality controls. Nasality d-prime decreases monotonically from control to severe in 6 of 7 severity-graded corpora. Mann-Whitney U tests confirm that all 12 features distinguish controls from severely dysarthric speakers (p < 0.001).The method requires no dysarthric training data and applies to any language with an existing MFA acoustic model (currently 29 languages). We release the full pipeline and phone feature configurations for six languages. 

---
# CircuitSynth: Reliable Synthetic Data Generation 

**Authors**: Zehua Cheng, Wei Dai, Jiahao Sun, Thomas Lukasiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2604.10114)  

**Abstract**: The generation of high-fidelity synthetic data is a cornerstone of modern machine learning, yet Large Language Models (LLMs) frequently suffer from hallucinations, logical inconsistencies, and mode collapse when tasked with structured generation. Existing approaches, such as prompting or retrieval-augmented generation, lack the mechanisms to balance linguistic expressivity with formal guarantees regarding validity and coverage. To address this, we propose CircuitSynth, a novel neuro-symbolic framework that decouples semantic reasoning from surface realization. By distilling the reasoning capabilities of a Teacher LLM into a Probabilistic Sentential Decision Diagram (PSDD), CircuitSynth creates a tractable semantic prior that structurally enforces hard logical constraints. Furthermore, we introduce a convex optimization mechanism to rigorously satisfy soft distributional goals. Empirical evaluations across diverse benchmarks demonstrate that CircuitSynth achieves 100% Schema Validity even in complex logic puzzles where unconstrained baselines fail (12.4%) while significantly outperforming state-of-the-art methods in rare-combination coverage. 

---
# Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry 

**Authors**: Jiang Li, Tian Lan, Shanshan Wang, Dongxing Zhang, Dianqing Lin, Guanglai Gao, Derek F. Wong, Xiangdong Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.10101)  

**Abstract**: The rapid development of large language models (LLMs) has extended text generation tasks into the literary domain. However, AI-generated literary creations has raised increasingly prominent issues of creative authenticity and ethics in literary world, making the detection of LLM-generated literary texts essential and urgent. While previous works have made significant progress in detecting AI-generated text, it has yet to address classical Chinese poetry. Due to the unique linguistic features of classical Chinese poetry, such as strict metrical regularity, a shared system of poetic imagery, and flexible syntax, distinguishing whether a poem is authored by AI presents a substantial challenge. To address these issues, we introduce ChangAn, a benchmark for detecting LLM-generated classical Chinese poetry that containing total 30,664 poems, 10,276 are human-written poems and 20,388 poems are generated by four popular LLMs. Based on ChangAn, we conducted a systematic evaluation of 12 AI detectors, investigating their performance variations across different text granularities and generation strategies. Our findings highlight the limitations of current Chinese text detectors, which fail to serve as reliable tools for detecting LLM-generated classical Chinese poetry. These results validate the effectiveness and necessity of our proposed ChangAn benchmark. Our dataset and code are available at this https URL. 

---
# SEPTQ: A Simple and Effective Post-Training Quantization Paradigm for Large Language Models 

**Authors**: Han Liu, Haotian Gao, Xiaotong Zhang, Changya Li, Feng Zhang, Wei Wang, Fenglong Ma, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10091)  

**Abstract**: Large language models (LLMs) have shown remarkable performance in various domains, but they are constrained by massive computational and storage costs. Quantization, an effective technique for compressing models to fit resource-limited devices while preserving generative quality, encompasses two primary methods: quantization aware training (QAT) and post-training quantization (PTQ). QAT involves additional retraining or fine-tuning, thus inevitably resulting in high training cost and making it unsuitable for LLMs. Consequently, PTQ has become the research hotspot in recent quantization methods. However, existing PTQ methods usually rely on various complex computation procedures and suffer from considerable performance degradation under low-bit quantization settings. To alleviate the above issues, we propose a simple and effective post-training quantization paradigm for LLMs, named SEPTQ. Specifically, SEPTQ first calculates the importance score for each element in the weight matrix and determines the quantization locations in a static global manner. Then it utilizes the mask matrix which represents the important locations to quantize and update the associated weights column-by-column until the appropriate quantized weight matrix is obtained. Compared with previous methods, SEPTQ simplifies the post-training quantization procedure into only two steps, and considers the effectiveness and efficiency simultaneously. Experimental results on various datasets across a suite of models ranging from millions to billions in different quantization bit-levels demonstrate that SEPTQ significantly outperforms other strong baselines, especially in low-bit quantization scenarios. 

---
# Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models 

**Authors**: Chao Xue, Yao Wang, Mengqiao Liu, Di Liang, Xingsheng Han, Peiyang Liu, Xianjie Wu, Chenyao Lu, Lei Jiang, Yu Lu, Haibo Shi, Shuang Liang, Minlong Peng, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2604.10079)  

**Abstract**: Supervised Fine-Tuning (SFT) is the standard approach for adapting large language models (LLMs) to downstream tasks. However, we observe a persistent failure mode: even after convergence, models often fail to correctly reproduce a subset of their own supervised training data. We refer to this behavior as the Incomplete Learning Phenomenon(ILP). This paper presents the first systematic study of ILP in LLM fine-tuning. We formalize ILP as post-training failure to internalize supervised instances and demonstrate its prevalence across multiple model families, domains, and datasets. Through controlled analyses, we identify five recurrent sources of incomplete learning: (1) missing prerequisite knowledge in the pre-trained model, (2) conflicts between SFT supervision and pre-training knowledge, (3) internal inconsistencies within SFT data, (4) left-side forgetting during sequential fine-tuning, and (5) insufficient optimization for rare or complex patterns. We introduce a diagnostic-first framework that maps unlearned samples to these causes using observable training and inference signals, and study several targeted mitigation strategies as causal interventions. Experiments on Qwen, LLaMA, and OLMo2 show that incomplete learning is widespread and heterogeneous, and that improvements in aggregate metrics can mask persistent unlearned subsets. The findings highlight the need for fine-grained diagnosis of what supervised fine-tuning fails to learn, and why. 

---
# Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty 

**Authors**: Chao Xue, Yao Wang, Mengqiao Liu, Di Liang, Xingsheng Han, Peiyang Liu, Xianjie Wu, Chenyao Lu, Lei Jiang, Yu Lu, Haibo Shi, Shuang Liang, Minlong Peng, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2604.10072)  

**Abstract**: Recent advancements in the Generative Reward Model (GRM) have demonstrated its potential to enhance the reasoning abilities of LLMs through Chain-of-Thought (CoT) prompting. Despite these gains, existing implementations of GRM suffer from two critical limitations. First, CoT prompting is applied indiscriminately to all inputs regardless of their inherent complexity. This introduces unnecessary computational costs for tasks amenable to fast, direct inference. Second, existing approaches primarily rely on voting-based mechanisms to evaluate CoT outputs, which often lack granularity and precision in assessing reasoning quality. In this paper, we propose E-GRM, an efficient generative reward modeling framework grounded in model-internal uncertainty. E-GRM leverages the convergence behavior of parallel model generations to estimate uncertainty and selectively trigger CoT reasoning only when needed, without relying on handcrafted features or task-dependent signals. To improve reward fidelity, we introduce a lightweight discriminative scorer trained with a hybrid regression--ranking objective to provide fine-grained evaluation of reasoning paths. Experiments on multiple reasoning benchmarks show that E-GRM substantially reduces inference cost while consistently improving answer accuracy, demonstrating that model-internal uncertainty is an effective and general signal for efficient reasoning-aware reward modeling. 

---
# ASPIRin: Action Space Projection for Interactivity-Optimized Reinforcement Learning in Full-Duplex Speech Language Models 

**Authors**: Chi-Yuan Hsiao, Ke-Han Lu, Yu-Kuan Fu, Guan-Ting Lin, Hsiao-Tsung Hung, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.10065)  

**Abstract**: End-to-end full-duplex Speech Language Models (SLMs) require precise turn-taking for natural interaction. However, optimizing temporal dynamics via standard raw-token reinforcement learning (RL) degrades semantic quality, causing severe generative collapse and repetition. We propose ASPIRin, an interactivity-optimized RL framework that explicitly decouples when to speak from what to say. Using Action Space Projection, ASPIRin maps the text vocabulary into a coarse-grained binary state (active speech vs. inactive silence). By applying Group Relative Policy Optimization (GRPO) with rule-based rewards, it balances user interruption and response latency. Empirical evaluations show ASPIRin optimizes interactivity across turn-taking, backchanneling, and pause handling. Crucially, isolating timing from token selection preserves semantic coherence and reduces the portion of duplicate n-grams by over 50% compared to standard GRPO, effectively eliminating degenerative repetition. 

---
# Linguistic Accommodation Between Neurodivergent Communities on Reddit:A Communication Accommodation Theory Analysis of ADHD and Autism Groups 

**Authors**: Saad Mankarious, Nour Zein, Iyad Ait Hou, Aya Zirikly  

**Link**: [PDF](https://arxiv.org/pdf/2604.10063)  

**Abstract**: Social media research on mental health has focused predominantly on detecting and diagnosing conditions at the individual level. In this work, we shift attention to \emph{intergroup} behavior, examining how two prominent neurodivergent communities, ADHD and autism, adjust their language when engaging with each other on Reddit. Grounded in Communication Accommodation Theory (CAT), we first establish that each community maintains a distinct linguistic profile as measured by Language Inquiry and Word Count Lexicon (LIWC). We then show that these profiles shift in opposite directions when users cross community boundaries: features that are elevated in one group's home community decrease when its members post in the other group's space, and vice versa, consistent with convergent accommodation. The involvement of topic-independent summary variables (Authentic, Clout) in these shifts provides partial evidence against a purely topical explanation. Finally, in an exploratory longitudinal analysis around the moment of public diagnosis disclosure, we find that its effects on linguistic style are small and, in some cases, directionally opposite to cross-community accommodation, providing initial evidence that situational audience adaptation and longer-term identity processes may involve different mechanisms. Our findings contribute to understanding intergroup communication dynamics among neurodivergent populations online and carry implications for community moderation and clinical perspectives on these conditions. 

---
# Computational Implementation of a Model of Category-Theoretic Metaphor Comprehension 

**Authors**: Fumitaka Iwaki, Miho Fuyama, Hayato Saigo, Tatsuji Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2604.10035)  

**Abstract**: In this study, we developed a computational implementation for a model of metaphor comprehension based on the theory of indeterminate natural transformation (TINT) proposed by Fuyama et al. We simplified the algorithms implementing the model to be closer to the original theory and verified it through data fitting and simulations. The outputs of the algorithms are evaluated with three measures: data-fitting with experimental data, the systematicity of the metaphor comprehension result, and the novelty of the comprehension (i.e. the correspondence of the associative structure of the source and target of the metaphor). The improved algorithm outperformed the existing ones in all the three measures. 

---
# CoSToM:Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models 

**Authors**: Mengfan Li, Xuanhua Shi, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.10031)  

**Abstract**: Theory of Mind (ToM), the ability to attribute mental states to others, is a hallmark of social intelligence. While large language models (LLMs) demonstrate promising performance on standard ToM benchmarks, we observe that they often fail to generalize to complex task-specific scenarios, relying heavily on prompt scaffolding to mimic reasoning. The critical misalignment between the internal knowledge and external behavior raises a fundamental question: Do LLMs truly possess intrinsic cognition, and can they externalize this internal knowledge into stable, high-quality behaviors? To answer this, we introduce CoSToM (Causal-oriented Steering for ToM alignment), a framework that transitions from mechanistic interpretation to active intervention. First, we employ causal tracing to map the internal distribution of ToM features, empirically uncovering the internal layers' characteristics in encoding fundamental ToM semantics. Building on this insight, we implement a lightweight alignment framework via targeted activation steering within these ToM-critical layers. Experiments demonstrate that CoSToM significantly enhances human-like social reasoning capabilities and downstream dialogue quality. 

---
# Weird Generalization is Weirdly Brittle 

**Authors**: Miriam Wanner, Hannah Collison, William Jurayj, Benjamin Van Durme, Mark Dredze, William Walden  

**Link**: [PDF](https://arxiv.org/pdf/2604.10022)  

**Abstract**: Weird generalization is a phenomenon in which models fine-tuned on data from a narrow domain (e.g. insecure code) develop surprising traits that manifest even outside that domain (e.g. broad misalignment)-a phenomenon that prior work has highlighted as a critical safety concern. Here, we present an extended replication study of key weird generalization results across an expanded suite of models and datasets. We confirm that surprising (and dangerous) traits can emerge under certain circumstances, but we find that weird generalization is exceptionally brittle: it emerges only for specific models on specific datasets, and it vanishes under simple training-time, prompt-based interventions. We find that the most effective interventions provide prompt context that makes the generalized behavior the expected behavior. However, we show that even very generic interventions that do not anticipate specific generalized traits can still be effective in mitigating weird generalization's effects. Our findings thus help clarify the nature of the safety threat that weird generalization poses and point toward an easily implemented set of solutions. 

---
# Human vs. Machine Deception: Distinguishing AI-Generated and Human-Written Fake News Using Ensemble Learning 

**Authors**: Samuel Jaeger, Calvin Ibeneye, Aya Vera-Jimenez, Dhrubajyoti Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2604.09960)  

**Abstract**: The rapid adoption of large language models has introduced a new class of AI-generated fake news that coexists with traditional human-written misinformation, raising important questions about how these two forms of deceptive content differ and how reliably they can be distinguished. This study examines linguistic, structural, and emotional differences between human-written and AI-generated fake news and evaluates machine learning and ensemble-based methods for distinguishing these content types. A document-level feature representation is constructed using sentence structure, lexical diversity, punctuation patterns, readability indices, and emotion-based features capturing affective dimensions such as fear, anger, joy, sadness, trust, and anticipation. Multiple classification models, including logistic regression, random forest, support vector machines, extreme gradient boosting, and a neural network, are applied alongside an ensemble framework that aggregates predictions across models. Model performance is assessed using accuracy and area under the receiver operating characteristic curve. The results show strong and consistent classification performance, with readability-based features emerging as the most informative predictors and AI-generated text exhibiting more uniform stylistic patterns. Ensemble learning provides modest but consistent improvements over individual models. These findings indicate that stylistic and structural properties of text provide a robust basis for distinguishing AI-generated misinformation from human-written fake news. 

---
# Should We be Pedantic About Reasoning Errors in Machine Translation? 

**Authors**: Calvin Bao, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2604.09890)  

**Abstract**: Across multiple language pairings (English $\to$ \{Spanish, French, German, Mandarin, Japanese, Urdu, Cantonese\}), we find reasoning errors in translation. To quantify how often these reasoning errors occur, we leverage an automated annotation protocol for reasoning evaluation wherein the goal is to detect if a reasoning step is any of three error categories: (1) source sentence-misaligned, (2) model hypothesis-misaligned, or (3) reasoning trace-misaligned. We probe the reasoning model with perturbed traces correcting for these identified reasoning errors using an array of weak-to-strong interventions: hedging, removal, re-reasoning after removal, hindsight, and oracle interventions. Experimenting with interventions on the reasoning traces suggests that small corrections to the reasoning have little impact on translation quality, but stronger interventions yield the highest resolution rates, despite translation quality gains being mixed. We find ultimately that reasoning errors in MT can be identified with high precision in Urdu but lower precision in Spanish, but that removing these reasoning errors does not resolve the initial errors significantly, suggesting limited reasoning faithfulness for machine translation. 

---
# Simulating Organized Group Behavior: New Framework, Benchmark, and Analysis 

**Authors**: Xinkai Zou, Yiming Huang, Zhuohang Wu, Jian Sha, Nan Huang, Longfei Yun, Jingbo Shang, Letian Peng  

**Link**: [PDF](https://arxiv.org/pdf/2604.09874)  

**Abstract**: Simulating how organized groups (e.g., corporations) make decisions (e.g., responding to a competitor's move) is essential for understanding real-world dynamics and could benefit relevant applications (e.g., market prediction). In this paper, we formalize this problem as a concrete research platform for group behavior understanding, providing: (1) a task definition with benchmark and evaluation criteria, (2) a structured analytical framework with a corresponding algorithm, and (3) detailed temporal and cross-group analysis. Specifically, we propose Organized Group Behavior Simulation, a task that models organized groups as collective entities from a practical perspective: given a group facing a particular situation (e.g., AI Boom), predict the decision it would take. To support this task, we present GROVE (GRoup Organizational BehaVior Evaluation), a benchmark covering 44 entities with 8,052 real-world context-decision pairs collected from Wikipedia and TechCrunch across 9 domains, with an end-to-end evaluation protocol assessing consistency, initiative, scope, magnitude, and horizon. Beyond straightforward prompting pipelines, we propose a structured analytical framework that converts collective decision-making events into an interpretable, adaptive, and traceable behavioral model, achieving stronger performance than summarization- and retrieval-based baselines. It further introduces an adapter mechanism for time-aware evolution and group-aware transfer, and traceable evidence nodes grounding each decision rule in originating historical events. Our analysis reveals temporal behavioral drift within individual groups, which the time-aware adapter effectively captures for stronger prediction, and structured cross-group similarity that enables knowledge transfer for data-scarce organizations. 

---
# Spoiler Alert: Narrative Forecasting as a Metric for Tension in LLM Storytelling 

**Authors**: Peiqi Sui, Yutong Zhu, Tianyi Cheng, Peter West, Richard Jean So, Hoyt Long, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2604.09854)  

**Abstract**: LLMs have so far failed both to generate consistently compelling stories and to recognize this failure--on the leading creative-writing benchmark (EQ-Bench), LLM judges rank zero-shot AI stories above New Yorker short stories, a gold standard for literary fiction. We argue that existing rubrics overlook a key dimension of compelling human stories: narrative tension. We introduce the 100-Endings metric, which walks through a story sentence by sentence: at each position, a model predicts how the story will end 100 times given only the text so far, and we measure tension as how often predictions fail to match the ground truth. Beyond the mismatch rate, the sentence-level curve yields complementary statistics, such as inflection rate, a geometric measure of how frequently the curve reverses direction, tracking twists and revelations. Unlike rubric-based judges, 100-Endings correctly ranks New Yorker stories far above LLM outputs. Grounded in narratological principles, we design a story-generation pipeline using structural constraints, including analysis of story templates, idea formulation, and narrative scaffolding. Our pipeline significantly increases narrative tension as measured by the 100-Endings metric, while maintaining performance on the EQ-Bench leaderboard. 

---
# Claim2Vec: Embedding Fact-Check Claims for Multilingual Similarity and Clustering 

**Authors**: Rrubaa Panchendrarajan, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2604.09812)  

**Abstract**: Recurrent claims present a major challenge for automated fact-checking systems designed to combat misinformation, especially in multilingual settings. While tasks such as claim matching and fact-checked claim retrieval aim to address this problem by linking claim pairs, the broader challenge of effectively representing groups of similar claims that can be resolved with the same fact-check via claim clustering remains relatively underexplored. To address this gap, we introduce Claim2Vec, the first multilingual embedding model optimized to represent fact-check claims as vectors in an improved semantic embedding space. We fine-tune a multilingual encoder using contrastive learning with similar multilingual claim pairs. Experiments on the claim clustering task using three datasets, 14 multilingual embedding models, and 7 clustering algorithms demonstrate that Claim2Vec significantly improves clustering performance. Specifically, it enhances both cluster label alignment and the geometric structure of the embedding space across different cluster configurations. Our multilingual analysis shows that clusters containing multiple languages benefit from fine-tuning, demonstrating cross-lingual knowledge transfer. 

---
# GIANTS: Generative Insight Anticipation from Scientific Literature 

**Authors**: Joy He-Yueya, Anikait Singh, Ge Gao, Michael Y. Li, Sherry Yang, Chelsea Finn, Emma Brunskill, Noah D. Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2604.09793)  

**Abstract**: Scientific breakthroughs often emerge from synthesizing prior ideas into novel contributions. While language models (LMs) show promise in scientific discovery, their ability to perform this targeted, literature-grounded synthesis remains underexplored. We introduce insight anticipation, a generation task in which a model predicts a downstream paper's core insight from its foundational parent papers. To evaluate this capability, we develop GiantsBench, a benchmark of 17k examples across eight scientific domains, where each example consists of a set of parent papers paired with the core insight of a downstream paper. We evaluate models using an LM judge that scores similarity between generated and ground-truth insights, and show that these similarity scores correlate with expert human ratings. Finally, we present GIANTS-4B, an LM trained via reinforcement learning (RL) to optimize insight anticipation using these similarity scores as a proxy reward. Despite its smaller open-source architecture, GIANTS-4B outperforms proprietary baselines and generalizes to unseen domains, achieving a 34% relative improvement in similarity score over gemini-3-pro. Human evaluations further show that GIANTS-4B produces insights that are more conceptually clear than those of the base model. In addition, SciJudge-30B, a third-party model trained to compare research abstracts by likely citation impact, predicts that insights generated by GIANTS-4B are more likely to lead to higher citations, preferring them over the base model in 68% of pairwise comparisons. We release our code, benchmark, and model to support future research in automated scientific discovery. 

---
# Generating High Quality Synthetic Data for Dutch Medical Conversations 

**Authors**: Cecilia Kuan, Aditya Kamlesh Parikh, Henk van den Heuvel  

**Link**: [PDF](https://arxiv.org/pdf/2604.09645)  

**Abstract**: Medical conversations offer insights into clinical communication often absent from Electronic Health Records. However, developing reliable clinical Natural Language Processing (NLP) models is hampered by the scarcity of domain-specific datasets, as clinical data are typically inaccessible due to privacy and ethical constraints. To address these challenges, we present a pipeline for generating synthetic Dutch medical dialogues using a Dutch fine-tuned Large Language Model, with real medical conversations serving as linguistic and structural reference. The generated dialogues were evaluated through quantitative metrics and qualitative review by native speakers and medical practitioners. Quantitative analysis revealed strong lexical variety and overly regular turn-taking, suggesting scripted rather than natural conversation flow. Qualitative review produced slightly below-average scores, with raters noting issues in domain specificity and natural expression. The limited correlation between quantitative and qualitative results highlights that numerical metrics alone cannot fully capture linguistic quality. Our findings demonstrate that generating synthetic Dutch medical dialogues is feasible but requires domain knowledge and carefully structured prompting to balance naturalness and structure in conversation. This work provides a foundation for expanding Dutch clinical NLP resources through ethically generated synthetic data. 

---
# HumorGen: Cognitive Synergy for Humor Generation in Large Language Models via Persona-Based Distillation 

**Authors**: Edward Ajayi, Prasenjit Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2604.09629)  

**Abstract**: Humor generation poses a significant challenge for Large Language Models (LLMs), because their standard training objective - predicting the most likely next word - inherently conflicts with the surprise and incongruity needed for comedy. To bridge this gap, we introduce the Cognitive Synergy Framework, a theoretically grounded methodology for generating high-quality humor data inspired by psychological theories of humor. Utilizing a Mixture-of-Thought (MoT) approach, we deploy six cognitive personas (e.g., The Absurdist, The Cynic) to synthesize diverse comedic perspectives for a given prompt. This framework creates a theoretically grounded dataset, which we use to fine-tune a 7B-parameter student model. We compare Direct Preference Optimization (DPO) and a novel Offline Group Relative Policy Optimization (O-GRPO); our 7B model significantly outperforms larger instruction-tuned baselines and achieves performance competitive with state-of-the-art proprietary models. We find that cognitive-driven data curation is far more critical than alignment algorithms or model scale for humor generation. Code and data will be available upon publication. 

---
# Toward Generalized Cross-Lingual Hateful Language Detection with Web-Scale Data and Ensemble LLM Annotations 

**Authors**: Dang H. Dang, Jelena Mitrovi, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2604.09625)  

**Abstract**: We study whether large-scale unlabelled web data and LLM-based synthetic annotations can improve multilingual hate speech detection. Starting from texts crawled via this http URL~(OWS) in four languages (English, German, Spanish, Vietnamese), we pursue two complementary strategies. First, we apply continued pre-training to BERT models by continuing masked language modelling on unlabelled OWS texts before supervised fine-tuning, and show that this yields an average macro-F1 gain of approximately 3% over standard baselines across sixteen benchmarks, with stronger gains in low-resource settings. Second, we use four open-source LLMs (Mistral-7B, Llama3.1-8B, Gemma2-9B, Qwen2.5-14B) to produce synthetic annotations through three ensemble strategies: mean averaging, majority voting, and a LightGBM meta-learner. The LightGBM ensemble consistently outperforms the other strategies. Fine-tuning on these synthetic labels substantially benefits a small model (Llama3.2-1B: +11% pooled F1), but provides only a modest gain for the larger Qwen2.5-14B (+0.6%). Our results indicate that the combination of web-scale unlabelled data and LLM-ensemble annotations is the most valuable for smaller models and low-resource languages. 

---
# Self-Calibrating Language Models via Test-Time Discriminative Distillation 

**Authors**: Mohamed Rissal Hedna, Jan Strich, Martin Semmann, Chris Biemann  

**Link**: [PDF](https://arxiv.org/pdf/2604.09624)  

**Abstract**: Large language models (LLMs) are systematically overconfident: they routinely express high certainty on questions they often answer incorrectly. Existing calibration methods either require labeled validation data, degrade under distribution shifts, or incur substantial inference costs. Recent work has shown that LLMs already contain a better-calibrated signal than the one they verbalize: the token probability of "True" when the model is asked "Is this answer correct?" ($P(\text{True})$) consistently outperforms their stated confidence, a gap that is theoretically grounded as generative error is lower-bounded by roughly twice the corresponding discriminative error. We introduce $\textbf{SECL}$ ($\textbf{SE}$lf-$\textbf{C}$alibrating $\textbf{L}$anguage Models), a test-time training (TTT) pipeline that exploits this gap as label-free self-supervision, requiring no labeled data or human supervision. SECL adapts only when the input distribution shifts, training on just 6--26% of the question stream at lower cost than the baseline it distills from. Across four small language models from three model families and four diverse domains, SECL reduces Expected Calibration Error (ECE) by 56--78%, outperforming its own supervision signal and matching or outperforming recent inference-time methods. SECL is the first method to apply TTT to calibration; seven ablations covering signal quality, gating strategy, weight accumulation, loss design, domain ordering, hyperparameter sensitivity, and layer selection confirm that each component is crucial and robust across configurations. Code: this https URL 

---
# Detecting Safety Violations Across Many Agent Traces 

**Authors**: Adam Stein, Davis Brown, Hamed Hassani, Mayur Naik, Eric Wong  

**Link**: [PDF](https://arxiv.org/pdf/2604.11806)  

**Abstract**: To identify safety violations, auditors often search over large sets of agent traces. This search is difficult because failures are often rare, complex, and sometimes even adversarially hidden and only detectable when multiple traces are analyzed together. These challenges arise in diverse settings such as misuse campaigns, covert sabotage, reward hacking, and prompt injection. Existing approaches struggle here for several reasons. Per-trace judges miss failures that only become visible across traces, naive agentic auditing does not scale to large trace collections, and fixed monitors are brittle to unanticipated behaviors. We introduce Meerkat, which combines clustering with agentic search to uncover violations specified in natural language. Through structured search and adaptive investigation of promising regions, Meerkat finds sparse failures without relying on seed scenarios, fixed workflows, or exhaustive enumeration. Across misuse, misalignment, and task gaming settings, Meerkat significantly improves detection of safety violations over baseline monitors, discovers widespread developer cheating on a top agent benchmark, and finds nearly 4x more examples of reward hacking on CyBench than previous audits. 

---
# ClawGUI: A Unified Framework for Training, Evaluating, and Deploying GUI Agents 

**Authors**: Fei Tang, Zhiqiong Lu, Boxuan Zhang, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11784)  

**Abstract**: GUI agents drive applications through their visual interfaces instead of programmatic APIs, interacting with arbitrary software via taps, swipes, and keystrokes, reaching a long tail of applications that CLI-based agents cannot. Yet progress in this area is bottlenecked less by modeling capacity than by the absence of a coherent full-stack infrastructure: online RL training suffers from environment instability and closed pipelines, evaluation protocols drift silently across works, and trained agents rarely reach real users on real devices. We present \textbf{ClawGUI}, an open-source framework addressing these three gaps within a single harness. \textbf{ClawGUI-RL} provides the first open-source GUI agent RL infrastructure with validated support for both parallel virtual environments and real physical devices, integrating GiGPO with a Process Reward Model for dense step-level supervision. \textbf{ClawGUI-Eval} enforces a fully standardized evaluation pipeline across 6 benchmarks and 11+ models, achieving 95.8\% reproduction against official baselines. \textbf{ClawGUI-Agent} brings trained agents to Android, HarmonyOS, and iOS through 12+ chat platforms with hybrid CLI-GUI control and persistent personalized memory. Trained end to end within this pipeline, \textbf{ClawGUI-2B} achieves 17.1\% Success Rate on MobileWorld GUI-Only, outperforming the same-scale MAI-UI-2B baseline by 6.0\%. 

---
# SWE-AGILE: A Software Agent Framework for Efficiently Managing Dynamic Reasoning Context 

**Authors**: Shuquan Lian, Juncheng Liu, Yazhe Chen, Yuhong Chen, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11716)  

**Abstract**: Prior representative ReAct-style approaches in autonomous Software Engineering (SWE) typically lack the explicit System-2 reasoning required for deep analysis and handling complex edge cases. While recent reasoning models demonstrate the potential of extended Chain-of-Thought (CoT), applying them to the multi-turn SWE task creates a fundamental dilemma: retaining full reasoning history leads to context explosion and ``Lost-in-the-Middle'' degradation, while discarding it would force the agent to redundantly re-reason at every step. To address these challenges, we propose SWE-AGILE, a novel software agent framework designed to bridge the gap between reasoning depth, efficiency, and context constraints. SWE-AGILE introduces a Dynamic Reasoning Context strategy, maintaining a ``sliding window'' of detailed reasoning for immediate continuity to prevent redundant re-analyzing, while compressing historical reasoning content into concise Reasoning Digests. Empirically, SWE-AGILE sets a new standard for 7B-8B models on SWE-Bench-Verified using only 2.2k trajectories and 896 tasks. Code is available at this https URL. 

---
# Agentic Driving Coach: Robustness and Determinism of Agentic AI-Powered Human-in-the-Loop Cyber-Physical Systems 

**Authors**: Deeksha Prahlad, Daniel Fan, Hokeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.11705)  

**Abstract**: Foundation models, including large language models (LLMs), are increasingly used for human-in-the-loop (HITL) cyber-physical systems (CPS) because foundation model-based AI agents can potentially interact with both the physical environments and human users. However, the unpredictable behavior of human users and AI agents, in addition to the dynamically changing physical environments, leads to uncontrollable nondeterminism. To address this urgent challenge of enabling agentic AI-powered HITL CPS, we propose a reactor-model-of-computation (MoC)-based approach, realized by the open-source Lingua Franca (LF) framework. We also carry out a concrete case study using the agentic driving coach as an application of HITL CPS. By evaluating the LF-based agentic HITL CPS, we identify practical challenges in reintroducing determinism into such agentic HITL CPS and present pathways to address them. 

---
# MimicLM: Zero-Shot Voice Imitation through Autoregressive Modeling of Pseudo-Parallel Speech Corpora 

**Authors**: Tao Feng, Yuxiang Wang, Yuancheng Wang, Xueyao Zhang, Dekun Chen, Chaoren Wang, Xun Guan, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11552)  

**Abstract**: Voice imitation aims to transform source speech to match a reference speaker's timbre and speaking style while preserving linguistic content. A straightforward approach is to train on triplets of (source, reference, target), where source and target share the same content but target matches the reference's voice characteristics, yet such data is extremely scarce. Existing approaches either employ carefully designed disentanglement architectures to bypass this data scarcity or leverage external systems to synthesize pseudo-parallel training data. However, the former requires intricate model design, and the latter faces a quality ceiling when synthetic speech is used as training targets. To address these limitations, we propose MimicLM, which takes a novel approach by using synthetic speech as training sources while retaining real recordings as targets. This design enables the model to learn directly from real speech distributions, breaking the synthetic quality ceiling. Building on this data construction approach, we incorporate interleaved text-audio modeling to guide the generation of content-accurate speech and apply post-training with preference alignment to mitigate the inherent distributional mismatch when training on synthetic data. Experiments demonstrate that MimicLM achieves superior voice imitation quality with a simple yet effective architecture, significantly outperforming existing methods in naturalness while maintaining competitive similarity scores across speaker identity, accent, and emotion dimensions. 

---
# Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach 

**Authors**: Haolin Li, Shuyang Jiang, Ruipeng Zhang, Jiangchao Yao, Ya Zhang, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11547)  

**Abstract**: While large language models hold promise for complex medical applications, their development is hindered by the scarcity of high-quality reasoning data. To address this issue, existing approaches typically distill chain-of-thought reasoning traces from large proprietary models via supervised fine-tuning, then conduct reinforcement learning (RL). These methods exhibit limited improvement on underrepresented domains like rare diseases while incurring substantial costs from generating complex reasoning chains. To efficiently enhance medical reasoning, we propose MedSSR, a Medical Knowledge-enhanced data Synthesis and Semi-supervised Reinforcement learning framework. Our framework first employs rare disease knowledge to synthesize distribution-controllable reasoning questions. We then utilize the policy model itself to generate high-quality pseudo-labels. This enables a two-stage, intrinsic-to-extrinsic training paradigm: self-supervised RL on the pseudo-labeled synthetic data, followed by supervised RL on the human-annotated real data. MedSSR scales model training efficiently without relying on costly trace distillation. Extensive experiments on Qwen and Llama demonstrate that our method outperforms existing methods across ten medical benchmarks, achieving up to +5.93% gain on rare-disease tasks. Our code is available at this https URL. 

---
# DuET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode 

**Authors**: Hojae Han, Jaejin Kim, Seung-won Hwang, Yu Jin Kim, Moontae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.11514)  

**Abstract**: This work addresses test output prediction, a key challenge in test case generation. To improve the reliability of predicted outputs by LLMs, prior approaches generate code first to ground predictions. One grounding strategy is direct execution of generated code, but even minor errors can cause failures. To address this, we introduce LLM-based pseudocode execution, which grounds prediction on more error-resilient pseudocode and simulates execution via LLM reasoning. We further propose DuET, a dual-execution framework that combines both approaches by functional majority voting. Our analysis shows the two approaches are complementary in overcoming the limitations of direct execution suffering from code errors, and pseudocode reasoning from hallucination. On LiveCodeBench, DuET achieves the state-of-the-art performance, improving Pass@1 by 13.6 pp. 

---
# Quantization Dominates Rank Reduction for KV-Cache Compression 

**Authors**: Samuel Salfati  

**Link**: [PDF](https://arxiv.org/pdf/2604.11501)  

**Abstract**: We compare two strategies for compressing the KV cache in transformer inference: rank reduction (discard dimensions) and quantization (keep all dimensions, reduce precision). At matched storage budgets across five models (124M-14B, MHA and GQA), we find that quantization consistently outperforms rank reduction by 4-364 PPL depending on model and compression level. The gap persists even when rank reduction is combined with quantization in hybrid baselines, and it grows with GQA aggressiveness. On LAMBADA, INT4 matches FP16 accuracy (+0.23 PPL on Mistral 7B, +0.58 on GPT-2) while rank-32 at identical storage collapses to 0.4%.
We trace this gap to a structural asymmetry: under softmax attention routing, removing a dimension can flip which token is attended (a discrete failure), while quantization noise is bounded and typically preserves score ordering. We formalize this via a perturbation result showing projection damage exceeds quantization damage by 3 x 2^(2b) per direction under the softmax Fisher metric. A basis ablation confirms the finding is basis-independent (spread <0.4 PPL), establishing that the advantage comes from preserving dimensions, not from a better coordinate system. Joint K+V INT4 quantization achieves 75% total KV reduction at only +0.18 PPL on Mistral 7B. 

---
# Revisiting Compositionality in Dual-Encoder Vision-Language Models: The Role of Inference 

**Authors**: Imanol Miranda, Ander Salaberria, Eneko Agirre, Gorka Azkune  

**Link**: [PDF](https://arxiv.org/pdf/2604.11496)  

**Abstract**: Dual-encoder Vision-Language Models (VLMs) such as CLIP are often characterized as bag-of-words systems due to their poor performance on compositional benchmarks. We argue that this limitation may stem less from deficient representations than from the standard inference protocol based on global cosine similarity. First, through controlled diagnostic experiments, we show that explicitly enforcing fine-grained region-segment alignment at inference dramatically improves compositional performance without updating pretrained encoders. We then introduce a lightweight transformer that learns such alignments directly from frozen patch and token embeddings. Comparing against full fine-tuning and prior end-to-end compositional training methods, we find that although these approaches improve in-domain retrieval, their gains do not consistently transfer under distribution shift. In contrast, learning localized alignment over frozen representations matches full fine-tuning on in-domain retrieval while yielding substantial improvements on controlled out-of-domain compositional benchmarks. These results identify global embedding matching as a key bottleneck in dual-encoder VLMs and highlight the importance of alignment mechanisms for robust compositional generalization. 

---
# Anthropogenic Regional Adaptation in Multimodal Vision-Language Model 

**Authors**: Samuel Cahyawijaya, Peerat Limkonchotiwat, Tack Hwa Wong, Hitesh Laxmichand Patel, Amit Agarwal, Manuel Antonio Rufino, Carlos Rafael Catalan, Muhammad Reza Qorib, Vicky Feliren, Holy Lovenia, Aye Hninn Khine, Frederikus Hudi, David Anugraha, Alham Fikri Aji, Romrawin Chumpu, Viet-Thanh Pham, Minghan Wang, Mohamed Fazli Imam, Ruochen Zhang, Joseph Marvin Imperial, Do Xuan Long, Musa Izzanardi Wijanarko, Joel Ruben Antony Moniz, Patrick Amadeus Irawan, Hanif Muhammad Zhafran, Isaiah Flores, Ira Salsabila, Jun Kevin, Jostin Jerico Rosal, Patricia Nicole Monderin, Kun Kerdthaisong, Ahmad Mustafid, My Chiffon Nguyen, Natchapon Jongwiriyanurak, Siva Worajitwannakul, Haochen Li, Adrian Xuan Wei Lim, Bin Wang, Muhammad Ravi Shulthan Habibi, Lynnette Hui Xian Ng, Mithil Bangera, Yeshil Bangera, Priyaranjan Pattnayak, Dun Li Chan, Sherissa Caren Djuniwar, Hee Ming Shan  

**Link**: [PDF](https://arxiv.org/pdf/2604.11490)  

**Abstract**: While the field of vision-language (VL) has achieved remarkable success in integrating visual and textual information across multiple languages and domains, there is still no dedicated framework for assessing human-centric alignment in vision-language systems. We offer two contributions to address this gap. First, we introduce Anthropogenic Regional Adaptation: a novel paradigm that aims to optimize model relevance to specific regional contexts while ensuring the retention of global generalization capabilities. Second, we present a simple, but effective adaptation method named Geographical-generalization-made-easy (GG-EZ), which utilizes regional data filtering and model merging. Through comprehensive experiments on 3 VL architectures: large vision-language models, text-to-image diffusion models, and vision-language embedding models, and a case study in Southeast Asia (SEA) regional adaptation, we demonstrate the importance of Anthropogenic Regional Adaptation and the effectiveness of GG-EZ, showing 5-15% gains in cultural relevance metrics across SEA while maintaining over 98% of global performance and even occasionally surpassing it. Our findings establish Anthropogenic Regional Alignment as a foundational paradigm towards applicability of multimodal vision-language models in diverse regions and demonstrate a simple-yet-effective baseline method that optimizes regional value alignment while preserving global generalization. 

---
# Low-rank Optimization Trajectories Modeling for LLM RLVR Acceleration 

**Authors**: Zhipeng Chen, Tao Qian, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11446)  

**Abstract**: Recently, scaling reinforcement learning with verifiable rewards (RLVR) for large language models (LLMs) has emerged as an effective training paradigm for significantly improving model capabilities, which requires guiding the model to perform extensive exploration and learning, leading to substantial computational overhead and becoming a key challenge. To reduce the number of training steps, Prior work performs linear extrapolation of model parameters. However, the dynamics of model parameter updates during RLVR training remain insufficiently understood. To further investigate the evolution of LLMs during RLVR training, we conduct empirical experiments and find that the rank-1 subspace of the model does not evolve linearly, and its dominance over the original parameters is further amplified during LoRA training. Based on the above insights, we propose the \textbf{N}onlinear \textbf{Ext}rapolation of low-rank trajectories (\textbf{NExt}), a novel framework that models and extrapolates low-rank parameter trajectories in a nonlinear manner. Concretely, we first train the model using LoRA and extract the rank-1 subspace of parameter differences at multiple training steps, which is then used for the subsequent nonlinear extrapolation. Afterward, we utilized the extracted rank-1 subspace to train a predictor, which can model the trajectory of parameter updates during RLVR, and then perform the predict-extend process to extrapolate model parameters, achieving the acceleration of RLVR. To further study and understand NExt, we conduct comprehensive experiments that demonstrate the effectiveness and robustness of the method. Our method reduces computational overhead by approximately 37.5\% while remaining compatible with a wide range of RLVR algorithms and tasks. We release our code in this https URL. 

---
# Reasoning Resides in Layers: Restoring Temporal Reasoning in Video-Language Models with Layer-Selective Merging 

**Authors**: Zihang Fu, Haonan Wang, Jian Kang, Kenji Kawaguchi, Jiaying Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11399)  

**Abstract**: Multimodal adaptation equips large language models (LLMs) with perceptual capabilities, but often weakens the reasoning ability inherited from language-only pretraining. This trade-off is especially pronounced in video-language models (VLMs), where visual alignment can impair temporal reasoning (TR) over sequential events. We propose MERIT, a training-free, task-driven model merging framework for restoring TR in VLMs. MERIT searches over layer-wise self-attention merging recipes between a VLM and its paired text-only backbone using an objective that improves TR while penalizing degradation in temporal perception (TP). Across three representative VLMs and multiple challenging video benchmarks, MERIT consistently improves TR, preserves or improves TP, and generalizes beyond the search set to four distinct benchmarks. It also outperforms uniform full-model merging and random layer selection, showing that effective recovery depends on selecting the right layers. Interventional masking and frame-level attribution further show that the selected layers are disproportionately important for reasoning and shift model decisions toward temporally and causally relevant evidence. These results show that targeted, perception-aware model merging can effectively restore TR in VLMs without retraining. 

---
# What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment? 

**Authors**: Koki Ryu, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2604.11374)  

**Abstract**: Personalized image aesthetics assessment (PIAA) is an important research problem with practical real-world applications. While methods based on vision-language models (VLMs) are promising candidates for PIAA, it remains unclear whether they internally encode rich, multi-level aesthetic attributes required for effective personalization. In this paper, we first analyze the internal representations of VLMs to examine the presence and distribution of such aesthetic attributes, and then leverage them for lightweight, individual-level personalization without model fine-tuning. Our analysis reveals that VLMs encode diverse aesthetic attributes that propagate into the language decoder layers. Building on these representations, we demonstrate that simple linear models can perform PIAA effectively. We further analyze how aesthetic information is transferred across layers in different VLM architectures and across image domains. Our findings provide insights into how VLMs can be utilized for modeling subjective, individual aesthetic preferences. Our code is available at this https URL. 

---
# Learning from Contrasts: Synthesizing Reasoning Paths from Diverse Search Trajectories 

**Authors**: Peiyang Liu, Zhirui Chen, Xi Wang, Di Liang, Youru Li, Zhi Cai, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.11365)  

**Abstract**: Monte Carlo Tree Search (MCTS) has been widely used for automated reasoning data exploration, but current supervision extraction methods remain inefficient. Standard approaches retain only the single highest-reward trajectory, discarding the comparative signals present in the many explored paths. Here we introduce \textbf{Contrastive Reasoning Path Synthesis (CRPS)}, a framework that transforms supervision extraction from a filtering process into a synthesis procedure. CRPS uses a structured reflective process to analyze the differences between high- and low-quality search trajectories, extracting explicit information about strategic pivots and local failure modes. These insights guide the synthesis of reasoning chains that incorporate success patterns while avoiding identified pitfalls. We show empirically that models fine-tuned on just 60K CRPS-synthesized examples match or exceed the performance of baselines trained on 590K examples derived from standard rejection sampling, a 20$\times$ reduction in dataset size. Furthermore, CRPS improves generalization on out-of-domain benchmarks, demonstrating that learning from the contrast between success and failure produces more transferable reasoning capabilities than learning from success alone. 

---
# Geometry-Aware Localized Watermarking for Copyright Protection in Embedding-as-a-Service 

**Authors**: Zhimin Chen, Xiaojie Liang, Wenbo Xu, Yuxuan Liu, Wei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11344)  

**Abstract**: Embedding-as-a-Service (EaaS) has become an important semantic infrastructure for natural language and multimedia applications, but it is highly vulnerable to model stealing and copyright infringement. Existing EaaS watermarking methods face a fundamental robustness--utility--verifiability tension: trigger-based methods are fragile to paraphrasing, transformation-based methods are sensitive to dimensional perturbation, and region-based methods may incur false positives due to coincidental geometric affinity.
To address this problem, we propose GeoMark, a geometry-aware localized watermarking framework for EaaS copyright protection. GeoMark uses a natural in-manifold embedding as a shared watermark target, constructs geometry-separated anchors with explicit target--anchor margins, and activates watermark injection only within adaptive local neighborhoods. This design decouples where watermarking is triggered from what ownership is attributed to, achieving localized triggering and centralized attribution.
Experiments on four benchmark datasets show that GeoMark preserves downstream utility and geometric fidelity while maintaining robust copyright verification under paraphrasing, dimensional perturbation, and CSE (Clustering, Selection, Elimination) attacks, with improved verification stability and low false-positive risk. 

---
# The Salami Slicing Threat: Exploiting Cumulative Risks in LLM Systems 

**Authors**: Yihao Zhang, Kai Wang, Jiangrong Wu, Haolin Wu, Yuxuan Zhou, Zeming Wei, Dongxian Wu, Xun Chen, Jun Sun, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.11309)  

**Abstract**: Large Language Models (LLMs) face prominent security risks from jailbreaking, a practice that manipulates models to bypass built-in security constraints and generate unethical or unsafe content. Among various jailbreak techniques, multi-turn jailbreak attacks are more covert and persistent than single-turn counterparts, exposing critical vulnerabilities of LLMs.
However, existing multi-turn jailbreak methods suffer from two fundamental limitations that affect the actual impact in real-world scenarios: (a) As models become more context-aware, any explicit harmful trigger is increasingly likely to be flagged and blocked; (b) Successful final-step triggers often require finely tuned, model-specific contexts, making such attacks highly context-dependent. To fill this gap, we propose \textit{Salami Slicing Risk}, which operates by chaining numerous low-risk inputs that individually evade alignment thresholds but cumulatively accumulate harmful intent to ultimately trigger high-risk behaviors, without heavy reliance on pre-designed contextual structures. Building on this risk, we develop Salami Attack, an automatic framework universally applicable to multiple model types and modalities.
Rigorous experiments demonstrate its state-of-the-art performance across diverse models and modalities, achieving over 90\% Attack Success Rate on GPT-4o and Gemini, as well as robustness against real-world alignment defenses. We also proposed a defense strategy to constrain the Salami Attack by at least 44.8\% while achieving a maximum blocking rate of 64.8\% against other multi-turn jailbreak attacks. Our findings provide critical insights into the pervasive risks of multi-turn jailbreaking and offer actionable mitigation strategies to enhance LLM security. 

---
# The Past Is Not Past: Memory-Enhanced Dynamic Reward Shaping 

**Authors**: Yang Liu, Enxi Wang, Yufei Gao, Weixin Zhang, Bo Wang, Zhiyuan Zeng, Yikai Zhang, Yining Zheng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11297)  

**Abstract**: Despite the success of reinforcement learning for large language models, a common failure mode is reduced sampling diversity, where the policy repeatedly generates similar erroneous behaviors. Classical entropy regularization encourages randomness under the current policy, but does not explicitly discourage recurrent failure patterns across rollouts. We propose MEDS, a Memory-Enhanced Dynamic reward Shaping framework that incorporates historical behavioral signals into reward design. By storing and leveraging intermediate model representations, we capture features of past rollouts and use density-based clustering to identify frequently recurring error patterns. Rollouts assigned to more prevalent error clusters are penalized more heavily, encouraging broader exploration while reducing repeated mistakes. Across five datasets and three base models, MEDS consistently improves average performance over existing baselines, achieving gains of up to 4.13 pass@1 points and 4.37 pass@128 points. Additional analyses using both LLM-based annotations and quantitative diversity metrics show that MEDS increases behavioral diversity during sampling. 

---
# RECIPER: A Dual-View Retrieval Pipeline for Procedure-Oriented Materials Question Answering 

**Authors**: Zhuoyu Wu, Wenhui Ou, Pei-Sze Tan, Wenqi Fang, Sailaja Rajanala, Raphaël C.-W. Phan  

**Link**: [PDF](https://arxiv.org/pdf/2604.11229)  

**Abstract**: Retrieving procedure-oriented evidence from materials science papers is difficult because key synthesis details are often scattered across long, context-heavy documents and are not well captured by paragraph-only dense retrieval. We present RECIPER, a dual-view retrieval pipeline that indexes both paragraph-level context and compact large language model-extracted procedural summaries, then combines the two candidate streams with lightweight lexical reranking. Across four dense retrieval backbones, RECIPER consistently improves early-rank retrieval over paragraph-only dense retrieval, achieving average gains of +3.73 in Recall@1, +2.85 in nDCG@10, and +3.13 in MRR. With BGE-large-en-v1.5, it reaches 86.82%, 97.07%, and 97.85% on Recall@1, Recall@5, and Recall@10, respectively. We further observe improved downstream question answering under automatic metrics, suggesting that procedural summaries can serve as a useful complementary retrieval signal for procedure-oriented materials question answering. Code and data are available at this https URL. 

---
# Sign Language Recognition in the Age of LLMs 

**Authors**: Vaclav Javorek, Jakub Honzik, Ivan Gruber, Tomas Zelezny, Marek Hruz  

**Link**: [PDF](https://arxiv.org/pdf/2604.11225)  

**Abstract**: Recent Vision Language Models (VLMs) have demonstrated strong performance across a wide range of multimodal reasoning tasks. This raises the question of whether such general-purpose models can also address specialized visual recognition problems such as isolated sign language recognition (ISLR) without task-specific training. In this work, we investigate the capability of modern VLMs to perform ISLR in a zero-shot setting. We evaluate several open-source and proprietary VLMs on the WLASL300 benchmark. Our experiments show that, under prompt-only zero-shot inference, current open-source VLMs remain far behind classic supervised ISLR classifiers by a wide margin. However, follow-up experiments reveal that these models capture partial visual-semantic alignment between signs and text descriptions. Larger proprietary models achieve substantially higher accuracy, highlighting the importance of model scale and training data diversity. All our code is publicly available on GitHub. 

---
# Hierarchical Textual Knowledge for Enhanced Image Clustering 

**Authors**: Yijie Zhong, Yunfan Gao, Weipeng Jiang, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11144)  

**Abstract**: Image clustering aims to group images in an unsupervised fashion. Traditional methods focus on knowledge from visual space, making it difficult to distinguish between visually similar but semantically different classes. Recent advances in vision-language models enable the use of textual knowledge to enhance image clustering. However, most existing methods rely on coarse class labels or simple nouns, overlooking the rich conceptual and attribute-level semantics embedded in textual space. In this paper, we propose a knowledge-enhanced clustering (KEC) method that constructs a hierarchical concept-attribute structured knowledge with the help of large language models (LLMs) to guide clustering. Specifically, we first condense redundant textual labels into abstract concepts and then automatically extract discriminative attributes for each single concept and similar concept pairs, via structured prompts to LLMs. This knowledge is instantiated for each input image to achieve the knowledge-enhanced features. The knowledge-enhanced features with original visual features are adapted to various downstream clustering algorithms. We evaluate KEC on 20 diverse datasets, showing consistent improvements across existing methods using additional textual knowledge. KEC without training outperforms zero-shot CLIP on 14 out of 20 datasets. Furthermore, the naive use of textual knowledge may harm clustering performance, while KEC provides both accuracy and robustness. 

---
# Use of AI Tools: Guidelines to Maintain Academic Integrity in Computing Colleges 

**Authors**: Hatem M. El-boghdadi, Toqeer Ali Syed, Ali Akarma, Qamar Wali  

**Link**: [PDF](https://arxiv.org/pdf/2604.11111)  

**Abstract**: The rapid adoption of AI tools such as ChatGPT has significantly transformed academic practices, offering considerable benefits for both students and faculty in computing disciplines. These tools have been shown to enhance learning efficiency, academic self-efficacy, and confidence. However, their increasing use also raises pressing concerns regarding the preservation of academic integrity -- an essential pillar of the educational process. This paper explores the implications of widespread AI tool usage within computing colleges, with a particular focus on how to align their use with the principles of academic honesty. We begin by classifying common assessment techniques employed in computing education and examine how each may be impacted by AI-assisted tools. Building on this foundation, we propose a set of general guidelines applicable across various assessment formats to help instructors responsibly integrate AI tools into their pedagogy. Furthermore, we provide targeted, assessment-specific recommendations designed to uphold educational objectives while mitigating risks of academic misconduct. These guidelines serve as a practical framework for instructors aiming to balance the pedagogical advantages of AI tools with the imperative of maintaining academic integrity in computing education. Finally, we introduce a formal model that provides a structured mathematical framework for evaluating student assessments in the presence of AI-assisted tools. 

---
# Do Agent Rules Shape or Distort? Guardrails Beat Guidance in Coding Agents 

**Authors**: Xing Zhang, Guanghui Wang, Yanwei Cui, Wei Qiu, Ziyuan Li, Bing Zhu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2604.11088)  

**Abstract**: Developers increasingly guide AI coding agents through natural language instruction files (e.g., this http URL, .cursorrules), yet no controlled study has measured whether these rules actually improve agent performance or which properties make a rule beneficial. We scrape 679 such files (25,532 rules) from GitHub and conduct the first large-scale empirical evaluation, running over 5,000 agent runs with a state-of-the-art coding agent on SWE-bench Verified. Rules improve performance by 7--14 percentage points, but random rules help as much as expert-curated ones -- suggesting rules work through context priming rather than specific instruction. Negative constraints ("do not refactor unrelated code") are the only individually beneficial rule type, while positive directives ("follow code style") actively hurt -- a pattern we analyze through the lens of potential-based reward shaping (PBRS). Moreover, individual rules are mostly harmful in isolation yet collectively helpful, with no degradation up to 50 rules. These findings expose a hidden reliability risk -- well-intentioned rules routinely degrade agent performance -- and provide a clear principle for safe agent configuration: constrain what agents must not do, rather than prescribing what they should. 

---
# Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation 

**Authors**: Chen Huang, Zitan Jiang, Changyi Zou, Wenqiang Lei, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.11077)  

**Abstract**: Customer service chatbots are increasingly expected to serve not merely as reactive support tools for users, but as strategic interfaces for harvesting high-value information and business intelligence. In response, we make three main contributions. 1) We introduce and define a novel task of Proactive Information Probing, which optimizes when to probe users for pre-specified target information while minimizing conversation turns and user friction. 2) We propose PROCHATIP, a proactive chatbot framework featuring a specialized conversation strategy module trained to master the delicate timing of probes. 3) Experiments demonstrate that PROCHATIP significantly outperforms baselines, exhibiting superior capability in both information probing and service quality. We believe that our work effectively redefines the commercial utility of chatbots, positioning them as scalable, cost-effective engines for proactive business intelligence. Our code is available at this https URL. 

---
# Min-$k$ Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics 

**Authors**: Yuanhao Ding, Meimingwei Li, Esteban Garces Arias, Matthias Aßenmacher, Christian Heumann, Chongsheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11012)  

**Abstract**: The quality of text generated by large language models depends critically on the decoding sampling strategy. While mainstream methods such as Top-$k$, Top-$p$, and Min-$p$ achieve a balance between diversity and accuracy through probability-space truncation, they share an inherent limitation: extreme sensitivity to the temperature parameter. Recent logit-space approaches like Top-$n\sigma$ achieve temperature invariance but rely on global statistics that are susceptible to long-tail noise, failing to capture fine-grained confidence structures among top candidates. We propose \textbf{Min-$k$ Sampling}, a novel dynamic truncation strategy that analyzes the local shape of the sorted logit distribution to identify "semantic cliffs": sharp transitions from high-confidence core tokens to uncertain long-tail tokens. By computing a position-weighted relative decay rate, Min-$k$ dynamically determines truncation boundaries at each generation step. We formally prove that Min-$k$ achieves strict temperature invariance and empirically demonstrate its low sensitivity to hyperparameter choices. Experiments on multiple reasoning benchmarks, creative writing tasks, and human evaluation show that Min-$k$ consistently improves text quality, maintaining robust performance even under extreme temperature settings where probability-based methods collapse. We make our code, models, and analysis tools publicly available. 

---
# K-Way Energy Probes for Metacognition Reduce to Softmax in Discriminative Predictive Coding Networks 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.11011)  

**Abstract**: We present this as a negative result with an explanatory mechanism, not as a formal upper bound.
Predictive coding networks (PCNs) admit a K-way energy probe in which each candidate class is fixed as a target, inference is run to settling, and the per-hypothesis settled energies are compared. The probe appears to read a richer signal source than softmax, since the per-hypothesis energy depends on the entire generative chain.
We argue this appearance is misleading under the standard Pinchetti-style discriminative PC formulation. We present an approximate reduction showing that with target-clamped CE-energy training and effectively-feedforward latent dynamics, the K-way energy margin decomposes into a monotone function of the log-softmax margin plus a residual that is not trained to correlate with correctness. The decomposition predicts that the structural probe should track softmax from below.
We test this across six conditions on CIFAR-10: extended deterministic training, direct measurement of latent movement during inference, a post-hoc decoder fairness control on a backpropagation network, a matched-budget PC vs BP comparison, a five-point Langevin temperature sweep, and trajectory-integrated MCPC training. In every condition the probe sat below softmax. The gap was stable across training procedures within the discriminative PC family. Final-state and trajectory-integrated training produced probes whose AUROC_2 values differed by less than 10^-3 at deterministic evaluation.
The empirical regime is small: single seed, 2.1M-parameter network, 1280 test images. We frame the result as a preprint inviting replication. We discuss conditions under which the decomposition does not apply (bidirectional PC, prospective configuration, generative PC, non-CE energy formulations) and directions for productive structural probing the analysis does not foreclose. 

---
# Back to the Barn with LLAMAs: Evolving Pretrained LLM Backbones in Finetuning Vision Language Models 

**Authors**: Sameera Horawalavithana, Lauren Phillips, Ian Stewart, Sai Munikoti, Karl Pazdernik  

**Link**: [PDF](https://arxiv.org/pdf/2604.10985)  

**Abstract**: Vision-Language Models (VLMs) have rapidly advanced by leveraging powerful pre-trained Large Language Models (LLMs) as core reasoning backbones. As new and more capable LLMs emerge with improved reasoning, instruction-following, and generalization, there is a pressing need to efficiently update existing VLMs to incorporate these advancements. However, the integration of new LLMs into VLMs, particularly how the evolving LLMs contribute to multimodal reasoning, alignment, and task-specific performance remains underexplored. Addressing this gap is important for VLM development, given the rapid evolution of pretrained LLM backbones. This study presents a controlled and systematic investigation of how changes in the pretrained LLM backbone affect downstream VLM task performance. By having the vision encoder, training data, and post-training algorithm remain same across LLAMA-1, LLAMA-2, and LLAMA-3 based VLMs, we find that newer LLM backbones do not always lead to better VLMs, but the performance depends on the downstream VLM task. For example, in visual question and answering tasks, newer LLM backbones tend to solve different questions rather than just more questions, and our analysis shows this is driven by differences in how the models process information, including better calibrated confidence and more stable internal representations. We also find that some VLM capabilities appear only in the newest LLM generation, while tasks that depend mainly on visual understanding see little benefit from a newer LLM backbone. 

---
# CFMS: A Coarse-to-Fine Multimodal Synthesis Framework for Enhanced Tabular Reasoning 

**Authors**: Qixian Huang, Hongqiang Lin, Tong Fu, Yingsen Wang, Zhenghui Fu, Qirui Wang, Yiding Sun, Dongxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10973)  

**Abstract**: Reasoning over tabular data is a crucial capability for tasks like question answering and fact verification, as it requires models to comprehend both free-form questions and semi-structured tables. However, while methods like Chain-of-Thought (CoT) introduce reasoning chains, purely symbolic methodes are inherently limited by their blindness to holistic visual patterns. To address this, we propose the Coarse-to-Fine Multimodal Synthesis framework (CFMS), a novel two-stage paradigm that hierarchically decouples high-level visual perception from granular symbolic reasoning. In the Coarse Stage, CFMS leverages the Multimodal Large Language Models (MLLMs) to perform a one-time synthesis of a multi-perspective knowledge tuple. This tuple subsequently serves as a dynamic reasoning map to guide the fine stage, where a symbolic engine executes a targeted and efficient sequence of iterative operations over the table. Extensive experiments on the WikiTQ and TabFact benchmarks demonstrate that CFMS achieves competitive accuracy. The framework exhibits particular robustness when handling large tables and when instantiated with smaller backbone models, validating its effectiveness and generalizability. 

---
# A molecular clock for writing systems reveals the quantitative impact of imperial power on cultural evolution 

**Authors**: Hiroki Fukui  

**Link**: [PDF](https://arxiv.org/pdf/2604.10957)  

**Abstract**: Writing systems are cultural replicators whose evolution has never been studied quantitatively at global scale. We compile the Global Script Database (GSD): 300 writing and notation systems, 50 binary structural characters, and 259 phylogenetic edges spanning 5,400 years. Applying four methods -- phenetics, cladistics, Bayesian inference, and neural network clustering -- we find that scripts exhibit a detectable molecular clock. The best-fitting model (Mk+Gamma strict clock) yields a substitution rate of q = 0.226 substitutions/character/millennium (95% CI: 0.034-1.22; Delta BIC = -4.1 versus relaxed clock; Delta BIC = -1,364.7 versus Mk without rate variation). Political interventions break this clock: deviation from expected divergence times correlates with intervention intensity (Spearman rho = 0.556, p < 10^{-4}), and per-character rate analysis reveals that intervention selectively rewrites deep structural features rather than merely accelerating change (rate profile correlation rho = 0.320). We identify 30 major script replacement events and rank their destructive impact. A ceiling effect suppresses independent invention wherever writing already exists (Fisher's exact OR = 0.054, p < 10^{-6}), and colonial contact predicts script extinction (Cox HR = 5.25, p = 0.0006). The Spanish Empire extinguished the most scripts (6 of 12 contacted, 50%), followed by the Empire of Japan (3 of 9, 33.3%). Feature coding was validated by inter-rater reliability testing with two independent human coders (Cohen's kappa = 0.877; human-LLM kappa = 0.929; Fleiss' kappa = 0.911). 

---
# Audio Flamingo Next: Next-Generation Open Audio-Language Models for Speech, Sound, and Music 

**Authors**: Sreyan Ghosh, Arushi Goel, Kaousheik Jayakumar, Lasha Koroshinadze, Nishit Anand, Zhifeng Kong, Siddharth Gururani, Sang-gil Lee, Jaehyeon Kim, Aya Aljafari, Chao-Han Huck Yang, Sungwon Kim, Ramani Duraiswami, Dinesh Manocha, Mohammad Shoeybi, Bryan Catanzaro, Ming-Yu Liu, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2604.10905)  

**Abstract**: We present Audio Flamingo Next (AF-Next), the next-generation and most capable large audio-language model in the Audio Flamingo series, designed to advance understanding and reasoning over speech, environmental sounds and music. Compared to Audio Flamingo 3, AF-Next introduces: (i) a stronger foundational audio-language model that significantly improves accuracy across diverse audio understanding tasks; (ii) scalable strategies for constructing large-scale audio understanding and reasoning data beyond existing academic benchmarks; (iii) support for long and complex audio inputs up to 30 minutes; and (iv) Temporal Audio Chain-of-Thought, a new reasoning paradigm that explicitly grounds intermediate reasoning steps to timestamps in long audio, enabling fine-grained temporal alignment and improved interpretability. To enable these capabilities, we first conduct a systematic analysis of Audio Flamingo 3 to identify key gaps in audio understanding and reasoning. We then curate and scale new large-scale datasets totaling over 1 million hours to address these limitations and expand the existing AudioSkills-XL, LongAudio-XL, AF-Think and AF-Chat datasets. AF-Next is trained using a curriculum-based strategy spanning pre-training, mid-training and post-training stages. Extensive experiments across 20 audio understanding and reasoning benchmarks, including challenging long-audio tasks, show that AF-Next outperforms similarly sized open models by large margins and remains highly competitive with and sometimes surpasses, much larger open-weight and closed models. Beyond benchmark performance, AF-Next exhibits strong real-world utility and transfers well to unseen tasks, highlighting its robustness and generalization ability. In addition to all data, code and methods, we open-source 3 variants of AF-Next, including AF-Next-Instruct, AF-Next-Think and AF-Next-Captioner. 

---
# ZoomR: Memory Efficient Reasoning through Multi-Granularity Key Value Retrieval 

**Authors**: David H. Yang, Yuxuan Zhu, Mohammad Mohammadi Amiri, Keerthiram Murugesan, Tejaswini Pedapati, Subhajit Chaudhury, Pin-Yu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.10898)  

**Abstract**: Large language models (LLMs) have shown great performance on complex reasoning tasks but often require generating long intermediate thoughts before reaching a final answer. During generation, LLMs rely on a key-value (KV) cache for autoregressive decoding. However, the memory footprint of the KV cache grows with output length. Prior work on KV cache optimization mostly focus on compressing the long input context, while retaining the full KV cache for decoding. For tasks requiring long output generation, this leads to increased computational and memory costs. In this paper, we introduce ZoomR, a novel approach that enables LLMs to adaptively compress verbose reasoning thoughts into summaries and uses a dynamic KV cache selection policy that leverages these summaries while also strategically "zooming in" on fine-grained details. By using summary keys as a coarse-grained index during decoding, ZoomR uses the query to retrieve details for only the most important thoughts. This hierarchical strategy significantly reduces memory usage by avoiding full-cache attention at each step. Experiments across math and reasoning tasks show that our approach achieves competitive performance compared to baselines, while reducing inference memory requirements by more than $4\times$. These results demonstrate that a multi-granularity KV selection enables more memory efficient decoding, especially for long output generation. 

---
# Speaking to No One: Ontological Dissonance and the Double Bind of Conversational AI 

**Authors**: Hugh Brosnahan, Izabela Lipinska  

**Link**: [PDF](https://arxiv.org/pdf/2604.10833)  

**Abstract**: Recent reports indicate that sustained interaction with conversational artificial intelligence (AI) systems can, in a small subset of users, contribute to the emergence or stabilisation of delusional experience. Existing accounts typically attribute such cases either to individual vulnerability or to failures of safety engineering. These explanations are incomplete. Drawing on phenomenology, psychiatry, and cognitive neuroscience, this paper argues that the risk arises from the relational and ontological structure of the interaction itself. Conversational AI generates ontological dissonance: a conflict between the appearance of relational presence and the absence of any subject capable of sustaining it. Maintained through a communicative double bind and amplified by attentional asymmetries, this dissonance tends, under conditions of affective vulnerability, to stabilise into a technologically mediated analogue of folie a deux. This account explains why explicit disclaimers often fail to disrupt delusional involvement and clarifies the ethical and clinical implications for the design and use of conversational AI. 

---
# Teaching Language Models How to Code Like Learners: Conversational Serialization for Student Simulation 

**Authors**: Charles Koutcheme, Arto Hellas, Juho Leinonen  

**Link**: [PDF](https://arxiv.org/pdf/2604.10720)  

**Abstract**: Artificial models that simulate how learners act and respond within educational systems are a promising tool for evaluating tutoring strategies and feedback mechanisms at scale. However, many existing approaches in programming education rely on prompting large, proprietary language models, raising concerns around privacy, cost, and dependence. In this work, we propose a method for training open-weight artificial programming learners using authentic student process data. Our approach serializes temporal log traces into a conversational format, representing each student's problem-solving process as a dialogue between the learner and their automated assessment system. Student code submissions and environment feedback, such as test outcomes, grades, and error traces, form alternating conversational turns, enabling models to learn from the iterative debugging process. We additionally introduce a training pipeline combining supervised fine-tuning with preference optimization to align models with authentic student debugging behavior. We evaluate our framework by training Qwen models at 4B and 8B scales on a large-scale dataset of real student submissions to Python programming assignments. Our results show that incorporating environment feedback strengthens the models' ability to replicate student debugging behavior, improving over both prior code-only approaches and prompted large language models baselines in functional alignment and code similarity. We release our code to support reproducibility. 

---
# Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game 

**Authors**: Yuanbo Xie, Yingjie Zhang, Yulin Li, Shouyou Song, Xiaokun Chen, Zhihan Liu, Liya Su, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10717)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems augment large language models with external knowledge, yet introduce a critical security vulnerability: RAG Knowledge Base Leakage, wherein adversarial prompts can induce the model to divulge retrieved proprietary content. Recent studies reveal that such leakage can be executed through adaptive and iterative attack strategies (named RAG extraction attack), while effective countermeasures remain notably lacking. To bridge this gap, we propose CanaryRAG, a runtime defense mechanism inspired by stack canaries in software security. CanaryRAG embeds carefully designed canary tokens into retrieved chunks and reformulates RAG extraction defense as a dual-path runtime integrity game. Leakage is detected in real time whenever either the target or oracle path violates its expected canary behavior, including under adaptive suppression and obfuscation. Extensive evaluations against existing attacks demonstrate that CanaryRAG provides robust defense, achieving substantially lower chunk recovery rates than state-of-the-art baselines while imposing negligible impact on task performance and inference latency. Moreover, as a plug-and-play solution, CanaryRAG can be seamlessly integrated into arbitrary RAG pipelines without requiring retraining or structural modifications, offering a practical and scalable safeguard for proprietary data. 

---
# Bringing Value Models Back: Generative Critics for Value Modeling in LLM Reinforcement Learning 

**Authors**: Zikang Shan, Han Zhong, Liwei Wang, Li Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.10701)  

**Abstract**: Credit assignment is a central challenge in reinforcement learning (RL). Classical actor-critic methods address this challenge through fine-grained advantage estimation based on a learned value function. However, learned value models are often avoided in modern large language model (LLM) RL because conventional discriminative critics are difficult to train reliably. We revisit value modeling and argue that this difficulty is partly due to limited expressiveness. In particular, representation complexity theory suggests that value functions can be hard to approximate under the one-shot prediction paradigm used by existing value models, and our scaling experiments show that such critics do not improve reliably with scale. Motivated by this observation, we propose Generative Actor-Critic (GenAC), which replaces one-shot scalar value prediction with a generative critic that performs chain-of-thought reasoning before producing a value estimate. We further introduce In-Context Conditioning, which helps the critic remain calibrated to the current actor throughout training. GenAC improves value approximation, ranking reliability, and out-of-distribution generalization, and these gains translate into stronger downstream RL performance than both value-based and value-free baselines. Overall, our results suggest that stronger value modeling is a promising direction for improving credit assignment in LLM reinforcement learning. 

---
# SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting 

**Authors**: Binbin Zheng, Xing Ma, Yiheng Liang, Jingqing Ruan, Xiaoliang Fu, Kepeng Lin, Benchang Zhu, Ke Zeng, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.10688)  

**Abstract**: On-policy reinforcement learning has become the dominant paradigm for reasoning alignment in large language models, yet its sparse, outcome-level rewards make token-level credit assignment notoriously difficult. On-Policy Distillation (OPD) alleviates this by introducing dense, token-level KL supervision from a teacher model, but typically applies this supervision uniformly across all rollouts, ignoring fundamental differences in signal quality. We propose Signal-Calibrated On-Policy Distillation Enhancement (SCOPE), a dual-path adaptive training framework that routes on-policy rollouts by correctness into two complementary supervision paths. For incorrect trajectories, SCOPE performs teacher-perplexity-weighted KL distillation to prioritize instances where the teacher demonstrates genuine corrective capability, while down-weighting unreliable guidance. For correct trajectories, it applies student-perplexity-weighted MLE to concentrate reinforcement on low-confidence samples at the capability boundary rather than over-reinforcing already mastered ones. Both paths employ a group-level normalization to adaptively calibrate weight distributions, accounting for the intrinsic difficulty variance across prompts. Extensive experiments on six reasoning benchmarks show that SCOPE achieves an average relative improvement of 11.42% in Avg@32 and 7.30% in Pass@32 over competitive baselines, demonstrating its consistent effectiveness. 

---
# Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents 

**Authors**: Hao Wang, Guozhi Wang, Han Xiao, Yufeng Zhou, Yue Pan, Jichao Wang, Ke Xu, Yafei Wen, Xiaohu Ruan, Xiaoxin Chen, Honggang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2604.10674)  

**Abstract**: Reinforcement learning (RL) has been widely used to train LLM agents for multi-turn interactive tasks, but its sample efficiency is severely limited by sparse rewards and long horizons. On-policy self-distillation (OPSD) alleviates this by providing dense token-level supervision from a privileged teacher that has access to ground-truth answers. However, such fixed privileged information cannot capture the diverse valid strategies in agent tasks, and naively combining OPSD with RL often leads to training collapse. To address these limitations, we introduce Skill-SD, a framework that turns the agent's own trajectories into dynamic training-only supervision. Completed trajectories are summarized into compact natural language skills that describe successful behaviors, mistakes, and workflows. These skills serve as dynamic privileged information conditioning only the teacher, while the student always acts under the plain task prompt and learns to internalize the guidance through distillation. To stabilize the training, we derive an importance-weighted reverse-KL loss to provide gradient-correct token-level distillation, and dynamically synchronize the teacher with the improving student. Experimental results on agentic benchmarks demonstrate that Skill-SD substantially outperforms the standard RL baseline, improving both vanilla GRPO (+14.0%/+10.9% on AppWorld/Sokoban) and vanilla OPD (+42.1%/+40.6%). Project page: this https URL 

---
# Omnimodal Dataset Distillation via High-order Proxy Alignment 

**Authors**: Yuxuan Gao, Xiaohao Liu, Xiaobo Xia, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10666)  

**Abstract**: Dataset distillation compresses large-scale datasets into compact synthetic sets while preserving training performance, but existing methods are largely restricted to single-modal or bimodal settings. Extending dataset distillation to scenarios involving more than two modalities, i.e., Omnimodal Dataset Distillation, remains underexplored and challenging due to increased heterogeneity and complex cross-modal interactions. In this work, we identify the key determinant that bounds the endpoint discrepancy in the omnimodal setting, which is exacerbated with an increasing number of modalities. To this end, we propose HoPA, a unified method that captures high-order cross-modal alignments via a compact proxy, which is compatible with trajectory matching as well. By abstracting omnimodal alignment with a shared similarity structure, our method avoids the combinatorial complexity of pairwise modality modeling and enables scalable joint distillation across heterogeneous modalities. Theoretical analysis from the spectral perspective reveals the rationality of our proposed method against bimodal dataset distillation techniques. Extensive experiments on various benchmarks demonstrate that the proposed method achieves superior compression-performance trade-offs compared to existing competitors. The source code will be publicly released. 

---
# SpectralLoRA: Is Low-Frequency Structure Sufficient for LoRA Adaptation? A Spectral Analysis of Weight Updates 

**Authors**: Rajveer Singh  

**Link**: [PDF](https://arxiv.org/pdf/2604.10649)  

**Abstract**: We present a systematic empirical study of the spectral structure of LoRA weight updates. Through 2D Discrete Cosine Transform (DCT) analysis of trained adaptation matrices across BERT-base and RoBERTa-base on four GLUE benchmarks (SST-2, MNLI, CoLA, QQP), we establish that LoRA updates are universally dominated by low-frequency components: on average, just 33% of DCT coefficients capture 90% of total spectral energy. Retaining only 10% of frequency coefficients reduces adapter storage by 10x while sacrificing only 1.95pp on SST-2. Notably, frequency masking at k=50% improves over full LoRA on 3 of 8 model-task pairs, suggesting high-frequency components act as adaptation noise. We further discover that RoBERTa-base is systematically more spectrally compressible than BERT-base across all tasks, and that task complexity governs spectral sensitivity -- NLI tasks require more frequency budget than sentiment classification. These findings motivate a new design principle for PEFT: spectral sparsity in adaptation. 

---
# BMdataset: A Musicologically Curated LilyPond Dataset 

**Authors**: Matteo Spanio, Ilay Guler, Antonio Rodà  

**Link**: [PDF](https://arxiv.org/pdf/2604.10628)  

**Abstract**: Symbolic music research has relied almost exclusively on MIDI-based datasets; text-based engraving formats such as LilyPond remain unexplored for music understanding. We present BMdataset, a musicologically curated dataset of 393 LilyPond scores (2,646 movements) transcribed by experts directly from original Baroque manuscripts, with metadata covering composer, musical form, instrumentation, and sectional attributes. Building on this resource, we introduce LilyBERT (weights can be found at this https URL), a CodeBERT-based encoder adapted to symbolic music through vocabulary extension with 115 LilyPond-specific tokens and masked language model pre-training. Linear probing on the out-of-domain Mutopia corpus shows that, despite its modest size (~90M tokens), fine-tuning on BMdataset alone outperforms continuous pre-training on the full PDMX corpus (~15B tokens) for both composer and style classification, demonstrating that small, expertly curated datasets can be more effective than large, noisy corpora for music understanding. Combining broad pre-training with domain-specific fine-tuning yields the best results overall (84.3% composer accuracy), confirming that the two data regimes are complementary. We release the dataset, tokenizer, and model to establish a baseline for representation learning on LilyPond. 

---
# NSFL: A Post-Training Neuro-Symbolic Fuzzy Logic Framework for Boolean Operators in Neural Embeddings 

**Authors**: Vladi Vexler, Ofer Idan, Gil Lederman, Dima Sivov  

**Link**: [PDF](https://arxiv.org/pdf/2604.10604)  

**Abstract**: Standard dense retrievers lack a native calculus for multi-atom logical constraints. We introduce Neuro-Symbolic Fuzzy Logic (NSFL), a framework that adapts formal t-norms and t-conorms to neural embedding spaces without requiring retraining. NSFL operates as a first-order hybrid calculus: it anchors logical operations on isolated zero-order similarity scores while actively steering representations using Neuro-Symbolic Deltas (NS-Delta) -- the first-order marginal differences derived from contextual fusion. This preserves pure atomic meaning while capturing domain reliance, preventing the representation collapse and manifold escape endemic to traditional geometric baselines. For scalable real-time retrieval, Spherical Query Optimization (SQO) leverages Riemannian optimization to project these fuzzy formulas into manifold-stable query vectors. Validated across six distinct encoder configurations and two modalities (including zero-shot and SOTA fine-tuned models), NSFL yields mAP improvements up to +81%. Notably, NSFL provides an additive 20% average and up to 47% boost even when applied to encoders explicitly fine-tuned for logical reasoning. By establishing a training-free, order-aware calculus for high-dimensional spaces, this framework lays the foundation for future dynamic scaling and learned manifold logic. 

---
# Calibration Collapse Under Sycophancy Fine-Tuning: How Reward Hacking Breaks Uncertainty Quantification in LLMs 

**Authors**: Subramanyam Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2604.10585)  

**Abstract**: Modern large language models (LLMs) are increasingly fine-tuned via reinforcement learning from human feedback (RLHF) or related reward optimisation schemes. While such procedures improve perceived helpfulness, we investigate whether sycophantic reward signals degrade calibration -- a property essential for reliable uncertainty quantification. We fine-tune Qwen3-8B under three regimes: no fine-tuning (base), neutral supervised fine-tuning (SFT) on TriviaQA, and sycophancy-inducing Group Relative Policy Optimisation (GRPO) that rewards agreement with planted wrong answers. Evaluating on $1{,}000$ MMLU items across five subject domains with bootstrap confidence intervals and permutation testing, we find that \textbf{sycophantic GRPO produces consistent directional calibration degradation} -- ECE rises by $+0.006$ relative to the base model and MCE increases by $+0.010$ relative to neutral SFT -- though the effect does not reach statistical significance ($p = 0.41$) at this training budget. Post-hoc matrix scaling applied to all three models reduces ECE by $40$--$64\%$ and improves accuracy by $1.5$--$3.0$ percentage points. However, the sycophantic model retains the highest post-scaling ECE relative to the neutral SFT control ($0.042$ vs.\ $0.037$), suggesting that reward-induced miscalibration leaves a structured residual even after affine correction. These findings establish a methodology for evaluating the calibration impact of reward hacking and motivate calibration-aware training objectives. 

---
# Evaluating Small Open LLMs for Medical Question Answering: A Practical Framework 

**Authors**: Avi-ad Avraam Buskila  

**Link**: [PDF](https://arxiv.org/pdf/2604.10535)  

**Abstract**: Incorporating large language models (LLMs) in medical question answering demands more than high average accuracy: a model that returns substantively different answers each time it is queried is not a reliable medical tool. Online health communities such as Reddit have become a primary source of medical information for millions of users, yet they remain highly susceptible to misinformation; deploying LLMs as assistants in these settings amplifies the need for output consistency alongside correctness. We present a practical, open-source evaluation framework for assessing small, locally-deployable open-weight LLMs on medical question answering, treating reproducibility as a first-class metric alongside lexical and semantic accuracy. Our pipeline computes eight quality metrics, including BERTScore, ROUGE-L, and an LLM-as-judge rubric, together with two within-model reproducibility metrics derived from repeated inference (N=10 runs per question). Evaluating three models (Llama 3.1 8B, Gemma 3 12B, MedGemma 1.5 4B) on 50 MedQuAD questions (N=1,500 total responses) reveals that despite low-temperature generation (T=0.2), self-agreement across runs reaches at most 0.20, while 87-97% of all outputs per model are unique -- a safety gap that single-pass benchmarks entirely miss. The clinically fine-tuned MedGemma 1.5 4B underperforms the larger general-purpose models on both quality and reproducibility; however, because MedGemma is also the smallest model, this comparison confounds domain fine-tuning with model scale. We describe the methodology in sufficient detail for practitioners to replicate or extend the evaluation for their own model-selection workflows. All code and data pipelines are available at this https URL. 

---
# VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions 

**Authors**: Hung-Ting Su, Ting-Jun Wang, Jia-Fong Yeh, Min Sun, Winston H. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10533)  

**Abstract**: Conventional Vision-and-Language Navigation (VLN) benchmarks assume instructions are feasible and the referenced target exists, leaving agents ill-equipped to handle false-premise goals. We introduce VLN-NF, a benchmark with false-premise instructions where the target is absent from the specified room and agents must navigate, gather evidence through in-room exploration, and explicitly output NOT-FOUND. VLN-NF is constructed via a scalable pipeline that rewrites VLN instructions using an LLM and verifies target absence with a VLM, producing plausible yet factually incorrect goals. We further propose REV-SPL to jointly evaluate room reaching, exploration coverage, and decision correctness. To address this challenge, we present ROAM, a two-stage hybrid that combines supervised room-level navigation with LLM/VLM-driven in-room exploration guided by a free-space clearance prior. ROAM achieves the best REV-SPL among compared methods, while baselines often under-explore and terminate prematurely under unreliable instructions. VLN-NF project page can be found at this https URL. 

---
# AI Patents in the United States and China: Measurement, Organization, and Knowledge Flows 

**Authors**: Hanming Fang, Xian Gu, Hanyin Yan, Wu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10529)  

**Abstract**: We develop a high-precision classifier to measure artificial intelligence (AI) patents by fine-tuning PatentSBERTa on manually labeled data from the USPTO's AI Patent Dataset. Our classifier substantially improves the existing USPTO approach, achieving 97.0% precision, 91.3% recall, and a 94.0% F1 score, and it generalizes well to Chinese patents based on citation and lexical validation. Applying it to granted U.S. patents (1976-2023) and Chinese patents (2010-2023), we document rapid growth in AI patenting in both countries and broad convergence in AI patenting intensity and subfield composition, even as China surpasses the United States in recent annual patent counts. The organization of AI innovation nevertheless differs sharply: U.S. AI patenting is concentrated among large private incumbents and established hubs, whereas Chinese AI patenting is more geographically diffuse and institutionally diverse, with larger roles for universities and state-owned enterprises. For listed firms, AI patents command a robust market-value premium in both countries. Cross-border citations show continued technological interdependence rather than decoupling, with Chinese AI inventors relying more heavily on U.S. frontier knowledge than vice versa. 

---
# Thinking Fast, Thinking Wrong: Intuitiveness Modulates LLM Counterfactual Reasoning in Policy Evaluation 

**Authors**: Yanjie He  

**Link**: [PDF](https://arxiv.org/pdf/2604.10511)  

**Abstract**: Large language models (LLMs) are increasingly used for causal and counterfactual reasoning, yet their reliability in real-world policy evaluation remains underexplored. We construct a benchmark of 40 empirical policy evaluation cases drawn from economics and social science, each grounded in peer-reviewed evidence and classified by intuitiveness -- whether the empirical finding aligns with (obvious), is unclear relative to (ambiguous), or contradicts (counter-intuitive) common prior expectations. We evaluate four frontier LLMs across five prompting strategies with 2,400 experimental trials and analyze the results using mixed-effects logistic regression. Our findings reveal three key results: (1) a chain-of-thought (CoT) paradox, where chain-of-thought prompting dramatically improves performance on obvious cases but this benefit is nearly eliminated on counter-intuitive ones (interaction OR = 0.053, $p < 0.001$); (2) intuitiveness as the dominant factor, explaining more variance than model choice or prompting strategy (ICC = 0.537); and (3) a knowledge-reasoning dissociation, where citation-based familiarity is unrelated to accuracy ($p = 0.53$), suggesting models possess relevant knowledge but fail to reason with it when findings contradict intuition. We frame these results through the lens of dual-process theory (System 1 vs. System 2) and argue that current LLMs' "slow thinking" may be little more than "slow talking" -- they produce the form of deliberative reasoning without the substance. 

---
# PatchRecall: Patch-Driven Retrieval for Automated Program Repair 

**Authors**: Mahir Labib Dihan, Faria Binta Awal, Md. Ishrak Ahsan  

**Link**: [PDF](https://arxiv.org/pdf/2604.10481)  

**Abstract**: Retrieving the correct set of files from a large codebase is a crucial step in Automated Program Repair (APR). High recall is necessary to ensure that the relevant files are included, but simply increasing the number of retrieved files introduces noise and degrades efficiency. To address this tradeoff, we propose PatchRecall, a hybrid retrieval approach that balances recall with conciseness. Our method combines two complementary strategies: (1) codebase retrieval, where the current issue description is matched against the codebase to surface potentially relevant files, and (2) history-based retrieval, where similar past issues are leveraged to identify edited files as candidate targets. Candidate files from both strategies are merged and reranked to produce the final retrieval set. Experiments on SWE-Bench demonstrate that PatchRecall achieves higher recall without significantly increasing retrieved file count, enabling more effective APR. 

---
# Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking 

**Authors**: Jingru Li, Wei Ren, Tianqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10299)  

**Abstract**: Large Vision-Language Models (LVLMs) rely on attention-based retrieval of safety instructions to maintain alignment during generation. Existing attacks typically optimize image perturbations to maximize harmful output likelihood, but suffer from slow convergence due to gradient conflict between adversarial objectives and the model's safety-retrieval mechanism. We propose Attention-Guided Visual Jailbreaking, which circumvents rather than overpowers safety alignment by directly manipulating attention patterns. Our method introduces two simple auxiliary objectives: (1) suppressing attention to alignment-relevant prefix tokens and (2) anchoring generation on adversarial image features. This simple yet effective push-pull formulation reduces gradient conflict by 45% and achieves 94.4% attack success rate on Qwen-VL (vs. 68.8% baseline) with 40% fewer iterations. At tighter perturbation budgets ($\epsilon=8/255$), we maintain 59.0% ASR compared to 45.7% for standard methods. Mechanistic analysis reveals a failure mode we term safety blindness: successful attacks suppress system-prompt attention by 80%, causing models to generate harmful content not by overriding safety rules, but by failing to retrieve them. 

---
# Hijacking Text Heritage: Hiding the Human Signature through Homoglyphic Substitution 

**Authors**: Robert Dilworth  

**Link**: [PDF](https://arxiv.org/pdf/2604.10271)  

**Abstract**: In what way could a data breach involving government-issued IDs such as passports, driver's licenses, etc., rival a random voluntary disclosure on a nondescript social-media platform? At first glance, the former appears more significant, and that is a valid assessment. The disclosed data could contain an individual's date of birth and address; for all intents and purposes, a leak of that data would be disastrous. Given the threat, the latter scenario involving an innocuous online post seems comparatively harmless--or does it? From that post and others like it, a forensic linguist could stylometrically uncover equivalent pieces of information, estimating an age range for the author (adolescent or adult) and narrowing down their geographical location (specific country). While not an exact science--the determinations are statistical--stylometry can reveal comparable, though noticeably diluted, information about an individual. To prevent an ID from being breached, simply sharing it as little as possible suffices. Preventing the leakage of personal information from written text requires a more complex solution: adversarial stylometry. In this paper, we explore how performing homoglyph substitution--the replacement of characters with visually similar alternatives (e.g., "h" $\texttt{[U+0068]}$ $\rightarrow$ "h" $\texttt{[U+04BB]}$)--on text can degrade stylometric systems. 

---
# The Amazing Agent Race: Strong Tool Users, Weak Navigators 

**Authors**: Zae Myung Kim, Dongseok Lee, Jaehyung Kim, Vipul Raheja, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10261)  

**Abstract**: Existing tool-use benchmarks for LLM agents are overwhelmingly linear: our analysis of six benchmarks shows 55 to 100% of instances are simple chains of 2 to 5 steps. We introduce The Amazing Agent Race (AAR), a benchmark featuring directed acyclic graph (DAG) puzzles (or "legs") with fork-merge tool chains. We release 1,400 instances across two variants: sequential (800 legs) and compositional (600 DAG legs). Agents must navigate Wikipedia, execute multi-step tool chains, and aggregate results into a verifiable answer. Legs are procedurally generated from Wikipedia seeds across four difficulty levels with live-API validation. Three complementary metrics (finish-line accuracy, pit-stop visit rate, and roadblock completion rate) separately diagnose navigation, tool-use, and arithmetic failures. Evaluating three agent frameworks on 1,400 legs, the best achieves only 37.2% accuracy. Navigation errors dominate (27 to 52% of trials) while tool-use errors remain below 17%, and agent architecture matters as much as model scale (Claude Code matches Codex CLI at 37% with 6x fewer tokens). The compositional structure of AAR reveals that agents fail not at calling tools but at navigating to the right pages, a blind spot invisible to linear benchmarks. The project page can be accessed at: this https URL 

---
# Visual Late Chunking: An Empirical Study of Contextual Chunking for Efficient Visual Document Retrieval 

**Authors**: Yibo Yan, Mingdong Ou, Yi Cao, Jiahao Huo, Xin Zou, Shuliang Liu, James Kwok, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10167)  

**Abstract**: Multi-vector models dominate Visual Document Retrieval (VDR) due to their fine-grained matching capabilities, but their high storage and computational costs present a major barrier to practical deployment. In this paper, we propose ColChunk, a plug-and-play framework that introduces multimodal late chunking to construct efficient, contextualized multi-vectors. Unlike existing pruning or fixed-token approaches, ColChunk employs hierarchical clustering on patch-level embeddings, fused with a 2D position prior to ensure spatial-semantic coherence. This adaptive grouping allows for a content-aware representation that preserves global context while drastically reducing the vector count. Evaluations across 24 VDR datasets demonstrate ColChunk achieves over a 90% reduction in storage requirements while simultaneously delivering a 9-point average improvement in nDCG@5 across representative single-vector models. ColChunk provides a practical solution for balancing retrieval accuracy and efficiency in visual document systems. 

---
# Learning from Emptiness: De-biasing Listwise Rerankers with Content-Agnostic Probability Calibration 

**Authors**: Hang Lv, Hongchao Gu, Ruiqing Yang, Liangyue Li, Zulong Chen, Defu Lian, Hao Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.10150)  

**Abstract**: Generative listwise reranking leverages global context for superior retrieval but is plagued by intrinsic position bias, where models exhibit structural sensitivity to input order independent of relevance. Existing mitigations present a dilemma: inference-time aggregation incurs prohibitive latency, while training-based methods often fail to eradicate ingrained priors, particularly in compact models. To resolve this dilemma, we propose CapCal (Content-Agnostic Probability Calibration), a training-free framework that mechanically decouples positional bias from ranking decisions. By estimating the bias distribution via content-free placeholders, CapCal rectifies output logits through an entropy-adaptive contrastive mechanism. Evaluations across 10 benchmarks confirm that CapCal achieves superior performance among training-free methods while preserving single-pass efficiency. Notably, it unlocks the latent potential of lightweight models (e.g., 0.6B), delivering absolute NDCG gains exceeding 10 points and outperforming both permutation-based aggregation and data-augmentation baselines. 

---
# FinTrace: Holistic Trajectory-Level Evaluation of LLM Tool Calling for Long-Horizon Financial Tasks 

**Authors**: Yupeng Cao, Haohang Li, Weijin Liu, Wenbo Cao, Anke Xu, Lingfei Qian, Xueqing Peng, Minxue Tang, Zhiyuan Yao, Jimin Huang, K.P. Subbalakshmi, Zining Zhu, Jordan W. Suchow, Yangyang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10015)  

**Abstract**: Recent studies demonstrate that tool-calling capability enables large language models (LLMs) to interact with external environments for long-horizon financial tasks. While existing benchmarks have begun evaluating financial tool calling, they focus on limited scenarios and rely on call-level metrics that fail to capture trajectory-level reasoning quality. To address this gap, we introduce FinTrace, a benchmark comprising 800 expert-annotated trajectories spanning 34 real-world financial task categories across multiple difficulty levels. FinTrace employs a rubric-based evaluation protocol with nine metrics organized along four axes -- action correctness, execution efficiency, process quality, and output quality -- enabling fine-grained assessment of LLM tool-calling behavior. Our evaluation of 13 LLMs reveals that while frontier models achieve strong tool selection, all models struggle with information utilization and final answer quality, exposing a critical gap between invoking the right tools and reasoning effectively over their outputs. To move beyond diagnosis, we construct FinTrace-Training, the first trajectory-level preference dataset for financial tool-calling, containing 8,196 curated trajectories with tool-augmented contexts and preference pairs. We fine-tune Qwen-3.5-9B using supervised fine-tuning followed by direct preference optimization (DPO) and show that training on FinTrace-Training consistently improves intermediate reasoning metrics, with DPO more effectively suppressing failure modes. However, end-to-end answer quality remains a bottleneck, indicating that trajectory-level improvements do not yet fully propagate to final output quality. 

---
# Demographic and Linguistic Bias Evaluation in Omnimodal Language Models 

**Authors**: Alaa Elobaid  

**Link**: [PDF](https://arxiv.org/pdf/2604.10014)  

**Abstract**: This paper provides a comprehensive evaluation of demographic and linguistic biases in omnimodal language models that process text, images, audio, and video within a single framework. Although these models are being widely deployed, their performance across different demographic groups and modalities is not well studied. Four omnimodal models are evaluated on tasks that include demographic attribute estimation, identity verification, activity recognition, multilingual speech transcription, and language identification. Accuracy differences are measured across age, gender, skin tone, language, and country of origin. The results show that image and video understanding tasks generally exhibit better performance with smaller demographic disparities. In contrast, audio understanding tasks exhibit significantly lower performance and substantial bias, including large accuracy differences across age groups, genders, and languages, and frequent prediction collapse toward narrow categories. These findings highlight the importance of evaluating fairness across all supported modalities as omnimodal language models are increasingly used in real-world applications. 

---
# Reproduction Beyond Benchmarks: ConstBERT and ColBERT-v2 Across Backends and Query Distributions 

**Authors**: Utshab Kumar Ghosh, Ashish David, Shubham Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2604.09982)  

**Abstract**: Reproducibility must validate architectural robustness, not just numerical accuracy. We evaluate ColBERT-v2 and ConstBERT across five dimensions, finding that while ConstBERT reproduces within 0.05% MRR@10 on MS-MARCO, both models show a drop of 86-97% on long, narrative queries (TREC ToT 2025). Ablations prove this failure is architectural: performance plateaus at 20 words because the MaxSim operator's uniform token weighting cannot distinguish signal from filler noise. Furthermore, undocumented backend parameters create an 8-point gap due to ConstBERT's sparse centroid coverage, and fine-tuning with 3x more data actually degrades performance by up to 29%. We conclude that architectural constraints in multi-vector retrieval cannot be overcome by adaptation alone. Code: this https URL. 

---
# Cross-Cultural Value Awareness in Large Vision-Language Models 

**Authors**: Phillip Howard, Xin Su, Kathleen C. Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2604.09945)  

**Abstract**: The rapid adoption of large vision-language models (LVLMs) in recent years has been accompanied by growing fairness concerns due to their propensity to reinforce harmful societal stereotypes. While significant attention has been paid to such fairness concerns in the context of social biases, relatively little prior work has examined the presence of stereotypes in LVLMs related to cultural contexts such as religion, nationality, and socioeconomic status. In this work, we aim to narrow this gap by investigating how cultural contexts depicted in images influence the judgments LVLMs make about a person's moral, ethical, and political values. We conduct a multi-dimensional analysis of such value judgments in five popular LVLMs using counterfactual image sets, which depict the same person across different cultural contexts. Our evaluation framework diagnoses LVLM awareness of cultural value differences through the use of Moral Foundations Theory, lexical analyses, and the sensitivity of generated values to depicted cultural contexts. 

---
# From UAV Imagery to Agronomic Reasoning: A Multimodal LLM Benchmark for Plant Phenotyping 

**Authors**: Yu Wu, Guangzeng Han, Ibra Niang Niang, Francia Ravelombola, Maiara Oliveira, Jason Davis, Dong Chen, Feng Lin, Xiaolei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09907)  

**Abstract**: To improve crop genetics, high-throughput, effective and comprehensive phenotyping is a critical prerequisite. While such tasks were traditionally performed manually, recent advances in multimodal foundation models, especially in vision-language models (VLMs), have enabled more automated and robust phenotypic analysis. However, plant science remains a particularly challenging domain for foundation models because it requires domain-specific knowledge, fine-grained visual interpretation, and complex biological and agronomic reasoning. To address this gap, we develop PlantXpert, an evidence-grounded multimodal reasoning benchmark for soybean and cotton phenotyping. Our benchmark provides a structured and reproducible framework for agronomic adaptation of VLMs, and enables controlled comparison between base models and their domain-adapted counterparts. We constructed a dataset comprising 385 digital images and more than 3,000 benchmark samples spanning key plant science domains including disease, pest control, weed management, and yield. The benchmark can assess diverse capabilities including visual expertise, quantitative reasoning, and multi-step agronomic reasoning. A total of 11 state-of-the-art VLMs were evaluated. The results indicate that task-specific fine-tuning leads to substantial improvement in accuracy, with models such as Qwen3-VL-4B and Qwen3-VL-30B achieving up to 78%. At the same time, gains from model scaling diminish beyond a certain capacity, generalization across soybean and cotton remains uneven, and quantitative as well as biologically grounded reasoning continue to pose substantial challenges. These findings suggest that PlantXpert can serve as a foundation for assessing evidence-grounded agronomic reasoning and for advancing multimodal model development in plant science. 

---
# Exploring Structural Complexity in Normative RAG with Graph-based approaches: A case study on the ETSI Standards 

**Authors**: Aiman Al Masoud, Marco Arazzi, Simone Germani, Antonino Nocera  

**Link**: [PDF](https://arxiv.org/pdf/2604.09868)  

**Abstract**: Industrial standards and normative documents exhibit intricate hierarchical structures, domain-specific lexicons, and extensive cross-referential dependencies, which making it challenging to process them directly by Large Language Models (LLMs).
While Retrieval-Augmented Generation (RAG) provides a computationally efficient alternative to LLM fine-tuning, standard "vanilla" vector-based retrieval may fail to capture the latent structural and relational features intrinsic in normative documents.
With the objective of shedding light on the most promising technique for building high-performance RAG solutions for normative, standards, and regulatory documents, this paper investigates the efficacy of Graph RAG architectures, which represent information as interconnected nodes, thus moving from simple semantic similarity toward a more robust, relation-aware retrieval mechanism.
Despite the promise of graph-based techniques, there is currently a lack of empirical evidence as to which is the optimal indexing strategy for technical standards. Therefore, to help solve this knowledge gap, we propose a specialized RAG methodology tailored to the unique structure and lexical characteristics of standards and regulatory documents.
Moreover, to keep our investigation grounded, we focus on well-known public standards, such as the ETSI EN 301 489 series. We evaluate several lightweight and low-latency strategies designed to embed document structure directly into the retrieval workflow.
The considered approaches are rigorously tested against a custom synthesized Q&A dataset, facilitating a quantitative performance analysis. Our experimental results demonstrate that the incorporation of structural and lexical information into the index can enhance, at least to some extent, retrieval performance, providing a scalable framework for automated normative and standards elaboration. 

---
# Instructing LLMs to Negotiate using Reinforcement Learning with Verifiable Rewards 

**Authors**: Shuze Daniel Liu, Claire Chen, Jiabao Sean Xiao, Lei Lei, Yuheng Zhang, Yisong Yue, David Simchi-Levi  

**Link**: [PDF](https://arxiv.org/pdf/2604.09855)  

**Abstract**: The recent advancement of Large Language Models (LLMs) has established their potential as autonomous interactive agents. However, they often struggle in strategic games of incomplete information, such as bilateral price negotiation. In this paper, we investigate if Reinforcement Learning from Verifiable Rewards (RLVR) can effectively teach LLMs to negotiate. Specifically, we explore the strategic behaviors that emerge during the learning process. We introduce a framework that trains a mid-sized buyer agent against a regulated LLM seller across a wide distribution of real-world products. By grounding reward signals directly in the maximization of economic surplus and strict adherence to private budget constraints, we reveal a novel four-phase strategic evolution. The agent progresses from naive bargaining to using aggressive starting prices, moves through a phase of deadlock, and ultimately develops sophisticated persuasive skills. Our results demonstrate that this verifiable training allows a 30B agent to significantly outperform frontier models over ten times its size in extracting surplus. Furthermore, the trained agent generalizes robustly to stronger counterparties unseen during training and remains effective even when facing hostile, adversarial seller personas. 

---
# COMPOSITE-Stem 

**Authors**: Kyle Waters, Lucas Nuzzi, Tadhg Looram, Alessandro Tomasiello, Ariel Ghislain Kemogne Kamdoum, Bikun Li, Damien Sileo, Egor Kretov, Francesco Fournier-Facio, Georgios Soloupis, Haile Kassahun, Hew Wolff, Jiaqi Cai, Lianghui Li, Marc Roth, Mohinder Naiya, Naixu Guo, Qicheng Tang, Richard Wheeler, Samuele Sala, Serguei Popov, Steven Dillman, Yuqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09836)  

**Abstract**: AI agents hold growing promise for accelerating scientific discovery; yet, a lack of frontier evaluations hinders adoption into real workflows. Expert-written benchmarks have proven effective at measuring AI reasoning, but most at this stage have become saturated and only measure performance on constrained outputs. To help address this gap, we introduce COMPOSITE-STEM, a benchmark of 70 expert-written tasks in physics, biology, chemistry, and mathematics, curated by doctoral-level researchers. Our benchmark combines exact-match grading and criterion-based rubrics with an LLM-as-a-jury grading protocol, allowing more flexible assessment of scientifically meaningful outputs. Using an adapted multimodal Terminus-2 agent harness within the Harbor agentic evaluation framework, we evaluate four frontier models. The top-performing model achieves 21%, demonstrating that COMPOSITE-STEM captures capabilities beyond current agent reach. All tasks are open-sourced with contributor permission to support reproducibility and to promote additional research towards AI's acceleration of scientific progress in these domains. 

---
# ProGAL-VLA: Grounded Alignment through Prospective Reasoning in Vision-Language-Action Models 

**Authors**: Nastaran Darabi, Amit Ranjan Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2604.09824)  

**Abstract**: Vision language action (VLA) models enable generalist robotic agents but often exhibit language ignorance, relying on visual shortcuts and remaining insensitive to instruction changes. We present Prospective Grounding and Alignment VLA (ProGAL-VLA), which constructs a 3D entity-centric graph (GSM), uses a slow planner to produce symbolic sub-goals, and aligns them with grounded entities via a Grounding Alignment Contrastive (GAC) loss. All actions are conditioned on a verified goal embedding $g_t$, whose attention entropy provides an intrinsic ambiguity signal. On LIBERO-Plus, ProGAL-VLA increases robustness under robot perturbations from 30.3 to 71.5 percent, reduces language ignorance by 3x-4x, and improves entity retrieval from 0.41 to 0.71 Recall@1. On the Custom Ambiguity Benchmark, it reaches AUROC 0.81 (vs. 0.52), AUPR 0.79, and raises clarification on ambiguous inputs from 0.09 to 0.81 without harming unambiguous success. The verification bottleneck increases mutual information of language-actions, the GAC loss imposes an entity-level InfoNCE bound, and attention entropy yields calibrated selective prediction, indicating that explicit verified grounding is an effective path toward instruction-sensitive, ambiguity-aware agents. 

---
# Pioneer Agent: Continual Improvement of Small Language Models in Production 

**Authors**: Dhruv Atreja, Julia White, Nikhil Nayak, Kelton Zhang, Henrijs Princis, George Hurn-Maloney, Ash Lewis, Urchade Zaratiana  

**Link**: [PDF](https://arxiv.org/pdf/2604.09791)  

**Abstract**: Small language models are attractive for production deployment due to their low cost, fast inference, and ease of specialization. However, adapting them to a specific task remains a challenging engineering loop, driven not by training itself but by surrounding decisions: data curation, failure diagnosis, regression avoidance, and iteration control. We present Pioneer Agent, a closed-loop system that automates this lifecycle. In cold-start mode, given only a natural-language task description, the agent acquires data, constructs evaluation sets, and iteratively trains models by jointly optimizing data, hyperparameters, and learning strategy. In production mode, given a deployed model with labeled failures, it diagnoses error patterns, constructs targeted training data, and retrains under explicit regression constraints. To evaluate this setting, we introduce AdaptFT-Bench, a benchmark of synthetic inference logs with progressively increasing noise, designed to test the full adaptation loop: diagnosis, curriculum synthesis, retraining, and verification. Across eight cold-start benchmarks spanning reasoning, math, code generation, summarization, and classification, Pioneer Agent improves over base models by 1.6-83.8 points. On AdaptFT-Bench, it improves or preserves performance in all seven scenarios, while naive retraining degrades by up to 43 points. On two production-style deployments built from public benchmark tasks, it raises intent classification from 84.9% to 99.3% and Entity F1 from 0.345 to 0.810. Beyond performance gains, the agent often discovers effective training strategies, including chain-of-thought supervision, task-specific optimization, and quality-focused data curation, purely from downstream feedback. 

---
# CONSCIENTIA: Can LLM Agents Learn to Strategize? Emergent Deception and Trust in a Multi-Agent NYC Simulation 

**Authors**: Aarush Sinha, Arion Das, Soumyadeep Nag, Charan Karnati, Shravani Nag, Chandra Vadhan Raj, Aman Chadha, Vinija Jain, Suranjana Trivedy, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2604.09746)  

**Abstract**: As large language models (LLMs) are increasingly deployed as autonomous agents, understanding how strategic behavior emerges in multi-agent environments has become an important alignment challenge. We take a neutral empirical stance and construct a controlled environment in which strategic behavior can be directly observed and measured. We introduce a large-scale multi-agent simulation in a simplified model of New York City, where LLM-driven agents interact under opposing incentives. Blue agents aim to reach their destinations efficiently, while Red agents attempt to divert them toward billboard-heavy routes using persuasive language to maximize advertising revenue. Hidden identities make navigation socially mediated, forcing agents to decide when to trust or deceive. We study policy learning through an iterative simulation pipeline that updates agent policies across repeated interaction rounds using Kahneman-Tversky Optimization (KTO). Blue agents are optimized to reduce billboard exposure while preserving navigation efficiency, whereas Red agents adapt to exploit remaining weaknesses. Across iterations, the best Blue policy improves task success from 46.0% to 57.3%, although susceptibility remains high at 70.7%. Later policies exhibit stronger selective cooperation while preserving trajectory efficiency. However, a persistent safety-helpfulness trade-off remains: policies that better resist adversarial steering do not simultaneously maximize task completion. Overall, our results show that LLM agents can exhibit limited strategic behavior, including selective trust and deception, while remaining highly vulnerable to adversarial persuasion. 

---
# Head-wise Modality Specialization within MLLMs for Robust Fake News Detection under Missing Modality 

**Authors**: Kai Qian, Weijie Shi, Jiaqi Wang, Mengze Li, Hao Chen, Yue Cui, Hanghui Guo, Ziyi Liu, Jia Zhu, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09711)  

**Abstract**: Multimodal fake news detection (MFND) aims to verify news credibility by jointly exploiting textual and visual evidence. However, real-world news dissemination frequently suffers from missing modality due to deleted images, corrupted screenshots, and similar issues. Thus, robust detection in this scenario requires preserving strong verification ability for each modality, which is challenging in MFND due to insufficient learning of the low-contribution modality and scarce unimodal annotations. To address this issue, we propose Head-wise Modality Specialization within Multimodal Large Language Models (MLLMs) for robust MFND under missing modality. Specifically, we first systematically study attention heads in MLLMs and their relationship with performance under missing modality, showing that modality-critical heads serve as key carriers of unimodal verification ability through their modality specialization. Based on this observation, to better preserve verification ability for the low-contribution modality, we introduce a head-wise specialization mechanism that explicitly allocates these heads to different modalities and preserves their specialization through lower-bound attention constraints. Furthermore, to better exploit scarce unimodal annotations, we propose a Unimodal Knowledge Retention strategy that prevents these heads from drifting away from the unimodal knowledge learned from limited supervision. Experiments show that our method improves robustness under missing modality while preserving performance with full multimodal input. 

---
# How LLMs Might Think 

**Authors**: Joseph Gottlieb, Ethan Kemp, Matthew Trager  

**Link**: [PDF](https://arxiv.org/pdf/2604.09674)  

**Abstract**: Do large language models (LLMs) think? Daniel Stoljar and Zhihe Vincent Zhang have recently developed an argument from rationality for the claim that LLMs do not think. We contend, however, that the argument from rationality not only falters, but leaves open an intriguing possibility: that LLMs engage only in arational, associative forms of thinking, and have purely associative minds. Our positive claim is that if LLMs think at all, they likely think precisely in this manner. 

---
# Digital hybridity and relics in cultural heritage: using corpus linguistics to inform design in emerging technologies from AI to VR 

**Authors**: Emma McClaughlin, Glenn McGarry, Alan Chamberlain, Geert De Wilde, Oliver Butler  

**Link**: [PDF](https://arxiv.org/pdf/2604.09669)  

**Abstract**: Hybrid technologies enable the blending of physical and digital elements, creating new ways to experience and interact with the world. Such technologies can transform engagement with relics, both secular and sacred but they present challenges for capturing faith, belief, and representation responsibly. Given the complexities of digital representation and the ethical challenges inherent in digitising culturally significant objects, a transdisciplinary understanding of these issues is needed. To inform this discussion from a linguistic perspective, we examined the representation of relics in historical and contemporary texts. Using a corpus linguistic approach to extract modifiers of the word relic in corpora of Early Modern English books and contemporary web sourced texts from 2021, we examined the multifaceted ways in which relics have been perceived and evaluated over time. Early texts consider relics as both objects of moral and spiritual significance, and tools of religious and political control, while they are more often framed as heritage symbols, reflecting past events, places, and traditions in contemporary texts. We discuss how hybrid, sometimes AI based technologies can enhance accessibility and engagement, whilst also challenging traditional sensitivities around authenticity and sensory experience, which are integral to the meaning and significance of relics. 

---
# Explainability and Certification of AI-Generated Educational Assessments 

**Authors**: Antoun Yaacoub, Zainab Assaghir, Anuradha Kar  

**Link**: [PDF](https://arxiv.org/pdf/2604.09622)  

**Abstract**: The rapid adoption of generative artificial intelligence (AI) in educational assessment has created new opportunities for scalable item creation, personalized feedback, and efficient formative evaluation. However, despite advances in taxonomy alignment and automated question generation, the absence of transparent, explainable, and certifiable mechanisms limits institutional and accreditation-level acceptance. This chapter proposes a comprehensive framework for explainability and certification of AI-generated assessment items, combining self-rationalization, attribution-based analysis, and post-hoc verification to produce interpretable cognitive-alignment evidence grounded in Bloom's and SOLO taxonomies. A structured certification metadata schema is introduced to capture provenance, alignment predictions, reviewer actions, and ethical indicators, enabling audit-ready documentation consistent with emerging governance requirements. A traffic-light certification workflow operationalizes these signals by distinguishing auto-certifiable items from those requiring human review or rejection. A proof-of-concept study on 500 AI-generated computer science questions demonstrates the framework's feasibility, showing improved transparency, reduced instructor workload, and enhanced auditability. The chapter concludes by outlining ethical implications, policy considerations, and directions for future research, positioning explainability and certification as essential components of trustworthy, accreditation-ready AI assessment systems. 

---
# LLM Nepotism in Organizational Governance 

**Authors**: Shunqi Mao, Wei Guo, Dingxin Zhang, Chaoyi Zhang, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.09620)  

**Abstract**: Large language models are increasingly used to support organizational decisions from hiring to governance, raising fairness concerns in AI-assisted evaluation. Prior work has focused mainly on demographic bias and broader preference effects, rather than on whether evaluators reward expressed trust in AI itself. We study this phenomenon as LLM Nepotism, an attitude-driven bias channel in which favorable signals toward AI are rewarded even when they are not relevant to role-related merit. We introduce a two-phase simulation pipeline that first isolates AI-trust preference in qualification-matched resume screening and then examines its downstream effects in board-level decision making. Across several popular LLMs, we find that resume screeners tend to favor candidates with positive or non-critical attitudes toward AI, discriminating skeptical, human-centered counterparts. These biases suggest a loophole: LLM-based hiring can produce more homogeneous AI-trusting organizations, whose decision-makers exhibit greater scrutiny failure and delegation to AI agents, approving flawed proposals more readily while favoring AI-delegation initiatives. To mitigate this behavior, we additionally study prompt-based mitigation and propose Merit-Attitude Factorization, which separates non-merit AI attitude from merit-based evaluation and attenuates this bias across experiments. 

---
# Assessing the Pedagogical Readiness of Large Language Models as AI Tutors in Low-Resource Contexts: A Case Study of Nepal's K-10 Curriculum 

**Authors**: Pratyush Acharya, Prasansha Bharati, Yokibha Chapagain, Isha Sharma Gauli, Kiran Parajuli  

**Link**: [PDF](https://arxiv.org/pdf/2604.09619)  

**Abstract**: The integration of Large Language Models (LLMs) into educational ecosystems promises to democratize access to personalized tutoring, yet the readiness of these systems for deployment in non-Western, low-resource contexts remains critically under-examined. This study presents a systematic evaluation of four state-of-the-art LLMs--GPT-4o, Claude Sonnet 4, Qwen3-235B, and Kimi K2--assessing their capacity to function as AI tutors within the specific curricular and cultural framework of Nepal's Grade 5-10 Science and Mathematics education. We introduce a novel, curriculum-aligned benchmark and a fine-grained evaluation framework inspired by the "natural language unit tests" paradigm, decomposing pedagogical efficacy into seven binary metrics: Prompt Alignment, Factual Correctness, Clarity, Contextual Relevance, Engagement, Harmful Content Avoidance, and Solution Accuracy. Our results reveal a stark "curriculum-alignment gap." While frontier models (GPT-4o, Claude Sonnet 4) achieve high aggregate reliability (approximately 97%), significant deficiencies persist in pedagogical clarity and cultural contextualization. We identify two pervasive failure modes: the "Expert's Curse," where models solve complex problems but fail to explain them clearly to novices, and the "Foundational Fallacy," where performance paradoxically degrades on simpler, lower-grade material due to an inability to adapt to younger learners' cognitive constraints. Furthermore, regional models like Kimi K2 exhibit a "Contextual Blindspot," failing to provide culturally relevant examples in over 20% of interactions. These findings suggest that off-the-shelf LLMs are not yet ready for autonomous deployment in Nepalese classrooms. We propose a "human-in-the-loop" deployment strategy and offer a methodological blueprint for curriculum-specific fine-tuning to align global AI capabilities with local educational needs. 

---
# Unifying Ontology Construction and Semantic Alignment for Deterministic Enterprise Reasoning at Scale 

**Authors**: Hongyin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09608)  

**Abstract**: While enterprises amass vast quantities of data, much of it remains chaotic and effectively dormant, preventing decision-making based on comprehensive information. Existing neuro-symbolic approaches rely on disjoint pipelines and struggle with error propagation. We introduce the large ontology model (LOM), a unified framework that seamlessly integrates ontology construction, semantic alignment, and logical reasoning into a single end-to-end architecture. LOM employs a construct-align-reason (CAR) pipeline, leveraging its unified architecture across all three stages: it first autonomously constructs a domain-specific ontological universe from raw data, then aligns neural generation with this structural reality using a graph-aware encoder and reinforcement learning, and finally executes deterministic reasoning over the constructed topology, node attributes and relation types. We evaluate LOM on a comprehensive benchmark constructed from diverse real-world enterprise datasets. Experimental results demonstrate that LOM-4B achieves 88.8% accuracy in ontology completion and 94% in complex graph reasoning tasks, significantly outperforming state-of-the-art LLMs. These findings validate that autonomous logical construction is essential for achieving deterministic, enterprise-grade intelligence. 

---
# CID-TKG: Collaborative Historical Invariance and Evolutionary Dynamics Learning for Temporal Knowledge Graph Reasoning 

**Authors**: Shuai-Long Lei, Xiaobin Zhu, Jiarui Liang, Guoxi Sun, Zhiyu Fang, Xu-Cheng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.09600)  

**Abstract**: Temporal knowledge graph (TKG) reasoning aims to infer future facts at unseen timestamps from temporally evolving entities and relations. Despite recent progress, existing approaches still suffer from inherent limitations due to their inductive biases, as they predominantly rely on time-invariant or weakly time-dependent structures and overlook the evolutionary dynamics. To overcome this limitation, we propose a novel collaborative learning framework for TKGR (dubbed CID-TKG) that integrates evolutionary dynamics and historical invariance semantics as an effective inductive bias for reasoning. Specifically, CID-TKG constructs a historical invariance graph to capture long-term structural regularities and an evolutionary dynamics graph to model short-term temporal transitions. Dedicated encoders are then employed to learn representations from each structure. To alleviate semantic discrepancies across the two structures, we decompose relations into view-specific representations and align view-specific query representations via a contrastive objective, which promotes cross-view consistency while suppressing view-specific noise. Extensive experiments verify that our CID-TKG achieves state-of-the-art performance under extrapolation settings. 

---
# DeepReviewer 2.0: A Traceable Agentic System for Auditable Scientific Peer Review 

**Authors**: Yixuan Weng, Minjun Zhu, Qiujie Xie, Zhiyuan Ning, Shichen Li, Panzhong Lu, Zhen Lin, Enhao Gu, Qiyao Sun, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09590)  

**Abstract**: Automated peer review is often framed as generating fluent critique, yet reviewers and area chairs need judgments they can \emph{audit}: where a concern applies, what evidence supports it, and what concrete follow-up is required. DeepReviewer~2.0 is a process-controlled agentic review system built around an output contract: it produces a \textbf{traceable review package} with anchored annotations, localized evidence, and executable follow-up actions, and it exports only after meeting minimum traceability and coverage budgets. Concretely, it first builds a manuscript-only claim--evidence--risk ledger and verification agenda, then performs agenda-driven retrieval and writes anchored critiques under an export gate. On 134 ICLR~2025 submissions under three fixed protocols, an \emph{un-finetuned 196B} model running DeepReviewer~2.0 outperforms Gemini-3.1-Pro-preview, improving strict major-issue coverage (37.26\% vs.\ 23.57\%) and winning 71.63\% of micro-averaged blind comparisons against a human review committee, while ranking first among automatic systems in our pool. We position DeepReviewer~2.0 as an assistive tool rather than a decision proxy, and note remaining gaps such as ethics-sensitive checks. 

---
# Generative UI: LLMs are Effective UI Generators 

**Authors**: Yaniv Leviathan, Dani Valevski, Matan Kalman, Danny Lumen, Eyal Segalis, Eyal Molad, Shlomi Pasternak, Vishnu Natchu, Valerie Nygaard, Srinivasan, Venkatachary, James Manyika, Yossi Matias  

**Link**: [PDF](https://arxiv.org/pdf/2604.09577)  

**Abstract**: AI models excel at creating content, but typically render it with static, predefined interfaces. Specifically, the output of LLMs is often a markdown "wall of text". Generative UI is a long standing promise, where the model generates not just the content, but the interface itself. Until now, Generative UI was not possible in a robust fashion. We demonstrate that when properly prompted and equipped with the right set of tools, a modern LLM can robustly produce high quality custom UIs for virtually any prompt. When ignoring generation speed, results generated by our implementation are overwhelmingly preferred by humans over the standard LLM markdown output. In fact, while the results generated by our implementation are worse than those crafted by human experts, they are at least comparable in 50% of cases. We show that this ability for robust Generative UI is emergent, with substantial improvements from previous models. We also create and release PAGEN, a novel dataset of expert-crafted results to aid in evaluating Generative UI implementations, as well as the results of our system for future comparisons. Interactive examples can be seen at this https URL 

---
# ACE-TA: An Agentic Teaching Assistant for Grounded Q&A, Quiz Generation, and Code Tutoring 

**Authors**: Himanshu Tripathi, Charlottee Crowell, Kaley Newlin, Subash Neupane, Shahram Rahimi, Jason Keith  

**Link**: [PDF](https://arxiv.org/pdf/2604.09572)  

**Abstract**: We introduce ACE-TA, the Agentic Coding and Explanations Teaching Assistant framework, that autonomously routes conceptual queries drawn from programming course material to grounded Q&A, stepwise coding guidance, and automated quiz generation using pre-trained Large Language Models (LLMs). ACE-TA consists of three coordinated modules: a retrieval grounded conceptual Q&A system that provides precise, context-aligned explanations; a quiz generator that constructs adaptive, multi-topic assessments targeting higher-order understanding; and an interactive code tutor that guides students through step-by-step reasoning with sandboxed execution and iterative feedback. 

---
# EvoDiagram: Agentic Editable Diagram Creation via Design Expertise Evolution 

**Authors**: Tianfu Wang, Leilei Ding, Ziyang Tao, Yi Zhan, Zhiyuan Ma, Wei Wu, Yuxuan Lei, Yuan Feng, Junyang Wang, Yin Wu, Yizhao Xu, Hongyuan Zhu, Qi Liu, Nicholas Jing Yuan, Yanyong Zhang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.09568)  

**Abstract**: High-fidelity diagram creation requires the complex orchestration of semantic topology, visual styling, and spatial layout, posing a significant challenge for automated systems. Existing methods also suffer from a representation gap: pixel-based models often lack precise control, while code-based synthesis limits intuitive flexibility. To bridge this gap, we introduce EvoDiagram, an agentic framework that generates object-level editable diagrams via an intermediate canvas schema. EvoDiagram employs a coordinated multi-agent system to decouple semantic intent from rendering logic, resolving conflicts across heterogeneous design layers. Additionally, we propose a design knowledge evolution mechanism that distills execution traces into a hierarchical memory of domain guidelines, enabling agents to retrieve context-aware expertise adaptively. We further release CanvasBench, a benchmark consisting of both data and metrics for canvas-based diagramming. Extensive experiments demonstrate that EvoDiagram exhibits excellent performance and balance against baselines in generating editable, structurally consistent, and aesthetically coherent diagrams. Our code is available at this https URL. 

---
# LETGAMES: An LLM-Powered Gamified Approach to Cognitive Training for Patients with Cognitive Impairment 

**Authors**: Jingwei Shi, Shengyu Tao, Xinxiang Yin, Chen Huang, Wenqiang Lei, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.09566)  

**Abstract**: The application of games as a therapeutic tool for cognitive training is beneficial for patients with cognitive impairments. However, effective game design for individual patient is resource-intensive. To this end, we propose an LLM-powered method, \ours, for automated and personalized therapeutic game design. Inspired by the Dungeons & Dragons, LETGAMES generates an open-world interactive narrative game. It not only generates game scenarios and challenges that target specific cognitive domains, but also employs conversational strategies to offer guidance and companionship. To validate its efficacy, we pioneer a psychology-grounded evaluation protocol LETGAMESEVAL, establishing comprehensive metrics for rehabilitative assessment. Building upon this, our experimental results from both LLM-based assessors and human expert evaluations demonstrate the significant potential of our approach, positioning LETGAMES as a promising solution to the widespread need for more accessible and tailored cognitive training tools. Our code will be open-sourced upon acceptance. 

---
# Seven simple steps for log analysis in AI systems 

**Authors**: Magda Dubois, Ekin Zorer, Maia Hamin, Joe Skinner, Alexandra Souly, Jerome Wynne, Harry Coppock, Lucas Satos, Sayash Kapoor, Sunischal Dev, Keno Juchems, Kimberly Mai, Timo Flesch, Lennart Luettgau, Charles Teague, Eric Patey, JJ Allaire, Lorenzo Pacchiardi, Jose Hernandez-Orallo, Cozmin Ududec  

**Link**: [PDF](https://arxiv.org/pdf/2604.09563)  

**Abstract**: AI systems produce large volumes of logs as they interact with tools and users. Analysing these logs can help understand model capabilities, propensities, and behaviours, or assess whether an evaluation worked as intended. Researchers have started developing methods for log analysis, but a standardised approach is still missing. Here we suggest a pipeline based on current best practices. We illustrate it with concrete code examples in the Inspect Scout library, provide detailed guidance on each step, and highlight common pitfalls. Our framework provides researchers with a foundation for rigorous and reproducible log analysis. 

---
# LABBench2: An Improved Benchmark for AI Systems Performing Biology Research 

**Authors**: Jon M Laurent, Albert Bou, Michael Pieler, Conor Igoe, Alex Andonian, Siddharth Narayanan, James Braza, Alexandros Sanchez Vassopoulos, Jacob L Steenwyk, Blake Lash, Andrew D White, Samuel G Rodriques  

**Link**: [PDF](https://arxiv.org/pdf/2604.09554)  

**Abstract**: Optimism for accelerating scientific discovery with AI continues to grow. Current applications of AI in scientific research range from training dedicated foundation models on scientific data to agentic autonomous hypothesis generation systems to AI-driven autonomous labs. The need to measure progress of AI systems in scientific domains correspondingly must not only accelerate, but increasingly shift focus to more real-world capabilities. Beyond rote knowledge and even just reasoning to actually measuring the ability to perform meaningful work. Prior work introduced the Language Agent Biology Benchmark LAB-Bench as an initial attempt at measuring these abilities. Here we introduce an evolution of that benchmark, LABBench2, for measuring real-world capabilities of AI systems performing useful scientific tasks. LABBench2 comprises nearly 1,900 tasks and is, for the most part, a continuation of LAB-Bench, measuring similar capabilities but in more realistic contexts. We evaluate performance of current frontier models, and show that while abilities measured by LAB-Bench and LABBench2 have improved substantially, LABBench2 provides a meaningful jump in difficulty (model-specific accuracy differences range from -26% to -46% across subtasks) and underscores continued room for performance improvement. LABBench2 continues the legacy of LAB-Bench as a de facto benchmark for AI scientific research capabilities and we hope that it continues to help advance development of AI tools for these core research functions. To facilitate community use and development, we provide the task dataset at this https URL and a public eval harness at this https URL. 

---
# MCERF: Advancing Multimodal LLM Evaluation of Engineering Documentation with Enhanced Retrieval 

**Authors**: Kiarash Naghavi Khanghah, Hoang Anh Nguyen, Anna C. Doris, Amir Mohammad Vahedi, Daniele Grandi, Faez Ahmed, Hongyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09552)  

**Abstract**: Engineering rulebooks and technical standards contain multimodal information like dense text, tables, and illustrations that are challenging for retrieval augmented generation (RAG) systems. Building upon the DesignQA framework [1], which relied on full-text ingestion and text-based retrieval, this work establishes a Multimodal ColPali Enhanced Retrieval and Reasoning Framework (MCERF), a system that couples a multimodal retriever with large language model reasoning for accurate and efficient question answering from engineering documents. The system employs the ColPali, which retrieves both textual and visual information, and multiple retrieval and reasoning strategies: (i) Hybrid Lookup mode for explicit rule mentions, (ii) Vision to Text fusion for figure and table guided queries, (iii) High Reasoning LLM mode for complex multi modal questions, and (iv) SelfConsistency decision to stabilize responses. The modular framework design provides a reusable template for future multimodal systems regardless of underlying model architecture. Furthermore, this work establishes and compares two routing approaches: a single case routing approach and a multi-agent system, both of which dynamically allocate queries to optimal pipelines. Evaluation on the DesignQA benchmark illustrates that this system improves average accuracy across all tasks with a relative gain of +41.1% from baseline RAG best results, which is a significant improvement in multimodal and reasoning-intensive tasks without complete rulebook ingestion. This shows how vision language retrieval, modular reasoning, and adaptive routing enable scalable document comprehension in engineering use cases. 

---
