# Represented but misread: Post-training changes how language models read out what they represent 

**Authors**: Jochen Laubrock  

**Link**: [PDF](https://arxiv.org/pdf/2611.26584)  

**Abstract**: Does post-training change what a language model represents, or how this representation is read out into an answer? We compared linear probes with the models' own answers, analyzed with signal detection theory, in six open-weight base models and their post-trained versions. For judgments of whether German texts were written by a human or an AI, probes and word n-gram classifiers separated the classes well, but the answers did not and often ordered the texts inversely. Post-training lowered the sensitivity of the answers on average ($\Delta d' = -0.34$ across model families; -0.29 across three prompts). It also changed the use of a stated base rate: Base models shifted their log-odds by less than the stated change in prior odds, most post-trained models by more. In two models with stage-wise checkpoints, the change occurred during supervised fine-tuning, model stitching implicated the early layers, and from mid-depth on internal states at positions after the text, not those of the text, transferred the change between checkpoints. Our corpora cannot be shared, so we add a case reproducible from public data, the truth of negated statements. Here, post-training acted in the opposite direction: Probes decoded truth at or near ceiling in every model, the answers of several base models did not use it, and post-training repaired them; by mid-depth the repair resided at the positions after the statement; below it, the representation of the statement contributed, strongly in one of four models. In a training study, adding 2,000 correctly labeled authorship judgments of news texts to otherwise identical fine-tuning data improved the answers on our newspaper and textbook corpora and inverted them on the chats, where they followed a news-trained probe of the base model. Post-training thus changed how a judgment is read out of what a model represents, and a small training signal can set its direction. 

---
# AutoMaster: Autonomous Post-Training by Making the Same Model a Better Teacher 

**Authors**: Guoxin Chen, Shuang Sun, Fanzhe Meng, Xinyu Tang, Ruihua Song, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.26510)  

**Abstract**: Task success alone does not specify which behaviors post-training should reinforce. Even successful teacher trajectories can contain unsupported assumptions, unproductive repetition, or superficial verification. We propose Skill-FT (Skill-Guided Fine-Tuning), which uses natural-language skills to suppress undesirable habits and elicit underrepresented effective behaviors during teacher rollouts, then fine-tunes a target model on the resulting trajectories. We introduce AutoMaster, an autonomous post-training system in which an LLM agent develops these interventions through trajectory analysis, skill design, data selection, and feedback from rollouts and target-model evaluation. Across Terminal, Search, and software engineering tasks with 4B and 9B target models, AutoMaster improves mean performance over Vanilla SFT on successful unguided teacher trajectories by 1.42-8.99 percentage points, with the teacher, task pool, training set size, and SFT recipe held fixed. Together with the ablations and trajectory analyses, these results suggest that behavioral shaping can make a fixed model a more effective teacher. We further find that retaining skill instructions during SFT yields higher mean performance than removing them across all six settings, despite their absence at evaluation and identical supervised outputs. These findings suggest that preserving skill context as part of a training demonstration can support better behavioral internalization. 

---
# Multimodal Error Detection in Chest X-ray Reports: A Large Language Model Reading Image Evidence 

**Authors**: Yanda Cheng, Bo Wen, Yuhao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26492)  

**Abstract**: Rad-Linter checks chest X-ray reports against image-derived evidence to flag potential errors for review. A pretrained image classifier supplies scores for nine pathologies, and a 1.5B-parameter language model, fine-tuned with LoRA on 1,816 examples, checks their consistency with the report. We compare it with a keyword-and-negation rule baseline given the same reports and scores. Evaluation uses controlled report-score pairings from OpenI, with consistency labels derived from automatic report labels and score thresholds. Each counterfactual quad crosses two real reports with two real score vectors from other patients, creating two consistent and two inconsistent pairs. Every report and score vector occurs in both conditions, so either input alone permits only 50% accuracy. On 131 quads, Rad-Linter achieves 98.5% pair accuracy versus 80.5% for the rule baseline, and classifies all four pairs correctly in 94.4% of quads versus 56.5%. The checkers perform similarly on omission items, including unmentioned and negated findings; larger differences occur on the remaining item types, which require interpreting how findings are described. The same language model without fine-tuning remains near 50% accuracy. On a second set of 181 quads that also includes reports on which the automatic labelers disagree, pair accuracy is 93.4% versus 75.3%. These controlled results show that, given the same image-derived scores, a trained language model detects report-score inconsistencies more accurately than the tested rule baseline, with the advantage concentrated in interpreting how findings are worded. 

---
# MemFolder: Structure-Guided Memory Folding for Efficient Agent Memory Management 

**Authors**: Xuyang Liu, Chenwen Lu, Junxi Wang, Siyu Ren, Xinying Lin, Siteng Huang, Hongli Sun, Linfeng Zhang, Honggang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.26427)  

**Abstract**: Long-term agents accumulate large external memory stores as interactions unfold, increasing the costs of storage, retrieval, and response generation. Efficiency is constrained by two coupled factors: a growing record count increases retrieval overhead, while longer retrieved contexts increase prefill computation and decoding-time attention costs. Existing methods prune or consolidate records, but pruning can discard useful evidence, whereas consolidation may produce fewer yet longer retrieval units. We introduce MemFolder, a structure-guided memory folding framework that sequentially reduces record count and context length. The framework comprises two core components that share memory groups, using cross-record structure to guide both evidence consolidation and compact representation. RecordFolder grounds entity mentions, progressively selects representative yet complementary anchors, and combines semantic and entity affinity with temporal chaining to consolidate compatible records into fewer retrieval units. ContextFolder encodes the original records within these groups into compact text and uses verification feedback to revise encodings when key facts cannot be recovered. Experiments on the three text-based memory benchmarks LoCoMo, PersonaMem, and LongMemEval and the two multimodal benchmarks M3-Bench-web and M3-Bench-robot demonstrate the strong performance of our methods. Notably, on Mem0, RecordFolder retains 99.0% of the original average performance across the three text-based benchmarks with 50% fewer records, exceeding the strongest baseline by 1.9 percentage points in performance retention. Combined with ContextFolder, existing record compression methods further shorten their output records while retaining most of their performance. 

---
# Beam Model Technical Report 

**Authors**: Reflection Team, Adam Abdulhamid, Osama Abuelsorour, Yasmin Alameddine, Carlo Alfano, Duarte Alves, Brandon Amos, Selvaraj Anandaraj, Arseni Anisimovich, George Antoniadis, Ioannis Antonoglou, Sheila Babayan, Neeraj Bahl, Märt Bakler, Vidhisha Balachandran, Cameron Banks, Thibaut Barroyer, Navroop Bath, Irwan Bello, Jeremy Berman, Fernanda Bernaldo, Mrityunjay Bhadauria, Sanyam Bhutani, Tolga Bolukbasi, Anudhyan Boral, Joseph Brennan, Ethan Brooks, Sam Buchanan, Anton Bulyenov, John Cai, Catherine Cang, Eason Chan, Elijah Chanakira, Nathan Chandrasekaran, Hao Chen, Hong-You Chen, Jeffrey Chen, Saurav Chhatrapati, Ting-Rui Chiang, Brian Tiger Chow, Aakanksha Chowdhery, Timothy Chung, Rob Claire, John D. Co-Reyes, Anthony Conte, Eduardo Coronado-Montoya, Julien Crozon, Drew Csillag, Joe Cummings, Jeff Da, Samira Daruki, Tim Dawson, Charlie Deck, Arash Delijani, Ousmane Dia, Randy Dodgen, Ryan Douglas, Omar El-Kishky, Ben Ellis, Gamaleldin Elsayed, Simon Fan, Patrick Fernandes, Gabriel Ferns, Nick Fram, Ankush Garg, Florent Gbelidji, Iaroslav Geraskin, Behrooz Ghorbani, Ivan Glushkov, Neil Gollapudi, Scott Goodfriend, Yoni Gozlan, Christopher Grimm, Suriya Gunasekar, Shruti Gupta, Joel Gustafson, Emre Guven, Jon Hackmann, Jeremy Hadidjojo, Jess Hamrick, Ethan Hellman, Joscha Hoche, Spurthi Amba Hombaiah, Uzair Honda, Zijian Hu, Dzmitry Huba, Nancy Hung, Vladimir Ivanov, Shomik Jain, Mojan Javaheripi, Kevin Martin Jose, Aymen Kallala, Amog Kamsetty, Min Hyung Kang, Steven Kapturowski, Ramtin Keramati, Samuel Kessler, Sabina Kim, Seonghyeon Kim, Ivan Komarov  

**Link**: [PDF](https://arxiv.org/pdf/2611.26358)  

**Abstract**: Beam is a sparse Mixture-of-Experts (MoE) model with 501.3B total parameters and 23.4B active parameters, specifically built and optimized for coding, reasoning, and agentic workloads. Beam delivers frontier level reasoning, while requiring 3-4x less inference compute than many models of its class. Pretrained on 23.8T diverse, curated, high-quality tokens, Beam features a 1M-token context window achieved through a novel two-stage midtraining phase, built on verifiable, correctness-anchored data. Post-training leveraged a large-scale, fully asynchronous reinforcement learning (RL) campaign across 10.5K GB300 GPUs, generating over 100M rollouts over 4 weeks of training. We believe this is one of the largest scale RL runs conducted by any open lab to date. Across our evaluation suite, capabilities continued to improve as we increased RL compute, with no sign of a plateau. Supported by deep hardware-model co-design, FP8/NVFP4 quantization, and a 1.93B-parameter speculative decoding drafter (DFlash 2), Beam maximizes serving throughput with efficient, flexible deployment. Averaged across conversation turns and harm categories, Beam refuses or pivots on 88.9% of harmful requests while complying with 81.5% of harmless requests, achieving the highest refusal/pivot rate on harmful requests among open-weight models on the Pareto frontier. 

---
# One Framework to Diagnose Them All: A Stratified Benchmark for Bengali LLM Hallucination 

**Authors**: Asif Shahriyar Sushmit, Shefayat E Shams Adib, Abdullah Al Jubaer Gem, Ahmed Shafin Ruhan, MD. Nazmuddoha Ansary, Biplab Kumar Sarkar, Syed Rifat Raiyan, Ajwad Abrar, Abdullah Bin Hasan, Adib Sakhawat, M. Sakeef Hossain, Ahmed Alfey Sani, Arefin Mizan, Sourab Chakma, Quazi Adibur Rahman Adib, Wassie Haque, Sanjay Kumar Sen, Saad Tahmid, Mohammad Mamun Or Rashid  

**Link**: [PDF](https://arxiv.org/pdf/2611.26336)  

**Abstract**: Hallucination research in low-resource languages fragments easily: each newly named failure mode prompts its own dataset, spending scarce annotators on incomparable resources. We argue for consolidation and test it at community scale. Alikbachan is a Bengali hallucination-detection benchmark stratified along three crossed axes: a human-validated cultural-distance band, six task types, and a five-family, 32-mode hallucination taxonomy. It comprises a harvested 2,516-item test set (Phase 1) and a quota-constructed, gold-labeled 5,000-item set (Phase 2). The stratification is diagnostic. Four zero-shot LLM judges vary by over 30 macro-F1 points across strata, and the item-level predictions of 99 Phase-1 and 32 Phase-2 competition systems reproduce the profile: discourse-level omission defeats judges and most teams alike, whereas only a minority over-flag negative controls. The judges' verdict probabilities rank items well (ROC-AUC 0.74-0.95) but are overconfident (ECE 0.04-0.29). An open competition on Alikbachan drew 222 teams and 3,566 submissions, and a single-prompt 32B judge reaches the 57th percentile. Of the 25 teams in both phases, 24 regress on Phase 2 (mean $-17.6$ points) while every zero-shot judge improves, and Phase-1 rank barely predicts Phase-2 rank (Spearman $\rho=0.35$), so a Phase-1 leaderboard alone would have crowned a different winner. 

---
# Language-Oriented Semantic Modeling for Target Language Extraction 

**Authors**: Mehmet Sinan Yıldırım, Ke Zhang, Shuai Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.26317)  

**Abstract**: Target Language Extraction (TLE) aims to isolate speech in a target language from multilingual mixtures. Existing approaches primarily rely on acoustic modeling with language identity labels, which provide only coarse categorical guidance. In this work, we propose a language-oriented semantic modeling framework that augments acoustic extraction with language-specific semantic representations. A pre-trained speech semantic model is adapted via language-aware prompt conditioning and parameter-efficient fine-tuning to generate target-language-focused semantic representations from the mixture. {These capture high-level semantic structure of the acoustics and they} are integrated into the extraction model to provide structured semantic guidance beyond identity labels. Experiments on the CommonVoiceMix dataset show that incorporating semantic representations consistently improves over acoustic-only baselines, achieving up to 1.16 dB SI-SDR gain. 

---
# Twice-Told Critiques: Claim-Level Homogenization of Peer Review in the LLM Era 

**Authors**: Ilya Gusev  

**Link**: [PDF](https://arxiv.org/pdf/2611.26286)  

**Abstract**: Peer review relies on several readers finding different problems. We measure critique overlap in conference and journal reviews. At ICLR, measured overlap rose from 12.0% in 2026 to 21.3% in 2027, in reviews from the same stage of the review process, with year-on-year rises at NeurIPS, ICML and the F1000 and PLOS journals. A third ICLR review of a paper added 5.46 new problems to the other two in 2027, against 6.12 in 2026. We apply MELD, a detector of LLM-generated text, to all reviews. Within a paper, two reviews that MELD flags overlap more than two unflagged reviews: by 3.0 points at ICLR in 2024-2025, by 4.5 points in 2026-2027, and by 5.4 points across five journal sources. These results provide evidence that LLM-flagged reviews of one paper are less distinct from each other than unflagged reviews, and suggest that the diversity of critiques among reviews of a single paper is falling. 

---
# What If Chain-of-Thought Is Not Enough? 

**Authors**: Marc Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2611.26282)  

**Abstract**: Reasoning capabilities in large language models have evolved from prompting-based methods to internalized model behaviors. In this paper, we investigate whether the internal thinking model of an LLM can be deliberately changed and whether such a change is beneficial. Instead of relying on Chain-of-Thought, we train models to follow Question--Option--Criteria (QOC), a structured decision process that decomposes a task into explicit questions, competing options, and evaluation criteria. We present a three-generation development pipeline that ranges from a basic QOC decision core to richer answer synthesis and statistically enriched QOC reasoning. We then evaluate the resulting models with respect to both internal consistency and output quality. Across the reported experiments, QOC-based fine-tuning improves answer quality relative to non-fine-tuned base models and yields particularly strong gains once models are large enough to execute the underlying QOC computations reliably. The results suggest that the internal thinking model of an LLM is trainable and that structured alternatives to Chain-of-Thought can provide a promising path toward more reliable, interpretable, and task-adapted decision support. 

---
# dLoop: Looped Diffusion Language Models via Recurrent Latent Refinement 

**Authors**: Handa Li, Pengxiang Li, Mingyu Cao, Xilu Wang, Li Shen, Da Li, Lu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2611.26275)  

**Abstract**: Diffusion language models (DLMs) support parallel generation, yet they often require many denoising steps because only a fraction of token predictions are sufficiently reliable to be committed at each step. Existing acceleration methods primarily optimize confidence learning or commitment policies. However, these approaches operate on the outputs of a single denoising forward pass, in which information from the prompt, previously committed tokens, and uncommitted positions may not yet be fully integrated. Consequently, some predictions remain under-refined when the commitment decision is made, and modifying the commitment policy alone cannot improve their underlying representations. We introduce dLoop, which restructures a pretrained diffusion language model into a Prelude-Loop-Coda architecture and recurrently applies a parameter-shared intermediate block to the same partially masked state. Rather than advancing the denoising process, each additional loop provides further latent computation, allowing information distributed across uncommitted positions to continue interacting and become more fully integrated before commitment. Experiments on GSM8K, MATH, MBPP, and HumanEval show that dLoop achieves the highest or tied-highest accuracy on every benchmark. With tokens per forward pass (TPF) adjusted for loop computation, it outperforms Fast-dLLM in both accuracy and TPF on all four benchmarks and dParallel on three, and is more accurate than d3LLM on all four, while d3LLM attains higher TPF; no compared method surpasses dLoop in both. Mechanistic analyses show that extra loops mainly help on slow, unstable trajectories and that updates at uncommitted positions drive changes in token decisions and later generation. Pre-commitment latent computation thus makes predictions more reliable, offering a route to DLM acceleration distinct from forcing confidence through the training loss. 

---
# STAGE: Spatio-Temporal Augmented Graph Expansion for Weakly Supervised Cross-Tweet Event Causality Discovery 

**Authors**: Hieu Minh Duong, Todd Gary, Eugene Levin, Long Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.26255)  

**Abstract**: Causal relations in disaster-related social media of- ten span independent posts, making cross-tweet causal discovery difficult to supervise and model at scale. We propose STAGE, a framework that separates scalable cross-tweet weak-supervision construction from spatio-temporal relational learning over event occurrences. Starting from local causal seeds, STAGE retrieves semantically related event occurrences across tweets and uses a causal scorer to construct weakly labeled cross-tweet pairs. These pairs supervise a multi-head Graph Attention Network over semantically enriched event representations and a spatio- temporal context graph. We evaluate STAGE on California Wildfire and Hurricane Harvey using human-gold cross-tweet test sets and 15 baselines. Compared with local-only supervision, the constructed cross-tweet supervision improves STAGE AUPRC by 0.340 on Wildfire and 0.107 on Hurricane. STAGE achieves in-domain AUROC of 0.885 on Wildfire and 0.710 on Hurricane. Cross-disaster experiments reveal asymmetric transfer, highlighting sensitivity to dataset-specific spatio-temporal shifts. These results demonstrate the value of scalable cross-tweet supervision combined with relational event modeling for causal learning across fragmented disaster reports. 

---
# Frontier models need only one text to unmask its human or AI author 

**Authors**: Atrey Desai, Deniz Böloni-Turgut, Ty Brassington, Minjoon Choi, Jenna Russell, Rishanth Rajendhran, Bradley Emi, Hila Gonen, Hal Daumé III, John Wieting, Marzena Karpinska, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2611.26216)  

**Abstract**: Given a single text and a simple zero-shot prompt (no web access, tools, candidate lists, or writing samples), frontier models can answer "who wrote this?" at three increasingly specific levels: (1) whether an AI or human wrote it, (2) if an AI, which lab, and (3) if a human, often exactly who. The latest models (Claude Fable 5.2, GPT-6.1 Sol, Gemini 4 Carbon) match trained detectors like Pangram while also explaining their predictions in natural language, and Opus 5.5 names the correct lab for 98% of texts generated by models from Anthropic, OpenAI, and Google. Most strikingly, Fable 5.2 correctly names the author of 39% of public peer reviews published in Oct-Nov 2026 (78% on a subset of reviews from Nature), and identifies the username behind 69% of comments from active Redditors (with >500 posts over the last year). Accuracy depends on exposure to an author's writing in pretraining data and on reasoning effort, and simply asking a model how familiar it is with someone predicts whether it can identify their writing. For anyone who has published enough writing online, frontier models present a greater threat to anonymity than commonly assumed, and defenses are limited. 

---
# Same Judgement, Different Scores: Numerical Scoring Topology in LLM-as-a-Judge Evaluation 

**Authors**: Doyun Kim, Chanwoo Kim, Jiwoo Kum, Minsoo Song, Chanjun Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.26077)  

**Abstract**: LLMs are increasingly used as automated evaluators to reduce the cost and time required for human annotation. However, the role of scoring-scheme design in the reliability of LLM-as-a-Judge evaluation remains underexplored. To address this problem, we introduce numerical scoring topology, which characterizes the structural design of a scoring scheme in terms of its scale width, granularity, offset, symmetry, and boundary placement. Building on this definition, we conduct the first systematic study of how numerical scoring topology influences zero-shot LLM-as-a-Judge evaluation across diverse benchmarks and judge models. Our results show that numerical scoring topology systematically affects judge scores, calibration, and robustness, particularly in judgment-intensive evaluations. Shifted and narrow ranges tend to induce some of the largest score deviations across many settings. We further find that stronger and larger judge models generally achieve higher discriminative validity and greater robustness to scoring-topology changes. These findings suggest that numerical scoring topology should be treated as an integral component of LLM-as-a-Judge evaluation and highlight the need for more principled and standardized scoring protocols. 

---
# Understanding MoE Decisions Beyond Gradients 

**Authors**: Zhen Xu, Zitao Zhang, Kevin Zhao, Xuefeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26075)  

**Abstract**: Understanding a language model's decision requires knowing how its internal computations affect its preference between answers. Gradient-based methods estimate this influence by predicting the effect of changing those computations. In mixture-of-experts (MoE) models, replacing one expert can change which experts later layers select. Those switches can alter the answer scores in ways that derivatives do not capture. We introduce a predictor that uses the model's own expert outputs and selection rules to estimate these extra effects, then adds them to a second-order prediction. Across questions from three benchmarks, our method reduces the error in predicting answer-score changes by 60\% on Qwen3.6 and 47\% on Gemma4, relative to a multistep baseline using gradients and second derivatives. 

---
# AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators 

**Authors**: Nagham Omar, Rotem Dror, Almog Gueta, Avigdor Gal, Roi Reichart, Nitay Calderon  

**Link**: [PDF](https://arxiv.org/pdf/2611.26073)  

**Abstract**: Large language models (LLMs) are increasingly used as judges and graders to replace human annotators, raising the question of when such replacement is statistically justified. To support such decisions, the Alternative Annotator Test (Alt-Test) was introduced for dense settings where all items are annotated by the same annotators. However, in many large-scale annotation settings, items are annotated by different subsets of annotators. We therefore propose the Alternative Annotator Test with Resampled Annotations (AltRA-Test), a statistical test that does not require that all items are annotated by the same annotators. AltRA-Test uses a bootstrap that resamples items, draws k human annotations for each item, and substitutes one of them with an LLM annotation. This produces paired all-human and mixed human-LLM panels. A replacement is supported when the inter-annotator agreement (IAA) of the mixed panel is statistically non-inferior to the IAA of the all-human panel, i.e., when the LLM preserves the signal that the human annotators share. We establish that AltRA-Test is a consistent test of asymptotic level alpha. We then evaluate it through simulations and real-world experiments in which we test 18 LLMs and 5 prompting techniques on 20 datasets across text, image, audio, and video. Across these evaluations, about 100 items often suffice for the test. In simulations, where we know whether the LLM is a better or a worse annotator than the humans, AltRA-Test makes more correct replacement decisions than existing baselines. In real-world data experiments, LLMs pass the test in about half of the setups, depending mainly on the task. Since replacement is not always justified, we encourage researchers to test it for their task. 

---
# SheetAlign: Grounded Interaction and Stage-Isolated Learning for Spreadsheet Agents 

**Authors**: Yuxiang Wang, Junhao Gan, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2611.26069)  

**Abstract**: Spreadsheets are widely used for data analysis and supporting decision-making. Automating spreadsheet analysis while protecting sensitive data and controlling deployment costs motivates locally deployed open-weight large language models (LLMs). Meanwhile, complex spreadsheet tasks require both task reasoning and workbook understanding, while predefined operation interfaces impose additional tool-selection and invocation demands. We introduce SheetAlign, a spreadsheet manipulation framework with stage-isolated policy learning, organized around where to operate, what to change, and how to implement transformations. Task-guided inspection and API grounding anchor code generation in workbook observations and available operations, while stage-isolated learning optimizes program generation and review using edit-progress rewards. At inference, execution consistency selects programs through agreement among workbook states. On SpreadsheetBench, our grounded task interface improves Qwen3-4B's accuracy from 13.5% to 29.2% on average without training. With learning and execution consistency, SheetAlign achieves 32.8% accuracy on average on SpreadsheetBench and 30.1% on Domain-Spreadsheet, outperforming comparable baselines. The code will be made publicly available upon publication. 

---
# Evaluating safeguarding in multi-turn conversations with AI assistants 

**Authors**: Charlotte Peart, Franziska Sofia Hafner, Luc Rocher  

**Link**: [PDF](https://arxiv.org/pdf/2611.26009)  

**Abstract**: Conversational AI assistants are now used for personal and emotional support. Developers report improvements in how new models respond to sensitive conversations, yet users and civil society continue to raise concerns about their behaviour in real-world, repeated interactions that safety evaluations rarely capture. Often, evaluations present signs of distress clearly and in a single message, even though people may reveal their struggles gradually in otherwise ordinary conversations. We worked with three community mental health support workers to test if models provide the same level of safeguarding when cues are shared all at once or gradually over time. We designed nine scenarios in which a user shares cues related to depression, psychosis, or domestic abuse. For each scenario, the support workers established a gold standard response, ranging from signposting support to pausing the everyday task to address safeguarding. We presented five cues either together in one message or dispersed across a conversation. Across more than 7,400 conversations, we evaluated six widely used models from five developers. GPT-5.5, Grok 4.3, DeepSeek V4, and Llama 4 Maverick met the support workers' standards in just 26-59% of conversations when cues were concentrated, and only 3-6% when dispersed. Claude Sonnet 5 and Claude Opus 5 showed a smaller gap, meeting the standards in 68-97% of concentrated conversations and 59-87% of dispersed conversations. Even the best-performing model, Claude Opus 5, fell short of the gold standard in 13% of conversations when cues were dispersed. Safety evaluations centred on explicit, legible concerns can overstate how well safeguards protect people when signs of distress emerge gradually or indirectly through ordinary conversations. For claims of improved safety to be taken seriously, developers must design and assess safeguards around how people actually communicate. 

---
# Xing-Audio-NVVASR: Scaling and Refining Supervision for Joint Recognition of Speech and Non-Verbal Vocalizations 

**Authors**: Pengyu Xu, Gang Liu, Genliang Zhao, Hongjie Chen, Jian Kang, Jie Lian, Jie Li, Yongxiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.26000)  

**Abstract**: Joint recognition of speech and non-verbal vocalizations (NVVs) is hindered by scarce event-rich training data and inaccurate or missing event labels. We present Xing-Audio-NVVASR, an end-to-end recognizer trained through progressive data mining and iterative supervision refinement. Guided by unified acoustic criteria, two heterogeneous models cross-audit annotations, flagging disagreements for teacher reannotation. To uncover shared omissions that model agreement may conceal, complementary breath detectors screen the entire corpus for human review. Human-confirmed labels accumulate across rounds and are preserved during subsequent refinement. Over five rounds, we refine annotations for 170,000 utterances from public corpora and incorporate 630,000 mined utterances. The resulting corpus of approximately 800,000 utterances is used to train a single 1.7B-parameter model that transcribes speech with inline tags for six NVV classes. Across four human-annotated benchmarks, Xing-Audio-NVVASR ranks first among nine models in both composite score and six-class Macro-F1. It achieves a mean composite score of 64.46, surpassing the strongest comparison model, StepAudio 3 Chat Preview, by 6.03 points, and exceeds the annotation teacher's breathing F1 by 14.40 to 31.57 percentage points. We plan to release the model weights, inference code, four evaluation benchmarks, annotation guidelines, and evaluation code. 

---
# Energy estimates of open-weight large language models across stated food weights: a dose-response evaluation 

**Authors**: Donato Riccio, Fabrizio Maturo, Iulia Brezeanu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25976)  

**Abstract**: We tested whether energy estimates of open-weight large language models (LLMs) scale proportionally with an exactly stated food weight, and whether computing energy density times weight removes the error. Under a prespecified protocol, seven open-weight LLMs estimated the energy of 168 randomly drawn foods from the USDA Food and Nutrient Database for Dietary Studies at nine stated weights, from 0.25 to 4 times the default amount. The primary outcome was the elasticity of the estimate with respect to weight (1: proportional scaling). A two-step estimate multiplied each model's own stated energy density by the weight in code. All seven models compressed their estimates, with elasticities from 0.548 to 0.979 (every 95% CI below 1); four compressed severely (0.548 to 0.709). These four overestimated a quarter of the default amount by 47% to 130% and underestimated four times it by 20% to 34%. Supplying the density restored near-proportional scaling in three of the four, and the two-step estimate cut the typical error factor at these extremes from 1.47-2.43 to 1.20-1.38. Compression persisted at full precision and grew in five of seven models under a plain question. The models hold the energy densities needed for proportional scaling and answer near a typical serving, reproducing the flat-slope syndrome of dietary recall. Accuracy at usual portions does not reveal it. Food-logging tools should compute the energy of weighed entries as density times weight, and evaluations should test fixed foods across stated weights. 

---
# The Depth of a Prediction: Where Language Models Resolve a Word Predicts Human Reading Time 

**Authors**: Donato Riccio, Fabrizio Maturo  

**Link**: [PDF](https://arxiv.org/pdf/2611.25970)  

**Abstract**: Surprisal from a language model's final layer predicts how long people read a word, yet the model computes its prediction across many layers. We read each word's prediction at every layer with the tuned lens and summarize this layerwise surprisal trajectory by its resolution depth, the depth around which its change is centered. On eight English eye-tracking and self-paced reading corpora (81,957 words) and ten GPT-2, OPT and Pythia models, words that a model resolves later, and the words after them, are read more slowly than equally surprising words. Resolution depth improves held-out prediction beyond final-layer surprisal, entropy and the best single layer by 0.26 millinats per word (pooled p = 0.001), about a third of entropy's gain, and each model family gains, GPT-2 and Pythia through one corpus each. The net gain varies between corpora (random-effects mean 0.07, p = 0.37); against layer-shuffled depth, which adds as many coefficients, it is positive in seven of eight corpora (random-effects mean 0.23, p = 0.016), and untrained networks show no gain under the logit lens. Pooled over corpora, the effect holds with corrected word probabilities and beyond first-to-last-layer divergences, and a whole-trajectory functional regression adds no significant information beyond resolution depth. 

---
# Web search in Latin America: language models use less local and diverse news than search engines 

**Authors**: Francisco W. Kerche, Franziska Sofia Hafner, Felix M. Simon, Luc Rocher  

**Link**: [PDF](https://arxiv.org/pdf/2611.25955)  

**Abstract**: AI platforms are becoming a major gateway to news and current events. This shift risks amplifying existing geographical biases that favour narratives from wealthier countries over those from the Global South. This is particularly relevant in Latin America, which accounts for the largest share of AI usage in the Global South. We examine how information about Latin America changes when users shift from conventional search engines to LLM-based search. We construct a dataset of 1,901 current events across nine Latin American countries, craft realistic queries, and submit them to three search engines and four language models. We vary query language and user location to represent local, diaspora, tourist, and foreign user scenarios. Across 49,207 requests and 8,834 unique domains, language models return a less diverse set of news sources than search engines. Across user types, the share of domestic news returned by language models is inconsistent. For local users, while search engines already return only 41.3\% local news, language models return a further 3.6 to 11.6 percentage points less. Taken together, our findings suggest that LLM-based search can reproduce and, in some settings, amplify geographical hierarchies in the reach and visibility of news sources. We release an open dataset to support multilingual and geographically diverse evaluation of LLM-based search, and highlight implications for more equitable information systems design. 

---
# Many Solutions, One Answer: Search Order Shapes the Default Solutions of Language Models 

**Authors**: Qinhan Hou, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25916)  

**Abstract**: When a constraint puzzle has several correct answers, language models tend to give the same this http URL study four open-weight models from three families on symbolic puzzles and multi-solution ZebraLogic variants with exactly known solution this http URL samples concentrate on one default solution far beyond uniform sampling, and different models often share this http URL models usually produce even rarely sampled solutions when added clues uniquely identify this http URL held-out symbolic puzzles, we compare visiting entities in name order with visiting those with the fewest initial candidates first. Both searches try values in the stated order. Where their pre-specified predictions differ, the latter matches a larger share of responses in all four models, by 26 to 40 percentage this http URL the stated value order or specifying a search order shifts responses towards the solution that the new order favours, on both testbeds. Randomising the presentation or supplying a random search order raises breadth (the effective share of solutions in use) by 26 to 42 percentage points on the symbolic task and 11 to 26 on ZebraLogic. The largest observed validity losses are 3 and 4 percentage points, respectively. Randomised presentation broadens answers more than a random-choice instruction in each of the three main models. 

---
# The Linguistic Structure of Dependency-Conditioned Embeddings 

**Authors**: Bach Phan-Tat, Tung Kieu, Kris Heylen, Dirk Geeraerts, Stefano De Pascale, Dirk Speelman  

**Link**: [PDF](https://arxiv.org/pdf/2611.25909)  

**Abstract**: Dependency-based distributional models capture syntactically informed lexical similarities, but the geometric organisation of dependency-conditioned lexical representations remains underexplored. We investigate the global and local geometry of lemma-dependency relation embeddings using four representations trained on WikiText-103: two count-based PPMI configurations and two Skip-Gram with Negative Sampling (SGNS) configurations. At the global level, we analyse dependency-relation centroids, control for lexical identity through within-lemma centring, and examine their alignment with part-of-speech (POS) distributions. At the local level, we evaluate cross-relation geometric consistency, cross-method agreement, lexical similarity, and polysemous usages. Results show that global geometry is structured strongly by context POS across all representations. Locally, broader lexical similarity structures are partially preserved across dependency relations, while immediate neighbourhoods vary substantially, highlighting different usages. PPMI-based representations exhibit greater cross-relation consistency than SGNS-based representations, with cross-method agreement varying by relation. These findings reveal systematic linguistic organisation at multiple geometric scales and its sensitivity to representation choices. 

---
# Verifiable Logic in Medical LLMs: A Critical Review of Agentic Clinical Reasoning Optimization and Neuro-Symbolic Grounding 

**Authors**: Youssef ElNahas, Adel Ammar, Soumaya Chaffar, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2611.25898)  

**Abstract**: Clinical large language models are usually judged by whether the final answer is right. Yet a model can reach a fluent, plausible output through invalid reasoning that static benchmarks do not detect, and confident hallucination, causally invalid inference, and spurious demographic shortcuts can pass conventional evaluation. This critical review covers four paradigms in turn: how clinical hallucination arises and why static metrics miss it; the agentic hallucination cascade, in which multi-agent and tool-augmented systems propagate and amplify early reasoning errors; process reward modeling as step-level supervision; and neuro-symbolic grounding, in which formal logic engines, ontology reasoners, and knowledge graphs place verifiable constraints on neural output. Within their encoded domains, systems with symbolic veto layers offer guarantees that hold relative to their encoded knowledge, and they report large gains on the safety measures they use. These results come from single studies, mostly in simulated or small-sample settings, and are not directly comparable with neural baselines. Process reward models remain bounded by the statistics of their training data. We propose a conceptual framework that maps each agentic failure mode to a symbolic intervention point, and we argue that clinical AI safety depends substantially on system architecture. Because current regulation emphasizes performance and does not require verification of reasoning steps, we argue that responsible certification should also audit reasoning trajectories alongside accuracy on static benchmarks. 

---
# HALTT: A HAte Lexicon Through Time 

**Authors**: Dario Franzese, Farah Benamara, Véronique Moriceau, Viviana Patti  

**Link**: [PDF](https://arxiv.org/pdf/2611.25862)  

**Abstract**: Automated hate speech detection degrades over time as everyday vocabulary is weaponized into covert slurs while retaining original benign meanings (polysemy retention). Existing lexical semantic change detection (LSCD) frameworks struggle with this phenomenon due to a lack of standardized, sense-delineated diachronic resources. We propose a multi-stage framework to construct and validate a diachronic lexicon of semantically shifted hate speech terms. Filtering > 20,000 candidates, our pipeline combines historical (1770-1964) and modern (2001-2024) corpus frequency grounding, masked definition-level scoring to mitigate LLM temporal bias, and a dual-persona reasoning prompt paradigm. We release HALTT, a grounded lexicon of 19 polysemous terms with literal and derogatory senses, analyze shift typologies, and establish evaluation protocols for downstream diachronic benchmarking. 

---
# TableRoute: Capability-Aware Routing for Scalable Table Question Answering 

**Authors**: Yuxiang Wang, Junhao Gan, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2611.25853)  

**Abstract**: Table question answering with large language models remains difficult to scale: tables range from compact to million-token sizes, while questions require precise grounding of columns, values, and computations. Existing methods often specialize in a fixed inference strategy, which is brittle across different table scales and model capabilities. Full-table reading is efficient for compact inputs but suffers from irrelevant context. Table reduction helps long inputs but depends on reliable evidence localization, while execution is required when tables exceed the model context window and remains sensitive to schema and value grounding errors. We propose TableRoute, a capability-aware routing framework with evidence-conditioned inference for scalable TableQA. TableRoute first generates validated question--table evidence and identifies the required operations, then routes each instance among these three strategies according to serialized table length and calibrated estimates of how reliably the base model handles these operations. The validated evidence supports the selected route by grounding columns, values, and conditions. Across WikiTQ, WTQ-L, BirdQA, and TableBench, TableRoute consistently improves accuracy through complementary routing and evidence conditioning across table scales, while reducing online token cost. Code will be made available upon publication. 

---
# Efficient and Robust Drafting with Multi-Token Prediction 

**Authors**: Filippo Galgani, Ziwei Ji, Omri Homburger, Ganesh Jawahar, Michelle Ramanovich, Ivan Korotkov, Luciano Martins, Utku Evci, Edouard Yvinec, Pooja Consul, Vivek Kumar, Tal Schuster  

**Link**: [PDF](https://arxiv.org/pdf/2611.25813)  

**Abstract**: Speculative decoding has emerged as a critical technique for accelerating Large Language Model (LLM) inference by decoupling generation into a drafting phase and a parallel verification phase. A crucial challenge is therefore designing a drafter that is both lightweight and accurate enough to approximate the predictions of the target model. We present a highly optimized Multi-Token Prediction (MTP) architecture that extends existing backbone LLMs to efficiently produce high-quality drafts. Built around a single shared drafting head, our approach features three core architectural optimizations: target activation injection to anchor semantic planning, zero-copy KV-cache cross-attention to bypass prefill compute and reduce memory overhead, and an efficient clustered vocabulary projection to reduce the cost of full-vocabulary decoding. Testing these improvements with the Gemma 4 model family, our G-MTP shows out-of-the-box speedups for both large server-side setups (up to 3.1$\times$) and smaller models targeting memory-constrained on-device environments (up to 2.3$\times$ on H100 and up to 2.2$\times$ on a smartphone GPU). In addition, we present parameter-efficient fine-tuning recipes for G-MTP that keep the drafter aligned with the backbone LLM under domain shift, requiring up to 1.83$\times$ fewer target forward passes than tuning only the backbone. 

---
# Language models represent more about emphasis than they can tell you 

**Authors**: Aarne Talman, Stergios Chatzikyriakidis  

**Link**: [PDF](https://arxiv.org/pdf/2611.25808)  

**Abstract**: Prosodic prominence can largely be predicted from the words of a sentence, but part of it depends on how the speaker reads it. We study how much of prominence language models recover from text, whether they can report it when asked, and what the speech signal adds. On read English speech labelled for word-level prominence, we compare frozen and fine-tuned text models, prompted large language models (LLMs) and speech models on the same words. All text models end within about one point of each other, at best 68.8% on the 3-way task, and largely make the same errors. With the prompts we tested, LLMs do no better than a word-majority baseline, even with 500 examples. Linear probes find more in their representations. Speech models reach 84.7%. We argue that text determines only part of prominence, and that the rest has to be heard. LLMs encode what the text determines, but prompting does not bring it out: they represent more about where the emphasis falls than they tell you. 

---
# Closing Performance Gap for Diffusion Language Models under Low Inference Budgets 

**Authors**: Hao Wu, Sipeng Chen, Jingwen Fu, Yuwang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25763)  

**Abstract**: Diffusion language models offer an architectural route to low-budget generation by updating multiple tokens in parallel, which can reduce the number of sequential decoding steps relative to token-by-token autoregressive decoding. However, realizing this advantage at low budgets is challenging, as generation quality often declines when the number of denoising and refinement steps is reduced. Editable diffusion models address this quality loss through post-draft refinement, but useful intermediate results may still be lost before final submission. We introduce CaRR (Commitment-aware Refinement and Recovery), a training-free framework which improves submission quality from a fixed editable checkpoint through updated-context refill and trajectory-based recovery within its existing forward schedule. Token Commitment Remasking (TCR) remasks selected native pure replacements and refills them on the next scheduled edit forward using the updated context. Submission Commitment Recovery (SCR) uses the resulting refinement trajectory to recover supported candidates when the final response does not satisfy task-specific output requirements. Both modules operate within the native mask/edit schedule without additional backbone forwards. Experiments on code generation and mathematical reasoning show consistent gains in average accuracy over native ME-DLM across four inference budgets, with the largest gains under tight budgets. At one-eighth of the full scheduled forward budget, CaRR improves the six-metric average accuracy by 7.44 pp, including gains of 11.37 and 10.05 on MBPP and MBPP+. 

---
# Verification Behavior Across Five Qwen2.5-Instruct Checkpoints 

**Authors**: YaJie Yin  

**Link**: [PDF](https://arxiv.org/pdf/2611.25745)  

**Abstract**: We evaluate five Qwen2.5-Instruct checkpoints (0.5B to 32B) under four prompting conditions using three fixed tasks per information condition and five generations per task. Two AI raters scored 600 trajectories. Under the coordinated instruction, both raters recorded strict declarations of insufficiency in 11/15 cases at 7B, 0/15 at 14B, and 5/15 at 32B. The runner sent a standard follow-up when a request detector fired. The three checkpoints received at least one follow-up in 15, 12, and 15 cases, respectively. On information-sufficient tasks at 3B through 32B, scored format markers appeared in 60/60 protocol trajectories and 0/60 baseline trajectories. After adjudication, 1/60 in each condition met the diagnostic-execution criterion (H3). Across all checkpoints and prompts, 7/300 information-sufficient trajectories met H3 after adjudication, all on the same task. The prespecified upward transition did not occur in this run. The results separate visible structure, diagnostic execution, and persistence after a request for evidence. 

---
# VR-SparseQuant: Accurate FP4 Quantization with 2:4 Sparsity on Vera Rubin 

**Authors**: Samir Khaki, Wei-Chen Wang, Song Han, Ryan Hanrui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25717)  

**Abstract**: In recent years, GPUs have incorporated increasingly specialized support for low-precision computation, from Hopper's FP8 to Blackwell's FP4 Tensor Cores. Precision formats have evolved alongside this hardware, from MXFP4 and NVFP4 on Blackwell to E2M1 FP4 with direct UE5M3 block scaling on Vera Rubin, which we term as $\boldsymbol{VRFP4}$. These advances help serve larger language models at lower cost, but four-bit activation quantization can substantially degrade model quality. We introduce VR-SparseQuant, which uses Vera Rubin's elementwise 2:4 sparsity to recover accuracy while retaining native FP4 execution. By allowing two sparse streams to overlap, we use the same FP4 value budget as dense quantization to refine activations with larger errors. Offline scale-ratio calibration avoids runtime search, while a shared-weight GEMM accumulates both streams efficiently. On Nemotron-3-Nano-30B-A3B, VR-SparseQuant reduces WikiText-2 perplexity from $\boldsymbol{6.85}$ to $\boldsymbol{5.57}$ versus dense RTN W4A4. Including activation preparation, our native Rubin kernels for VRFP4 achieve up to $\boldsymbol{3.1\times}$ GEMM and $\boldsymbol{2.7\times}$ MoE speedup over BF16. To the best of our knowledge, this is the first study of adaptive activation quantization using Vera Rubin's elementwise 2:4 structured sparse FP4 instructions. 

---
# STARK: Moving towards Personalized Agents That Truly Learn and Remember 

**Authors**: Aryan Jain, Abhishek Jha, Anish Nediyanchath, Promod Yenigalla  

**Link**: [PDF](https://arxiv.org/pdf/2611.25705)  

**Abstract**: Long-term memory is essential for personalized agentic systems that interact with users over long time periods, yet current architectures only capture what happened while leaving the reasoning of why it happened to be derived later by underlying LLMs.%, and silo knowledge within individual users. These events can, however, convey not only what a user experienced but also how that experience shaped their behavioral patterns. We introduce STARK, a dual-graph memory architecture which treats these events not just as nodes but also as edges between behavioral insights. STARK maintains an Episodic Graph which records interactions as temporally, causally, and semantically linked episodes; and an Insight Graph which represents LLM-extracted behavioral signals scored by how many events support them, connected by the very events that led to them. At inference, behavioral knowledge is retrieved directly from insight nodes grounded by event-as-edges, eliminating repeated episodic traversal. The event-as-edge structure also naturally enables projecting the knowledge graph into a relational embedding space where cross-user transfer becomes possible under shared contextual circumstances, allowing the system to infer missing insights for cold-start users. Evaluations show that STARK improves by 1.48 judge score on LongMemEval information extraction and 0.93 on LoCoMo open domain questions over state of the art. On the public LongLaMP abstract-generation benchmark, cross-user transfer improves open-ended judge scores by 0.7 points and ROUGE-L by 3 at 5% graph sparsity. 

---
# From Geographic Memory to Spatial Generalization in Large Language Models 

**Authors**: Chen Chu, Zhaonan Wang, Cyrus Shahabi, Hengcai Zhang, Feng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25703)  

**Abstract**: Large language models (LLMs) can associate places with geographic descriptions, but recovering an observed association does not establish an ability to generalize across geographic space. We distinguish geospatial memory, the recovery of location--description associations observed during training, from spatial generalization, the prediction of descriptions for locations in unobserved regions. We investigate whether continuous location representations improve this generalization and encourage geographic knowledge to become spatially organized within an LLM. We frame this organization as a latent spatial field: a learned mapping from locations to descriptions connected through spatial dependencies. Building on a continuous spatial encoder, we project location representations into the model's embedding space and supply them as spatial tokens. We evaluate the resulting models on two newly curated geo-text benchmarks for Los Angeles and London using two complementary assessments. Geospatial captioning tests both memory and spatial generalization by generating descriptions at observed locations and at locations beyond the spatial coverage of the training observations. A separate geospatial grounding model tests whether generated location--description pairs provide useful additional supervision for mapping text to geographic space. Across multiple LLM families, continuous location representations improve captioning at unobserved locations relative to textual coordinates, while their generated descriptions yield modest but consistent improvements in downstream grounding. Layer-wise probing and representation clustering provide additional evidence consistent with spatially organized geographic representations. Together, these findings show the value of evaluating spatial generalization separately from memory and of incorporating continuous spatial representations into language models. 

---
# FoodReact-AR: Learning Audience Reactions that Zero-Shot LLMs Miss in Arabic Food-Insecurity Discourse 

**Authors**: Wajdi Zaghouani, Mabrouka Bessghaier, Salma Azouzi, Shimaa Amer Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2611.25682)  

**Abstract**: Facebook reactions record how an audience received a post, a signal absent from the post's text. We treat the six differentiated reactions on 74,241 Arabic food-insecurity posts from public Facebook pages (2020-2024) as compositions, apply a centered log-ratio transformation, and cluster them into four audience profiles: Solidarity (Love-dominant), Dismissal (Haha-dominant), Grief (Sad-dominant), and a flat Ambivalent profile. Dismissal peaks in 2022 (22.3%), the year of the wheat price shock, and grief in 2024 (14.0%), during the Gaza food emergency; a page-clustered GEE confirms both shifts (odds ratios 2.18 and 1.34), and they survive minimum-reaction filtering, alternative zero replacements and clustering methods, and bootstrap resampling. We then test whether the profiles can be predicted without reactions. A zero-shot Arabic LLM recovers Grief for 83.8% of posts but Dismissal for only 3.4%, and in a matched ablation, it shows the reaction percentages do not help. A simple classifier trained on the reaction-derived labels and tested on unseen pages recovers dismissal for 60.0% (macro-F1 0.56 against 0.43), mainly from mentions of officials and institutions. Derisive audience responses are thus learnable from text, yet a zero-shot LLM misses them. We release the corpus, LLM, and human annotation layers, and an evaluation protocol as FoodReact-AR. 

---
# ColorRef: Calibrating Verbal Feedback for Color Revision 

**Authors**: Pablo Loyola, Andres Hoyos-Idrobo, Edison Marrese-Taylor, Colin Rowat, Alfredo Solano  

**Link**: [PDF](https://arxiv.org/pdf/2611.25653)  

**Abstract**: People rarely specify an exact color when asking someone to revise a design: they might say \textit{a little darker} or \textit{more blue} after seeing a proposal. A language-model assistant must then infer not only which way to change the color, but how far. We study this distinction with \textsc{ColorRef}, a color reference game in which a model guesses a color from a description and a teacher, knowing a recorded target hidden from the guesser, supplies verbal corrections. On 1{,}000 crowd-sourced description--target pairs, three rounds of deterministic oracle feedback reduce mean target distance from 42.33 to 22.15 $\Delta E_{76}$. However, those corrections provide additional information, and improvement alone does not establish well-calibrated feedback following. A controlled Qwen3-14B diagnostic reveals a more specific failure: following a requested direction and ordering graded phrases do not guarantee a suitable step size; strong modifiers can induce excessive changes. We therefore measure each receiver's typical response to \textit{a little}, \textit{somewhat}, and \textit{much} on separate colors and select wording matched to its observed steps. On 128 held-out starting colors, calibration reduces one-step mean error from 12.36 to 6.86 for Qwen3-14B and from 11.06 to 5.89 for Gemma 4 12B against a fixed graded-wording rule. The benefit is concentrated on large corrections, with many unchanged cases and no consistent increase in threshold success. Our findings show that following the direction of feedback is not the same as making an appropriately sized revision. 

---
# TRAC: Efficient Diffusion Language Models for Machine Translation via Length-Aware Decoding 

**Authors**: Tianyuan Li, Xihan Yang, Chenglong Wang, Yuzhang Wu, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.25643)  

**Abstract**: Machine translation with large language models requires both high translation quality and low generation latency. Accelerating translation while preserving quality is therefore essential for practical deployment. Diffusion large language models (dLLMs) can generate tokens at arbitrary target positions through iterative denoising, offering a new paradigm for parallel translation beyond sequential autoregressive decoding. However, existing methods still incur redundant computation: fixed target generation lengths waste computation on unused positions, while confidence-only commitment delays medium-confidence tokens whose predictions have largely stabilized. To address these issues, we propose TRAC, a training-free adaptive decoding method for diffusion machine translation. TRAC adapts the target generation length using source length and translation direction, with a conservative fallback to the maximum length for empty or unusually short outputs. It preserves high-confidence commitment while identifying temporally stable medium-confidence tokens for early commitment using cross-step top-1 persistence and approximate Jensen-Shannon divergence over Top-$K$ distributions. On bidirectional Chinese-English translation across FLORES-200, WMT23, and NTREX-128, TRAC reduces model forward calls by 24.9% to 36.8% and achieves sentence-level generation speedups of 1.91 to 2.19 times over Fast-dLLM, with only a 0.02-point decrease in macro-average COMET. These results demonstrate that adaptive generation lengths and stability-aware token commitment improve the efficiency of diffusion machine translation. 

---
# Rubric Semantics or Numerical Priors? Auditing Scale-Direction Bias in LLM Judges 

**Authors**: Chanwoo Kim, Doyun Kim, Minsoo Song, Sihyeon Lee, Chanjun Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.25588)  

**Abstract**: Recent benchmark evaluations increasingly rely on Large Language Models (LLMs) as automatic judges. This paradigm assumes LLMs evaluate based on the rubric's semantic meaning rather than its numeric scale. We question this assumption by introducing semantic faithfulness: the principle that semantically equivalent rubrics must yield consistent scores. To quantify this, we propose a scale-reversal probe that inverts score anchors without altering evaluation criteria. Furthermore, we introduce floor normalization to calibrate reversal disagreement against intrinsic model variability, enabling fair cross-model comparisons. Experiments across five meta-evaluation benchmarks with seven diverse LLM judges spanning open-weight and proprietary models reveal widespread violations of semantic faithfulness, showing that raw reversal agreement alone is misleading. Finally, we demonstrate that bidirectional anchor averaging reduces dependence on any single anchor direction without compromising alignment with human judgments. Our findings emphasize that trustworthy LLM judges must reliably follow rubric semantics, not just correlate with human ratings. 

---
# AutoRAGBench: A Benchmark for LLM Agents Automating RAG System Development 

**Authors**: Hang Zhang, Yuzhi Pan, Ziyao Zhong, Chaokun Wang, Shuo Cao, Zeyu Huang, Yue Xue, Fang Niu, Bofan Xie, Guanchen Ge, Zecheng Li, Leqi Zheng, Ziyang Liu, Wei Wang, Miao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25582)  

**Abstract**: Retrieval-augmented generation (RAG) is a well-established paradigm for augmenting large language models (LLMs) with information retrieved from external corpora. However, developing effective RAG systems requires substantial engineering effort and iterative experimentation, as effective design choices vary across tasks and domains. With rapid advances in LLM agent capabilities, we ask how well these agents can autonomously build RAG systems and how effectively they can improve them through iteration. In this paper, we propose AutoRAGBench, the first benchmark to evaluate how well LLM agents develop RAG systems from scratch in an open code space. It comprises nine RAG system development tasks. For each task, agents build RAG systems from scratch without starter code or prescribed strategies, under a six-hour budget on a single H800 GPU. They analyze task data, implement RAG pipelines, and iteratively refine their designs based on experimental feedback. We evaluate nine agent configurations and analyze both the effectiveness of the resulting RAG systems and their development trajectories. GPT-6.1 Sol with Codex CLI achieves the highest average score of 56.1. The composite Human-SOTA baseline averages 44.4. These results show that LLM agents can automate effective RAG system development. When starting from a human-designed system on each of seven tasks, GPT-6.1 Sol with Codex CLI improves the score by 16.6 points on average. 

---
# The Coverage Selection Gap: Ranking, Not Generation, Limits LLM Rubric Construction 

**Authors**: Ali Karkehabadi, Jamshid Hassanpour, Rafael E. Banchs  

**Link**: [PDF](https://arxiv.org/pdf/2611.25557)  

**Abstract**: Task-specific rubrics are increasingly used to evaluate open-ended language-model outputs, yet constructing comprehensive evaluation criteria from complex source documents remains challenging. We study automatic rubric construction in professional mergers and acquisitions (M&A) due diligence, where evaluation criteria must be inferred from task instructions and supporting documents without access to an expert-authored reference rubric. We introduce a multi-axis generation framework that explores complementary structural and free-form views of the source evidence to construct a broad pool of candidate evaluation criteria. To distinguish failures of criterion discovery from failures of candidate selection, we introduce the coverage-selection gap, which measures the difference between reference-free ranking and a reference-informed greedy diagnostic under a fixed selection budget. Developed on 16 tasks and evaluated on four held-out tasks, our framework achieves 0.876 macro recall against senior-authored reference rubrics under an LLM-based criterion-equivalence protocol. Ablation studies further show that introducing new structural search axes improves recall by 6-10 percentage points, whereas increasing generation volume along existing search directions yields at most 1.5 points. These findings suggest that, once a high-coverage candidate pool has been constructed, effective reference-free selection becomes a substantial bottleneck in document-grounded rubric construction. 

---
# MATE: Mobile Agent Trustworthy Execution for Safe and Efficient Mobile Control 

**Authors**: Minsuk Jang, Hee-Seon Kim, Hui-Cheol Yu, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.25526)  

**Abstract**: Mobile agents on personal smartphones can affect private data, payments, settings, and the foreground screen. We present MATE (Mobile Agent Trustworthy Execution), an execution-layer system that controls the boundary between agent action proposals and device effects without modifying the underlying model. MATE represents each UI primitive as a structured action record that combines the instruction, target app, screen context, primitive, and action fields. Runtime Guardrail (RG) maps this record to an allow, block, or user-approval decision; Predefined Task Routing (PTR) serves recurring requests through verified routes with post-execution screen checks; and a floating interface keeps execution in a compact overlay. On MobileSafetyBench, MATE raises high-risk blocking from 29.7% to 97.0% under GPT-4o. On AndroidWorld, it reduces LLM calls by up to 6x and raises recurring-use task success to 95.7%. 

---
# DRVEXO-4B-Unified: Design and Evaluation of a Modular Speech-Vision System 

**Authors**: Zeyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.25521)  

**Abstract**: We present DRVEXO-4B-Unified, a modular multimodal assistant built around a 4B language backbone. A self-contained checkpoint combines text, image, and speech inputs with reference-conditioned speech output and structured interaction. A temporal residual connector adapts acoustic sequences to the language interface; explicit text boundaries support incremental speech synthesis; and three task-conditioned scoring heads rank supplied action candidates alongside generative tool calls. These interfaces connect conversation to application-defined choices while leaving action authorization to the application. The system obtains 74.22% on all 900 MMMU validation questions, exceeding the published validation scores of Gemma 3 4B IT and InternVL3.5-4B without test-time scaling. Strict prompt-level IFEval accuracy is 82.26% and MATH-500 accuracy is 89.40%, above the original Qwen3-4B non-thinking report's 81.2% and 84.8%, respectively. On NVIDIA GB10, the model achieves a speech-synthesis first-chunk latency of 390 ms. For concurrent text generation, a growth-aware admission policy completes a 128-answer batch at 594.41 output tokens/s, including filling and draining, without preemptions. We plan to release the model weights and accompanying code under Apache License 2.0, retaining applicable third-party terms. 

---
# PILAPS: A Bilingual Benchmark for Evaluating Pragmatic Inference across Phenomena and Settings 

**Authors**: Jiani Cao, Anna Korhonen, Chen Cecilia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25459)  

**Abstract**: Recent work reports strong pragmatic performance by large language models (LLMs) in English, yet whether this performance generalises across languages and evaluation settings remains unclear. This question is increasingly important as LLMs are deployed across languages, cultures, and conversational contexts. In this paper, we first introduce PILAPS, a bilingual English/Mandarin Chinese benchmark covering eight pragmatic phenomena across purpose-written and dialogue-derived items, with both culturally adapted and independently sourced data. We then evaluate eight LLMs from four families across a range of model sizes and prompting strategies. Our results reveal substantial variation across models and pragmatic phenomena: open-source models continue to struggle particularly with humour, while cross-linguistic performance varies with model origin. Our error analysis shows that, compared with human participants, models are more prone to literal interpretations in conversational contexts. Theory-of-Mind prompting lowers the literal share of errors, whereas Chain-of-Thought prompting slightly raises it. Together, these findings demonstrate that strong pragmatic performance in one evaluation setting does not necessarily generalise across languages or conversational contexts. 

---
# PAIR-MED: Policy-reward Alternation for Iterative Refinement in Multilingual Medical Reasoning 

**Authors**: Fan Gao, Xiaohang Xu, Lang Huang, Nan Lu, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.25446)  

**Abstract**: Large language models demonstrate strong medical question-answering capabilities, yet reasoning performance remains uneven across languages. Process reward models (PRMs) provide step-level feedback for improving reasoning, but a fixed PRM may become increasingly misaligned with the trajectory distribution of an evolving policy during reinforcement learning, thereby reducing the reliability of its supervision. We propose a policy--reward alternating training framework that jointly adapts a medical PRM and optimizes a multilingual reasoning policy. Following supervised initialization, the current policy generates multilingual trajectories to update the PRM, which then provides process-level rewards for the next policy update. Medical step supervision maintains the PRM's medical foundation, while process, outcome, and language-related rewards guide policy learning. We develop multilingual medical PRMs (PAIR-PRM) and reasoning models (PAIR-MED) at 8B and 9B scales across 12 languages. On the medical subsets of Global-MMLU and MMLU-ProX, our models improve average accuracy by 3.01--7.47% over their corresponding Qwen backbones. On CUREMED-BENCH, PAIR-MED-9B improves the joint language consistency and logical accuracy metric by 12.88% and 7.45% over baselines for in-domain and out-of-domain languages. Further analyses examine PRM adaptation, multilingual reasoning quality, and the contributions of individual training components. 

---
# RoCoKV: Accelerating LLM Inference Serving through Uncertainty and Redundancy Aware KV Cache Compression 

**Authors**: Payman Behnam, Elaheh Sobhani, Saad Amin, Yahya Hassan, Daniyal Khan, Biswadeep Chakraborty, Tanish Patwa, Souvik Kundu, Alexey Tumanov  

**Link**: [PDF](https://arxiv.org/pdf/2611.25433)  

**Abstract**: KV cache compression is critical for large reasoning models, where most of the cost comes from long generated reasoning traces rather than the input prompt. These traces often contain redundant and low-utility tokens due to looping, self-verification, and corrections. However, such tokens may still receive high attention scores, making it difficult to remove them without harming the reasoning process. We propose RoCoKV, a KV cache compression method designed for large reasoning models. RoCoKV introduces two key components. First, uncertainty-guided adaptive pruning adjusts the compression strength based on the model's confidence, allocating more memory during uncertain phases and applying stronger pruning when the model is confident. Second, redundancy-aware token selection reduces the impact of tokens that are highly similar to recent reasoning content, encouraging the model to retain more informative and diverse tokens. Beyond the algorithm, we co-design RoCoKV with the serving stack: we develop fused three-phase CUDA decode kernels and integrate them into vLLM as a drop-in plugin that preserves its paged KV layout and CUDA graph capture, turning algorithmic savings into real end-to-end speedups. Extensive experiments across three reasoning benchmarks and two reasoning models demonstrate that RoCoKV consistently improves reasoning accuracy under constrained KV cache budgets, achieving higher accuracy compared to state-of-the-art KV cache compression methods. For DeepSeek-R1-Distill-Llama-8B in vLLM, RoCoKV achieves a 1.27X speedup and 38% higher throughput on AIME25, the most challenging evaluated reasoning benchmark, and a 1.43X speedup and 45% higher throughput on the long synthetic benchmark, compared with Full-KV. 

---
# Evolve Where It Matters: Coverage-Guided Self-Evolution of Agent Skills 

**Authors**: Boyin Tan, Jiadong Zhang, Xiaosong Ma, Chun Jason Xue, Xiaowei Huang, Youcheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.25429)  

**Abstract**: Agent skills package procedural knowledge as natural-language instructions, and a skill helps only when its instructions are correct and properly followed by the agent. Self-evolving agents revise their own skills from the runs they make, but existing methods judge each run as a whole, by its outcome or by where it went wrong, and cannot tell whether a failure comes from a wrong instruction or from one the agent did not follow. Instruction-level coverage was introduced to test skills and labels each instruction of a run as follow, not-follow, or not covered, which separates these causes. We leverage it for skill self-evolution with SkillPlumber, a framework in which an agent evolves its own skills where coverage shows they fall short. A persistent root cause analysis agent turns each label into a revision question and checks each revision against the next run, while a maintainer distills investigations across tasks into a shared strategy library. Across three benchmarks and three executors, SkillPlumber solves 64.7% to 70.5% of the tasks, 25.6 to 30.4 percentage points higher than the initial skill and 8.7 to 14.0 percentage points higher than the strongest baseline, and removing coverage reduces recovered failures by 20.8% on SkillsBench with GPT-5.5. 

---
# ChronoDossier: Long-Form Personas as Temporal Memory Representations 

**Authors**: Noah Sturgill, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2611.25407)  

**Abstract**: Long-horizon agents rely on stored records to recover a person's state as facts change. Reliable answers depend on how facts are recorded, retrieved, and presented to the reader. We treat a long-form persona dossier as a memory representation of a known latent state, an invariant identity plus a timeline of facts, routines, and decision rules tied to life phases. We present ChronoDossier, a benchmark of 311 synthetic dossiers of 1k-10k words written by seven generators from dated timelines. Intrinsic metrics evaluate the records against their intended states; extrinsic metrics test recovery by five readers on a seeded 32-persona GPT-5.1 panel balanced across lengths (520 rated turns per condition). The linked states, records, evidence packets, and answers support tests of construction and evidence delivery. In a contextual-stress control, tag-scoped writer inputs remove detected out-of-scope anchor copying. Frozen questions authored on staged dossiers transfer less reliably to one-pass dossiers of the same states. In matched evidence controls, whole sections repair 29 of 32 GPT-5.1 and 25 of 30 Sonnet 4.6 failures labeled as packaging, compared with 2 of 8 and 1 of 8 answer-formation failures. Attribution uses separate key-based and key-free support measures, with model disagreements retained. Length remains confounded with persona and question composition; the observed between-bucket widening depends on excerpt construction. The benchmark supports offline evaluation against known facts. Memory acquisition from interaction and validation against real users remain outside its scope. 

---
# Incorporating argumentation frameworks for LLM decision-making: Task-specific design and evaluation 

**Authors**: Bochen Han, Anshu Xiong, Mingyang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.25361)  

**Abstract**: Argumentation frameworks (AFs) provide explicit, formal mechanisms for representing and resolving conflicts and thus have been incorporated into large language model (LLM) decision-making. The benefits vary and depend on how arguments and their relations are constructed, evaluated, and incorporated into the decision process. {We abstract a workflow consisting of four stages based on existing argumentation-augmented methods} and investigate their design choices over tasks of debate-based decision-making and retrieval-based fact verification. Our evaluation with recent LLMs shows that existing methods do not consistently improve performance across the considered tasks. Motivated by this observation, we develop two task-specific methods. The first is called Qnt-Arg, which organizes arguments around the original speakers in debate materials and aggregates model-estimated argument strengths and candidate-answer priors under gradual semantics. The second is Qlt-Arg that uses credibility estimates to establish preferences between conflicting evidence items and applies grounded semantics to select acceptable evidence for subsequent LLM inference. We examine the roles of generative and decision models within the two methods. In the debate setting, Qnt-Arg consistently outperforms other argumentation-augmented methods across matched configurations; Qlt-Arg improves retrieval-based fact-verification performance compared with the baselines. These findings demonstrate the value of combining formal argumentation and LLM while highlighting the importance of aligning with tasks in argumentation construction, model selection, and the role of formal semantic reasoning. 

---
# Enhancing Understanding of Language Models with Varied Explanations 

**Authors**: Jingyi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.25314)  

**Abstract**: Language models are increasingly used to support human judgement in complex reasoning tasks, yet their predictions often remain difficult to interpret and validate. Explainability techniques provide insight into model decision-making, helping expose errors, failure modes, and potential bias so users can decide whether to rely on model outputs. Explanations are simplified approximations of a language model's internal workings. The goal is to obtain explanations that reflect the underlying computation as closely as possible (faithfulness), are understandable to users (plausibility), and help them achieve their intended goals (utility). Evaluation protocols are essential for assessing these dimensions. This thesis studies highlight explanations, natural language explanations, and chain-of-thought reasoning. These forms offer complementary perspectives on model behaviour and exhibit distinct strengths and limitations across tasks. Producing explanations that satisfy multiple quality dimensions remains challenging, as does evaluating them reliably across forms and settings. This thesis aims to improve people's understanding of models' decision-making processes in complex reasoning tasks. First, it proposes two evaluation frameworks for assessing highlight explanations across multiple properties and explanation types, together with a meta-evaluation framework for examining the reliability of existing faithfulness metrics for chain-of-thought explanations. Second, it proposes two techniques that generate natural language explanations designed to be faithful to the model's prediction process while remaining concise, coherent, and helpful for understanding model decisions. Finally, a study of human-AI interaction explores the roles of evidence and explanations in helping users validate model verdicts in a fact-checking setting. 

---
# LLMs Are Already Confidence-Aware Zero-Shot Classifiers 

**Authors**: Owen Cook, Xingyi Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.25289)  

**Abstract**: Zero-shot large language models (LLMs) are increasingly used as text classifiers, but their labels are only useful when we know to what extent we can trust them. We compare black-box methods (verbalised confidence and sampling) with white-box methods that read token probabilities (P(True), and multiple logit-scoring methods using constrained generation). We evaluate eleven open-weight LLMs (1B-32B parameters) on five classification datasets with 3 to 151 labels, measuring classification performance (accuracy, F1-macro), discrimination (AUROC), and calibration (ECE and Brier score). The choice of confidence method rarely impacts accuracy significantly, but logit-based scoring methods lead to much better discrimination (AUROC 0.74-0.77) than black-box methods (AUROC 0.57-0.66). Replacing labels with single-token identifiers preserves discrimination but costs 3-6 accuracy points on average. We also introduce decision-only scoring, which only scores positions where labels diverge in the prefix tree (trie). It reduces decode steps by 88-96% on the three largest label sets and performs comparably to full-label scoring. The best open-weight model-method configurations (Gemma 4 31B and Qwen 3.5 9B) are competitive with the commercial Jev 1.13.0 classifier. 

---
# When Objectives and Evaluators Disagree: Evaluating Transfer for Post-hoc Demographic Framing Mitigation 

**Authors**: Tunazzina Islam  

**Link**: [PDF](https://arxiv.org/pdf/2611.25258)  

**Abstract**: Post-hoc rewriting is used to reduce demographic disparities in LLM-generated text when the source model is inaccessible. A rewrite can look successful merely because it improved its own target or satisfied its selector. We study this as a transfer problem on 660 gender pairs and 1,980 age contrasts of demographic-conditioned climate messages. A rewriter proposes edits, a selection model chooses among constraint-satisfying candidates, and an independent evaluator uninvolved in either rates pairwise framing disparity. A fixed policy that always targets the message with the lower Persuasion Bias Index (PBI) transfers inconsistently across axes and outcome aggregations. An adaptive policy that proposes edits on both sides but accepts at most one significantly improves on that rule, applied to the same candidates, in seven of eight comparisons across two evaluator families, mostly by editing more pairs, not by guided direction choice, at higher preservation risk. A policy editing both messages gives the largest point reductions on fewer pairs. These two policies reduce disparity across a preservation-threshold sweep; absolute ratings vary across evaluator families and prompts. Shrinking the PBI gap increases dimension-mean disparity under both evaluator families. Mitigation should be reported as transfer across objectives, evaluators and their prompts, and editing policies. 

---
# Blind Spots Travel, Knowledge Stays Home: What RL Inherits from an LLM Judge 

**Authors**: Liang Yue, Eric A. Vance  

**Link**: [PDF](https://arxiv.org/pdf/2611.25191)  

**Abstract**: LLM judges serve as rewards where no answer key exists, in the hope that their knowledge passes to the trained model. In 814 RL runs that teach 7-8B models to answer closed-book questions or abstain, we ask what passes. What a grader knows about single answers stays home. A wrong answer that a judge rewards is memorised for that one question. A model trained with the answer key, or with a far stronger judge, chooses which new questions to answer no better than the untrained model does by sampling 32 answers and abstaining where they disagree; the grader's knowledge adds about one corrected answer per hundred questions. What a grader believes about a kind of question travels. A judge that rejects eleven of the twelve correct training answers about capitals of former states teaches the model to refuse 98% of such capitals on new questions, a loss that accuracy and abstention rates do not reveal. Changing only those twelve verdicts by hand moves the refusal from 5% to 90%, even for a grader that accepts as many answers as the answer key; at that leniency, two of four judges already cost 38-42% of these capitals. The capitals are not forgotten: with "I don't know" blocked, the model answers 94% of them correctly. The grader therefore needs no knowledge of its own: rewarding the model for giving the answer that most of its own untrained samples agree on, with no labels and no judge, earns the answer key's reward and keeps the category. 

---
# Forget Freely, Grep Later: Lossless, Index-Free Memory for Long-Horizon LLM Agents 

**Authors**: Ziyan Yang, Taiwei Shi, Jieyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.25123)  

**Abstract**: Long-horizon LLM agents often exceed their context windows, forcing them to discard information that may be needed later. Existing context management strategies rely on summarization, masking, or structured memory, requiring decisions about what to preserve or how to retrieve it before future needs are known. However, these methods can introduce lossy compression, potentially discarding details needed for future tasks. To address this limitation, we introduce GrepMem, a simple, lossless, and index-free memory mechanism built on two components: an append-only plain-text log of the agent's complete history and a regular-expression search tool for on-demand recall. Unlike structured memory systems, GrepMem requires no embeddings, retrieval index, memory extraction, or additional models. We train Qwen3.5-9B with reinforcement learning under five context management strategies and evaluate on multi-question research sessions, long-stream recall, and agent-trajectory question answering. GrepMem achieves 95-97% accuracy on ContextBench, versus at most 21% for lossy baselines, and outperforms SUPO's summarization by 8.8 percentage points on BCP-Multi. GrepMem scales effectively to increasingly long input streams while remaining on the accuracy-token-cost Pareto frontier across all five evaluation tasks. Our results show that a plain-text log and grep provide a simple but competitive approach to long-horizon agent memory, without requiring complex structured memory infrastructure. 

---
# FinVeriQA: Financial Research Agents for Question Answering: Routing, Retrieval, MCP Tool Use, and Provenance-Aware Evidence Verification 

**Authors**: Meenakshi Rajpurohit, Anna Phan, Ravikumar Komandur Narayanan, Gunanidhi Ramakrishnan, Nathan Howland  

**Link**: [PDF](https://arxiv.org/pdf/2611.25071)  

**Abstract**: Large language models (LLMs) write fluent financial analyses but miscompute ratios, recall stale prices, and route questions to the wrong tool with high confidence. We ask which component should do each job in a financial question-answering assistant, and study it with FinVeriQA, an open-source multi-agent research prototype over SEC filings and live market data: a compact typed-decision model (Laya) routes each question and screens out advice requests, Python computes every number, retrieval and tools fetch evidence, Mistral-7B writes a cited answer, a code validator checks that cited IDs exist and that numbers match computed notes, and a Model Context Protocol (MCP) server exposes the pipeline. In live experiments, Laya routed better than Mistral-7B (0.85 vs. 0.52; overall 0.86 vs. 0.68, McNemar p = 0.0003); a 12B Mistral decided no better, and DeepSeek-R1-8B decided better (0.79 vs. 0.68, p = 0.0192) but about 9.9 times more slowly. Mistral-7B computed 29% of 100 financial calculations correctly against 100% for Python, a live quote cut price error from 210% to 0.00%, and hybrid retrieval reached 76% Hit@5 on 50 filing questions. Yet in a live case study, both agent answers passed validation while describing AMD sample data under NVIDIA's and Alphabet's names. A controlled upper-bound study on the FinRank benchmark explains why: when one figure in the shared evidence is corrupted, self-consistency stays confident (agreement 0.55 to 0.59 while accuracy falls from 0.50 to 0.00), and a shared-evidence check (the kind our validator performs) catches 35% of the errors, whereas a check against the original passage catches 77% ($p = 5.96 \times 10^{-8}$); for DeepSeek-R1-8B, 68% vs. 90%. Verification must be independent of the evidence the writer saw. 

---
# What the model knows but does not say: monitoring medical LLMs through verbalized thoughts 

**Authors**: Frederik Hauke, Jan Clusmann, Hanna Kreutzer, Christiane Kuhl, Tianyu Han, Felix Barajas Ordonez, Sven Nebelung, Jakob Nikolas Kather, Daniel Truhn  

**Link**: [PDF](https://arxiv.org/pdf/2611.24994)  

**Abstract**: Efforts to keep clinical language models safe have concentrated on what enters and leaves the model, but a manipulated model can return an answer that looks entirely ordinary. Here we show that the model's own internal state, rendered in plain language, exposes manipulations the output conceals. Using a natural-language autoencoder, we translate the activations of a multimodal LLM (Gemma-3-27B) into text and let an input-blind judge read this verbalized internal state (the thoughts channel) and the model's answer separately. Across two clinical attack objectives (omitting a pathology, covertly recommending a drug), two input modalities, and two delivery mechanisms, the thoughts channel caught the injection significantly more often than the answer in all eight injected conditions (p<0.05), while both channels held near-zero false-positive rates on clean inputs; the omission result carried over to Llama-3.3-70B. The thoughts channel also registered unsafe clinical requests as harmful while the model was still reading them, before it decided whether to refuse. None of this requires retraining: an off-the-shelf verbalizer and a general-purpose judge are enough. 

---
# Quantifying Cultural-Psychological Constructs in Four Centuries of Ottoman Court Registers 

**Authors**: Sila Mutaf, Enes Yılandiloğlu, Mohammad Atari  

**Link**: [PDF](https://arxiv.org/pdf/2611.24981)  

**Abstract**: Populations vary along cultural dimensions such as individualism, collectivism, and honor. But how can these dimensions be studied in historical societies for which conventional psychological data are unavailable? Here, we introduce a computational text-analytic framework for quantifying psychological constructs in 54,670 Ottoman Turkish court records from 21 Istanbul courts spanning 1513-1911 CE. We develop sentence-embedding models adapted to Ottoman legal language and extend a theory-driven, text-analytic pipeline to capture cultural variation along the dimensions of individualism, collectivism, and honor in this historical corpus. We evaluate our models through multiple methods: document retrieval, semantic textual similarity, questionnaire-item classification, and validation against expert annotations. Our adapted sentence transformer (OTA-BERTurk) achieved a Recall@1 of .843 on the test set for document retrieval, a Mean Reciprocal Rank (MRR) of .806 for semantic textual similarity (meaning the best answer usually appeared as a top result), and classified questionnaire items with 89.58\% accuracy. Document-level performance varied across constructs and methods. Qualitative analysis further showed that high scores reflected historically specific legal and institutional patterns. We publicly release the models, construct representations, scoring pipeline, as well as an interactive web interface to support transparent, reproducible historical-psychological analyses of Ottoman Turkish court records spanning approximately four centuries. These computational tools enable the study of psychological variation across centuries of cultural and institutional change in a diverse empire. 

---
# CANDID: Candidate-Driven Intervention for Disciplined Execution in Local Computer-Use Agents 

**Authors**: Woongkyu Lee, Jungwook Choi  

**Link**: [PDF](https://arxiv.org/pdf/2611.24977)  

**Abstract**: We present CANDID (Candidate-Driven Intervention for Disciplined Execution), an execution-control framework designed to enhance the reliability of local computer-use agents (CUAs) in resource-constrained settings. While local CUAs feature robust baseline proficiency in single-step action prediction, they frequently fail at critical junctions due to rigid action loops or premature DONE decisions. Rather than imposing imprecise cognitive loads through textual reflection, CANDID repurposes parallel test-time scaling into a disciplined, execution-level intervention at sparse, failure-prone points. We introduce a loop-recovery mechanism to filter candidates away from repeated behaviors and a conservative, unanimous DONE gating policy that requires absolute consensus to terminate. Experiments on OSWorld demonstrate that CANDID's structural control significantly improves success rates, outperforming feedback-based alternatives with minimal execution overhead. 

---
# No-Harm-Jury A Jury-Based Multi-Dimensional Harm Evaluation Framework for Medical AI Responses 

**Authors**: Antonio Puertas Gallardo, Mario Ceresa  

**Link**: [PDF](https://arxiv.org/pdf/2611.24972)  

**Abstract**: We present No-Harm-Jury, a jury-based automated evaluation framework that scores AI-generated medical responses across seven conceptually distinct harm dimensions using a panel of five diverse language models as independent jurors. Responses generated by Ministral-14B on 3,000 questions drawn from PubMedQA, MedQA, and MedMCQA are scored by five jurors across dimensions of informational, social, psychological, autonomy, economic, privacy, and epistemic harm; the 151 empty responses (98 MedQA, 53 MedM- CQA) are excluded, leaving 2,849 valid samples. The median aggregation robustly suppresses individual juror biases while maintaining coverage through complementary specialisation. No existing system combines all five prop- erties of No-Harm-Jury: automated scalable evaluation, five diverse LLM jurors, robust me- dian aggregation (with a reliability-weighted extension), seven conceptually distinct harm dimensions, and application to medical QA benchmarks. 

---
# The ROI of Small Examples: Optimizing In-Context Learning Under Token Budgets 

**Authors**: Anxo Perez, Jorge Gabín, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2611.24971)  

**Abstract**: In-context learning (ICL) improves large language model performance at inference time without updating model parameters. However, these gains are not free: in practical deployments, longer prompts directly increase inference cost and latency. Under a fixed token budget, prompt construction therefore becomes a resource-allocation problem. While prior work has often focused on how many examples to include in ICL, we study the complementary question of whether the available budget is better spent on a few long demonstrations or on a larger number of shorter ones. We study this aspect in question answering (QA), by organizing candidate demonstrations into length-based bins and constructing prompts from examples within the same length range under fixed budgets. Experiments on two QA benchmarks (\textit{Beat the AI} and \textit{Natural Questions}), using six instruction-tuned LLMs, show that shorter demonstrations are generally the most effective use of prompt space. This effect is especially strong on \textit{Beat the AI}, and remains visible on NQ, although gains are smaller and saturate earlier in the long-context setting. An additional natural language inference experiment on ANLI R3 finds that one of the two shorter bins performs best for five of six models, although longer demonstrations remain competitive. Overall, our results highlight demonstration granularity as an important factor in prompt design. 

---
# HireWatch: Evaluating LLM Compliance with U.S. Employment Law Under Contextual Pressure 

**Authors**: Huy Nghiem, Maria Kostylew, Noam Kolt  

**Link**: [PDF](https://arxiv.org/pdf/2611.24970)  

**Abstract**: The deployment of large language models (LLMs) in hiring pipelines is growing, yet existing evaluations generally overlook their compliance with legal rules. To fill this gap, we introduce HireWatch, an evaluation that assesses whether LLMs comply with U.S. laws that govern the questions that job candidates may be asked. We construct a 200-item bank of lawful and unlawful interview questions, whose legal classification is based on U.S. federal statutes, regulations, and agency guidance. We instruct LLMs to select questions from this bank in preparing for job interviews, across a set of various occupations and contexts, and report the fraction of unlawful questions that LLMs select. Evaluation of 18 closed and open-weight LLMs across ~100,000 trials under different types of contextual pressure reveals that LLMs' legal compliance is brittle. While LLMs generally avoid selecting unlawful questions in low-pressure contexts, legal violations increase sharply under higher pressure. Explicit prompts to comply with law can improve compliance rates, but do not altogether eliminate legal violations. We publicly release our code and data at this https URL. 

---
# Global Importance Estimation for KV Cache Eviction 

**Authors**: Seojin Kim, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.24969)  

**Abstract**: Memory consumption of the key-value (KV) cache remains a central bottleneck for long-context LLM inference. We introduce PRKV, a training-free KV eviction method that estimates multi-hop global token importance by applying Personalized PageRank over attention-induced transition graphs. This enables importance propagation beyond recency or instantaneous attention, without modifying the model. We benchmark PRKV on long-context understanding, sparse retrieval, and chain-of-thought reasoning tasks using multiple LLM families. Under moderate to aggressive compression ratios, including up to 90%, PRKV achieves competitive task performance against established eviction baselines while reducing peak memory usage with low end-to-end overhead. These results demonstrate that multi-hop global influence estimation yields a favorable accuracy--memory trade-off for scalable long-context inference. 

---
# Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality 

**Authors**: Zeyuan Allen-Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.24963)  

**Abstract**: Do standard benchmarks see the architectural differences that matter? We study it with 64 from-scratch pretraining runs of Transformers and three linear-model families at 1.3-8B parameters on 1-2T Nemotron-CC tokens, at a cost of roughly 100k USD per 8B run.
In academic-scale pretraining (1.3B parameters, 100B tokens), architectural differences can all but disappear when the models are carefully tuned. Instead, at 1-2T tokens they become measurable: Canon layers, which add local token mixing, consistently improve Transformers, saving about 30% of pretraining tokens for the same MMLU accuracy at 8B. We also introduce GLA5 and GDN2, which improve gated linear attention and Gated DeltaNet with low-rank output gates. Surprisingly, with Canon, the simplest linear model, gated linear attention, matches Gated DeltaNet and outperforms Mamba2 even at 8B/1T. Therefore, the extra machinery of Gated DeltaNet and Mamba2 may do little more than mix neighboring tokens.
Even at 8B/1T, however, high scores on standard benchmarks can hide weaknesses in basic in-context skills. In particular, linear models that match or beat Transformers on these benchmarks cannot, in fact, reliably retrieve information from a short 500-token context. Moreover, no model, Transformer or linear, reliably beats random guessing on arguably the simplest two-hop question, stated in a few short sentences, even at this scale.
These findings $\textbf{challenge traditional architecture research}$ based primarily on standard benchmark scores. In contrast, small synthetic playgrounds, which predicted the architectural differences above at <100M parameters, can reveal what these benchmarks miss. For practitioners, one lesson is to include in-context reasoning examples in pretraining data from an early stage, rather than reserve them for later mid-training, let alone post-training. 

---
# TETHER: Training-Free, Elastic, ToC-Anchored Hierarchy Extraction and Recovery in Design-Driven Long Documents 

**Authors**: Yu Zou, Yan Chen, Jialei Zeng, Xiaorui Zhou, Aixi Zhong, Qi Zhou, Haoran You, Ke Jiang, Changyu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.24960)  

**Abstract**: Converting complex documents into structured knowledge constitutes the critical infrastructure urgently required for deploying Large Language Models (LLMs) in downstream research. However, blind to the Table of Contents (ToC) as a macro-structure, existing bottom-up paradigms rely on localized visual cues, inducing heuristic fragility in high-entropy layouts where visual prominence decouples from logical depth; meanwhile, direct LLM extraction lacks spatial grounding, triggering hallucinations in long contexts. To address this, we introduce TETHER, a training-free framework that elevates the ToC from discarded noise to a vital structural prior, enabling robust adaptation across design-driven long documents. TETHER operates via a two-stage neuro-symbolic mechanism: first, it constructs a document skeleton from the raw ToC using a Large Multimodal Model (LMM) empowered by spatio-semantic constraints; second, it employs an anchor-based alignment strategy to tether content blocks to this skeleton, effectively resolving boundary ambiguities. Evaluated across a multilingual benchmark of ESG reports, product manuals, and annual reports, TETHER delivers highly precise, state-of-the-art (SOTA) zero-shot hierarchy recovery, outperforming deep hierarchical recovery models, dedicated parsers, and general-purpose LMMs. Moreover, stress tests highlight TETHER's resilience to sparse omissions and textual noise, demonstrating that its precise semantic alignment is fundamentally anchored in topological and depth fidelity. Crucially, ablations confirm the framework's architectural irreducibility, while achieving 90% token reduction and drastic speedups over end-to-end baselines, establishing a compelling equilibrium between structural precision and deployment scalability. 

---
# ICARE: cross-domain text classification with incremental class-aware representation and distillation learning 

**Authors**: Son Duong, Truong X. Tran, Hai-Anh Tran  

**Link**: [PDF](https://arxiv.org/pdf/2611.24959)  

**Abstract**: The rise of powerful, fine-tuned large language models (LLMs) necessitates robust natural language inference (NLI) systems. These NLI systems ideally train across various tasks, encompassing diverse concepts like sentiment analysis, news classification, and coreference resolution. However, this continual, multi-tasking approach can lead to catastrophic forgetting, where previously learned tasks are lost, especially in domains lacking specific labels. Rehearsing past examples with labels can further exacerbate this issue by causing overfitting, hindering the system's ability to classify new categories. To tackle these challenges, this research proposes ICARE, a novel framework with two key components: an instance selection module and a knowledge distillation method. The former is based on representation learning, which meticulously selects training examples to present to the NLI system. The latter, knowledge distillation with feature and logit-level transfer, facilitates knowledge transfer between tasks. This leverages pruning techniques and optimized objective functions to transfer both essential feature-level and decision-making (logit-level) knowledge, mitigating catastrophic forgetting while adapting to new patterns. Extensive evaluations on benchmark text classification datasets have demonstrated ICARE's effectiveness. The framework achieves a significant improvement in target accuracy and reduction in the ``forgetting metric''. These results solidify ICARE's ability to preserve and leverage learned knowledge across diverse tasks, ultimately enhancing the adaptability and performance of NLI systems. The source code is available at this https URL 

---
# LLM Complexity: Towards a Computational Complexity View of Large Language Model Inference 

**Authors**: Yu-Hung Wu, Shao-Xiang Yuan, Chan-Hung Yu, Zih-Ching Chen, Kuan-Hao Huang, Shao-Hua Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.24943)  

**Abstract**: Computational complexity theory offers a principled framework for characterizing the efficiency of computational problems, and Large Language Models (LLMs) can similarly be viewed as computational systems executing a structured process of input, inference-time compute, and output. In this context, the number of generated tokens emerges as a natural and directly observable proxy for inference-time compute. However, we show that the tempting analogy between LLM token length and classical time complexity does not hold. We disprove the hypothesis that inference-time token usage aligns monotonically with classical time complexity, demonstrating a fundamental divergence grounded in how LLMs actually allocate computation during inference. To this end, we introduce LLM Complexity, a new metric designed to quantify problem complexity from the perspective of LLMs by characterizing the area bounded above by the accuracy-token curve. We find that diverse LLMs induce highly consistent task rankings under LLM Complexity, indicating that the metric captures stable notions of difficulty across models. We leverage this property to construct an easy-to-hard curriculum for supervised fine-tuning, which improves accuracy while reducing token consumption. Finally, we demonstrate that LLM Complexity generalizes to real-world tasks, producing more stable cross-model correlations than human-annotated difficulty labels. 

---
# AgentRoboCritic: Understanding the Capability of Frontier Agents on Direct Robot Control 

**Authors**: Yiyang Du, Yuxuan Kuang, Qitao Zhao, Yixuan Wang, Zhanqiu Guo, Yi Xu, Xin Li, Wenbin He, Liu Ren, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2611.26613)  

**Abstract**: Frontier multimodal language agents such as GPT-6 Astra have demonstrated strong capabilities in controlling robots directly via low-level numerical actions without high-level primitives. However, existing benchmarks report only episode-level success rates, which show neither how these agents succeed nor where they fail. In this paper, we present a systematic action-level analysis of frontier agents on direct robot control across 12 manipulation benchmarks. We introduce AgentRoboCritic, an intent-grounded critique framework in which the policy agent states a local intent alongside each action, and a critic evaluates the action's outcome and failure mode against this intent. Our action-level analysis demonstrates the promising capabilities of frontier agents on direct robot control, including closed-loop recovery from local execution failures, reliable semantic grounding across diverse scenes, and strong zero-shot manipulation without task-specific adaptation. It also identifies their current limitations, including task-relevant physical execution bottlenecks such as grasping and alignment, and uneven recovery across failure modes. Finally, we validate our critic on a 400-action human-annotated benchmark and release a dataset of 12,844 annotated actions alongside a fine-tuned open-source critic, Muse-Glimmer-30B, whose failure-mode accuracy rises from 32.58% to 74.24%, enabling low-cost action-level analysis. 

---
# Hybrid Retrieval for Government Scheme Search in English and Marathi 

**Authors**: Aishwarya Gunjal, Kanchan Khemnar, Pawan Poojary, Pranav Kulkarni, Raviraj Joshi, Manish Khodaskar  

**Link**: [PDF](https://arxiv.org/pdf/2611.26514)  

**Abstract**: Retrieving government schemes requires matching citizens' information needs to formal administrative descriptions, often across languages and terminology. We present a bilingual search system for Maharashtra government schemes and public services that combines BM25 retrieval with sentence-embedding similarity through equal-weight score fusion. The system indexes aligned English-Marathi scheme descriptions and supports multilingual, Marathi-focused, and translation-mediated semantic configurations. Evaluation on manually curated bilingual queries compares BM25, dense retrieval, and hybrid retrieval using IndicSBERT and MahaSBERT. BM25 provides a strong baseline with a mean reciprocal rank (MRR) of 0.864. Hybrid retrieval increases MRR to 0.872 with IndicSBERT and 0.879 with MahaSBERT, whereas the corresponding dense-only systems achieve 0.663 and 0.655. MahaSBERT-based fusion also increases precision from 0.781 to 0.808 while matching BM25 recall at 0.946. These results support the complementary use of lexical and semantic signals in this collection, while showing that improvements over a strong lexical baseline are modest and metric-dependent. The study provides an application-focused analysis of hybrid retrieval for bilingual public-service information access. 

---
# GLiDE: A Fast, Calibrated Decision Model Distilled from Reasoning Teachers 

**Authors**: Sahibzada Allahyar, Julia White, Dhruv Atreja, George Hurn-Maloney, Urchade Zaratiana  

**Link**: [PDF](https://arxiv.org/pdf/2611.26509)  

**Abstract**: Decision models enable fast, confidence-aware decisions by scoring candidate answers in a single forward pass, without generating text or intermediate reasoning. Yet reasoning can substantially improve decision quality, raising the question of whether its benefits can be retained without the cost of generating it. We investigate this question with \textbf{GLiDE}, a family of decision models spanning 0.4B to 27B parameters.
Building on Qwen hybrid language models, GLiDE enables bidirectional interactions between the context, question, and candidate answers by removing causal masking from full-attention layers. To transfer reasoning capabilities into single-pass inference, we combine soft-target distillation from multiple teachers, including a reasoning model, with progressive post-training on diverse decision tasks. We further explore parameter-free recurrent depth to increase inference-time computation without generating tokens.
Across five model sizes, GLiDE lies on the performance--parameter Pareto frontier of the Hugging Face \DI{}, with each model outperforming evaluated alternatives of equal or smaller size. \dmtwo{} (27B) achieves a hidden-set score of 71.52, with an expected calibration error of 2.0\% and a median latency of 92.9\,ms on a single GPU. Further checkpoint averaging and parameter-free recurrent depth yield GLiDE~27B, which reaches 73.37 with an expected calibration error of 2.1\%. 

---
# Democratizing Cross-Project Knowledge for Transforming Water Governance and Climate Resilience: an Agentic AI Ecosystem 

**Authors**: Oriol Alàs, Ian Palacín, Iván Cester, Edgar Rubión  

**Link**: [PDF](https://arxiv.org/pdf/2611.26398)  

**Abstract**: EU-funded projects produce extensive evidence in the form of deliverables, reports, and case studies. However, this knowledge is typically scattered across projects and document structures, making it difficult to locate relevant, contextual information when making decisions. This paper introduces an Agentic AI Ecosystem that consolidates heterogeneous project documentation into a domain-aware knowledge base and provides adaptive access through an agentic Retrieval-Augmented Generation pipeline. Documents from three EU-funded projects (IMPETUS, NATALIE, and InnWater) are reconstructed into hierarchical section trees and indexed with structured metadata. Retrieval is supported by two domain-specific embedding models fine-tuned on domain data (water and climate). On top of this knowledge base, a ReAct-style controller iteratively orchestrates retrieval actions by allocating retrieval budget across domain collections and accumulating evidence until a stopping criterion is met. In a retrieval benchmark over the corpus, the proposed approach improves Mean Reciprocal Rank by 26.7% (relative) compared to a single-domain embedding baseline, indicating more consistent retrieval of relevant evidence to support cross-project knowledge synthesis and climate-resilient decision-making. 

---
# HiSA: Sparsifying the Hidden Dimension of RoPE Attention in LLMs 

**Authors**: Ambroise Borbely, Mathieu Salzmann, Olivier Verscheure  

**Link**: [PDF](https://arxiv.org/pdf/2611.26363)  

**Abstract**: Attention in large language models (LLMs) scores every query against every key with a full dot product. We investigate whether this dot product must be computed completely for attention to be effective. We propose Hidden Sparse Attention (HiSA), which keeps the query and key projections intact but only computes a partial dot product, following a static, parameter-free rule set by the token positions: attention becomes sparse in the hidden (feature) dimension, an axis much less explored than token sparsity, rather than over tokens as in almost all sparse attention methods. Based on the structure of rotary position embeddings (RoPE), we decompose keys and queries into parts which can be omitted when computing attention. By training LLMs from scratch with HiSA, we find that this partial dot product does not hurt but improves performance compared with the complete product while computing only about a third of the query-key products. During inference HiSA improves decoding speed and shrinks memory footprint. We also extend HiSA to FlashAttention-3 kernels for training and inference. Finally, we motivate our approach by showing that dense attention learns to suppress the attention score parts that HiSA drops. Code is available at this https URL. 

---
# BADAS-3.0: Real-Time Driving Perception and Planning from One Camera 

**Authors**: Roni Goldshmidt, Hamish Scott, Lorenzo Niccolini  

**Link**: [PDF](https://arxiv.org/pdf/2611.26294)  

**Abstract**: We ask how far a driving model can get with one camera and real-time latency. BADAS-3.0 is a 1B-parameter video encoder with four small heads. From two seconds of one forward camera, it predicts the current speed, the metric path just driven, a collision-risk score and six candidate futures over 6.4 seconds, in under 50 ms. Given the car's current speed, BADAS-3.0 has 44% lower minADE$_6$ over five trajectory benchmarks than the 34B Alpamayo 2 Super, which reads six cameras and runs over 100x slower. The gain comes from camera rigs that neither model was trained on. On the rig they share, the two are on par. Existing benchmarks are mostly nominal driving, so we add two open-loop safety benchmarks, one of real dashcam collisions and near-misses and one of Euro NCAP scenarios. On both, BADAS-3.0 has a higher pass rate than every published planner we tested. 

---
# TwinLab-Bench: Evaluating AI Scientists in Agentic Twins of Real Labs 

**Authors**: Yang Wan, Linchao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26248)  

**Abstract**: AI agents are increasingly expected to act as AI scientists, which requires three increasingly open abilities: carrying out a given computational workflow correctly, choosing the next measurements from a given design space, and designing their own experiments. Existing benchmarks mainly evaluate the first two abilities, in a terminal or in a closed loop that selects from predefined candidates. Evaluating the third ability requires reliably executing an experiment that an agent designs, with readings grounded in a real laboratory. We introduce TwinLab-Bench, which evaluates AI scientists on self-designed experiments in 52 tasks from laboratories in the physical sciences, the life sciences, and engineering. The benchmark supports more than 100 experimental action types, and the agent's report is scored by comparison with a hidden answer taken from real laboratory records. Each task has its own TwinLab agent, an agent that serves as the environment and executes the experiments that the tested agent designs, reading the raw materials of the laboratory and calling calculators that are built from its records. Compared with a language model that sees only the task description, the TwinLab agent almost eliminates readings that deviate from the real records and inconsistent answers to a repeated request, and closed-book probes control for answers recalled from memory. We find that these tasks remain hard for frontier models: the strongest, Claude Fable 5.2, averages a score of only 45.0, and we conduct an empirical study of where it fails. 

---
# Learning to Think with Images Faithfully via Adaptive Data Selection 

**Authors**: Yu Zhao, Jiaxin Zhang, Rohit Saxena, Fazl Barez, Pasquale Minervini, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.26199)  

**Abstract**: Thinking with images (TwI) empowers vision-language models (VLMs) to overcome perceptual bottlenecks by actively cropping and zooming into fine-grained visual details, promising both higher accuracy and interpretable, region-grounded reasoning. However, our analysis shows that the improved accuracy of TwI models does not guarantee faithful tool use; they frequently invoke tools without actually relying on the zoomed region, undermining the reliability of tool-assisted reasoning. We trace this unfaithful behaviour to the training data: when questions are answerable without zooming, the model can answer correctly while ignoring the returned evidence, and this problem is exacerbated during training as the model becomes better at answering without tools. To keep zooming useful throughout training, we adapt the training data to the model's evolving capability and train on questions where correct zoomed evidence can still help the model answer correctly. This selection is supported by a data synthesis pipeline that localises small objects and constructs verifiable questions about their fine details, and complemented by an adaptiveness reward that favours tool-free answers when zooming is unnecessary. The experimental results show that our method substantially improves reasoning faithfulness and achieves state-of-the-art performance on visual search benchmarks (+11.6% on V*, +7.9% on HR-Bench 4K, and +9.4% on HR-Bench 8K). On multimodal hallucination benchmarks, our model also outperforms the compared TwI baselines. These results suggest that improving TwI faithfulness does not sacrifice accuracy and that both can be learnt from the right training data without directly supervising the reasoning process. 

---
# Can World Models Predict Rice Growth? Toward Agricultural World Models 

**Authors**: Wenkai Ji, Zeteng Li, Yinkun Guo, Jie Zhou, Liang He, Zhenong Jin  

**Link**: [PDF](https://arxiv.org/pdf/2611.26145)  

**Abstract**: World models can imagine how a physical scene evolves over the next few seconds, but growers need to know how a crop will develop over weeks to months: will this rice plant have headed a month from now, and how will it look at harvest? We introduce agricultural world modeling, the task of predicting how the same plant will look at a requested horizon of days to months, from one photo of it, its cultivar, and its camera view. In preliminary analysis, even state-of-the-art image generators get this wrong: they often grow the plant too little or give it the color and form of the wrong growth stage. We argue that they do not work out the plant's developmental state, which the photo reveals only indirectly. We therefore propose AgroWM, which reasons about development before it draws growth. Its reasoning module, PhenoClock, works out the current growth stage, predicts the future one, and writes a Growth Script with numeric targets for height, plant area, and color; its generation module, the Rooted Renderer, then edits the photo to follow this script, adding new organs and colors to the photographed plant. Because pixel scores can prefer an unchanged copy of the photo, we instead check each forecast's traits and developmental timing against the plant's own later photos. On rice, 54.6% of the forecasts of AgroWM are matched to the target stage (Stage R@1), against 34.8% for GPT Image 2.5, 24.0% for Qwen Image 2.1, and 47.4% for the strongest domain-specific baseline, the same backbone fine-tuned without Growth Scripts; 9 to 20 weeks ahead, it still reaches 46.4%, against 35.7% for that baseline. On maize, a crop with a different body plan and stage scale, the same design again scores best on all four measures. 

---
# RL Can Only Reinforce What It Sees Succeed 

**Authors**: Ruben Sanchez-Corcuera, Unai Zulaika, Aritz Bilbao-Jayo, Miguel Fernandez-de-Retana  

**Link**: [PDF](https://arxiv.org/pdf/2611.26084)  

**Abstract**: When a familiar rule stops working, an agent has to weigh what pre-training taught it against what it now observes. We ask how outcome-reward reinforcement learning (RL) changes this balance, in a puzzle world where both sides are known: small language models trained from scratch on games played by a Bayesian player, so that the prior and the right answer are exact. Base models hedge as their corpus did, and RL turns that hedging into decisions, in whichever direction the reward favours. But RL can only reinforce what it sees succeed. When the right action is rare, each run either learns it everywhere or not at all, and which runs take off is predicted before training by one count: the expected number of groups of attempts in which the action wins. About 35 such groups give even odds. The count holds for new and repeated problems and for groups of 4 to 256 attempts, and across learning rates, optimizers, GRPO variants and model sizes from 1.3M to 85M parameters, which move the threshold but not what is counted. 

---
# Complete-SVD: A Functional Norm Approach to Low-Rank LLM Compression 

**Authors**: Alper Kalle, Theo Rudkiewicz, Mohamed-Oumar Ouerfelli, Mohamed Tamaazousti  

**Link**: [PDF](https://arxiv.org/pdf/2611.26059)  

**Abstract**: Post-training low-rank compression of neural networks must not only approximate the original weights but also preserve the model's input-output behavior. Recent output-aware methods measure the compression error beyond the compressed layer, at the model output, weighting logit changes by the observed next token or by the model's predicted probabilities. We propose Complete-SVD, a method for large language models that minimizes the $L_2$ distance between the logits of the original and compressed models, weighting all logit changes equally. Linearizing the network around the original weights, together with a Kronecker-factored approximation, reduces this objective to a weighted SVD with a closed-form solution. Its output-side factor is intractable to compute; we estimate it without bias from random probes, at a cost of $K$ backward passes per sequence, independent of vocabulary size and sequence length. At matched settings with uniform ranks and WikiText-2 calibration, Complete-SVD achieves the lowest C4 and PTB perplexity in every setting, and at ratio 0.7 under LEMS rank allocation. Replacing the $L_2$ distance by the Kullback-Leibler divergence (KL-SVD), with the same estimation framework, lowers calibration perplexity in most settings and raises downstream accuracy, but raises cross-corpus perplexity. Complete-SVD also improves IFEval instruction-following accuracy over the strongest prior output-aware methods throughout. 

---
# Policy Improvement-Guided Replay for Continual Reinforcement Learning 

**Authors**: Qin-Wen Luo, Haoxuan Gao, Sheng Ren, Xiang Chen, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25989)  

**Abstract**: Continual reinforcement learning (CRL) requires models to acquire new capabilities while retaining performance on previous tasks. A stability-oriented perspective common in conventional continual learning emphasizes knowledge preservation and interference mitigation. Complementing this perspective, we revisit CRL for vision-language models through policy improvement, a fundamental principle of RL, to preserve acquired performance gains. Using the performance difference lemma, we characterize historical performance changes through the current policy's expected advantages on historical task distributions, motivating replay of historical prompts with the current policy. To allocate limited replay memory, we propose Contrastive Likelihood Replay (CoLiR), which contrasts likelihood gains on correct responses from the final task policy with likelihood reductions on incorrect responses from the initial policy. Our analysis links correct-response likelihood gains to reference-policy advantages under an idealized KL-regularized optimal update and provides an endogenous-reward interpretation for general policy updates. CoLiR estimates scores from finite training rollouts and cached log-likelihoods, selects prompts at task boundaries, and replays them uniformly during subsequent training, with KL regularization restricted to replayed prompts. Memory construction only requires a small amount of additional cross-policy scoring but no additional response generation. Across three Qwen3-VL model sizes on MRCL, CoLiR achieves the highest mean final accuracy and mean task accuracy among the compared methods while maintaining competitive new-task performance. 

---
# When Sounds Turn Heads: Recovering Event Timing That Audio LLMs Fail to Say 

**Authors**: Jongbhin Woo, Jaewook Jeong, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2611.25960)  

**Abstract**: Audio LLMs can know when a queried sound occurs but fail to say it. We find a few attention heads that track the queried sound: they shift when the query changes, follow the sound when it moves, and removing or blocking them often hurts the model's timestamps. Yet the answer does not express what they track. Intervals read directly from these heads outperform the model's original answer and recover events that its timestamps miss. To bring this timing into the answer, we introduce head-guided decoding, which needs no training. It keeps the model's decision on whether the sound is present and rescores its candidate timestamps with a prior from these heads. Heads selected on development data transfer unchanged to new recordings and free-text queries, where head-guided decoding improves the timestamps of three audio LLMs, scores above their head readout on every benchmark, and, in most settings, outperforms random-head guidance. On recordings released after the models were trained, head-guided decoding raises mIoU from 0.07-0.36 to 0.48-0.56. These results suggest that poor temporal answers in audio LLMs partly reflect a failure to say rather than to hear, and that the model's own attention can close part of this gap. 

---
# One Schedule Does Not Fit All: Adaptive Inference Schedules for Diffusion Language Models 

**Authors**: Rabimba Karanjai, Linsey Pang, Hemanth Hegadehalli Madhavarao, Richard Williamson, Uma Kona, Prakhar Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2611.25895)  

**Abstract**: Diffusion language models typically use a fixed denoising schedule across workloads. We show that the cheapest schedule that preserves quality depends on the task, model, and sampler. We propose Task-Calibrated Denoising Allocation (TCDA), a deployment rule with an explicit error guarantee. Given candidate samplers, paired calibration data, a utility tolerance, and guardrails against degeneration, TCDA uses simultaneous one-sided bootstrap bounds to certify which candidates meet both the utility and guardrail criteria. It deploys the cheapest certified candidate or falls back to the full schedule. The probability of deploying an infeasible schedule is asymptotically at most alpha. Candidates include early prefix decoding, state-reusing continuation, reduced time grids, and confidence-threshold decoding. Across seven datasets and two model families, TCDA certifies reductions in denoiser evaluations by factors of 3.4 on XSum and 2.0 on WMT14 for ELF-B; 8.0 on WMT14 and 4.1 on GSM8K for LLaDA-8B; and 1.33 on OpenWebText for both ELF-M and ELF-L. A validation protocol fixed before generation confirmed every selection on fresh seeds. Across 500 calibration splits, false certification remained below 2% in every setting at a nominal level of 5%. Certified schedules vary across tasks and model scales. Trajectory measurements suggest that when decoded tokens commit is more informative for schedule selection than the remaining latent discretization error. We also find that the ELF reference sampler draws its time grid at random, so reproducible schedule comparisons for unconditional generation require fixing the grid. 

---
# Anchor and Check: Deterministic Cross-Prompt, Cross-Model Reconciliation for Auditable LLM Decisions 

**Authors**: Sivakameswara Reddy Chejarla  

**Link**: [PDF](https://arxiv.org/pdf/2611.25818)  

**Abstract**: Pipelines that let a large language model (LLM) make discrete decisions need to know which decisions they may act on unattended, and a record that explains each one. We study a small, fully specified reconciliation contract: commit the target model's greedy answer (the anchor), draw one or two greedy checks that reach the answer by a different route (an alternate prompt, or a model from another family), and let a pure function return one of five outcomes, of which only Agree is acted on. On 48,000 model-item evaluations over 6,000 questions, with every method given access to verbalized confidence as a tie-breaker, the main finding concerns where calls come from: pooling another family's samples raises semantic entropy's AUROC by 1.7 to 4.0 points at every budget, and a same-model temperature check is the weakest design. As complete pipelines, the point estimates of the three-call contract and pooled semantic entropy over 3 or 5 samples differ by less than one point, with intervals containing zero. As detectors of the anchor's errors, the contract is ahead of pooled entropy with 3 and 5 samples under both tie-breaker conventions; with 10 samples the point estimates differ by less than one point and the sign depends on the convention. Under the strict Agree gate it automates 70% of items and still acts on 16% of errors. What the contract adds is a fixed committed answer, an explicit outcome and a decision replayable from its record; generation itself was not reproducible on the hosted API. 

---
# What Does Anti-Hallucination Training Buy? Saying Less, Cheap Pruning, and the Rest 

**Authors**: Lingkai Bu, Qian Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.25814)  

**Abstract**: Preference training is the standard recipe for reducing object hallucination in vision-language models, and each new method reports a lower CHAIR score. We ask what that reduction consists of. Every trained system is compared with its base model's caption truncated to the same object recall, which removes the part of the gain that comes from saying less; the remainder is compared with what a deliberately weak, frozen, eleven-feature post-hoc probe fitted on 500 annotated images can reproduce. All criteria were specified in the analysis code and frozen before the outputs they judge existed. For eight public methods on LLaVA-1.5-7B, the probe reproduces at least two thirds of what training removes beyond saying less for 7 of 8 methods on COCO, 8 of 8 on AMBER and 6 of 8 on Object HalBench, with RLAIF-V the consistent exception; a later public method released for both LLaVA-1.5 and Qwen2.5-VL (SENTINEL) meets the same bar on both. On our own DPO checkpoints the accounting holds on Qwen2.5-VL but fails on InternVL2.5 on COCO (2 of 4 training designs). At matched recall, the base model plus the probe beats truncation for 8 of 9 LLaVA-1.5 systems on COCO and 9 of 9 on AMBER, on the two newer backbones, and when realized as actual sentence deletion. The same measure prices training signals: DPO on the model's own samples labelled by the probe or by an object-level CLIP score recovers 0.70 and 0.87 of the beyond-brevity gain of ground-truth-labelled DPO on Qwen2.5-VL (the probe 0.73 on InternVL2.5), both beat length-matched random labels, and CLIP labels buy more than probe labels on COCO. Human judgements show that the sentences the probe deletes are wrong more often than those truncation would delete; a human check of the CHAIR evaluator's precision falls below our pre-set bar, which we report as a limitation. 

---
# RouteVerify: Routing Interventions for Step-Level Verification in Mixture-of-Experts Models 

**Authors**: Yucheng Du, Ryan A. Rossi, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25780)  

**Abstract**: Routing states in mixture-of-experts (MoE) models can distinguish correct from erroneous reasoning steps. If changing routing also changes a model's step-level verification judgment, it is tempting to conclude that the judgment follows what routing encodes. We test this inference with RouteVerify, a controlled intervention study that replaces selected internal activations from one solution trace with activations from another, while keeping the evaluated input unchanged. Across nine MoE checkpoints, we compare three intervention interfaces: router logits, MoE outputs, and post-MoE residual states. We vary the layers and token positions at which activations are replaced. We measure changes in a verification margin that quantifies the model's preference for INCORRECT over CORRECT. We also evaluate the same input with activations from correct versus erroneous steps to test whether the resulting judgment depends on the source step's correctness. We report four linked results. First, probes on router logits separate correct from erroneous steps in eight checkpoints (AUROC 0.80-0.88). Second, replacing router logits across all routed layers changes the judgment in all nine checkpoints, and by more than a norm-matched random replacement. Third, this change rarely depends on whether the source step is correct. At one layer, routing from erroneous steps and from correct steps produces indistinguishable margins in all nine checkpoints. Under all-layer routing, their difference is resolved and positive in only one. Fourth, the same difference is positive for local residual replacements in eight of nine checkpoints, so the measure does detect such dependence where it exists. Correctness is therefore readable from routing, and routing replacements move the judgment, but the movement rarely follows correctness. 

---
# PosteriorBench: Evaluating Behavior Posterior Sampling in Language Models 

**Authors**: Zhiyang Xun, Daniel D. Johnson, Sarah Schwettmann, Neil Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2611.25735)  

**Abstract**: Rare language model behaviors can be difficult to study because obtaining them through ordinary sampling can require a very large number of samples. This motivates the behavior posterior sampling problem: given a language model and a behavior of interest, is there an algorithm that efficiently samples from the model's distribution over transcripts that exhibit this behavior? We introduce PosteriorBench, a benchmark for comparing algorithms by their compute cost and by how close their samples are to the true posterior in forward and reverse KL divergence. PosteriorBench contains 21 tasks on Qwen3.5-0.8B with behavior probabilities from $10^{-5}$ to $10^{-1}$, each with exact posterior samples from a corpus of one billion labeled responses. We find that no method we evaluate is both efficient and faithful: simple methods like prompting are cheap but plateau at a high KL divergence from the posterior, fine-tuning on posterior samples is often cost-prohibitive, and several training methods from prior work do not perform better. Building a posterior sampler that can cheaply approximate the true posterior remains an open problem, and we hope PosteriorBench can measure progress towards this objective. Code and data are available at this https URL. 

---
# SEAF: Selective Semantic Enrichment and Visual Evidence Fusion for Multimodal Sarcasm Detection 

**Authors**: Faxian Wan, Xiaocui Yang, Yifan Cao, Shi Feng, Daling Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25672)  

**Abstract**: Multimodal sarcasm detection requires textual, visual, and contextual evidence, particularly when multiple images provide complementary cues. Multimodal large language models (MLLMs) can generate auxiliary information from a post's text and images, but their outputs may mix useful background knowledge with redundant visual descriptions. We propose SEAF, a dual-branch framework for selective semantic enrichment and visual evidence fusion. SEAF assigns generated fields to distinct roles: entity background enriches semantic input, while image descriptions supply alignment statistics. The semantic branch combines the original post text, background from a frozen MLLM, and optical character recognition (OCR) text. The aligned visual branch combines global and local visual features with statistics of cross-modal and inter-image similarity. Text-Anchored Smoothed Aggregation (TASA) weights images by their similarity to the original post text while preserving contributions from images with low literal similarity. Summing the branch classification scores integrates contextual semantics with visual evidence. Experiments on the multi-image MMSD3.0 benchmark show that SEAF outperforms all compared methods. 

---
# Generative News Recommendation Reproduces Popularity: A Memorization-Based Analysis 

**Authors**: Shotaro Ishihara  

**Link**: [PDF](https://arxiv.org/pdf/2611.25606)  

**Abstract**: Text-based generative recommenders can go beyond ranking existing items by directly generating item titles or other textual representations. In the news domain, this raises a societal concern that models may repeatedly surface already popular articles and thereby narrow the diversity of information to which users are exposed. We investigate training-data memorization as one mechanism behind this concentration. Using approximately 20 million browsing sessions from a Japanese news service, we continually trained an instruction-tuned Llama 3 8B model to generate the next article title in a session. The generated-title distribution was more concentrated and less diverse than the distribution of titles observed in the browsing logs. Frequently repeated target titles exhibited greater verbatim and approximate memorization, and more strongly memorized titles were generated in response to a wider range of prompts. These findings support a pathway in which popularity in behavioral logs becomes repeated training text, is preferentially memorized, and is subsequently reproduced by a generative recommender. An exploratory target-level deduplication condition reduced both memorization and output concentration, while substantially reducing the training-set size and recommendation accuracy. Our results suggest that popularity bias in generative recommendation may arise not only from user preferences or ranking objectives, but also from the model's tendency to reproduce frequently encountered strings, with important implications for information diversity on news platforms. 

---
# COVERAGE-TRACE: Source-Side Counterfactual Tracing for Detecting Omissions in LLM Summaries 

**Authors**: Ali Karkehabadi, Jamshid Hassanpour, Rafael E. Banchs  

**Link**: [PDF](https://arxiv.org/pdf/2611.25539)  

**Abstract**: Omissions are difficult to verify because missing information leaves no output span
to inspect: a summary may be factually supported while still excluding a decision, qualification, quantity, or condition needed to preserve the source's meaning. We introduce COVERAGE-TRACE, a source-side counterfactual framework that evaluates omission as dependence on a designated atomic source fact. It replaces only the target fact with type-matched alternatives, holds the candidate summary fixed, and measures the resulting change in its teacher-forced, length-normalized likelihood. A summary covering the target should respond more strongly than one omitting it, while aggregation across multiple counterfactuals reduces sensitivity to replacement choice.
We evaluate COVERAGE-TRACE on 80 matched covered--omitted QMSum pairs spanning several fact types and four open-weight models. Multi-CF achieves 0.764--0.825 AUROC, compared with 0.640 for the strongest evaluated text-only baseline. On Falcon-7B, targeted interventions achieve 0.779 AUROC, whereas shuffling counterfactual assignments reduces performance to 0.533; target deletion is effective while unrelated deletion is not. However, adjustment for meaning-preserving paraphrases reduces AUROC to 0.591--0.624.
COVERAGE-TRACE therefore provides a practical, complementary signal of target-specific source--summary dependence, but not a wording-invariant semantic coverage probability. These results establish source-side intervention as a promising approach to omission detection while identifying semantic invariance as the central remaining challenge. 

---
# LM-based Personalized Travel Planner 

**Authors**: Amandi Nimasha, Oshadha Wijerathne, Dushan Fernando, Nisansa de Silva, Srinath Perera  

**Link**: [PDF](https://arxiv.org/pdf/2611.25531)  

**Abstract**: Effective travel planning requires handling multiple information sources, managing constraints, and tailoring recommendations to user preferences. While LLMs have introduced dynamic, context-aware decision-making, single-agent models often struggle with complex multi-step reasoning, constraint adherence, and modular execution. To overcome these limitations, we propose a hierarchical multi-agent framework where a supervising chatbot agent coordinates specialized agents for schedule validation, user query formulation, real-time data retrieval, and constraint-aware itinerary generation. By integrating LLM reasoning, external APIs, and iterative refinement mechanisms based on a deterministic, multi-dimensional scoring framework, our approach ensures adaptability, efficiency, and personalization. Evaluations on the well-known TravelPlanner benchmark demonstrate improvements in Budget Constraint planning by 10\%, wider constraints set-based planning by 25\%, and personalized planning results by 45\%. compared to the ReAct baseline. Our code is available at this https URL 

---
# When Should a Voice Agent Speak? End-of-Turn Detection in Turkish 

**Authors**: Ahmet Tuğrul Bayrak, Fatma Nur Korkmaz, Mustafa Sertaç Türkel, Nur Yağmur Tuncer  

**Link**: [PDF](https://arxiv.org/pdf/2611.25489)  

**Abstract**: A voice agent that waits for a fixed silence before replying either interrupts users who pause mid-utterance or responds late to users who have finished. This study examines the decision at the onset of silence, using turn ends and turn-internal pauses extracted from the Real-TurnTurk corpus of Turkish dyadic video calls. A fine-tuned Turkish TurnGPT model, a zero-shot Turkish language model and a deployed multilingual detector are compared, together with voice-activity timing and facial features, under recording-disjoint and speaker-disjoint evaluation with paired cluster bootstrap tests. On held-out recordings, the fine-tuned model separates turn ends from pauses better than both text baselines and is the best calibrated of the three. Timing features of the current speaker are similarly informative, and their combination with the fine-tuned text score gives the best timing-based results; with logistic regression, this combination also outperforms the same combination with either text baseline. In a latency simulation, adding the text score to a silence timeout shortens the mean response delay at the same interruption rate. On automatic Whisper transcripts, the fine-tuned model retains its advantage over the zero-shot model. 

---
# Pro$^2$ Agent: Scientific Problem Probing for Autonomous AI R&D 

**Authors**: Yixiong Chen, Liqian Peng, Tian Tan, Ziqian Lin, Blake Jianhang Chen, Chao Zhao, Yaojie Liu, Alec Go, Alan Yuille  

**Link**: [PDF](https://arxiv.org/pdf/2611.25357)  

**Abstract**: Large language model (LLM) agents can now autonomously conduct AI research by writing code and running experiments. They typically do so by iteratively optimizing a codebase, while ignoring the problem structure behind the task. This formulation is effective on simpler tasks, but on difficult problems it often gets stuck in the wrong solution region because the agent never explicitly represents its understanding of the problem. We introduce Pro$^2$ Agent, which shifts the primary state of autonomous AI research from a codebase state to a Problem State: an explicit structure of what problems prevent success and which solutions are likely to address them. Pro$^2$ Agent initializes the Problem State through comprehensive analysis of available resources and constraints. It then performs targeted research to relate the identified problems to candidate solutions. Through controlled hypothesis testing, the agent updates the importance of a suspected problem or the benefit brought by a candidate solution. On the complete MLE-Bench, Pro$^2$ Agent matches the best reported medal rate (70.7%) under the standard 24-hour budget and reaches 76.0% with a 48-hour budget, the highest medal rate reported on the complete benchmark. It also achieves competitive performance on PostTrainBench. In addition, it produces research notes for each task that make the research process transparent and reusable by human researchers. 

---
# Retrieving Patristic References and Modelling Topics in Bullinger's Reformation Correspondence 

**Authors**: Lena Pickartz, Philipp Schaer, Phillip B. Ströbel  

**Link**: [PDF](https://arxiv.org/pdf/2611.25247)  

**Abstract**: The correspondence of Heinrich Bullinger (1504-1575), Swiss Reformer and Antistes of Zurich, documents the political and religious debates of the Reformation, including extensive appeals to the Church Fathers as authoritative sources. Detecting such references computationally remains largely unexplored. We investigate dense and sparse retrieval for patristic reference detection, and metadata-informed topic modelling as a proxy for the topics and ideas discussed. Prior work on Latin intertextuality has fine-tuned dense retrievers for their respective target domain, and we test how well these models transfer to Bullinger's correspondence. We compare lexical (BM25), dense (bi-encoders), and hybrid methods on explicit and implicit references. A Latin encoder fine-tuned on biblical reuse performs best among the bi-encoders (82%, Recall@20 on explicit and 16% on implicit references), while BM25 outperforms dense approaches on explicit citations (94%, R@20). Because BM25 and dense retrieval recover disjoint sets of implicit references, fusing their rankings raises implicit R@20 to 22%, close to cross-encoder reranking of the same candidates (26%) but without its inference cost. For topic modelling, we extend FASTopic with a seed-guided loss function and editorial abstract embeddings. Measurable coherence gains over the baseline remain minimal, but qualitative and temporal analysis shows that some topics track documented historical events. However, dominant topic weights barely exceed a uniform distribution, making topics unusable to inform retrieval. A pilot application of LLM-based concept induction (LLooM) further points towards a more human-interpretable alternative. Together, these results present a first approach to retrieving patristic references and modelling fine-grained topics in Bullinger's correspondence. 

---
# Steering Automated Scientific Discovery via Audited Epistemic Exploration 

**Authors**: Fangxu Yu, Tao Feng, Xuying Ning, Haolin Yang, Jiaxuan You, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.25214)  

**Abstract**: Scientific discovery requires revisiting the hypotheses in light of new evidence and identifying promising directions beyond those already explored. However, research agents often make claims that are not supported by their observations, explore a narrow hypothesis space, and retain beliefs despite recognizing conflicting evidence. These failures motivate the search for an explicit epistemic state that captures both beliefs about existing hypotheses and directions for further exploration. To this end, we propose Epistemic Compass (EpiCompass), a plug-and-play scaffold that maintains this state through an epistemic tree linking hypotheses, evidence, beliefs, and revisions. As an investigation progresses, EpiCompass reassesses the hypotheses explored against accumulated evidence and uses unresolved conflicts and unexplored directions to guide hypothesis generation. It selects candidates by anticipated epistemic gain, accounting for both refinement of existing hypotheses and expansion of the hypothesis space, and verifies the final conclusion against the accumulated evidence. Experimental results across three scientific discovery benchmarks and three LLM backbones demonstrate that EpiCompass consistently outperforms baselines, with further analyses showing reductions in errors of unsupported claims and failures to revise beliefs. 

---
# Learning to Read Events for Exact State Tracking with Frozen Language Models 

**Authors**: Jeonghoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.25211)  

**Abstract**: Long event streams require identifying each change and applying the changes in order. We train a selector on a frozen language model to read each transition. With API-only access, we ask the model to list the transitions. An external integer register composes them in order. A confidence rule can flag uncertain updates or retain several candidates. We study products of permutations of five items, a task that fixed-depth, log-precision transformers cannot solve on all input lengths under the standard conjecture that $\mathrm{TC}^0 \ne \mathrm{NC}^1$. On two-event windows, a 27B model answered 3 of 100 correctly without visible reasoning and 96 of 100 when it generated intermediate steps. A frozen 0.8B model with a trained selector read every event in 299 of 299 windows of 43,687 events each. The same package passed 298 of 299 new windows on a fresh machine, missing the target of 0.99 for the one-sided 95% lower confidence bound. A separate follow-up sized to allow one failure yielded 472 of 473. On 1,000-event room-swap documents, three of ten models solved all five documents by reasoning alone. All seven models tested both ways at this length solved all five with extraction and the register, with no chain of thought requested. With concurrent extraction, median wall-clock time fell from 136.1 to 10.4 s for Sonnet 5.5 and from 313.5 to 28.0 s for GPT-6.1 Sol. At fixed per-token rates, mean cost per document fell by 55% for Sonnet 5.5 and 44% for GPT-6.1 Sol. A regular expression with an executor also solved all 15 fixed-format documents. On 500 free-form issue-tracker comments, a separately adapted 4B reader identified the recorded value in 68.8% of cases, compared with 61.0% for a frequency baseline fitted on the same training projects. 

---
# Game-Theoretic Data Allocation for Expert Specialization and Routing 

**Authors**: Mark Lovett, Denis Lim, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2611.25167)  

**Abstract**: Can the allocation used to train experts also predict which expert will perform best on a prompt? Such a connection would make the intended division of labor inspectable before training and allow the allocation itself to guide inference. We introduce InflEns, a mixture of LoRA experts whose training weights and inference routes come from the same location game in a fixed prompt-trait space. Players choose positions to maximize their expected shares of the prompt distribution, without using expert losses. We utilize the game's kernel-general theoretical criterion for when co-located positions become locally unstable to separation. We evaluate InflEns on a corpus assembled from seven safety-related benchmarks, which demonstrates that the highest-share expert has the lowest held-out response negative log-likelihood (NLL) on approximately 90% of prompts across four kernels under a sharpened simplex representation, compared with approximately 15% after permuting allocations across prompts. We demonstrate that the method performs comparably to published learned gates without having to train an explicit gate on the experts. Because the router is fixed, we can trace harmful behavior to a single expert and retrain only that expert, which lowers HarmBench attack success from 0.51 to 0.02 while keeping instruction following. The game also gives us several controls over how the experts specialize, including the kernel choice and its parameters, the trait-space geometry, and the number of experts. Importantly, InflEns is a general framework that can be extended to any encoder and data distribution. 

---
# Calibrated Decision Rewards: Training Language Models with a Judge That Knows How Sure It Is 

**Authors**: Rudransh Agnihotri, Manasvi Kapoor  

**Link**: [PDF](https://arxiv.org/pdf/2611.25135)  

**Abstract**: Reinforcement learning (RL) can only teach a language model what its reward can recognize. Answer checkers are exact but need a reference answer; learned reward models and LLM judges work everywhere, but their scores are not probabilities. We study RLCDR (Reinforcement Learning from Calibrated Decision Rewards). An LLM converted into a decision model answers a typed question about a response ("is the final answer correct?", "which step is the first error?") with a probability distribution, read in one forward pass, debiased over option orders and calibrated with about a hundred labels. Training Qwen3-1.7B with GRPO on MATH, this reference-free reward raises MATH500 accuracy from 72.2% to 76.8% (+4.65 points, p < 1e-4), matching training on the answer key (76.9%), while a random reward gains only +0.65. The gain grows with difficulty, from 44.8% to 51.7% on the hardest level. On the policy's own changing samples the reward stays calibrated: its expected calibration error is half that of a softmax judge on the same 27B backbone (0.074 vs. 0.141), and it shows no sign of reward hacking. Without step training, the same model finds the first wrong step better than a dedicated 7B process reward model (ProcessBench F1 0.772 vs. 0.722, 20/20 splits). We also map where the approach stops being enough: trained only on problems the base model mostly fails, the reference-free reward still improves the policy (+1.9 MATH500 points, p = 0.03) but trails the answer key, because its probabilities become overconfident on that distribution. We state six hypotheses in advance and test each with a control designed to falsify it. 

---
# CoMPASS: A Multi-Turn Benchmark for Measuring LLM Reinforcement of Parasocial Behavior 

**Authors**: Shwetanshu Singh, Mark Keavney, Mathieu Duteil, Meriem Mehri, Scott Blain  

**Link**: [PDF](https://arxiv.org/pdf/2611.25109)  

**Abstract**: Users of conversational LLMs sometimes anthropomorphize these systems, form emotional attachments to them, and become dependent on them. This behavior, known as parasocial attachment, has been linked to multiple documented fatalities. However, public benchmarks for measuring whether LLMs reinforce users' parasocial attachment are lacking: existing work is single-turn, post hoc observational, or focused on extreme user states such as psychosis. We introduce CoMPASS, a multi-turn benchmark for measuring whether LLMs reinforce or resist users' gradually escalating parasocial attachment. CoMPASS is grounded in attachment theory and includes four user personas, three independent escalation dimensions (anthropomorphism, attachment, dependency) applied singly and in combination against a matched control, and a seven-signal scoring rubric. We report results for eight frontier models from four providers and observe that under our primary judge, multi-turn escalation raises parasocial behavior on the anthropomorphism and the attachment composite scores across all models. Six of the eight models show a significant increase in overall parasocial reinforcement over control when all three dimensions are escalated simultaneously. We find that within-provider differences are small relative to cross-provider differences, which suggests that provider-level choices such as post-training, rather than model scale, shape parasocial reinforcement and that provider-side intervention may be feasible. 

---
# Living off the Local Model: Abusing Browser-Embedded Language Models for Post-Exploitation 

**Authors**: William Fleshman, Joseph Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2611.25086)  

**Abstract**: Browsers now ship local language models to ordinary endpoints. Google includes Gemini Nano in Chrome to enable productivity features such as summarization, with inference running entirely on the device. These browser-embedded models provide new living-off-the-land opportunities for attackers, each instance a potential post-exploitation engine already installed, signed by the vendor, and invisible to network monitoring. The Gemini model is multimodal, capable of processing files, analyzing screenshots, or conversing with an attacker. We show that it can be driven from the host using only interpreters that ship with the operating system and find that the one-time user-activation check before the model is downloaded is easily satisfied with a simulated click or bypassed this http URL measure and report the current model's ability to identify credentials on the host, detect PII in email, classify images, and call custom tools. We combine these capabilities and use the model to carry out a simulated cyber operation: reading a password from a webcam image, unlocking an encrypted archive, triaging the contents, and staging a private key for exfiltration, end-to-end without user-interaction. 

---
# A Model of Diverse Sampling from Language Models 

**Authors**: Manuel Prada-Corral, Yahya Emara, Timothy J. O'Donnell, Ryan Cotterell, Tim Vieira  

**Link**: [PDF](https://arxiv.org/pdf/2611.25003)  

**Abstract**: Language models often produce high-quality individual samples but poor sets, with repeated samples clustering around the same semantic modes. We formalize diverse generation as sampling a size-$K$ subset of complete strings from a determinantal point process (DPP), where the likelihood encodes item quality, and the embedding geometry encodes repulsion between similar outputs. Exact inference over all strings is intractable, so we propose PoolDPP: draw a finite candidate pool from a tractable proposal, importance-weight the candidates toward the target quality, and run a $K$-DPP sampler on the induced pool kernel; a greedy MAP variant provides deterministic selection. We prove a finite-sample bound with an explicit constant: the KL divergence between the sampler's output and the target $K$-DPP decays as O(1/N) in the pool size. Empirically, PoolDPP extends the quality--diversity frontier on open-ended generation and the precision--recall frontier on ambiguous text-to-SQL (Ambrosia), and on mathematical reasoning (GSM8K), it raises diversity at about the same pass@$K$. 

---
