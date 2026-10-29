# Reference list for the A2 report (verified 29 Oct 2026)

This list replaces the plan's draft list (`docs/plan/sections/c-references.tex`) for the report. It follows the user's
rule (29 Oct 2026): **every source as recent as possible and no more than 7 years old**, i.e. published in 2020 or
later. The only exceptions are the unit's own documents, which are undated, and one dataset record, which needs a
decision (see the end of this file).

**How each entry was checked:** journal and conference DOIs against Crossref's metadata (title, authors, year, venue,
volume, issue, pages). Software releases against their Zenodo records. arXiv preprints against arXiv. Two claims
used in our tables against the papers' text: LLaMA's architecture, and EvolveGCN on Elliptic step 43. Where a
preprint and the published version have different titles, the published title is used.

**Style:** A1's numbered Harvard style. The list is alphabetical by first author and numbered when the report is
built. Citations in the text are author-date. Entries with more than four authors are shortened to "et al.";
expand them if Swinburne's Harvard guide asks for every author. The report keeps only the entries it cites.

## Main list

| Ref | Supports in the report |
|---|---|
| Akcora, C.G., Li, Y., Gel, Y.R. and Kantarcioglu, M. (2020) 'BitcoinHeist: topological data analysis for ransomware prediction on the Bitcoin blockchain', in *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI-20)*, pp. 4439–4445. doi: 10.24963/ijcai.2020/612. | DS4 source, its six graph features and label families |
| Angelopoulos, A.N. and Bates, S. (2023) 'Conformal prediction: a gentle introduction', *Foundations and Trends in Machine Learning*, 16(4), pp. 494–591. doi: 10.1561/2200000101. | Conformal ranges for the regression method (CQR), coverage |
| Ansel, J. et al. (2024) 'PyTorch 2: faster machine learning through dynamic Python bytecode transformation and graph compilation', in *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24)*, pp. 929–947. doi: 10.1145/3620665.3640366. | PyTorch, used for both language models |
| Antonopoulos, A.M. and Harding, D.A. (2023) *Mastering Bitcoin: programming the open blockchain*. 3rd edn. Sebastopol, CA: O'Reilly Media. | Background: transactions, inputs and outputs, fees, blocks |
| Ben Allal, L. et al. (2025) 'SmolLM2: when smol goes big — data-centric training of a small language model', *arXiv preprint*, arXiv:2502.02737. | DS6 (smol-smoltalk) for the chat stage; small-model training |
| Blázquez-García, A., Conde, A., Mori, U. and Lozano, J.A. (2022) 'A review on outlier/anomaly detection in time series data', *ACM Computing Surveys*, 54(3), pp. 1–33. doi: 10.1145/3444690. | Own-history (point) outliers in time series; robust statistics baselines |
| Blockstream (n.d.) *Esplora HTTP API*. Available at: https://github.com/Blockstream/esplora/blob/master/API.md (Accessed: 22 October 2026). | DS1 and DS10 collection (DS1 snapshot built 22 Oct 2026) |
| Boukerche, A., Zheng, L. and Alfandi, O. (2021) 'Outlier detection: methods, models, and classification', *ACM Computing Surveys*, 53(3), pp. 1–37. doi: 10.1145/3381028. | Taxonomy of outlier methods (Isolation Forest, LOF, clustering-based) |
| Bouthillier, X. et al. (2021) 'Accounting for variance in machine learning benchmarks', *Proceedings of Machine Learning and Systems*, 3 (MLSys 2021). arXiv:2103.03098. | Repeating evaluations over seeds (statistics A, B, C) |
| Brown, T.B. et al. (2020) 'Language models are few-shot learners', *Advances in Neural Information Processing Systems*, 33, pp. 1877–1901. | Decoder-only language models (the 152M model, TxGPT) |
| Bushra, A.A. and Yi, G. (2021) 'Comparative analysis review of pioneering DBSCAN and successive density-based clustering algorithms', *IEEE Access*, 9, pp. 87918–87935. doi: 10.1109/ACCESS.2021.3089036. | DBSCAN and HDBSCAN (clustering model type) |
| Card, D. et al. (2020) 'With little power comes great responsibility', in *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 9263–9274. doi: 10.18653/v1/2020.emnlp-main.745. | Paired tests (McNemar), bootstrap intervals and power for the writer comparisons |
| Chicco, D. and Jurman, G. (2020) 'The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation', *BMC Genomics*, 21, 6. doi: 10.1186/s12864-019-6413-7. | Why accuracy misleads on imbalanced labels; F1, precision, recall (DS3, DS4) |
| Elmougy, Y. and Liu, L. (2023) 'Demystifying fraudulent transactions and illicit nodes in the Bitcoin network for financial forensics', in *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23)*, pp. 3979–3990. doi: 10.1145/3580305.3599803. | Elliptic transactions (DS3) and Elliptic++ (DS8): description, labels, baselines |
| Fabbri, A.R. et al. (2021) 'SummEval: re-evaluating summarization evaluation', *Transactions of the Association for Computational Linguistics*, 9, pp. 391–409. doi: 10.1162/tacl_a_00373. | ROUGE-L and its limits (why our strict checks go beyond it) |
| Gebru, T. et al. (2021) 'Datasheets for datasets', *Communications of the ACM*, 64(12), pp. 86–92. doi: 10.1145/3458723. | DATA_CARD.md |
| Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022) 'Why do tree-based models still outperform deep learning on typical tabular data?', *Advances in Neural Information Processing Systems*, 35, Datasets and Benchmarks Track. arXiv:2207.08815. | Model selection: tree ensembles for the tabular classifiers and regression |
| Han, S., Hu, X., Huang, H., Jiang, M. and Zhao, Y. (2022) 'ADBench: anomaly detection benchmark', *Advances in Neural Information Processing Systems*, 35, Datasets and Benchmarks Track. arXiv:2206.09426. | Unsupervised detectors compared; synthetic anomaly types; no method wins everywhere |
| Hariri, S., Carrasco Kind, M. and Brunner, R.J. (2021) 'Extended isolation forest', *IEEE Transactions on Knowledge and Data Engineering*, 33(4), pp. 1479–1489. doi: 10.1109/TKDE.2019.2947676. | Isolation Forest (A1 extra) and its known biases |
| Harris, C.R. et al. (2020) 'Array programming with NumPy', *Nature*, 585(7825), pp. 357–362. doi: 10.1038/s41586-020-2649-2. | Software |
| Hilal, W., Gadsden, S.A. and Yawney, J. (2022) 'Financial fraud: a review of anomaly detection techniques and recent advances', *Expert Systems with Applications*, 193, 116429. doi: 10.1016/j.eswa.2021.116429. | Unsupervised anomaly detection where labels are scarce; framing as an aid, not a verdict |
| Hoffmann, J. et al. (2022) 'An empirical analysis of compute-optimal large language model training', *Advances in Neural Information Processing Systems*, 35. arXiv:2203.15556. | Token budget (about 20 tokens per parameter) for the 152M model |
| Hu, S. et al. (2024) 'MiniCPM: unveiling the potential of small language models with scalable training strategies', *arXiv preprint*, arXiv:2404.06395. | The warmup-stable-decay (WSD) schedule of the continued pretraining |
| HuggingFaceFW (2024) *FineWeb-Edu* [Dataset]. Hugging Face. doi: 10.57967/hf/2497. Available at: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu (Accessed: 20 October 2026). | DS5 |
| HuggingFaceTB (n.d.) *smol-smoltalk* [Dataset]. Hugging Face. Available at: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk (Accessed: 20 October 2026). | DS6 |
| Ikotun, A.M., Ezugwu, A.E., Abualigah, L., Abuhaija, B. and Heming, J. (2023) 'K-means clustering algorithms: a comprehensive review, variants analysis, and advances in the era of big data', *Information Sciences*, 622, pp. 178–210. doi: 10.1016/j.ins.2022.11.139. | K-Means and choosing k (clustering model type) |
| Ji, Z. et al. (2023) 'Survey of hallucination in natural language generation', *ACM Computing Surveys*, 55(12), pp. 1–38. doi: 10.1145/3571730. | Why every generated explanation passes the code guardrail (number faithfulness) |
| Kaplan, J. et al. (2020) 'Scaling laws for neural language models', *arXiv preprint*, arXiv:2001.08361. | The 6N FLOPs-per-token compute estimate |
| Kapoor, S. and Narayanan, A. (2023) 'Leakage and the reproducibility crisis in machine-learning-based science', *Patterns*, 4(9), 100804. doi: 10.1016/j.patter.2023.100804. | Time-safe features and temporal splits (DR-02); frozen test files |
| Lai, K.-H., Zha, D., Xu, J., Zhao, Y., Wang, G. and Hu, X. (2021) 'Revisiting time series outlier detection: definitions and benchmarks', *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*, 1 (NeurIPS 2021). Available at: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ec5decca5ed3d6b8079e2e7e7bacc9f2-Abstract-round1.html (Accessed: 29 October 2026). | Planting point and pattern outliers to evaluate without labels (spike, burst, dormant) |
| Le, V.-H. and Zhang, H. (2022) 'Log-based anomaly detection with deep learning: how far are we?', in *Proceedings of the 44th International Conference on Software Engineering (ICSE '22)*, pp. 1356–1367. doi: 10.1145/3510003.3510155. | Sequence models over event logs as anomaly scorers (TxGPT's idea) and their evaluation pitfalls |
| Liang, W. et al. (2024) 'Systematic analysis of 32,111 AI model cards characterizes documentation practice in AI', *Nature Machine Intelligence*, 6(7), pp. 744–753. doi: 10.1038/s42256-024-00857-z. | MODEL_CARD.md |
| mempool.space (n.d.) *REST API documentation*. Available at: https://mempool.space/docs/api/rest (Accessed: 22 October 2026). | DS2 crawl (22-24 Oct 2026); DS1's fallback host |
| Mielke, S.J. et al. (2021) 'Between words and characters: a brief history of open-vocabulary modeling and tokenization in NLP', *arXiv preprint*, arXiv:2112.10508. | Our byte-pair-encoding tokenizer |
| Pareja, A. et al. (2020) 'EvolveGCN: evolving graph convolutional networks for dynamic graphs', *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(04), pp. 5363–5370. doi: 10.1609/aaai.v34i04.5984. | Elliptic's collapse from step 43 (a dark-market shutdown), as in our classifiers |
| Penedo, G. et al. (2024) 'The FineWeb datasets: decanting the web for the finest text data at scale', *Advances in Neural Information Processing Systems*, 37, pp. 30811–30849. doi: 10.52202/079017-0970. | DS5 (FineWeb-Edu) |
| Pineau, J. et al. (2021) 'Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program)', *Journal of Machine Learning Research*, 22(164), pp. 1–20. | Reproducibility: pinned versions, seeds, the rebuild check, the Mac replication |
| scikit-learn developers (2025) *scikit-learn* (Version 1.6.1) [Software]. Zenodo. doi: 10.5281/zenodo.14627164. | Every classical model (named by the brief) |
| Shazeer, N. (2020) 'GLU variants improve Transformer', *arXiv preprint*, arXiv:2002.05202. | SwiGLU feed-forward layer |
| Su, J. et al. (2024) 'RoFormer: enhanced transformer with rotary position embedding', *Neurocomputing*, 568, 127063. doi: 10.1016/j.neucom.2023.127063. | Rotary position embeddings |
| The Matplotlib Development Team (2025) *Matplotlib: visualization with Python* (Version 3.10.1) [Software]. Zenodo. doi: 10.5281/zenodo.14940554. | Figures (named by the brief) |
| The pandas development team (2024) *pandas-dev/pandas: Pandas* (Version 2.2.3) [Software]. Zenodo. doi: 10.5281/zenodo.13819579. | Data processing (named by the brief) |
| Touvron, H. et al. (2023) 'LLaMA: open and efficient foundation language models', *arXiv preprint*, arXiv:2302.13971. | The 152M architecture: pre-norm RMSNorm, SwiGLU, rotary embeddings, AdamW (checked in its section 2) |
| UCI Machine Learning Repository (2020) *Bitcoin Heist Ransomware Address* [Dataset]. doi: 10.24432/C5BG8V. Available at: https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset (Accessed: 20 October 2026). | DS4 |
| Virtanen, P. et al. (2020) 'SciPy 1.0: fundamental algorithms for scientific computing in Python', *Nature Methods*, 17(3), pp. 261–272. doi: 10.1038/s41592-019-0686-2. | Statistical tests (binomial / McNemar) |
| Wu, R. and Keogh, E.J. (2023) 'Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress', *IEEE Transactions on Knowledge and Data Engineering*, 35(3), pp. 2421–2429. doi: 10.1109/TKDE.2021.3112126. | Why planted tests need strict rules (no trivial plants, flag budgets, frozen seeds) |

Unit documents (undated; kept as in A1): Li, N. (n.d.) *Week 4–7 lecture slides*, COS30049 Computing Technology
Innovation Project, Swinburne University of Technology (copy the exact titles from each deck's first slide);
Swinburne University of Technology (n.d.) *COS30049 Assignment 2 brief* and *Assignment 3 brief*.

Optional, only if the report names them: Bellei, C. et al. (2024) 'The shape of money laundering: subgraph
representation learning on the blockchain with the Elliptic2 dataset', *KDD Workshop on Machine Learning in Finance
(MLF '24)*. arXiv:2404.19109.

## One decision for the user: the Elliptic dataset record (2019)

DS3 is Elliptic's Kaggle dataset, released in 2019. Its licence (CC BY-NC-ND 4.0) and its page ask users to credit
it, and DATA_CARD.md and LICENSES.md do (Weber et al., 2019). That is licence attribution and stays. For the report,
choose one:
1. **Recommended:** cite the dataset record itself as the data source: Elliptic (2019) *Elliptic Data Set*
   [Dataset]. Kaggle. Available at: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set. A data source is
   not literature. Every claim about the data is then supported by the 2020+ papers above (Elmougy and Liu 2023;
   Pareja et al. 2020).
2. **Strict 7-year rule:** name the dataset in the text only, and cite Elmougy and Liu (2023), who distribute and
   describe the same transactions.

## Older entries in the plan's draft list and what replaces them

| Plan's entry (older than 7 years) | Replaced by |
|---|---|
| Liu, Ting and Zhou (2008) Isolation Forest | Hariri et al. (2021); Han et al. (2022) |
| Breunig et al. (2000) LOF | Boukerche et al. (2021); Han et al. (2022) |
| Ester et al. (1996) DBSCAN | Bushra and Yi (2021) |
| Lloyd (1982) K-Means; Caliński and Harabasz (1974); Davies and Bouldin (1979); Rousseeuw (1987) silhouettes | Ikotun et al. (2023) |
| Iglewicz and Hoaglin (1993) robust z | Blázquez-García et al. (2022) |
| Breiman (2001); Friedman (2001); Ke et al. (2017); Cover and Hart (1967); Cox (1958); Hoerl and Kennard (1970) | scikit-learn developers (2025); Grinsztajn et al. (2022) |
| Koenker and Bassett (1978) quantile regression | Angelopoulos and Bates (2023) |
| Saito and Rehmsmeier (2015) PR curves | Chicco and Jurman (2020) |
| Cohen (1960); Mann and Whitney (1947) | Card et al. (2020) (paired tests); Virtanen et al. (2020) |
| Vaswani et al. (2017); Radford et al. (2019); Zhang and Sennrich (2019); Loshchilov and Hutter (2019) | Touvron et al. (2023); Brown et al. (2020) |
| Sennrich, Haddow and Birch (2016) BPE | Mielke et al. (2021) |
| Lin (2004) ROUGE | Fabbri et al. (2021) |
| Mitchell et al. (2019) model cards | Liang et al. (2024) |
| Paszke et al. (2019) PyTorch; Pedregosa et al. (2011); McKinney (2010); Hunter (2007) | Ansel et al. (2024); the 2024-2025 software releases |
| Weber et al. (2019) (Elliptic's own paper) | Elmougy and Liu (2023); Pareja et al. (2020); the dataset record (decision above) |
| Conti et al. (2018); Huang et al. (2018); Paquet-Clouston et al. (2019) (BitcoinHeist's label sources) | Akcora et al. (2020), which names them |
