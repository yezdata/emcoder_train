# EmCoder
> **Probabilistic Emotion Recognition & Uncertainty Quantification**<br>**28 Emotion multi-label classifier**  



Unlike standard classifiers, EmCoder quantifies what it doesn't know using Monte Carlo Dropout, making it suitable for high-stakes AI pipelines.<br>


<!-- TODO: add HOW TO USE SECTION -->



## SOTA benchmark
### Evaluation on the GoEmotions test split (macro avg metrics)
EmCoder achieves competitive F1-scores while being ~35% smaller than RoBERTa-base and ~45% smaller than ModernBERT, offering a superior efficiency-to-uncertainty ratio.
| Model | Precision | Recall | F1-Score | Params |
| :--- | :--- | :--- | :--- | :--- |
| **EmCoder (v1)** | **0.408** | **0.495** | **0.440** | **82.1M** |
| Google BERT (Original) | 0.400 | 0.630 | 0.460 | 110M |
| RoBERTa-base | 0.575 | 0.396 | 0.450 | 125M |
| ModernBERT-base | 0.652 | 0.443 | 0.500 | 149M |



## Model Architecture
```mermaid
flowchart LR

subgraph InputGroup["Input Operations"]
    direction TB
    MCD_Loop(["MC-Inference Loop: N_samples"]):::LoopNode
    ids["Batch IDs"]
    mask["Batch Mask"]
end

subgraph EmCoderCore["EmCoder Core"]
    direction LR
    tok_emb["Token Embedding"]
    ln_in["Input LayerNorm"]
    Transformer["Transformer Encoder"]
    final_norm["Final LayerNorm"]
    Dropout1[("MC-Dropout")]
    Dropout2[("MC-Dropout")]
end

subgraph Row1[" "]
    direction LR
    InputGroup
    EmCoderCore
end

subgraph MLP["Classifier MLP"]
    L_lin["Linear 1"]
    Dropout3[("MC-Dropout")]
    GELU["GELU"]
    F_lin["Final Linear"]
end

subgraph ClassifierHead[" "]
    direction TB
    pool["Masked Mean Pooling"]
    MLP
end

subgraph Row2[" "]
    direction LR
    ClassifierHead
    Out(["Class LogitsMC
    (n_samples, B, 28)"])

    Avg["Bayesian Post-processing"]
end

tok_emb ==> ln_in
ln_in -.-> Dropout1
Dropout1 ==> Transformer
Transformer -.-> Dropout2
Dropout2 ==> final_norm
MCD_Loop -.-> ids
ids ==> tok_emb
final_norm ==> pool
mask ==> pool
pool ==> L_lin
L_lin -.-> Dropout3
Dropout3 ==> GELU
GELU ==> F_lin
F_lin ==> Out
Out ==> Avg
mask ==> Transformer

classDef MCD fill:#424242,stroke:#fbc02d,stroke-width:2px,stroke-dasharray: 5 5,color:#fff
classDef OutNode fill:#0d47a1,stroke:#1976d2,stroke-width:3px,color:#fff,font-weight:bold
classDef BayesNode fill:#3e2723,stroke:#8d6e63,stroke-width:2px,stroke-dasharray: 3 3,color:#fff
classDef LoopNode fill:#263238,stroke:#78909c,stroke-width:2px,color:#fff,font-style:italic
classDef LightNode fill:#212121,stroke:#90a4ae,color:#fff

class MCD_Loop LoopNode
class ids,mask,tok_emb,ln_in,Transformer,final_norm,L_lin,GELU,F_lin,pool LightNode
class Dropout1,Dropout2,Dropout3 MCD
class Out OutNode
class Avg BayesNode

style InputGroup fill:#1a1a1a,stroke:#444,color:#fff
style EmCoderCore fill:#2d1a2d,stroke:#6a1b9a,color:#fff
style MLP fill:#212121,stroke:#455a64,color:#fff
style ClassifierHead fill:#012a4a,stroke:#01497c,color:#fff
style Row1 fill:none,stroke:none
style Row2 fill:none,stroke:none

linkStyle 2 stroke:#fbc02d,stroke-width:2px,fill:none
linkStyle 5 stroke:#fbc02d,stroke-width:2px,fill:none
linkStyle 11 stroke:#fbc02d,stroke-width:2px,fill:none
```

### Optimization
The model is trained using a Weighted Bayesian Binary Cross Entropy loss:
$$ \mathcal{L}_{Bayesian} = \frac{1}{T} \sum_{t=1}^{T} \text{BCEWithLogits}(z^{(t)}, y; w) $$
where weights $w$ are calculated using a logarithmic class-balancing scale to handle extreme label imbalance.
$$ w_c = 1 + \log\left(\frac{N_{neg, c} + \epsilon}{N_{pos, c} + \epsilon}\right)$$



## Performance
**Using threshold of 0.5 for binarizing predictions**
|                |   precision |   recall |   f1-score |   support |
|:---------------|------------:|---------:|-----------:|----------:|
| admiration     |       0.541 |    0.673 |      0.599 |       504 |
| amusement      |       0.688 |    0.909 |      0.783 |       264 |
| anger          |       0.419 |    0.47  |      0.443 |       198 |
| annoyance      |       0.31  |    0.25  |      0.277 |       320 |
| approval       |       0.304 |    0.271 |      0.287 |       351 |
| caring         |       0.229 |    0.281 |      0.252 |       135 |
| confusion      |       0.26  |    0.497 |      0.342 |       153 |
| curiosity      |       0.432 |    0.764 |      0.552 |       284 |
| desire         |       0.453 |    0.518 |      0.483 |        83 |
| disappointment |       0.176 |    0.152 |      0.163 |       151 |
| disapproval    |       0.279 |    0.404 |      0.33  |       267 |
| disgust        |       0.447 |    0.545 |      0.491 |       123 |
| embarrassment  |       0.325 |    0.351 |      0.338 |        37 |
| excitement     |       0.288 |    0.427 |      0.344 |       103 |
| fear           |       0.47  |    0.692 |      0.56  |        78 |
| gratitude      |       0.834 |    0.943 |      0.885 |       352 |
| grief          |       0     |    0     |      0     |         6 |
| joy            |       0.445 |    0.652 |      0.529 |       161 |
| love           |       0.724 |    0.895 |      0.801 |       238 |
| nervousness    |       0.24  |    0.261 |      0.25  |        23 |
| optimism       |       0.483 |    0.543 |      0.511 |       186 |
| pride          |       0.667 |    0.375 |      0.48  |        16 |
| realization    |       0.226 |    0.166 |      0.191 |       145 |
| relief         |       0.222 |    0.182 |      0.2   |        11 |
| remorse        |       0.516 |    0.857 |      0.644 |        56 |
| sadness        |       0.405 |    0.545 |      0.464 |       156 |
| surprise       |       0.429 |    0.539 |      0.478 |       141 |
| neutral        |       0.602 |    0.695 |      0.645 |      1787 |
| micro avg      |       0.494 |    0.596 |      0.54  |      6329 |
| macro avg      |       0.408 |    0.495 |      0.44  |      6329 |
| weighted avg   |       0.492 |    0.596 |      0.535 |      6329 |
| samples avg    |       0.525 |    0.616 |      0.544 |      6329 |


**Model uncertainty estimation**
[epistemic_unc](outputs/v1/epistemic_unc_scatter.png)


**Confusion matrix**
[multi-label-confusion-matrix](outputs/v1/confusion_matrix.png)


## Workflow
```mermaid
flowchart LR
    classDef StageNode fill:#121212,stroke:#546e7a,color:#fff;
    classDef HighlightNode fill:#4e342e,stroke:#ff7043,stroke-width:2px,color:#fff,font-weight:bold;

    subgraph PT ["Phase 1: Pre-training"]
        direction TB
        OWT[(OpenWebText)]:::StageNode --> MLM[Masked Language Modeling]:::StageNode
        MLM --> Core[Save EmCoderCore]:::StageNode
    end

    subgraph FT ["Phase 2: Fine-tuning"]
        direction TB
        Core --> Init[Init ClassificationHead]:::StageNode
        GE[(GoEmotions)]:::StageNode --> WBT[Bayesian Fine-tuning]:::HighlightNode
        WBT --> LogW[Log-weighted BCE Loss]:::StageNode
        LogW --> Freeze[Step 0-500: Encoder Frozen]:::StageNode
    end

    subgraph EV ["Phase 3: Testing & Inference"]
        direction TB
        Freeze --> MCD[MC Dropout Inference]:::HighlightNode
        MCD --> Unc[Uncertainty Estimation]:::HighlightNode
        
        subgraph Metrics ["Analysis"]
            Unc --> EPI[Epistemic: Model Confidence]:::StageNode
            Unc --> ALE[Aleatoric: Data Ambiguity]:::StageNode
            Unc --> CM[Test set metrics]:::StageNode
        end
    end

    style PT fill:#0d1b2a,stroke:#1b263b,color:#fff
    style FT fill:#2e1500,stroke:#5d2a00,color:#fff
    style EV fill:#1b2e1b,stroke:#2d4a2d,color:#fff
    style Metrics fill:#000,stroke:#333,color:#fff

    linkStyle default stroke:#aaa,stroke-width:2px;
```


## Citation
If you use this model, please cite it as follows:

```bibtex
@software{jez2026emcoder,
  author = {Václav Jež},
  title = {EmCoder: Probabilistic Emotion Recognition & Uncertainty Quantification},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yezdata/emcoder}},
  version = {1.0.0}
}
```