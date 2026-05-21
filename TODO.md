# V2 OR V2.5
## README
- Changes in architecture for V2
  - fix dropout in attention, ModernArchitecture
- Changes in training for V2
  - mix data...
- Performance of V2


**Krátce popsat:**
Uncertainty Decomposition
EmCoder computes probabilistic uncertainty using Information Theory metrics over $N$ stochastic forward passes:Total Uncertainty (Predictive Entropy): $H(\mathbb{E}[y|x])$ — measures the overall randomness of the model's prediction.Aleatoric Uncertainty (Expected Entropy): $\mathbb{E}[H(y|x)]$ — captures the inherent data ambiguity (label noise, overlapping semantics).Epistemic Uncertainty (Mutual Information): $I(y; \omega | x) = H(\mathbb{E}[y|x]) - \mathbb{E}[H(y|x)]$ — isolates the model's structural lack of knowledge, driving the F1-Rejection pipeline.


**Publish nejspis V2 ale zminit i experiment s V2_5:**
The Baseline (V1): Standardní BERT architektura, solidní základ (Macro F1: 0.463).The Architecture Leap (V2): Implementace moderních LLM prvků (RMSNorm, RoPE, SwiGLU) + optimalizovaný data-mix. Výsledek: Skok na 0.488 Macro F1  (překonání původního Google BERT o téměř 9 % relativně). Model vykazuje excelentní kalibraci nejistoty, kdy při odfiltrování 50 % nejistých dat stoupá Macro F1 nad 0.7.  The Optimization Experiment (V2.5): Nasazení Concrete Dropoutu pro automatické učení regularizace. Zde ukaž kritické inženýrské myšlení: "Experiment ukázal, že Concrete Dropout sice zpřesnil identifikaci nejistoty u hraničních případů (strmější počátek rejection curve), ale ustálená hodnota $p \approx 0.15$ vedla k příliš silné regularizaci klasifikační hlavy a narušila synergii s encoderem ($p=0.1$). Z toho důvodu byl jako finální produkční model zvolen V2."


**PRIDAT K F1 REJECTION JAK DO README TAK NA LINKEDIN:**
The steep increase in Macro F1 during epistemic uncertainty rejection is partially driven by the dataset's extreme class imbalance. The stochastic uncertainty estimation effectively identifies and filters out high-entropy underrepresented emotions (e.g., grief, pride, nervousness), leaving a clean, majority-dominated subset where the model performs with high confidence.


**REMOVE MEAN PROB vs EPISTEMIC PLOT:**
**INSTEAD: PER-CLASS PLOTS**
Udělej to jako Grid 2x2 nebo 1x3 vybraných archetypů:
Vyber 3 emoce, které reprezentují odlišné chování sítě, a dej je vedle sebe do jednoho kombinovaného obrázku. Tím demonstruješ chování modelu na různých typech distribucí:
  - neutral (Zástupce majoritní třídy s vysokou jistotou).
  - admiration nebo gratitude (Zástupce dobře naučené, specifické emocionální třídy s ukázkovou parabolou nejistoty).
  - disgust nebo approval (Zástupce těžké, málo zastoupené nebo ambivalentní třídy, kde je vidět vysoká epistemická nejistota i při nižších pravděpodobnostech).

The plots below demonstrate perfect Bayesian calibration: epistemic uncertainty naturally peaks around the decision boundary ($\mu \approx 0.5$) where the model's stochastic passes yield the highest variance, while aleatoric uncertainty correctly captures the inherent data ambiguity

**REMOVE MEAN PROB vs ALEATORIC PLOT:**
  
  




# V3
## FineTuning
Oversampling minority classes?
