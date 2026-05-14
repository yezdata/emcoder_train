# V2
Temperature scaling pro kalibraci pravděpodobností
    - Temperature Scaling slouží exkluzivně k optimalizaci Expected Calibration Error (ECE) a Negative Log-Likelihood (NLL) pro bodové odhady. Používá se k tomu, aby průměrná predikovaná pravděpodobnost odpovídala empirické přesnosti (accuracy) na daném binu dat.
    - TS neléčí příčinu (špatný odhad posteriorního rozdělení nebo strukturální zkreslení kvůli class imbalance), TS pouze zahlazuje symptomy přehnaného sebevědomí (overconfidence) plošně.

NEW Class-Balanced Loss
  - Udělat soubor losses.py -> tam nechat starou loss Class + pridat novou a vzdy jen vybrat kterou importovat a pouzit pro train
  - nejak zanest i do train_config.json jaka loss byla pouzita pro dany trenink
  - Asymmetric Loss (maybe focal?)


Grouped Query Attention (GQA)
rotační embeddings (RoPE)


Oversampling minority classes /
  Syntetická data pro minority (grief, relief, nervousness)


Variable / větší max_seq_len
