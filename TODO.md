# V2
**! LONGER PRETRAIN - TAKE 1.5 EPOCH PRETRAIN -> CONTINUE ON NEW DATA (MIXED OPENWEBTEXT + NEW)**


Temperature scaling pro kalibraci pravděpodobností
    - Temperature Scaling slouží exkluzivně k optimalizaci Expected Calibration Error (ECE) a Negative Log-Likelihood (NLL) pro bodové odhady. Používá se k tomu, aby průměrná predikovaná pravděpodobnost odpovídala empirické přesnosti (accuracy) na daném binu dat.
    - TS neléčí příčinu (špatný odhad posteriorního rozdělení nebo strukturální zkreslení kvůli class imbalance), TS pouze zahlazuje symptomy přehnaného sebevědomí (overconfidence) plošně.


Grouped Query Attention (GQA)

Variable / větší max_seq_len
- rotační embeddings (RoPE)


Oversampling minority classes