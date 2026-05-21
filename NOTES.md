# V1
Smaller architecture + basic MC Dropout + bugs in lr scheduler (opting for 5 epoch, doesnt include in accelerator.prepare)
# V1_5
optimizing only finetune + classification head
# V2
optimizing primarly pretrain + Encoder:
  - Modern Transformer Architecture (SwiGLU, RMSNorm, RoPE) + fix missing mc dropout in attention (in .eval) + mix data and longer pretrain (opting for 1 epoch)
# V2.5
optimizing only finetune + classification head
  - Concrete Dropout






# Proc EmCoder?
- kompaktni model -> moznost inference i na spotrebitelskem HW \ jednoducha a rychla domain adaptace
    - efektivita - inference / training speed + nižší latence, menší nároky na paměť GPU
- moje own in-house reseni / zadny pripraveny popr. spatne zdokumentovany "black box"
- custom weights pro BCE loss eliminuji potrebu downsamplingu u tak nevyvazeneho datasetu jako goEmotions
## Proc MC Dropout?
- uncertainty quantification je dulezita soucast posouzeni vykonnosti modelu a dava nam vhled na epistemic unc
- efektivni aproximace deep gauss process - BNN, ktera narokuje mnohem vetsi vypocetni kapacitu
- umozni mi napr zahodit predikce dane nejakym thresholdem pro epistemic_unc (std) -> zamezeni halucinanci modelu v produkci
- presnejsi aproximace skutecneho vysledku modelu kdyz se bere mean z vicero Dropout dependent forwardu
# MC Dropout
- V klasickém Bayesovském přístupu hledáš posterior $P(w|D) \propto P(D|w)P(w)$.$P(w)$ (Prior): Schován v kombinaci Dropoutu a Weight Decay.$P(D|w)$ (Likelihood): Tvá loss funkce (Weighted BCE).
    - 1. Vztah mezi Dropoutem a $L_2$ regularizacíZásadní vědecký průlom (Gal & Ghahramani, 2016) dokázal, že trénování neuronové sítě s Dropoutem je matematicky ekvivalentní aproximaci Bayesovské inference v Gaussovských procesech.Weight Decay ($L_2$ regularizace): Ve svém kódu máš weight_decay=0.01. V Bayesovském světě odpovídá $L_2$ regularizace logaritmu Gaussova prioru nad váhami modelu.Prior: Tím, že penalizuješ velké váhy, efektivně říkáš: "Předpokládám (prior), že váhy jsou malé a centrované kolem nuly ($w \sim \mathcal{N}(0, \lambda^{-1})$)."
    - 2. Prior jako Bernoulliho distribuce (Dropout maska)V metodě MC Dropout definuješ architekturu, kde každá váha může být "vypnuta".Priorem je zde Bernoulliho distribuce s parametrem $p$ (tvůj dropout_rate = 0.15).Říkáš tím: "Můj prior předpokládá, že libovolná část sítě může chybět, a já hledám konfiguraci vah, která je robustní vůči tomuto výpadku."
