# Proc?
- kompaktni model -> moznost inference i na spotrebitelskem HW \ jednoducha a rychla domain adaptace
- moje own reseni \ zadny pripraveny "black box"
- custom weights pro BCE loss eliminuji potrebu downsamplingu u tak nevyvazeneho datasetu jako goEmotions
## Proc MC Dropout?
- umozni mi napr zahodit predikce dane nejakym thresholdem pro epistemic_unc (std) -> zamezeni halucinanci modelu
- presnejsi aproximace skutecneho vysledku modelu kdyz se bere mean z vicero Dropout dependent forwardu
# MC Dropout
- technicka prace se spolehlivosti modelu - rozptyl rika o spolehlivosti dane predikce
- V klasickém Bayesovském přístupu hledáš posterior $P(w|D) \propto P(D|w)P(w)$.$P(w)$ (Prior): Schován v kombinaci Dropoutu a Weight Decay.$P(D|w)$ (Likelihood): Tvá loss funkce (Weighted BCE).
    - 1. Vztah mezi Dropoutem a $L_2$ regularizacíZásadní vědecký průlom (Gal & Ghahramani, 2016) dokázal, že trénování neuronové sítě s Dropoutem je matematicky ekvivalentní aproximaci Bayesovské inference v Gaussovských procesech.Weight Decay ($L_2$ regularizace): Ve svém kódu máš weight_decay=0.01. V Bayesovském světě odpovídá $L_2$ regularizace logaritmu Gaussova prioru nad váhami modelu.Prior: Tím, že penalizuješ velké váhy, efektivně říkáš: "Předpokládám (prior), že váhy jsou malé a centrované kolem nuly ($w \sim \mathcal{N}(0, \lambda^{-1})$)."
    - 2. Prior jako Bernoulliho distribuce (Dropout maska)V metodě MC Dropout definuješ architekturu, kde každá váha může být "vypnuta".Priorem je zde Bernoulliho distribuce s parametrem $p$ (tvůj dropout_rate = 0.15).Říkáš tím: "Můj prior předpokládá, že libovolná část sítě může chybět, a já hledám konfiguraci vah, která je robustní vůči tomuto výpadku."


### 82.1M Params
* 81.5M EmCoderCore 
    - 38.6M Token Embeddings
    - 0.39M Pos Embeddings
    - 42.47M Transformer Layers 
* 0.61M ClassifierHead
    - 0.59M First Linear 
    - 0.02M Final Linear


# Metrika,Zdroj,Co ti to říká o textu/modelu
Mean Probs,mean(outputs),Sémantický obsah. (Co autor cítí/píše).
Entropy,Rozptyl mezi třídami,Ambiguitita textu. (Jak moc je emoce v textu nejasná).
MC Variance,std(outputs),Kvalita modelu. (Jak moc můžeme modelu v tomto případě věřit).



# CLASS HEAD
    - Někdy je užitečné mezi mean_pooled vektor a finální vrstvu vložit ještě jednu nelinearitu (tzv. Projection Head):

        Linear(d_model, d_model)
        Tanh() nebo GELU()
        Dropout(0.1)
        Linear(d_model, 28)

        To dává modelu trochu více "kapacity" na to, aby se naučil separovat oněch 27 jemných emocí, aniž by musel drasticky měnit naučené váhy v enkodéru.