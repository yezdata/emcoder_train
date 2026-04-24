# Proc?
- kompaktni model -> moznost inference i na spotrebitelskem HW \ jednoducha a rychla domain adaptace
- moje own reseni \ zadny pripraveny "black box"
## Proc MC Dropout?
- umozni mi napr zahodit predikce dane nejakym thresholdem pro epistemic_unc (std) -> zamezeni halucinanci modelu
- presnejsi aproximace skutecneho vysledku modelu kdyz se bere mean z vicero Dropout dependent forwardu
# MC Dropout
- technicka prace se spolehlivosti modelu - rozptyl rika o spolehlivosti dane predikce




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