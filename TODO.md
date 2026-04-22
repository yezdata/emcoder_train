# Emotions spread index / Entropy
- roztpyl ci spread primo z mean probs kazde emoce 
- pracuji s vyslednymi probs



# MC Dropout
- technicka prace se spolehlivosti modelu - rozptyl rika o spolehlivosti dane predikce
- N<=50
Metrika,Zdroj,Co ti to říká o textu/modelu
Mean Probs,mean(outputs),Sémantický obsah. (Co autor cítí/píše).
Entropy,Rozptyl mezi třídami,Ambiguitita textu. (Jak moc je emoce v textu nejasná).
MC Variance,std(outputs),Kvalita modelu. (Jak moc můžeme modelu v tomto případě věřit).
- Rozptyl (Variance/Std) z MC Dropoutu používáš jako filtr kvality. Je to tvůj "detektor lži" pro model.
    Vyřazení predikcí: Pokud je rozptyl u daného textu příliš vysoký, můžeš automaticky říct: "Tuto predikci ignorujeme, model si ji vymýšlí (halucinuje), protože tento typ textu nezná."

# Pre Layer Norm
# Pre-Norm Residual Connection
x = x + self.attention(self.norm1(x))
x = x + self.ffn(self.norm2(x))


## SDPA: Implementuj přes torch.nn.functional.scaled_dot_product_attention
import torch.nn.functional as F

# v rámci tvé Attention třídy:
attn_output = F.scaled_dot_product_attention(
    q, k, v, 
    attn_mask=mask, 
    dropout_p=self.dropout if self.training else 0,
    is_causal=False # Protože děláš Encoder
)
## BFLOAT16
- torch.cuda.amp.autocast
## Accelerate
## keep_in_memory=True pro dataset.map
- nastavit num_workers v DataLoaderu na vysoké číslo (např. 16 nebo 32), což zajistí, že batche budou připraveny v RAM dříve, než GPU stihne dokončit předchozí krok (tzv. prefetching).


# Pre-training - OpenWebText
Tímto procesem se váhy v Attention hlavách a embeddingy vylepšují tak, aby obsahovaly sémantickou informaci.
- Learning Rate Warmup
- Weight Tieing:
    - V pre-trainingu doporučuji sdílet váhy mezi embedding vrstvou na vstupu a lineární vrstvou MLM headu na výstupu. Ušetříš 38M parametrů a urychlíš konvergenci.
- MLM
    - Dynamic Masking? (Tedy že v každé epoše zamaskuješ jiná slova ve stejné větě).
    - Z každého inputu (OpenWebText) náhodně vybereš 15 % tokenů pro modifikaci:

    80 % z těchto vybraných tokenů nahradíš speciálním tokenem <mask>.

    10 % nahradíš úplně náhodným tokenem ze slovníku (nutí model udržovat reprezentaci i pro slova, která tam "nepatří").

    10 % ponecháš beze změny (nutí model pamatovat si skutečný vstup).

    Loss Function
    Model na výstupu vrací tenzor o velikosti [batch_size, seq_len, vocab_size]. Loss počítáš pouze nad těmi 15 % pozic, které byly vybrány k modifikaci. Použiješ standardní Cross Entropy Loss.
- Max Sequence Length: 512 tokenů (standard).Batch Size: Cílí na celkový batch size kolem 2048 až 4096 (využij Gradient Accumulation, pokud se ti do VRAM nevejde nativní batch).Optimizer: AdamW s weight_decay=0.01.Learning Rate Schedule: Lineární warmup (prvních 6–10 % kroků) následovaný lineárním nebo kosinovým poklesem. Špičková LR kolem $1 \cdot 10^{-4}$ až $6 \cdot 10^{-4}$.


# Class training - GoEmotions
- **Weighted BCEWithLogitsLoss** \ Downsampling Neutral na napr prumer ostatnich emoci

- change MLM na Class hlavu
    - Zahoď MLM Head: Odstraň lineární vrstvu, která mapovala embeddingy zpět na slovník (50k).
- mean-pooling
    - Vezmeš výstupy z poslední vrstvy enkodéru pro všechny tokeny (např. tenzor [batch, seq_len, 768]).
    Aplikuješ Masked Mean Pooling (musíš ignorovat padding tokeny, jinak ti průměr zkreslí!).
    Výsledný vektor [batch, 768] pošleš do finální lineární vrstvy pro 28 emocí.
    - Výstupem tvého Transformer enkodéru pro jednu větu o délce $L$ je matice o rozměrech $[L \times d_{model}]$. Mean pooling tuto matici „sploští“ (zprůměruje přes osu sekvence) na jediný vektor o délce $d_{model}$, který reprezentuje celou větu.Zde je technický rozbor a varování před nejčastějšími chybami v této fázi:1. Matematická operacePokud máš výstupy z poslední vrstvy $H = [h_1, h_2, ..., h_L]$, tvůj pooling vypadá takto:$$\text{sentence\_vector} = \frac{1}{L} \sum_{i=1}^{L} h_i$$Tento vektor pak skutečně pošleš do nn.Linear(d_model, 28).
    - **MASKED PRO PAD TOKENY**
        - # last_hidden_state: [batch, seq_len, d_model]
        # attention_mask: [batch, seq_len] (1 pro slovo, 0 pro padding)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask  # Výsledek: [batch, d_model]
    - Někdy je užitečné mezi mean_pooled vektor a finální vrstvu vložit ještě jednu nelinearitu (tzv. Projection Head):

        Linear(d_model, d_model)
        Tanh() nebo GELU()
        Dropout(0.1)
        Linear(d_model, 28)

        To dává modelu trochu více "kapacity" na to, aby se naučil separovat oněch 27 jemných emocí, aniž by musel drasticky měnit naučené váhy v enkodéru.

    - Nezprůměrovávej pravděpodobnosti ($probs$) po každém průchodu. Ulož si všech $N$ výstupních vektorů (logity) a teprve pak nad nimi počítej statistiku. Proč? Protože rozptyl v logitech ti řekne o chování modelu víc než rozptyl v už „zmáčknutých“ pravděpodobnostech po sigmoidě/softmaxu.