test_data = load_dataset("json", data_files={"test": "data/test_data.jsonl"})["test"]




logits_samples = model.mc_forward(input_ids, mask, n_samples=10)
mean_logits = results.mean(dim=0)  # Průměrná predikce pro každou větu v batchi
std_logits = results.std(dim=0)

probs = torch.sigmoid(logits_samples) # Pro GoEmotions (multi-label)
mean_probs = probs.mean(dim=0)
uncertainty = probs.std(dim=0)




model.eval()              # 1. Celý model do eval módu (vypne LayerNorm update)
model.set_mc_dropout(True) # 2. Selektivně zapne dropouty

all_logits = []
with torch.inference_mode():     # 3. Vypne výpočet gradientů (šetříš VRAM na RTX 6000)
    for _ in range(N):     # N = 50
        logits = model(batch_input, batch_mask)
        all_logits.append(logits)

# Nyní máš v all_logits 50 průchodů pro statistiku (mean, std)