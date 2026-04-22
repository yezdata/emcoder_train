model.eval()              # 1. Celý model do eval módu (vypne LayerNorm update)
model.set_mc_dropout(True) # 2. Selektivně zapne dropouty

all_logits = []
with torch.inference_mode():     # 3. Vypne výpočet gradientů (šetříš VRAM na RTX 6000)
    for _ in range(N):     # N = 50
        logits = model(batch_input, batch_mask)
        all_logits.append(logits)

# Nyní máš v all_logits 50 průchodů pro statistiku (mean, std)