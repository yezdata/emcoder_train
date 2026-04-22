encoder = MyEncoder(config) # Nová instance
encoder.load_state_dict(torch.load("encoder_weights.pth")) # Načti "chytré" váhy
model = MyModelForClassification(encoder, config) # Obal do klasifikační hlavy
# Teď máš model s naučeným enkodérem a náhodnou hlavou.