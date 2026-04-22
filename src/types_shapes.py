from torch import Tensor
from typing import Annotated


BatchIDs = Annotated[Tensor, "(B, S)"]
TokenEmbeds = Annotated[Tensor, "(B, S, D)"]
MeanPooled = Annotated[Tensor, "(B, D)"]
Logits = Annotated[Tensor, "(B, num_labels)"]

# TODO
# BatchMask