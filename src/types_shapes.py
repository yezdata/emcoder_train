from torch import Tensor
from typing import Annotated


BatchIDs = Annotated[Tensor, "(B, S)"]
BatchMask = Annotated[Tensor, "(B, S)"]

TokenEmbeds = Annotated[Tensor, "(B, S, D)"]

MLMLogits = Annotated[Tensor, "(B, S, V)"]
MLMLabels = Annotated[Tensor, "(B, S)"]

ClassLogits = Annotated[Tensor, "(B, num_labels)"]
MeanPooled = Annotated[Tensor, "(B, D)"]