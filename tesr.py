import torch

#Verficiar si CUDA esta disponible
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA esta disponible')
else:
    device = torch.device('cpu')
    print('CUDA no esta disponible')

x = torch.rand(5, 3, device=device)
print(x)

#Si usamos GPU, tendremos mas velocidad en el procesamiento de datos.

#TENSORES

#Un tensor de 2 dimensiones con valores enteros predefinidos

tensor1 = torch.tensor([[5,7,9], [8,4,2]], dtype=torch.int32)
print(tensor1)

#Un tensor de 2 dimensiones con valores decimales predefinidos
tensor2 = torch.tensor([[5.3,6.2,1.5],[5.5,1.4,3.8]], dtype = torch.float32)
print(tensor2)

#Un tensor de 3 dimensiones con valores aleatorios:
tensor3 = torch.rand(3,4,2)
print(tensor3)

import pandas as pd

RUTA = '/content/XY_clasificacion.csv'
df = pd.read_csv(RUTA)
df