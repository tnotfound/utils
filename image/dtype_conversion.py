------- PIL -> PyTorch ------
from PIL import Image
import torchvision.transforms

img = Image.opne('img.jpg')
transforms.functional.to_tensor(img)
----------------------------

------- PIL -> NumPy -------
import numpy as np
from PIL import Image

img = Image.opne('img.jpg')
np.array(img)
----------------------------


------- Numpy -> PIL -------
import numpy as np
from PIL import Image

arr = np.zeros((256,256,3))
Image.fromarray(np.unit8(arr))
----------------------------


------ Numpy -> PyTorch ------
import numpy as np
import torch

arr = np.zeros((256,256,3))
torch.from_numpy(arr)
------------------------------


------ PyTorch -> PIL ------
import torch
import torchvision
 
tensor = torch.zeros((256, 256, 3))
transforms.functional.to_pil_image(tensor)
------------------------------


------ PyTorch -> Numpy ------
import torch

tensor = torch.zeros((256,256,3))
tensor.to('cpu').detach().numpy()
------------------------------
