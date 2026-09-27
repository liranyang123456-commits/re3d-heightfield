import torch
print(torch.__version__)
print("cuda", int(torch.cuda.is_available()))
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
