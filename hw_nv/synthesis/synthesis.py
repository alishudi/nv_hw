import numpy as np
import torch

def synthesis(model, text, device, waveglow, n, speed=1.0, alpha_e=1.0, alpha_p=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=speed, alpha_e=alpha_e, alpha_p=alpha_p)
    path = f"results/s={speed}_{n}_e={alpha_e}_p={alpha_p}.wav"
    inference(mel.contiguous().transpose(1, 2), waveglow, path)
    return mel, path

       
