import torch
def process_caption(caption, max_caption_length=200, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "):
    caption = caption.lower()
    alpha_to_num = {k:v+1 for k, v in zip(alphabet, range(len(alphabet)))}
    labels = torch.zeros(max_caption_length).long()
    max_i = min(max_caption_length, len(caption))
    for i in range(max_i):
        labels[i] =alpha_to_num(caption[i], alpha_to_num[' '])
    labels = labels.unsqueeze(1)
    one_hot =one_hot[:,1:]
    one_hot = one_hot.permute(1,0)
    return one_hot
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') !=-1:
        m.weight.data.normal(0.0,0.02)
    elif classname.find('BatchNorm') !=-1:
        m.weight.data.normal(1.0, 0.02)
        m.bias.data.fill(0)
        