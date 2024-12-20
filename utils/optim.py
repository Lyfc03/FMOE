from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_optim(max_steps,optim_name, model, lr,warmup_rate,weight_decay):
    warmup_steps = int(warmup_rate * max_steps)  

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(max_steps - current_step) / float(max(1, max_steps - warmup_steps)))

    if optim_name == 'Adam':
        optim = Adam(model.parameters(), lr)

    elif optim_name == 'SGD':
        optim = SGD(model.parameters(), lr)

    elif optim_name == "AdamW":
        base_learning_rate = lr
        optim = AdamW(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay)

    scheduler = LambdaLR(optim, lr_lambda)
    return optim, scheduler
