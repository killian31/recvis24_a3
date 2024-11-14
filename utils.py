class WarmupScheduler:
    def __init__(self, optimizer, warmup_iters, target_lr):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.target_lr = target_lr
        self.current_iter = 0

    def step(self):
        self.current_iter += 1
        if self.current_iter <= self.warmup_iters:
            # exponentially increase the learning rate
            warmup_lr = self.target_lr * (self.current_iter / self.warmup_iters) ** 2
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr

    def is_warmup_done(self):
        return self.current_iter >= self.warmup_iters
