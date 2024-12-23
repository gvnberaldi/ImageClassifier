import torch
import wandb


class WandBLogger:
    def __init__(self, api_key: str, enabled=True, model: torch.nn.modules=None,
                 run_name: str=None,
                 project_name: str=None,
                 entity_name: str = None) -> None:

        wandb.login(key=api_key)

        self.enabled = enabled

        if self.enabled:
            wandb.init(entity=entity_name, project=project_name)
            if run_name is None:
                wandb.run.name = wandb.run.id    
            else:
                wandb.run.name = run_name  

            if model is not None:
                self.watch(model)

    def watch(self, model, log_freq: int=1):
        wandb.watch(model, log="all", log_freq=log_freq)

    def log(self, log_dict: dict, commit=False, step=None):
        if self.enabled:
            if step:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)

    def finish(self):
        if self.enabled:
            wandb.finish()
