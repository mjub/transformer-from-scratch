import datetime
import hashlib
import json
import os
import types

import torch
import tqdm
import transformers

import model


class Run:
    def __init__(self, config):
        self.config = config
        self.model = model.Transformer(config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )
        self.scheduler = transformers.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.05 * config.max_steps),
            num_training_steps=config.max_steps,
        )
        self.global_step = 0
        self.tokens_seen = 0

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # We use the first 8 characters of the hash of the configuratio to give
        # a somewhat unique name to this run
        hash = hashlib.sha256(
            bytes(repr(config), encoding="utf-8"), usedforsecurity=False
        ).hexdigest()[:8]
        self.name = f"{config.name}-{num_params/1e6:.1f}M-{hash}"

        self.loss_history = []

    @classmethod
    def from_file(cls, path):
        states = torch.load(path)

        run = cls(states["config"])
        # run.config gets overwritten but it doesn't matter
        for attr in states:
            if hasattr(run, attr):
                if hasattr(getattr(run, attr), "load_state_dict"):
                    getattr(run, attr).load_state_dict(states[attr])
                else:
                    setattr(run, attr, states[attr])
        return run

    def save(self, path):
        torch.save(
            {
                "name": self.name,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "config": self.config,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "tokens_seen": self.tokens_seen,
                "loss_history": self.loss_history,
            },
            path,
        )


class Trainer:
    def __init__(self, config, run, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.config = config
        self.run = run

        self.run.model.to(self.device)

        self.train_data = torch.load(config.train_data).to(self.device)
        self.val_data = torch.load(config.val_data).to(self.device)

    def get_batch(self, mode="train"):
        data = self.train_data if mode == "train" else self.val_data

        N = data.size(0)
        start = torch.randint(
            0,
            N - self.config.max_position_embeddings - 1,
            (self.config.per_device_train_batch_size,),
            device=data.device,
        )
        pos = (
            start[:, None]
            + torch.arange(self.config.max_position_embeddings, device=data.device)[
                None, :
            ]
        )
        return data[pos], data[pos + 1]

    def step(self):
        x, y = self.get_batch("train")
        self.run.tokens_seen += x.numel()

        _, loss = self.run.model(x, labels=y)

        self.run.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.run.model.parameters(), 1.0)

        self.run.optimizer.step()
        self.run.scheduler.step()
        self.run.global_step += 1

    def train(self):
        run_dir = os.path.join(self.config.runs_dir, self.run.name)
        os.makedirs(run_dir, exist_ok=True)
        for _ in tqdm.tqdm(range(self.run.global_step, self.config.max_steps)):
            self.step()
            if self.run.global_step % self.config.eval_steps == 0:
                losses = self.evaluate()

                print(
                    f'Step {self.run.global_step}: train loss = {losses["train"]:.6f}, val loss = {losses["val"]:.6f}'
                )

                self.run.loss_history.append(
                    {
                        "global_step": self.run.global_step,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "tokens_seen": self.run.tokens_seen,
                    }
                )

                self.run.save(
                    os.path.join(
                        self.config.runs_dir,
                        self.run.name,
                        f"{self.run.name}-{self.run.global_step}.pt",
                    )
                )

    @torch.no_grad()
    def evaluate(self):
        losses = {}
        self.run.model.eval()
        for mode in ["train", "val"]:
            total = 0.0
            for _ in range(self.config.max_eval_samples):
                x, y = self.get_batch(mode)
                _, loss = self.run.model(x, labels=y)
                total += loss.item()
            losses[mode] = total / self.config.max_eval_samples
        self.run.model.train()
        return losses


if __name__ == "__main__":
    with open("config.json") as fd:
        config = types.SimpleNamespace(**json.load(fd))
    trainer = Trainer(config, Run(config))
    trainer.train()
