import datetime
import hashlib
import json
import logging as log
import math
import os
import pprint
import shutil
import time
import types

import torch
import torch.utils.tensorboard
import tqdm
import tqdm.contrib.logging
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

        self.best_validation_loss = float("inf")
        self.loss_history = []

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # We use the first 8 characters of the hash of the configuration to give
        # a somewhat unique name to this run
        config_hash = hashlib.sha256(
            bytes(repr(config), encoding="utf-8"), usedforsecurity=False
        ).hexdigest()[:8]
        self.name = f"{config.name}-{num_params/1e6:.1f}M-{config_hash}"

    @classmethod
    def from_file(cls, path):
        states = torch.load(path)

        run = cls(types.SimpleNamespace(**states["config"]))
        # A quick and dirty way to set the attributes
        for attr in states:
            if attr == "config":
                continue
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
                "config": vars(self.config),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "tokens_seen": self.tokens_seen,
                "best_validation_loss": self.best_validation_loss,
                "loss_history": self.loss_history,
            },
            path,
        )


class Trainer:
    def __init__(self, config, run, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.config = config
        self.run = run

        assert vars(self.config) == vars(self.run.config)

        num_params = sum(
            p.numel() for p in self.run.model.parameters() if p.requires_grad
        )
        log.info(f"🧠 Model has {num_params:,} trainable parameters")

        self.run.model.to(self.device)
        log.info(f"🖥️  Using device: {self.device}")

        log.info("📊 Loading training and validation data...")
        self.train_data = torch.load(config.train_data).to(self.device)
        self.val_data = torch.load(config.val_data).to(self.device)
        log.info(
            f"    → Train: {self.train_data.numel():,} tokens | Val: {self.val_data.numel():,} tokens"
        )

        self.run_dir = os.path.join(self.config.runs_dir, self.run.name)
        os.makedirs(self.run_dir, exist_ok=True)
        # Save the current config within the run directory
        with open(os.path.join(self.run_dir, "config.json"), "w") as fd:
            json.dump(vars(self.config), fd)
        # Also copy the tokenizer
        shutil.copy(self.config.tokenizer, os.path.join(self.run_dir, "tokenizer.json"))

        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.run_dir)

        self.run.model.eval()
        self.writer.add_graph(
            self.run.model,
            torch.randint(
                0,
                config.vocab_size,
                (1, config.max_position_embeddings),
                device=self.device,
            ),
        )
        self.run.model.train()

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

        # We log everything because the dataset is small
        self.writer.add_scalar("loss/train", loss.item(), self.run.global_step)

    def train(self):
        log.info(
            f"🚀 Starting training for {self.config.max_steps - self.run.global_step:,} steps"
        )
        log.info(
            f"    → Batch size: {self.config.per_device_train_batch_size:,} | Context: {self.config.max_position_embeddings:,} | Initial LR: {self.config.learning_rate:.2e}"
        )
        log.info(f"📋 Full configuration:\n{pprint.pformat(vars(self.config))}")

        try:
            starting_time = time.time()
            train_data_size = self.train_data.numel()

            with tqdm.contrib.logging.logging_redirect_tqdm():
                with tqdm.tqdm(
                    range(self.run.global_step, self.config.max_steps),
                    desc="Training",
                    unit="steps",
                ) as pbar:
                    for _ in pbar:
                        self.step()

                        pbar.set_postfix(
                            epoch=f"{self.run.tokens_seen / train_data_size:.1%}",
                            speed=f"{round(self.run.tokens_seen / (time.time() - starting_time)):,} tokens/s",
                            tokens_seen=f"{self.run.tokens_seen:,}",
                        )

                        if self.run.global_step % self.config.eval_steps == 0:
                            losses = self.evaluate()

                            log.info(
                                f'📈 Step {self.run.global_step:,}: train loss = {losses["train"]:.4f}, val loss = {losses["val"]:.4f}'
                            )
                            self.run.loss_history.append(
                                {
                                    "global_step": self.run.global_step,
                                    "tokens_seen": self.run.tokens_seen,
                                    "epochs": self.run.tokens_seen / train_data_size,
                                    "lr": self.run.optimizer.param_groups[0]["lr"],
                                    "loss": losses,
                                }
                            )

                            if losses["val"] < self.run.best_validation_loss:
                                self.run.best_validation_loss = losses["val"]
                                self._save(f'best-{math.exp(losses["val"]):.2f}')

                            self.writer.add_scalar(
                                "loss/val", losses["val"], self.run.global_step
                            )

                            self.writer.add_scalar(
                                "perplexity/train",
                                math.exp(losses["train"]),
                                self.run.global_step,
                            )

                            self.writer.add_scalar(
                                "perplexity/val",
                                math.exp(losses["val"]),
                                self.run.global_step,
                            )

                            self.writer.add_scalar(
                                "lr",
                                self.run.optimizer.param_groups[0]["lr"],
                                self.run.global_step,
                            )
                            self.writer.flush()

                        if self.run.global_step % self.config.checkpoint_steps == 0:
                            self._save()

        except KeyboardInterrupt:
            log.warning("⚠️  Training interrupted by user")
        except BaseException as e:
            log.error(f"❌ Training failed: {e.__class__.__name__}: {e}")
            raise
        finally:
            self.writer.close()

        self._save("final")
        log.info(f"🎉 Training complete! Final step: {self.run.global_step:,}")

    def _save(self, suffix=None):
        path = os.path.join(
            self.run_dir,
            f"{self.run.name}-{self.run.global_step}{'-' + suffix if suffix else ''}.pt",
        )
        self.run.save(path)
        log.info(f"💾 Checkpoint saved at {path}")

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
    # TODO Use argparse
    # TODO Override config from command line
    # TODO Resume from checkpoint
    # TODO Check config logic
    with open("config.json") as fd:
        config = json.load(fd, object_hook=lambda d: types.SimpleNamespace(**d))

    run = Run(config)

    run_dir = os.path.join(config.runs_dir, run.name)
    os.makedirs(run_dir, exist_ok=True)
    log.basicConfig(
        level=log.INFO,
        format="\033[2m%(asctime)s\033[0m \033[1m\033[36m%(levelname)s\033[0m \033[1m[%(name)s]\033[0m %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            # Write logs into the run directory as well
            log.FileHandler(os.path.join(run_dir, "train.log")),
            log.StreamHandler(),
        ],
        force=True,
    )

    trainer = Trainer(config, run)
    assert run_dir == trainer.run_dir

    log.info(f"📁 Run directory: {trainer.run_dir}")

    # Run a brand-new training round
    trainer.train()
