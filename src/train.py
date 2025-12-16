import argparse
import datetime
import hashlib
import json
import logging as log
import math
import os
import pprint
import time
import types

import tokenizers
import torch
import torch.utils.tensorboard
import torchinfo
import tqdm
import tqdm.contrib.logging
import transformers

try:
    from . import aux, model
except ImportError:
    import aux
    import model


class Run:
    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.datetime.now(datetime.UTC).isoformat()
        self.model = model.Transformer(config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.scheduler = transformers.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.05 * self.config.max_steps),
            num_training_steps=self.config.max_steps,
        )
        self.global_step = 0
        self.tokens_seen = 0

        self.best_validation_loss = float("inf")
        self.loss_history = []

        self.tokenizer = tokenizers.Tokenizer.from_file(config.tokenizer_path)

        self._info = torchinfo.summary(
            self.model,
            input_size=(
                self.config.per_device_train_batch_size,
                self.config.max_position_embeddings,
            ),
            dtypes=[torch.long],
            mode="eval",
            verbose=0,
        )

        # We use the first 8 characters of the hash of the configuration to give
        # a somewhat unique name to this run
        config_hash = hashlib.sha256(
            bytes(repr(config), encoding="utf-8"), usedforsecurity=False
        ).hexdigest()[:8]
        self.name = (
            f"{self.config.name}-{self._info.trainable_params/1e6:.1f}M-{config_hash}"
        )

    @classmethod
    def from_file(cls, path, device="cpu"):
        states = torch.load(path, map_location=torch.device(device))
        config = aux.config_from_dict(states["config"], validate=False)

        run = cls(config)
        run.tokenizer = tokenizers.Tokenizer.from_str(states["tokenizer"])
        for attr in states:
            if attr in ("config", "tokenizer"):
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
                "tokenizer": self.tokenizer.to_str(),
            },
            path,
        )


class Trainer:
    def __init__(self, run, device):
        self.device = torch.device(device)

        self.run = run
        self.config = self.run.config

        log.info(
            f"🧠 Model has {self.run._info.trainable_params:,} trainable parameters"
        )
        log.info(f"🏗️ Architecture:\n{self.run._info}")

        self.run.model.to(self.device)
        log.info(f"🖥️  Using device: {self.device}")

        log.info("📊 Loading training and validation data...")
        self.train_data = torch.load(self.config.train_data).to(self.device)
        self.val_data = torch.load(self.config.val_data).to(self.device)
        self.train_data_size = self.train_data.numel()
        log.info(
            f"    → Train: {self.train_data_size:,} tokens | Val: {self.val_data.numel():,} tokens"
        )

        self.run_dir = os.path.join(self.config.runs_dir, self.run.name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.run_dir)
        self.writer_closed = False

        self.run.model.eval()
        self.writer.add_graph(
            self.run.model,
            torch.randint(
                0,
                self.config.vocab_size,
                (1, self.config.max_position_embeddings),
                device=self.device,
            ),
        )
        self.run.model.train()

        self._loss = math.log(self.config.vocab_size)

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

        self._loss = loss.item()
        # We log everything because the dataset is small
        self.writer.add_scalar("loss/train", self._loss, self.run.global_step)

    def train(self, jupyter_notebook=False, no_warmup=False):
        log.info(
            f"🚀 Starting training for {self.config.max_steps - self.run.global_step:,} steps"
        )
        log.info(
            f"    → Batch size: {self.config.per_device_train_batch_size:,} | Context: {self.config.max_position_embeddings:,} | Initial LR: {self.config.learning_rate:.2e}"
        )
        log.info(f"📋 Full configuration:\n{pprint.pformat(vars(self.config))}")

        try:
            # Run a "warmup" evaluation if it is only the beginning of the run
            if self.run.global_step == 0 and not no_warmup:
                log.info(
                    f"🎲 Random baseline loss: {math.log(self.config.vocab_size):.2f} (running warmup eval...)"
                )
                self.evaluate()

            starting_time = time.time()

            pbar = (tqdm.notebook.tqdm if jupyter_notebook else tqdm.tqdm)(
                range(self.run.global_step, self.config.max_steps),
                desc="Training",
                unit="steps",
            )

            with tqdm.contrib.logging.logging_redirect_tqdm():
                for _ in pbar:
                    self.step()

                    pbar.set_postfix(
                        epoch=f"{self.run.tokens_seen / self.train_data_size:.1%}",
                        # lr=f'{self.run.optimizer.param_groups[0]["lr"]:.1e}',
                        speed=f"{round(self.run.tokens_seen / (time.time() - starting_time)):,} tokens/s",
                        tokens_seen=f"{self.run.tokens_seen:,}",
                        train_loss=f"{self._loss:.2f}",
                    )

                    if self.run.global_step % self.config.eval_steps == 0:
                        self.evaluate()

                    if self.run.global_step % self.config.checkpoint_steps == 0:
                        self._save()

        except KeyboardInterrupt:
            log.warning("⚠️  Training interrupted by user")
        except BaseException as e:
            log.error(f"❌ Training failed: {e.__class__.__name__}: {e}")
            raise
        finally:
            pbar.close()
            self.writer.close()
            self.writer_closed = True

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

        log.info(
            f'📈 Step {self.run.global_step:,}: train loss = {losses["train"]:.4f}, val loss = {losses["val"]:.4f}'
        )
        self.run.loss_history.append(
            {
                "global_step": self.run.global_step,
                "tokens_seen": self.run.tokens_seen,
                "epochs": self.run.tokens_seen / self.train_data_size,
                "lr": self.run.optimizer.param_groups[0]["lr"],
                "loss": losses,
            }
        )
        # Save a checkpoint if we have the best validation loss
        if losses["val"] < self.run.best_validation_loss:
            self.run.best_validation_loss = losses["val"]
            self._save(f'best-{math.exp(losses["val"]):.2f}')

        if not self.writer_closed:
            self.writer.add_scalar("loss/val", losses["val"], self.run.global_step)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Transformer model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m src.train -c config.json
  python -m src.train -c config.json --set learning_rate=1e-4 --set max_steps=10000
  python -m src.train -c config.json -d runs/my_experiment
""",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="path to the configuration JSON file",
        metavar="PATH",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="override config values with key=value (can be used multiple times)",
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "-d",
        "--run-dir",
        default=None,
        help="override the run directory (default: {runs_dir}/{run_name})",
        metavar="PATH",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="device to run on (default: cuda if available)",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config, load model and data, print summary, then exit",
    )
    # TODO Resume from checkpoint
    args = parser.parse_args()

    try:
        config = aux.load_config(args.config)
        aux.apply_overrides(config, **dict(s.split("=", 1) for s in args.set))
    except aux.ConfigError as e:
        parser.error(str(e))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run = Run(config)

    if not args.dry_run:
        run_dir = args.run_dir or os.path.join(run.config.runs_dir, run.name)
        os.makedirs(run_dir, exist_ok=True)

    handlers = [log.StreamHandler()]
    if not args.dry_run:
        handlers.append(log.FileHandler(os.path.join(run_dir, "train.log")))

    log.basicConfig(
        level=log.INFO,
        format="\033[2m%(asctime)s\033[0m \033[1m\033[36m%(levelname)s\033[0m \033[1m[%(name)s]\033[0m %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    if args.run_dir:
        run.config.runs_dir = os.path.dirname(args.run_dir)
        run.name = os.path.basename(args.run_dir)

    trainer = Trainer(run, device)

    if not args.dry_run:
        log.info(f"📁 Run directory: {trainer.run_dir}")
        trainer.train()
    else:
        log.info("✅ Dry run complete. Config is valid and ready to train.")
