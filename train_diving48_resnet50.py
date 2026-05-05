#!/usr/bin/env python3
import argparse
import os
import torch
from torch import save

from diving48_utils import DivingConfig, create_loaders, train_model, _load_model_from_subfolder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="run", help="Experiment run name")
    parser.add_argument("--model-name", default=None, help="Model folder under models/ to load")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-dir", default="results")
    args = parser.parse_args()

    cfg = DivingConfig("config.json")
    # override training params from CLI if provided
    if args.epochs is not None:
        cfg.config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.config.setdefault("training", {})["batch_size"] = args.batch_size
    cfg.config.setdefault("training", {})["save_dir"] = args.save_dir
    cfg.config.setdefault("training", {})["save_model"] = True

    data_cfg = cfg.getDataConfig()
    batch_size = cfg.get_batch_size() if hasattr(cfg, "get_batch_size") else cfg.get_epochs()
    if args.batch_size is not None:
        batch_size = args.batch_size

    train_loader, test_loader = create_loaders(data_cfg, batch_size=batch_size, num_workers=args.num_workers)

    if args.model_name is None:
        args.model_name = cfg.get_model_name() if hasattr(cfg, "get_model_name") else None
    if args.model_name is None:
        raise RuntimeError("Please provide --model-name or set training.model in config.json")

    model = _load_model_from_subfolder(args.model_name, cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model = train_model(
        model,
        train_loader,
        test_loader,
        device=device,
        num_epochs=cfg.get_epochs(),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        visualize_history=True,
        history_save_path=os.path.join(args.save_dir, f"{args.run_name}_history.png"),
    )

    save_path = os.path.join(args.save_dir, f"{args.run_name}.pt")
    os.makedirs(args.save_dir, exist_ok=True)
    try:
        save(model.state_dict(), save_path)
        print(f"Saved model checkpoint to {save_path}")
    except Exception as e:
        print(f"Failed to save model: {e}")


if __name__ == "__main__":
    main()
