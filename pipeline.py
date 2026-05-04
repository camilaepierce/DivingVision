import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import torch
from src.config import DivingConfig
from src.dataloader import create_loaders
from src.training import train_model
from torch import nn


def _load_model_from_subfolder(model_name: str, cfg_obj: DivingConfig = None):
    """Dynamically load a model from models/<model_name>/model.py.

    Loading strategy:
    - If module provides `build_model(cfg)` or `get_model(cfg)`, call it.
    - Else, find the first class that subclasses `nn.Module` and instantiate it.
      Try to pass `(in_channels, num_classes)` if supported, otherwise no args.
    """
    model_dir = Path("models") / model_name
    model_file = model_dir / "model.py"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    spec = importlib.util.spec_from_file_location(f"models.{model_name}.model", str(model_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # prefer explicit factory functions
    if hasattr(mod, "build_model") and callable(mod.build_model):
        return mod.build_model(cfg_obj) if cfg_obj is not None else mod.build_model()
    if hasattr(mod, "get_model") and callable(mod.get_model):
        return mod.get_model(cfg_obj) if cfg_obj is not None else mod.get_model()

    # fallback: find nn.Module subclass
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        try:
            if issubclass(obj, nn.Module) and obj is not nn.Module:
                # attempt to instantiate with (in_channels, num_classes) if signature allows
                sig = inspect.signature(obj.__init__)
                params = list(sig.parameters.keys())[1:]
                kwargs = {}
                if cfg_obj is not None:
                    try:
                        num_classes = cfg_obj.get_labels().get("categories", [])
                        num_classes = len(num_classes) if num_classes else cfg_obj.get_training().get("num_classes", None)
                    except Exception:
                        num_classes = None
                else:
                    num_classes = None

                if "in_channels" in params and "num_classes" in params:
                    args = []
                    # default to 3 channels
                    in_ch = 3
                    nc = int(num_classes) if num_classes else 5
                    try:
                        return obj(in_ch, nc)
                    except Exception:
                        pass
                # try no-arg constructor
                try:
                    return obj()
                except Exception:
                    continue
        except Exception:
            continue

    raise RuntimeError(f"No suitable model class found in {model_file}")


def main(model_name: str = None, config_path: str = None):
    """Pipeline entrypoint.

    Usage: `python pipeline.py ModelName` where `models/ModelName/model.py` exists.
    Optionally provide `-c config.json` to override the model-local config.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # If there is a model-local config at models/<name>/config.json, prefer it.
    cfg_obj = None
    if model_name:
        model_config_path = Path("models") / model_name / "config.json"
        if model_config_path.exists():
            cfg_obj = DivingConfig(filename=str(model_config_path))

    # If no model-local config found, use provided config_path or default
    if cfg_obj is None:
        cfg_obj = DivingConfig(filename=config_path) if config_path else DivingConfig()

    data_cfg = cfg_obj.getDataConfig()
    batch_size = cfg_obj.get_batch_size()
    epochs = cfg_obj.get_epochs()
    save_settings = cfg_obj.get_save_settings()
    history_save_path = os.path.join(save_settings["save_dir"], "training_history.png")

    # create dataloaders
    train_loader, test_loader = create_loaders(data_cfg, batch_size=batch_size)

    # model and training
    if model_name:
        model = _load_model_from_subfolder(model_name, cfg_obj)
    else:
        # fallback to default model in models/model.py
        default_model_path = Path("models") / "model.py"
        if default_model_path.exists():
            # import the shared file
            spec = importlib.util.spec_from_file_location("models.model", str(default_model_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # try to find any nn.Module class
            model = None
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                try:
                    if issubclass(obj, nn.Module) and obj is not nn.Module:
                        model = obj()
                        break
                except Exception:
                    continue
            if model is None:
                raise RuntimeError("No default model found in models/model.py")
        else:
            raise RuntimeError("No model specified and no default model found")

    model = model.to(device)
    model = train_model(
        model,
        train_loader,
        test_loader,
        device,
        num_epochs=epochs,
        visualize_history=True,
        history_save_path=history_save_path,
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Model name matching models/<model_name>/")
    args = parser.parse_args()
    main(args.model_name)