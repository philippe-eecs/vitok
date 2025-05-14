import importlib
from typing import Any, Callable


def from_config(config, predict_fns):
  """Creates a list of evaluators based on `config`."""
  evaluators = []
  specs = config.get("evals", {})

  for name, cfg in specs.items():

    # Pop all generic settings off so we're left with eval's kwargs in the end.
    cfg = cfg.to_dict()
    module = cfg.pop("type", name)
    pred_key = cfg.pop("pred", "predict")
    pred_kw = cfg.pop("pred_kw", None)
    prefix = cfg.pop("prefix", f"{name}/")
    cfg.pop("skip_first", None)
    logsteps = cfg.pop("log_steps")
    for typ in ("steps", "epochs", "examples", "percent"):
      cfg.pop(f"log_{typ}", None)

    # Use same batch_size as eval by default, to reduce fragmentation.
    # TODO: eventually remove all the deprecated names...
    cfg["batch_size"] = config.get("batch_size")  # pylint: disable=line-too-long

    module = importlib.import_module(f"video_compression.evaluators.{module}")

    api_type = getattr(module, "API", "pmap")
    if api_type == "pmap" and "devices" in cfg:
      raise RuntimeError(
          "You are seemingly using the old pmap-based evaluator, but with "
          "jit-based train loop, see (internal link) for more details.")
    if api_type == "jit" and "devices" not in cfg:
      raise RuntimeError(
          "You are seemingly using new jit-based evaluator, but with "
          "old pmap-based train loop, see (internal link) for more details.")

    try:
      predict_fn = predict_fns[pred_key]
    except KeyError as e:
      raise ValueError(
          f"Unknown predict_fn '{pred_key}'. Available predict_fns are:\n"
          + "\n".join(predict_fns)) from e
    evaluator = module.Evaluator(predict_fn, **cfg)
    evaluators.append((name, evaluator, logsteps, prefix))

  return evaluators