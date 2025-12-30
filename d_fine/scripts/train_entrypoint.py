from __future__ import annotations

import subprocess
import sys
import torch
import click

def get_n_gpus(requested: int | None) -> int:
  return requested if requested is not None else torch.cuda.device_count()

def build_cmd(n: int) -> list[str]:
  return [
    "torchrun",
    "--nproc_per_node", str(n),
    "--master_port", "29500",
    "-m", "d_fine.scripts.train",
  ] if n > 1 else [sys.executable, "-m", "d_fine.scripts.train"]

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option("--num-gpus", type=int, help="Number of GPUs for DDP. Defaults to auto-detection.")
@click.argument("rest", nargs=-1, type=click.UNPROCESSED)
def main(num_gpus: int | None, rest: tuple[str, ...]) -> None:
  """Principled training entrypoint with automatic DDP selection."""
  n = get_n_gpus(num_gpus)
  subprocess.run(build_cmd(n) + list(rest), check=True)

if __name__ == "__main__":
  main()
