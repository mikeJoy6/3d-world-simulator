"""The interactive demo with the inference optimisations of the public demo that do not change the picture:

    python -m demo.server_fast --release release

  bf16                     decoder, Gaussian head and dynamics in bfloat16
  sparse finest level      the last decoder level only on the cells near occupied parents (models/sparse_decoder.py),
                           with exact GroupNorm statistics: about 85 % of the decoding time removed
  compiled dynamics        torch.compile on the dynamics step, on the dense parts of the decoder and on the first network
                           of the kinematic arm model

Measured on one GPU: 42 to 49 dB PSNR between this render and the reference server's (demo/server.py); frames per second
in the browser while a key is held: cloth 46 (reference 12), box 42, pusht 58, assembly 57. The first open of a task takes
about a minute longer (compilation). Both servers draw the kinematic robot arms (models/arm_rig.py); the public demo additionally runs TensorRT engines.
"""
from demo.server import main

if __name__ == "__main__":
    main(fast=True)
