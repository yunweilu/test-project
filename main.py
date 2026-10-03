import json
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

use_gpu = True  # True on the cluster, False on your laptop

matrix_size = 10000
dtype = jnp.float32

if use_gpu:
    device = jax.devices("cuda")[0]
    output_root = Path("/net/nfs-iq/data/yunwei/gpu-speed-test")
else:
    device = jax.devices("cpu")[0]
    output_root = Path("data/gpu-speed-test")

# Create a folder with date/time of the run
run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir.mkdir(parents=True, exist_ok=True)

print(f"Using device: {device}", flush=True)
print(f"Saving results to: {run_dir}", flush=True)


@jax.jit
def accelerated_step(x):
    return x @ x


x = jax.device_put(
    jnp.ones((matrix_size, matrix_size), dtype=dtype),
    device,
)

# Warmup: compiles the function for this device
# Not necessary, but useful here since we're comparing runtimes
accelerated_step(x).block_until_ready()

# Actual run
t0 = time.perf_counter()
y = accelerated_step(x).block_until_ready()
dt = time.perf_counter() - t0

# Save example results
sample_results = np.asarray(y[:4, :4])

summary = {
    "use_gpu": use_gpu,
    "device": str(device),
    "result_devices": [str(d) for d in y.devices()],
    "matrix_size": matrix_size,
    "dtype": str(dtype),
    "elapsed_seconds": dt,
    "sample_00": float(sample_results[0, 0]),
}

(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
np.savez(run_dir / "result_sample.npz", sample=sample_results)

print("Result device:", y.devices())
print(f"Time: {dt:.3f} s")
