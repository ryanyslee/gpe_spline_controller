# trt_runner.py
from pathlib import Path
import numpy as np
import tensorrt as trt
import cupy as cp

class TrtRunner:
    """Static-shape runner for (1,80,8)->(1,2), supports TRT 10.x tensor API and TRT 8.x bindings API."""

    def __init__(self, engine_path: str | Path, input_name: str | None = None, output_name: str | None = None):
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(engine_path)

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # 1 for static & 2 for batch-2
        self.target_shape = (1, 80, 8)
        # self.target_shape = (2, 80, 8)

        # --- Decide API ERA based on ENGINE capabilities (more reliable on TRT10)
        has_tensor_api = any([
            hasattr(self.engine, "get_tensor_names"),
            hasattr(self.engine, "num_io_tensors"),
        ])
        has_bindings_api = all([
            hasattr(self.engine, "num_bindings"),
            hasattr(self.engine, "get_binding_name"),
            hasattr(self.engine, "binding_is_input"),
        ])

        if has_tensor_api:
            self.api = "tensor"   # TRT 10.x+
        elif has_bindings_api:
            self.api = "bindings" # TRT 8.x/7.x
        else:
            raise RuntimeError("Cannot detect a supported TensorRT API on this engine.")

        print(f"[TrtRunner] TRT {trt.__version__} | API: {self.api}")

        # --- I/O discovery
        self.input_names, self.output_names = [], []
        if self.api == "tensor":
            names = self.engine.get_tensor_names() if hasattr(self.engine, "get_tensor_names") else [
                self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
            ]
            for name in names:
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_names.append(name)
                elif mode == trt.TensorIOMode.OUTPUT:
                    self.output_names.append(name)
        else:
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    self.input_names.append(self.engine.get_binding_name(i))
                else:
                    self.output_names.append(self.engine.get_binding_name(i))

        # Allow manual override
        if input_name:  self.input_names  = [input_name]
        if output_name: self.output_names = [output_name]
        if len(self.input_names) != 1 or len(self.output_names) != 1:
            raise RuntimeError(f"Expected 1 input & 1 output, got IN={self.input_names}, OUT={self.output_names}")

        self.in_name  = self.input_names[0]
        self.out_name = self.output_names[0]
        print(f"[TrtRunner] IN: {self.in_name}  OUT: {self.out_name}")

        # --- Allocate & bind
        self.stream = cp.cuda.Stream(non_blocking=True)

        if self.api == "tensor":
            # Ensure concrete shapes
            try:
                self.context.set_input_shape(self.in_name, self.target_shape)
            except Exception:
                pass

            in_shape = tuple(self.context.get_tensor_shape(self.in_name))
            if -1 in in_shape:
                in_shape = self.target_shape
                self.context.set_input_shape(self.in_name, in_shape)

            out_shape = tuple(self.context.get_tensor_shape(self.out_name))
            if -1 in out_shape:
                out_shape = tuple(self.context.get_tensor_shape(self.out_name))

            self.d_in  = cp.empty(in_shape,  dtype=cp.float32)
            self.d_out = cp.empty(out_shape, dtype=cp.float32)

            # Bind by tensor name
            self.context.set_tensor_address(self.in_name,  int(self.d_in.data.ptr))
            self.context.set_tensor_address(self.out_name, int(self.d_out.data.ptr))

            # Use enqueue_v3 if available, else fallback to execute_v2 if present (rare)
            self._runner = ("enqueue_v3" if hasattr(self.context, "enqueue_v3") else
                            "execute_v2" if hasattr(self.context, "execute_v2") else None)
            if self._runner is None:
                raise RuntimeError("No supported enqueue/execute method found for tensor API context.")

            print(f"[TrtRunner] Runner: {self._runner}")

        else:
            # Legacy bindings path
            self.in_idx  = self.engine.get_binding_index(self.in_name)
            self.out_idx = self.engine.get_binding_index(self.out_name)

            in_shape = self.engine.get_binding_shape(self.in_idx)
            if any(d == -1 for d in in_shape):
                self.context.set_binding_shape(self.in_idx, self.target_shape)
                in_shape = self.context.get_binding_shape(self.in_idx)
            out_shape = self.context.get_binding_shape(self.out_idx)

            self.d_in  = cp.empty(in_shape,  dtype=cp.float32)
            self.d_out = cp.empty(out_shape, dtype=cp.float32)

            self.bindings = [None] * self.engine.num_bindings
            self.bindings[self.in_idx]  = int(self.d_in.data.ptr)
            self.bindings[self.out_idx] = int(self.d_out.data.ptr)

    def infer(self, window: np.ndarray) -> np.ndarray:
        # Accept (80,8) or (1,80,8)
        if window.dtype != np.float32:
            window = window.astype(np.float32, copy=False)
        if window.ndim == 2:
            window = window[None, ...]
        if window.shape != self.target_shape:
            raise ValueError(f"expected {self.target_shape}, got {window.shape}")

        with self.stream:
            self.d_in.set(window)  # H2D
            if self.api == "tensor":
                if self._runner == "enqueue_v3":
                    if not self.context.enqueue_v3(self.stream.ptr):
                        raise RuntimeError("enqueue_v3 failed")
                else:
                    if not self.context.execute_v2([]):
                        raise RuntimeError("execute_v2 (tensor API fallback) failed")
            else:
                if not self.context.execute_v2(self.bindings):
                    raise RuntimeError("execute_v2 (bindings) failed")
            out = cp.asnumpy(self.d_out)  # D2H
        return out
