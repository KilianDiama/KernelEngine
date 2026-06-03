import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Final

import numpy as np

T = TypeVar("T")


# =========================
# Core Data Structures
# =========================

@dataclass(frozen=True, slots=True)
class BrickOutput(Generic[T]):
    """
    Immutable container for brick outputs.

    Attributes
    ----------
    content:
        Payload produced by the brick (arbitrary type).
    reliability:
        Self-reported reliability in [0, 1].
    timestamp:
        UTC timestamp of production.
    metadata:
        Auxiliary structured information (e.g., diagnostics, brick internals).
    """
    content: T
    reliability: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class TechBrick(ABC):
    """
    Abstract base class for a reasoning / tool brick.

    Each brick exposes:
      - name: human-readable identifier
      - embedding: latent vector on the unit hypersphere (shape: (D,))
      - run: async execution returning a BrickOutput[T]
    """

    def __init__(self, name: str, embedding: np.ndarray) -> None:
        self.name: str = name

        embedding = np.asarray(embedding, dtype=float)
        if embedding.ndim != 1:
            raise ValueError(f"Embedding for brick '{name}' must be 1D, got shape {embedding.shape}.")

        norm = float(np.linalg.norm(embedding))
        if norm < 1e-12:
            raise ValueError(f"Embedding for brick '{name}' has near-zero norm.")

        # Unit hypersphere projection
        self.embedding: np.ndarray = embedding / norm

    @abstractmethod
    async def run(self, input_data: Any) -> BrickOutput[Any]:
        """
        Execute the brick on the given input_data.

        Implementations should:
          - Be side-effect aware.
          - Return a BrickOutput with reliability in [0, 1].
        """
        ...


# =========================
# Kernel Engine
# =========================

class KernelEngine:
    """
    Latent reasoning kernel with:
      - Spectral encoding of tasks
      - GRU-like gated state update
      - Scaled dot-product attention over bricks
      - Bayesian refinement of reliability
      - Depth-limited recursive reasoning

    The kernel is concurrency-safe for its internal state and uses a thread pool
    for CPU-bound operations.
    """

    ENTROPY_FLOOR: Final[float] = 0.42
    LATENT_DIM: Final[int] = 128
    MAX_DEPTH: Final[int] = 5
    TIMEOUT_SECONDS: Final[float] = 15.0
    MAX_WORKERS: Final[int] = 12

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        max_workers: int = MAX_WORKERS,
        entropy_floor: float = ENTROPY_FLOOR,
        max_depth: int = MAX_DEPTH,
        timeout_seconds: float = TIMEOUT_SECONDS,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive.")
        if not (0.0 <= entropy_floor <= 1.0):
            raise ValueError("entropy_floor must be in [0, 1].")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive.")

        self.dim: int = latent_dim
        self.state_memory: np.ndarray = np.zeros(self.dim, dtype=float)

        self.entropy_floor: float = float(entropy_floor)
        self.max_depth: int = int(max_depth)
        self.timeout_seconds: float = float(timeout_seconds)

        # Executor for CPU-bound operations
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()  # Concurrency safety for state_memory

        # Optional logger (e.g., logging.info)
        self._log: Callable[[str], None] = logger or (lambda msg: None)

        # SVD Orthogonal Initialization for the Spectral Mask
        rnd = np.random.default_rng()
        random_matrix = rnd.standard_normal((self.dim, self.dim))
        u, _, vh = np.linalg.svd(random_matrix, full_matrices=False)
        self._spectral_mask: np.ndarray = u @ vh

        # He-Initialization for Gated Logic
        gain = np.sqrt(2.0 / self.dim)
        self._gate_w: np.ndarray = rnd.standard_normal((self.dim, self.dim)) * gain
        self._context_proj: np.ndarray = rnd.standard_normal((self.dim, self.dim)) * gain
        self._residual_alpha: float = 0.15

    # =========================
    # Low-level utilities
    # =========================

    @staticmethod
    def _layer_norm(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Simple layer normalization over a 1D vector.
        """
        if vec.ndim != 1:
            raise ValueError(f"_layer_norm expects 1D vector, got shape {vec.shape}.")
        mean = float(np.mean(vec))
        std = float(np.std(vec))
        return (vec - mean) / (std + eps)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax over a 1D vector.
        """
        if x.ndim != 1:
            raise ValueError(f"_softmax expects 1D vector, got shape {x.shape}.")
        x_shifted = x - np.max(x)
        e_x = np.exp(x_shifted)
        denom = float(e_x.sum())
        if denom <= 0.0 or not np.isfinite(denom):
            # Fallback to uniform if something goes numerically wrong
            return np.ones_like(e_x) / e_x.size
        return e_x / denom

    def _attention_score(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """
        Scaled dot-product attention over brick embeddings.

        Parameters
        ----------
        query:
            Shape (D,)
        keys:
            Shape (N, D)

        Returns
        -------
        probs:
            Shape (N,), attention probabilities summing to 1.
        """
        if query.ndim != 1:
            raise ValueError(f"query must be 1D, got shape {query.shape}.")
        if keys.ndim != 2 or keys.shape[1] != query.shape[0]:
            raise ValueError(
                f"keys must be (N, D) with D={query.shape[0]}, got {keys.shape}."
            )

        d_k = float(query.shape[0])
        scores = (keys @ query) / np.sqrt(d_k)
        return self._softmax(scores)

    @staticmethod
    def bayesian_refinement(prior: float, likelihood: float, depth: int) -> float:
        """
        Depth-aware refinement of reliability.

        As depth increases, the Kalman-like gain decreases → more stability.

        Parameters
        ----------
        prior:
            Prior confidence in [0, 1].
        likelihood:
            Observed reliability in [0, 1].
        depth:
            Current recursion depth (>= 0).

        Returns
        -------
        updated:
            Refined confidence in [0, 1].
        """
        prior = float(np.clip(prior, 0.0, 1.0))
        likelihood = float(np.clip(likelihood, 0.0, 1.0))
        depth = max(int(depth), 0)

        k_gain = 0.8 / (1.0 + depth * 0.2)
        updated = prior + k_gain * (likelihood - prior)
        return float(np.clip(updated, 0.0, 1.0))

    def _encode_spectral(self, task: str) -> np.ndarray:
        """
        Spectral encoding of the task string into latent space.

        Uses a deterministic RNG seeded from the first characters of the task.
        """
        if not task:
            return np.zeros(self.dim, dtype=float)

        seed = 0
        for i, c in enumerate(task[:10]):
            seed += (ord(c) & 0xFF) << (i * 3)

        rng = np.random.default_rng(seed)
        signal = rng.standard_normal(self.dim)
        projected = self._spectral_mask @ signal
        return self._layer_norm(np.tanh(projected))

    # =========================
    # Brick selection
    # =========================

    async def select_brick(self, intent_vec: np.ndarray, bricks: List[TechBrick]) -> TechBrick:
        """
        Select the most relevant brick via attention over embeddings.
        """
        if not bricks:
            raise ValueError("Brick registry is empty.")

        keys = np.stack([b.embedding for b in bricks], axis=0)
        loop = asyncio.get_running_loop()
        probs = await loop.run_in_executor(
            self._executor,
            self._attention_score,
            intent_vec.astype(float),
            keys.astype(float),
        )
        idx = int(np.argmax(probs))
        selected = bricks[idx]
        self._log(f"[KernelEngine] Selected brick='{selected.name}' with p={probs[idx]:.3f}")
        return selected

    # =========================
    # Recursive reasoning
    # =========================

    async def _update_state(self, current_intent: np.ndarray) -> np.ndarray:
        """
        GRU-like gated update of the internal state.

        Protected by a lock at the call site.
        """
        if current_intent.shape != (self.dim,):
            raise ValueError(
                f"current_intent must have shape ({self.dim},), got {current_intent.shape}."
            )

        # z_gate = sigmoid(W * h_{t-1})
        pre_act = np.dot(self._gate_w, self.state_memory)
        z_gate = 1.0 / (1.0 + np.exp(-pre_act))

        candidate = np.tanh(self._context_proj @ current_intent)

        new_state = (
            (1.0 - z_gate) * self.state_memory
            + z_gate * candidate
            + self._residual_alpha * self.state_memory
        )
        self.state_memory = self._layer_norm(new_state)
        return self.state_memory

    async def recursive_reasoning(
        self,
        task: str,
        bricks: List[TechBrick],
        depth: int = 0,
    ) -> BrickOutput[Any]:
        """
        Main reasoning loop:
          - Encode task
          - Update recurrent state
          - Select brick via attention
          - Execute brick with timeout
          - Refine reliability; recurse if below entropy floor
        """
        if depth >= self.max_depth:
            self._log(
                f"[KernelEngine] Max depth reached (depth={depth}). Returning stability protocol message."
            )
            return BrickOutput(
                content="Stability Protocol: Reasoning depth limit reached.",
                reliability=0.4,
                metadata={
                    "depth": depth,
                    "status": "max_depth_reached",
                    "kernel_entropy_floor": self.entropy_floor,
                },
            )

        # Encode + update state under lock
        current_intent = self._encode_spectral(task)
        async with self._lock:
            intent_vec = await self._update_state(current_intent)

        # Select brick outside the lock
        brick = await self.select_brick(intent_vec, bricks)

        try:
            result = await asyncio.wait_for(
                brick.run(task),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            self._log(
                f"[KernelEngine] Timeout in brick='{brick.name}' at depth={depth} "
                f"(timeout={self.timeout_seconds}s)."
            )
            return BrickOutput(
                content="Kernel Fault: Brick execution timed out.",
                reliability=0.0,
                metadata={
                    "brick": brick.name,
                    "error": "timeout",
                    "depth": depth,
                    "timeout_seconds": self.timeout_seconds,
                },
            )
        except Exception as e:
            self._log(
                f"[KernelEngine] Exception in brick='{brick.name}' at depth={depth}: {type(e).__name__}: {e}"
            )
            return BrickOutput(
                content="Kernel Fault: Brick raised an exception.",
                reliability=0.0,
                metadata={
                    "brick": brick.name,
                    "error": "exception",
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "depth": depth,
                },
            )

        # Bayesian confidence refinement
        prior_conf = 0.9 if depth == 0 else 0.5
        current_conf = self.bayesian_refinement(prior_conf, result.reliability, depth)

        self._log(
            f"[KernelEngine] depth={depth}, brick='{brick.name}', "
            f"raw_rel={result.reliability:.3f}, refined_conf={current_conf:.3f}"
        )

        if current_conf < self.entropy_floor:
            refined_task = (
                f"Sub-optimal result detected (conf={current_conf:.2f}). "
                f"Refine based on: {result.content}"
            )
            return await self.recursive_reasoning(
                refined_task,
                bricks,
                depth=depth + 1,
            )

        # Attach refined confidence in metadata, but keep original reliability intact
        enriched_metadata = dict(result.metadata)
        enriched_metadata.update(
            {
                "depth": depth,
                "refined_confidence": current_conf,
                "brick": getattr(brick, "name", None),
                "kernel_entropy_floor": self.entropy_floor,
            }
        )
        return BrickOutput(
            content=result.content,
            reliability=result.reliability,
            timestamp=result.timestamp,
            metadata=enriched_metadata,
        )

    # =========================
    # Lifecycle
    # =========================

    def shutdown(self) -> None:
        """
        Cleanly shutdown the underlying executor.
        """
        self._log("[KernelEngine] Shutting down executor.")
        self._executor.shutdown(wait=True)


# =========================
# Bootstrapping
# =========================

def reboot_system(logger: Optional[Callable[[str], None]] = None) -> KernelEngine:
    """
    Factory for a fresh KernelEngine instance with a small boot log.
    """
    log = logger or (lambda msg: print(msg, flush=True))

    header = " KERNEL SYNCHRONIZATION 10/10 "
    log(f"\n{header:=^60}")
    log("STATUS: ZENITH-CLASS / TRANSFORMER ATTENTION ACTIVE")
    log("CORE: CONCURRENCY SAFE | SCALED DOT-PRODUCT | GRU-GATING")
    log(f"TS: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'':=^60}\n")

    return KernelEngine(logger=logger)


