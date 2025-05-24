import numpy as np
from typing import List, Literal, Optional
from loguru import logger


class InterpolationEngine:
    """Handles various methods for interpolating between embedding vectors."""

    @staticmethod
    def linear_interpolation(
        emb1: np.ndarray, emb2: np.ndarray, steps: int = 10
    ) -> np.ndarray:
        """
        Performs linear interpolation (LERP) between two embedding vectors.

        Args:
            emb1: The starting embedding vector (1D numpy array).
            emb2: The ending embedding vector (1D numpy array).
            steps: The number of interpolation steps (including start and end points).

        Returns:
            A 2D numpy array where each row is an interpolated embedding vector.
            Shape will be (steps, embedding_dimension).
        """
        if emb1.shape != emb2.shape:
            raise ValueError("Start and end embeddings must have the same shape.")
        if steps < 2:
            raise ValueError("Number of steps must be at least 2 (start and end).")

        alphas = np.linspace(0, 1, steps)
        interpolated_vectors = []
        for alpha in alphas:
            interpolated_vector = (1 - alpha) * emb1 + alpha * emb2
            interpolated_vectors.append(interpolated_vector)
        return np.array(interpolated_vectors)

    @staticmethod
    def slerp(emb1: np.ndarray, emb2: np.ndarray, steps: int = 10) -> np.ndarray:
        """
        Performs Spherical Linear Interpolation (SLERP) between two embedding vectors.
        SLERP is useful for interpolating on a hypersphere, maintaining magnitude if vectors are normalized.
        This implementation also interpolates magnitudes linearly if they differ.

        Args:
            emb1: The starting embedding vector (1D numpy array).
            emb2: The ending embedding vector (1D numpy array).
            steps: The number of interpolation steps (including start and end points).

        Returns:
            A 2D numpy array where each row is an interpolated embedding vector.
        """
        if emb1.shape != emb2.shape:
            raise ValueError("Start and end embeddings must have the same shape.")
        if steps < 2:
            raise ValueError("Number of steps must be at least 2.")

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            logger.warning("SLERP called with zero-norm vector. Falling back to LERP.")
            return InterpolationEngine.linear_interpolation(emb1, emb2, steps)

        e1_normalized = emb1 / norm1
        e2_normalized = emb2 / norm2

        dot_product = np.dot(e1_normalized, e2_normalized)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        omega = np.arccos(dot_product)
        sin_omega = np.sin(omega)

        alphas = np.linspace(0, 1, steps)
        interpolated_vectors = []

        for t in alphas:
            current_norm = (1 - t) * norm1 + t * norm2

            if np.abs(sin_omega) < 1e-6:
                if t == 0:
                    interp_normalized = e1_normalized.copy()
                elif t == 1:
                    interp_normalized = e2_normalized.copy()
                else:
                    interp_normalized = (1 - t) * e1_normalized + t * e2_normalized
                    norm_interp = np.linalg.norm(interp_normalized)
                    if norm_interp > 1e-6:
                        interp_normalized = interp_normalized / norm_interp
                    else:
                        pass
            else:
                term1 = np.sin((1 - t) * omega) / sin_omega
                term2 = np.sin(t * omega) / sin_omega
                interp_normalized = term1 * e1_normalized + term2 * e2_normalized

            interpolated_vectors.append(interp_normalized * current_norm)

        return np.array(interpolated_vectors)

    # Bezier curve will be added later as per the plan
    # def bezier_curve(self, start_emb: np.ndarray, end_emb: np.ndarray,
    #                  control_point_embs: Optional[List[np.ndarray]] = None,
    #                  steps: int = 50) -> np.ndarray:
    #     pass
