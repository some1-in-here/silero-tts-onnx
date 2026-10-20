"""Russian stress helpers.

Two providers exist for compatibility:

- ``get_accentor()`` / ``apply_stress()`` use the optional pip package
  ``silero-stress`` (legacy prototype path, torch-free).
- ``OnnxStressProvider`` uses the ONNX stress/ngram models through
  ``silero_onnx.homograph_solver.FullStress`` (the runtime path used by
  ``Language_Ru``).

Neither imports torch.
"""
_accentor = None


def get_accentor():
    global _accentor
    if _accentor is None:
        from silero_stress import load_accentor
        _accentor = load_accentor()
    return _accentor


def apply_stress(text: str, **flags) -> str:
    """Stress + yo + homographs via the official silero-stress package."""
    acc = get_accentor()
    kw = dict(put_stress=True, put_stress_homo=True, put_yo=True,
              put_yo_homo=True, stress_single_vowel=True)
    kw.update(flags)
    return acc(text, **kw)


class OnnxStressProvider:
    """Lazy ``FullStress`` wrapper (no model I/O until first call)."""

    def __init__(self, models_dir=None, data_dir=None):
        self._models_dir = models_dir
        self._data_dir = data_dir
        self._stress = None

    def __call__(self, text):
        if self._stress is None:
            from ...homograph_solver import FullStress
            self._stress = FullStress(models_dir=self._models_dir,
                                      data_dir=self._data_dir)
        return self._stress(text)


__all__ = ["get_accentor", "apply_stress", "OnnxStressProvider"]
