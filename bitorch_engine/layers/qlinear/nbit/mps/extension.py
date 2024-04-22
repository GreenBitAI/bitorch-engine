from pathlib import Path

from bitorch_engine.utils.mlx_extension import get_mlx_extension

MLX_REQUIRED = True

def get_ext(path: Path):
    return get_mlx_extension(
        path,
        relative_name='mpq_linear_mlx',
        relative_sources=[
            'mlx_bindings.cpp',
            'mpq_linear_mlx.cpp',
        ]
    )