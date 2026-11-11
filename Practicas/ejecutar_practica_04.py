"""Ejecuta el notebook completo con el mismo Python que invoca este script."""

from pathlib import Path
from time import perf_counter
import sys

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    base = Path(__file__).resolve().parent
    ruta = base / 'practica_04_metaaprendizaje.ipynb'
    nb = nbformat.read(ruta, as_version=4)
    for celda in nb.cells:
        if celda.cell_type == 'code':
            celda.outputs = []
            celda.execution_count = None
    inicio = perf_counter()

    def empezar(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            print(f'[{perf_counter()-inicio:.0f}s] Celda {cell_index}: '
                  f'{cell.source.splitlines()[0][:100]}', flush=True)

    def guardar(**kwargs):
        nbformat.write(nb, ruta)
        print(f'[{perf_counter()-inicio:.0f}s] Celda terminada; notebook guardado.', flush=True)

    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv[0] = sys.executable
    cliente = NotebookClient(nb, km=km, timeout=None,
        resources={'metadata': {'path': str(base)}}, allow_errors=False,
        on_cell_start=empezar, on_cell_executed=guardar)
    try:
        cliente.execute()
    finally:
        nbformat.write(nb, ruta)
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    print(f'Notebook completo: {perf_counter()-inicio:.1f}s', flush=True)


if __name__ == '__main__':
    main()
