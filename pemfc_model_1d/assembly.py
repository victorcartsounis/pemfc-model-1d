"""Stack the layers into one boundary value problem for scipy's solve_bvp.

Every layer is mapped onto its own coordinate xi in [0, 1], so all layers
share one mesh. The stacked unknown vector holds, layer after layer, each
solved unknown followed by its flux, both divided by their scale. Interface
and outer conditions all become boundary conditions at xi = 0 and xi = 1.
"""

import functools
from dataclasses import dataclass

import numpy as np
import sympy as sp

from pemfc_model_1d import layers as ly
from pemfc_model_1d.parameters import Parameters


@dataclass(frozen=True)
class CompiledLayer:
    rhs: object  # (*y, *args) -> d(y)/d(xi), one entry per row of y
    jacobian: object  # (*y, *args) -> the nonzero entries of d(rhs)/d(y)
    nonzero: list[tuple[int, int]]
    initial_guess: object  # (*args) -> y


@dataclass(frozen=True)
class Block:
    """Where one layer's unknowns sit in the stacked vector."""

    layer: ly.Layer
    start: int

    @property
    def size(self):
        return 2 * len(self.layer.variables)

    def value(self, variable):
        return self.start + 2 * self.layer.variables.index(variable)

    def flux(self, variable):
        return self.value(variable) + 1


def _blocks():
    blocks, start = [], 0
    for layer in ly.LAYERS:
        blocks.append(Block(layer, start))
        start += 2 * len(layer.variables)
    return tuple(blocks)


def _scales(layer):
    scales = []
    for v in layer.variables:
        _, value_scale, flux_scale = ly.VARIABLES[v]
        scales += [value_scale, flux_scale]
    return scales


def _lambdify(args, exprs):
    return sp.lambdify(args, exprs, modules="numpy", cse=True)


class Model:
    """The compiled boundary value problem, valid for any parameter values.

    `args` everywhere is (U, *Parameters.values()).
    """

    def __init__(self):
        p = Parameters.symbols()
        U = sp.Symbol("U")
        args = (U, *p.values())
        self.blocks = _blocks()
        self.size = sum(b.size for b in self.blocks)
        self.scale = np.concatenate([_scales(b.layer) for b in self.blocks])

        self._layers = [self._compile_layer(b.layer, p, U, args) for b in self.blocks]
        self._compile_boundary_conditions(p, U, args)

    def _compile_layer(self, layer, p, U, args):
        guess = ly.uniform_state(p, layer.side, U)
        u = dict(guess)
        y, j, initial = [], {}, []
        for v in layer.variables:
            flux_name, value_scale, flux_scale = ly.VARIABLES[v]
            y += [sp.Symbol(v), sp.Symbol(flux_name)]
            u[v] = value_scale * y[-2]
            j[v] = flux_scale * y[-1]
            initial += [guess[v] / value_scale, 0]

        L = ly.thickness(layer, p)
        rhs = []
        for v, (gradient, source) in ly.equations(layer, p, u, j).items():
            _, value_scale, flux_scale = ly.VARIABLES[v]
            rhs += [L / value_scale * gradient, L / flux_scale * source]

        jacobian = sp.Matrix(rhs).jacobian(y)
        nonzero = [(r, c) for r, c in np.ndindex(jacobian.shape) if jacobian[r, c] != 0]
        return CompiledLayer(
            rhs=_lambdify((*y, *args), rhs),
            jacobian=_lambdify((*y, *args), [jacobian[r, c] for r, c in nonzero]),
            nonzero=nonzero,
            initial_guess=_lambdify(args, initial),
        )

    def _compile_boundary_conditions(self, p, U, args):
        ya = sp.symbols(f"ya:{self.size}")
        yb = sp.symbols(f"yb:{self.size}")
        channel = {"A": ly.channel_state(p, "A", U), "C": ly.channel_state(p, "C", U)}
        residuals = []
        for v, (_, value_scale, _) in ly.VARIABLES.items():
            for run in self._runs(v):
                first, last = run[0], run[-1]
                if first is self.blocks[0]:
                    residuals.append(ya[first.value(v)] - channel["A"][v] / value_scale)
                else:
                    residuals.append(ya[first.flux(v)])
                for left, right in zip(run, run[1:], strict=False):
                    residuals.append(yb[left.value(v)] - ya[right.value(v)])
                    residuals.append(yb[left.flux(v)] - ya[right.flux(v)])
                if last is self.blocks[-1]:
                    residuals.append(yb[last.value(v)] - channel["C"][v] / value_scale)
                else:
                    residuals.append(yb[last.flux(v)])
        assert len(residuals) == self.size
        residuals = sp.Matrix(residuals)
        bc_args = (*ya, *yb, *args)
        self._bc = _lambdify(bc_args, list(residuals))
        self._bc_jacobian = _lambdify(
            bc_args, [residuals.jacobian(ya), residuals.jacobian(yb)]
        )

    def _runs(self, variable):
        """Maximal sequences of adjacent blocks that solve for `variable`."""
        runs, current = [], []
        for block in self.blocks:
            if variable in block.layer.variables:
                current.append(block)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        return runs

    def rhs(self, y, args):
        out = np.empty_like(y)
        for block, layer in zip(self.blocks, self._layers, strict=True):
            rows = slice(block.start, block.start + block.size)
            for k, value in enumerate(layer.rhs(*y[rows], *args)):
                out[block.start + k] = value
        return out

    def rhs_jacobian(self, y, args):
        out = np.zeros((self.size, self.size, y.shape[1]))
        for block, layer in zip(self.blocks, self._layers, strict=True):
            rows = slice(block.start, block.start + block.size)
            values = layer.jacobian(*y[rows], *args)
            for (r, c), value in zip(layer.nonzero, values, strict=True):
                out[block.start + r, block.start + c] = value
        return out

    def bc(self, ya, yb, args):
        return np.array(self._bc(*ya, *yb, *args), dtype=float)

    def bc_jacobian(self, ya, yb, args):
        dya, dyb = self._bc_jacobian(*ya, *yb, *args)
        return np.asarray(dya, dtype=float), np.asarray(dyb, dtype=float)

    def initial_guess(self, args):
        """Flat profiles with zero fluxes, as a stacked (scaled) vector."""
        return np.concatenate([layer.initial_guess(*args) for layer in self._layers])


@functools.cache
def build():
    """The compiled model; built once per process."""
    return Model()
