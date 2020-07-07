# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: file has been modified.

"""JAX-based Dormand-Prince ODE integration with adaptive stepsize.

Integrate systems of ordinary differential equations (ODEs) using the JAX
autograd/diff library and the Dormand-Prince method for adaptive integration
stepsize calculation. Provides improved integration accuracy over fixed
stepsize integration methods.

Adjoint algorithm based on Appendix C of https://arxiv.org/pdf/1806.07366.pdf
"""


from functools import partial
import operator as op

import jax
import jax.numpy as np
from jax import lax
from jax import ops
from jax.util import safe_map, safe_zip
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from jax import linear_util as lu

map = safe_map
zip = safe_zip

_ADAMS_MAX_ORDER = 12


def ravel_first_arg(f, unravel):
  return ravel_first_arg_(lu.wrap_init(f), unravel).call_wrapped

@lu.transformation
def ravel_first_arg_(unravel, y_flat, *args):
  y = unravel(y_flat)
  ans = yield (y,) + args, {}
  ans_flat, _ = ravel_pytree(ans)
  yield ans_flat

def interp_fit_dopri(y0, y1, k, dt):
  # Fit a polynomial to the results of a Runge-Kutta step.
  dps_c_mid = np.array([
      6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2,
      -2691868925 / 45128329728 / 2, 187940372067 / 1594534317056 / 2,
      -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2])
  y_mid = y0 + dt * np.dot(dps_c_mid, k)
  return np.array(fit_4th_order_polynomial(y0, y1, y_mid, k[0], k[-1], dt))

def interp_fit_bosh(y0, y1, k, dt):
    """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
    bs_c_mid = np.array([0., 0.5, 0., 0.])
    y_mid = y0 + dt * np.dot(bs_c_mid, k)
    return np.array(fit_4th_order_polynomial(y0, y1, y_mid, k[0], k[-1], dt))

def interp_fit_heun(y0, y1, k, dt):
    """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
    # from torchdiffeq
    bs_c_mid = np.array([0.5, 0])
    y_mid = y0 + dt * np.dot(bs_c_mid, k)
    return np.array(fit_4th_order_polynomial(y0, y1, y_mid, k[0], k[-1], dt))

def fit_4th_order_polynomial(y0, y1, y_mid, dy0, dy1, dt):
  a = -2.*dt*dy0 + 2.*dt*dy1 -  8.*y0 -  8.*y1 + 16.*y_mid
  b =  5.*dt*dy0 - 3.*dt*dy1 + 18.*y0 + 14.*y1 - 32.*y_mid
  c = -4.*dt*dy0 +    dt*dy1 - 11.*y0 -  5.*y1 + 16.*y_mid
  d = dt * dy0
  e = y0
  return a, b, c, d, e

def initial_step_size(fun, t0, y0, order, rtol, atol, f0):
  # Algorithm from:
  # E. Hairer, S. P. Norsett G. Wanner,
  # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
  scale = atol + np.abs(y0) * rtol
  d0 = np.linalg.norm(y0 / scale)
  d1 = np.linalg.norm(f0 / scale)

  h0 = np.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)

  y1 = y0 + h0 * f0
  f1 = fun(y1, t0 + h0)
  d2 = np.linalg.norm((f1 - f0) / scale) / h0

  h1 = np.where((d1 <= 1e-15) & (d2 <= 1e-15),
                np.maximum(1e-6, h0 * 1e-3),
                (0.01 / np.max(d1 + d2)) ** (1. / (order + 1.)))

  return np.minimum(100. * h0, h1)

def runge_kutta_step(func, y0, f0, t0, dt):
  # Dopri5 Butcher tableaux
  alpha = np.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1., 0])
  beta = np.array([
      [1 / 5, 0, 0, 0, 0, 0, 0],
      [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
      [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
      [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
  ])
  c_sol = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
  c_error = np.array([35 / 384 - 1951 / 21600, 0, 500 / 1113 - 22642 / 50085,
                      125 / 192 - 451 / 720, -2187 / 6784 - -12231 / 42400,
                      11 / 84 - 649 / 6300, -1. / 60.])

  def body_fun(i, k):
    ti = t0 + dt * alpha[i-1]
    yi = y0 + dt * np.dot(beta[i-1, :], k)
    ft = func(yi, ti)
    return ops.index_update(k, jax.ops.index[i, :], ft)

  k = ops.index_update(np.zeros((7, f0.shape[0])), ops.index[0, :], f0)
  k = lax.fori_loop(1, 7, body_fun, k)

  y1 = dt * np.dot(c_sol, k) + y0
  y1_error = dt * np.dot(c_error, k)
  f1 = k[-1]
  return y1, f1, y1_error, k

def bosh_step(func, y0, f0, t0, dt):
  # Bosh tableau
  alpha = np.array([1/2, 3/4, 1., 0])
  beta = np.array([
    [1/2, 0,   0,   0],
    [0.,  3/4, 0,   0],
    [2/9, 1/3, 4/9, 0]
    ])
  c_sol = np.array([2/9, 1/3, 4/9, 0.])
  c_error = np.array([2/9-7/24, 1/3-1/4, 4/9-1/3, -1/8])

  def body_fun(i, k):
    ti = t0 + dt * alpha[i-1]
    yi = y0 + dt * np.dot(beta[i-1, :], k)
    ft = func(yi, ti)
    return ops.index_update(k, jax.ops.index[i, :], ft)

  k = ops.index_update(np.zeros((4, f0.shape[0])), ops.index[0, :], f0)
  k = lax.fori_loop(1, 4, body_fun, k)

  y1 = dt * np.dot(c_sol, k) + y0
  y1_error = dt * np.dot(c_error, k)
  f1 = k[-1]
  return y1, f1, y1_error, k

def heun_step(func, y0, f0, t0, dt):
  # Heun tableau
  alpha = np.array([1., 0])
  beta = np.array([
    [1/2, 0]
    ])
  c_sol = np.array([1/2, 1/2])
  c_error = np.array([1/2 - 1, 1/2])

  def body_fun(i, k):
    ti = t0 + dt * alpha[i-1]
    yi = y0 + dt * np.dot(beta[i-1, :], k)
    ft = func(yi, ti)
    return ops.index_update(k, jax.ops.index[i, :], ft)

  k = ops.index_update(np.zeros((2, f0.shape[0])), ops.index[0, :], f0)
  k = lax.fori_loop(1, 2, body_fun, k)

  y1 = dt * np.dot(c_sol, k) + y0
  y1_error = dt * np.dot(c_error, k)
  f1 = k[-1]
  return y1, f1, y1_error, k

def fehlberg_step(func, y0, f0, t0, dt):
  # Fehlberg tableau
  alpha = np.array([1/2, 1, 0])
  beta = np.array([
    [1/2, 0, 0],
    [1/256, 255/256, 0]
    ])
  c_sol = np.array([1/512, 255/256, 1/512])
  c_error = np.array([1/512 - 1/256, 0., 1/512])

  def body_fun(i, k):
    ti = t0 + dt * alpha[i-1]
    yi = y0 + dt * np.dot(beta[i-1, :], k)
    ft = func(yi, ti)
    return ops.index_update(k, jax.ops.index[i, :], ft)

  k = ops.index_update(np.zeros((3, f0.shape[0])), ops.index[0, :], f0)
  k = lax.fori_loop(1, 3, body_fun, k)

  y1 = dt * np.dot(c_sol, k) + y0
  y1_error = dt * np.dot(c_error, k)
  f1 = k[-1]
  return y1, f1, y1_error, k

def rk_fehlberg_step(func, y0, f0, t0, dt):
  # Fehlberg tableau
  alpha = np.array([1/4, 3/8, 12/13, 1, 1/2, 0])
  beta = np.array([
    [1/4, 0, 0, 0, 0, 0],
    [3/32, 9/32, 0, 0, 0, 0],
    [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
    [439/216, -8, 3680/513, -845/4104, 0, 0],
    [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
    ])
  c_sol = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
  c_error = np.array([16/135 - 25/216, 0, 6656/12825 - 1408/2565, 28561/56430 - 2197/4104, -9/50 - - 1/5, 2/55])

  def body_fun(i, k):
    ti = t0 + dt * alpha[i-1]
    yi = y0 + dt * np.dot(beta[i-1, :], k)
    ft = func(yi, ti)
    return ops.index_update(k, jax.ops.index[i, :], ft)

  k = ops.index_update(np.zeros((6, f0.shape[0])), ops.index[0, :], f0)
  k = lax.fori_loop(1, 6, body_fun, k)

  y1 = dt * np.dot(c_sol, k) + y0
  y1_error = dt * np.dot(c_error, k)
  f1 = k[-1]
  return y1, f1, y1_error, k

def cash_karp_step(func, y0, f0, t0, dt):
  # Cash-Karp tableau
  alpha = np.array([1/5, 3/10, 3/5, 1, 7/8, 0])
  beta = np.array([
    [1/5, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0],
    [3/10, -9/10, 6/5, 0, 0, 0],
    [-11/54, 5/2, -70/27, 35/27, 0, 0],
    [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096, 0]
    ])
  c_sol = np.array([37/378, 0, 250/621, 125/594, 0, 512/1771])
  c_error = np.array([37/378 - 2825/27648, 0, 250/621 - 18575/48384, 125/594 - 13525/55296, -277/14336, 512/1771 - 1/4])

  def body_fun(i, k):
    ti = t0 + dt * alpha[i-1]
    yi = y0 + dt * np.dot(beta[i-1, :], k)
    ft = func(yi, ti)
    return ops.index_update(k, jax.ops.index[i, :], ft)

  k = ops.index_update(np.zeros((6, f0.shape[0])), ops.index[0, :], f0)
  k = lax.fori_loop(1, 6, body_fun, k)

  y1 = dt * np.dot(c_sol, k) + y0
  y1_error = dt * np.dot(c_error, k)
  f1 = k[-1]
  return y1, f1, y1_error, k

def owrenzen_step(func, y0, f0, t0, dt):
  # Owrenzen tableau
  alpha = np.array([1/6, 11/37, 11/17, 13/15, 1, 0])
  beta = np.array([
    [1/6, 0, 0, 0, 0, 0],
    [44/1369, 363/1369, 0, 0, 0, 0],
    [3388/4913, -8349/4913, 8140/4913, 0, 0, 0],
    [-36764/408375, 767/1125, -32708/136125, 210392/408375, 0, 0],
    [1697/18876, 0, 50653/116160, 299693/1626240, 3375/11648, 0]
    ])
  c_sol = np.array([1697/18876, 0, 50653/116160, 299693/1626240, 3375/11648, 0])
  c_error = np.array([-1185/6292, 0, 4107/7744, -68493/108416, 3375/11648, 0])

  def body_fun(i, k):
    ti = t0 + dt * alpha[i-1]
    yi = y0 + dt * np.dot(beta[i-1, :], k)
    ft = func(yi, ti)
    return ops.index_update(k, jax.ops.index[i, :], ft)

  k = ops.index_update(np.zeros((6, f0.shape[0])), ops.index[0, :], f0)
  k = lax.fori_loop(1, 6, body_fun, k)

  y1 = dt * np.dot(c_sol, k) + y0
  y1_error = dt * np.dot(c_error, k)
  f1 = k[-1]
  return y1, f1, y1_error, k

def owrenzen5_step(func, y0, f0, t0, dt):
  # Owrenzen5 tableau
  alpha = np.array([1/6, 1/4, 1/2, 1/2, 9/14, 7/8, 1, 0])
  beta = np.array([
    [1/6, 0, 0, 0, 0, 0, 0, 0],
    [1/16, 3/16, 0, 0, 0, 0, 0, 0],
    [1/4, -3/4, 1, 0, 0, 0, 0, 0],
    [-3/4, 15/4, -3, 1/2, 0, 0, 0, 0],
    [369/1372, -243/343, 297/343, 1485/9604, 297/4802, 0, 0, 0],
    [-133/4512, 1113/6016, 7945/16544, -12845/24064, -315/24064, 156065/198528, 0, 0],
    [83/945, 0, 248/825, 41/180, 1/36, 2401/38610, 6016/20475, 0]
    ])
  c_sol = np.array([-1/9 + 188/945, 0, 40/33 - 752/825, -7/4 + 89/45, -1/12 + 1/9, 343/198 - 32242/19305, 6016/20475, 0])
  c_error = np.array([188/945, 0, -752/825, 89/45, 1/9, -32242/19305, 6016/20475, 0])

  def body_fun(i, k):
    ti = t0 + dt * alpha[i-1]
    yi = y0 + dt * np.dot(beta[i-1, :], k)
    ft = func(yi, ti)
    return ops.index_update(k, jax.ops.index[i, :], ft)

  k = ops.index_update(np.zeros((8, f0.shape[0])), ops.index[0, :], f0)
  k = lax.fori_loop(1, 8, body_fun, k)

  y1 = dt * np.dot(c_sol, k) + y0
  y1_error = dt * np.dot(c_error, k)
  f1 = k[-1]
  return y1, f1, y1_error, k

def tanyam_step(func, y0, f0, t0, dt):
  # Tanyam7 tableau
  alpha = np.array([0.07816646510113846,
                    0.1172496976517077,
                    0.17587454647756157,
                    0.4987401101913988,
                    0.772121690184484,
                    0.9911856696047768,
                    0.9995019582097662,
                    1,
                    0,
                    0])
  beta = np.array([
    [0.07816646510113846, 0, 0, 0, 0, 0,
      0, 0, 0, 0],
    [0.029312424412926925, 0.08793727323878078, 0, 0, 0, 0,
      0, 0, 0, 0],
    [0.04396863661939039, 0, 0.13190590985817116, 0, 0, 0,
      0, 0, 0, 0],
    [0.7361834837738382, 0, -2.8337999624233303, 2.5963565888408913, 0, 0,
      0, 0, 0, 0],
    [-12.062819391370866, 0, 48.20838100175243, -38.05863046463434, 2.6851905444372632, 0,
      0, 0, 0, 0],
    [105.21957276320198, 0, -417.92888626241256, 332.3155504499333, -19.827591183572938, 1.2125399024549859,
      0, 0, 0, 0],
    [114.67755718631742, 0, -455.5612169896097, 362.24095553923144, -21.67190442182809, 1.3189132007137807,
      -0.0048025566150346555, 0, 0, 0],
    [115.21334870553768, 0, -457.69356568613233, 363.93688218862735, -21.776682078900294, 1.3250670887878468,
      -0.004518190986768983, -0.0005320269334859959, 0, 0],
    [115.18928245800194, 0, -457.598022271643, 363.8610256312148, -21.77212754027556, 1.3248804645074317,
      -0.0045057252106918315, -0.0005330165949429136, 0, 0]
    ])
  c_sol = np.array([0.05126014249744686, 0,
                    0, 0.27521638456212627,
                    0.33696650340710543, 0.18986072244906577,
                    8.461099418514403, -130.15941672640542,
                    121.84501355497527, 0]) +\
          np.array([0.0002577249070696835, 0,
                      0, -0.0009229104845391819,
                      0.0031779013374194105, -0.01210458894174817,
                      2.706023959472591, -44.541445641266066,
                      121.84501355497527, -80])
  c_error = np.array([0.0002577249070696835, 0,
                      0, -0.0009229104845391819,
                      0.0031779013374194105, -0.01210458894174817,
                      2.706023959472591, -44.541445641266066,
                      121.84501355497527, -80])

  def body_fun(i, k):
    ti = t0 + dt * alpha[i-1]
    yi = y0 + dt * np.dot(beta[i-1, :], k)
    ft = func(yi, ti)
    return ops.index_update(k, jax.ops.index[i, :], ft)

  k = ops.index_update(np.zeros((10, f0.shape[0])), ops.index[0, :], f0)
  k = lax.fori_loop(1, 10, body_fun, k)

  y1 = dt * np.dot(c_sol, k) + y0
  y1_error = dt * np.dot(c_error, k)
  f1 = k[-1]
  return y1, f1, y1_error, k

def _g_and_explicit_phi(prev_t, next_t, implicit_phi, k):
  curr_t = prev_t[0]
  dt = next_t - prev_t[0]

  beta = 1.

  explicit_phi = np.zeros_like(implicit_phi)
  explicit_phi = jax.ops.index_update(explicit_phi, 0, implicit_phi[0])

  c = 1 / np.arange(1, _ADAMS_MAX_ORDER + 2)

  g = np.zeros(_ADAMS_MAX_ORDER + 1)
  g = jax.ops.index_update(g, 0, 1)

  def body_fun(i, val):
    beta, explicit_phi, c, g = val

    beta = (next_t - prev_t[i - 1]) / (curr_t - prev_t[i]) * beta
    explicit_phi = jax.ops.index_update(explicit_phi, i, implicit_phi[i] * beta)

    idxs = np.arange(_ADAMS_MAX_ORDER + 1)
    c_q = np.where(idxs < k - i + 1, c, 0)   # c[:k - i + 1]
    c_q_1 = np.where(idxs < k + 1 - i + 1, np.where(idxs >= 1, c, 0), 0)  # c[1:k + 1 - i + 1]
    # shift so that it lines up with diff1
    c_q_1 = jax.ops.index_update(c_q_1, jax.ops.index[:-1], c_q_1[1:])
    # c[:k - i + 1] - c[1:k + 1 - i + 1]
    c = lax.cond(i == 1, None, lambda _: c_q - c_q_1, None, lambda _: c_q - c_q_1 * dt / (next_t - prev_t[i - 1]))
    g = jax.ops.index_update(g, i, c[0])

    val = beta, explicit_phi, c, g
    return val

  beta, explicit_phi, c, g = lax.fori_loop(1, k, body_fun, (beta, explicit_phi, c, g))

  # do the c and g update for i = k
  c = jax.ops.index_update(c, jax.ops.index[:1], c[:1] - c[1:2] * dt / (next_t - prev_t[k - 1]))
  g = jax.ops.index_update(g, k, c[0])

  return g, explicit_phi

def _compute_implicit_phi(explicit_phi, f_n, phi_order, k):
  k = lax.min(phi_order + 1, k)
  implicit_phi = np.zeros_like(explicit_phi)
  implicit_phi = jax.ops.index_update(implicit_phi, 0, f_n)
  def body_fun(i, val):
    implicit_phi = val
    implicit_phi = jax.ops.index_update(implicit_phi, i, implicit_phi[i - 1] - explicit_phi[i - 1])
    return implicit_phi
  implicit_phi = lax.fori_loop(1, k, body_fun, implicit_phi)
  return implicit_phi

def adaptive_adams_step(func, y0, prev_f, prev_t, next_t, prev_phi, order, target_t, rtol, atol):
  gamma_star = np.array([
    1, -1 / 2, -1 / 12, -1 / 24, -19 / 720, -3 / 160, -863 / 60480, -275 / 24192, -33953 / 3628800, -0.00789255,
    -0.00678585, -0.00592406, -0.00523669, -0.0046775, -0.00421495, -0.0038269
  ])
  next_t = lax.min(next_t, target_t)  # TODO: integrate as far as possible and interpolate
  dt = next_t - prev_t[0]

  # explicit predictor step
  g, phi = _g_and_explicit_phi(prev_t, next_t, prev_phi, order)

  # compute y0 + dt * np.dot(phi[:max(1, order - 1)].T, g[:max(1, order - 1)])
  p_next = y0 + dt * np.dot(np.where(np.arange(_ADAMS_MAX_ORDER) < lax.max(1, order - 1), phi.T, 0),
                            np.where(np.arange(_ADAMS_MAX_ORDER + 1) < lax.max(1, order - 1), g, 0)[:-1])

  # update phi to implicit
  next_f0 = func(p_next, next_t)
  implicit_phi_p = _compute_implicit_phi(phi, next_f0, order, order + 1)

  # Implicit corrector step.
  y_next = p_next + dt * g[order - 1] * implicit_phi_p[order - 1]

  def compute_error_estimate(order):
    return dt * (g[order] - g[order - 1]) * implicit_phi_p[order]

  # Error estimation.
  local_error = compute_error_estimate(order)
  tolerance = error_tolerance(rtol, atol, y0, y_next)
  error_k = error_ratio_tol(local_error, tolerance)

  def accept(tpl):
    prev_t, prev_f = tpl
    next_f0 = func(y_next, next_t)
    implicit_phi = _compute_implicit_phi(phi, next_f0, order, order + 2)
    next_order = \
      lax.cond(
        len(prev_t) <= 4 or order < 3,
          None,
          lambda _: lax.min(order + 1, lax.min(3, _ADAMS_MAX_ORDER)),
          (error_ratio_tol(compute_error_estimate(order - 1), tolerance),
          error_ratio_tol(compute_error_estimate(order - 2), tolerance)),
          lambda errs:
          lax.cond(
            np.min(errs[0] + errs[1]) < np.max(error_k),
              None,
              lambda _: order - 1,
              None,
              lambda _:
              lax.cond(
                order < _ADAMS_MAX_ORDER,
                  error_ratio_tol(dt * gamma_star[order] * implicit_phi_p[order], tolerance),
                  lambda error_kp1:
                  lax.cond(
                    np.max(error_kp1) < np.max(error_k),
                      None,
                      lambda _: order + 1,
                      None,
                      lambda _: order
                  ),
                  None,
                  lambda _: order
              )
          )
      )

    # Keep step size constant if increasing order. Else use adaptive step size.
    dt_next = lax.cond(next_order > order,
                       None, lambda _: dt,
                       None, lambda _: optimal_step_size(dt, error_k, order=next_order))

    # shift right and insert at 0

    prev_f = jax.ops.index_update(prev_f, jax.ops.index[1:], prev_f[:-1])
    prev_f = jax.ops.index_update(prev_f, 0, next_f0)

    prev_t = jax.ops.index_update(prev_t, jax.ops.index[1:], prev_t[:-1])
    prev_t = jax.ops.index_update(prev_t, 0, next_t)

    return p_next, prev_f, prev_t, next_t + dt_next, implicit_phi, next_order, 2

  def reject(_):
    dt_next = optimal_step_size(dt, error_k, order=order)
    return y0, prev_f, prev_t, prev_t[0] + dt_next, prev_phi, order, 1

  # TODO: why is scoping only needed for some of the variables? and in only one of the cases?
  return lax.cond(error_k <= 1, (prev_t, prev_f), accept, None, reject)

def error_ratio(error_estimate, rtol, atol, y0, y1):
  return error_ratio_tol(error_estimate, error_tolerance(rtol, atol, y0, y1))

def error_tolerance(rtol, atol, y0, y1):
  return atol + rtol * np.maximum(np.abs(y0), np.abs(y1))

def error_ratio_tol(error_estimate, error_tolerance):
  err_ratio = error_estimate / error_tolerance
  # return np.square(np.max(np.abs(err_ratio)))  # (square since optimal_step_size expects squared norm)
  return np.mean(np.square(err_ratio))

def optimal_step_size(last_step, mean_error_ratio, safety=0.9, ifactor=10.0,
                      dfactor=0.2, order=5.0):
  """Compute optimal Runge-Kutta stepsize."""
  mean_error_ratio = np.max(mean_error_ratio)
  dfactor = np.where(mean_error_ratio < 1, 1.0, dfactor)

  err_ratio = np.sqrt(mean_error_ratio)
  factor = np.maximum(1.0 / ifactor,
                      np.minimum(err_ratio**(1.0 / order) / safety, 1.0 / dfactor))
  return np.where(mean_error_ratio == 0, last_step * ifactor, last_step / factor)

def odeint(func, y0, t, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=np.inf):
  """Adaptive stepsize (Dormand-Prince) Runge-Kutta odeint implementation.

  Args:
    func: function to evaluate the time derivative of the solution `y` at time
      `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: array or pytree of arrays representing the initial value for the state.
    t: array of float times for evaluation, like `np.linspace(0., 10., 101)`,
      in which the values must be strictly increasing.
    *args: tuple of additional arguments for `func`.
    rtol: float, relative local error tolerance for solver (optional).
    atol: float, absolute local error tolerance for solver (optional).
    mxstep: int, maximum number of steps to take for each timepoint (optional).

  Returns:
    Values of the solution `y` (i.e. integrated system values) at each time
    point in `t`, represented as an array (or pytree of arrays) with the same
    shape/structure as `y0` except with a new leading axis of length `len(t)`.
  """
  return _odeint_wrapper(func, rtol, atol, mxstep, y0, t, *args)

def odeint_grid(func, y0, t, *args, step_size):
  """Adaptive stepsize (Dormand-Prince) Runge-Kutta odeint implementation.

  Args:
    func: function to evaluate the time derivative of the solution `y` at time
      `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: array or pytree of arrays representing the initial value for the state.
    t: array of float times for evaluation, like `np.linspace(0., 10., 101)`,
      in which the values must be strictly increasing.
    *args: tuple of additional arguments for `func`.
    step_size: step size for the fixed-grid solver
  Returns:
    Values of the solution `y` (i.e. integrated system values) at each time
    point in `t`, represented as an array (or pytree of arrays) with the same
    shape/structure as `y0` except with a new leading axis of length `len(t)`.
  """
  return _odeint_grid_wrapper(func, step_size, y0, t, *args)

def odeint_grid_sepaux(fwd_func, rev_func, y0, t, *args, step_size):
  """Fixed-grid solver for integrating augmented dynamics. Assumes two auxiliary terms since we want to do the
  sum for Finlay.

  Args:
    func: function to evaluate the time derivative of the solution `y` at time
      `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: array or pytree of arrays representing the initial value for the state.
    t: array of float times for evaluation, like `np.linspace(0., 10., 101)`,
      in which the values must be strictly increasing.
    *args: tuple of additional arguments for `func`.
    step_size: step size for the fixed-grid solver
  Returns:
    Values of the solution `y` (i.e. integrated system values) at each time
    point in `t`, represented as an array (or pytree of arrays) with the same
    shape/structure as `y0` except with a new leading axis of length `len(t)`.
  """
  return _odeint_grid_sepaux_wrapper(fwd_func, rev_func, step_size, y0, t, *args)

def odeint_grid_sepaux_one(fwd_func, rev_func, y0, t, *args, step_size):
  """Fixed-grid solver for integrating augmented dynamics. Assumes two auxiliary terms since we want to do the
  sum for Finlay.

  Args:
    func: function to evaluate the time derivative of the solution `y` at time
      `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: array or pytree of arrays representing the initial value for the state.
    t: array of float times for evaluation, like `np.linspace(0., 10., 101)`,
      in which the values must be strictly increasing.
    *args: tuple of additional arguments for `func`.
    step_size: step size for the fixed-grid solver
  Returns:
    Values of the solution `y` (i.e. integrated system values) at each time
    point in `t`, represented as an array (or pytree of arrays) with the same
    shape/structure as `y0` except with a new leading axis of length `len(t)`.
  """
  return _odeint_grid_sepaux_one_wrapper(fwd_func, rev_func, step_size, y0, t, *args)

def odeint_grid_aux(fwd_func, rev_func, y0, t, *args, step_size):
  """Fixed-grid solver for integrating augmented dynamics. Assumes two auxiliary terms since we want to do the
  sum for Finlay.

  Args:
    func: function to evaluate the time derivative of the solution `y` at time
      `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: array or pytree of arrays representing the initial value for the state.
    t: array of float times for evaluation, like `np.linspace(0., 10., 101)`,
      in which the values must be strictly increasing.
    *args: tuple of additional arguments for `func`.
    step_size: step size for the fixed-grid solver
  Returns:
    Values of the solution `y` (i.e. integrated system values) at each time
    point in `t`, represented as an array (or pytree of arrays) with the same
    shape/structure as `y0` except with a new leading axis of length `len(t)`.
  """
  return _odeint_grid_aux_wrapper(fwd_func, rev_func, step_size, y0, t, *args)

def odeint_aux(func, y0, t, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=np.inf):
  """Adaptive stepsize (Dormand-Prince) Runge-Kutta odeint implementation.

  Args:
    func: function to evaluate the time derivative of the solution `y` at time
      `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: array or pytree of arrays representing the initial value for the state.
    t: array of float times for evaluation, like `np.linspace(0., 10., 101)`,
      in which the values must be strictly increasing.
    *args: tuple of additional arguments for `func`.
    rtol: float, relative local error tolerance for solver (optional).
    atol: float, absolute local error tolerance for solver (optional).
    mxstep: int, maximum number of steps to take for each timepoint (optional).

  Returns:
    Values of the solution `y` (i.e. integrated system values) at each time
    point in `t`, represented as an array (or pytree of arrays) with the same
    shape/structure as `y0` except with a new leading axis of length `len(t)`.
  """
  return _odeint_aux_wrapper(func, rtol, atol, mxstep, y0, t, *args)

def odeint_aux_one(fwd_func, rev_func, y0, t, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=np.inf):
  """Adaptive stepsize (Dormand-Prince) Runge-Kutta odeint implementation.

  Args:
    func: function to evaluate the time derivative of the solution `y` at time
      `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: array or pytree of arrays representing the initial value for the state.
    t: array of float times for evaluation, like `np.linspace(0., 10., 101)`,
      in which the values must be strictly increasing.
    *args: tuple of additional arguments for `func`.
    rtol: float, relative local error tolerance for solver (optional).
    atol: float, absolute local error tolerance for solver (optional).
    mxstep: int, maximum number of steps to take for each timepoint (optional).

  Returns:
    Values of the solution `y` (i.e. integrated system values) at each time
    point in `t`, represented as an array (or pytree of arrays) with the same
    shape/structure as `y0` except with a new leading axis of length `len(t)`.
  """
  return _odeint_aux_one_wrapper(fwd_func, rev_func, rtol, atol, mxstep, y0, t, *args)

def odeint_sepaux(fwd_func, rev_func, y0, t, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=np.inf):
  """Adaptive stepsize (Dormand-Prince) Runge-Kutta odeint implementation.

  Args:
    func: function to evaluate the time derivative of the solution `y` at time
      `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: array or pytree of arrays representing the initial value for the state.
    t: array of float times for evaluation, like `np.linspace(0., 10., 101)`,
      in which the values must be strictly increasing.
    *args: tuple of additional arguments for `func`.
    rtol: float, relative local error tolerance for solver (optional).
    atol: float, absolute local error tolerance for solver (optional).
    mxstep: int, maximum number of steps to take for each timepoint (optional).

  Returns:
    Values of the solution `y` (i.e. integrated system values) at each time
    point in `t`, represented as an array (or pytree of arrays) with the same
    shape/structure as `y0` except with a new leading axis of length `len(t)`.
  """
  return _odeint_sepaux_wrapper(fwd_func, rev_func, rtol, atol, mxstep, y0, t, *args)

def odeint_fin_sepaux(fwd_func, rev_func, y0, t, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=np.inf):
  """Adaptive stepsize (Dormand-Prince) Runge-Kutta odeint implementation.

  Args:
    func: function to evaluate the time derivative of the solution `y` at time
      `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: array or pytree of arrays representing the initial value for the state.
    t: array of float times for evaluation, like `np.linspace(0., 10., 101)`,
      in which the values must be strictly increasing.
    *args: tuple of additional arguments for `func`.
    rtol: float, relative local error tolerance for solver (optional).
    atol: float, absolute local error tolerance for solver (optional).
    mxstep: int, maximum number of steps to take for each timepoint (optional).

  Returns:
    Values of the solution `y` (i.e. integrated system values) at each time
    point in `t`, represented as an array (or pytree of arrays) with the same
    shape/structure as `y0` except with a new leading axis of length `len(t)`.
  """
  return _odeint_fin_sepaux_wrapper(fwd_func, rev_func, rtol, atol, mxstep, y0, t, *args)

def odeint_sepaux2(fwd_func, rev_func, y0, t, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=np.inf):
  """Adaptive stepsize (Dormand-Prince) Runge-Kutta odeint implementation.

  Args:
    func: function to evaluate the time derivative of the solution `y` at time
      `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: array or pytree of arrays representing the initial value for the state.
    t: array of float times for evaluation, like `np.linspace(0., 10., 101)`,
      in which the values must be strictly increasing.
    *args: tuple of additional arguments for `func`.
    rtol: float, relative local error tolerance for solver (optional).
    atol: float, absolute local error tolerance for solver (optional).
    mxstep: int, maximum number of steps to take for each timepoint (optional).

  Returns:
    Values of the solution `y` (i.e. integrated system values) at each time
    point in `t`, represented as an array (or pytree of arrays) with the same
    shape/structure as `y0` except with a new leading axis of length `len(t)`.
  """
  _init_nfe = 0.
  return _odeint_sepaux2_wrapper(fwd_func, rev_func, rtol, atol, mxstep, _init_nfe, y0, t, *args)

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _odeint_wrapper(func, rtol, atol, mxstep, y0, ts, *args):
  y0, unravel = ravel_pytree(y0)
  func = ravel_first_arg(func, unravel)
  out, nfe = _dopri5_odeint(func, rtol, atol, mxstep, y0, ts, *args)
  return jax.vmap(unravel)(out), nfe

@partial(jax.jit, static_argnums=(0, 1))
def _odeint_grid_wrapper(func, step_size, y0, ts, *args):
  y0, unravel = ravel_pytree(y0)
  func = ravel_first_arg(func, unravel)
  out, nfe = _rk4_odeint(func, step_size, y0, ts, *args)
  return jax.vmap(unravel)(out), nfe

@partial(jax.jit, static_argnums=(0, 1, 2))
def _odeint_grid_sepaux_wrapper(fwd_func, rev_func, step_size, y0, ts, *args):
  fwd_func = ravel_first_arg(fwd_func, ravel_pytree(y0[0])[1])
  rev_func = ravel_first_arg(rev_func, ravel_pytree(y0)[1])
  return _rk4_odeint_sepaux(fwd_func, rev_func, step_size, y0, ts, *args)

@partial(jax.jit, static_argnums=(0, 1, 2))
def _odeint_grid_sepaux_one_wrapper(fwd_func, rev_func, step_size, y0, ts, *args):
  fwd_func = ravel_first_arg(fwd_func, ravel_pytree(y0[0])[1])
  rev_func = ravel_first_arg(rev_func, ravel_pytree(y0)[1])
  return _rk4_odeint_sepaux_one(fwd_func, rev_func, step_size, y0, ts, *args)

@partial(jax.jit, static_argnums=(0, 1, 2))
def _odeint_grid_aux_wrapper(fwd_func, rev_func, step_size, y0, ts, *args):
  fwd_func = ravel_first_arg(fwd_func, ravel_pytree(y0[0])[1])
  rev_func = ravel_first_arg(rev_func, ravel_pytree(y0)[1])
  return _rk4_odeint_aux(fwd_func, rev_func, step_size, y0, ts, *args)

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _odeint_aux_wrapper(func, rtol, atol, mxstep, y0, ts, *args):
  return _dopri5_odeint_aux(func, rtol, atol, mxstep, y0, ts, *args)

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _odeint_aux_one_wrapper(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args):
  fwd_func = ravel_first_arg(fwd_func, ravel_pytree(y0[0])[1])
  rev_func = ravel_first_arg(rev_func, ravel_pytree(y0)[1])
  return _dopri5_odeint_aux_one(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args)

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _odeint_sepaux_wrapper(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args):
  fwd_func = ravel_first_arg(fwd_func, ravel_pytree(y0[0])[1])
  rev_func = ravel_first_arg(rev_func, ravel_pytree(y0)[1])
  return _dopri5_odeint_sepaux(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args)

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _odeint_fin_sepaux_wrapper(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args):
  fwd_func = ravel_first_arg(fwd_func, ravel_pytree(y0[0])[1])
  rev_func = ravel_first_arg(rev_func, ravel_pytree(y0)[1])
  return _dopri5_odeint_fin_sepaux(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args)

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _odeint_sepaux2_wrapper(fwd_func, rev_func, rtol, atol, mxstep, _init_nfe, y0, ts, *args):
  fwd_func = ravel_first_arg(fwd_func, ravel_pytree(y0[0])[1])
  rev_func = ravel_first_arg(rev_func, ravel_pytree(y0)[1])
  return _dopri5_odeint_sepaux2(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args)

def _euler_odeint(func, num_steps, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  # TODO: does not work for multiple ts

  length = ts[-1] - ts[0]
  step_size = length / num_steps

  f0 = func_(y0, ts[0])
  init_val = [y0, f0, ts[0]]
  def for_body_fun(i, val):
    cur_y, cur_f, cur_t = val
    next_t = cur_t + step_size
    next_y = cur_y + cur_f * step_size
    next_f = func_(next_y, next_t)
    return [next_y, next_f, next_t]
  final_y, _, final_t = lax.fori_loop(0, num_steps, for_body_fun, init_val)

  return final_y


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _dopri5_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      next_y, next_f, next_y_error, k = runge_kutta_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      new_interp_coeff = interp_fit_dopri(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 6 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _rk4_odeint(func, step_size, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def step_func(cur_t, dt, cur_y):
    """
    Take one step of RK4.
    """
    k1 = func_(cur_y, cur_t)
    k2 = func_(cur_y + dt * k1 / 2, cur_t + dt / 2)
    k3 = func_(cur_y + dt * k2 / 2, cur_t + dt / 2)
    k4 = func_(cur_y + dt * k3, cur_t + dt)
    return (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

  def cond_fun(carry):
    """
    Check if we've reached the last timepoint.
    """
    cur_y, cur_t, cur_nfe = carry
    return cur_t < ts[-1]

  # TODO: this doesn't work for multiple time points
  def body_fun(carry):
    """
    Take one step of RK4.
    """
    cur_y, cur_t, cur_nfe = carry
    next_t = lax.min(cur_t + step_size, ts[-1])
    dt = next_t - cur_t
    dy = step_func(cur_t, dt, cur_y)
    next_y = cur_y + dy
    new_nfe = cur_nfe + 4
    new_carry = [next_y, next_t, new_nfe]
    return new_carry

  init_t = ts[0]
  init_nfe = 0
  init_carry = [y0, init_t, init_nfe]
  y1, t1, nfe = lax.while_loop(cond_fun, body_fun, init_carry)
  return np.concatenate((y0[None], y1[None])), nfe


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _rk4_odeint_sepaux_one(fwd_func, rev_func, step_size, y0, ts, *args):
  flat_y0, unravel = ravel_pytree(y0[0])
  flat_out, nfe = _rk4_odeint(fwd_func, step_size, flat_y0, ts, *args)
  out = jax.vmap(unravel)(flat_out)
  def aug_init(y):
    """
    Initialize dynamics with 0 for logpx and regs.
    TODO: this is copied from nodes_ffjord.py
    """
    batch_size = y.shape[0]
    return y, np.zeros((batch_size, 1)), np.zeros((batch_size, 1))
  return jax.vmap(aug_init)(out), nfe

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _rk4_odeint_aux(fwd_func, rev_func, step_size, y0, ts, *args):
  flat_y0, unravel = ravel_pytree(y0[0])
  flat_out, nfe = _rk4_odeint(fwd_func, step_size, flat_y0, ts, *args)
  out = jax.vmap(unravel)(flat_out)
  def aug_init(y):
    """
    Initialize dynamics with 0 for logpx and regs.
    TODO: this is copied from nodes_ffjord.py
    """
    batch_size = y.shape[0]
    return y, np.zeros((batch_size, 1))
  return jax.vmap(aug_init)(out), nfe

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _rk4_odeint_sepaux(fwd_func, rev_func, step_size, y0, ts, *args):
  flat_y0, unravel = ravel_pytree(y0[0])
  flat_out, nfe = _rk4_odeint(fwd_func, step_size, flat_y0, ts, *args)
  out = jax.vmap(unravel)(flat_out)
  def aug_init(y):
    """
    Initialize dynamics with 0 for logpx and regs.
    TODO: this is copied from nodes_ffjord.py
    """
    batch_size = y.shape[0]
    return y, np.zeros((batch_size, 1)), np.zeros((batch_size, 1)), np.zeros((batch_size, 1))
  return jax.vmap(aug_init)(out), nfe

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _dopri5_odeint_aux(func, rtol, atol, mxstep, y0, ts, *args):
  y0, unravel = ravel_pytree(y0)
  func = ravel_first_arg(func, unravel)

  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      next_y, next_f, next_y_error, k = runge_kutta_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      new_interp_coeff = interp_fit_dopri(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 6 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  out = np.concatenate((y0[None], ys))
  return jax.vmap(unravel)(out), nfe

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _dopri5_odeint_aux_one(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args):
  flat_y0, unravel = ravel_pytree(y0[0])
  flat_out, nfe = _dopri5_odeint(fwd_func, rtol, atol, mxstep, flat_y0, ts, *args)
  out = jax.vmap(unravel)(flat_out)
  def aug_init(y):
    """
    Initialize dynamics with 0 for reg.
    """
    batch_size = y.shape[0]
    return y, np.zeros((batch_size, 1))
  return jax.vmap(aug_init)(out), nfe

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _dopri5_odeint_sepaux(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args):
  flat_y0, unravel = ravel_pytree(y0[0])
  flat_out, nfe = _dopri5_odeint(fwd_func, rtol, atol, mxstep, flat_y0, ts, *args)
  out = jax.vmap(unravel)(flat_out)
  def aug_init(y):
    """
    Initialize dynamics with 0 for logpx and regs.
    TODO: this is copied from nodes_ffjord.py
    """
    batch_size = y.shape[0]
    return y, np.zeros((batch_size, 1)), np.zeros((batch_size, 1))
  return jax.vmap(aug_init)(out), nfe

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _dopri5_odeint_fin_sepaux(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args):
  flat_y0, unravel = ravel_pytree(y0[0])
  flat_out, nfe = _dopri5_odeint(fwd_func, rtol, atol, mxstep, flat_y0, ts, *args)
  out = jax.vmap(unravel)(flat_out)
  def aug_init(y):
    """
    Initialize dynamics with 0 for logpx and regs.
    TODO: this is copied from nodes_ffjord.py
    """
    batch_size = y.shape[0]
    return y, np.zeros((batch_size, 1)), np.zeros((batch_size, 1)), np.zeros((batch_size, 1))
  return jax.vmap(aug_init)(out), nfe

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _dopri5_odeint_sepaux2(fwd_func, rev_func, rtol, atol, mxstep, _init_nfe, y0, ts, *args):
  flat_y0, unravel = ravel_pytree(y0[0])
  flat_out, nfe = _dopri5_odeint(fwd_func, rtol, atol, mxstep, flat_y0, ts, *args)
  out = jax.vmap(unravel)(flat_out)
  def aug_init(y):
    """
    Initialize dynamics with 0 for logpx and regs.
    TODO: this is copied from nodes_ffjord.py
    """
    batch_size = y.shape[0]
    return y, np.zeros((batch_size, 1)), np.zeros((batch_size, 1))
  return jax.vmap(aug_init)(out), nfe

def _heun_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = heun_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = heun_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=2)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 1 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 1, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

def _fehlberg_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = fehlberg_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = fehlberg_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=2)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 2 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 1, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

def _bosh_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = bosh_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = bosh_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=3)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 3 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 2, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

def _owrenzen_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = owrenzen_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = owrenzen_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=4)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 5 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 3, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

def _rk_fehlberg_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = rk_fehlberg_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = rk_fehlberg_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=5)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 5 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

def _cash_karp_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = cash_karp_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = cash_karp_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=5)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 5 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

def _owrenzen5_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = owrenzen5_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = owrenzen5_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=5)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 7 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

def _tanyam_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, dt, _, _ = state
      return (t < target_t) & (i < mxstep) & (dt > 0)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      dt = np.where(t + dt > target_t, target_t - t, dt)
      next_y, next_f, next_y_error, k = tanyam_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      y_mid, _, _, _ = tanyam_step(func_, y, f, t, dt / 2)
      new_interp_coeff = np.array(fit_4th_order_polynomial(y0, next_y, y_mid, k[0], k[-1], dt))
      # new_interp_coeff = interp_fit_bosh(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios, order=5)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 9 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  # init_nfe = 1.
  # dt = 0.1
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

# @partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _adams_odeint(func, rtol, atol, mxstep, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, prev_t, _, _, _, _ = state
      return (prev_t[0] < target_t) & (i < mxstep)

    def body_fun(state):
      i, y, prev_f, prev_t, next_t, prev_phi, order, nfe = state
      y, prev_f, prev_t, next_t, prev_phi, order, cur_nfe = \
        adaptive_adams_step(func_, y, prev_f, prev_t, next_t, prev_phi, order, target_t, rtol, atol)
      return [i + 1, y, prev_f, prev_t, next_t, prev_phi, order, nfe + cur_nfe]

    _, *carry = lax.while_loop(cond_fun, body_fun, [0] + carry)
    y_target, *_ = carry
    return carry, y_target

  t0 = ts[0]
  f0 = func_(y0, t0)
  ode_dim = f0.shape[0]
  init_nfe = 2.
  dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)

  prev_f = np.empty((_ADAMS_MAX_ORDER + 1, ode_dim))
  prev_f = jax.ops.index_update(prev_f, 0, f0)

  prev_t = np.empty(_ADAMS_MAX_ORDER + 1)
  prev_t = jax.ops.index_update(prev_t, 0, t0)

  prev_phi = np.empty((_ADAMS_MAX_ORDER, ode_dim))
  prev_phi = jax.ops.index_update(prev_phi, 0, f0)

  next_t = t0 + dt
  init_order = 1

  init_carry = [y0,
                prev_f,
                prev_t,
                next_t,
                prev_phi,
                init_order,
                init_nfe]

  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return np.concatenate((y0[None], ys)), nfe

def _odeint_fwd(func, rtol, atol, mxstep, y0, ts, *args):
  ys, nfe = _dopri5_odeint(func, rtol, atol, mxstep, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _rk4_odeint_fwd(func, step_size, y0, ts, *args):
  ys, nfe = _rk4_odeint(func, step_size, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _rk4_odeint_sepaux_fwd(fwd_func, rev_func, step_size, y0, ts, *args):
  ys, nfe = _rk4_odeint_sepaux(fwd_func, rev_func, step_size, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _rk4_odeint_sepaux_one_fwd(fwd_func, rev_func, step_size, y0, ts, *args):
  ys, nfe = _rk4_odeint_sepaux_one(fwd_func, rev_func, step_size, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _rk4_odeint_aux_fwd(fwd_func, rev_func, step_size, y0, ts, *args):
  ys, nfe = _rk4_odeint_aux(fwd_func, rev_func, step_size, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _odeint_aux_fwd(func, rtol, atol, mxstep, y0, ts, *args):
  def aug_init(y):
    """
    Initialize dynamics with 0 for logpx and regs.
    TODO: this is copied from nodes_ffjord.py
    """
    batch_size = y.shape[0]
    return y, np.zeros((batch_size, 1)), np.zeros((batch_size, 1))
  # this doesn't fully implement the finlay trick since we're still paying the price of evaluating
  # the augmented dynamics, but we're only integrating it's unaugmented portion
  ys, nfe = _dopri5_odeint_aux(lambda y, t, *args, **kwargs: func(aug_init(y), t, *args, **kwargs)[0],
                               rtol, atol, mxstep, y0[0], ts, *args)
  # assumes state has two augmented variables (one for div, one for reg)
  aug_ys = (ys, np.zeros((ys.shape[0], *y0[1].shape)), np.zeros((ys.shape[0], *y0[2].shape)))
  return (aug_ys, nfe), (aug_ys, ts, args)

def _odeint_aux_one_fwd(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args):
  ys, nfe = _dopri5_odeint_aux_one(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _odeint_sepaux_fwd(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args):
  ys, nfe = _dopri5_odeint_sepaux(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _odeint_fin_sepaux_fwd(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args):
  ys, nfe = _dopri5_odeint_fin_sepaux(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _odeint_sepaux2_fwd(fwd_func, rev_func, rtol, atol, mxstep, _init_nfe, y0, ts, *args):
  ys, nfe = _dopri5_odeint_sepaux(fwd_func, rev_func, rtol, atol, mxstep, y0, ts, *args)
  return (ys, nfe), (ys, ts, args)

def _odeint_rev(func, rtol, atol, mxstep, res, g):
  ys, ts, args = res
  g, _ = g

  def aug_dynamics(augmented_state, t, *args):
    """Original system augmented with vjp_y, vjp_t and vjp_args."""
    y, y_bar, *_ = augmented_state
    # `t` here is negatice time, so we need to negate again to get back to
    # normal time. See the `odeint` invocation in `scan_fun` below.
    y_dot, vjpfun = jax.vjp(func, y, -t, *args)
    return (-y_dot, *vjpfun(y_bar))

  y_bar = g[-1]
  ts_bar = []
  t0_bar = 0.

  def scan_fun(carry, i):
    y_bar, t0_bar, args_bar = carry
    # Compute effect of moving measurement time
    t_bar = np.dot(func(ys[i], ts[i], *args), g[i])
    t0_bar = t0_bar - t_bar
    # Run augmented system backwards to previous observation
    _, y_bar, t0_bar, args_bar = odeint(
        aug_dynamics, (ys[i], y_bar, t0_bar, args_bar),
        np.array([-ts[i], -ts[i - 1]]),
        *args, rtol=rtol, atol=atol, mxstep=mxstep)[0]
    y_bar, t0_bar, args_bar = tree_map(op.itemgetter(1), (y_bar, t0_bar, args_bar))
    # Add gradient from current output
    y_bar = y_bar + g[i - 1]
    return (y_bar, t0_bar, args_bar), t_bar

  init_carry = (g[-1], 0., tree_map(np.zeros_like, args))
  (y_bar, t0_bar, args_bar), rev_ts_bar = lax.scan(
      scan_fun, init_carry, np.arange(len(ts) - 1, 0, -1))
  ts_bar = np.concatenate([np.array([t0_bar]), rev_ts_bar[::-1]])
  return (y_bar, ts_bar, *args_bar)

def _rk4_odeint_rev(func, step_size, res, g):
  ys, ts, args = res
  g, _ = g

  def aug_dynamics(augmented_state, t, *args):
    """Original system augmented with vjp_y, vjp_t and vjp_args."""
    y, y_bar, *_ = augmented_state
    # `t` here is negatice time, so we need to negate again to get back to
    # normal time. See the `odeint` invocation in `scan_fun` below.
    y_dot, vjpfun = jax.vjp(func, y, -t, *args)
    return (-y_dot, *vjpfun(y_bar))

  y_bar = g[-1]
  ts_bar = []
  t0_bar = 0.

  def scan_fun(carry, i):
    y_bar, t0_bar, args_bar = carry
    # Compute effect of moving measurement time
    t_bar = np.dot(func(ys[i], ts[i], *args), g[i])
    t0_bar = t0_bar - t_bar
    # Run augmented system backwards to previous observation
    _, y_bar, t0_bar, args_bar = odeint_grid(
        aug_dynamics, (ys[i], y_bar, t0_bar, args_bar),
        np.array([-ts[i], -ts[i - 1]]),
        *args, step_size=step_size)[0]
    y_bar, t0_bar, args_bar = tree_map(op.itemgetter(1), (y_bar, t0_bar, args_bar))
    # Add gradient from current output
    y_bar = y_bar + g[i - 1]
    return (y_bar, t0_bar, args_bar), t_bar

  init_carry = (g[-1], 0., tree_map(np.zeros_like, args))
  (y_bar, t0_bar, args_bar), rev_ts_bar = lax.scan(
      scan_fun, init_carry, np.arange(len(ts) - 1, 0, -1))
  ts_bar = np.concatenate([np.array([t0_bar]), rev_ts_bar[::-1]])
  return (y_bar, ts_bar, *args_bar)

def _rk4_odeint_sepaux_rev(fwd_func, rev_func, step_size, res, g):
  flatten_func = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])  # flatten everything but time dim
  flat_res = flatten_func(res[0])
  flat_g = flatten_func(g[0])
  result = _rk4_odeint_rev(rev_func, step_size, (flat_res, *res[1:]), (flat_g, g[1]))

  unravel = ravel_pytree(tree_map(op.itemgetter(0), res[0]))[1]
  return (unravel(result[0]), *result[1:])

def _rk4_odeint_sepaux_one_rev(fwd_func, rev_func, step_size, res, g):
  flatten_func = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])  # flatten everything but time dim
  flat_res = flatten_func(res[0])
  flat_g = flatten_func(g[0])
  result = _rk4_odeint_rev(rev_func, step_size, (flat_res, *res[1:]), (flat_g, g[1]))

  unravel = ravel_pytree(tree_map(op.itemgetter(0), res[0]))[1]
  return (unravel(result[0]), *result[1:])

def _rk4_odeint_aux_rev(fwd_func, rev_func, step_size, res, g):
  flatten_func = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])  # flatten everything but time dim
  flat_res = flatten_func(res[0])
  flat_g = flatten_func(g[0])
  result = _rk4_odeint_rev(rev_func, step_size, (flat_res, *res[1:]), (flat_g, g[1]))

  unravel = ravel_pytree(tree_map(op.itemgetter(0), res[0]))[1]
  return (unravel(result[0]), *result[1:])

def _odeint_rev2(func, rtol, atol, mxstep, res, g):
  ys, ts, args = res
  g, _ = g

  def aug_dynamics(augmented_state, t, *args):
    """Original system augmented with vjp_y, vjp_t and vjp_args."""
    y, y_bar, *_ = augmented_state
    # `t` here is negatice time, so we need to negate again to get back to
    # normal time. See the `odeint` invocation in `scan_fun` below.
    y_dot, vjpfun = jax.vjp(func, y, -t, *args)
    return (-y_dot, *vjpfun(y_bar))

  y_bar = g[-1]
  ts_bar = []
  t0_bar = 0.

  def scan_fun(carry, i):
    y_bar, t0_bar, args_bar, nfe = carry
    # Compute effect of moving measurement time
    t_bar = np.dot(func(ys[i], ts[i], *args), g[i])
    t0_bar = t0_bar - t_bar
    # Run augmented system backwards to previous observation
    (_, y_bar, t0_bar, args_bar), cur_nfe = odeint(
        aug_dynamics, (ys[i], y_bar, t0_bar, args_bar),
        np.array([-ts[i], -ts[i - 1]]),
        *args, rtol=rtol, atol=atol, mxstep=mxstep)
    nfe += cur_nfe + 1
    y_bar, t0_bar, args_bar = tree_map(op.itemgetter(1), (y_bar, t0_bar, args_bar))
    # Add gradient from current output
    y_bar = y_bar + g[i - 1]
    return (y_bar, t0_bar, args_bar, nfe), t_bar

  init_carry = (g[-1], 0., tree_map(np.zeros_like, args), 0.)
  (y_bar, t0_bar, args_bar, nfe), rev_ts_bar = lax.scan(
      scan_fun, init_carry, np.arange(len(ts) - 1, 0, -1))
  ts_bar = np.concatenate([np.array([t0_bar]), rev_ts_bar[::-1]])
  return (nfe, y_bar, ts_bar, *args_bar)

def _odeint_aux_rev(func, rtol, atol, mxstep, res, g):
  aug_ys, ts, args = res

  # we want to ravel the tuple after indexing each of its elements in time
  # too difficult, and not worth it since it's less efficient
  ys, unravel = ravel_pytree(aug_ys)
  func = ravel_first_arg(func, unravel)
  aug_g, _ = g
  g, _ = ravel_pytree(aug_g)  # don't need the unravel, it's the same one for ys

  def aug_dynamics(augmented_state, t, *args):
    """Original system augmented with vjp_y, vjp_t and vjp_args."""
    y, y_bar, *_ = augmented_state
    # `t` here is negatice time, so we need to negate again to get back to
    # normal time. See the `odeint` invocation in `scan_fun` below.
    y_dot, vjpfun = jax.vjp(func, y, -t, *args)
    return (-y_dot, *vjpfun(y_bar))

  y_bar = g[-1]
  ts_bar = []
  t0_bar = 0.

  def scan_fun(carry, i):
    y_bar, t0_bar, args_bar = carry
    # Compute effect of moving measurement time
    t_bar = np.dot(func(ys[i], ts[i], *args), g[i])
    t0_bar = t0_bar - t_bar
    # Run augmented system backwards to previous observation
    _, y_bar, t0_bar, args_bar = odeint(
        aug_dynamics, (ys[i], y_bar, t0_bar, args_bar),
        np.array([-ts[i], -ts[i - 1]]),
        *args, rtol=rtol, atol=atol, mxstep=mxstep)[0]
    y_bar, t0_bar, args_bar = tree_map(op.itemgetter(1), (y_bar, t0_bar, args_bar))
    # Add gradient from current output
    y_bar = y_bar + g[i - 1]
    return (y_bar, t0_bar, args_bar), t_bar

  init_carry = (g[-1], 0., tree_map(np.zeros_like, args))
  (y_bar, t0_bar, args_bar), rev_ts_bar = lax.scan(
      scan_fun, init_carry, np.arange(len(ts) - 1, 0, -1))
  ts_bar = np.concatenate([np.array([t0_bar]), rev_ts_bar[::-1]])
  y_bar = unravel(y_bar)
  return (y_bar, ts_bar, *args_bar)

def _odeint_aux_one_rev(fwd_func, rev_func, rtol, atol, mxstep, res, g):
  flatten_func = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])  # flatten everything but time dim
  flat_res = flatten_func(res[0])
  flat_g = flatten_func(g[0])
  result = _odeint_rev(rev_func, rtol, atol, mxstep, (flat_res, *res[1:]), (flat_g, g[1]))

  unravel = ravel_pytree(tree_map(op.itemgetter(0), res[0]))[1]
  return (unravel(result[0]), *result[1:])

def _odeint_sepaux_rev(fwd_func, rev_func, rtol, atol, mxstep, res, g):
  flatten_func = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])  # flatten everything but time dim
  flat_res = flatten_func(res[0])
  flat_g = flatten_func(g[0])
  result = _odeint_rev(rev_func, rtol, atol, mxstep, (flat_res, *res[1:]), (flat_g, g[1]))

  unravel = ravel_pytree(tree_map(op.itemgetter(0), res[0]))[1]
  return (unravel(result[0]), *result[1:])

def _odeint_sepaux2_rev(fwd_func, rev_func, rtol, atol, mxstep, res, g):
  flatten_func = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])  # flatten everything but time dim
  flat_res = flatten_func(res[0])
  flat_g = flatten_func(g[0])
  nfe, *result = _odeint_rev2(rev_func, rtol, atol, mxstep, (flat_res, *res[1:]), (flat_g, g[1]))

  unravel = ravel_pytree(tree_map(op.itemgetter(0), res[0]))[1]
  return (nfe, unravel(result[0]), *result[1:])

methods = {
  "dopri5": _dopri5_odeint,
  "adams": _adams_odeint
}
_dopri5_odeint.defvjp(_odeint_fwd, _odeint_rev)
_rk4_odeint.defvjp(_rk4_odeint_fwd, _rk4_odeint_rev)
_rk4_odeint_sepaux.defvjp(_rk4_odeint_sepaux_fwd, _rk4_odeint_sepaux_rev)
_rk4_odeint_sepaux_one.defvjp(_rk4_odeint_sepaux_one_fwd, _rk4_odeint_sepaux_one_rev)
_rk4_odeint_aux.defvjp(_rk4_odeint_aux_fwd, _rk4_odeint_aux_rev)
_dopri5_odeint_aux_one.defvjp(_odeint_aux_one_fwd, _odeint_aux_one_rev)
_dopri5_odeint_sepaux.defvjp(_odeint_sepaux_fwd, _odeint_sepaux_rev)
_dopri5_odeint_fin_sepaux.defvjp(_odeint_fin_sepaux_fwd, _odeint_sepaux_rev)
_dopri5_odeint_sepaux2.defvjp(_odeint_sepaux2_fwd, _odeint_sepaux2_rev)
# _dopri5_odeint_aux.defvjp(_odeint_aux_fwd, _odeint_aux_rev)
# _adams_odeint.defvjp(partial(_odeint_fwd, _adams_odeint), partial(_odeint_rev, "adams"))

def pend(np, y, _, m, g):
  theta, omega = y
  return [omega, -m * omega - g * np.sin(theta)]

def benchmark_odeint(fun, y0, tspace, *args, **kwargs):
  """Time performance of JAX odeint method against scipy.integrate.odeint."""
  n_trials = 10
  n_repeat = 100
  y0, tspace = onp.array(y0), onp.array(tspace)
  onp_fun = partial(fun, onp)
  scipy_times = []
  for k in range(n_trials):
    start = time.time()
    for _ in range(n_repeat):
      scipy_result = osp_integrate.odeint(onp_fun, y0, tspace, args)
    end = time.time()
    print('scipy odeint elapsed time ({} of {}): {}'.format(k+1, n_trials, end-start))
    scipy_times.append(end - start)
  scipy_result, infodict = osp_integrate.odeint(onp_fun, y0, tspace, args,
                                                full_output=True,
                                                rtol=kwargs["rtol"],
                                                atol=kwargs["atol"])
  sc_nfe = infodict["nfe"][-1]
  y0, tspace = np.array(y0), np.array(tspace)
  jax_fun = partial(fun, np)
  jax_times = []
  for k in range(n_trials):
    start = time.time()
    for _ in range(n_repeat):
      jax_result, jax_nfe = odeint(jax_fun, y0, tspace, *args, **kwargs)
    jax_result.block_until_ready()
    end = time.time()
    print('JAX odeint elapsed time ({} of {}): {}'.format(k+1, n_trials, end-start))
    jax_times.append(end - start)
  print('(avg scipy time) / (avg jax time) = {}'.format(
      onp.mean(scipy_times[1:]) / onp.mean(jax_times[1:])))
  print('norm(scipy result-jax result): {}'.format(
      np.linalg.norm(np.asarray(scipy_result) - jax_result)))
  print("jax nfe, scipy nfe", jax_nfe, sc_nfe)
  return scipy_result, jax_result

def pend_benchmark_odeint(**kwargs):
  _, _ = benchmark_odeint(pend, [np.pi - 0.1, 0.0], np.linspace(0., 10., 2),
                          0.25, 9.8, **kwargs)

def pend_check_grads():
  def f(y0, ts, *args):
    return odeint(partial(pend, np), y0, ts, *args)[0]

  y0 = [np.pi - 0.1, 0.0]
  ts = np.linspace(0., 1., 11)
  args = (0.25, 9.8)

  check_grads(f, (y0, ts, *args), modes=["rev"], order=2,
              atol=1e-1, rtol=1e-1)

def _max_abs(tensor):
    return np.max(np.abs(tensor))

def _rel_error(true, estimate):
    return _max_abs((true - estimate) / true)

def const_test(**kwargs):
  a = 0.2
  b = 3.
  f = lambda y, t: a + (y - (a * t + b)) ** 5
  exact = lambda t: a * t + b

  t_points = np.linspace(1, 8, 2)
  sol = exact(t_points)
  y0 = sol[0]
  ys, nfe = odeint(f, y0, t_points, **kwargs)
  sc_ys, infodict = osp_integrate.odeint(f, onp.array(y0), onp.array(t_points),
                                         full_output=True,
                                         rtol=kwargs["rtol"],
                                         atol=kwargs["atol"])
  sc_nfe = infodict["nfe"][-1]
  print("Constant\t(abs, rel, nfe, sc_nfe)\t%.4e, %.4e, %d, %d" % (_max_abs(sol - ys), _rel_error(sol, ys), nfe, sc_nfe))

def rk4_const_test():
  a = 0.2
  b = 3.
  f = lambda y, t: a + (y - (a * t + b)) ** 5
  exact = lambda t: a * t + b

  t_points, step_size = np.linspace(1, 8, 10, retstep=True)
  sol = exact(t_points)
  sol = np.concatenate((sol[0][None], sol[-1][None]))
  y0 = sol[0]
  ys, nfe = _rk4_odeint(f, step_size, y0, t_points)
  print("Constant\t(abs, rel, nfe)\t%.4e, %.4e, %d" % (_max_abs(sol - ys), _rel_error(sol, ys), nfe))

def linear_test(**kwargs):
  dim = 10
  rng = jax.random.PRNGKey(0)
  U = jax.random.normal(rng, (dim, dim)) * 0.1
  A = 2 * U - (U + U.T)
  initial_val = np.ones((dim, 1))

  f = lambda np, y, t: np.dot(A, y)
  def exact(t):
    ans = []
    for t_i in t:
        ans.append(np.matmul(scipy.linalg.expm(A * t_i), initial_val))
    return np.stack([np.array(ans_) for ans_ in ans]).reshape(len(t), dim)

  t_points = np.linspace(1, 8, 2)
  sol = exact(t_points)
  y0 = sol[0]
  ys, nfe = odeint(partial(f, np), y0, t_points, **kwargs)
  sc_ys, infodict = osp_integrate.odeint(partial(f, onp), onp.array(y0), onp.array(t_points),
                                         full_output=True,
                                         rtol=kwargs["rtol"],
                                         atol=kwargs["atol"])
  sc_nfe = infodict["nfe"][-1]
  print("Linear\t\t(abs, rel, nfe, sc_nfe)\t%.4e, %.4e, %d, %d" % (_max_abs(sol - ys), _rel_error(sol, ys), nfe, sc_nfe))

def sin_test(**kwargs):
  f = lambda np, y, t: 2 * y / t + t**4 * np.sin(2 * t) - t**2 + 4 * t**3
  exact = lambda t: -0.5 * t**4 * np.cos(2 * t) + 0.5 * t**3 * np.sin(2 * t) + 0.25 * t**2 * np.cos(2 * t) - \
                    t**3 + 2 * t**4 + (np.pi - 0.25) * t**2

  t_points = np.linspace(1, 8, 2)
  sol = exact(t_points)
  y0 = sol[0]
  ys, nfe = odeint(partial(f, np), y0, t_points, **kwargs)
  sc_ys, infodict = osp_integrate.odeint(partial(f, onp), onp.array(y0), onp.array(t_points),
                                         full_output=True,
                                         rtol=kwargs["rtol"],
                                         atol=kwargs["atol"])
  sc_nfe = infodict["nfe"][-1]
  print("Sine\t\t(abs, rel, nfe, sc_nfe)\t%.4e, %.4e, %d, %d" % (_max_abs(sol - ys), _rel_error(sol, ys), nfe, sc_nfe))

def weird_time_pendulum_check_grads():
  """Test that gradients are correct when the dynamics depend on t."""
  def f(y0, ts):
    return odeint(lambda y, t: np.array([y[1] * -t, -1 * y[1] - 9.8 * np.sin(y[0])]), y0, ts)[0]

  y0 = [np.pi - 0.1, 0.0]
  ts = np.linspace(0., 1., 11)

  check_grads(f, (y0, ts), modes=["rev"], order=2)

if __name__ == '__main__':
  # kwargs = {
  #   # "method": "dopri5",
  #   # "init_step": 0.1,
  #   "atol": 1.4e-8,
  #   "rtol": 1.4e-8
  # }
  # const_test(**kwargs)
  # linear_test(**kwargs)
  # sin_test(**kwargs)
  # pend_benchmark_odeint(**kwargs)
  # pend_check_grads()
  # weird_time_pendulum_check_grads()
  # print("method\t\t", kwargs["method"])
  rk4_const_test()
