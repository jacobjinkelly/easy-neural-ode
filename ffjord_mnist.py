"""
FFJORD on MNIST, implemented with Haiku.
"""
import argparse
import collections
import os
import pickle
import sys
import time

import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from jax import lax
from jax.experimental import optimizers
from jax.experimental.jet import jet
from jax.flatten_util import ravel_pytree
from jax.scipy.special import expit as sigmoid
from jax.tree_util import tree_flatten

from lib.ode import odeint, odeint_sepaux, odeint_fin_sepaux, odeint_grid, odeint_grid_sepaux, odeint_grid_sepaux_one

parser = argparse.ArgumentParser('FFJORd MNIST')
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--test_batch_size', type=int, default=200)
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--warmup_itrs', type=float, default=1e3)
parser.add_argument("--max_grad_norm", type=float, default=1e10)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--lam_w', type=float, default=0)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument('--vmap', action="store_true")
parser.add_argument('--reg', type=str, choices=['none', 'r2', 'r3', 'r4'], default='none')
parser.add_argument('--test_freq', type=int, default=30000)  # save time boi
parser.add_argument('--save_freq', type=int, default=30000)  # save time boi
parser.add_argument('--dirname', type=str, default='tmp')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--no_count_nfe', action="store_true")
parser.add_argument('--ckpt_freq', type=int, default=30000)  # divide test and save, divisible by num_batches
parser.add_argument('--ckpt_path', type=str, default="./ck.pt")
parser.add_argument('--lam_fro', type=float, default=0)
parser.add_argument('--lam_kin', type=float, default=0)
parser.add_argument('--reg_type', type=str, choices=['our', 'fin'], default='our')
parser.add_argument('--num_steps', type=int, default=2)
parse_args = parser.parse_args()


if not os.path.exists(parse_args.dirname):
    os.makedirs(parse_args.dirname)

# set up config

reg = parse_args.reg
lam = parse_args.lam
lam_fro = parse_args.lam_fro
lam_kin = parse_args.lam_kin
reg_type = parse_args.reg_type
lam_w = parse_args.lam_w
seed = parse_args.seed
rng = jax.random.PRNGKey(seed)
dirname = parse_args.dirname
count_nfe = not parse_args.no_count_nfe
vmap = parse_args.vmap
grid = False
if grid:
    _odeint = odeint_grid
    _odeint_aux2 = odeint_grid_sepaux_one   # finlay trick with 2 augmented states
    _odeint_aux3 = odeint_grid_sepaux       # finlay trick with 3 augmented states
    ode_kwargs = {
        "step_size": 1 / parse_args.num_steps
    }
else:
    _odeint = odeint
    _odeint_aux2 = odeint_sepaux
    _odeint_aux3 = odeint_fin_sepaux
    ode_kwargs = {
        "atol": parse_args.atol,
        "rtol": parse_args.rtol
    }


def _logaddexp(x1, x2):
  """
  Logaddexp while ignoring the custom_jvp rule.
  """
  amax = lax.max(x1, x2)
  delta = lax.sub(x1, x2)
  return lax.select(jnp.isnan(delta),
                    lax.add(x1, x2),  # NaNs or infinities of the same sign.
                    lax.add(amax, lax.log1p(lax.exp(-lax.abs(delta)))))


softplus = lambda x: _logaddexp(x, jnp.zeros_like(x))


def sol_recursive(f, z, t):
  """
  Recursively compute higher order derivatives of dynamics of ODE.
  """
  z_shape = z.shape
  z_t = jnp.concatenate((jnp.ravel(z), jnp.array([t])))

  def g(z_t):
    """
    Closure to expand z.
    """
    z, t = jnp.reshape(z_t[:-1], z_shape), z_t[-1]
    dz = jnp.ravel(f(z, t))
    dt = jnp.array([1.])
    dz_t = jnp.concatenate((dz, dt))
    return dz_t

  (y0, [y1h]) = jet(g, (z_t, ), ((jnp.ones_like(z_t), ), ))
  (y0, [y1, y2h]) = jet(g, (z_t, ), ((y0, y1h,), ))

  return (jnp.reshape(y0[:-1], z_shape), [jnp.reshape(y1[:-1], z_shape)])


# set up modules
class ConcatConv2D(hk.Module):
    """
    Convolution with extra channel and skip connection for time.
    """

    def __init__(self, *args, **kwargs):
        super(ConcatConv2D, self).__init__()
        self._layer = hk.Conv2D(*args, **kwargs)

    def __call__(self, x, t):
        tt = jnp.ones_like(x[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, x], axis=-1)
        return self._layer(ttx)


def logdetgrad(x, alpha):
    """
    Log determinant grad of the logit function for propagating logpx.
    """
    s = alpha + (1 - 2 * alpha) * x
    logdetgrad_ = -jnp.log(s - s * s) + jnp.log(1 - 2 * alpha)
    return jnp.sum(jnp.reshape(logdetgrad_, (x.shape[0], -1)), axis=1, keepdims=True)


def get_epsilon(key, shape):
    """
    Sample epsilon from the desired distribution.
    """
    # rademacher
    return jax.random.randint(key, shape, minval=0, maxval=2).astype(jnp.float32) * 2 - 1


class ForwardPreODE(hk.Module):
    """
    Module applied before the ODE layer.
    """

    def __init__(self, alpha=1e-5):
        super(ForwardPreODE, self).__init__()
        self.alpha = alpha

    def __call__(self, x, logpx):
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = jnp.log(s) - jnp.log(1 - s)
        logpy = logpx - logdetgrad(x, self.alpha)
        return y, logpy


class ReversePreODE(hk.Module):
    """
    Inverse of module applied before the ODE layer.
    """

    def __init__(self, alpha=1e-6):
        super(ReversePreODE, self).__init__()
        self.alpha = alpha

    def __call__(self, y, logpy):
        x = (sigmoid(y) - self.alpha) / (1 - 2 * self.alpha)
        logpx = logpy + logdetgrad(y, self.alpha)
        return x, logpx


class NN_Dynamics(hk.Module):
    """
    NN_Dynamics of the ODENet.
    """

    def __init__(self,
                 hidden_dims=(64, 64, 64),
                 input_shape=(28, 28, 1),
                 strides=(1, 1, 1, 1)):
        super(NN_Dynamics, self).__init__()
        self.input_shape = input_shape
        layers = []
        activation_fns = []
        base_layer = ConcatConv2D
        nonlinearity = softplus

        for layer_num, (dim_out, stride) in enumerate(zip(hidden_dims + (input_shape[-1],), strides)):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"kernel_shape": 3, "stride": 1, "padding": lambda _: (1, 1)}
            elif stride == 2:
                layer_kwargs = {"kernel_shape": 4, "stride": 2, "padding": lambda _: (1, 1)}
            elif stride == -2:
                # note: would need to use convtranspose instead here
                layer_kwargs = {"kernel_shape": 4, "stride": 2, "padding": lambda _: (1, 1), "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            if layer_num == len(strides) - 1:
                layer_kwargs["w_init"] = jnp.zeros  # initialize last layer to zeros (bias is always init to 0)
            layer = base_layer(dim_out, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(nonlinearity)

        self.layers = layers
        self.activation_fns = activation_fns[:-1]

    def __call__(self, x, t):
        x = jnp.reshape(x, (-1, *self.input_shape))
        dx = x
        for l, layer in enumerate(self.layers):
            dx = layer(dx, t)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx


def wrap_module(module, *module_args, **module_kwargs):
    """
    Wrap the module in a function to be transformed.
    """
    def wrap(*args, **kwargs):
        """
        Wrapping of module.
        """
        model = module(*module_args, **module_kwargs)
        return model(*args, **kwargs)
    return wrap


def initialization_data(input_shape):
    """
    Data for initializing the modules.
    """
    input_shape = (parse_args.test_batch_size, ) + input_shape[1:]
    data = {
        "pre_ode": aug_init(jnp.zeros(input_shape))[:2],        # (z, delta_logp)
        "ode": aug_init(jnp.zeros(input_shape))[:1] + (0., )    # (z, t)
    }
    return data


def init_model():
    """
    Instantiates transformed submodules of model and their parameters.
    """
    ts = jnp.array([0., 1.])

    input_shape = (-1, 28, 28, 1)

    initialization_data_ = initialization_data(input_shape)

    pre_ode = hk.without_apply_rng(hk.transform(wrap_module(ForwardPreODE)))
    pre_ode_params = pre_ode.init(rng, *initialization_data_["pre_ode"])
    pre_ode_fn = pre_ode.apply

    dynamics = hk.without_apply_rng(hk.transform(wrap_module(NN_Dynamics)))
    dynamics_params = dynamics.init(rng, *initialization_data_["ode"])
    dynamics_wrap = lambda x, t, params: dynamics.apply(params, x, t)

    def reg_dynamics(y, t, params):
        """
        NN_Dynamics of regularization for ODE integration.
        """
        if reg == "none":
            y = jnp.reshape(y, input_shape)
            return jnp.zeros((y.shape[0], 1))
        else:
            # do r3 regularization
            y0, y_n = sol_recursive(lambda _y, _t: dynamics_wrap(_y, _t, params), y, t)
            r = y_n[-1]
            return jnp.mean(jnp.square(r), axis=[axis_ for axis_ in range(1, r.ndim)])

    def ffjord_dynamics(yp, t, eps, params):
        """
        Dynamics of augmented ffjord state.
        """
        y, p = yp
        f = lambda y: dynamics_wrap(y, t, params)
        dy, eps_dy = jax.jvp(f, (y,), (eps,))
        div = jnp.sum(jnp.reshape(eps_dy * eps, (y.shape[0], -1)), axis=1, keepdims=True)
        return dy, -div

    def ffjord2_dynamics(yp, t, eps, params):
        """
        Dynamics of augmented ffjord state.
        """
        y, p = yp
        f = lambda y: dynamics_wrap(y, t, params)
        dy, eps_dy = jax.jvp(f, (y,), (eps,))
        div = jnp.sum(jnp.reshape(eps_dy * eps, (y.shape[0], -1)), axis=1, keepdims=True)
        return dy, -div, eps_dy

    def aug_dynamics(ypr, t, eps, params):
        """
        NN_Dynamics augmented with logp and regularization.
        """
        y, p, *_ = ypr

        dy, dp, eps_dy = ffjord2_dynamics((y, p), t, eps, params)

        if reg_type == "our":
            dr = reg_dynamics(y, t, params)
            return dy, dp, dr
        else:
            dfro = jnp.mean(jnp.square(eps_dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
            dkin = jnp.mean(jnp.square(dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
            return dy, dp, dfro, dkin

    def all_aug_dynamics(ypr, t, eps, params):
        """
        NN_Dynamics augmented with logp and regularization.
        """
        y, p, *_ = ypr

        dy, dp, eps_dy = ffjord2_dynamics((y, p), t, eps, params)

        dr = reg_dynamics(y, t, params)
        dfro = jnp.mean(jnp.square(eps_dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
        dkin = jnp.mean(jnp.square(dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
        return dy, dp, dr, dfro, dkin

    if reg_type == 'our':
        _odeint_aux = _odeint_aux2
    else:
        _odeint_aux = _odeint_aux3
    nodeint_aux = lambda y0, ts, eps, params: _odeint_aux(lambda y, t, eps, params: dynamics_wrap(y, t, params),
                                                          aug_dynamics, y0, ts, eps, params, **ode_kwargs)[0]
    all_nodeint = lambda y0, ts, eps, params: _odeint(all_aug_dynamics, y0, ts, eps, params, **ode_kwargs)[0]

    def ode_aux(params, y, delta_logp, eps):
        """
        Apply the ODE block.
        """
        ys, delta_logps, *rs = nodeint_aux(reg_init(y, delta_logp), ts, eps, params)
        return (ys[-1], delta_logps[-1], *(rs_[-1] for rs_ in rs))

    def all_ode(params, y, delta_logp, eps):
        """
        Apply the ODE block.
        """
        ys, delta_logps, *rs = all_nodeint(all_reg_init(y, delta_logp), ts, eps, params)
        return (ys[-1], delta_logps[-1], *(rs_[-1] for rs_ in rs))

    if count_nfe:
        # TODO: w/ finlay trick this is not true NFE
        if vmap:
            unreg_nodeint = jax.vmap(lambda z, delta_logp, t, eps, params:
                                     _odeint(ffjord_dynamics, (z, delta_logp), t, eps, params, **ode_kwargs)[1],
                                     (0, 0, None, 0, None))
        else:
            unreg_nodeint = lambda z, delta_logp, t, eps, params: \
                _odeint(ffjord_dynamics, (z, delta_logp), t, eps, params, **ode_kwargs)[1]

        @jax.jit
        def nfe_fn(key, params, _images):
            """
            Function to return NFE.
            """
            eps = get_epsilon(key, _images.shape)

            z, detla_logp = pre_ode_fn(params["pre_ode"], *aug_init(_images)[:2])
            f_nfe = unreg_nodeint(z, detla_logp, ts, eps, params["ode"])
            return jnp.mean(f_nfe)

    else:
        nfe_fn = None

    def forward_aux(key, params, _images):
        """
        Forward pass of the model.
        """
        eps = get_epsilon(key, _images.shape)

        z, detla_logp = pre_ode_fn(params["pre_ode"], *aug_init(_images)[:2])
        return ode_aux(params["ode"], z, detla_logp, eps)

    def forward_all(key, params, _images):
        """
        Forward pass of the model.
        """
        eps = get_epsilon(key, _images.shape)

        z, detla_logp = pre_ode_fn(params["pre_ode"], *aug_init(_images)[:2])
        return all_ode(params["ode"], z, detla_logp, eps)

    model = {"model": {
        "pre_ode": pre_ode_fn,
        "ode": all_ode
    }, "params": {
        "pre_ode": pre_ode_params,
        "ode": dynamics_params
    }, "nfe": nfe_fn,
        "forward_all": forward_all
    }

    return forward_aux, model


def aug_init(y):
    """
    Initialize dynamics with 0 for logpx and regs.
    """
    batch_size = y.shape[0]
    if reg_type == "our":
        return y, jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1))
    else:
        return y, jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1))


def reg_init(y, delta_logp):
    """
    Initialize dynamics with 0 for and regs.
    """
    batch_size = y.shape[0]
    if reg_type == "our":
        return y, delta_logp, jnp.zeros((batch_size, 1))
    else:
        return y, delta_logp, jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1))


def all_reg_init(y, delta_logp):
    """
    Initialize dynamics with 0 for and regs.
    """
    batch_size = y.shape[0]
    return y, delta_logp, jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1))


def _acc_fn(logits, labels):
    """
    Classification accuracy of the model.
    """
    predicted_class = jnp.argmax(logits, axis=1)
    return jnp.mean(predicted_class == labels)


def standard_normal_logprob(z):
    """
    Log probability of standard normal.
    """
    logz = -0.5 * jnp.log(2 * jnp.pi)
    return logz - jnp.square(z) / 2


def _loss_fn(z, delta_logp):
    logpz = jnp.sum(jnp.reshape(standard_normal_logprob(z), (z.shape[0], -1)), axis=1, keepdims=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = jnp.sum(logpx) / z.size  # averaged over batches
    bits_per_dim = -(logpx_per_dim - jnp.log(256)) / jnp.log(2)

    return bits_per_dim


def _reg_loss_fn(reg):
    return jnp.mean(reg)


def _weight_fn(params):
    flat_params, _ = ravel_pytree(params)
    return 0.5 * jnp.sum(jnp.square(flat_params))


def loss_fn(forward, params, images, key):
    """
    The loss function for training.
    """
    if reg_type == "our":
        z, delta_logp, regs = forward(key, params, images)
        loss_ = _loss_fn(z, delta_logp)
        reg_ = _reg_loss_fn(regs)
        weight_ = _weight_fn(params)
        return loss_ + lam * reg_ + lam_w * weight_
    else:
        z, delta_logp, fro_regs, kin_regs = forward(key, params, images)
        loss_ = _loss_fn(z, delta_logp)
        fro_reg_ = _reg_loss_fn(fro_regs)
        kin_reg_ = _reg_loss_fn(kin_regs)
        weight_ = _weight_fn(params)
        return loss_ + lam_fro * fro_reg_ + lam_kin * kin_reg_ + lam_w * weight_


def init_data():
    """
    Initialize data.
    """
    (ds_train, ds_test), ds_info = tfds.load('mnist',
                                             split=['train', 'test'],
                                             shuffle_files=True,
                                             with_info=True,
                                             as_supervised=True,
                                             read_config=tfds.ReadConfig(shuffle_seed=parse_args.seed,
                                                                         try_autocache=False))

    num_train = ds_info.splits['train'].num_examples
    num_test = ds_info.splits['test'].num_examples

    # make sure all batches are the same size to minimize jit compilation cache
    assert num_train % parse_args.batch_size == 0
    num_batches = num_train // parse_args.batch_size
    assert num_test % parse_args.test_batch_size == 0
    num_test_batches = num_test // parse_args.test_batch_size

    # make sure we always save the model on the last iteration
    assert num_batches * parse_args.nepochs % parse_args.save_freq == 0

    ds_train = ds_train.cache().repeat().shuffle(1000, seed=seed).batch(parse_args.batch_size)
    ds_test_eval = ds_test.batch(parse_args.test_batch_size).repeat()

    ds_train, ds_test_eval = tfds.as_numpy(ds_train), tfds.as_numpy(ds_test_eval)

    meta = {
        "num_batches": num_batches,
        "num_test_batches": num_test_batches
    }

    return iter(ds_train), iter(ds_test_eval), meta


def run():
    """
    Run the experiment.
    """
    # init the model first so that jax gets enough GPU memory before TFDS
    forward, model = init_model()
    grad_fn = jax.grad(lambda *args: loss_fn(forward, *args))

    ds_train, ds_test_eval, meta = init_data()
    num_batches = meta["num_batches"]
    num_test_batches = meta["num_test_batches"]

    def lr_schedule(itr):
        """
        Learning rate schedule.
        Slowly warm-up with a small learning rate.
        """
        iter_frac = lax.min((itr.astype(jnp.float32) + 1.) / lax.max(parse_args.warmup_itrs, 1.), 1.)
        _epoch = itr // num_batches
        id = lambda x: x
        return lax.cond(_epoch < 80, parse_args.lr * iter_frac, id, parse_args.lr / 10, id)

    opt_init, opt_update, get_params = optimizers.adam(step_size=lr_schedule)
    unravel_opt = ravel_pytree(opt_init(model["params"]))[1]
    if os.path.exists(parse_args.ckpt_path):
        outfile = open(parse_args.ckpt_path, 'rb')
        state_dict = pickle.load(outfile)
        outfile.close()

        opt_state = unravel_opt(state_dict["opt_state"])

        load_itr = state_dict["itr"]
    else:
        init_params = model["params"]
        opt_state = opt_init(init_params)

        load_itr = 0

    @jax.jit
    def update(_itr, _opt_state, _key, _batch):
        """
        Update the params based on grad for current batch.
        """
        grad_ = jax.experimental.optimizers.clip_grads(grad_fn(get_params(_opt_state), _batch, _key),
                                                       parse_args.max_grad_norm)
        return opt_update(_itr, grad_, _opt_state)

    @jax.jit
    def sep_losses(_opt_state, _batch, _key):
        """
        Convenience function for calculating losses separately.
        """
        z, delta_logp, r2_regs, fro_regs, kin_regs = model["forward_all"](_key, get_params(_opt_state), _batch)
        loss_ = _loss_fn(z, delta_logp)
        r2_reg_ = _reg_loss_fn(r2_regs)
        fro_reg_ = _reg_loss_fn(fro_regs)
        kin_reg_ = _reg_loss_fn(kin_regs)
        total_loss_ = loss_ + lam * r2_reg_ + lam_fro * fro_reg_ + lam_kin * kin_reg_
        return total_loss_, loss_, r2_reg_, fro_reg_, kin_reg_

    def evaluate_loss(opt_state, _key, ds_eval):
        """
        Convenience function for evaluating loss over train set in smaller batches.
        """
        sep_loss_aug_, sep_loss_, sep_loss_r2_reg_, sep_loss_fro_reg_, sep_loss_kin_reg_, nfe = [], [], [], [], [], []

        for test_batch_num in range(num_test_batches):
            _key, _key2 = jax.random.split(_key, num=2)
            test_batch = next(ds_eval)[0]
            test_batch = (test_batch.astype(jnp.float32) + jax.random.uniform(_key2,
                                                                              minval=1e-15,
                                                                              maxval=1 - 1e-15,
                                                                              shape=test_batch.shape)) / 256.

            test_batch_loss_aug_, test_batch_loss_, \
            test_batch_loss_r2_reg_, test_batch_loss_fro_reg_, test_batch_loss_kin_reg_ = \
                sep_losses(opt_state, test_batch, _key)

            if count_nfe:
                nfe.append(model["nfe"](_key, get_params(opt_state), test_batch))
            else:
                nfe.append(0)

            sep_loss_aug_.append(test_batch_loss_aug_)
            sep_loss_.append(test_batch_loss_)
            sep_loss_r2_reg_.append(test_batch_loss_r2_reg_)
            sep_loss_fro_reg_.append(test_batch_loss_fro_reg_)
            sep_loss_kin_reg_.append(test_batch_loss_kin_reg_)

        sep_loss_aug_ = jnp.array(sep_loss_aug_)
        sep_loss_ = jnp.array(sep_loss_)
        sep_loss_r2_reg_ = jnp.array(sep_loss_r2_reg_)
        sep_loss_fro_reg_ = jnp.array(sep_loss_fro_reg_)
        sep_loss_kin_reg_ = jnp.array(sep_loss_kin_reg_)
        nfe = jnp.array(nfe)

        return jnp.mean(sep_loss_aug_), jnp.mean(sep_loss_),\
               jnp.mean(sep_loss_r2_reg_), jnp.mean(sep_loss_fro_reg_), jnp.mean(sep_loss_kin_reg_), jnp.mean(nfe)

    itr = 0
    info = collections.defaultdict(dict)

    key = rng

    for epoch in range(parse_args.nepochs):
        for i in range(num_batches):
            key, key2 = jax.random.split(key, num=2)
            batch = next(ds_train)[0]
            batch = (batch.astype(jnp.float32) + jax.random.uniform(key2,
                                                                    minval=1e-15,
                                                                    maxval=1 - 1e-15,
                                                                    shape=batch.shape)) / 256.

            itr += 1

            if itr <= load_itr:
                continue

            update_start = time.time()
            opt_state = update(itr, opt_state, key, batch)
            tree_flatten(opt_state)[0][0].block_until_ready()
            update_end = time.time()
            time_str = "%d %.18f %d\n" % (itr, update_end - update_start, load_itr)
            outfile = open("%s/reg_%s_%s_lam_%.18e_lam_fro_%.18e_lam_kin_%.18e_time.txt"
                           % (dirname, reg, reg_type, lam, lam_fro, lam_kin), "a")
            outfile.write(time_str)
            outfile.close()

            if itr % parse_args.test_freq == 0:

                loss_aug_, loss_, loss_r2_reg_, loss_fro_reg_, loss_kin_reg_, nfe_ = \
                    evaluate_loss(opt_state, key, ds_test_eval)

                print_str = 'Iter {:04d} | Total (Regularized) Loss {:.6f} | Loss {:.6f} | ' \
                            'r {:.6f} | fro {:.6f} | kin {:.6f} | ' \
                            'NFE {:.6f}'.format(itr, loss_aug_, loss_, loss_r2_reg_, loss_fro_reg_, loss_kin_reg_, nfe_)

                print(print_str)

                outfile = open("%s/reg_%s_%s_lam_%.18e_lam_fro_%.18e_lam_kin_%.18e_info.txt"
                               % (dirname, reg, reg_type, lam, lam_fro, lam_kin), "a")
                outfile.write(print_str + "\n")
                outfile.close()

                info[itr]["loss_aug"] = loss_aug_
                info[itr]["loss"] = loss_
                info[itr]["loss_r2_reg"] = loss_r2_reg_
                info[itr]["loss_fro_reg"] = loss_fro_reg_
                info[itr]["loss_kin_reg"] = loss_kin_reg_
                info[itr]["nfe"] = nfe_

            if itr % parse_args.save_freq == 0:
                param_filename = "%s/reg_%s_%s_lam_%.18e_lam_fro_%.18e_lam_kin_%.18e_%d_fargs.pickle" \
                                 % (dirname, reg, reg_type, lam, lam_fro, lam_kin, itr)
                fargs = get_params(opt_state)
                outfile = open(param_filename, "wb")
                pickle.dump(fargs, outfile)
                outfile.close()

                state_dict = {
                    "opt_state": ravel_pytree(opt_state)[0],
                    "itr": itr,
                }
                param_filename = "%s/reg_%s_%s_lam_%.18e_lam_fro_%.18e_lam_kin_%.18e_%d_fargs_ckpt.pickle" \
                                 % (dirname, reg, reg_type, lam, lam_fro, lam_kin, itr)
                outfile = open(param_filename, "wb")
                pickle.dump(state_dict, outfile)
                outfile.close()

            if itr % parse_args.ckpt_freq == 0:
                state_dict = {
                    "opt_state": ravel_pytree(opt_state)[0],
                    "itr": itr,
                }
                # only save ckpts if a directory has been made for them (allow easy switching between v1 and v2)
                try:
                    outfile = open(parse_args.ckpt_path, 'wb')
                    pickle.dump(state_dict, outfile)
                    outfile.close()
                except IOError:
                    print("Unable to save ck.pt %d" % itr, file=sys.stderr)
    meta = {
        "info": info,
        "args": parse_args
    }
    outfile = open("%s/reg_%s_%s_lam_%.18e_lam_fro_%.18e_lam_kin_%.18e_%d_meta.pickle"
                   % (dirname, reg, reg_type, lam, lam_fro, lam_kin, itr), "wb")
    pickle.dump(meta, outfile)
    outfile.close()


if __name__ == "__main__":
    run()
