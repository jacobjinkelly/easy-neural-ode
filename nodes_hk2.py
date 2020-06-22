"""
Neural ODEs on MNIST with no downsampling before ODE, implemented with Haiku.
"""
import argparse
import collections
import os
import pickle
import sys
import time
from glob import glob

import haiku as hk
import tensorflow_datasets as tfds
from jax.tree_util import tree_flatten

import jax
from jax import lax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.experimental import optimizers
from jax.experimental.ode import \
    odeint, odeint_aux_one, odeint_sepaux, ravel_first_arg, odeint_grid, odeint_grid_sepaux_one, odeint_grid_aux
from jax.experimental.jet import jet

float64 = False
from jax.config import config
config.update("jax_enable_x64", float64)

REGS = ["r2", "r3", "r4", "r5", "r6"]

parser = argparse.ArgumentParser('Neural ODE')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--lam_w', type=float, default=0)
parser.add_argument('--atol', type=float, default=1.4e-8)
parser.add_argument('--rtol', type=float, default=1.4e-8)
parser.add_argument('--method', type=str, default="dopri5")
parser.add_argument('--no_vmap', action="store_true")
parser.add_argument('--init_step', type=float, default=1.)
parser.add_argument('--reg', type=str, choices=['none'] + REGS, default='none')
parser.add_argument('--reg_result', type=str, choices=['none'] + REGS, default=None)  # TODO: for plotting
parser.add_argument('--test_freq', type=int, default=3000)
parser.add_argument('--save_freq', type=int, default=3000)
parser.add_argument('--dirname', type=str, default='tmp')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resnet', action="store_true")
parser.add_argument('--no_count_nfe', action="store_true")
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--load_ckpt', type=str, default=None)
parser.add_argument('--ckpt_path', type=str, default="./ck.pt")
parser.add_argument('--lam_fro', type=float, default=0)
parser.add_argument('--lam_kin', type=float, default=0)
parser.add_argument('--reg_type', type=str, choices=['our', 'fin'], default='our')
parser.add_argument('--num_steps', type=int, default=2)
parser.add_argument('--eval', action="store_true")
parser.add_argument('--eval_dir', type=str)

parse_args = parser.parse_args()


assert os.path.exists(parse_args.dirname)

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
odenet = False if parse_args.resnet is True else True
count_nfe = False if parse_args.no_count_nfe or (not odenet) is True else True
vmap = False if parse_args.no_vmap is True else True
vmap = False
num_blocks = parse_args.num_blocks
grid = False
if grid:
    all_odeint = odeint_grid
    odeint_aux1 = odeint_grid_aux           # finlay trick w/ 1 augmented state
    odeint_aux2 = odeint_grid_sepaux_one    # odeint_grid_sepaux_onefinlay trick w/ 2 augmented states
    ode_kwargs = {
        "step_size": 1 / parse_args.num_steps
    }
else:
    all_odeint = odeint
    odeint_aux1 = odeint_aux_one
    odeint_aux2 = odeint_sepaux
    ode_kwargs = {
        "atol": parse_args.atol,
        "rtol": parse_args.rtol
    }


# some primitive functions
def sigmoid(z):
  """
  Numerically stable sigmoid.
  """
  return 1/(1 + jnp.exp(-z))


def softmax_cross_entropy(logits, labels):
  """
  Cross-entropy loss applied to softmax.
  """
  one_hot = hk.one_hot(labels, logits.shape[-1])
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def sol_recursive(f, z, t):
  """
  Recursively compute higher order derivatives of dynamics of ODE.
  """
  # TODO: flatten z and concat t once?
  # z_t = jnp.concatenate((z, jnp.repeat(jnp.array([[t]]), z.shape[0], axis=0)), axis=1)
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
  # (y0, [y1, y2, y3h]) = jet(g, (z_t, ), ((y0, y1, y2h), ))
  # (y0, [y1, y2, y3, y4h]) = jet(g, (z_t, ), ((y0, y1, y2, y3h), ))
  # (y0, [y1, y2, y3, y4, y5h]) = jet(g, (z_t, ), ((y0, y1, y2, y3, y4h), ))
  # (y0, [y1, y2, y3, y4, y5, y6h]) = jet(g, (z_t, ), ((y0, y1, y2, y3, y4, y5h), ))

  # TODO: shape this correctly! this will fail silently
  return (jnp.reshape(y0[:-1], z_shape), [jnp.reshape(y1[:-1], z_shape)])
                                          # jnp.reshape(y2[:-1], z_shape),
                                          # jnp.reshape(y3[:-1], z_shape),
                                          # jnp.reshape(y4[:-1], z_shape),
                                          # jnp.reshape(y5[:-1], z_shape)])


# set up modules
class Flatten(hk.Module):
    """
    Flatten all dimensions except batch dimension.
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def __call__(self, x):
        return jnp.reshape(x, (x.shape[0], -1))


class ConcatConv2D(hk.Module):
    """
    Convolution with extra channel and skip connection for time.
    """

    def __init__(self, **kwargs):
        super(ConcatConv2D, self).__init__()
        self._layer = hk.Conv2D(**kwargs)

    def __call__(self, x, t):
        tt = jnp.ones_like(x[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, x], axis=-1)
        return self._layer(ttx)


def get_epsilon(key, shape):
    """
    Sample epsilon from the desired distribution.
    """
    # normal
    # return jax.random.normal(key, shape)
    # rademacher
    if float64:
        return jax.random.randint(key, shape, minval=0, maxval=2).astype(jnp.float64) * 2 - 1
    else:
        return jax.random.randint(key, shape, minval=0, maxval=2).astype(jnp.float32) * 2 - 1


class MLPBlock(hk.Module):
    """
    Standard ResBlock.
    """

    def __init__(self, input_shape):
        super(MLPBlock, self).__init__()
        self.input_shape = input_shape
        self.dim = jnp.prod(input_shape[1:])
        self.hidden_dim = 100
        self.lin1 = hk.Linear(self.hidden_dim)
        self.lin2 = hk.Linear(self.hidden_dim)
        self.lin3 = hk.Linear(self.dim)

    def __call__(self, x):
        # vmapping means x will be a single batch element, so need to expand dims at 0
        x = jnp.reshape(x, (-1, self.dim))

        out = sigmoid(x)
        out = self.lin1(out)
        out = sigmoid(out)
        out = self.lin2(out)
        out = sigmoid(out)
        out = self.lin3(out)

        return out


class PreODE(hk.Module):
    """
    Module applied before the ODE layer.
    """

    def __init__(self):
        super(PreODE, self).__init__()
        # self.model = hk.Sequential([
        #     lambda x: x.astype(jnp.float32) / 255.,
        #     hk.Conv2D(output_channels=64,
        #               kernel_shape=3,
        #               stride=1,
        #               padding="VALID"),
        #     sigmoid,
        #     hk.Conv2D(output_channels=64,
        #               kernel_shape=4,
        #               stride=2,
        #               padding=lambda _: (1, 1)),
        #     sigmoid,
        #     hk.Conv2D(output_channels=64,
        #               kernel_shape=4,
        #               stride=2,
        #               padding=lambda _: (1, 1))
        # ])
        if float64:
            self.model = hk.Sequential([
                lambda x: x.astype(jnp.float64) / 255.,
                Flatten()
            ])
        else:
            self.model = hk.Sequential([
                lambda x: x.astype(jnp.float32) / 255.,
                Flatten()
            ])

    def __call__(self, x):
        return self.model(x)


class Dynamics(hk.Module):
    """
    Dynamics of the ODENet.
    """

    def __init__(self, input_shape):
        super(Dynamics, self).__init__()
        self.input_shape = input_shape
        output_channels = input_shape[-1]
        self.conv1 = ConcatConv2D(output_channels=output_channels,
                                  kernel_shape=3,
                                  stride=1,
                                  padding=lambda _: (1, 1))
        self.conv2 = ConcatConv2D(output_channels=output_channels,
                                  kernel_shape=3,
                                  stride=1,
                                  padding=lambda _: (1, 1),
                                  w_init=jnp.zeros,
                                  b_init=jnp.zeros)

    def __call__(self, x, t):
        # vmapping means x will be a single batch element, so need to expand dims at 0
        x = jnp.reshape(x, self.input_shape)

        out = sigmoid(x)
        out = self.conv1(out, t)
        out = sigmoid(out)
        out = self.conv2(out, t)

        return out


class MLPDynamics(hk.Module):
    """
    Dynamics for ODE as an MLP.
    """

    def __init__(self, input_shape):
        super(MLPDynamics, self).__init__()
        self.input_shape = input_shape
        self.dim = jnp.prod(input_shape[1:])
        self.hidden_dim = 100
        self.lin1 = hk.Linear(self.hidden_dim)
        self.lin2 = hk.Linear(self.dim)

    def __call__(self, x, t):
        # vmapping means x will be a single batch element, so need to expand dims at 0
        x = jnp.reshape(x, (-1, self.dim))

        out = sigmoid(x)
        tt = jnp.ones_like(x[:, :1]) * t
        t_out = jnp.concatenate([tt, out], axis=-1)
        out = self.lin1(t_out)

        out = sigmoid(out)
        tt = jnp.ones_like(out[:, :1]) * t
        t_out = jnp.concatenate([tt, out], axis=-1)
        out = self.lin2(t_out)

        return out


class PostODE(hk.Module):
    """
    Module applied after the ODE layer.
    """

    def __init__(self):
        super(PostODE, self).__init__()
        self.model = hk.Sequential([
            sigmoid,
            # hk.AvgPool(window_shape=(1, 6, 6, 1),
            #            strides=(1, 1, 1, 1),
            #            padding="VALID"),
            # Flatten(),
            hk.Linear(10)
        ])

    def __call__(self, x):
        return self.model(x)


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


def initialization_data(input_shape, ode_shape):
    """
    Data for initializing the modules.
    """
    ode_shape = (1, ) + ode_shape[1:]
    ode_dim = jnp.prod(ode_shape)
    data = {
        "pre_ode": jnp.zeros(input_shape),
        "ode": (jnp.zeros(ode_dim), 0.),
        "res": jnp.zeros(ode_shape),
        "post_ode": jnp.zeros(ode_dim)
    }
    return data


def init_model(model_reg=None):
    """
    Instantiates transformed submodules of model and their parameters.
    """
    ts = jnp.array([0., 1.])

    input_shape = (1, 28, 28, 1)
    ode_shape = (-1, 28, 28, 1)
    ode_dim = jnp.prod(ode_shape[1:])

    initialization_data_ = initialization_data(input_shape, ode_shape)

    if odenet:
        pre_ode = hk.transform(wrap_module(PreODE))
        pre_ode_params = pre_ode.init(rng, initialization_data_["pre_ode"])
        pre_ode_fn = pre_ode.apply
    else:
        pre_ode = hk.transform(wrap_module(PreODE))
        pre_ode_params = pre_ode.init(rng, initialization_data_["pre_ode"])
        pre_ode_fn = pre_ode.apply

    if odenet:
        dynamics = hk.transform(wrap_module(MLPDynamics, ode_shape))
        dynamics_params = dynamics.init(rng, *initialization_data_["ode"])
        dynamics_wrap = lambda x, t, params: dynamics.apply(params, x, t)
        def reg_dynamics(y, t, params):
            """
            Dynamics of regularization for ODE integration.
            """
            if reg == "none":
                dydt = dynamics_wrap(y, t, params)
                y = jnp.reshape(y, (-1, ode_dim))
                return dydt, jnp.zeros(y.shape[0])
            else:
                # do r3 regularization
                y0, y_n = sol_recursive(lambda _y, _t: dynamics_wrap(_y, _t, params), y, t)
                if model_reg is None:
                    r = y_n[-1]
                else:
                    r = y_n[REGS.index(model_reg)]
                return y0, jnp.mean(jnp.square(r), axis=[axis_ for axis_ in range(1, r.ndim)])

        def fin_dynamics(y, t, eps, params):
            """
            Dynamics of finlay reg.
            """
            f = lambda y: dynamics_wrap(y, t, params)
            dy, eps_dy = jax.jvp(f, (y,), (eps,))
            return dy, eps_dy

        def aug_dynamics(yr, t, eps, params):
            """
            Dynamics augmented with regularization.
            """
            y, *_ = yr
            if reg_type == "our":
                res = reg_dynamics(y, t, params)
                return reg_dynamics(y, t, params)
            else:
                dy, eps_dy = fin_dynamics(y, t, eps, params)
                dfro = jnp.mean(jnp.square(eps_dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
                dkin = jnp.mean(jnp.square(dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
                return dy, dfro, dkin

        def all_aug_dynamics(yr, t, eps, params):
            """
            Dynamics augmented with all regularizations for tracking.
            """
            y, *_ = yr
            dy, eps_dy = fin_dynamics(y, t, eps, params)
            _, drdt = reg_dynamics(y, t, params)
            dfro = jnp.mean(jnp.square(eps_dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
            dkin = jnp.mean(jnp.square(dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
            return dy, drdt, dfro, dkin

        if reg_type == "our":
            _odeint = odeint_aux1
        else:
            _odeint = odeint_aux2
        nodeint_aux = lambda y0, ts, eps, params: \
            _odeint(lambda y, t, eps, params: dynamics_wrap(y, t, params),
                    aug_dynamics, y0, ts, eps, params, **ode_kwargs)[0]
        all_nodeint = lambda y0, ts, eps, params: all_odeint(all_aug_dynamics,
                                                              y0, ts, eps, params, **ode_kwargs)[0]

        def ode(params, out_pre_ode, eps):
            """
            Apply the ODE block.
            """
            out_ode, *out_ode_rs = nodeint_aux(aug_init(out_pre_ode), ts, eps, params)
            return (out_ode[-1], *(out_ode_r[-1] for out_ode_r in out_ode_rs))

        def all_ode(params, out_pre_ode, eps):
            """
            Apply ODE block for all regularizations.
            """
            out_ode, *out_ode_rs = all_nodeint(all_aug_init(out_pre_ode), ts, eps, params)
            return (out_ode[-1], *(out_ode_r[-1] for out_ode_r in out_ode_rs))

        if count_nfe:
            if vmap:
                unreg_nodeint = jax.vmap(lambda y0, t, params: all_odeint(dynamics_wrap, y0, t, params, **ode_kwargs)[1],
                                         (0, None, None))
            else:
                unreg_nodeint = lambda y0, t, params: all_odeint(dynamics_wrap, y0, t, params, **ode_kwargs)[1]

            @jax.jit
            def nfe_fn(params, _images, _labels):
                """
                Function to return NFE.
                """
                in_ode = pre_ode_fn(params["pre_ode"], _images)
                f_nfe = unreg_nodeint(in_ode, ts, params["ode"])
                return jnp.mean(f_nfe)

            def plot_nfe_fn(_method, params, _images, _labels):
                """
                Function to return NFE.
                """
                rtol = ode_kwargs["rtol"]
                atol = ode_kwargs["atol"]
                mxstep = jnp.inf
                in_ode = pre_ode_fn(params["pre_ode"], _images)
                def unreg_nodeint(y0, t, params):
                    y0, unravel = ravel_pytree(y0)
                    func = ravel_first_arg(dynamics_wrap, unravel)
                    return _method(func, rtol, atol, mxstep, y0, t, params)[1]
                f_nfe = jax.vmap(unreg_nodeint, (0, None, None))(in_ode, ts, params["ode"])
                return jnp.mean(f_nfe)

            def plot_nfe_per_ex_fn(_method, params, _images, _labels):
                """
                Function to return NFE.
                """
                rtol = ode_kwargs["rtol"]
                atol = ode_kwargs["atol"]
                mxstep = jnp.inf
                in_ode = pre_ode_fn(params["pre_ode"], _images)
                def unreg_nodeint(y0, t, params):
                    y0, unravel = ravel_pytree(y0)
                    func = ravel_first_arg(dynamics_wrap, unravel)
                    return _method(func, rtol, atol, mxstep, y0, t, params)
                out_ode, f_nfe = jax.vmap(unreg_nodeint, (0, None, None))(in_ode, ts, params["ode"])
                out_ode = out_ode[:, -1]
                logits = post_ode_fn(params["post_ode"], out_ode)
                loss_ = jax.vmap(_loss_fn, in_axes=(0, 0))(logits, _labels)
                acc_ = jax.vmap(_acc_fn, in_axes=(0, 0))(jnp.expand_dims(logits, axis=1), _labels)
                return loss_, acc_, f_nfe

        else:
            nfe_fn = None

    else:
        resnet = hk.transform(wrap_module(
            lambda: hk.Sequential([MLPBlock(ode_shape) for _ in range(num_blocks)])))
        resnet_params = resnet.init(rng, initialization_data_["res"])
        resnet_fn = resnet.apply

    post_ode = hk.transform(wrap_module(PostODE))
    post_ode_params = post_ode.init(rng, initialization_data_["post_ode"])
    post_ode_fn = post_ode.apply

    # return a dictionary of the three components of the model
    model = {
        "model": {
            "pre_ode": pre_ode_fn,
            "post_ode": post_ode_fn
        },
        "params": {
            "pre_ode": pre_ode_params,
            "post_ode": post_ode_params
        }
    }

    if odenet:
        model["model"]["ode"] = ode
        model["model"]["all_ode"] = all_ode
        model["params"]["ode"] = dynamics_params
        model["nfe"] = nfe_fn
        model["plot_nfe"] = plot_nfe_fn  # TODO: plot or nah?
        model["plot_nfe_per_ex"] = plot_nfe_per_ex_fn
    else:
        model["model"]["res"] = resnet_fn
        model["params"]["res"] = resnet_params

    def forward(key, params, _images):
        """
        Forward pass of the model.
        """
        model_ = model["model"]

        if odenet:
            out_pre_ode = model_["pre_ode"](params["pre_ode"], _images)
            out_ode, *regs = model_["ode"](params["ode"], out_pre_ode, get_epsilon(key, out_pre_ode.shape))
            logits = model_["post_ode"](params["post_ode"], out_ode)
        else:
            out_pre_ode = model_["pre_ode"](params["pre_ode"], _images)
            out_ode = model_["res"](params["res"], out_pre_ode)
            regs = jnp.zeros(_images.shape[0])
            logits = model_["post_ode"](params["post_ode"], out_ode)

        return (logits, *regs)

    def forward_all(key, params, _images):
        """
        Forward pass of the model.
        """
        model_ = model["model"]

        if odenet:
            out_pre_ode = model_["pre_ode"](params["pre_ode"], _images)
            out_ode, *regs = model_["all_ode"](params["ode"], out_pre_ode, get_epsilon(key, out_pre_ode.shape))
            logits = model_["post_ode"](params["post_ode"], out_ode)
        else:
            out_pre_ode = model_["pre_ode"](params["pre_ode"], _images)
            out_ode = model_["res"](params["res"], out_pre_ode)
            regs = jnp.zeros(_images.shape[0])
            logits = model_["post_ode"](params["post_ode"], out_ode)

        return (logits, *regs)

    model["model"]["forward_all"] = forward_all
    model["model"]["forward"] = forward

    return forward, model


def aug_init(y, batch_size=-1):
    """
    Flatten the dynamics and append regularization dynamics.
    We need to flatten the dynamics first since they may be convolutional
    (has width, height, and channels).
    """
    if batch_size == -1:
        batch_size = y.shape[0]
    if reg_type == "our":
        return y, jnp.zeros(batch_size)
    else:
        return y, jnp.zeros(batch_size), jnp.zeros(batch_size)


def all_aug_init(y, batch_size=-1):
    """
    Flatten the dynamics and append regularization dynamics.
    We need to flatten the dynamics first since they may be convolutional
    (has width, height, and channels).
    """
    if batch_size == -1:
        batch_size = y.shape[0]
    return y, jnp.zeros(batch_size), jnp.zeros(batch_size), jnp.zeros(batch_size)


def _acc_fn(logits, labels):
    """
    Classification accuracy of the model.
    """
    predicted_class = jnp.argmax(logits, axis=1)
    return jnp.mean(predicted_class == labels)


def _loss_fn(logits, labels):
    return jnp.mean(softmax_cross_entropy(logits, labels))


def _reg_loss_fn(reg):
    return jnp.mean(reg)


def _weight_fn(params):
    flat_params, _ = ravel_pytree(params)
    return 0.5 * jnp.sum(jnp.square(flat_params))


def loss_fn(forward, params, images, labels, key):
    """
    The loss function for training.
    """
    if reg_type == "our":
        logits, regs = forward(key, params, images)
        loss_ = _loss_fn(logits, labels)
        reg_ = _reg_loss_fn(regs)
        weight_ = _weight_fn(params)
        return loss_ + lam * reg_ + lam_w * weight_
    else:
        logits, fro_regs, kin_regs = forward(key, params, images)
        loss_ = _loss_fn(logits, labels)
        fro_reg_ = _reg_loss_fn(fro_regs)
        kin_reg_ = _reg_loss_fn(kin_regs)
        weight_ = _weight_fn(params)
        return loss_ + lam_fro * fro_reg_ + lam_kin * kin_reg_ + lam_w * weight_


def init_data():
    """
    Initialize data.
    """
    (ds_train,), ds_info = tfds.load('mnist',
                                     split=['train'],
                                     shuffle_files=True,
                                     as_supervised=True,
                                     with_info=True,
                                     read_config=tfds.ReadConfig(shuffle_seed=parse_args.seed))

    num_train = ds_info.splits['train'].num_examples

    assert num_train % parse_args.batch_size == 0
    num_batches = num_train // parse_args.batch_size

    test_batch_size = parse_args.test_batch_size if odenet else 10000
    assert num_train % test_batch_size == 0
    num_test_batches = num_train // test_batch_size

    # make sure we always save the model on the last iteration
    assert num_batches * parse_args.nepochs % parse_args.save_freq == 0

    ds_train = ds_train.cache()
    ds_train = ds_train.repeat()
    ds_train = ds_train.shuffle(1000, seed=seed)
    # ds_train = ds_train.batch(parse_args.batch_size)
    # ds_train = tfds.as_numpy(ds_train)
    # ds_train_eval = ds_test.batch(test_batch_size).repeat()
    # ds_train_eval = tfds.as_numpy(ds_train_eval)
    ds_train, ds_train_eval = ds_train.batch(parse_args.batch_size), ds_train.batch(test_batch_size)
    ds_train, ds_train_eval = tfds.as_numpy(ds_train), tfds.as_numpy(ds_train_eval)

    meta = {
        "num_batches": num_batches,
        "num_test_batches": num_test_batches
    }

    return ds_train, ds_train_eval, meta


def run():
    """
    Run the experiment.
    """
    print("Reg: %s\tLambda %.4e" % (reg, lam))
    print("Reg: %s\tLambda %.4e" % (reg, lam), file=sys.stderr)

    ds_train, ds_train_eval, meta = init_data()
    num_batches = meta["num_batches"]
    num_test_batches = meta["num_test_batches"]

    forward, model = init_model()
    forward_all = model["model"]["forward_all"]
    grad_fn = jax.grad(lambda *args: loss_fn(forward, *args))

    def lr_schedule(train_itr):
        _epoch = train_itr // num_batches
        id = lambda x: x
        return lax.cond(_epoch < 60, 1e-1, id, 0,
                        lambda _: lax.cond(_epoch < 100, 1e-2, id, 0,
                                           lambda _: lax.cond(_epoch < 140, 1e-3, id, 1e-4, id)))

    opt_init, opt_update, get_params = optimizers.momentum(step_size=lr_schedule, mass=0.9)
    if parse_args.load_ckpt:
        file_ = open(parse_args.load_ckpt, 'rb')
        init_params = pickle.load(file_)
        file_.close()

        # parse itr from the checkpoint
        load_itr = int(os.path.basename(parse_args.load_ckpt).split("_")[-2])
    else:
        init_params = model["params"]
        load_itr = 0
    opt_state = opt_init(init_params)

    @jax.jit
    def update(_itr, _opt_state, _key, _batch):
        """
        Update the params based on grad for current batch.
        """
        images, labels = _batch
        return opt_update(_itr, grad_fn(get_params(_opt_state), images, labels, _key), _opt_state)

    @jax.jit
    def sep_losses(_opt_state, _batch, key):
        """
        Convenience function for calculating losses separately.
        """
        params = get_params(_opt_state)
        images, labels = _batch
        logits, r2_regs, fro_regs, kin_regs = forward_all(key, params, images)
        loss_ = _loss_fn(logits, labels)
        r2_reg_ = _reg_loss_fn(r2_regs)
        fro_reg_ = _reg_loss_fn(fro_regs)
        kin_reg_ = _reg_loss_fn(kin_regs)
        total_loss_ = loss_ + lam * r2_reg_ + lam_fro * fro_reg_ + lam_kin * kin_reg_
        acc_ = _acc_fn(logits, labels)
        return acc_, total_loss_, loss_, r2_reg_, fro_reg_, kin_reg_

    def evaluate_loss(opt_state, _key, ds_train_eval):
        """
        Convenience function for evaluating loss over train set in smaller batches.
        """
        sep_acc_, sep_loss_aug_, sep_loss_, \
        sep_loss_r2_reg_, sep_loss_fro_reg_, sep_loss_kin_reg_, nfe = [], [], [], [], [], [], []

        for test_batch_num in range(num_test_batches):
            test_batch = next(ds_train_eval)
            _key, = jax.random.split(_key, num=1)

            test_batch_acc_, test_batch_loss_aug_, test_batch_loss_, \
            test_batch_loss_r2_reg_, test_batch_loss_fro_reg_, test_batch_loss_kin_reg_ = \
                sep_losses(opt_state, test_batch, _key)

            if count_nfe:
                nfe.append(model["nfe"](get_params(opt_state), *test_batch))
            else:
                nfe.append(0)

            sep_acc_.append(test_batch_acc_)
            sep_loss_aug_.append(test_batch_loss_aug_)
            sep_loss_.append(test_batch_loss_)
            sep_loss_r2_reg_.append(test_batch_loss_r2_reg_)
            sep_loss_fro_reg_.append(test_batch_loss_fro_reg_)
            sep_loss_kin_reg_.append(test_batch_loss_kin_reg_)

        sep_acc_ = jnp.array(sep_acc_)
        sep_loss_aug_ = jnp.array(sep_loss_aug_)
        sep_loss_ = jnp.array(sep_loss_)
        sep_loss_r2_reg_ = jnp.array(sep_loss_r2_reg_)
        sep_loss_fro_reg_ = jnp.array(sep_loss_fro_reg_)
        sep_loss_kin_reg_ = jnp.array(sep_loss_kin_reg_)
        nfe = jnp.array(nfe)

        return jnp.mean(sep_acc_), jnp.mean(sep_loss_aug_), jnp.mean(sep_loss_), \
               jnp.mean(sep_loss_r2_reg_), jnp.mean(sep_loss_fro_reg_), jnp.mean(sep_loss_kin_reg_), jnp.mean(nfe)

    itr = 0
    info = collections.defaultdict(dict)

    key = rng

    for epoch in range(parse_args.nepochs):
        for i in range(num_batches):
            batch = next(ds_train)

            key, = jax.random.split(key, num=1)

            itr += 1

            if parse_args.load_ckpt:
                if itr <= load_itr:
                    continue

            if not parse_args.eval:
                update_start = time.time()
                opt_state = update(itr, opt_state, key, batch)
                tree_flatten(opt_state)[0][0].block_until_ready()
                update_end = time.time()
                time_str = "%d %.18f %d\n" % (itr, update_end - update_start, load_itr)
                outfile = open("%s/reg_%s_%s_lam_%.18e_lam_fro_%.18e_lam_kin_%.18e_time.txt"
                               % (dirname, reg, reg_type, lam, lam_fro, lam_kin), "a")
                outfile.write(time_str)
                outfile.close()
            else:
                # go immediately to testing
                itr = 0

            if itr % parse_args.test_freq == 0:
                if parse_args.eval:
                    # find params in eval_dir
                    files = glob("%s/*96000_fargs.pickle" % parse_args.eval_dir)
                    if len(files) != 1:
                        print("Couldn't find param file!!")
                        print("Couldn't find param file!!", file=sys.stderr)
                        return
                    eval_pth = files[0]
                    eval_param_file = open(eval_pth, "rb")
                    eval_params = pickle.load(eval_param_file)
                    eval_param_file.close()
                    # shove them in opt_state
                    opt_state = opt_init(eval_params)

                acc_, loss_aug_, loss_, \
                loss_r2_reg_, loss_fro_reg_, loss_kin_reg_, nfe_ = evaluate_loss(opt_state, key, ds_train_eval)

                print_str = 'Iter {:04d} | Total (Regularized) Loss {:.6f} | Loss {:.6f} | ' \
                            'r {:.6f} | fro {:.6f} | kin {:.6f} | ' \
                            'NFE {:.6f}'.format(itr, loss_aug_, loss_, loss_r2_reg_, loss_fro_reg_, loss_kin_reg_, nfe_)

                print(print_str)

                if parse_args.eval:
                    outfile = open("%s/eval_info.txt" % parse_args.eval_dir, "w")
                    outfile.write(print_str + "\n")
                    outfile.close()
                    return

                outfile = open("%s/reg_%s_%s_lam_%.18e_lam_fro_%.18e_lam_kin_%.18e_info.txt"
                               % (dirname, reg, reg_type, lam, lam_fro, lam_kin), "a")
                outfile.write(print_str + "\n")
                outfile.close()

                info[itr]["acc"] = acc_
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

            outfile = open("%s/reg_%s_%s_lam_%.18e_lam_fro_%.18e_lam_kin_%.18e_iter.txt"
                           % (dirname, reg, reg_type, lam, lam_fro, lam_kin), "a")
            outfile.write("Iter: {:04d}\n".format(itr))
            outfile.close()
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
