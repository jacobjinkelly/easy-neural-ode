"""
Implementing Latent ODE on PhysioNet.
"""
import argparse
import collections
import os
import pickle
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax
from jax.config import config
from jax.experimental import optimizers
from jax.experimental.jet import jet
from jax.flatten_util import ravel_pytree

from lib.optimizers import exponential_decay
from lib.ode import odeint
from physionet_data import init_physionet_data

config.update("jax_enable_x64", True)

REGS = ["r2", "r3", "r4", "r5"]

parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--data_root', type=str, default="./")
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--lam_w', type=float, default=0)
parser.add_argument('--atol', type=float, default=1.4e-8)
parser.add_argument('--rtol', type=float, default=1.4e-8)
parser.add_argument('--init_step', type=float, default=1.)
parser.add_argument('--reg', type=str, choices=['none'] + REGS, default='none')
parser.add_argument('--test_freq', type=int, default=640)
parser.add_argument('--save_freq', type=int, default=640)
parser.add_argument('--dirname', type=str, default='tmp')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--no_count_nfe', action="store_true")
parse_args = parser.parse_args()


if not os.path.exists(parse_args.dirname):
    os.makedirs(parse_args.dirname)

# set up config

reg = parse_args.reg
lam = parse_args.lam
lam_rec = lam_gen = lam
lam_w = parse_args.lam_w
seed = parse_args.seed
rng = jax.random.PRNGKey(seed)
dirname = parse_args.dirname
count_nfe = not parse_args.no_count_nfe
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


def logsumexp(x, axis=-1):
    """
    Numerically stable logsumexp.
    """
    x_max = x.max(axis)
    return x_max + jnp.log(jnp.sum(jnp.exp(x - x_max), axis, keepdims=False))


def sol_recursive(f, z, t):
  """
  Recursively compute higher order derivatives of dynamics of ODE.
  """
  if reg == "none":
      return f(z, t), jnp.zeros_like(z)

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

  reg_ind = REGS.index(reg)

  (y0, [*yns]) = jet(g, (z_t, ), ((jnp.ones_like(z_t), ), ))
  for _ in range(reg_ind + 1):
      (y0, [*yns]) = jet(g, (z_t, ), ((y0, *yns), ))

  return (jnp.reshape(y0[:-1], z_shape), jnp.reshape(yns[-2][:-1], z_shape))


class LatentGRU(hk.Module):
    """
    Modified GRU unit to deal with latent state.
    """
    def __init__(self,
                 latent_dim,
                 n_units,
                 **init_kwargs):
        super(LatentGRU, self).__init__()
        self.latent_dim = latent_dim

        self.update_gate = hk.Sequential([
           hk.Linear(n_units, **init_kwargs),
           jnp.tanh,
           hk.Linear(latent_dim, **init_kwargs),
           sigmoid
        ])

        self.reset_gate = hk.Sequential([
           hk.Linear(n_units, **init_kwargs),
           jnp.tanh,
           hk.Linear(latent_dim, **init_kwargs),
           sigmoid
        ])

        self.new_state_net = hk.Sequential([
           hk.Linear(n_units, **init_kwargs),
           jnp.tanh,
           hk.Linear(latent_dim * 2, **init_kwargs)
        ])

    def __call__(self, y_mean, y_std, x):
        y_concat = jnp.concatenate([y_mean, y_std, x], axis=-1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = jnp.concatenate([y_mean * reset_gate, y_std * reset_gate, x], axis=-1)

        new_state = self.new_state_net(concat)
        new_state_mean, new_state_std = new_state[..., :self.latent_dim], new_state[..., self.latent_dim:]
        new_state_std = jnp.abs(new_state_std)

        new_y_mean = (1-update_gate) * new_state_mean + update_gate * y_mean
        new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

        # update the hidden state only if at least one feature is present for the current time point
        n_data_dims = x.shape[-1] // 2
        mask = x[n_data_dims:]
        mask = jnp.sum(mask, axis=-1, keepdims=True) > 0

        new_y_mean = mask * new_y_mean + (1-mask) * y_mean
        new_y_std = mask * new_y_std + (1-mask) * y_std

        new_y_std = jnp.abs(new_y_std)
        return new_y_mean, new_y_std


class RecDynamics(hk.Module):
    """
    ODE dynamics for encoder.
    """

    def __init__(self,
                 latent_dim,
                 layers,
                 units):
        super(RecDynamics, self).__init__()
        self.latent_dim = latent_dim
        self.model = hk.Sequential([unit for _ in range(layers + 1) for unit in
                                    [jnp.tanh, hk.Linear(units)]] +
                                   [jnp.tanh, hk.Linear(latent_dim)]
                                   )

    def __call__(self, y, t):
        # use time-independent dynamics
        # need to reshape so regularization knows what to take mean over
        y = jnp.reshape(y, (-1, self.latent_dim))
        return self.model(y)


class GenDynamics(hk.Module):
    """
    ODE dynamics for decoder.
    """

    def __init__(self,
                 latent_dim,
                 layers,
                 units):
        super(GenDynamics, self).__init__()
        self.latent_dim = latent_dim
        self.model = hk.Sequential([unit for _ in range(layers + 1) for unit in
                                    [jnp.tanh, hk.Linear(units)]] +
                                   [jnp.tanh, hk.Linear(latent_dim)]
                                   )

    def __call__(self, y, t):
        return self.model(y)


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


def initialization_data(rec_dim, gen_dim, data_dim):
    """
    Creates data for initializing each of the modules based on the shapes of init_data.
    """
    data = {
        "gru_rnn": (jnp.zeros(rec_dim), jnp.zeros(rec_dim), jnp.zeros((2 * data_dim + 1, ))),  # 2x+1 for mask and delta
        "rec_dynamics": (jnp.zeros(rec_dim), 0.),
        "rec_to_gen": (jnp.zeros(rec_dim), jnp.zeros(rec_dim)),
        "gen_dynamics": (jnp.zeros(gen_dim), 0.),
        "gen_to_data": jnp.zeros(gen_dim)
    }
    return data


def augment_dynamics(dynamics):
    """
    Closure to augment dynamics.
    """
    def reg_dynamics(y, t, params):
        """
        Dynamics of regularization.
        """
        y0, r = sol_recursive(lambda _y, _t: dynamics(_y, _t, params), y, t)
        return y0, jnp.mean(r ** 2)

    def aug_dynamics(yr, t, params):
        """
        Dynamics augmented with regularization.
        """
        y, r = yr
        dydt, drdt = reg_dynamics(y, t, params)
        return dydt, drdt
    return aug_dynamics


def init_model(gen_ode_kwargs,
               rec_dim=40,
               gen_dim=20,
               data_dim=37,
               gen_layers=3,
               dynamics_units=50,
               gru_units=50):
    """
    Instantiates transformed submodules of model and their parameters.
    """
    initialization_data_ = initialization_data(rec_dim,
                                               gen_dim,
                                               data_dim)

    init_kwargs = {
        "w_init": hk.initializers.RandomNormal(mean=0, stddev=0.1),
        "b_init": jnp.zeros
    }

    gru_rnn = hk.without_apply_rng(hk.transform(wrap_module(LatentGRU,
                                                            latent_dim=rec_dim,
                                                            n_units=gru_units,
                                                            **init_kwargs)))
    gru_rnn_params = gru_rnn.init(rng, *initialization_data_["gru_rnn"])

    # note: the ODE-RNN version uses double
    rec_to_gen = hk.without_apply_rng(hk.transform(wrap_module(lambda: hk.Sequential([
        lambda x, y: jnp.concatenate((x, y), axis=-1),
        hk.Linear(50, **init_kwargs),
        jnp.tanh,
        hk.Linear(2 * gen_dim, **init_kwargs)
    ]))))
    rec_to_gen_params = rec_to_gen.init(rng, *initialization_data_["rec_to_gen"])

    gen_dynamics = hk.without_apply_rng(hk.transform(wrap_module(GenDynamics,
                                                                 latent_dim=gen_dim,
                                                                 units=dynamics_units,
                                                                 layers=gen_layers)))
    gen_dynamics_params = gen_dynamics.init(rng, *initialization_data_["gen_dynamics"])
    gen_dynamics_wrap = lambda x, t, params: gen_dynamics.apply(params, x, t)

    gen_to_data = hk.without_apply_rng(hk.transform(wrap_module(hk.Linear,
                                                                output_size=data_dim,
                                                                **init_kwargs)))
    gen_to_data_params = gen_to_data.init(rng, initialization_data_["gen_to_data"])

    init_params = {
        "gru_rnn": gru_rnn_params,
        "rec_to_gen": rec_to_gen_params,
        "gen_dynamics": gen_dynamics_params,
        "gen_to_data": gen_to_data_params
    }

    def forward(count_nfe_, params, data, data_timesteps, timesteps, mask, num_samples=3):
        """
        Forward pass of the model.
        y are the latent variables of the recognition model
        z are the latent variables of the generative model
        """

        data_mask = jnp.concatenate((data, mask), axis=-1)

        # ode-rnn encoder
        final_y, final_y_std, rec_r, rec_nfes = \
            jax.vmap(rnn, in_axes=(None, 0, None))(params, data_mask, data_timesteps)

        # translate
        z0 = rec_to_gen.apply(params["rec_to_gen"], final_y, final_y_std)
        mean_z0, std_z0 = z0[..., :gen_dim], z0[..., gen_dim:]
        std_z0 = jnp.abs(std_z0)

        def sample_z0(key):
            """
            Sample generative latent variable using reparameterization trick.
            """
            return mean_z0 + std_z0 * jax.random.normal(key, shape=mean_z0.shape)
        z0 = jax.vmap(sample_z0)(jax.random.split(rng, num=num_samples))

        def integrate_sample(z0_):
            """
            Integrate one sample of z0 (for one batch).
            """
            if count_nfe_:
                dynamics = gen_dynamics_wrap
                init_fn = lambda x: x
            else:
                dynamics = augment_dynamics(gen_dynamics_wrap)
                init_fn = aug_init
            return jax.vmap(lambda z_, t_: odeint(dynamics, init_fn(z_), t_,
                                                  params["gen_dynamics"], **gen_ode_kwargs),
                            in_axes=(0, None))(z0_, timesteps)
        z_r, gen_nfes = jax.vmap(integrate_sample)(z0)
        if count_nfe_:
            z, gen_r = z_r, 0.
        else:
            z, gen_r = z_r
            gen_r = gen_r[:, :, -1]  # take only regularization at final time point

        # decode latent to data, vmapping over batch and timepoints
        pred = jax.vmap(jax.vmap(partial(gen_to_data.apply, params["gen_to_data"]), in_axes=1, out_axes=1))(z)

        z0_params = {
            "mean": mean_z0,
            "std": std_z0
        }

        nfe = {
            "rec": jnp.mean(jnp.sum(rec_nfes, axis=1)),
            "gen": jnp.mean(gen_nfes)
        }

        return z, pred, rec_r, gen_r, z0_params, nfe

    def rnn(params, data, timesteps):
        """
        ODE-RNN model.
        """
        # concatenate time_delta
        delta = jnp.expand_dims(jnp.concatenate((timesteps[1:] - timesteps[:-1], jnp.zeros(1))), axis=1)
        data_delta = jnp.concatenate((data, delta), axis=1)

        init_state = jnp.zeros(rec_dim), jnp.zeros(rec_dim)

        def scan_fun(prev_state, xi):
            yi_mean, yi_std = prev_state
            yi_mean, yi_std = gru_rnn.apply(params["gru_rnn"], yi_mean, yi_std, xi)
            return (yi_mean, yi_std), None
        (final_y_mean, final_y_std), _ = lax.scan(scan_fun, init_state, data_delta[::-1])

        return final_y_mean, final_y_std, 0., jnp.zeros(data.shape[0] - 1)

    model = {
        "forward": partial(forward, False),
        "params": init_params,
        "nfe": lambda *args: partial(forward, count_nfe)(*args)[-1]
    }

    return model


def aug_init(y):
    """
    Flatten the dynamics and append regularization dynamics.
    We need to flatten the dynamics first since they may be convolutional
    (has width, height, and channels).
    """
    return y, jnp.zeros(1)


def _likelihood(preds, data, mask):
    """
    Compute log-likelihood of data w/ mask under current predictions.
    """
    def sample_likelihood(data_, mu, std=0.01):
        """
        Log-Likelihood of one sample.
        """
        return -((data_ - mu) ** 2) / (2 * std ** 2) - jnp.log(std) - jnp.log(2 * jnp.pi) / 2
    return jnp.sum(sample_likelihood(data[None] * mask, preds * mask), axis=[1, 2, 3]) / jnp.sum(mask)


def _mse(preds, data, mask):
    """
    Return mean squared error w/ mask.
    """
    return jnp.mean(jnp.sum(jnp.square(data[None] * mask - preds * mask), axis=[1, 2, 3]) / jnp.sum(mask))


def _kl_div(params):
    """
    Analytically compute KL between z0 distribution and prior.
    """
    mean_prior = 0
    std_prior = 1
    var_ratio = (params["std"] / std_prior) ** 2
    t1 = ((params["mean"] - mean_prior) / std_prior) ** 2
    return jnp.mean(0.5 * (var_ratio + t1 - 1 - jnp.log(var_ratio)))


def _reg_loss_fn(reg):
    return jnp.mean(reg)


def _weight_fn(params):
    flat_params, _ = ravel_pytree(params)
    return 0.5 * jnp.sum(jnp.square(flat_params))


def loss_fn(forward, params, batch, kl_coef):
    """
    The loss function for training.
    """
    preds, rec_r, gen_r, z0_params, nfe = forward(params,
                                                  batch["observed_data"],
                                                  batch["observed_tp"],
                                                  batch["tp_to_predict"],
                                                  batch["observed_mask"])
    likelihood_ = _likelihood(preds, batch["observed_data"], batch["observed_mask"])
    kl_ = _kl_div(z0_params)
    rec_reg_ = _reg_loss_fn(rec_r)
    gen_reg_ = _reg_loss_fn(gen_r)
    return -logsumexp(likelihood_ - kl_coef * kl_, axis=0) + lam_rec * rec_reg_ + lam_gen * gen_reg_


def run():
    """
    Run the experiment.
    """

    ds_train, ds_test, meta = init_physionet_data(rng, parse_args)
    num_batches = meta["num_batches"]
    num_test_batches = meta["num_test_batches"]

    model = init_model(ode_kwargs)
    forward = lambda *args: model["forward"](*args)[1:]
    grad_fn = jax.grad(lambda *args: loss_fn(forward, *args))

    lr_schedule = exponential_decay(step_size=parse_args.lr,
                                    decay_steps=1,
                                    decay_rate=0.999,
                                    lowest=parse_args.lr / 10)
    opt_init, opt_update, get_params = optimizers.adamax(step_size=lr_schedule)
    opt_state = opt_init(model["params"])

    def get_kl_coef(epoch_):
        """
        Tuning schedule for KL coefficient. (annealing)
        """
        return max(0., 1 - 0.99 ** (epoch_ - 10))

    @jax.jit
    def update(_itr, _opt_state, _batch, kl_coef):
        """
        Update the params based on grad for current batch.
        """
        return opt_update(_itr, grad_fn(get_params(_opt_state), _batch, kl_coef), _opt_state)

    @jax.jit
    def sep_losses(_opt_state, _batch, kl_coef):
        """
        Convenience function for calculating losses separately.
        """
        params = get_params(_opt_state)
        preds, rec_r, gen_r, z0_params, nfe = forward(params,
                                                      batch["observed_data"],
                                                      batch["observed_tp"],
                                                      batch["tp_to_predict"],
                                                      batch["observed_mask"])
        likelihood_ = _likelihood(preds, batch["observed_data"], batch["observed_mask"])
        mse_ = _mse(preds, batch["observed_data"], batch["observed_mask"])
        kl_ = _kl_div(z0_params)
        return -logsumexp(likelihood_ - kl_coef * kl_, axis=0), likelihood_, kl_, mse_, rec_r, gen_r

    def evaluate_loss(opt_state, ds_test, kl_coef):
        """
        Convenience function for evaluating loss over train set in smaller batches.
        """
        loss, likelihood, kl, mse, rec_r, gen_r, rec_nfe, gen_nfe = [], [], [], [], [], [], [], []

        for test_batch_num in range(num_test_batches):
            test_batch = next(ds_test)

            batch_loss, batch_likelihood, batch_kl, batch_mse, batch_rec_r, batch_gen_r = \
                sep_losses(opt_state, test_batch, kl_coef)

            if count_nfe:
                nfes = model["nfe"](get_params(opt_state),
                                    test_batch["observed_data"],
                                    test_batch["observed_tp"],
                                    test_batch["tp_to_predict"],
                                    test_batch["observed_mask"])
                rec_nfe.append(nfes["rec"])
                gen_nfe.append(nfes["gen"])
            else:
                rec_nfe.append(0)
                gen_nfe.append(0)

            loss.append(batch_loss)
            likelihood.append(batch_likelihood)
            kl.append(batch_kl)
            mse.append(batch_mse)
            rec_r.append(batch_rec_r)
            gen_r.append(batch_gen_r)

        loss = jnp.array(loss)
        likelihood = jnp.array(likelihood)
        kl = jnp.array(kl)
        mse = jnp.array(mse)
        rec_r = jnp.array(rec_r)
        gen_r = jnp.array(gen_r)
        rec_nfe = jnp.array(rec_nfe)
        gen_nfe = jnp.array(gen_nfe)

        return jnp.mean(loss), jnp.mean(likelihood), jnp.mean(kl), jnp.mean(mse), jnp.mean(rec_r), jnp.mean(gen_r), \
               jnp.mean(rec_nfe), jnp.mean(gen_nfe)

    itr = 0
    info = collections.defaultdict(dict)
    for epoch in range(parse_args.nepochs):
        for i in range(num_batches):
            batch = next(ds_train)

            itr += 1

            opt_state = update(itr, opt_state, batch, get_kl_coef(epoch))

            if itr % parse_args.test_freq == 0:
                loss_, likelihood_, kl_, mse_, rec_r_, gen_r_, rec_nfe_, gen_nfe_ = \
                    evaluate_loss(opt_state, ds_test, get_kl_coef(epoch))

                print_str = 'Iter {:04d} | Loss {:.6f} | ' \
                            'Likelihood {:.6f} | KL {:.6f} | MSE {:.6f} | Enc. r {:.6f} | Dec. r {:.6f} | ' \
                            'Enc. NFE {:.6f} | Dec. NFE {:.6f}'.\
                    format(itr, loss_, likelihood_, kl_, mse_, rec_r_, gen_r_, rec_nfe_, gen_nfe_)

                print(print_str)

                outfile = open("%s/reg_%s_lam_%.12e_info.txt" % (dirname, reg, lam), "a")
                outfile.write(print_str + "\n")
                outfile.close()

                info[itr]["loss"] = loss_
                info[itr]["likelihood"] = likelihood_
                info[itr]["kl"] = kl_
                info[itr]["mse"] = mse_
                info[itr]["rec_r"] = rec_r_
                info[itr]["gen_r"] = gen_r_
                info[itr]["rec_nfe"] = rec_nfe_
                info[itr]["gen_nfe"] = gen_nfe_

            if itr % parse_args.save_freq == 0:
                param_filename = "%s/reg_%s_lam_%.12e_%d_fargs.pickle" % (dirname, reg, lam, itr)
                fargs = get_params(opt_state)
                outfile = open(param_filename, "wb")
                pickle.dump(fargs, outfile)
                outfile.close()

            outfile = open("%s/reg_%s_lam_%.12e_iter.txt" % (dirname, reg, lam), "a")
            outfile.write("Iter: {:04d}\n".format(itr))
            outfile.close()
    meta = {
        "info": info,
        "args": parse_args
    }
    outfile = open("%s/reg_%s_lam_%.12e_meta.pickle" % (dirname, reg, lam), "wb")
    pickle.dump(meta, outfile)
    outfile.close()


if __name__ == "__main__":
    run()
