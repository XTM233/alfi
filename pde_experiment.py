# %% [markdown]
# # Spatio-temporal Transcriptomics
# 
# Toy dataset from López-Lopera et al. (2019)
# 
# - Data download: https://github.com/anfelopera/PhysicallyGPDrosophila
# 
# ### Imports

# %%
import numpy as np
import torch
import random
import os
import json
import argparse

from torch.nn import Parameter
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path
from scipy.interpolate import interp1d
from dolfin import *

from torch.optim import Adam
from gpytorch.optim import NGD

from alfi.models import TrainMode, MultiOutputGP, PartialLFM, generate_multioutput_gp
from alfi.impl.pdes import ReactionDiffusion
from alfi.utilities.data import dros_ground_truth
from alfi.utilities.fenics import interval_mesh
from alfi.datasets import DrosophilaSpatialTranscriptomics, HomogeneousReactionDiffusion
from alfi.trainers import PartialPreEstimator, PDETrainer
from alfi.plot import plot_spatiotemporal_data, Plotter1d
from alfi.utilities.torch import softplus, inv_softplus, q2, cia, smse
from alfi.configuration import VariationalConfiguration

# %%
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def evaluate_model(lfm, trainer):
    tx = trainer.tx
    disc = trainer.disc
    h = lfm(tx, step=disc)
    f_mean = h.mean.detach()
    f_var = h.variance.detach()
    y_target = trainer.y_target[0]
    f_mean_test = f_mean.squeeze()
    f_var_test = f_var.squeeze()

    prot_q2 = q2(y_target, f_mean.squeeze()).item()
    prot_cia = cia(y_target, f_mean_test, f_var_test).item()
    prot_smse = smse(y_target, f_mean_test).mean().item()

    f = lfm.gp_model(trainer.tx.t())
    lf_target = trainer.lf_target.squeeze()
    num_t = trainer.num_t
    num_x = trainer.num_x
    f_mean = f.mean.view(num_t, num_x)[::disc].reshape(-1)
    f_var = f.variance.view(num_t, num_x)[::disc].reshape(-1)

    mrna_q2 = q2(lf_target, f_mean.squeeze()).item()
    mrna_cia = cia(lf_target, f_mean.squeeze(), f_var.squeeze()).item()
    mrna_smse = smse(lf_target, f_mean.squeeze()).mean().item()

    return prot_q2, prot_cia, prot_smse, mrna_q2, mrna_cia, mrna_smse
# %%
def main(gene, SEED, drosophila=True, use_lhs=True, train_ratio=0.3, skip=True, learn_inducing_locations=False, gp_use_tril=True, params_grad=False, experiment_dir="experiment_results", note="", max_epoch=None, debug=False):
    # Set random seed for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)

    # Load data and configuration
    from time import time
    experiment_name = f"{gene}_{SEED}_{str(time())}{note}"
    os.makedirs(experiment_dir + "/" + experiment_name, exist_ok=True)
    dataset = DrosophilaSpatialTranscriptomics(
        gene=gene, data_dir='./data', scale=True, disc=20) # data folder
    print("Gene name:", gene)
    # :HP: params, matched with author's script
    if gene == 'kni':
        params = dict(lengthscale=10,
                **dros_ground_truth(gene),
                parameter_grad=False,
                warm_epochs=-1,
                natural=False,
                clamp=True)
        num_epochs = 300
    elif gene == 'gt':
        params = dict(lengthscale=10,
                **dros_ground_truth(gene),
                parameter_grad=False,
                natural=False,
                clamp=True)
        num_epochs = 80
    elif gene == "kr":
        # in author's script, params are kept as known
        num_epochs = 100
        params = dict(lengthscale=10,
                **dros_ground_truth(gene),
                parameter_grad=False,
                warm_epochs=-1,
                natural=True,
                zero_mean=True,
                clamp=True)
    if params_grad:
        params['parameter_grad'] = True
        params["sensitivity"] = 0.1
        params["decay"] = 0.1
        params["diffusion"] = 0.01
    if max_epoch > 0:
        num_epochs = max_epoch


    disc = dataset.disc

    data = next(iter(dataset))
    tx, y_target = data
    lengthscale = params['lengthscale']
    zero_mean = params['zero_mean'] if 'zero_mean' in params else False

    # #### Generate FENiCS 1D mesh from the data
    # Alternatively, if our spatial data is not uniformly spaced, we can define a custom mesh as follows.
    # We calculate a mesh that contains all possible spatial locations in the dataset

    spatial = np.unique(tx[1, :])
    mesh = interval_mesh(spatial)
    # The mesh coordinates should match up to the data:
    assert (spatial == mesh.coordinates().reshape(-1)).all()

    # Set up GP model
    # :HP: percentage of data points to use as inducing points
    if tx.shape[1] > 1000:
        num_inducing = int(tx.shape[1] * 3/6)
    else:
        num_inducing = int(tx.shape[1] * 5/6)
    # whether inducing points should be sampled using LHS or random choice
    if use_lhs:
        print('tx', tx.shape)
        from smt.sampling_methods import LHS
        ts = tx[0, :].unique().sort()[0].numpy()
        xs = tx[1, :].unique().sort()[0].numpy()
        xlimits = np.array([[ts[0], ts[-1]],[xs[0], xs[-1]]])
        sampling = LHS(xlimits=xlimits, random_state=np.random.RandomState(SEED))
        inducing_points = torch.tensor(sampling(num_inducing)).unsqueeze(0)
    else:
        inducing_points = torch.stack([
            tx[0, torch.randperm(tx.shape[1], generator=torch.Generator().manual_seed(SEED))[:num_inducing]],
            tx[1, torch.randperm(tx.shape[1], generator=torch.Generator().manual_seed(SEED))[:num_inducing]]
        ], dim=1).unsqueeze(0)
    gp_kwargs = dict(learn_inducing_locations=learn_inducing_locations,
                    natural=params['natural'],
                    use_tril=gp_use_tril)

    gp_model = generate_multioutput_gp(
        1, inducing_points,
        initial_lengthscale=lengthscale,
        ard_dims=2,
        zero_mean=zero_mean,
        gp_kwargs=gp_kwargs)
    gp_model.covar_module.lengthscale = lengthscale
    # lengthscale_constraint=Interval(0.1, 0.3),
    gp_model.double()

    # Set up PDE (fenics module)

    # %%
    # Define fenics model
    ts = tx[0, :].unique().sort()[0].numpy()
    t_range = (ts[0], ts[-1])
    # :HP: number of time steps to discretise the PDE of the dataset
    time_steps = dataset.num_discretised
    fenics_model = ReactionDiffusion(t_range, time_steps, mesh)

    # :HP: configuration for the variational inference, initial conditions, number of samples
    config = VariationalConfiguration(
        initial_conditions=False,
        num_samples=5
    )

    # Set up the LFM
    # include DE parameters
    parameter_grad = params['parameter_grad'] if 'parameter_grad' in params else True
    sensitivity = Parameter(
        inv_softplus(torch.tensor(params['sensitivity'])) * torch.ones((1, 1), dtype=torch.float64),
        requires_grad=False)
    decay = Parameter(
        inv_softplus(torch.tensor(params['decay'])) * torch.ones((1, 1), dtype=torch.float64),
        requires_grad=parameter_grad)
    diffusion = Parameter(
        inv_softplus(torch.tensor(params['diffusion'])) * torch.ones((1, 1), dtype=torch.float64),
        requires_grad=parameter_grad)

    fenics_params = [sensitivity, decay, diffusion]
    num_training = int(train_ratio * tx.shape[1])

    lfm = PartialLFM(1, gp_model, fenics_model, fenics_params, config, num_training_points=num_training)

    # :HP: learning rate for each optimizer, matched with author's script
    if params['natural']:
        variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.03)
        parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.03)
        optimizers = [variational_optimizer, parameter_optimizer]
    else:
        optimizers = [Adam(lfm.parameters(), lr=0.005)]

    track_parameters = list(lfm.fenics_named_parameters.keys()) +\
                    list(map(lambda s: f'gp_model.{s}', dict(lfm.gp_model.named_hyperparameters()).keys()))

    # As in Lopez-Lopera et al., we take 30% of data for training
    train_mask = torch.zeros_like(tx[0, :])
    train_mask[torch.randperm(tx.shape[1], generator=torch.Generator().manual_seed(SEED))[:int(train_ratio * tx.shape[1])]] = 1

    orig_data = dataset.orig_data.squeeze().t()
    # :HP: warm_variational, matched with author's script
    trainer = PDETrainer(lfm, optimizers, dataset,
                        clamp=params['clamp'],
                        track_parameters=track_parameters,
                        train_mask=train_mask.bool(),
                        warm_variational=-1,
                        lf_target=orig_data)
    # trainer.tx is tx from the dataset, added with discretion

    tx = trainer.tx
    time_unique = orig_data[:, 0].unique()
    num_t_orig = orig_data[:, 0].unique().shape[0]
    num_x_orig = orig_data[:, 1].unique().shape[0]
    num_t = tx[0, :].unique().shape[0]
    num_x = tx[1, :].unique().shape[0]
    latent = torch.tensor(orig_data[trainer.t_sorted, 2]).unsqueeze(0)
    latent = latent.repeat(lfm.config.num_samples, 1, 1)
    latent = latent.view(lfm.config.num_samples, 1, num_t_orig, num_x_orig)
    time_interp = tx[0].unique()
    time_interp[-1] -= 1e-5 
    latent = torch.from_numpy(interp1d(time_unique, latent, axis=2)(time_interp))

    y_target = trainer.y_target[0]
    y_matrix = y_target.view(num_t_orig, num_x_orig)
    pde_func, pde_target = lfm.fenics_model.interpolated_gradient(tx, y_matrix, disc=disc, plot=False)
    # Training
    # PreEstimator matches the gradient

    # %%
    num_training = int(train_ratio * tx.shape[1])
    # :HP: optimizers for the pre-estimator, matched with author's script
    if params['natural']:
        variational_optimizer = NGD(lfm.variational_parameters(), num_data=num_training, lr=0.03)
        parameter_optimizer = Adam(lfm.nonvariational_parameters(), lr=0.03)
        optimizers = [variational_optimizer, parameter_optimizer]
    else:
        optimizers = [Adam(lfm.parameters(), lr=0.05)]

    if debug and gene == 'kr' and params['natural']:
        print('Debug mode enabled, using smaller learning rate')
        optimizers = [Adam(lfm.parameters(), lr=0.015)]

    pre_estimator = PartialPreEstimator(
        lfm, optimizers, dataset, pde_func,
        input_pair=(trainer.tx, trainer.y_target), target=pde_target,
        train_mask=trainer.train_mask
    )
    print("Start training pre-estimator")
    lfm.set_mode(TrainMode.GRADIENT_MATCH)
    lfm.config.num_samples = 50
    if not skip:
        if debug:
            times = pre_estimator.train(60, report_interval=5)  # [(timestamp, loss) x 60]
        else:
            times = pre_estimator.train(80, report_interval=10) # [(timestamp, loss) x 60]
    lfm.config.num_samples = 5

    tx_orig = orig_data[:, :2].t()

    # Compare tx and tx_orig, find the indices of tx that correspond to tx_orig
    indices = torch.where(torch.all(torch.eq(tx.unsqueeze(-1), tx_orig.unsqueeze(1)), dim=0))[1]
    # the first 512 points of tx correspond to tx_orig
    print(f'Evaluating LFM after gradient matching')
    if not drosophila:
        lfm.eval()
        f = lfm(tx)
        print(f.mean.shape)
        f_mean = f.mean.detach()
        f_var = f.variance.detach()
        y_target = trainer.y_target[0]

        print(f_mean.shape, y_target.shape, f_var.shape)
        print('prot Q2', q2(y_target.squeeze(), f_mean.squeeze()))
        print('prot CA', cia(y_target.squeeze(), f_mean.squeeze(), f_var.squeeze()))
        gp = lfm.gp_model(tx.t())
        lf_target = orig_data[trainer.t_sorted, 2]
        f_mean = gp.mean.detach().view(num_t, num_x)[::disc].reshape(-1)
        f_var = gp.variance.detach().view(num_t, num_x)[::disc].reshape(-1)

        print('mrna Q2', q2(lf_target.squeeze(), f_mean.squeeze()))
        print('mrna CA', cia(lf_target.squeeze(), f_mean.squeeze(), f_var.squeeze()))


        loss_traj = np.array([tup[1][0] for tup in times])
        plt.plot(loss_traj)
    if drosophila:
        lfm.eval()
        prot_q2, prot_cia, _, mrna_q2, mrna_cia, _ = evaluate_model(lfm, trainer)
        print('prot Q2', prot_q2)
        print('prot CA', prot_cia)
        print('mrna Q2', mrna_q2)
        print('mrna CA', mrna_cia)

    # Train LFM
    # repeat untinl convergence
    print(f'Training LFM for {num_epochs} epochs')

    times = trainer.train(num_epochs)

    # %%
    time = np.array([tup[0] for tup in times])
    loss_traj = np.array([tup[1][0] for tup in times])
    plt.savefig(experiment_dir+"/"+ experiment_name + "/prot.pdf", dpi=300, bbox_inches='tight')

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Loss')
    ax1.plot(time, loss_traj)
    ax1.tick_params(axis='y')
    fig.savefig(experiment_dir+"/"+ experiment_name + "/loss.pdf", dpi=300, bbox_inches='tight')

    ell = np.array([tup[1][1][0] for tup in times])
    kl = np.array([tup[1][1][1] for tup in times])
    times = np.stack((time, loss_traj, ell, kl)).T
    np.save(experiment_dir + "/" + experiment_name + "/training_times.npy", times)
    # %%
    # save the trained model
    filepath = str(num_epochs) +"epochs_trained_"   # Specify the file path to save the model
    lfm.save(experiment_dir+"/"+ experiment_name + "/" + filepath)

    # ### Evaluate trained model

    # %%
    with torch.no_grad():
        lfm.config.num_samples = 5
        lfm.eval()
        u = lfm.gp_model(trainer.tx.t()).sample(torch.Size([5])).permute(0, 2, 1)
        u = u.view(*u.shape[:2], num_t, num_x)
        dy_t_ = pde_func(
            trainer.y_target,
            u[:, :, ::disc].contiguous(),
            sensitivity,
            decay,
            diffusion)[0]

        out_predicted = lfm.solve_pde(u.view(5, 1, num_t, num_x)).mean(0)

    ts = tx[0, :].unique().numpy()
    xs = tx[1, :].unique().numpy()
    extent = [ts[0], ts[-1], xs[0], xs[-1]]
    # axes = plot_spatiotemporal_data(
    #     [
    #         trainer.y_target.view(num_t_orig, num_x_orig).t(),
    #         pde_target.reshape(num_t_orig, num_x_orig).t(),
    #         dy_t_.view(num_t_orig, num_x_orig).t(),
    #         latent[0].view(num_t, num_x).t(),
    #         u.mean(0).view(num_t, num_x).t(),
    #     ],
    #     extent, titles=['y', 'target dy_t', 'pred dy_t_', 'target latent', 'pred latent']
    # )
    # plot_spatiotemporal_data(
    #     [
    #         # real.t().detach(),
    #         trainer.y_target.view(num_t_orig, num_x_orig).t(),
    #         out_predicted.t().detach()
    #     ],
    #     extent, titles=['target y', 'pred y']
    # )
    # print([softplus(param).item() for param in lfm.fenics_parameters])

    # %%
    # load trained model
    gp_args = [1, inducing_points]
    gp_kwargs = dict(learn_inducing_locations=learn_inducing_locations,
                    natural=params['natural'],
                    use_tril=gp_use_tril)
    gp_model = generate_multioutput_gp(
        *gp_args,
        initial_lengthscale=lengthscale,
        ard_dims=2,
        zero_mean=zero_mean,
        gp_kwargs=gp_kwargs)
    gp_args = [gp_model.mean_module, gp_model.covar_module] + [gp_args[1], gp_args[0]]

    lfm = PartialLFM.load(experiment_dir+"/"+ experiment_name + "/" + filepath,
                        gp_cls=MultiOutputGP,
                        gp_args=gp_args,
                        gp_kwargs=gp_kwargs,
                        lfm_args=[1, fenics_model, fenics_params, config])
    # lfm = PartialLFM(1, gp_model, fenics_model, fenics_params, config)

    gp_model = lfm.gp_model
    optimizer = [torch.optim.Adam(lfm.parameters(), lr=0.07)]
    trainer = PDETrainer(lfm, optimizer, dataset, track_parameters=list(lfm.fenics_named_parameters.keys()), lf_target=orig_data)
    lfm.eval()
    prot_q2, prot_cia, prot_smse, mrna_q2, mrna_cia, mrna_smse = evaluate_model(lfm, trainer)
    result_dict = {
        'prot_q2': prot_q2,
        'prot_cia': prot_cia,
        'prot_smse': prot_smse,
        'mrna_q2': mrna_q2,
        'mrna_cia': mrna_cia,
        'mrna_smse': mrna_smse
    }
    with open(experiment_dir + "/" + experiment_name + "/result.json", "w") as f:
        json.dump(result_dict, f, indent=4)
if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Run Spatio-temporal Transcriptomics experiment.")
    parser.add_argument("--gene", type=str, required=True, help="Gene to analyze (e.g., 'kr', 'kni', 'gt').")
    parser.add_argument("--SEED", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--drosophila", type=bool, default=True, help="Whether to use Drosophila dataset.")
    parser.add_argument("--use_lhs", type=bool, default=True, help="Use Latin Hypercube Sampling for inducing points.")
    parser.add_argument("--train_ratio", type=float, default=0.3, help="Ratio of data used for training.")
    parser.add_argument("--skip", type=bool, default=False, help="Skip pre-estimator training.")
    parser.add_argument("--learn_inducing_locations", type=bool, default=False, help="Learn inducing point locations.")
    parser.add_argument("--gp_use_tril", type=bool, default=True, help="Use Cholesky decomposition in GP.")
    parser.add_argument("--params_grad", type=bool, default=False, help="Enable gradient for PDE parameters.")
    parser.add_argument("--experiment_dir", type=str, default="experiment_results", help="Directory to save experiment results.")
    parser.add_argument("--note", type=str, default="", help="Additional note for experiment naming.")
    parser.add_argument("--max_epoch", type=int, default=0, help="Maximum number of training epochs.")
    parser.add_argument("--debug", type=bool, default=False, help="Enable debug mode for smaller learning rates.")

    args = parser.parse_args()

    main(
        gene=args.gene,
        SEED=args.SEED,
        drosophila=args.drosophila,
        use_lhs=args.use_lhs,
        train_ratio=args.train_ratio,
        skip=args.skip,
        learn_inducing_locations=args.learn_inducing_locations,
        gp_use_tril=args.gp_use_tril,
        params_grad=args.params_grad,
        experiment_dir=args.experiment_dir,
        note=args.note,
        max_epoch=args.max_epoch,
        debug=args.debug
    )
