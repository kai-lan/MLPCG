import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'font.weight': 'bold',
          'axes.labelweight': "bold",
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# count fraction of time and fraction of cases where our solver is the fastest
def fraction_time_and_count_fastest(model, scenes):
    i, j = 0, 0
    ii, jj = 0, 0
    for scene in scenes:
        data = pd.read_excel('results/Performance.xlsx', sheet_name=scene)
        AMG_time = data['AMG time'].tolist()[:200]
        AMGX_time = data['AMGX time'].tolist()[:200]
        IC_time = data['IC time'].tolist()[:200]
        NPSD2_time = data[f'{model} time'].tolist()[:200]
        CG_time = data['CG time'].tolist()[:200]
        for amg, amgx, ic, npsd, cg in zip(AMG_time, AMGX_time, IC_time, NPSD2_time, CG_time):
            if pd.isna(amg): continue
            if npsd < amg and npsd < ic and npsd < cg:
                i += npsd
                ii += 1
            j += npsd
            jj += 1
    print('Fraction of time:', i/j, 'fraction of cases:', ii/jj)

def extract_data(model):

    # Different system size
    data = pd.read_excel('results/Profile.xlsx', sheet_name=f"smoke_solid_192_f20")
    np.savez(f"results/res_iters_smoke_solid_192_f20.npz",
            amg_res=data["AMG res"], amg_time=data["AMG time"],
            ic_res=data["IC res"], ic_time=data["IC time"],
            cg_res=data["CG res"], cg_time=data["CG time"],
            npsd_res=data[f"{model} res"], npsd_time=data[f"{model} time"],
            )

def fetch_data(model, scenes, solvers=['AMG', 'AMGX', 'IC', 'CG', 'DCDM']):
    results = [[] for _ in range(len(solvers))]
    for scene in scenes:
        data = pd.read_excel('results/Performance.xlsx', sheet_name=scene)
        data = data.dropna()
        for i, solver in enumerate(solvers):
            results[i].extend(data[f'{solver} speedup'].tolist())
    for i in range(len(results)):
        results[i] = np.asarray(results[i], dtype=float)
    return results

def order_data(scenes):
    sizes = []
    for scene in scenes:
        data = pd.read_excel('results/Performance.xlsx', sheet_name=scene)
        data = data.dropna()
        sizes.extend(data['Sizes'].tolist()[:200])
    sizes = np.asarray(sizes, dtype=int)
    s_25, s_50, s_75 = np.percentile(sizes, 25), np.percentile(sizes, 50), np.percentile(sizes, 75)
    return s_25, s_50, s_75

def classify_data(scenes, partitions, solvers=['AMG', 'AMGX', 'IC', 'CG']):
    num_divisions = len(partitions)+1
    results = [[[] for _ in range(len(solvers))] for _ in range(num_divisions)]
    for scene in scenes:
        data = pd.read_excel('results/Performance.xlsx', sheet_name=scene)
        data = data.dropna()

        speedup_list = []
        for solver in solvers:
            speedup_list.append(data[f'{solver} speedup'].tolist())
        size = data['Sizes'].tolist()[:200]

        for i in range(len(size)):
            if size[i] < partitions[0]:
                for j in range(len(speedup_list)):
                    results[0][j].append(speedup_list[j][i])
                continue
            if size[i] >= partitions[-1]:
                for j in range(len(speedup_list)):
                    results[-1][j].append(speedup_list[j][i])
                continue
            for k in range(1, len(partitions)):
                if size[i] >= partitions[k-1] and size[i] < partitions[k]:
                    for j in range(len(speedup_list)):
                        results[k][j].append(speedup_list[j][i])
                    continue
    return results

def plot_hist(speedup_list, labels, colors, save=None, num_bins=150, ticks=None):
    x_min = np.min(speedup_list)
    x_max = np.max(speedup_list)
    if ticks is None:
        ticks = np.arange(np.floor(x_min), np.ceil(x_max)+1, 2)
    num_bins = num_bins
    bins = np.linspace(x_min, x_max, num_bins)
    weights = np.zeros_like(speedup_list[0]) + 1 / len(speedup_list[0])
    for speedup, color, label in zip(speedup_list, colors, labels):
        plt.hist(speedup, bins=bins, color=color, alpha=0.5, label=label, weights=weights, cumulative=False)
    # plt.gca().set_xmargin(0.04)
    # plt.ylim([0,1])
    plt.xlim([0,18])
    plt.xticks(ticks)
    plt.gca().get_xticklabels()[1].set_color('red')
    plt.legend()
    plt.xlabel("Speedup over CG")
    plt.ylabel("Fraction of systems")
    plt.tight_layout()
    if save:
        plt.savefig(f"{save}.pdf", bbox_inches='tight')
    plt.clf()

def fetch_data_res(scene, solvers=['AMG', 'IC', 'CG']):
    data = pd.read_excel('results/Profile.xlsx', sheet_name=scene)
    results = []
    for solver in solvers:
        results.append([data[f"{solver} res"].dropna(), data[f"{solver} time"].dropna()])
    return results

def plot_res(res, labels, colors=None, save=None):
    if colors is None:
        for r, lab in zip(res, labels):
            plt.plot(r, label=lab)
    else:
         for r, lab, c in zip(res, labels, colors):
            plt.plot(r, label=lab, c=c)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel(r"log $|\mathbf{r}|/|\mathbf{r}_0|$")
    if save:
        plt.savefig(f"{save}.pdf", bbox_inches='tight')
    # plt.show()
    plt.clf()
def plot_time(res, time, labels, colors=None, save=None):
    if colors is None:
        for r, t, lab in zip(res, time, labels):
            plt.plot(t, r, label=lab)
    else:
        for r, t, lab, c in zip(res, time, labels, colors):
            plt.plot(t, r, label=lab, c=c)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(r"log $|\mathbf{r}|/|\mathbf{r}_0|$")
    if save:
        plt.savefig(f"{save}.pdf", bbox_inches='tight')
    # plt.show()
    plt.clf()
def plot_res_and_time(res, time, labels, colors=None, save=None):
    fig, axes = plt.subplots(2, constrained_layout=True)
    if colors is None:
        for r, t, lab in zip(res, time, labels):
            l1 = axes[0].plot(r, label=lab)
            l2 = axes[1].plot(t, r)
    else:
        for r, t, lab, c in zip(res, time, labels, colors):
            l1 = axes[0].plot(r, c=c, label=lab)
            l2 = axes[1].plot(t, r, c=c)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].legend()
    axes[0].set_xlabel("Iterations")
    axes[1].set_xlabel("Time")
    axes[0].set_ylabel(r"log $|\mathbf{r}|/|\mathbf{r}_0|$")
    axes[1].set_ylabel(r"log $|\mathbf{r}|/|\mathbf{r}_0|$")
    axes[0].set_yticks(np.logspace(-6, 0, 3))
    axes[1].set_yticks(np.logspace(-6, 0, 3))
    if save:
        plt.savefig(f"{save}.pdf", bbox_inches='tight')
    # plt.show()
    plt.clf()
def plot_cholesky(model, scene):
    # Cholesky solver
    data = pd.read_excel('results/Performance.xlsx', sheet_name=scene)
    data = data.dropna()
    cholesky_time = data['Cholesky time'].iloc[:200]
    cg_time = data['CG time'].iloc[:200]
    amg_time = data['AMG time'].iloc[:200]
    amgx_time = data['AMGX time'].iloc[:200]
    ic_time = data['IC time'].iloc[:200]
    npsd_time = data[f'{model} time'].iloc[:200]
    sizes = data['Sizes'].tolist()[:200]

    plt.scatter(sizes, amg_time, label='AMG', c='red', s=4)
    plt.scatter(sizes, amgx_time, label='AMGX', c='darkorange', s=4)
    plt.scatter(sizes, ic_time, label='IC', c='blue', s=4)
    plt.scatter(sizes, cg_time, label='CG', c='magenta', s=4)
    plt.scatter(sizes, npsd_time, label='Ours', c='green', s=4)
    plt.scatter(sizes, cholesky_time, label='Cholesky', c='cyan', s=4)
    plt.xscale('log')
    plt.yscale('log', base=2)
    plt.grid()
    plt.xlabel("Number of unknowns")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.savefig(f"cholesky_comparison.pdf", bbox_inches='tight')
    plt.clf()

scenes = [
    "smoke_solid_128",
    "smoke_solid_256",
    "smoke_bunny_128",
    "smoke_bunny_256",
    "standing_pool_scooping_128",
    "standing_pool_scooping_256",
    "dambreak_pillars_128_256",
    "dambreak_bunny_128_256",
    "waterflow_spiky_torus_128",
    "waterflow_spiky_torus_256",
    "waterflow_ball_128",
    "waterflow_ball_256"
    ]

model = "l5 trilinear 25"

# fraction_time_and_count_fastest(model, scenes)
# fraction_time_and_count_fastest(model, scenes[4:])

#################
# Plot histogram
#################
# solver_list = ['AMG', 'IC', model, 'AMGX', 'DCDM']
# solver_labels = ['AMG', 'IC', 'Ours', 'AMGX', 'DCDM']
# solver_colors = ['red', 'blue', 'green', 'darkorange', 'gold']
# results = fetch_data(model, scenes, solver_list[:4])
# x_min = np.min(results)
# x_max = np.max(results)
# ticks = np.arange(np.floor(x_min), np.ceil(x_max)+1, 2)
# ticks = list(ticks)
# ticks.insert(1, 1)
# print("Ticks:", ticks)
# plot_hist(results, solver_labels, solver_colors, save="hist_all", ticks=ticks)

# s_25, s_50, s_75 = order_data(scenes)
# print("Partition:", s_25, s_50, s_75)
# results = classify_data(scenes, [s_25], solver_list[:4])

# plot_hist(results[0], solver_labels, solver_colors, num_bins=100, ticks=ticks, save="hist_all_low")
# plot_hist(results[1], solver_labels, solver_colors, num_bins=100, ticks=ticks, save="hist_all_high")

# results = fetch_data(model, ["smoke_solid_128", "smoke_bunny_128"], solver_list)
# plot_hist(results, solver_labels, solver_colors, num_bins=100, save="hist_smoke_128", ticks=ticks)
# results = fetch_data(model, ["smoke_solid_256", "smoke_bunny_256"], solver_list[:4])
# plot_hist(results, solver_labels, solver_colors, num_bins=100, save="hist_smoke_256", ticks=ticks)

# results = fetch_data(model, scenes[4:], solver_list[:4])
# plot_hist(results, solver_labels, solver_colors, num_bins=100, save="hist_water", ticks=ticks)
# plot_hist(fetch_data(model, ["standing_pool_scooping_128", "waterflow_ball_128"], solver_list[:4]), solver_labels, solver_colors, num_bins=100, save="hist_water_128", ticks=ticks)
# plot_hist(fetch_data(model, ["dambreak_pillars_128_256", "dambreak_bunny_128_256"], solver_list[:4]), solver_labels, solver_colors, num_bins=100, save="hist_water_128_256", ticks=ticks)
# plot_hist(fetch_data(model, ["standing_pool_scooping_256", "waterflow_ball_256"], solver_list[:4]), solver_labels, solver_colors, num_bins=100, save="hist_water_256", ticks=ticks)

# s_25, s_100, s_75 = order_data(scenes[4:])
# results = classify_data(scenes[4:], [s_25], solver_list[:4])
# plot_hist(results[0], solver_labels, solver_colors, num_bins=100, ticks=ticks, save="hist_water_small25")
# plot_hist(results[1], solver_labels, solver_colors, num_bins=100, ticks=ticks, save="hist_water_large75")

####################
# Residual and time
####################
solver_list = ['AMG', 'AMGX', 'IC', 'CG', model, 'DCDM', 'FN']
solver_labels = ['AMG', 'AMGX', 'IC', 'CG', 'Ours', 'DCDM', 'FN']
solver_colors = ['red', 'darkorange', 'blue', 'magenta', 'green', 'gold', 'brown']

results = fetch_data_res("smoke_solid_128_f100", solver_list)
plot_res([result[0] for result in results], solver_labels, solver_colors, "smoke_solid_128_f100_res_iters")
plot_time([results[0][0], results[1][0], results[2][0], results[3][0], results[4][0], results[5][0][:61], results[6][0][:61]],
          [results[0][1], results[1][1], results[2][1], results[3][1], results[4][1], results[5][1][:61], results[6][1][:61]],
          solver_labels,
          solver_colors,
          "smoke_solid_128_f100_res_time")

# results = fetch_data_res("waterflow_ball_128_f200", solver_list)
# plot_res([result[0] for result in results],
#          solver_labels, solver_colors, "waterflow_ball_128_f200_res_iters")
# plot_time([results[0][0], results[1][0], results[2][0], results[3][0], results[4][0], results[5][0][:41], results[6][0][:43]],
#           [results[0][1], results[1][1], results[2][1], results[3][1], results[4][1], results[5][1][:41], results[6][1][:43]],
#           solver_labels,
#           solver_colors,
#           "waterflow_ball_128_f200_res_time")

# results = fetch_data_res("smoke_bunny_256_f200", solver_list[:5])
# plot_res([result[0] for result in results],
#          solver_labels, solver_colors, "smoke_bunny_256_f200_res_iters")
# plot_time([result[0] for result in results],
#           [result[1] for result in results],
#           solver_labels,
#           solver_colors,
#           "smoke_bunny_256_f200_res_time")

# results = fetch_data_res("waterflow_ball_256_f200", solver_list[:5])
# plot_res([result[0] for result in results],
#          solver_labels, solver_colors, "waterflow_ball_256_f200_res_iters")
# plot_time([result[0] for result in results],
#           [result[1] for result in results],
#           solver_labels,
#           solver_colors,
#           "waterflow_ball_256_f200_res_time")


# solver_list[4] = 'l4 trilinear 62'

# results = fetch_data_res("smoke_solid_192_f20", solver_list[:5])
# plot_res([result[0] for result in results], solver_labels,
#          solver_colors, "smoke_solid_192_f20_res_iters")
# plot_time([result[0] for result in results],
#           [result[1] for result in results],
#           solver_labels,
#           solver_colors,
#           "smoke_solid_192_f20_res_time")

# results = fetch_data_res("damreak_pillars_128_384_f5", solver_list[:5])
# plot_res([result[0] for result in results], solver_labels,
#          solver_colors, "damreak_pillars_128_384_f5_res_iters")
# plot_time([result[0] for result in results],
#           [result[1] for result in results],
#           solver_labels,
#           solver_colors,
#           "damreak_pillars_128_384_f5_res_time")

# plot_cholesky(model, "waterflow_spiky_torus_256")


##################################
# Additional histograms
##################################
# solver_list = ['AMG', 'IC', model, 'AMGX']
# solver_labels = ['AMG', 'IC', 'Ours', 'AMGX']
# solver_colors = ['red', 'blue', 'green', 'darkorange']
# for scene in scenes:
#     print(scene)
#     results = fetch_data(model, [scene], solver_list)
#     x_min = np.min(results)
#     x_max = np.max(results)
#     ticks = np.arange(np.floor(x_min), np.ceil(x_max)+1, 2)
#     ticks = list(ticks)
#     ticks.insert(1, 1)
#     plot_hist(results, solver_labels, solver_colors, save=f"hist_{scene}", ticks=ticks)


#####################
# Different l levels
#####################
# solver_list = ['l5 trilinear 25', 'l4 trilinear 62', 'l3 trilinear 57', 'l2 trilinear 57']
# solver_labels = ['L=6', 'L=5', 'L=4', 'L=3']
# solver_colors = ['green', 'cyan', 'magenta', 'darkorange']

# results = fetch_data_res("smoke_solid_128_f100", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "smoke_solid_128_f100_res_time_levels")
# results = fetch_data_res("smoke_bunny_128_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "smoke_bunny_128_f200_res_time_levels")
# results = fetch_data_res("standing_pool_128_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"standing_pool_128_f200_res_time_levels")
# results = fetch_data_res("spiky_torus_128_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"spiky_torus_128_f200_res_time_levels")
# results = fetch_data_res("waterflow_ball_128_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "waterflow_ball_128_f200_res_time_levels")
# results = fetch_data_res("dambreak_pillars_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "dambreak_pillars_f200_res_time_levels")
# results = fetch_data_res("dambreak_bunny_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"dambreak_bunny_f200_res_time_levels")
# results = fetch_data_res("smoke_solid_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"smoke_solid_256_f200_res_time_levels")
# results = fetch_data_res("smoke_bunny_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "smoke_bunny_256_f200_res_time_levels")
# results = fetch_data_res("standing_pool_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "standing_pool_256_f200_res_time_levels")
# results = fetch_data_res("spiky_torus_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"spiky_torus_256_f200_res_time_levels")
# results = fetch_data_res("waterflow_ball_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"waterflow_ball_256_f200_res_time_levels")

######################
# Different PCGs
######################
# solver_list = ['l5 trilinear 25', 'ML cg', 'ML flex cg', 'ML npsd', 'CG']
# solver_labels = ['NPSDO', 'PCG', 'Flexible PCG', 'NPSD', 'CG']
# solver_colors = ['green', 'cyan', 'brown', 'darkorange', 'magenta']

# results = fetch_data_res("smoke_solid_128_f100", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "smoke_solid_128_f100_res_time_pcgs")
# results = fetch_data_res("smoke_bunny_128_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "smoke_bunny_128_f200_res_time_pcgs")
# results = fetch_data_res("standing_pool_128_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"standing_pool_128_f200_res_time_pcgs")
# results = fetch_data_res("spiky_torus_128_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"spiky_torus_128_f200_res_time_pcgs")
# results = fetch_data_res("waterflow_ball_128_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "waterflow_ball_128_f200_res_time_pcgs")
# results = fetch_data_res("dambreak_pillars_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "dambreak_pillars_f200_res_time_pcgs")
# results = fetch_data_res("dambreak_bunny_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"dambreak_bunny_f200_res_time_pcgs")
# results = fetch_data_res("smoke_solid_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"smoke_solid_256_f200_res_time_pcgs")
# results = fetch_data_res("smoke_bunny_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "smoke_bunny_256_f200_res_time_pcgs")
# results = fetch_data_res("standing_pool_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "standing_pool_256_f200_res_time_pcgs")
# results = fetch_data_res("spiky_torus_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"spiky_torus_256_f200_res_time_pcgs")
# results = fetch_data_res("waterflow_ball_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"waterflow_ball_256_f200_res_time_pcgs")


############################################
# Different search iterations
############################################
# solver_list = ['ML npsd', 'iters 1', 'l5 trilinear 25', 'iters 3', 'iters 4']
# solver_labels = ['0', '1', '2', '3', '4']
# solver_colors = ['cyan', 'darkorange', 'green', 'magenta', 'red']

# results = fetch_data_res("smoke_solid_128_f100", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "smoke_solid_128_f100_res_time_ortho")
# results = fetch_data_res("smoke_bunny_128_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "smoke_bunny_128_f200_res_time_ortho")
# results = fetch_data_res("standing_pool_128_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"standing_pool_128_f200_res_time_ortho")
# results = fetch_data_res("spiky_torus_128_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"spiky_torus_128_f200_res_time_ortho")
# results = fetch_data_res("waterflow_ball_128_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "waterflow_ball_128_f200_res_time_ortho")
# results = fetch_data_res("dambreak_pillars_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "dambreak_pillars_f200_res_time_ortho")
# results = fetch_data_res("dambreak_bunny_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"dambreak_bunny_f200_res_time_ortho")
# results = fetch_data_res("smoke_solid_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"smoke_solid_256_f200_res_time_ortho")
# results = fetch_data_res("smoke_bunny_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "smoke_bunny_256_f200_res_time_ortho")
# results = fetch_data_res("standing_pool_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors, "standing_pool_256_f200_res_time_ortho")
# results = fetch_data_res("spiky_torus_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"spiky_torus_256_f200_res_time_ortho")
# results = fetch_data_res("waterflow_ball_256_f200", solver_list)
# plot_res_and_time([result[0] for result in results], [result[1] for result in results], solver_labels, solver_colors,"waterflow_ball_256_f200_res_time_ortho")

#####################
# Symmetry
#####################
# solver_list = ['AMG', 'AMGX', 'IC', 'CG', 'l5 trilinear 25', 'spd 72']
# solver_labels = ['AMG', 'AMGX', 'IC', 'CG', 'Ours', 'Symmtric variant']
# solver_colors = ['red', 'darkorange', 'blue', 'magenta', 'green', 'gold']

# results = fetch_data_res("waterflow_ball_128_f200", solver_list)
# plot_res([result[0] for result in results],
#          solver_labels, solver_colors, "smoke_solid_128_f100_iters_symmetry")
# plot_time([result[0] for result in results],
#           [result[1] for result in results],
#           solver_labels,
#           solver_colors,
#           "smoke_solid_128_f100_time_symmetry")

# results = fetch_data_res("waterflow_ball_128_f200", solver_list)
# plot_res([result[0] for result in results],
#          solver_labels, solver_colors, "waterflow_ball_128_f200_iters_symmetry")
# plot_time([result[0] for result in results],
#           [result[1] for result in results],
#           solver_labels,
#           solver_colors,
#           "waterflow_ball_128_f200_time_symmetry")

# results = fetch_data_res("smoke_bunny_256_f200", solver_list)
# plot_res([result[0] for result in results],
#          solver_labels, solver_colors, "smoke_bunny_256_f200_iters_symmetry")
# plot_time([result[0] for result in results],
#           [result[1] for result in results],
#           solver_labels,
#           solver_colors,
#           "smoke_bunny_256_f200_time_symmetry")

# results = fetch_data_res("waterflow_ball_256_f200", solver_list)
# plot_res([result[0] for result in results],
#          solver_labels, solver_colors, "waterflow_ball_256_f200_iters_symmetry")
# plot_time([result[0] for result in results],
#           [result[1] for result in results],
#           solver_labels,
#           solver_colors,
#           "waterflow_ball_256_f200_time_symmetry")

