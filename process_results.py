import numpy as np
import pandas as pd
import pickle
import itertools as it

from matplotlib import pyplot as plt
from matplotlib import rcParams
plt.rcParams["font.family"] = "Times New Roman"

rcParams.update({'figure.autolayout': True})
rcParams.update({'font.size': 12})

pd.set_option('display.max_colwidth', 400)


def plot_biases(layers, model_names, model_titles, df):
    models = model_names
    colors = ["#A93226", "#884EA0", "#2471A3", "#1ABC9C", "#D4AC0D", "#E59866", "#839192", "#5D6D7E", "#0000FF",
              "#00FF00", '#FF0000', "#BBA512", "#951523"]
    markers = ['+', '+', '+', 'o', 'o', 'o', '>', '>', '>', '>', '>', '.', '|']
    for layer in layers:
        fig, ax = plt.subplots()
        for i, model in enumerate(models):
            df[df[f"p_{model}_{layer}"] < 0.1][f"p_{model}_{layer}"].reset_index().dropna().sort_values(
                "index", ascending=False).plot(kind='scatter', y='index', x=f"p_{model}_{layer}", color=colors[i],
                                               marker=markers[i], label=model_titles[i], figsize=(10, 7), ax=ax)
            ax.set_xlabel("p-value")
            ax.set_ylabel("Name of the bias")
            fig.savefig(f'marker_biases_{layer}_times_font.pdf', format='pdf')

    plt.show()


def plot_evolution_of_biases(layers, model_names, df):

    models = model_names
    colors = ["#A93226", "#884EA0", "#2471A3", "#1ABC9C", "#D4AC0D", "#E59866", "#839192", "#5D6D7E", "#17202A",
              "#000000"]
    layer_labels = [' ', 'Block 1', 'Block 2', 'Block 3', 'Block 4', 'Block 5', 'GAP']
    for i, model in enumerate(models):
        depth = 0
        fig, ax = plt.subplots()
        for j, layer in enumerate(layers):
            newdf = df[df[f"p_{model}_{layer}"] < 0.1][f"p_{model}_{layer}"].dropna().notnull().astype('int')+depth
            newdf.reset_index().sort_values(
                "index", ascending=False).plot(kind='scatter', y='index', x=f"p_{model}_{layer}",
                                               color=colors[j],
                                               label=layer_labels[j], figsize=(10, 7), ax=ax)
            depth+=1
        ax.get_legend().remove()
        ax.set_xlabel("Block")
        ax.set_xticklabels(layer_labels)
        ax.set_ylabel("Name of the bias")
        fig.savefig(f'cvpr_evolution_of_biases_model_{model}.pdf', format='pdf')
    plt.show()


def to_latex(df, layer='avg_pool'):
    cols = ['random', 'jigsaw', 'rotation', 'relative_loc', 'cluster_fit', 'odc', 'swav', 'npid', 'moco_v1',
            'moco_v2', 'simclr_vissl_200',
            'byol', 'supervised']
    combo = pd.DataFrame()

    def cell(row):
        ans = ""
        if row[0] < 0.01:
            ans += "\\cellcolor[RGB]{92,92,255} "
        elif row[0] < 0.05:
            ans += "\\cellcolor[RGB]{151, 151, 255} "
        elif row[0] < 0.1:
            ans += "\\cellcolor[RGB]{213, 213, 255} "
        p = round(row[0], 2)
        d = str(round(row[1], 2))
        if str(p) == '0.0':
            p = '<0.01'
        if str(p) == '1.0':
            p = '0.99'
        ans += "{\\begin{tabular}[c]{@{}c@{}}" + f"\\textbf{{{p}}} \\\\ {d}" + "\\end{tabular}}"
        return ans

    for model in cols:
        combo = pd.concat(
            [combo, df[[f'p_{model}_{layer}', f'd_{model}_{layer}']].apply(cell, axis=1).rename(model)],
            axis=1)

    return combo.to_latex(escape=False)


def plot_cumulative(df, layer, models, labels, threshold=0.1, save=False, extra_msg=""):

    # apply threshold
    df = df[df.filter(regex='^p_') < threshold]
    ps = set()
    for col in df.columns:
        if col[0] == 'p':
            values = df[col].dropna().values
            for value in values:
                ps.add(value)

    ps = sorted(list(ps))

    ys = []
    pass_control_bias = []
    max_biases = 0

    for model in models:
        y = np.zeros(len(ps))
        col = df[f'p_{model}_{layer}']
        buffer = 0
        if df.loc['Insect-Flower'][f'p_{model}_{layer}'] < 0.1:
            pass_control_bias.append(True)
        else:
            pass_control_bias.append(False)
        for i, t in enumerate(ps):
            buffer += col[col < t].count()
            y[i] = col[col < t].count()
        max_biases = max(y) if max(y) > max_biases else max_biases
        ys.append(y)

    styles = ["dashed"] * 6 + ["solid"]*6 + ["dotted"]*5
    thickness = [1 for _ in range(15)]
    thickness[6:10] = [2] * 5

    colors = [(1, 0, 0), (0, 0.3448, 0), (0, 1, 0), (0, 0, 0.1724), (1.0000, 0.1034, 0.7241),
              (1.0000, 0.8276, 0), (0, 0, 1), (0.5172, 0.5172, 1.0000),
              (0.6207, 0.3103, 0.2759), (0, 1.0000, 0.7586),]

    for i in range(len(models)):
        if i < len(colors):
            plt.step(ps, ys[i], label=labels[i], linestyle=styles[i], color=colors[i], linewidth=thickness[i])
        else:
            plt.step(ps, ys[i], label=labels[i], linestyle=styles[i], linewidth=thickness[i])

    fig = plt.gcf()
    fig.set_size_inches(8, 6.5)
    plt.legend()
    plt.xlabel("$\it{p_t}$")
    plt.ylabel("Number of identified biases in the model")
    plt.yticks(np.arange(0, 21, 1.0))
    ax = plt.gca()
    ax.set_xscale('log')
    # ax.set_ylim([0, 20])
    # title = "the Global Average Pooling layer" if layer=='avg_pool' else "ResNet's " + layer.replace('_', ' ')
    # plt.title(f"Cumulative number of biases detected in the embeddings\n taken from {title} \n{extra_msg}")
    if save:
        fig.savefig(f'step_biases_{layer}.pdf', format='pdf')
    plt.show()

    p_final = []

    # compute if the contrastive models are as biased as other models
    if layer == 'avg_pool':
        for i in range(len(ps)):
            contrastives = np.concatenate([ys[6:11]])
            contrastives_numpy = np.asarray([int(contrastives[c][i]) for c in range(contrastives.shape[0])])

            non_contrastives = np.concatenate([ys[0:5]])
            non_contrastives_numpy = np.asarray([int(non_contrastives[c][i]) for c in range(non_contrastives.shape[0])])
            ans, p = significance_test(contrastives_numpy, non_contrastives_numpy)
            p_final.append(p)
        print(f'contrastive models: {labels[6:11]}')
        print(f'non-contrastive models: {labels[0:5]}')
        print(f"computed p-value for testing hypothesis that contrastive models aren't\
            more biased: {sum(p_final) / len(p_final)}")


def significance_test(X: np.array, Y: np.array, n_samples=10000):
    mu_X = X.mean()
    mu_Y = Y.mean()
    s = mu_X - mu_Y
    XY = np.concatenate((X, Y))

    total_true = 0
    total_equal = 0
    total = 0

    import scipy.special
    num_partitions = int(scipy.special.binom(2 * X.shape[0], X.shape[0]))
    if num_partitions > n_samples:
        total_true += 1
        total += 1

        for i in range(n_samples - 1):
            a = XY.shape[0]
            np.random.shuffle(a)
            Xi = np.asarray(a[:X.shape[0]])
            assert 2 * len(Xi) == len(XY)
            Yi = XY[~np.isin(XY, Xi)]
            si = Xi.mean() - Yi.mean()
            if si > s:
                total_true += 1
            elif si == s:  # use conservative test
                total_true += 1
                total_equal += 1
            total += 1
    else:
        for Xi in it.combinations(XY, X.shape[0]):
            assert 2 * len(Xi) == len(XY)
            Xi = np.asarray(Xi)
            Yi = list(XY)
            for el in list(Xi):
                Yi.remove(el)
            Yi = np.asarray(Yi)
            si = Xi.mean() - Yi.mean()
            if si > s:
                total_true += 1
            elif si == s:  # use conservative test
                total_true += 1
                total_equal += 1
            total += 1

    # print(total)
    p = total_true / total
    return f"p-value for the hypothesis \"contrastives aren't more biasy than otheres\" is {p}", p




def drop_repeats(df: pd.DataFrame, layer, threshold=0.1):
    dropped_indices = []
    for i, row in df.iterrows():
        if row[f'p_random_torch_1_{layer}'] < threshold \
                and row[f'p_random_torch_2_{layer}'] < threshold \
                and row[f'p_random_torch_3_{layer}'] < threshold:
            for j in range(len(row)):
                if layer in row.index[j]:
                    row[j] = np.nan
            dropped_indices.append(i)
        df.loc[i] = row

    return df, dropped_indices


def plot_strength(df, models, labels, layers, threshold=0.1, save=True, plot_number_biases=False):
    res = {}
    bar_labels = np.zeros(len(models))
    for layer in layers:
        for model in models:
            col = model + f"_{layer}"
            p = "p_" + col
            d = "d_" + col
            if plot_number_biases:
                res[col] = df[df[p] < threshold][d].count()
            else:
                res[col] = df[df[p] < threshold][d].sum()

    l1, l2, l3, l4, l5, lavg = [], [], [], [], [], []

    for k, v in res.items():
        if "layer_0" in k:
            l1.append(v)
        elif "layer_1" in k:
            l2.append(v)
        elif "layer_2" in k:
            l3.append(v)
        elif "layer_3" in k:
            l4.append(v)
        elif "layer_4" in k:
            l5.append(v)
        elif "avg_pool" in k:
            lavg.append(v)

    fig, ax = plt.subplots()

    ax.bar(labels, l1, label='Block 1')
    ax.bar(labels, l2, bottom=l1, label='Block 2')
    ax.bar(labels, l3, bottom=np.array(l1)+np.array(l2), label='Block 3')
    ax.bar(labels, l4, bottom=np.array(l1)+np.array(l2)+np.array(l3), label='Block 4')
    ax.bar(labels, l5, bottom=np.array(l1)+np.array(l2)+np.array(l3)+np.array(l4), label='Block 5')
    ax.bar(labels, lavg, bottom=np.array(l1)+np.array(l2)+np.array(l3)+np.array(l4)+np.array(l5), label='Layer GAP')
    # ax.set_title(f"Cumulative strength of biases with p<{threshold}")

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    for y in [l1, l2, l3, l4, l5, lavg]:
        for i, m in enumerate(y):
            bar_labels[i] += m

    for bar in ax.patches:
        if bar.get_height() > 1:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height()/2 + bar.get_y(),
                f"{bar.get_height():.1f}",
                ha='center',
                va='center',
                color='w',
                size=10
            )
    ax.set_ylabel("Strength of biases (d-value)")
    # ax.set_ylabel("Number of biases")
    # ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    if save:
        fig.savefig(f'barchart_strength_p_{threshold}_no_legend.pdf', format='pdf')
    plt.show()


def process_results(data, model):
    results_df = pd.DataFrame(data).transpose()
    results_df.columns = ["X", "Y", "A", "B", "d", "p", "n_t", "n_a"]
    results_df = results_df[results_df.index.get_level_values(1).str.contains(model)]
    for c in results_df.columns[:4]:
        results_df[c] = results_df[c].str.split("/").str[-1]
    results_df["sig"] = ""
    for l in [0.10, 0.05, 0.01, 1.1]:
        results_df.sig[results_df.p < l] += "*"
    significant = results_df[results_df["sig"].str.contains("\*")]

    return significant, results_df


def compute_errors(a, b, c):
    maxs = []
    names = []

    out = pd.DataFrame(index=a.index)
    # plot only p-value errors as d-values are exact
    for col in a.filter(regex="^p_*").columns:
        col_a, col_b, col_c = a[col].astype(float).to_numpy(), b[col].astype(float).to_numpy(), c[col].astype(float).to_numpy()
        d = np.zeros((len(col_a), 4))
        d[:, 0] = col_a
        d[:, 1] = col_b
        d[:, 2] = col_c
        d[:, 3] = d[:,:3].max(axis=1) - d[:,:3].min(axis=1)
        out[col] = d[:,:3].max(axis=1) - d[:,:3].min(axis=1)

        names.append(col)

        maxs.append(max(d[:, 3]))

    # [print(f"{k}: max diff {v}") for k, v in dict(zip(list(names), maxs)).items()]
    return out


def read_ieat_pickle(fname, layer, models, load_random=False):
    results = pickle.load(open(fname, "rb"))

    all_results = []
    for model in models:
        _, all_res = process_results(data=results, model=model)
        all_results.append(all_res.reset_index(level=1, drop=True))

    ans = pd.DataFrame(columns=[f"p_{m}_{layer}" for m in models] + [f"d_{m}_{layer}" for m in models])

    for i, model in enumerate(models):
        ans[f"p_{model}_{layer}"] = all_results[i]["p"]
        ans[f"d_{model}_{layer}"] = all_results[i]["d"]

    if load_random:
        ans[f"p_random_{layer}"] = (all_results[models.index('random_torch_1')]["p"] +all_results[models.index('random_torch_2')]["p"] + all_results[models.index('random_torch_3')]["p"]) /3
        ans[f"d_random_{layer}"] = (all_results[models.index('random_torch_1')]["d"] +
                                    all_results[models.index('random_torch_2')]["d"] +
                                    all_results[models.index('random_torch_3')]["d"]) / 3
        ans.drop(columns=[f'p_random_torch_1_{layer}', f'p_random_torch_2_{layer}', f'p_random_torch_3_{layer}'],
                 inplace=True)
        ans.drop(columns=[f'd_random_torch_1_{layer}', f'd_random_torch_2_{layer}', f'd_random_torch_3_{layer}'],
                 inplace=True)
    return ans

def plot_raw_ieat_results_as_simple_bars(df, cols, layer, titles):

    def plot_group(df, name, cols, layer, ax, titles, shift):
        gs_df = df.loc[df.index.str.startswith(name)]
        gs_sum = np.zeros(len(cols))
        x = np.arange(len(titles))
        for i in range(gs_df.shape[0]):
            gs = [0 for _ in range(len(cols))]
            for idx, model in enumerate(cols):
                p = gs_df.iloc[i][f'p_{model}_{layer}']
                if p < 0.01:
                    gs[idx] += 1
            gs_sum += np.asarray(gs)
        ax.bar(x + shift, gs_sum, width=0.2, label=name)#label=gs_df.iloc[i].name)
        ax.legend()
        ax1.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(titles)
        for tick in ax.get_xticklabels():
            tick.set_rotation(75)


    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(5, 4.5)
    ax1.set_ylabel("Number of biases with p < 0.01")
    plot_group(df, 'Intersectional-Gender-Science', cols, layer, ax1, titles, shift=-0.2)
    plot_group(df, 'Intersectional-Gender-Career', cols, layer, ax1, titles, shift=0)
    plot_group(df, 'Intersectional-Valence', cols, layer, ax1, titles, shift=0.2)
    ax1.legend(bbox_to_anchor=(0.3, 1.4), loc='upper center')
    ax1.axvline(2.5, linestyle="--", ymin=-.5, clip_on=False, color='black', linewidth=0.7)
    ax1.axvline(5.5, linestyle="--", ymin=-.5, clip_on=False, color='black', linewidth=0.7)
    ax1.axvline(10.5, linestyle="--", ymin=-.5, clip_on=False, color='black', linewidth=0.7)
    fig1.text(0.16, 0.02, "Geometric", fontsize=8)
    fig1.text(0.38, 0.02, "Clustering", fontsize=8)
    fig1.text(0.65, 0.02, "Contrastive", fontsize=8)

    plt.tight_layout()
    plt.show()

    fig1.savefig(f'intersectional_biases_bar_camera_ready.pdf', format='pdf')


def main():
    layers = ['layer_0', 'layer_1', 'layer_2', 'layer_3', 'layer_4', 'avg_pool']

    models = ['relative_loc', 'rotation', 'odc',
              'npid', 'moco_v1',
              'moco_v2', 'byol',
              'supervised',
              'jigsaw', 'cluster_fit',
              'swav', 'simclr_vissl_200',
              'random_torch_1', 'random_torch_2', 'random_torch_3',
              ]

    n_validations = 3

    # Compute and plot error bars
    # for layer in layers:
    #     runs = []
    #     for i in range(n_validations):
    #         runs.append(read_ieat_pickle(f'validation/validation{i}_{layer}_features_with_random_w1_resnet50.pkl',
    #                                      layer=layer,
    #                                      models=models,
    #                                      load_random=True))
    #     df = compute_errors(*runs)
    #
    #     df.plot(title=f"Max. differences for p- and d-values \nbetween 3 runs. Layer name: {layer}",
    #             kind='barh', figsize=(8, 13)
    #             )
    #     plt.legend(bbox_to_anchor=(1.0, 1.0))
    #     plt.show()
    #     fig = plt.gcf()
    #     fig.savefig(f'error_bars_{layer}.pdf', format='pdf')

    summary = pd.DataFrame()

    model_names = ['Random',   'Jigsaw', 'RL', 'ClusterFit', 'Rotation',
                   'NPID', 'ODС',
                   'MoCo_v1', 'SimCLR', 'MoCo_v2',  'BYOL', 'SwAV', 'Supervised',

                   ]

    models_to_plot = ['random', 'jigsaw', 'relative_loc', 'cluster_fit', 'rotation',
                      'npid', 'odc',
                       'moco_v1',  'simclr_vissl_200', 'moco_v2', 'byol', 'swav', 'supervised',
                     ]

    # load bias-detections
    for layer in layers[::-1]:
        df = read_ieat_pickle(f'bias_detection_code/results/validation0_{layer}_features_with_random_w1_resnet50.pkl',
                              layer=layer,
                              models=models,
                              load_random=True
                              )

        for i in range(1, n_validations):
            df += read_ieat_pickle(f'bias_detection_code/results/validation{i}_{layer}_features_with_random_w1_resnet50.pkl',
                                   layer=layer,
                                   models=models,
                                   load_random=True
                                   )

        df /= n_validations

        summary = pd.concat([summary, df], axis=1)

        # plot intersectional biases as bars
        # model_names = ['Jigsaw',
        #                'Rotation',
        #                'RL',
        #                'ClusterFit',
        #                'ODС',
        #                'SwAV*',
        #                'NPID',
        #                'MoCo_v1',
        #                'MoCo_v2',
        #                'SimCLR',
        #                'BYOL',
        #                'Supervised',
        #                ]
        #
        # models_to_plot = ['jigsaw',
        #                   'rotation',
        #                   'relative_loc',
        #                   'cluster_fit',
        #                   'odc',
        #                   'swav',
        #                   'npid',
        #                   'moco_v1',
        #                   'moco_v2',
        #                   'simclr_vissl_200',
        #                   'byol',
        #                   'supervised',
        #                   ]
        # plot_raw_ieat_results_as_simple_bars(summary,
        #                                      models_to_plot,
        #                                      layer,
        #                                      model_names)



        # plot cumulative number of biases
        model_names = ['Jigsaw',
                       'Rotation',
                       'Relative Location',
                       'ClusterFit',
                       'ODС',
                       'SwAV',
                       'NPID',
                       'MoCo_v1',
                       'MoCo_v2',
                       'SimCLR',
                       'BYOL',
                       'Supervised',
                       'Random'
                       ]

        models_to_plot = ['jigsaw',
                          'rotation',
                          'relative_loc',
                          'cluster_fit',
                          'odc',
                          'swav',
                          'npid',
                          'moco_v1',
                          'moco_v2',
                          'simclr_vissl_200',
                          'byol',
                          'supervised',
                          'random'
                          ]
        plot_cumulative(summary, layer, models=models_to_plot, labels=model_names, save=True)

    # plot strength of biases
    # plot_strength(summary, models_to_plot, model_names, layers, threshold=0.01, save=True, plot_number_biases=False)

    # plot number of biases
    # plot_strength(summary, models_to_plot, model_names, layers, threshold=0.01, save=True, plot_number_biases=True)

    # plot all biases using markers
    # plot_biases(df=summary, layers=layers, model_names=models_to_plot, model_titles=model_names)

    # plot evolution of biases through the layers
    # plot_evolution_of_biases(df=summary, layers=layers, model_names=models_to_plot)


if __name__ == '__main__':
    main()