import ctfishpy

import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path
import pandas as pd
import monai
import math
import torch
import seaborn as sns
import scipy


if __name__ == '__main__':
    dataset_path = '/home/wahab/Data/HDD/uCT/'
    # dataset_path = '/home/ak18001/Data/HDD/uCT/'
    # dataset_path = '/mnt/scratch/ak18001/uCT/'


    ctreader = ctfishpy.CTreader(dataset_path)
    master = ctreader.master
    print(master)

    data_path = "output/results/jaw/jawunet_data230124.csv"
    cols = ["Dens1","Dens2","Dens3","Dens4","Vol1","Vol2","Vol3","Vol4"]
    sub_bones = ["L_Dentary", "R_Dentary", "L_Quadrate", "R_Quadrate"]
    all_genes = ['wt', 'barx1', 'arid1b', 'col11a2', 'panther', 'giantin', 'chst11', 'runx2',
    'wnt16', 'ncoa3', 'gdf5', 'mcf2l', 'dot1', 'scxb', 'scxa', 'col9', 'sp7', 'col11 ',
    'vhl', 'tert', 'chsy1', 'col9a1', 'ras', 'atg', 'ctsk', 'spp1', 'il10', 'il1b',
    'irf6',]
    genes_to_include = ['wt', 'barx1', 'arid1b', 'col11a2', 'giantin', 'chst11', 'runx2',
    'wnt16', 'ncoa3', 'gdf5', 'mcf2l', 'scxa', 'sp7', 'col11 ',
    'chsy1', 'atg', 'ctsk', 'spp1']
    genes_to_analyse = ['wt']
    # age_bins = [6,12,24,36] # for age in months
    age_bins = [6,12] # for age in months

    # import pdb; pdb.set_trace()

    # bin  ages
    # master.age = pd.cut(master.age, bins = len(age_bins), labels=age_bins, 
    #             include_lowest=True)
    # ages = [ctreader.trim(master, 'age', [b]) for b in bins]
    # print(master)

    # clean strain naming
    for n in master.index:
        strain = master['strain'].loc[n]
        geno = master['genotype'].loc[n]
        # print(strain, geno)
        if geno == 'wt': 
            master['strain'].loc[n] = 'wt'

    df = pd.read_csv(data_path, index_col=0)
    dens = df[cols[:len(sub_bones)]]
    vols = df[cols[len(sub_bones):]]
    dens.columns = sub_bones
    vols.columns = sub_bones

    # included = df.index
    # included = [i-1 for i in included]
    # master = master.iloc[included]

    dens_df = pd.concat([dens, master], axis=1)
    # dens_df = ctreader.trim(dens_df, 'genotype', ['wt', 'het', 'hom'])
    dens_df = ctreader.trim(dens_df, 'strain', genes_to_analyse)
    vols_df = pd.concat([vols, master], axis=1)
    # vols_df = ctreader.trim(vols_df, 'genotype', ['wt', 'het', 'hom'])
    vols_df = ctreader.trim(vols_df, 'strain', genes_to_analyse)

    id_vars = ['age', 'strain', 'genotype', 'length']
    value_vars = sub_bones

    # melt dataframes
    dens_df = dens_df.melt(id_vars = id_vars, value_vars = value_vars, var_name="Bone", value_name="Density ($g.cm^{3}HA$)")
    vols_df = vols_df.melt(id_vars = id_vars, value_vars = value_vars, var_name="Bone", value_name="Volume ($mm^{3}$)")

    bones_to_analyse = ["Dentary","Quadrate"]
    b = "Dentary"

    dens_df.rename({'age': 'Age (months)', 'length': 'Length (mm)'}, axis=0, inplace=True)
    vols_df.rename({'age': 'Age (months)', 'length': 'Length (mm)'}, axis=0, inplace=True)
    dens_df.dropna(inplace=True)
    vols_df.dropna(inplace=True)

    fig , axs = plt.subplots(2, 2, sharex="col", sharey="row")
    plt.tight_layout()
    # axs = axs.flatten()
    dens_df = ctreader.trim(dens_df, "Bone", [f"L_{b}", f"R_{b}"])
    vols_df = ctreader.trim(vols_df, "Bone", [f"L_{b}", f"R_{b}"])
    
    sns.regplot(data=vols_df, x="age", y="Volume ($mm^{3}$)", ax=axs[0,0], color="gold")
    axs[0,0].set_xlabel("")
    x = np.array(dens_df["age"])
    y = np.array(vols_df["Volume ($mm^{3}$)"])
    results = scipy.stats.linregress(x,y)
    print(results)
    print(f"{x.mean():.2f}, {x.std():.2f}, {y.mean():.2f}, {y.std():.2f}")
    axs[0,0].set_title(f"$A$ - $R^{2}$ = {results.rvalue:.2f} slope = {results.slope:.2f}, intercept = {results.intercept:.2f}")
    
    sns.regplot(data=vols_df, x="length", y="Volume ($mm^{3}$)", ax=axs[0,1], color="yellowgreen")
    axs[0,1].set_xlabel("")
    axs[0,1].set_ylabel("")
    x = np.array(dens_df["length"])
    y = np.array(vols_df["Volume ($mm^{3}$)"])
    results = scipy.stats.linregress(x,y)
    print(results)
    print(f"{x.mean():.2f}, {x.std():.2f}, {y.mean():.2f}, {y.std():.2f}")
    axs[0,1].set_title(f"$B$ - $R^{2}$ = {results.rvalue:.2f} slope = {results.slope:.2f}, intercept = {results.intercept:.2f}")
    
    sns.regplot(data=dens_df, x="age", y="Density ($g.cm^{3}HA$)", ax=axs[1,0], color="red")
    axs[1,0].set_xlabel("Age (months)")
    x = np.array(dens_df["age"])
    y = np.array(dens_df["Density ($g.cm^{3}HA$)"])
    results = scipy.stats.linregress(x,y)
    print(results)
    print(f"{x.mean():.2f}, {x.std():.2f}, {y.mean():.2f}, {y.std():.2f}")
    axs[1,0].set_title(f"$C$ - $R^{2}$ = {results.rvalue:.2f} slope = {results.slope:.2f}, intercept = {results.intercept:.2f}")
    
    sns.regplot(data=dens_df, x="length", y="Density ($g.cm^{3}HA$)", ax=axs[1,1], color="firebrick")
    axs[1,1].set_xlabel("Length (mm)")
    axs[1,1].set_ylabel("")
    x = np.array(dens_df["length"])
    y = np.array(dens_df["Density ($g.cm^{3}HA$)"])
    results = scipy.stats.linregress(x,y)
    print(results)
    print(f"{x.mean():.2f}, {x.std():.2f}, {y.mean():.2f}, {y.std():.2f}")
    axs[1,1].set_title(f"$D$ - $R^{2}$ = {results.rvalue:.2f} slope = {results.slope:.2f}, intercept = {results.intercept:.2f}")

    
    fig.set_figwidth(9)
    fig.set_figheight(7)
    plt.savefig("output/results/jaw/age_v_len_dv.png")
    
    """
    OUTPUT
    
    LinregressResult(slope=0.007850857253695224, intercept=0.7189302826173424, rvalue=0.45673568557198807, pvalue=5.925494228367233e-20, stderr=0.0008081771947683731, intercept_stderr=0.016945831120487403)
LinregressResult(slope=0.016084221016186874, intercept=0.4110383904059282, rvalue=0.3536962459672694, pvalue=4.7694207483899124e-12, stderr=0.002248055454161584, intercept_stderr=0.06353435164384306)
LinregressResult(slope=0.004355738704920829, intercept=0.17196821171908905, rvalue=0.5385569631379681, pvalue=1.8240182554377528e-28, stderr=0.00036016748139913567, intercept_stderr=0.007551979138226342)
LinregressResult(slope=0.01696370899172899, intercept=-0.2237909886296559, rvalue=0.7928174380528079, pvalue=5.432388670238397e-79, stderr=0.0006892086823124486, intercept_stderr=0.019478357038287472)
    
    """
    print(results.rvalue, results.slope, results.intercept)



    # bones_to_analyse = ["Dentary","Quadrate"]
    # bones_to_analyse_short = ["Dent","Quad"]
    # fig_index=0
    # for i, b in enumerate(bones_to_analyse):
    #     df = ctreader.trim(dens_df, "Bone", [f"L_{b}", f"R_{b}"])
    #     for j, bin_ in enumerate(age_bins):
    #         print(i,j,b,bin_)

    #         final_df = df.loc[df['age'] == bin_]
    #         final_df.to_csv("output/test.csv")
    #         # dropna?
    #         if len(df)>0:
    #             # sns.violinplot(data=dens_df, x="genotype", y="Density ($g.cm^{3}HA$)", ax=axs[i,j], inner='stick',)
    #             sns.boxplot(data=final_df,  x="genotype", y="Density ($g.cm^{3}HA$)",ax=axs[i,j])
    #         else: print("\nSKIPPED\n")
    #         if b == "Dentary": 
    #             axs[i,j].set_xlabel("")
    #             axs[i,j].set_ylim(0,1.9)
    #         elif b == "Quadrate":
    #             axs[i,j].set_ylim(0,1.9)
    #         if bin_ > 6: axs[i,j].set_ylabel("")
    #         names, ages = get_names(final_df)
    #         sub_plot_title = f"{chr(fig_index+65)} - {bones_to_analyse_short[i]} {bin_} months, n = {ages}"
    #         axs[i,j].set_title(sub_plot_title)

    #         fig_index += 1
   
    #         # STATS
    #         stats = {}
    #         for name in names:
    #             stats[name] = {}
    #             stats[name]["array"] = final_df.loc[final_df['genotype'] == name, 'Density ($g.cm^{3}HA$)'].to_list()
    #             stats[name]["array"] = np.array(stats[name]["array"])
    #             if len(stats[name]["array"]) > 2:
    #                 stats[name]["normality"] = scipy.stats.shapiro(stats[name]["array"]).pvalue < 0.05
    #             else:
    #                 stats[name]["normality"] = False
            
    #         normal = False
    #         for name in names[1:]:
    #             if stats[name]['normality'] and stats['wt']['normality']: 
    #                 normal = True
    #             wt = stats['wt']["array"]
    #             target = stats[name]["array"]
    #             if normal :
    #                 stats[name]["sig"] = scipy.stats.ttest_ind(wt, target, equal_var=False)
    #             else:
    #                 stats[name]["sig"] = scipy.stats.mannwhitneyu(wt, target, nan_policy="omit")
                    
    #             sig = stats[name]["sig"].pvalue
    #             print(f"{b,bin_} wt vs {name}  |   normal = {normal}   |   p = {sig}")
    #             # if name == "hom": print(stats[name]["array"])
                
                
    #         # wt = densities.loc[(densities['genotype'] == 'wt 6(m)') & (densities['Otoliths'] == oto), 'Density ($g.cm^{3}HA$)'].tolist()
    #         # mut = densities.loc[(densities['genotype'] == '$col11a2$ -/- 6(m)') & (densities['Otoliths'] == oto), 'Density ($g.cm^{3}HA$)'].tolist()
    #         # wt = np.array(wt)
    #         # mut = np.array(mut)
    #         # normality[oto] = [scipy.stats.shapiro(wt), scipy.stats.shapiro(mut)]

    # # import pdb;pdb.set_trace()
    
    # fig.set_figwidth(7)
    # fig.set_figheight(9)
    # fig.suptitle(f"Densities {genes_to_analyse[-1]}", fontsize=16, y=1.025)

    # # plt.subplots_adjust(left=0.125, bottom=0.125, right=0.25, top=0.25, wspace=0.25, hspace=0.25)
    # # plt.gcf().subplots_adjust(bottom=0.5)
    # plt.savefig(f"output/results/jaw/genes/dens_{genes_to_analyse[-1]}.png", bbox_inches="tight")
    # plt.clf()
    
    


