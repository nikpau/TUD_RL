import csv
import os

#matplotlib.use("agg")
import matplotlib.pyplot as plt
import pandas as pd
from helper_fnc import exponential_smoothing
from collections import namedtuple
from pathlib import Path

# Green, brown
COLORS = namedtuple("COLORS", ["green","yellow","brown","bluish","Cream","whiteish"])
colors = COLORS("#205855","#FF7D00","#6C534E","#8DA9C4", "#EEF4ED","#f8f9fa")

def plot_from_progress(dir, alg, env_str, info=None):
    """Plots based on a given 'progress.txt' the evaluation return, Q_values and losses.

    Args:
        dir (string):     directory of 'progress.txt', most likely something like experiments/some_number
        env_str (string): name of environment 
        alg (string):     used algorithm
        info (string):    further information to display in the header
    """

    # Load logo
    dirname = Path(os.path.dirname(__file__))
    logo = plt.imread(f"{dirname.parents[0]}/assets/logo.png")

    # open progress file and load it into pandas
    with open(f"{dir}/progress.txt") as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)

    df = pd.DataFrame(d)
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df = df.astype(float)

    runtime = df["Runtime_in_h"].iloc[-1].round(3)

    # create plot
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))

    fig.set_facecolor(colors.whiteish)
    ax[0,0].set_facecolor(colors.whiteish)
    ax[0,1].set_facecolor(colors.whiteish)
    ax[1,0].set_facecolor(colors.whiteish)
    ax[1,1].set_facecolor(colors.whiteish)
    
    # define title
    if info is not None:
        fig.suptitle(f"{alg} ({info}) | {env_str} | Runtime (h): {runtime}")
    else:
        fig.suptitle(f"{alg} | {env_str} | Runtime (h): {runtime}")

    # first axis
    ax[0,0].plot(
        df["Timestep"], 
        df["Avg_Eval_ret"], 
        label = "Avg. test return",
        color = colors.green
    )
    ax[0,0].plot(
        df["Timestep"], 
        exponential_smoothing(df["Avg_Eval_ret"].values), 
        label = "Exp. smooth. return",
        color = colors.yellow
    )
    ax[0,0].legend()
    ax[0,0].set_xlabel("Timestep")
    ax[0,0].set_ylabel("Test return")

    # second axis
    if "Avg_Q_val" in df.columns:
        ax[0,1].plot(df["Timestep"], df["Avg_Q_val"],color = colors.green)
        ax[0,1].set_ylabel("Avg_Q_val")
        ax[0,1].set_xlabel("Timestep")
    
    # third axis
    if "Loss" in df.columns:
        ax[1,0].plot(df["Timestep"], df["Loss"],color = colors.green)
        ax[1,0].set_xlabel("Timestep")
        ax[1,0].set_ylabel("Loss")

    if "Critic_loss" in df.columns and "Actor_loss" in df.columns:
        ax[1,0].plot(df["Timestep"], df["Critic_loss"], label="Critic", color = colors.brown)
        ax[1,0].plot(df["Timestep"], df["Actor_loss"], label="Actor", color = colors.bluish)
        ax[1,0].legend()

    # fourth axis
    ax[1,1].set_xlabel("Timestep")
    
    if all(ele in df.columns for ele in ["Avg_bias", "Std_bias", "Max_bias", "Min_bias"]):
        ax[1,1].plot(df["Timestep"], df["Avg_bias"], label="Avg. bias",color = colors.green)
        ax[1,1].plot(df["Timestep"], df["Std_bias"], label="Std. bias",color = colors.yellow)
        ax[1,1].plot(df["Timestep"], df["Max_bias"], label="Max. bias",color = colors.brown)
        ax[1,1].plot(df["Timestep"], df["Min_bias"], label="Min. bias",color = colors.bluish)
        ax[1,1].legend()

    
    # Add logo to plot
    logoax = fig.add_axes([0.7,0.72,0.2,0.2], anchor='NE', zorder=1)
    logoax.imshow(logo)
    logoax.axis('off')
    
    # safe figure and close
    plt.savefig(f"{dir}/{alg}_{env_str}1.pdf")
    plt.close()