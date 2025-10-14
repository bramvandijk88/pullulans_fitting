"""

Fit Erlang-chain ODE model for A. pullulans
===========================================

To fit the data of differentiation of A. pullulans cells, we model three cell types: 
Blastoconidia (B), Swollen cells (S), and Hyphae (H). However, instead of using only
three variables, we represent each cell population as a chain of sub-stages to capture
the "development" of cells more accurately. Biologically, these chains may represent
the completion of different cell stages, such as DNA replication, cell growth, and the
machinery for cell division or differentiation.  

We varied the lenghts of the three chains (the B, S, and H chain), and fitted parameters
for the transitions, blastoconidia birth rates, and waiting times. We also fitted decay, but
constrained these to low values to avoid overfitting based on unrealistic rates of 
cell death. 

Parameter names: 
    p_BS, p_SH (transitions per hour),
    r_SB, r_HB, r_BB (birth rates per hour),
    tau_B, tau_S, tau_H (waiting times) in minutes
Fitted with differential evolution (scipy.optimize) with loss function:
    SSE(B,S,H counts vs observed data up to T=15 h) + nongrowth_penalty
    Because we fit percentage data, we add the nongrowth penalty to avoid 
    trivial solutions.

"""

import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from dataclasses import dataclass
import argparse
import os

COLORS = {
    "B": "#41aa8a",   # teal
    "S": "#fa8b63",   # orange
    "H": "#ff9be8"    # pink
}

def make_outdir(args):
    # strip ".csv" from path to get dataset name
    dataset = os.path.basename(DATA_PATH).replace(".csv","")

    folder = (f"{dataset}_"
              f"s{SEED}_"
              f"kB{kB}_kS{kS}_kH{kH}_"
              f"p{POPSIZE}_i{MAXITER}")
    os.removedirs(folder) if os.path.exists(folder) else None
    os.makedirs(folder, exist_ok=True)
    return folder

# =========================
# 1) GLOBAL CONFIG
# =========================
DATA_PATH = "CBS58475.csv"   # change to your strain CSV
#DATA_PATH = "CBS100280.csv"   # change to your strain CSV
#DATA_PATH = "CBS109810.csv"   # change to your strain CSV
#DATA_PATH = "CBS140240.csv"   # change to your strain CSV

T_FIT_MAX = 15.0    # fit only up to 15 h (data horizon)
T_TARGET  = 72.0    # enforce %B≈TARGET_B here

# Initial conditions (from CSV at t=0, will be set later)
B0_init, S0_init, H0_init = None, None, None

# Optimizer knobs
POPSIZE  = 32
MAXITER  = 32

SEED     = 1
POLISH   = False    # set True to enable L-BFGS-B polishing (can be slow!)
MANUAL   = False     # do not fit parameters but use defaults and make plots
PLOT = True

# Erlang sub-stage counts (user can modify)
kB, kS, kH = 5, 10, 10

# Use defaults when MANUAL=TRUE, else fit parameters
PARAMS = {
    # These are the parameters of the best fit
    # identified in 10,000 fits that varied chain lengths. 
    # It was found for chain length B=5, S=10, H=10 (as set above)
    "p_BS": 7.04595, 
    "p_SH": 3.27765,
    "r_SB": 0.0296091,
    "r_HB": 7.55313,
    "r_BB": 0.101539,
    "tau_B": 528.083,
    "tau_S": 651.442,
    "tau_H": 989.059,
    "d_B": 0.00594637,
    "d_S": 0.00373772,
    "d_H": 0.00950423,
} 

d_B=0.00594637
d_H=0.00950423
d_S=0.00373772
p_BS=7.04595
p_SH=3.27765
r_BB=0.101539
r_HB=7.55313
r_SB=0.0296091
tau_B=528.083
tau_H=989.059
tau_S=651.442

# Plotting toggle
SHOW_SUBSTAGES = False

# Solver tolerances
RTOL_FIT, ATOL_FIT = 1e-6, 1e-8    # during fitting (fast)
RTOL_FINAL, ATOL_FINAL = 1e-12, 1e-16  # for final plots


# =========================
# 2) DATA LOADING
# =========================
def load_counts(path):
    df = pd.read_csv(path, sep=";")
    mapping = {"1Blastoconidia":"B","2Swollen cells":"S","4Hyphae":"H"}
    df = df[df["Category"].isin(mapping)].copy()
    df["cat"] = df["Category"].map(mapping)
    agg = df.groupby(["Tijd","cat"], as_index=False)["n"].sum()
    wide = agg.pivot_table(index="Tijd", columns="cat", values="n", fill_value=0).reset_index()
    wide = wide.rename(columns={"Tijd":"t"}).sort_values("t").reset_index(drop=True)
    for k in ["B","S","H"]:
        if k not in wide.columns: wide[k]=0
    return wide[["t","B","S","H"]]


# =========================
# 3) ERLANG-CHAIN ODE
# =========================
@dataclass
class ChainConfig:
    kB:int; kS:int; kH:int
    alpha_B:float; alpha_S:float; alpha_H:float

def minutes_to_hours(m): return m/60.0

def build_chain_config(params, kB,kS,kH):
    tau_B = minutes_to_hours(params["tau_B"])
    tau_S = minutes_to_hours(params["tau_S"])
    tau_H = minutes_to_hours(params["tau_H"])
    return ChainConfig(
        kB, kS, kH,
        alpha_B = kB/max(tau_B,1e-9),
        alpha_S = kS/max(tau_S,1e-9),
        alpha_H = kH/max(tau_H,1e-9)
    )

def rhs_factory(params, C):
    p_BS, p_SH = params["p_BS"], params["p_SH"]
    r_SB, r_HB, r_BB = params["r_SB"], params["r_HB"], params["r_BB"]
    d_B, d_S, d_H   = params["d_B"], params["d_S"], params["d_H"]
    kB,kS,kH = C.kB,C.kS,C.kH
    aB,aS,aH = C.alpha_B,C.alpha_S,C.alpha_H
    n_state = kB+kS+kH

    def rhs(t,y):
        y = np.maximum(y,0)
        B,S,H = y[0:kB], y[kB:kB+kS], y[kB+kS:n_state]
        dB,dS,dH = np.zeros_like(B), np.zeros_like(S), np.zeros_like(H)

        # Erlang flows
        for arr, a, d in [(B, aB, dB), (S, aS, dS), (H, aH, dH)]:
            if len(arr) > 1:
                d[0] += -a * arr[0]
                for i in range(1, len(arr)-1):
                    d[i] += a*arr[i-1] - a*arr[i]
                d[-1] += a*arr[-2]

        # last-stage eligibles
        B_last = B[-1] if kB>0 else 0
        S_last = S[-1] if kS>0 else 0
        H_last = H[-1] if kH>0 else 0

        # transitions
        #ft=1-t/(t+10)
        #ft=1-0.01*t
        ft=1
        flow_BS = p_BS*B_last*ft
        if kB: dB[-1]-=flow_BS
        if kS: dS[0]+=flow_BS
        flow_SH = p_SH*S_last
        if kS: dS[-1]-=flow_SH
        if kH: dH[0]+=flow_SH

        # births
        #N = B.sum() + S.sum() + H.sum()
        births_from_S = r_SB*S_last
        births_from_H = r_HB*H_last
        births_from_B = r_BB*B_last
        births_total = births_from_S+births_from_H+births_from_B
        if kB>0: dB[0]+=births_total

        # decay (applies only to last stages)
        if kB: dB[-1] -= d_B*B_last
        if kS: dS[-1] -= d_S*S_last
        if kH: dH[-1] -= d_H*H_last
        
        # Store as attributes for later use
        rhs.last_births = (births_from_B, births_from_S, births_from_H, births_total)
        return np.concatenate([dB,dS,dH])
    return rhs

def simulate(params, B0,S0,H0, t_end=72, dt_out=0.1, rtol=1e-6, atol=1e-8):
    C = build_chain_config(params,kB,kS,kH)
    rhs = rhs_factory(params,C)
    y0 = np.zeros(C.kB+C.kS+C.kH); y0[0]=B0
    if C.kS: y0[C.kB]=S0
    if C.kH: y0[C.kB+C.kS]=H0
    t_eval = np.arange(0,t_end+1e-9,dt_out)
    sol = solve_ivp(rhs,(0,t_end),y0,method="BDF",t_eval=t_eval,rtol=rtol,atol=atol)
    
    # --- births record ---
    births_records = []
    for ti, yi in zip(sol.t, sol.y.T):
        rhs(ti, yi)   # update rhs.last_births
        by_B, by_S, by_H, total = rhs.last_births
        births_records.append((ti, by_B, by_S, by_H, total))
    births_df = pd.DataFrame(births_records, columns=["t","by_B","by_S","by_H","total"])


    Y = np.maximum(sol.y,0)
    Bsol = Y[0:C.kB,:]
    Ssol = Y[C.kB:C.kB+C.kS,:]
    Hsol = Y[C.kB+C.kS:,:]
    Btot,Stot,Htot = Bsol.sum(0),Ssol.sum(0),Hsol.sum(0)
    Ntot = Btot+Stot+Htot
    sim_df = pd.DataFrame({
        "t":sol.t,"B":Btot,"S":Stot,"H":Htot,
        "B_pct":100*Btot/np.maximum(Ntot,1e-12),
        "S_pct":100*Stot/np.maximum(Ntot,1e-12),
        "H_pct":100*Htot/np.maximum(Ntot,1e-12)
    })
    return sim_df,(Bsol,Ssol,Hsol), births_df

def find_crossover_time(sim_df, Bsol, Ssol, Hsol, params, C):
    """Return time when H→B births exceed S→B births. 
    If never, return 72h (or end of sim)."""
    r_SB, r_HB = params["r_SB"], params["r_HB"]

    # last-stage subpops
    S_last = Ssol[-1,:]
    H_last = Hsol[-1,:]

    births_S = r_SB * S_last
    births_H = r_HB * H_last
    
    #print("S→B births:", births_S)
    #print("H→B births:", births_H)

    # Find first t where births_H > births_S
    mask = births_H > births_S
    if np.any(mask):
        idx = np.argmax(mask)   # first True
        return float(sim_df["t"].iloc[idx])
    else:
        return float(sim_df["t"].iloc[-1])  # last time (e.g. 72h)

# =========================
# 4) FITTING
# =========================
def fit_params(df):
    global B0_init,S0_init,H0_init
    B0_init = df.iloc[0]["B"]
    S0_init = df.iloc[0]["S"]
    H0_init = 0
    print(B0_init,S0_init,H0_init)

    obs = df[df["t"] <= T_FIT_MAX].copy()

    bounds = [
        (0, 10),  # p_BS
        (0, 10),  # p_SH
        (0, 10),  # r_SB
        (0, 10),  # r_HB
        (0, 10),  # r_BB
        (5, 1000), # tau_B
        (5, 1000), # tau_S
        (5, 1000), # tau_H
        (0, 0.01),   # d_B
        (0, 0.01),   # d_S
        (0, 0.01)    # d_H
    ]
    keys = ["p_BS","p_SH","r_SB","r_HB","r_BB",
            "tau_B","tau_S","tau_H",
            "d_B","d_S","d_H"]

    loss_history = []
    nonlocal_iter = [0]

    def loss(theta):
        params = dict(zip(keys, theta))
        sim_df, _, _ = simulate(params, B0_init, S0_init, H0_init,
                             t_end=T_TARGET, rtol=RTOL_FIT, atol=ATOL_FIT)
        sim_fit = sim_df.set_index("t").loc[obs["t"]]
        sse = ((sim_fit[["B","S","H"]].values - obs[["B","S","H"]].values)**2).sum()
        B72 = sim_df.loc[sim_df["t"] == T_TARGET, "B_pct"].values[0]
        row = sim_df.iloc[-1]
        penalty = 0.0
        # growth penalty
        if sim_df["B"].iloc[-1] + sim_df["S"].iloc[-1] + sim_df["H"].iloc[-1] < \
           (B0_init + S0_init + H0_init):
            penalty += 1e6  # large penalty for non-growth

        return sse + penalty

    def progress_callback(xk, convergence):
        nonlocal_iter[0] += 1
        current_loss = loss(xk)
        loss_history.append(current_loss)
        pct = 100 * nonlocal_iter[0] / MAXITER
        print(f"\r>>> Iter {nonlocal_iter[0]:3d}/{MAXITER} "
              f"({pct:5.1f}%) | Current loss = {current_loss:.3e}",
              end="", flush=True)

    print(">>> Starting parameter fitting ...")

    res = differential_evolution(
        loss, bounds,
        popsize=POPSIZE, maxiter=MAXITER,
        seed=SEED, disp=False, 
        mutation=(0.5, 1.5),
        callback=progress_callback,
        polish=POLISH
    )
    print("\n>>> Finished fitting.")

    return dict(zip(keys, res.x)), res.fun, loss_history

#####
# Summarize fitting progress in ASCII
#####
def ascii_summary(loss_history):
    max_loss = max(loss_history)
    print("\n>>> Fitting summary")
    print(f"Initial loss: {loss_history[0]:.3e}")
    print(f"Final loss:   {loss_history[-1]:.3e}")
    improvement = 100 * (1 - loss_history[-1]/loss_history[0])
    print(f"Improvement:  {improvement:.1f} %\n")

    for i, L in enumerate(loss_history[::2]):  # every 2nd point
        bar_len = int(30 * (L/max_loss))
        print(f"{i*2:3d} [{'#'*bar_len}{' '*(30-bar_len)}] {L:.2e}")
        
        
# =========================
# Command line options
# =========================
def get_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manual",action="store_true",help="Run with fixed manual params, skip fitting")
    ap.add_argument("--noplot",action="store_false",help="Skip matplotlib plotting window (png still produced!)")
    ap.add_argument("--seed",type=int,help="RNG seed for fitting")
    ap.add_argument("--kB",type=int,help="Length of the B chain")
    ap.add_argument("--kS",type=int,help="Length of the S chain")
    ap.add_argument("--kH",type=int,help="Length of the H chain")
    for k in PARAMS.keys():
        ap.add_argument(f"--{k}",type=float,help=f"Set {k} manually")
    return ap.parse_args()


# =========================
# Plot results
# =========================
def plot_results(df, sim_df, sub_stages, log_counts=False, crossover_time=None,outdir=None,fname="fit_results.png", show=True):
    Bsol,Ssol,Hsol,blasto_only_prediction=sub_stages
    eps=1e-12
    N_tot=sim_df["B"]+sim_df["S"]+sim_df["H"]

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7,8),sharex=True)

    # (A) Percentages
    ax1.plot(sim_df["t"],sim_df["B_pct"],label="B% total",color=COLORS["B"],lw=2.0)
    ax1.plot(sim_df["t"],sim_df["S_pct"],label="S% total",color=COLORS["S"],lw=2.0)
    ax1.plot(sim_df["t"],sim_df["H_pct"],label="H% total",color=COLORS["H"],lw=2.0)
    if SHOW_SUBSTAGES:
        for arr,color in [(Bsol,"tab:blue"),(Ssol,"tab:orange"),(Hsol,"tab:green")]:
            for i in range(arr.shape[0]):
                ax1.plot(sim_df["t"],100*arr[i,:]/np.maximum(N_tot,eps),color=color,lw=0.8,alpha=0.2)
    tot=df[["B","S","H"]].sum(1)
    ax1.scatter(df["t"],100*df["B"]/tot,color=COLORS["B"],marker="o")
    ax1.scatter(df["t"],100*df["S"]/tot,color=COLORS["S"],marker="s")
    ax1.scatter(df["t"],100*df["H"]/tot,color=COLORS["H"],marker="^")
    ax1.set_ylim(0,100); ax1.set_ylabel("% of total")
    # Add crossover_time
    if crossover_time is not None:
        for ax in (ax1, ax2):   # both percentage and count plots
            ax.axvline(crossover_time, color="red", ls="--", lw=1.5,
                    label="S→B = H→B")

    # (B) Counts
    ax2.plot(sim_df["t"],sim_df["B"],label="B sim",color=COLORS["B"])
    ax2.plot(sim_df["t"],sim_df["S"],label="S sim",color=COLORS["S"])
    ax2.plot(sim_df["t"],sim_df["H"],label="H sim",color=COLORS["H"])
    #ax2.plot(sim_df["t"],blasto_only_prediction,label="Predicted B's",color="blue",lw=1.0,linestyle="--")
    # Set axis to max of B, S, or H (not blasto_only_prediction)
    ymax = max(sim_df[["B","S","H"]].max())
    ax2.set_ylim(0.1,ymax*1.1)
    ax2.scatter(df["t"],df["B"],color=COLORS["B"],marker="o")
    ax2.scatter(df["t"],df["S"],color=COLORS["S"],marker="s")
    ax2.scatter(df["t"],df["H"],color=COLORS["H"],marker="^")
    if log_counts: ax2.set_yscale("log")
    ax2.set_xlabel("Time (h)"); ax2.set_ylabel("Count")

    fig.tight_layout(); 
    if outdir is not None:
        fig.savefig(os.path.join(outdir,fname),dpi=300); 
    if show:
        plt.show()


# =========================
# MAIN
# =========================
def main():
    global SEED
    global kB,kS,kH
    args=get_args()
    print(args)
    if args.seed is not None:
        SEED = args.seed
    if args.kB is not None:
        kB = args.kB
    if args.kS is not None:
        kS = args.kS
    if args.kH is not None:
        kH = args.kH
    df=load_counts(DATA_PATH)
    global B0_init,S0_init,H0_init
    B0_init,S0_init,H0_init=df.iloc[0][["B","S","H"]]
    
    if MANUAL or args.manual:
        params=PARAMS.copy()
        for k,v in vars(args).items():
            if k in params and v is not None: params[k]=v
        print(">>> Manual run with params:",params)
        
        # Simulate with final params
        sim_df,(Bsol,Ssol,Hsol), births_df=simulate(params,B0_init,S0_init,H0_init,
                                     t_end=T_TARGET,rtol=RTOL_FINAL,atol=ATOL_FINAL)
        # Find crossover time assuming we started with any hyphae
        sim_df_crossover,(Bsolp,Ssolp,Hsolp), births_df=simulate(params,B0_init,S0_init,0,
                                     t_end=T_TARGET,rtol=RTOL_FINAL,atol=ATOL_FINAL)
        C = build_chain_config(params, kB, kS, kH)  # to pass into helper
        crossover_time = find_crossover_time(sim_df_crossover, Bsolp, Ssolp, Hsolp, params, C)
        
        # Add a column for the predicted blastoconidia births if only blastocodia would be dividing at 0.15/h
        blastodivision_only_prediction = 365**(0.15*births_df["t"])
        plot_results(df,sim_df,(Bsol,Ssol,Hsol,blastodivision_only_prediction),log_counts=True,crossover_time=crossover_time,outdir=None, show=args.noplot)
    else:
        # Make output directory
        outdir = make_outdir(args)
        print(f">>> Results will be saved in: {outdir}")
        # Fit the parameters
        

        params, loss, loss_history = fit_params(df)
        

        ### Rename folder to include loss value
        new_outdir = f"loss_{loss:.5f}_{outdir}"
        # Remove new outdir if it already exists (regardless of content)
        if os.path.exists(new_outdir):
            shutil.rmtree(new_outdir)

        os.rename(outdir, new_outdir)
        outdir = new_outdir
        print(f">>> Renamed results folder to: {outdir}")
        # Report the best parameters found
        print(">>> Best params:")
        for k in sorted(params.keys()):
            print(f"   {k:12s}: {params[k]:.6g}")
        # Save params for re-use
        with open(os.path.join(outdir,"fitted_params.txt"),"w") as f:
            for k,v in sorted(params.items()):
                f.write(f"{k}={v:.6g}\n")
        # Save loss history to a files
        pd.DataFrame({"iter":range(len(loss_history)),"loss":loss_history})\
            .to_csv(os.path.join(outdir,"loss_history.csv"), index=False)
        # Ascii summary of fitting process
        ascii_summary(loss_history)
        # Simulate with final params
        sim_df,(Bsol,Ssol,Hsol), births_df=simulate(params,B0_init,S0_init,H0_init,
                                     t_end=T_TARGET,rtol=RTOL_FINAL,atol=ATOL_FINAL)
        # Find and save crossover time
        C = build_chain_config(params, kB, kS, kH)  # to pass into helper
        print(params, C)
        crossover_time = find_crossover_time(sim_df, Bsol, Ssol, Hsol, params, C)
        print(f">>> Cross-over time (S→B vs H→B): {crossover_time:.2f} h")
        with open(os.path.join(outdir,"crossover_time.txt"),"w") as f:
            f.write(f"{crossover_time:.3f}\n")
        # Save curves to CSV
        sim_df.to_csv(os.path.join(outdir,"sim_run.csv"), index=False)
        births_df.to_csv(os.path.join(outdir,"new_blastoconidia.csv"),index=False)
        # Plot the results
        blastodivision_only_prediction = 365**(0.15*births_df["t"])
        plot_results(df,sim_df,(Bsol,Ssol,Hsol,blastodivision_only_prediction),log_counts=True,crossover_time=crossover_time,outdir=outdir,show=args.noplot)

    

if __name__=="__main__":
    main()