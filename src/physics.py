import numpy as np, pandas as pd

def index_nodes(nodes_e):
    ids = list(nodes_e["Номер"].astype(int))
    return ids, {k: i for i, k in enumerate(ids)}

def voltages(nodes_e):
    return nodes_e["V"].to_numpy() * np.exp(1j*np.deg2rad(nodes_e["Delta"].to_numpy()))

def branch_calc(V, lines_e, ix):
    S = np.zeros(len(ix), complex); out = []
    for r in lines_e.itertuples(index=False):
        s, t = ix[int(r.N_s)], ix[int(r.N_e)]
        y = 1/complex(r.R, r.X); b = r.B*1e-6
        if r.type == 0:
            Is = y*(V[s]-V[t]) - 1j*b*0.5*V[s]
            It = y*(V[t]-V[s]) - 1j*b*0.5*V[t]
        else:
            k = r.Kt; Iz = y*(V[s]-V[t]/k)
            Is = Iz - 1j*b*V[s]; It = -Iz/k
        S[s] += V[s]*np.conj(Is); S[t] += V[t]*np.conj(It)
        out.append((int(r.Номер), int(r.type), abs(Is)/np.sqrt(3)*1000, abs(It)/np.sqrt(3)*1000, r.I_max))
    return S, pd.DataFrame(out, columns=["line","type","I_s","I_t","I_max"])
