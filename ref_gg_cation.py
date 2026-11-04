"""
Классический эталон: локализация дырки в GG⁺ теломерного G-квадруплекса.

Геометрия: остатки G2 (5′) и G3 (3′) тракта GGG из кристаллической структуры
параллельного теломерного G-квадруплекса человека с K⁺ (PDB 1KF1, Parkinson et al.
2002). Тяжёлые атомы оснований -> на них накладывается (Kabsch) гуанин с H,
оптимизированный B3LYP/cc-pVDZ (results/g_cation.json); N9 закрыт атомом H.

Окружения: gas | K (ионы K⁺ кристалла как точечные заряды +1) | PCM (вода, ε=78.4)

Методы (cc-pVDZ):
  UHF, UB3LYP, UCAM-B3LYP — дырка на каждом G по Малликену:
      h_f = n_f(GG) − n_f(GG⁺)   (сумма = 1)
  SA2-CASSCF(7,6) + SC-NEVPT2 (gas, K) — π HOMO/HOMO-1 каждого G + 2 π*;
      два нижних дублета GG⁺, их дырки, и FCD-модель (Voityuk):
      V, Δ (диабатические) из NEVPT2-энергий и CASSCF (переходных) плотностей.
"""
import argparse
import json
import os
import time

import numpy as np
from pyscf import gto, scf, dft, mcscf, mrpt, qmmm, solvent

HA2EV = 27.211386
MEM = 18000
OUT = "results/gg_cation.json"
BASE_NAMES = ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"]
# индексы тяжёлых атомов гуанина в SMILES "Nc1nc2[nH]cnc2c(=O)[nH]1" (RDKit)
TEMPLATE_IDX = {"N2": 0, "C2": 1, "N3": 2, "C4": 3, "N9": 4, "C8": 5, "N7": 6,
                "C5": 7, "C6": 8, "O6": 9, "N1": 10}


# ---------- геометрия ----------
def read_pdb(path, chain="A"):
    res, ions = {}, []
    for line in open(path):
        rec = line[:6].strip()
        if rec in ("ATOM", "HETATM") and line[21] == chain:
            name, resn, resi = line[12:16].strip(), line[17:20].strip(), int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if rec == "HETATM" and resn == "K":
                ions.append(xyz)
            elif rec == "ATOM":
                res.setdefault(resi, {"resn": resn})[name] = xyz
    return res, np.array(ions)


def kabsch(P, Q):
    """R, t: R·P + t ≈ Q."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def build_gg(residues, template_sym, template_xyz):
    sym_all, xyz_all, frag, rmsd = [], [], [], []
    tpl = np.asarray(template_xyz)
    for k, resi in enumerate(residues):
        r = residues_pdb[resi]
        assert r["resn"] == "DG", f"остаток {resi} не G"
        P = np.array([tpl[TEMPLATE_IDX[n]] for n in BASE_NAMES])
        Q = np.array([r[n] for n in BASE_NAMES])
        R, t = kabsch(P, Q)
        fit = tpl @ R.T + t
        rmsd.append(float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, 1)))))
        # проверка: H на N9 смотрит на C1′ (кэп вместо сахара)
        n9 = fit[TEMPLATE_IDX["N9"]]
        h9 = min((i for i, s in enumerate(template_sym) if s == "H"),
                 key=lambda i: np.linalg.norm(fit[i] - n9))
        cosang = np.dot(fit[h9] - n9, r["C1'"] - n9) / (np.linalg.norm(fit[h9] - n9) * np.linalg.norm(r["C1'"] - n9))
        assert cosang > 0.95, f"N9–H не по направлению N9–C1′ (cos={cosang:.2f})"
        sym_all += list(template_sym)
        xyz_all += fit.tolist()
        frag += [k] * len(template_sym)
    return sym_all, np.array(xyz_all), np.array(frag), rmsd


# ---------- анализ ----------
def frag_pop(mol, dm_ao, S, frag_of_atom):
    """Малликеновская электронная заселённость фрагментов для (переходной) плотности."""
    ps = np.einsum("ij,ji->i", dm_ao, S)
    out = np.zeros(frag_of_atom.max() + 1)
    for a, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        out[frag_of_atom[a]] += ps[p0:p1].sum()
    return out


def total_dm(mf):
    dm = mf.make_rdm1()
    return dm[0] + dm[1] if np.ndim(dm) == 3 else dm


def pi_weights(mol, mo, normals, frag_of_atom):
    """Доля МО на p-функциях вдоль локальной нормали к плоскости своего основания."""
    labels = mol.ao_labels(fmt=False)
    w = np.zeros(mo.shape[1])
    tot = np.sum(mo ** 2, axis=0)
    by_shell = {}
    for i, (atm, _, shell, comp) in enumerate(labels):
        if shell.endswith("p") and comp in ("x", "y", "z"):
            by_shell.setdefault((atm, shell), {})[comp] = i
    for (atm, _), comps in by_shell.items():
        if len(comps) != 3:
            continue
        n = normals[frag_of_atom[atm]]
        proj = n[0] * mo[comps["x"]] + n[1] * mo[comps["y"]] + n[2] * mo[comps["z"]]
        w += proj ** 2
    return w / tot


def plane_normal(xyz):
    _, _, vt = np.linalg.svd(xyz - xyz.mean(0))
    return vt[2]


# ---------- расчёты ----------
def make_mol(sym, xyz, charge):
    return gto.M(atom=list(zip(sym, map(tuple, xyz))), basis="cc-pvdz", charge=charge,
                 spin=charge % 2, verbose=0, max_memory=MEM)


def with_env(mf, env, ions):
    if env == "K":
        return qmmm.mm_charge(mf, ions, np.ones(len(ions)))
    if env == "PCM":
        mf = solvent.PCM(mf)
        mf.with_solvent.eps = 78.3553
        return mf
    return mf


def scf_method(mol, method, env, ions):
    open_shell = mol.spin != 0
    if method == "HF":
        mf = scf.UHF(mol) if open_shell else scf.RHF(mol)
    else:
        mf = dft.UKS(mol) if open_shell else dft.RKS(mol)
        mf.xc = {"B3LYP": "b3lyp", "CAM-B3LYP": "camb3lyp"}[method]
    mf = with_env(mf.density_fit(), env, ions)
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        mf = mf.newton().run()
    assert mf.converged, f"{method}/{env} не сошёлся"
    return mf


def hole_scf(sym, xyz, frag, method, env, ions):
    m0, m1 = make_mol(sym, xyz, 0), make_mol(sym, xyz, 1)
    mf0, mf1 = scf_method(m0, method, env, ions), scf_method(m1, method, env, ions)
    S = m0.intor("int1e_ovlp")
    h = frag_pop(m0, total_dm(mf0), S, frag) - frag_pop(m1, total_dm(mf1), S, frag)
    s2 = float(mf1.spin_square()[0]) if method == "HF" else None
    ip = (mf1.e_tot - mf0.e_tot) * HA2EV
    return {"hole": h.round(4).tolist(), "IP_eV": round(ip, 4), "S2": s2}


def hole_cas(sym, xyz, frag, env, ions):
    m0, m1 = make_mol(sym, xyz, 0), make_mol(sym, xyz, 1)
    mf0 = scf_method(m0, "HF", env, ions)
    normals = [plane_normal(xyz[frag == k]) for k in (0, 1)]
    w = pi_weights(m0, mf0.mo_coeff, normals, np.array(frag))
    nocc = m0.nelectron // 2
    occ_pi = [i for i in range(nocc - 1, -1, -1) if w[i] > 0.5][:4]
    vir_pi = [i for i in range(nocc, len(w)) if w[i] > 0.5][:2]
    act = sorted(occ_pi + vir_pi)
    info = [(int(i), round(float(mf0.mo_energy[i]) * HA2EV, 2), round(float(w[i]), 2)) for i in act]

    # нейтраль CASSCF(8,6) — референс заселённостей
    mc0 = mcscf.CASSCF(mf0, 6, 8)
    mc0.kernel(mc0.sort_mo([i + 1 for i in act]))
    S = m0.intor("int1e_ovlp")
    n_neu = frag_pop(m0, mc0.make_rdm1(), S, frag)

    # катион: SA2-CASSCF(7,6), старт с орбиталей нейтрали
    mf1 = with_env(scf.ROHF(m1).density_fit(), env, ions)
    mf1.kernel()
    sa = mcscf.CASSCF(mf1, 6, (4, 3)).state_average_([0.5, 0.5])
    sa.fix_spin_(ss=0.75)
    sa.max_cycle_macro = 150
    sa.kernel(mc0.mo_coeff)
    assert sa.converged, "SA-CASSCF не сошёлся"

    ci = mcscf.CASCI(mf1, 6, (4, 3))
    ci.fcisolver.nroots = 2
    ci.fix_spin_(ss=0.75)
    ci.kernel(sa.mo_coeff)
    e_cas = list(map(float, ci.e_tot))
    e_pt2 = [e_cas[r] + float(mrpt.NEVPT(ci, root=r).kernel()) for r in (0, 1)]

    C = ci.mo_coeff[:, ci.ncore:ci.ncore + ci.ncas]
    n_state = [frag_pop(m1, ci.make_rdm1(ci=ci.ci[r]), S, frag) for r in (0, 1)]
    t_act = ci.fcisolver.trans_rdm1(ci.ci[0], ci.ci[1], 6, (4, 3))
    n_tr = frag_pop(m1, C @ t_act @ C.T, S, frag)

    holes = [(n_neu - n).round(4).tolist() for n in n_state]
    # FCD: Δn_ab = n_G1(ab) − n_G2(ab) в базисе двух адиабатических состояний
    d11, d22 = n_state[0][0] - n_state[0][1], n_state[1][0] - n_state[1][1]
    d12 = n_tr[0] - n_tr[1]
    den = np.sqrt((d11 - d22) ** 2 + 4 * d12 ** 2)
    out = {"active": info, "E_CASSCF": e_cas, "E_NEVPT2": e_pt2, "hole_states_CASSCF": holes}
    for tag, e in (("CASSCF", e_cas), ("NEVPT2", e_pt2)):
        gap = (e[1] - e[0]) * HA2EV
        out[f"gap_{tag}_eV"] = round(gap, 4)
        out[f"V_{tag}_meV"] = round(1000 * gap * abs(d12) / den, 1)
        out[f"Delta_{tag}_meV"] = round(1000 * gap * abs(d11 - d22) / den, 1)
    return out


def main():
    global residues_pdb
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="data/1KF1.pdb")
    ap.add_argument("--res", type=int, nargs=2, default=[2, 3], help="5′ и 3′ остатки")
    ap.add_argument("--k-cutoff", type=float, default=8.0)
    args = ap.parse_args()

    ref = json.load(open("results/g_cation.json"))
    from ref_g_cation import rdkit_start
    tsym, _ = rdkit_start()
    residues_pdb, ions_all = read_pdb(args.pdb)
    sym, xyz, frag, rmsd = build_gg(args.res, tsym, ref["geom_neutral"])
    c = xyz.mean(0)
    ions = np.array([k for k in ions_all if np.linalg.norm(k - c) < args.k_cutoff])
    rise = abs(np.dot(xyz[frag == 1].mean(0) - xyz[frag == 0].mean(0), plane_normal(xyz[frag == 0])))
    print(f"GG из {args.pdb}: остатки {args.res} (5′, 3′), {len(sym)} атомов, "
          f"RMSD наложения {np.round(rmsd, 3).tolist()} Å, межплоскостное ≈ {rise:.2f} Å, "
          f"K⁺ в {args.k_cutoff} Å: {len(ions)}", flush=True)
    with open("results/gg_1KF1.xyz", "w") as f:
        f.write(f"{len(sym) + len(ions)}\nGG from 1KF1 res {args.res} + K+\n")
        f.writelines(f"{s} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n" for s, p in zip(sym, xyz))
        f.writelines(f"K {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n" for p in ions)

    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    key0 = f"res{args.res[0]}-{args.res[1]}"
    R.setdefault(key0, {})

    def step(key, fn):
        if key not in R[key0]:
            t0 = time.time()
            R[key0][key] = fn()
            R[key0][key]["t"] = round(time.time() - t0)
            json.dump(R, open(OUT, "w"), indent=1, ensure_ascii=False)
        print(f"  {key:22s} {R[key0][key]}", flush=True)

    for env in ("gas", "K", "PCM"):
        for meth in ("HF", "B3LYP", "CAM-B3LYP"):
            step(f"{meth}/{env}", lambda: hole_scf(sym, xyz, frag, meth, env, ions))
    for env in ("gas", "K"):
        step(f"SA-CASSCF/{env}", lambda: hole_cas(sym, xyz, frag, env, ions))

    print("\nДырка на (5′-G, 3′-G); основное состояние GG⁺:")
    for k, v in R[key0].items():
        h = v.get("hole") or v["hole_states_CASSCF"][0]
        extra = ""
        if "V_NEVPT2_meV" in v:
            extra = (f"   NEVPT2: V = {v['V_NEVPT2_meV']} мэВ, Δ = {v['Delta_NEVPT2_meV']} мэВ, "
                     f"щель = {v['gap_NEVPT2_eV']} эВ")
        print(f"  {k:22s} {h[0]:6.3f} {h[1]:6.3f}{extra}")


if __name__ == "__main__":
    main()
