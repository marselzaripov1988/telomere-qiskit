"""
Классический эталон: локализация дырки в GG⁺ теломерного G-квадруплекса.

Геометрия: остатки G2 (5′) и G3 (3′) тракта GGG из кристаллической структуры
параллельного теломерного G-квадруплекса человека с K⁺ (PDB 1KF1, Parkinson et al.
2002). Тяжёлые атомы оснований -> на них накладывается (Kabsch) гуанин с H,
оптимизированный B3LYP/cc-pVDZ (results/g_cation.json); N9 закрыт атомом H.

Окружения (точечные заряды через qmmm.mm_charge):
  gas | K   — ионы K⁺ кристалла (+1) в 8 Å
      | KP  — все K⁺ (+1) и все 21 фосфат цепи (−0.5 на OP1 и OP2), заряд среды −18
      | KPn — то же, заряды фосфатов масштабированы до электронейтральности
              (−3/21 на фосфат: неявная конденсация противоионов)
      | PCM — вода, ε=78.4

Методы (cc-pVDZ):
  UHF, UB3LYP, UCAM-B3LYP — дырка на каждом G по Малликену:
      h_f = n_f(GG) − n_f(GG⁺)   (сумма = 1)
  SA4-CASSCF(7,6) — π HOMO/HOMO-1 каждого G + 2 π*; 4 нижних дублета GG⁺.
  SA-DSRG-MRPT2 (pyscf-forge, s=0.5) с релаксацией референса: динамическая корреляция
      и перемешивание квазивырожденных состояний в CAS (аналог QD/XMS-NEVPT2).
  FCD-модель (Voityuk) для пары нижних состояний с дыркой на разных G:
      V, Δ из энергий и (переходных) плотностей — CASSCF и DSRG-MRPT2 (релакс. CI).
"""
import argparse
import json
import os
import time

import numpy as np
from pyscf import gto, scf, dft, mcscf, mrpt, qmmm, solvent, fci
from pyscf.dsrg_mrpt2 import DSRG_MRPT2

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
    """ions: dict env -> (координаты Å, заряды) для окружений из точечных зарядов."""
    if env in ions:
        xyz, q = ions[env]
        return qmmm.mm_charge(mf, xyz, q)
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


def cas_dm_ao(mc, casdm1):
    c_core = mc.mo_coeff[:, :mc.ncore]
    c_act = mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
    return 2 * c_core @ c_core.T + c_act @ casdm1 @ c_act.T


def fcd(e, dq):
    """FCD (Voityuk) для двух адиабатических состояний.
    e — энергии (Ha); dq = (Δq11, Δq22, Δq12), Δq = n(5′-G) − n(3′-G)."""
    d11, d22, d12 = dq
    den = np.sqrt((d11 - d22) ** 2 + 4 * d12 ** 2)
    gap = (e[1] - e[0]) * HA2EV
    return {"gap_eV": round(gap, 4), "V_meV": round(1000 * abs(gap * d12 / den), 1),
            "Delta_meV": round(1000 * abs(gap * (d11 - d22) / den), 1)}


def analyze_states(mc, vecs, energies, n_neu, S, m1, frag, nelec):
    """Дырки всех состояний, выбор CT-пары и FCD."""
    def pop(dm_act):
        return frag_pop(m1, cas_dm_ao(mc, dm_act), S, frag)
    nr = len(vecs)
    n_all = [pop(fci.direct_spin1.make_rdm1(v, mc.ncas, nelec)) for v in vecs]
    holes = [(n_neu - n).round(4).tolist() for n in n_all]
    on = [[r for r in range(nr) if holes[r][k] > 0.5] for k in (0, 1)]
    out = {"E": [float(x) for x in energies], "holes": holes}
    if not (on[0] and on[1]):
        out["note"] = "нет пары состояний с дыркой на разных G"
        return out
    p = sorted((on[0][0], on[1][0]))
    c_act = mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
    t = fci.direct_spin1.trans_rdm1(vecs[p[0]], vecs[p[1]], mc.ncas, nelec)
    n_tr = frag_pop(m1, c_act @ t @ c_act.T, S, frag)
    dq = (n_all[p[0]][0] - n_all[p[0]][1], n_all[p[1]][0] - n_all[p[1]][1], n_tr[0] - n_tr[1])
    out.update({"pair": p, **fcd([energies[r] for r in p], dq)})
    return out


def hole_cas(sym, xyz, frag, env, ions, tag):
    m0, m1 = make_mol(sym, xyz, 0), make_mol(sym, xyz, 1)
    mf0 = scf_method(m0, "HF", env, ions)
    normals = [plane_normal(xyz[frag == k]) for k in (0, 1)]
    w = pi_weights(m0, mf0.mo_coeff, normals, np.array(frag))
    nocc = m0.nelectron // 2
    occ_pi = [i for i in range(nocc - 1, -1, -1) if w[i] > 0.5][:4]
    vir_pi = [i for i in range(nocc, len(w)) if w[i] > 0.5][:2]
    act = sorted(occ_pi + vir_pi)
    info = [(int(i), round(float(mf0.mo_energy[i]) * HA2EV, 2), round(float(w[i]), 2)) for i in act]
    chk = f"chk/gg_{tag}_{env}.npz"
    saved = np.load(chk) if os.path.exists(chk) else None

    # нейтраль: CASSCF(8,6) — референс заселённостей; DSRG-MRPT2 — для IP
    mc0 = mcscf.CASSCF(mf0, 6, 8)
    mc0.max_cycle_macro = 150
    mc0.kernel(saved["mo0"] if saved is not None else mc0.sort_mo([i + 1 for i in act]))
    assert mc0.converged, "CASSCF нейтрали не сошёлся"
    e_neu_dsrg = float(DSRG_MRPT2(mc0, s=0.5).kernel())
    S = m0.intor("int1e_ovlp")
    n_neu = frag_pop(m0, mc0.make_rdm1(), S, frag)

    # катион: SA4-CASSCF(7,6), равные веса, старт с орбиталей нейтрали (или из chk)
    # (SA2 смещён: оба корня — дырка на одном G, CT-состояние выталкивается вверх)
    nr, nelec = 4, (4, 3)
    mf1 = with_env(scf.ROHF(m1).density_fit(), env, ions)
    mf1.kernel()
    sa = mcscf.CASSCF(mf1, 6, nelec).state_average_([1 / nr] * nr)
    sa.fix_spin_(ss=0.75)
    sa.max_cycle_macro = 150
    sa.kernel(saved["mo1"] if saved is not None else mc0.mo_coeff)
    assert sa.converged, "SA-CASSCF не сошёлся"
    os.makedirs("chk", exist_ok=True)
    np.savez(chk, mo0=mc0.mo_coeff, mo1=sa.mo_coeff)

    out = {"active": info, "nroots": nr, "E_neutral_CASSCF": float(mc0.e_tot),
           "E_neutral_DSRG": e_neu_dsrg}
    cas = analyze_states(sa, sa.ci, sa.e_states, n_neu, S, m1, frag, nelec)
    out["CASSCF"] = cas

    # SA-DSRG-MRPT2 с однократной релаксацией референса: эффективный гамильтониан,
    # диагонализованный в CAS, — состояния заново перемешиваются (квазивырожденная PT2)
    d = DSRG_MRPT2(sa, s=0.5, relax="once")
    d.kernel()
    e_d = list(map(float, d.e_relax_eigval_shifted))
    out["DSRG"] = analyze_states(sa, d.ci_vecs, e_d, n_neu, S, m1, frag, nelec)
    out["IP_DSRG_eV"] = round((min(e_d) - e_neu_dsrg) * HA2EV, 4)
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
    phos = np.array([r[o] for r in residues_pdb.values() for o in ("OP1", "OP2") if o in r])
    sym, xyz, frag, rmsd = build_gg(args.res, tsym, ref["geom_neutral"])
    c = xyz.mean(0)
    ions = np.array([k for k in ions_all if np.linalg.norm(k - c) < args.k_cutoff])
    nk, nph = len(ions_all), len(phos) // 2
    envs = {
        "K": (ions, np.ones(len(ions))),
        "KP": (np.vstack([ions_all, phos]), np.r_[np.ones(nk), -0.5 * np.ones(2 * nph)]),
        "KPn": (np.vstack([ions_all, phos]), np.r_[np.ones(nk), -0.5 * nk / nph * np.ones(2 * nph)]),
    }
    dmin = min(np.linalg.norm(xyz[:, None] - phos[None], axis=2).min(axis=1))
    print(f"Фосфатов: {nph}, K⁺: {nk}; суммарный заряд KP {envs['KP'][1].sum():+.2f}, "
          f"KPn {envs['KPn'][1].sum():+.2f}; мин. расстояние OP–атом QM {dmin:.2f} Å", flush=True)
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

    for env in ("gas", "K", "PCM", "KP", "KPn"):
        for meth in ("HF", "B3LYP", "CAM-B3LYP"):
            step(f"{meth}/{env}", lambda: hole_scf(sym, xyz, frag, meth, env, envs))
    for env in ("gas", "KPn", "KP"):
        step(f"SA4-DSRG/{env}", lambda: hole_cas(sym, xyz, frag, env, envs, key0))

    print("\nДырка на (5′-G, 3′-G) в основном состоянии GG⁺:")
    for k, v in R[key0].items():
        if "hole" in v:
            print(f"  {k:22s} {v['hole'][0]:6.3f} {v['hole'][1]:6.3f}")
        elif k.startswith("SA4-DSRG"):
            for lvl in ("CASSCF", "DSRG"):
                r = v[lvl]
                h = r["holes"][int(np.argmin(r["E"]))]
                ex = (f"   V = {r['V_meV']} мэВ, Δ = {r['Delta_meV']} мэВ, щель = {r['gap_eV']} эВ"
                      if "V_meV" in r else "   " + r.get("note", ""))
                print(f"  {k.replace('SA4-DSRG', lvl):22s} {h[0]:6.3f} {h[1]:6.3f}{ex}")


if __name__ == "__main__":
    main()
