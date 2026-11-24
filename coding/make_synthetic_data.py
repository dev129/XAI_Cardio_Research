"""Synthetic data shaped like the REAL files, for testing the pipeline only.
Creates:
  <root>/ptbxl/            ptbxl_database.csv, scp_statements.csv, records500/..., records100/... (WFDB, mV)
  <root>/chapman_figshare/ Diagnostics.xlsx, ECGData/<FileName>.csv (5000x12, microvolts)
  <root>/chapman_wfdb/     WFDBRecords/01/010/JSxxxxx.{hea,mat}, RECORDS, ConditionNames_SNOMED-CT.csv
Signals carry weak label-dependent structure so a model can learn something.
NUMBERS PRODUCED ON THIS DATA ARE MEANINGLESS FOR THE PAPER.
"""
import os, sys, json, numpy as np, pandas as pd
from pathlib import Path
import scipy.io as sio

LEADS_PTB = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEADS_CH = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def synth_ecg(rng, fs=500, dur=10, hr=70, af=False, wide_qrs=False, big=False, noise=0.02, drift=0.1):
    n = fs * dur
    t = np.arange(n) / fs
    rr = 60.0 / hr
    beats, x = [], 0.3
    while x < dur:
        beats.append(x)
        x += rr * (rng.uniform(0.6, 1.4) if af else rng.uniform(0.97, 1.03))
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    qw = 0.025 if wide_qrs else 0.010
    amp = 1.8 if big else 1.0
    for bt in beats:
        g = lambda mu, sd: np.exp(-0.5 * ((t - mu) / sd) ** 2)
        p = 0 if af else 0.12 * g(bt - 0.16, 0.02)
        qrs = amp * (g(bt, qw) - 0.2 * g(bt - 0.03, 0.008) - 0.25 * g(bt + 0.03, 0.01))
        tw = 0.3 * g(bt + 0.28, 0.05)
        a += p + qrs + tw
        b += 0.6 * p + 0.5 * qrs + 0.25 * tw
        c += -0.3 * qrs + (0.4 * qrs if wide_qrs else 0) * g(bt + 0.02, 0.03) + 0.2 * tw
    if af:
        a += 0.05 * np.sin(2 * np.pi * 6.3 * t + rng.uniform(0, 6))
    for v in (a, b, c):  # noise/drift on independent electrodes, so Einthoven still holds for derived leads
        v += noise * rng.standard_normal(n) + drift * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 6))
    I, III = a, b
    II = I + III
    aVR, aVL, aVF = -(I + II) / 2, (I - III) / 2, (II + III) / 2
    V = [c * w1 + a * w2 for w1, w2 in [(1.2, 0.1), (1.0, 0.4), (0.5, 0.8), (0.0, 1.2), (-0.3, 1.1), (-0.4, 0.9)]]
    sig = np.stack([I, II, III, aVR, aVL, aVF] + V, axis=1)
    return sig.astype(np.float64)


def make_ptbxl(root, n=400, seed=0):
    import wfdb
    rng = np.random.default_rng(seed)
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    rows = []
    n_pat = int(n * 0.87)
    pats = rng.integers(1, n_pat + 1, size=n)
    pat_fold = {p: int(rng.integers(1, 11)) for p in np.unique(pats)}
    for i in range(1, n + 1):
        r = rng.random()
        rhythm = 'AFIB' if r < 0.12 else 'STACH' if r < 0.2 else 'SBRAD' if r < 0.28 else 'SARRH' if r < 0.33 else 'AFLT' if r < 0.35 else 'SR'
        codes = {rhythm: 0.0}
        rb = rng.random() < 0.12; lvh = rng.random() < 0.15
        if rb: codes['CRBBB'] = 100.0
        if lvh: codes['LVH'] = 100.0
        if rng.random() < 0.1: codes['1AVB'] = 100.0
        if rng.random() < 0.08: codes['PVC'] = 100.0
        if rng.random() < 0.1: codes['IRBBB'] = 100.0
        if rng.random() < 0.4 and rhythm == 'SR' and not (rb or lvh): codes['NORM'] = 100.0
        hr = 115 if rhythm == 'STACH' else 50 if rhythm == 'SBRAD' else 75
        sig = synth_ecg(rng, hr=hr, af=rhythm in ('AFIB', 'AFLT'), wide_qrs=rb, big=lvh, noise=0.02, drift=0.15)
        sub = f'{(i // 1000) * 1000:05d}'
        hr_name = f'records500/{sub}/{i:05d}_hr'
        lr_name = f'records100/{sub}/{i:05d}_lr'
        for name, fs, s in [(hr_name, 500, sig), (lr_name, 100, sig[::5])]:
            d = root / Path(name).parent; d.mkdir(parents=True, exist_ok=True)
            wfdb.wrsamp(Path(name).name, fs=fs, units=['mV'] * 12, sig_name=LEADS_PTB, p_signal=s,
                        fmt=['16'] * 12, adc_gain=[1000.0] * 12, baseline=[0] * 12, write_dir=str(d))
        p = int(pats[i - 1])
        rows.append(dict(ecg_id=i, patient_id=float(p), age=float(rng.integers(20, 90)), sex=int(rng.integers(0, 2)),
                         scp_codes=str(codes), heart_axis=rng.choice(['MID', 'LAD', 'ALAD', 'RAD', np.nan]),
                         strat_fold=pat_fold[p], validated_by_human=True, baseline_drift=np.nan, static_noise=np.nan,
                         burst_noise=np.nan, electrodes_problems=np.nan, filename_lr=lr_name, filename_hr=hr_name))
    pd.DataFrame(rows).to_csv(root / 'ptbxl_database.csv', index=False)
    pd.DataFrame({'Unnamed: 0': ['NORM', 'AFIB'], 'description': ['normal ECG', 'atrial fibrillation']}).to_csv(
        root / 'scp_statements.csv', index=False)


def _chapman_case(rng):
    r = rng.random()
    rhythm = 'AFIB' if r < 0.17 else 'SB' if r < 0.5 else 'ST' if r < 0.62 else 'AF' if r < 0.66 else 'SI' if r < 0.7 else 'SVT' if r < 0.74 else 'SR'
    beats = []
    if rng.random() < 0.10: beats.append('RBBB')
    if rng.random() < 0.10: beats.append('LVH')
    if rng.random() < 0.06: beats.append('1AVB')
    if rng.random() < 0.06: beats.append('VPB')
    if rng.random() < 0.05: beats.append('IRBBB')
    if rng.random() < 0.20: beats.append('TWC')
    if rng.random() < 0.03: beats.append('XYZQ')  # unknown acronym -> must be reported as unmapped
    hr = 115 if rhythm in ('ST', 'SVT') else 50 if rhythm == 'SB' else 75
    # domain shift: different noise, drift, slight gain
    sig = synth_ecg(rng, hr=hr, af=rhythm in ('AFIB', 'AF'), wide_qrs='RBBB' in beats, big='LVH' in beats,
                    noise=0.05, drift=0.4) * 1.15
    return rhythm, beats, sig


def make_chapman_figshare(root, n=300, seed=1):
    rng = np.random.default_rng(seed)
    root = Path(root); (root / 'ECGData').mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        rhythm, beats, sig = _chapman_case(rng)
        fn = f'MUSE_20180113_{i:06d}_{int(rng.integers(10000, 99999))}'
        df = pd.DataFrame(sig * 1000.0, columns=LEADS_CH)  # microvolts, as per Zheng et al. 2020
        if i == 5: df.iloc[:, 3] = 0.0  # a flat lead, as found in real data
        if i != 7:  # record 7: missing signal file (real-world glitch)
            df.to_csv(root / 'ECGData' / f'{fn}.csv', index=False, float_format='%.2f')
        rows.append(dict(FileName=fn, Rhythm=rhythm, Beat=' '.join(beats) if beats else 'NONE',
                         PatientAge=int(rng.integers(4, 98)), Gender=rng.choice(['MALE', 'FEMALE']),
                         VentricularRate=75, AtrialRate=75, QRSDuration=90, QTInterval=400, QTCorrected=420,
                         RAxis=30, TAxis=40, QRSCount=12, QOnset=200, QOffset=245, TOffset=400))
    pd.DataFrame(rows).to_excel(root / 'Diagnostics.xlsx', index=False)


SNOMED_CH = {  # ONLY for synthetic test files; real file ships with the dataset
    'SB': '426177001', 'SR': '426783006', 'AFIB': '164889003', 'ST': '427084000', 'AF': '164890007',
    'SI': '427393009', 'SVT': '426761007', 'RBBB': '59118001', 'LVH': '164873001', '1AVB': '270492004',
    'VPB': '17338001', 'IRBBB': '713426002', 'TWC': '164934002', 'XYZQ': '999999999'}


def make_chapman_wfdb(root, n=300, seed=2):
    rng = np.random.default_rng(seed)
    root = Path(root); rec_list = []
    for i in range(1, n + 1):
        rhythm, beats, sig = _chapman_case(rng)
        name = f'JS{i:05d}'
        d = root / 'WFDBRecords' / f'{(i - 1) // 1000 + 1:02d}' / f'{(i - 1) // 100 + 10:03d}'
        d.mkdir(parents=True, exist_ok=True)
        val = np.round(sig.T * 1000).astype(np.int16)  # 1000 adu/mV
        sio.savemat(d / f'{name}.mat', {'val': val}, format='4')
        dx = ','.join(SNOMED_CH[c] for c in [rhythm] + beats)
        lines = [f'{name} 12 500 5000 05-May-2020 14:50:55']
        for k, ld in enumerate(LEADS_CH):
            lines.append(f'{name}.mat 16+24 1000/mV 16 0 {int(val[k, 0])} 0 0 {ld}')
        lines += [f'#Age: {int(rng.integers(4, 98))}', f'#Sex: {rng.choice(["Male", "Female"])}', f'#Dx: {dx}',
                  '#Rx: Unknown', '#Hx: Unknown', '#Sx: Unknown']
        (d / f'{name}.hea').write_text('\n'.join(lines) + '\n')
        rec_list.append(str((d / name).relative_to(root)))
    (root / 'RECORDS').write_text('\n'.join(rec_list) + '\n')
    pd.DataFrame([{'Acronym Name': k, 'Full Name': k, 'Snomed_CT': v} for k, v in SNOMED_CH.items()]).to_csv(
        root / 'ConditionNames_SNOMED-CT.csv', index=False)


if __name__ == '__main__':
    out = Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/synth')
    n_ptb = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    n_ch = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    make_ptbxl(out / 'ptbxl', n_ptb)
    make_chapman_figshare(out / 'chapman_figshare', n_ch)
    make_chapman_wfdb(out / 'chapman_wfdb', n_ch)
    for d in ['ptbxl', 'chapman_figshare', 'chapman_wfdb']:
        (out / d / 'SYNTHETIC_DATA_MARKER').write_text('synthetic test data - numbers are meaningless')
    print('synthetic data written to', out)
