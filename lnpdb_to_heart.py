#!/usr/bin/env python3
"""LNPDB ionizable-lipid SMILES -> audited HEART 10-slot LMDB dataset.

Independent adapter, without changing HEART source. The actual geometry generator
is imported from the user-provided HEART repository's
model/pretrain/lipid_library/conformers.py (same as Ugi Stage 04).

Commands:
    prepare   deduplicate IL_SMILES, optional vocabulary screen, fixed identity split
    probe     real RDKit geometry smoke test; no LMDB dependency
    build     create 10-slot integer-keyed LMDBs for one parallel shard
    finalize  verify all shard receipts and create HEART-compatible splits.json
    verify    independently inspect created LMDB and training-time 2D alignment

Requires RDKit, NumPy; build/finalize/verify require python-lmdb (HPC heart_gpu).
No performance readouts, formulation covariates or cargo labels are used.
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for buf in iter(lambda: f.read(1 << 20), b''):
            h.update(buf)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)


def add_repo(repo):
    pretrain = Path(repo).resolve() / 'model' / 'pretrain'
    conf = pretrain / 'lipid_library' / 'conformers.py'
    if not conf.exists():
        raise FileNotFoundError('Cannot find HEART conformers.py at ' + str(conf))
    if str(pretrain) not in sys.path:
        sys.path.insert(0, str(pretrain))
    from lipid_library.conformers import smi2coords_pretrain, smi2_2Dcoords, CONFORMER_POLICY_VERSION
    return smi2coords_pretrain, smi2_2Dcoords, CONFORMER_POLICY_VERSION


def dictionary_symbols(path):
    if not path:
        return None
    with open(path, encoding='utf-8') as f:
        return {line.split()[0] for line in f if line.split()}


def stable_id(smi):
    return hashlib.sha256(smi.encode('utf-8')).hexdigest()[:20]


def group_tag(mol):
    """Descriptive chemistry tags, NOT a validated scaffold/route classification."""
    from rdkit.Chem import Descriptors, rdMolDescriptors
    # Helps future stratification diagnostics; no heuristic 'true Ugi' label.
    return {
        'rings': rdMolDescriptors.CalcNumRings(mol),
        'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'logp': round(Descriptors.MolLogP(mol), 2),
    }


def prepare(args):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / 'lnpdb_unique_prepared.csv'
    rejected = out / 'lnpdb_rejected.csv'
    report_path = out / 'preparation_report.json'
    if (dest.exists() or rejected.exists() or report_path.exists()) and not args.overwrite:
        raise FileExistsError('Prepared files already exist; use a new --out or --overwrite intentionally')
    allowed = dictionary_symbols(args.dict_path)
    rows = {}
    rejected_rows = []
    reasons = collections.Counter()
    raw_rows = 0
    missing = 0
    for r in csv.DictReader(open(args.csv, newline='', encoding='utf-8-sig')):
        raw_rows += 1
        smi = (r.get('IL_SMILES') or '').strip()
        if not smi or smi.upper() in ('NA', 'N/A', 'NAN', 'NONE'):
            missing += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            reasons['rdkit_parse_or_sanitize_failed'] += 1
            rejected_rows.append((raw_rows, smi, 'rdkit_parse_or_sanitize_failed'))
            continue
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        hmol = Chem.AddHs(mol)
        atoms = [a.GetSymbol() for a in hmol.GetAtoms()]
        heavy = mol.GetNumHeavyAtoms()
        bad = sorted(set(atoms) - allowed) if allowed is not None else []
        if heavy > args.max_atoms:
            reasons['too_many_heavy_atoms'] += 1
            rejected_rows.append((raw_rows, smi, 'too_many_heavy_atoms'))
            continue
        if bad:
            reasons['outside_model_vocabulary'] += 1
            rejected_rows.append((raw_rows, smi, 'outside_model_vocabulary:' + ','.join(bad)))
            continue
        if canon not in rows:
            descr = group_tag(mol)
            rows[canon] = {'smiles': canon, 'molecule_id': stable_id(canon),
                           'n_source_rows': 0, 'n_heavy_atoms': heavy,
                           'n_atoms_with_h': len(atoms),
                           'rings': descr['rings'], 'rotatable_bonds': descr['rotatable_bonds'],
                           'logp': descr['logp']}
        rows[canon]['n_source_rows'] += 1
    if not rows:
        raise ValueError('No valid unique ionizable lipid molecules after screening')
    # Hash split is fixed by molecular identity; this is NOT structural-family holdout.
    prepared = list(rows.values())
    for row in prepared:
        rank = int(hashlib.sha256(('HEART-LNPDB-split-v1:' + row['smiles']).encode()).hexdigest()[:16], 16) / 16**16
        row['split'] = 'valid' if rank < args.valid_fraction else 'train'
    prepared.sort(key=lambda r: r['molecule_id'])
    if not any(r['split'] == 'valid' for r in prepared) or not any(r['split'] == 'train' for r in prepared):
        raise RuntimeError('Cannot produce nonempty train and valid sets; adjust --valid-fraction')
    fields = list(prepared[0])
    with open(dest, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(prepared)
    with open(rejected, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f); wr.writerow(['input_row','input_smiles','reason']); wr.writerows(rejected_rows)
    report = {
        'input_path': str(Path(args.csv).resolve()), 'input_sha256': sha256_file(args.csv),
        'prepared_sha256': sha256_file(dest), 'vocabulary_sha256': sha256_file(args.dict_path) if args.dict_path else None,
        'max_heavy_atoms': args.max_atoms, 'n_raw_formulation_rows': raw_rows, 'n_missing_smiles_rows': missing,
        'n_unique_eligible_molecules': len(prepared), 'n_train': sum(r['split']=='train' for r in prepared),
        'n_valid': sum(r['split']=='valid' for r in prepared),
        'n_duplicate_formulation_rows_collapsed': sum(r['n_source_rows']-1 for r in prepared),
        'n_rejected_rows': len(rejected_rows), 'rejected_reasons': dict(reasons),
        'stereochemistry': 'preserve as in IL_SMILES; canonical isomeric SMILES',
        'protonation': 'preserve IL_SMILES (do not switch to IL_protonated_SMILES)',
        'validation_split': 'stable molecular identity hash; not a structural-family holdout',
        'uses_experimental_targets': False,
        'dictionary_checked': bool(args.dict_path),
    }
    save_json(report_path, report)
    print(json.dumps(report, indent=2))
    print('Prepared:', dest)
    print('Report:', report_path)
    return report


def read_prepared(path):
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        need = {'smiles','molecule_id','split'}
        if not need.issubset(r.fieldnames or []):
            raise ValueError('Prepared input is missing columns ' + repr(need))
        for row in r:
            if row['split'] not in ('train', 'valid'):
                raise ValueError('Unexpected split: ' + row['split'])
            yield row


def generate_record(row, opts, g):
    smi2coords_pretrain, smi2_2d, _ = g
    record, stats = smi2coords_pretrain(row['smiles'], n_3d=10, store_2d=False,
                                        max_try=opts.max_try, max_iteration=4000,
                                        max_atoms_for_3d=400, seed_start=0)
    if stats['n_3d_success'] < opts.min_3d or stats['n_distinct_3d'] < opts.min_distinct_3d:
        raise ValueError('insufficient_real_3d: n_success={} n_distinct={}'.format(
            stats['n_3d_success'], stats['n_distinct_3d']))
    if set(record) != {'smi','atoms','coordinates'}:
        raise AssertionError('HEART record unexpected fields')
    if record['smi'] != row['smiles'] or len(record['coordinates']) != 10:
        raise AssertionError('HEART 10-coordinate-slot contract not met')
    if any(np.asarray(c).shape != (len(record['atoms']), 3) for c in record['coordinates']):
        raise AssertionError('HEART coordinate/atom mismatch')
    twod = smi2_2d(record['smi'])
    if twod.shape != (len(record['atoms']), 3):
        raise AssertionError('Training-time 2D representation would have wrong atom count')
    from rdkit import Chem
    mol_h = Chem.AddHs(Chem.MolFromSmiles(record['smi']))
    if [a.GetSymbol() for a in mol_h.GetAtoms()] != record['atoms']:
        raise AssertionError('Training-time 2D atom order does not match')
    return record, stats


def probe(args):
    # Probes actual production 10-conformer generator, but deliberately no LMDB writes.
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    g = add_repo(args.repo)
    rows = list(read_prepared(args.prepared))
    # One non-cherry-picked deterministic spread across molecular size range.
    rows = [r for r in rows if args.min_probe_heavy <= int(r['n_heavy_atoms']) <= args.max_probe_heavy]
    if not rows: raise ValueError('No molecules at or below --max-probe-heavy')
    rows.sort(key=lambda r: (int(r['n_heavy_atoms']), r['molecule_id']))
    n = min(args.count, len(rows))
    chosen = [rows[int(round(i * (len(rows)-1) / max(1,n-1)))] for i in range(n)]
    results = []
    for ix, row in enumerate(chosen):
        t0 = time.monotonic()
        try:
            record, s = generate_record(row, args, g)
            entry = {'molecule_id': row['molecule_id'], 'n_heavy_atoms': int(row['n_heavy_atoms']),
                     'status': 'PASS', 'n_slots': len(record['coordinates']),
                     'n_3d_success': s['n_3d_success'], 'n_2d_filled': s['n_2d_filled'],
                     'n_distinct_3d': s['n_distinct_3d']}
        except Exception as exc:
            entry = {'molecule_id': row['molecule_id'], 'n_heavy_atoms': int(row['n_heavy_atoms']),
                     'status': 'FAIL', 'reason': f'{type(exc).__name__}: {exc}'}
        entry['seconds'] = round(time.monotonic()-t0, 3)
        results.append(entry)
        print('PROBE {}/{}: {}'.format(ix+1, n, entry), flush=True)
    report = {'tested': len(results), 'passed': sum(r['status']=='PASS' for r in results),
              'failed': sum(r['status']=='FAIL' for r in results), 'results': results,
              'production_slots': 10, '2d_added_by_loader': True,
              'conformer_policy': g[2]}
    if args.json_out:
        save_json(args.json_out, report)
    return 0 if report['passed'] == len(results) else 2


def build(args):
    try:
        import lmdb
    except ModuleNotFoundError as exc:
        raise RuntimeError('python-lmdb is required for --build. Use the HPC heart_gpu environment.') from exc
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError('Require 0 <= --shard-id < --num-shards')
    if not args.dict_path:
        raise ValueError('--dict must point to the real General model dict.txt')
    vocab = dictionary_symbols(args.dict_path)
    g = add_repo(args.repo)
    all_rows = list(read_prepared(args.prepared))
    expected = [r for r in all_rows if int(r['molecule_id'], 16) % args.num_shards == args.shard_id]
    if args.limit is not None:
        # Smoke test includes N from EACH split so that both datasets can be verified.
        expected = ([r for r in expected if r['split'] == 'train'][:args.limit] +
                    [r for r in expected if r['split'] == 'valid'][:args.limit])
    destdir = Path(args.out) / ('shard-{:05d}'.format(args.shard_id))
    destdir.mkdir(parents=True, exist_ok=True)
    rp = destdir / 'receipt.json'
    if rp.exists():
        raise FileExistsError('Refusing to overwrite existing receipt: '+str(rp))
    pending = [destdir / ('{}.lmdb'.format(split)) for split in ('train','valid')]
    if any(p.exists() or (destdir / (p.name+'.partial')).exists() for p in pending):
        raise FileExistsError('Output LMDB exists. Use a new output directory; no in-place overwrite.')
    counts = collections.Counter(); bad = collections.Counter(); geometry = collections.Counter()
    lmdbs = []
    for split in ('train','valid'):
        rows = [r for r in expected if r['split'] == split]
        if not rows:
            continue
        path = destdir / (split + '.lmdb')
        partial = destdir / (split + '.lmdb.partial')
        env = lmdb.open(str(partial), subdir=False, readonly=False, lock=True, readahead=False,
                        map_size=int(args.map_size_gb * (1<<30)), max_readers=4)
        csvpath = destdir / (split + '_records.csv')
        try:
            with open(csvpath, 'w', newline='', encoding='utf-8') as rf:
                wr = csv.writer(rf)
                wr.writerow(['key','molecule_id','smiles','n_3d_success','n_distinct_3d','n_2d_filled'])
                txn = env.begin(write=True)
                key = 0
                for r in rows:
                    try:
                        mol = Chem.MolFromSmiles(r['smiles'])
                        if mol is None: raise ValueError('unparsable')
                        atoms = [a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()]
                        if mol.GetNumHeavyAtoms() > args.max_atoms:
                            raise ValueError('too_many_heavy_atoms')
                        if set(atoms) - vocab:
                            raise ValueError('outside_model_vocabulary')
                        rec, stats = generate_record(r, args, g)
                    except Exception as exc:
                        bad[type(exc).__name__ + ':' + str(exc)[:120]] += 1
                        continue
                    txn.put(str(key).encode('ascii'), pickle.dumps(rec, protocol=4), overwrite=False)
                    wr.writerow([key, r['molecule_id'], r['smiles'], stats['n_3d_success'],
                                 stats['n_distinct_3d'], stats['n_2d_filled']])
                    geometry['n_real_3d_slots'] += stats['n_3d_success']
                    geometry['n_2d_fallback_slots'] += stats['n_2d_filled']
                    geometry['n_distinct_3d_total'] += stats['n_distinct_3d']
                    counts[split] += 1
                    key += 1
                    if key % 100 == 0:
                        txn.commit();txn = env.begin(write=True)
                txn.commit()
            # Compact copy avoids sparse 4-GB output files and mirrors Ugi writer.
            compact = destdir / (split + '.lmdb.compact')
            env.copy(str(compact), compact=True)
        finally:
            env.close()
        try:
            os.remove(partial)
            lockfile = str(partial)+'-lock'
            if os.path.exists(lockfile): os.remove(lockfile)
            os.replace(compact, path)
        except Exception:
            raise
        lmdbs.append({'split': split,'path':str(path.relative_to(args.out)),
                      'n_records':counts[split], 'sha256':sha256_file(path)})
    receipt = {'shard_id':args.shard_id, 'num_shards':args.num_shards, 'source_prepared_sha256':sha256_file(args.prepared),
               'real_dict_sha256':sha256_file(args.dict_path), 'n_input_rows':len(expected),
               'n_written':sum(counts.values()), 'n_skipped':len(expected)-sum(counts.values()),
               'counts_by_split':dict(counts), 'skip_reasons':dict(bad),
               'geometry':dict(geometry), 'lmdbs':lmdbs,
               'geometry_policy':g[2], 'n_3d':10, 'store_2d':False,
               'min_3d':args.min_3d,'min_distinct_3d':args.min_distinct_3d,
               'max_try':args.max_try, 'max_heavy_atoms':args.max_atoms,
               'smoke':bool(args.limit is not None)}
    save_json(rp, receipt)
    print(json.dumps(receipt,indent=2))
    return receipt


def finalize(args):
    root = Path(args.out)
    all_rows = list(read_prepared(args.prepared))
    if args.num_shards < 1:
        raise ValueError('num shards must be positive')
    train, valid, receipts = [], [], []
    seen = set()
    for i in range(args.num_shards):
        rp = root / ('shard-{:05d}'.format(i)) / 'receipt.json'
        if not rp.exists(): raise FileNotFoundError('Missing shard receipt: '+str(rp))
        with open(rp, encoding='utf-8') as f: r=json.load(f)
        if r['shard_id'] != i or r['num_shards'] != args.num_shards:
            raise ValueError('Shard geometry mismatch at shard '+str(i))
        if r['source_prepared_sha256'] != sha256_file(args.prepared):
            raise ValueError('Prepared input fingerprint mismatch at shard '+str(i))
        expected_n = sum(int(row['molecule_id'], 16) % args.num_shards == i for row in all_rows)
        if r['n_input_rows'] != expected_n and not (args.allow_smoke and r.get('smoke') and r['n_input_rows'] <= expected_n):
            raise ValueError('Shard input cardinality mismatch at shard {}'.format(i))
        if r['n_written'] + r['n_skipped'] != r['n_input_rows']:
            raise ValueError('Shard count reconciliation failed at shard {}'.format(i))
        for x in r['lmdbs']:
            p = root / x['path']
            if not p.exists() or sha256_file(p) != x['sha256']:
                raise ValueError('LMDB missing/different: '+str(p))
            (train if x['split']=='train' else valid).append(x['path'])
        if i in seen: raise ValueError('Duplicate shard ID')
        seen.add(i)
        receipts.append(r)
    totals = {'input':sum(r['n_input_rows'] for r in receipts),
              'records':sum(r['n_written'] for r in receipts),
              'skipped':sum(r['n_skipped'] for r in receipts)}
    if not all(train) or not all(valid): raise ValueError('Empty train or valid LMDB paths')
    if totals['input'] != len(all_rows):
        if not args.allow_smoke:
            raise ValueError('Not all input rows covered: {} != {}. Smoke needs --allow-smoke'.format(totals['input'],len(all_rows)))
    if not train or not valid: raise ValueError('Must have nonempty train and valid LMDBs')
    smoke = any(r['smoke'] for r in receipts) or totals['input'] != len(all_rows)
    if smoke and not args.allow_smoke: raise ValueError('Smoke data cannot be finalized without --allow-smoke')
    for r in receipts:
        if (r['geometry_policy'],r['real_dict_sha256'],r['max_heavy_atoms'],r['min_3d'],r['min_distinct_3d']) != (
            receipts[0]['geometry_policy'],receipts[0]['real_dict_sha256'],receipts[0]['max_heavy_atoms'],receipts[0]['min_3d'],receipts[0]['min_distinct_3d']):
            raise ValueError('Shard policies/dictionary differ')
    meta = {'complete':True, 'smoke':smoke, 'source':'LNPDB IL_SMILES structural-only',
            'train':sorted(train), 'valid':sorted(valid),
            'coordinate_slots_per_record':[10], 'stored_2d_conformer':False,
            'n_records':totals['records'], 'n_input_rows':totals['input'],
            'n_skipped':totals['skipped'], 'n_shards':args.num_shards,
            'is_structural_family_holdout':False,
            'note':'Stand-alone LNPDB dataset. Current HEART continual task does not yet support separate Ugi and LNPDB sampling ratios; do not use --lipid-lmdb for both train and valid.'}
    save_json(root/'splits.json', meta)
    print(json.dumps(meta,indent=2))
    return meta


def verify(args):
    # The original HEART validation function is reused for a real LMDB read.
    _ = add_repo(args.repo)
    # import after repository is on sys.path
    from lipid_library.validate_lmdb import check_lmdb
    vocab = dictionary_symbols(args.dict_path)
    root = Path(args.out)
    with open(root/'splits.json', encoding='utf-8') as f: splits=json.load(f)
    bad = []; stats=[]
    if not splits.get('complete'): raise ValueError('Not a finalized dataset')
    all_ids = {'train':set(), 'valid':set()}
    for split in ('train','valid'):
        for p in splits[split]:
            lp = root/p
            qc=check_lmdb(str(lp), dict_symbols=vocab, sample=args.sample,
                          check_all=args.check_all, max_atoms=args.max_atoms,
                          expect_slots=10, only_polar=0)
            stats.append({'split':split,'path':p,'n_records':qc['n_records'],'n_checked':qc['n_checked'],
                          'n_problems':len(qc['problems'])})
            if qc['problems']: bad.extend([p+': '+v for v in qc['problems'][:10]])
            # cross-check the saved identity index for train/valid leakage.
            cp=lp.parent/(split+'_records.csv')
            if not cp.is_file(): bad.append('Missing identity index '+str(cp));continue
            with open(cp,newline='',encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row['molecule_id'] in all_ids[split]: bad.append('duplicate identity within '+split+': '+row['molecule_id'])
                    all_ids[split].add(row['molecule_id'])
    overlap=all_ids['train'] & all_ids['valid']
    if overlap: bad.append('{} molecular identities overlap train/valid'.format(len(overlap)))
    report={'status':'PASS' if not bad else 'FAIL', 'n_train_unique':len(all_ids['train']),
            'n_valid_unique':len(all_ids['valid']), 'checked':stats,'problems':bad[:100]}
    save_json(root/'verification.json',report)
    print(json.dumps(report,indent=2))
    return 0 if not bad else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Canonicalize and deduplicate LNPDB IL_SMILES')
    p.add_argument('--csv',required=True);p.add_argument('--out',required=True)
    p.add_argument('--dict',dest='dict_path');p.add_argument('--max-atoms',type=int,default=256)
    p.add_argument('--valid-fraction',type=float,default=0.05)
    p.add_argument('--overwrite',action='store_true')
    q=sub.add_parser('probe',help='Test production 10-conformer generator without LMDB')
    q.add_argument('--repo',required=True);q.add_argument('--prepared',required=True)
    q.add_argument('--count',type=int,default=5);q.add_argument('--json-out')
    q.add_argument('--min-probe-heavy',type=int,default=0)
    q.add_argument('--max-probe-heavy',type=int,default=75,help='avoid expensive huge molecules during initial probe')
    q.add_argument('--max-try',type=int,default=15);q.add_argument('--min-3d',type=int,default=1)
    q.add_argument('--min-distinct-3d',type=int,default=1)
    b=sub.add_parser('build',help='Generate LMDB for one parallel shard (python-lmdb required)')
    b.add_argument('--repo',required=True);b.add_argument('--prepared',required=True)
    b.add_argument('--dict',dest='dict_path',required=True);b.add_argument('--out',required=True)
    b.add_argument('--shard-id',type=int,required=True);b.add_argument('--num-shards',type=int,required=True)
    b.add_argument('--limit',type=int);b.add_argument('--map-size-gb',type=float,default=4)
    b.add_argument('--max-atoms',type=int,default=256);b.add_argument('--max-try',type=int,default=15)
    b.add_argument('--min-3d',type=int,default=1);b.add_argument('--min-distinct-3d',type=int,default=1)
    f=sub.add_parser('finalize',help='Check all shards, create HEART splits.json')
    f.add_argument('--out',required=True);f.add_argument('--prepared',required=True)
    f.add_argument('--num-shards',type=int,required=True);f.add_argument('--allow-smoke',action='store_true')
    v=sub.add_parser('verify',help='Read LMDB and check record schema, atom order, split overlap')
    v.add_argument('--repo',required=True);v.add_argument('--dict',dest='dict_path',required=True)
    v.add_argument('--out',required=True);v.add_argument('--sample',type=int,default=100)
    v.add_argument('--check-all',action='store_true');v.add_argument('--max-atoms',type=int,default=256)
    args=ap.parse_args()
    if args.command=='prepare':prepare(args);return 0
    if args.command=='probe':return probe(args)
    if args.command=='build':build(args);return 0
    if args.command=='finalize':finalize(args);return 0
    if args.command=='verify':return verify(args)
    return 1

if __name__=='__main__':
    sys.exit(main())
