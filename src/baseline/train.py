"""Offline nested development validation. No inference/submission entry point."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from flag2027.common import ROOT, device_for, load_pack, save_json, seed_all, sha256
from flag2027.data import assert_training
from flag2027.metrics import verification_metrics
from flag2027.model import Association, joint_loss
from .model import BlockAssociation
from .losses import loss as block_loss, same_gender_rank


def development_pack(path, split_path):
    provenance = assert_training(path)
    if provenance['feature_set'] != 'official_plus_facenet_wavlm_age':
        raise ValueError('Expected the original frozen demographic training feature pack')
    p = load_pack(path)
    split = json.loads(Path(split_path).read_text())
    allowed = sorted(set(split['train']) | set(split['validation']))
    excluded = set(split['audit'])
    if set(allowed) & excluded:
        raise ValueError('Development and reserved audit identities overlap')
    selected = np.isin(p['labels'], allowed)
    p = {k: v[selected] for k, v in p.items()}
    assert set(np.unique(p['labels'])) == set(allowed)
    assert not set(np.unique(p['labels'])) & excluded
    return p, split


def balanced_trials(pack, identities, seed, anchors=128):
    rng = np.random.default_rng(seed)
    groups = {i: np.flatnonzero(pack['labels'] == i) for i in identities}
    sexes = {i: int(pack['genders'][ix[0]]) for i, ix in groups.items()}
    result = {}
    for track in ['no_gender', 'gender']:
        f, v, y = [], [], []
        for identity, ix in groups.items():
            others = [j for j in identities if j != identity and
                      (track == 'no_gender' or sexes[j] == sexes[identity])]
            if not others or len(ix) < 2:
                raise ValueError('Need >=2 examples/identity and >=2 identities/gender')
            for _ in range(anchors):
                vi = int(rng.choice(ix))
                pi = int(rng.choice(ix[ix != vi]))
                ni = int(rng.choice(groups[rng.choice(others)]))
                f.extend([pi,ni]); v.extend([vi,vi]); y.extend([1,0])
        result[track] = tuple(np.asarray(x) for x in [f,v,y])
    return result


@torch.inference_mode()
def evaluate(model, kind, pack, ids, device, cfg, seed):
    trials = balanced_trials(pack, ids, seed)
    ix = np.flatnonzero(np.isin(pack['labels'],ids))
    remap = np.full(len(pack['labels']),-1); remap[ix]=np.arange(len(ix))
    model.eval()
    outputs = None
    for start in range(0,len(ix),256):
        rows=ix[start:start+256]
        out=model(torch.as_tensor(pack['faces'][rows],device=device),torch.as_tensor(pack['voices'][rows],device=device))
        if outputs is None: outputs=[[] for _ in out]
        for dest,t in zip(outputs,out):dest.append(t.cpu())
    encoded=[torch.cat(x) for x in outputs]
    result={}
    for track,(fi,vi,labels) in trials.items():
        fi,vi=remap[fi],remap[vi]
        if kind in ['control_v1','v1_rank']:
            f,v,gf,gv=encoded
            score=(f[fi]*v[vi]).sum(-1)
        else:
            f,v,cf,cv,gf,gv=encoded
            gate=float((0.25*model.context_gate.sigmoid()).item())
            score=(f[fi]*v[vi]).sum(-1)+gate*(cf[fi]*cv[vi]).sum(-1)
        if track=='no_gender':
            score+=cfg['gender_score_weight']*(gf[fi].softmax(-1)*gv[vi].softmax(-1)).sum(-1)
        result[track]=verification_metrics(labels,score.numpy())
    result['mean_english_eer_percent']=float(np.mean([r['eer_percent'] for r in result.values()]))
    return result


def construct(kind, classes, cfg, device):
    if kind in ['control_v1','v1_rank']:
        model=Association(5376,1728,classes,embedding_dim=128,hidden_dim=256,dropout=0.6)
    else:
        model=BlockAssociation(classes,**{k:cfg[k] for k in ['embedding_dim','block_dim','context_dim','dropout','block_dropout']})
    return model.to(device)


def fit(kind, pack, fit_ids, cfg, out, device, seed, validation_ids=None, fixed_epochs=None):
    seed_all(seed)
    rng=np.random.default_rng(seed)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    classes=sorted(fit_ids)
    groups={i:np.flatnonzero(pack['labels']==i) for i in classes}
    by_gender={g:[i for i in classes if int(pack['genders'][groups[i][0]])==g] for g in [0,1]}
    if min(map(len,by_gender.values()))<2:raise ValueError('Need >=2 fitting identities of each gender')
    model=construct(kind,len(classes),cfg,device)
    control_cfg={'temperature':0.07,'proxy_weight':0.5,'gender_weight':0.15}
    lr=0.001 if kind in ['control_v1','v1_rank'] else cfg['learning_rate']
    wd=0.01 if kind in ['control_v1','v1_rank'] else cfg['weight_decay']
    optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    epochs=int(fixed_epochs or cfg['epochs'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=epochs)
    best,best_epoch,best_state=1e9,0,None
    history=[];started=time.time()
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for _ in range(cfg['steps_per_epoch']):
            identities=[]
            for g in [0,1]:
                identities.extend(rng.choice(by_gender[g],min(len(by_gender[g]),cfg['identities_per_gender']),replace=False))
            fi,vi,labels=[],[],[]
            for identity in identities:
                ix=groups[identity];k=cfg['samples_per_identity']
                fi.extend(rng.choice(ix,k,replace=True));vi.extend(rng.choice(ix,k,replace=True))
                labels.extend([classes.index(identity)]*k)
            f=torch.as_tensor(pack['faces'][fi],device=device)
            v=torch.as_tensor(pack['voices'][vi],device=device)
            lab=torch.tensor(labels,dtype=torch.long,device=device)
            sex=torch.as_tensor(pack['genders'][fi],dtype=torch.long,device=device)
            optim.zero_grad(set_to_none=True)
            output=model(f,v)
            if kind in ['control_v1','v1_rank']:
                total=joint_loss(model,output,lab,sex,control_cfg)
                if kind=='v1_rank':
                    scores=output[0]@output[1].T
                    total=total+cfg['rank_addon_weight']*same_gender_rank(scores,lab,sex,cfg['temperature'],cfg['ranking_margin'])
            else:
                total=block_loss(model,output,f,v,lab,sex,cfg,relation=kind.endswith('_relation'))
            if not torch.isfinite(total).item():raise RuntimeError('Non-finite research loss')
            total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optim.step()
            losses.append(total.item())
        scheduler.step()
        row={'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.time()-started}
        if validation_ids is not None:
            report=evaluate(model,kind,pack,validation_ids,device,cfg,seed+10_000)
            metric=report['mean_english_eer_percent'];row['inner_validation']=report
            if metric<best-1e-5:
                best,best_epoch=metric,epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            message=f'inner_EER={metric:.2f}% best={best:.2f}%'
        else:
            best_epoch=epoch;message='fixed refit; outer validation unopened'
        history.append(row);save_json(out/'history.json',history)
        print(f'{out.name} {kind} epoch {epoch}/{epochs} loss={row["loss"]:.3f} {message}',flush=True)
        if validation_ids is not None and epoch-best_epoch>=cfg['patience']:break
    if best_state is None:best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state);model.eval()
    torch.save({'kind':kind,'state_dict':best_state,'config':cfg,'seed':seed,'classes':classes,
        'epoch':best_epoch,'parameters':sum(p.numel() for p in model.parameters()),
        'role':'offline_research_not_a_final_evaluation_system'},out/'best.pt')
    return model,{'kind':kind,'epoch':best_epoch,'inner_mean_eer':best if validation_ids is not None else None,
                   'checkpoint':str(out/'best.pt'),'parameters':sum(p.numel() for p in model.parameters())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='research_v2/config.yaml')
    parser.add_argument('--out',default='research_v2/runs/nested_v2')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    pack,old=development_pack(cfg['pack'],cfg['original_split'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Completed research run; use another output directory')
    device=device_for(cfg['device'])
    ids=np.unique(pack['labels'])
    sex=np.asarray([pack['genders'][np.flatnonzero(pack['labels']==i)[0]] for i in ids])
    folds=StratifiedKFold(n_splits=cfg['outer_folds'],shuffle=True,random_state=cfg['seed'])
    before=sha256(ROOT/'artifacts/final/system.json')
    records=[]
    for fold,(fit_idx,val_idx) in enumerate(folds.split(ids,sex)):
        directory=out/f'fold{fold}'
        outer_fit,outer_val=ids[fit_idx],ids[val_idx]
        inner_fit,inner_val=train_test_split(outer_fit,test_size=cfg['inner_validation_fraction'],
            stratify=sex[fit_idx],random_state=cfg['seed']+fold+1)
        assert not set(outer_val)&set(inner_fit) and not set(outer_val)&set(inner_val)
        assert not set(ids)&set(old['audit'])
        save_json(directory/'split.json',{'inner_fit':inner_fit.tolist(),'inner_validation':inner_val.tolist(),
            'outer_fit':outer_fit.tolist(),'outer_validation':outer_val.tolist(),'excluded_original_audit':old['audit']})
        inners=[]
        for kind in cfg['variants']:
            _,r=fit(kind,pack,inner_fit.tolist(),cfg,directory/f'inner_{kind}',device,cfg['seed']+fold,
                    validation_ids=inner_val.tolist())
            inners.append(r)
        new=min([r for r in inners if r['kind']!='control_v1'],key=lambda r:r['inner_mean_eer'])
        control=next(r for r in inners if r['kind']=='control_v1')
        save_json(directory/'inner_selection.json',{'candidates':inners,'selected_proposal':new,
            'selection_uses':'inner_validation_only'})
        record={'fold':fold,'outer_identities':len(outer_val),'selected_kind':new['kind'],'results':{}}
        for label,chosen in [('control',control),('proposal',new)]:
            model,r=fit(chosen['kind'],pack,outer_fit.tolist(),cfg,directory/f'outer_{label}',device,
                        cfg['seed']+fold,fixed_epochs=chosen['epoch'])
            result=evaluate(model,chosen['kind'],pack,outer_val.tolist(),device,cfg,cfg['seed']+20_000+fold)
            record['results'][label]={**r,'outer_validation':result}
        records.append(record);save_json(out/'folds.json',records)
        print('OUTER FOLD',json.dumps(record),flush=True)
    summary={'role':'offline_nested_development_validation','folds':records,'development_identities':len(ids),
        'original_audit_excluded':len(old['audit']),'pack_sha256':sha256(cfg['pack']),
        'config':cfg,'baseline_frozen_manifest_sha256':before,'no_evaluation_data_access':True,
        'limitations':'Previously researched English development identities. Not a fresh external test or a Bengali estimate. Outer folds have few identities; do not use final leaderboard results to select this research system.'}
    for label in ['control','proposal']:
        values=[r['results'][label]['outer_validation']['mean_english_eer_percent'] for r in records]
        summary[label]={'mean_english_eer_percent':float(np.mean(values)),
                        'fold_sample_std_percent':float(np.std(values,ddof=1)),'fold_means':values}
    assert sha256(ROOT/'artifacts/final/system.json')==before
    save_json(out/'summary.json',summary)
    print('RESEARCH SUMMARY',json.dumps({k:summary[k] for k in ['control','proposal']},indent=2),flush=True)


if __name__=='__main__':main()
