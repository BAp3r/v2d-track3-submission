"""Run all ready scoring episodes with existing learned actors, then export actual poses."""
from pathlib import Path
import argparse,json,subprocess,time,os,shutil,sys
import numpy as np
import pyarrow.parquet as pq
from experiments.nvidia_v2d.sharpa_bridge import digest
from experiments.nvidia_v2d.policy_trajectory import validate_trajectory
from experiments.nvidia_v2d.policy_metric_logging import JsonMetricLogger


def export(case, row, dest, kit, checkpoint):
    sys.path.insert(0,str(kit));from v2dlb.submission_parquet import read_trajectory
    p=case/'rollout-001/result';src=p/'diagnostic.object_trajectories.parquet';m=json.loads((p/'diagnostic.manifest.json').read_text());d=np.load(p/'diagnostic.npz');ref=case/'official-001/data.parquet'
    frames=row['frames'];validate_trajectory(src,frames,1)
    assert m['action_mode']=='policy' and m['checkpoint_sha256']==digest(checkpoint)
    assert d['policy_actions'].shape==(frames,1,56) and np.any(d['policy_actions']!=0)
    assert np.isfinite(d['achieved_pose_w']).all() and np.all(d['auxiliary_scales']==0)
    tab=pq.read_table(src);md=dict(tab.schema.metadata or {});md.update({b'flash_chord.episode_index':str(row['episode']).encode(),b'local.selection':b'fixed world0; learned zero-shot; no success filtering'})
    target=dest/'results'/f"episode_{row['episode']:06d}.parquet";pq.write_table(tab.replace_schema_metadata(md),target)
    tr=read_trajectory(target);assert tr.episode_index==row['episode'];assert np.array_equal(tr.pose_xyzw,d['achieved_pose_w'].reshape(-1,7))
    reference=pq.read_table(ref).to_pylist()[0];assert reference['sequence_id']==row['episode_id']
    meshes=dest/'meshes'/f"episode_{row['episode']:06d}";meshes.mkdir()
    binds={}
    for name,path in zip(reference['object_body_names'],reference['object_mesh_paths'],strict=True):
        mesh=Path(path).with_name('collision.stl');out=meshes/(name+'.stl');shutil.copyfile(mesh,out);binds[name]={'source':str(mesh),'sha256':digest(mesh),'output_sha256':digest(out)}
    receipt={'episode':row['episode'],'frames':frames,'rows':tab.num_rows,'source_sha256':digest(src),'output_sha256':digest(target),'checkpoint_sha256':digest(checkpoint),'reference_sha256':digest(ref),'assistance':0,'meshes':binds,'roundtrip_max_abs':0.,'status':'exported','policy':'learned zero-shot; no quality claim','hand_inference':row['hand_inference']}
    (case/'delivery-receipt.json').write_text(json.dumps(receipt,indent=2));return receipt


def run(root, repo, legacy, kit):
    root=root.resolve();repo=repo.resolve();base=root.parent
    rows=json.loads((root/'inputs-002/manifest.json').read_text())['episodes']
    dest=root/'candidate-complete-001';(dest/'results').mkdir(parents=True);(dest/'meshes').mkdir()
    code=root/'rollout-code-001';shutil.copytree(repo/'experiments',code/'experiments',ignore=shutil.ignore_patterns('__pycache__'))
    pending={x['episode']:x for x in rows};status=[];deadline=time.monotonic()+14400
    while pending and time.monotonic()<deadline:
        ready=[]
        for ep,row in pending.items():
            case=root/f'episode-{ep:06d}';f=case/'cpu-status.json'
            if f.exists():
                st=json.loads(f.read_text())
                if st.get('status')=='completed':ready.append(ep)
                elif any(x['exit_code'] for x in st['stages']):ready.append(ep)
        if not ready:time.sleep(5);continue
        for ep in ready:
            row=pending.pop(ep);case=root/f'episode-{ep:06d}';st=json.loads((case/'cpu-status.json').read_text())
            if st.get('status')!='completed':status.append({'episode':ep,'status':'cpu_failed'});continue
            checkpoint=(base/'coverage-20261030-001/train03-official-001/train/policy_131072.safetensors' if len(row['jobs'])>1 else base/'threeway-rollout-20261029-001/run-official-seed42/train/policy_2097152.safetensors')
            out=case/'rollout-001';out.mkdir();image='sha256:7fdce3ad594fbad8b5983fde7846908be2ed6179ba4ebcfd1edefbcbcf4c628c'
            argv=['docker','run','--rm','--init','--name',f'track3-submit-{ep}','--gpus','device=0','--network','none','--memory','20g','--cpus','8','--shm-size','2g','--user','1000:1000','-v',f'{repo}:{repo}:ro','-v',f'{code}:/adapter:ro','-v',f'{out}:/run:rw','-v',f'{legacy}:{legacy}:ro','-v',f'{base}/cache:/cache:rw','-e','PYTHONPATH=/adapter','-e','PYTHONUNBUFFERED=1','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false','-e','FLASH_CHORD_CACHE_DIR=/cache/assets','-e','JAX_COMPILATION_CACHE_DIR=/cache/jax','-e','WARP_CACHE_PATH=/cache/warp','-e','XDG_CACHE_HOME=/cache/xdg',image,'timeout','--signal=TERM','--kill-after=10s','600s','python','-m','experiments.nvidia_v2d.submission_rollout','--checkpoint',str(checkpoint),'--output','/run/result','--worlds','1','--action-mode','policy','--reference',str(case/'official-001/data.parquet'),'--capture-contacts','--motion-speed','1.0']
            (out/'command.json').write_text(json.dumps(argv,indent=2));start=time.monotonic()
            with (out/'runtime.log').open('w') as log:result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            result_row={'episode':ep,'exit_code':result.returncode,'elapsed_s':time.monotonic()-start,'status':'runtime_failed'}
            try:
                if result.returncode==0:
                    receipt=export(case,row,dest,kit,checkpoint);result_row.update(receipt)
                    report=json.loads((out/'result/diagnostic.metrics.json').read_text());metrics=report['metrics'];result_row['local_metrics']=metrics
                    context={'series_id':f'episode-{ep:06d}-zero-shot','episode':ep,'reference_parquet_sha256':digest(case/'official-001/data.parquet'),'checkpoint_sha256':digest(checkpoint),'assistance':'none','world_count':1,'step_count':row['frames'],'new_training_steps':0,'mode':'zero_shot_submission','report_path':str(out/'result/diagnostic.metrics.json')}
                    logger=JsonMetricLogger(root/'evaluation-metrics.jsonl',resume=(root/'evaluation-metrics.jsonl').exists(),context=context)
                    logger.log({'evaluation/'+k:float(v) for k,v in metrics.items() if isinstance(v,(int,float))},step=int(report.get('environment_steps',0)));logger.close()
            except Exception as exc:result_row.update(status='export_failed',failure=f'{type(exc).__name__}: {exc}')
            (out/'execution.json').write_text(json.dumps(result_row,indent=2));status.append(result_row);(root/'rollout-status.json').write_text(json.dumps(status,indent=2));print(json.dumps({k:result_row[k] for k in ('episode','status','elapsed_s')}),flush=True)
    (root/'batch-execution.json').write_text(json.dumps({'status':'completed' if not pending and all(x['status']=='exported' for x in status) else 'incomplete','episodes':status,'pending':list(pending)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--legacy',type=Path,required=True);p.add_argument('--kit',type=Path,required=True);a=p.parse_args();run(a.root,a.repo,a.legacy,a.kit)
