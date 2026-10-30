"""Prepare explicit inferred-hand delivery inputs from original reconstructed bundles.

No object mesh/pose comes from the competition template. Unobserved hands are
labeled inferred, never promoted to observations or used as quality evidence.
"""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import numpy as np
from experiments.nvidia_v2d.sharpa_bridge import digest,write_json,read_source,export_episode
from experiments.nvidia_v2d.sharpa_multi_object import assemble


def prepare(run: Path, plan: Path, output: Path) -> dict:
    output = output.resolve()
    state=json.loads((run/'run.json').read_text());output.mkdir(exist_ok=False,parents=True)
    grouped={}
    for job in state['jobs']:
        if 'track_3-evaluation-' not in job['run_key']:continue
        path=run/('preserved' if job['status']=='failed' else 'workspace/jobs')/job['job_id']
        bundle=path/'reconstruction/result_slam'
        if not (bundle/'result.npz').exists():bundle=path/'reconstruction/result'
        with np.load(bundle/'result.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        grouped.setdefault(job['run_key'],[]).append((job,path,bundle,arrays))
    episodes=[]
    for sequence,items in grouped.items():
        source=read_source(plan,sequence);count=source.frame_count
        if digest(source.source_video)!=source.source_video_sha256:raise ValueError('video hash mismatch')
        by_name={x[0]['target_name']:x for x in items}
        if set(by_name)!=set(source.object_names):raise ValueError('object roster mismatch')
        first=items[0][3]
        for _,_,_,a in items:
            for k in ('camera_to_world_transform','camera_intrinsics','camera_is_valid'):
                if not np.array_equal(first[k],a[k]):raise ValueError('camera gauge mismatch')
        canonical={};provenance={}
        for side in ('left','right'):
            prefix='hand_'+side;choices=[]
            for job,path,bundle,a in items:
                records=[]
                for file in (path/'reconstruction/hamer_aligned').glob('*/*.json'):
                    value=json.loads(file.read_text());idx=int(value['frame_idx'])
                    if value['is_right']==(side=='right') and 0<=idx<count and a[prefix+'_is_valid'][idx]:records.append(idx)
                choices.append((len(set(records)),path,bundle,a,sorted(set(records))))
            _,path,bundle,a,observed=max(choices,key=lambda x:x[0])
            src=bundle/'result.npz';fields={k:v.copy() for k,v in a.items() if k.startswith(prefix+'_')}
            codes=np.full(count,3,dtype=np.uint8)
            if observed:
                indices=np.asarray(observed);nearest=indices[np.abs(np.arange(count)[:,None]-indices[None,:]).argmin(axis=1)]
                for k,v in fields.items():
                    if k!=prefix+'_betas' and v.shape and v.shape[0]==count:fields[k]=v[nearest].copy()
                codes[indices]=1
            else:
                # An absent camera hand is kept away from the manipulated objects.
                fields[prefix+'_betas']=np.zeros(10)
                fields[prefix+'_wrist_orient_in_camera']=np.zeros((count,3))
                fields[prefix+'_finger_pose']=np.zeros((count,15,3))
                fields[prefix+'_wrist_trans_in_camera']=np.tile([3.,0.,3.],(count,1))
                fields[prefix+'_scale']=np.ones(count)
            fields[prefix+'_source_code']=codes
            fields[prefix+'_is_valid']=np.ones(count,dtype=bool)
            fields[prefix+'_is_observed']=codes==1
            # These transforms are retained consistently, although the bridge uses MANO fields.
            from scipy.spatial.transform import Rotation
            tf=np.tile(np.eye(4),(count,1,1));tf[:,:3,:3]=Rotation.from_rotvec(fields[prefix+'_wrist_orient_in_camera']).as_matrix();tf[:,:3,3]=fields[prefix+'_wrist_trans_in_camera']
            fields[prefix+'_wrist_to_camera_transform']=tf
            fields[prefix+'_wrist_to_world_transform']=first['camera_to_world_transform']@tf
            canonical.update(fields)
            provenance[side]={'method':'original reconstructed shape; nearest observed pose for missing frames' if observed else 'unobserved hand parked outside task; neutral MANO shape',
                'source_job':str(path),'observed_frame_indices':observed,'inferred_frame_indices':np.flatnonzero(codes==3).tolist(),
                'inputs':[{'path':str(src),'sha256':digest(src)}]}
        jobs={}
        for name in source.object_names:
            job,path,bundle,a=by_name[name];dest=output/sequence/name;dest.mkdir(parents=True)
            q=dest/'hand-quality';q.mkdir();recon=dest/'reconstruction/result';recon.mkdir(parents=True)
            manifest=json.loads((bundle/'manifest.json').read_text())
            for filename in ['manifest.json','mesh.obj',*manifest['mesh'].get('materials',[]),*manifest['mesh'].get('textures',[])]:
                if Path(filename).name!=filename:raise ValueError('nonlocal mesh asset')
                shutil.copyfile(bundle/filename,recon/filename)
            merged={**a,**canonical};np.savez_compressed(q/'quality_result.npz',**merged)
            write_json(q/'manifest.json',{'schema':'submission_derived_numerical_input.v1','geometry_quality_accepted':False,'artifacts':[{'path':'quality_result.npz','sha256':digest(q/'quality_result.npz')}],
                'original_npz':str(bundle/'result.npz'),'original_npz_sha256':digest(bundle/'result.npz')})
            write_json(q/'provenance.json',{'shape_provenance':provenance,'claim':'inferred reference inputs only; actual submission poses must come from learned physics rollout'})
            write_json(dest/'experiment-config.json',job['config']);jobs[name]=str(dest)
        entry={'episode_id':sequence,'episode':int(sequence[-6:]),'jobs':jobs,'anchor':source.object_names[0],'frames':count,'hand_inference':{s:len(v['inferred_frame_indices']) for s,v in provenance.items()}}
        episodes.append(entry)
    receipt={'schema':'nvidia_v2d.submission_inputs.v1','source_run':str(run),'source_run_sha256':digest(run/'run.json'),'episodes':episodes}
    write_json(output/'manifest.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','bridge']);p.add_argument('--run',type=Path);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--inputs',type=Path);p.add_argument('--episode',type=int);p.add_argument('--upstream',type=Path);p.add_argument('--mano-root',type=Path);a=p.parse_args()
    if a.mode=='prepare':result=prepare(a.run,a.plan,a.output);print('prepared',len(result['episodes']));return
    row=next(x for x in json.loads((a.inputs/'manifest.json').read_text())['episodes'] if x['episode']==a.episode)
    source=read_source(a.plan,row['episode_id']);jobs={k:Path(v) for k,v in row['jobs'].items()}
    if len(jobs)>1:result=assemble(source,jobs,row['anchor'],a.upstream,a.mano_root,a.output,delivery=True)
    else:result=export_episode(source,next(iter(jobs.values())),a.upstream,a.mano_root,a.output,delivery=True)
    print(json.dumps({'status':result['status'],'episode':a.episode}))

if __name__=='__main__':main()
