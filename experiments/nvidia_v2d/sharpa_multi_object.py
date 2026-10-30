"""Assemble independent reconstructed rigid bodies in one camera/world gauge.

The anchor job supplies one hand trajectory and the single scene transform.
Per-object contacts are diagnostic nearest-surface estimates, not measured force.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.nvidia_v2d.sharpa_bridge import (
    _assets, _contacts, digest, export_episode, read_source, write_json,
)
from experiments.nvidia_v2d.sharpa_episode import EpisodeSource, audit_bundle, require_new_output


def shared_poses(arrays: list[dict[str, np.ndarray]], alignment: np.ndarray) -> np.ndarray:
    """Reject unregistered camera gauges; apply the same transform to every body."""
    first = arrays[0]
    for item in arrays[1:]:
        for key in ('camera_to_world_transform', 'camera_intrinsics', 'camera_is_valid'):
            if not np.array_equal(first[key], item[key]):
                raise ValueError(f'objects do not share camera/frame gauge: {key}')
    return np.stack([alignment[None] @ item['object_to_world_transform'] for item in arrays], axis=1)


def merge_contacts(contacts: list[dict[str, Any]]) -> dict[str, list]:
    """Choose the closest active per-body estimate for each fixed hand-link slot.

    Body IDs are 1-based; zero means inactive. All geometry fields come from
    the same selected body so normals/positions/IDs cannot be mixed.
    """
    active = np.stack([np.asarray(c['object_contact_part_ids']) > 0 for c in contacts])
    distances = np.stack([np.linalg.norm(np.asarray(c['link_contact_positions']) -
                            np.asarray(c['object_contact_positions']), axis=-1) for c in contacts])
    choice = np.where(active, distances, np.inf).argmin(axis=0)
    valid = active.any(axis=0)
    result = {'tips_distance': np.min([c['tips_distance'] for c in contacts], axis=0).tolist(),
              'object_contact_part_ids': np.where(valid, choice + 1, 0).tolist()}
    for key in ('link_contact_positions', 'link_contact_normals',
                'object_contact_positions', 'object_contact_normals'):
        stack = np.stack([c[key] for c in contacts])
        picked = np.take_along_axis(stack, choice[None, ..., None], axis=0)[0]
        result[key] = np.where(valid[..., None], picked, 0).tolist()
    return result


def assemble(source: EpisodeSource, jobs: dict[str, Path], anchor: str,
             upstream: Path, mano_root: Path, output: Path, *, delivery: bool = False) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import trimesh
    from scipy.spatial.transform import Rotation

    names = source.object_names
    if len(names) < 2 or len(set(names)) != len(names) or set(jobs) != set(names) or anchor not in names:
        raise ValueError('require all distinct episode bodies and an explicit hand anchor')
    require_new_output(output, [*jobs.values(), source.source_video.parent, upstream, mano_root])
    arrays, audits, inputs = [], [], {}
    for name in names:
        job = jobs[name]
        config_path = job / 'experiment-config.json'
        config = json.loads(config_path.read_text())
        expected = {'run_key': source.sequence_id, 'target_name': name,
                    'source_video_sha256': source.source_video_sha256,
                    'frame_count': source.frame_count, 'fps': source.fps}
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError(f'job identity/time mismatch: {name}')
        quality = job / 'hand-quality'
        npz = quality / 'quality_result.npz'
        manifest = json.loads((quality / 'manifest.json').read_text())
        entries = [r for r in manifest['artifacts'] if r['path'] == npz.name]
        if len(entries) != 1 or entries[0]['sha256'] != digest(npz):
            raise ValueError(f'quality hash mismatch: {name}')
        with np.load(npz, allow_pickle=False) as f:
            item = {k: f[k] for k in f.files}
        audit = audit_bundle(source.model_copy(update={'object_names': [name]}), item,
                             allow_inferred_hands=delivery)
        if not audit.input_contract_passed:
            raise ValueError(f'{name}: {audit.blockers}')
        arrays.append(item)
        audits.append(audit.model_dump(mode='json'))
        for p in (config_path, npz, quality / 'manifest.json', job / 'reconstruction/result/mesh.obj',
                  job / 'reconstruction/result/manifest.json'):
            inputs[str(p)] = digest(p)
    # Check before doing expensive hand FK/export.
    shared_poses(arrays, np.eye(4))
    output.mkdir(parents=True, exist_ok=False)
    anchor_output = output / 'anchor-only'
    receipt = export_episode(source.model_copy(update={'object_names': [anchor]}),
                             jobs[anchor], upstream, mano_root, anchor_output, delivery=delivery)
    if not receipt['input_contract_passed']:
        raise ValueError('anchor hand export failed')
    row_table = pq.read_table(anchor_output / 'loaded.parquet')
    row = row_table.to_pylist()[0]
    with np.load(anchor_output / 'geometry.npz', allow_pickle=False) as f:
        geometry = {k: f[k] for k in f.files}
    alignment = geometry['source_to_scene_transform']
    poses = shared_poses(arrays, alignment)
    meshes, paths, urdfs = [], [], []
    for i, name in enumerate(names):
        if name == anchor:
            path = Path(row['object_mesh_paths'][0]);urdf = Path(row['object_urdf_paths'][0])
            mesh = trimesh.load(path, force='mesh', process=False)
        else:
            mesh, path, urdf = _assets(jobs[name] / 'reconstruction/result', output,
                                      upstream, name, float(arrays[i]['object_scale']))
        meshes.append(mesh);paths.append(str(path));urdfs.append(str(urdf))
    n = source.frame_count
    row.update(object_name='+'.join(names), safe_object_name='+'.join(names),
               object_body_names=names, safe_object_body_names=names,
               object_mesh_paths=paths, object_urdf_paths=urdfs,
               object_mesh_radius=[float(np.linalg.norm(m.vertices-m.vertices.mean(0), axis=1).max()) for m in meshes],
               object_body_position=poses[:, :, :3, 3].tolist(),
               object_body_wxyz=Rotation.from_matrix(poses[:, :, :3, :3].reshape(-1,3,3)).as_quat()
                   .reshape(n,len(names),4)[:, :, [3,0,1,2]].tolist())
    # Root fields remain the explicitly chosen anchor; body arrays are authoritative.
    for side in ('left', 'right'):
        hand = {k: geometry[f'{side}_{k}'] for k in ('vertices','faces','joints')}
        per_body = [_contacts(hand, mesh, poses[:, i], upstream) for i, mesh in enumerate(meshes)]
        for key, value in merge_contacts(per_body).items():
            row[f'mano_{side}_{key}'] = value
    parquet = output / 'loaded.parquet'
    pq.write_table(pa.Table.from_pylist([row], schema=row_table.schema), parquet, compression='zstd')
    geometry['object_to_world_transform'] = poses
    np.savez_compressed(output / 'geometry.npz', **geometry)
    calibration = json.loads((anchor_output / 'calibration.json').read_text())
    calibration.update(hand_source_job=str(jobs[anchor]), alignment_anchor=anchor,
                       alignment_source='single_anchor_pca_applied_to_all_bodies',
                       object_names=names)
    write_json(output / 'calibration.json', calibration)
    before = np.linalg.inv(arrays[0]['object_to_world_transform'].astype(float)) @ arrays[1]['object_to_world_transform']
    after = np.linalg.inv(poses[:,0]) @ poses[:,1]
    relative_error = float(np.max(np.abs(before-after)))
    if relative_error > 1e-5:
        raise ValueError('relative body transform was not preserved')
    gate = dict(audits[names.index(anchor)])
    gate.update(source=source.model_dump(mode='json'), assembled_multi_object_scene=True,
                hand_anchor=anchor, per_object_audits=audits,
                shared_camera_exact=True, relative_transform_max_abs_error=relative_error)
    write_json(output / 'input-gate.json', gate)
    inputs.update(receipt['input_sha256'])
    if not all(digest(Path(p)) == h for p,h in inputs.items()):
        raise ValueError('source changed during assembly')
    result = dict(schema='nvidia_v2d.sharpa_shared_scene.v1', status='succeeded',
                  source=source.model_dump(mode='json'), parquet=str(parquet),
                  input_contract_passed=True, geometry_quality_accepted=False,
                  input_sha256=inputs, input_files_unchanged=True,
                  alignment_anchor=anchor, hand_source=str(jobs[anchor]),
                  shared_camera_exact=True, relative_transform_max_abs_error=relative_error,
                  contact_definition='closest active original 1cm per-body estimate; one body per hand-link slot',
                  contact_part_id_counts={s:{str(i):int((np.asarray(row[f'mano_{s}_object_contact_part_ids'])==i).sum())
                                            for i in range(len(names)+1)} for s in ('left','right')},
                  output_sha256={str(p):digest(p) for p in [parquet,output/'calibration.json',output/'geometry.npz',output/'input-gate.json']})
    write_json(output / 'manifest.json', result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--sequence-id',required=True)
    p.add_argument('--jobs',type=Path,required=True);p.add_argument('--anchor',required=True)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--mano-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=assemble(read_source(a.plan,a.sequence_id), {k:Path(v) for k,v in json.loads(a.jobs.read_text()).items()},
                    a.anchor,a.upstream,a.mano_root,a.output)
    print(json.dumps({k:result[k] for k in ('status','parquet','relative_transform_max_abs_error','contact_part_id_counts')}))

if __name__=='__main__':main()
