from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.nvidia_v2d.policy_geometry_ablation import (
    apply_normal_factor,
    audit_factor,
    compose_factor,
)


def normal_example():
    cells={'object_body_position':np.zeros((249,1,3)).tolist(),
           'object_body_wxyz':np.tile([1.,0,0,0],(249,1,1)).tolist()}
    patch={'source_object_body_position':np.asarray(cells['object_body_position']),
           'source_object_body_wxyz':np.asarray(cells['object_body_wxyz'])}
    for side in ('left','right'):
        ids=np.zeros((249,5),int);ids[100,2]=9  # unknown supervision must survive
        status=np.zeros_like(ids);status[100,2]=3
        normals=np.zeros((249,5,3));normals[100,2]=[0,0,1]
        if side=='left':
            ids[[143,201],3]=7;status[[143,201],3]=2;normals[[143,201],3]=[1,0,0]
        cells[f'mano_{side}_object_contact_part_ids']=ids.tolist()
        cells[f'mano_{side}_object_contact_positions']=np.ones((249,5,3)).tolist()
        cells[f'mano_{side}_object_contact_normals']=normals.tolist()
        patch.update({f'{side}_part_ids':ids,f'{side}_status':status,f'{side}_object_local_normals':-normals})
    return cells,patch


def test_normal_changes_only_two_normals_and_retains_unknown_ids_and_points():
    cells,patch=normal_example();before=deepcopy(cells);result=apply_normal_factor(cells,patch)
    assert cells==before
    for side in ('left','right'):
        np.testing.assert_array_equal(result[f'mano_{side}_object_contact_positions'],before[f'mano_{side}_object_contact_positions'])
        np.testing.assert_array_equal(result[f'mano_{side}_object_contact_part_ids'],before[f'mano_{side}_object_contact_part_ids'])
        assert result[f'mano_{side}_object_contact_normals'][100][2]==[0,0,1]
    changed=np.argwhere(np.any(np.asarray(result['mano_left_object_contact_normals'])!=before['mano_left_object_contact_normals'],axis=-1))
    assert changed.tolist()==[[143,3],[201,3]]


def test_normal_rejects_wrong_source_ids_or_unreviewed_extra_edit():
    cells,patch=normal_example();patch['left_part_ids'][100,2]=0
    with pytest.raises(AssertionError):apply_normal_factor(cells,patch)
    cells,patch=normal_example();patch['right_status'][144,3]=2
    with pytest.raises(ValueError,match='only reviewed'):apply_normal_factor(cells,patch)


def test_actual_factor_isolation_and_detects_tampering():
    import pyarrow.parquet as pq

    from experiments.nvidia_v2d.support_reference_repair import load_support_solution
    root=Path(__file__).resolve().parents[2];base=root/'.e2e/zero_success_20261027'
    source=base/'inputs/legacy/retarget/public44-official-002/data.parquet'
    if not source.exists():pytest.skip('Repository experiment fixture absent')
    repair=base/'reference-repair-20261029-001';cells=pq.read_table(source).to_pylist()[0]
    patch=dict(np.load(repair/'normals/training-candidate-002/contact-repair.npz'))
    support=load_support_solution(repair/'support')
    for factor in ('normal','support','normal-support'):
        result=compose_factor(cells,factor,patch=patch,support_solution=support)
        report=audit_factor(cells,result,factor,patch=patch,support_solution=support)
        assert all(report['checks'].values())
        result['robot_right_finger_joints'][210][0]+=.001
        with pytest.raises((AssertionError,ValueError)):audit_factor(cells,result,factor,patch=patch,support_solution=support)
    result=compose_factor(cells,'support',patch=patch,support_solution=support)
    result['mano_left_joints'][210][3][0]+=.001
    with pytest.raises(AssertionError):audit_factor(cells,result,'support',patch=patch,support_solution=support)
