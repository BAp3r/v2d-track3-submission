import numpy as np
import pytest
from experiments.nvidia_v2d.sharpa_multi_object import shared_poses, merge_contacts


def test_shared_scene_preserves_relative_motion_and_rejects_different_camera():
    t=np.tile(np.eye(4),(3,1,1));a={'camera_to_world_transform':t.copy(),'camera_intrinsics':np.array([1,1,0,0]),'camera_is_valid':np.ones(3,dtype=bool),'object_to_world_transform':t.copy()}
    b={k:v.copy() for k,v in a.items()};b['object_to_world_transform'][:,0,3]=[.1,.2,.3]
    align=np.eye(4);align[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];align[:3,3]=[2,3,4]
    out=shared_poses([a,b],align)
    np.testing.assert_allclose(np.linalg.inv(out[:,0])@out[:,1],b['object_to_world_transform'],atol=1e-14)
    b['camera_to_world_transform'][1,0,3]=1
    with pytest.raises(ValueError,match='camera/frame gauge'):shared_poses([a,b],align)


def test_contact_body_identity_and_normals_follow_selected_surface():
    def c(offset, ids):
        return {'tips_distance':[[offset]*5],'object_contact_part_ids':[ids],
                'link_contact_positions':np.zeros((1,3,3)).tolist(),
                'object_contact_positions':np.full((1,3,3),offset).tolist(),
                'link_contact_normals':np.tile([0,1,0],(1,3,1)).tolist(),
                'object_contact_normals':np.tile([0,0,1],(1,3,1)).tolist()}
    a=c(.009,[1,0,0]);b=c(.002,[1,1,0]);b['object_contact_normals']=[[[1,0,0]]*3]
    out=merge_contacts([a,b]);assert out['object_contact_part_ids']==[[2,2,0]]
    assert out['object_contact_normals']==[[[1,0,0],[1,0,0],[0,0,0]]]
    assert out['object_contact_positions'][0][2]==[0,0,0]
    assert out['tips_distance']==[[.002]*5]
