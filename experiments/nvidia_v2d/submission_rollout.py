"""Native-speed zero-shot submission rollout; no checkpoint mutation or object assistance."""
import json,runpy,sys
from dataclasses import replace
from pathlib import Path
from flash_chord.evaluation import harness
old_config=harness.checkpoint_config
def config(metadata):
    value=old_config(metadata)
    value["task"]["motion_speed"]=1.0
    value["scene"]["support"]=False
    for scope in (value,value["env"]):
        obj=scope.get("objective",{})
        if "release_contact_objective" in obj.get("_target_",""):
            obj["_target_"]="flash_chord.objectives.config.ObjectiveConfig"
            obj.pop("release_start_frame",None)
    return value
harness.checkpoint_config=config
old=harness.setup_scene
receipt=[]
def setup(**kwargs):
    # A first new object diagnostic uses the official infinite-ground scene, not the pot's table.
    kwargs['include_support']=False
    kwargs['collision']=replace(kwargs['collision'],ground=True)
    kwargs['motion_speed']=1.0
    result=old(**kwargs)
    model=result.scene.model
    limits=[]
    for i,label in enumerate(model.joint_label):
        if '_wrist_' in label:
            di=int(model.joint_qd_start.numpy()[i])
            limits.append(dict(label=label, dof=di, lower=float(model.joint_limit_lower.numpy()[di]), upper=float(model.joint_limit_upper.numpy()[di])))
    receipt.append({'joint_limits':limits})
    receipt.append({'parquet':str(kwargs['parquet']),'motion_speed':1.,'include_support':False,
        'ground':True,'objective':'default ObjectiveConfig; pot release407 removed for new episode','num_frames':result.reference.num_frames,'source':'explicit_diagnostic_scene_override'})
    return result
harness.setup_scene=setup

try:runpy.run_module('experiments.nvidia_v2d.policy_physics_diagnostic',run_name='__main__')
finally:
    if '--output' in sys.argv:
        out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(exist_ok=True,parents=True)
        (out/'scene-override-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    harness.setup_scene=old
    harness.checkpoint_config=old_config
