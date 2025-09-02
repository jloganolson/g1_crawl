import os
import sys
import time

sys.path.append(os.getcwd())

# Constants - hardcoded configuration values
HUMANOID_XML = "./g1_description/g1_mjx_alt.xml"
SCENE_XML = "./g1_description/scene_mjx_alt.xml"
MOTION_FILE = "../data/amass_all.pkl"


import numpy as np
import mujoco
import mujoco.viewer
import joblib

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_connector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                        point1.astype(np.float64), point2.astype(np.float64))

def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif chr(keycode) == "T":
        print("next")
        motion_id = (motion_id + 1) % len(motion_data_keys)
        curr_motion_key = motion_data_keys[motion_id]
        print(curr_motion_key)
    else:
        print("not mapped", chr(keycode))
    
    
        
def main() -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    humanoid_xml = HUMANOID_XML
    scene_xml = SCENE_XML
    
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1/30, False
    
    motion_file = MOTION_FILE
    print(motion_file)
    motion_data = joblib.load(motion_file)
    motion_data_keys = list(motion_data.keys())

    mj_model = mujoco.MjModel.from_xml_path(scene_xml)
    mj_data = mujoco.MjData(mj_model)

    
    mj_model.opt.timestep = dt
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        for _ in range(50):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        # for _ in range(24):
        #     add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([0, 1, 0, 1]))
        # Close the viewer automatically after 30 wall-seconds.
        while viewer.is_running():
            step_start = time.time()
            curr_motion_key = motion_data_keys[motion_id]
            curr_motion = motion_data[curr_motion_key]
            curr_time = int(time_step/dt) % curr_motion['dof'].shape[0]
            
            mj_data.qpos[:3] = curr_motion['root_trans_offset'][curr_time]
            mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]]
            mj_data.qpos[7:] = curr_motion['dof'][curr_time]
                
            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt

            # pose_aa_h1_new = torch.cat([torch.from_numpy(sRot.from_quat(root_rot).as_rotvec()[None, :, None]), H1_ROTATION_AXIS * dof_pos[None, ..., None], torch.zeros((1, 1, 2, 3))], axis = 2).float()
            # for i in range(rg_pos_t.shape[1]):
            #     if not i in [20, 21, 22]:
            #         continue
            #     viewer.user_scn.geoms[i].pos = rg_pos_t[0, i]
                # viewer.user_scn.geoms[i].pos = mj_data.xpos[:][i]
            
            joint_gt = motion_data[curr_motion_key]['smpl_joints']
            
            for i in range(joint_gt.shape[1]):
                viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]
                
            # joints_opt = kps_dict['j3d_h1'].squeeze()#[h1_joint_pick_idx]
            # for i in range(len(joints_opt)):
            #     viewer.user_scn.geoms[i].pos = joints_opt[i]
                

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            # if RECORDING and time_step > motion_len:
            #     curr_start += num_motions
            #     motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
            #     time_step = 0

if __name__ == "__main__":
    main()
