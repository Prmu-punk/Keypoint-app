import torch
import json

def load_parameters(self, human_params_file, org_path, camera_path, object_params_file=None):
    """
    从JSON文件加载人体和物体参数
    :param human_params_file: 人体参数JSON文件路径
    :param object_params_file: 物体参数JSON文件路径(可选)
    :return: 解析后的参数字典
    """
    # 加载人体参数
    with open(human_params_file, 'r') as f:
        human_params = json.load(f)

    # 将字符串键转换为整数并排序
    sorted_frames = sorted([int(k) for k in human_params['body_pose'].keys()])

    # 重新组织参数，按帧序列排列
    organized_human_params = {
        'body_pose': [],
        'betas': [],
        'global_orient': [],
        'transl': [],
        'left_hand_pose': [],
        'right_hand_pose': [],
    }
    incam_cam = []
    global_cam = []
    camera_params = torch.load(camera_path)
    incam_params = camera_params["smpl_params_incam"]
    global_params = camera_params["smpl_params_global"]
    for id, frame_idx in enumerate(sorted_frames):
        frame_str = str(frame_idx)
        organized_human_params['body_pose'].append(
            torch.tensor(human_params['body_pose'][frame_str], dtype=torch.float32))
        organized_human_params['betas'].append(torch.tensor(human_params['betas'][frame_str], dtype=torch.float32))
        organized_human_params['global_orient'].append(
            torch.tensor(human_params['global_orient'][frame_str], dtype=torch.float32))
        organized_human_params['transl'].append(torch.tensor(human_params['transl'][frame_str], dtype=torch.float32))
        organized_human_params['left_hand_pose'].append(
            torch.tensor(human_params['left_hand_pose'][frame_str], dtype=torch.float32))
        organized_human_params['right_hand_pose'].append(
            torch.tensor(human_params['right_hand_pose'][frame_str], dtype=torch.float32))
        incam_cam.append((incam_params['global_orient'][id], incam_params['transl'][id]))
        global_cam.append((global_params['global_orient'][id], global_params['transl'][id]))

    result = {
        'human_params': organized_human_params,
        'frame_indices': sorted_frames,
        'total_frames': len(sorted_frames),
        'global_params': global_cam,
        'incam_params': incam_cam
    }

    # 如果提供了物体参数文件，也加载它
    if object_params_file and os.path.exists(object_params_file):
        with open(object_params_file, 'r') as f:
            object_params = json.load(f)
        org_params = json.load(open(org_path, 'r'))

        organized_object_params = {
            'poses': [],  # R_final
            'centers': [],  # t_final
            'scale': org_params.get('scale', 1.0),  # 获取scale参数，默认为1.0
            'scale_init': None
        }
        organized_object_params['scale_init'] = object_params['scale'] if object_params['scale'] is not None else 1.0

        for frame_idx in sorted_frames:
            frame_str = str(frame_idx)
            if frame_str in object_params['poses']:
                organized_object_params['poses'].append(
                    torch.tensor(object_params['poses'][frame_str], dtype=torch.float32))
                organized_object_params['centers'].append(
                    torch.tensor(object_params['centers'][frame_str], dtype=torch.float32))
            else:
                # 如果某帧没有物体参数，填充零
                organized_object_params['poses'].append(torch.eye(3, dtype=torch.float32))
                organized_object_params['centers'].append(torch.zeros(3, dtype=torch.float32))

        result['object_params'] = organized_object_params
        print(f"Loaded object parameters with scale: {organized_object_params['scale']}")

    print(f" Loaded parameters for {result['total_frames']} frames")
    return result