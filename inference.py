import os
import sys
import time
import torch
import yaml
import numpy as np
from tqdm import tqdm
import subprocess
import numpy as np
import time


from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from scripts.dataset import BartenderDataset
from scripts.network import ConditionalUnet1D
from scripts.vision_encoder import get_resnet, replace_bn_with_gn
from scripts.get_observations import ObservationSubscriber
# from robot_state_publisher import RobotStatePublisher

import rospy
from std_msgs.msg import Float32MultiArray 
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from collections import deque

import rtde_io

## Add the path to the teleop methods.
sys.path.append("/home/rpmdt05/Code/the-real-bartender/Spark/TeleopMethods")
from Spark.TeleopMethods.UR.arms import UR
from Spark.TeleopMethods.UR.dashboard import rtde_dashboard
from Spark.TeleopMethods.UR.gripper import RobotiqGripper

ROBOT_SPEED = 0.3           ## Speed of the robot in m/s
ROBOT_ACCELERATION = 0.3    ## Acceleration of the robot in m/s^2
ROBOT_BLEND = 0.001          ## Blend value for the robot
ROBOT_VELOCITY_SCALE = 1.0  

LIGHTNING_HOME = [-3.2225098609924316, -0.6799390912055969, -2.337024450302124, -1.1205544471740723, 1.7932333946228027, 1.2651606798171997]
THUNDER_HOME = [-2.82061505317688, -2.5057523250579834, 2.3684492111206055, -1.921329140663147, -1.6988776922225952, -1.2206653356552124]


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_network(config, device):
    '''
    Load the Models
    '''
    vision_encoder_front = replace_bn_with_gn(get_resnet('resnet18'))
    vision_encoder_thunder_wrist = replace_bn_with_gn(get_resnet('resnet18'))
    vision_encoder_lightning_wrist = replace_bn_with_gn(get_resnet('resnet18'))

    vision_feature_dim = 512 * 3
    state_dim = 14
    observation_dim = vision_feature_dim + state_dim
    action_dim = 14

    noise_prediction_network = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=observation_dim * config['observation_horizon'],
    )

    networks = torch.nn.ModuleDict({
        'vision_encoder_front': vision_encoder_front,
        'vision_encoder_thunder_wrist': vision_encoder_thunder_wrist,
        'vision_encoder_lightning_wrist': vision_encoder_lightning_wrist,
        'noise_prediction_network': noise_prediction_network
    }).to(device)

    checkpoint_path = config['model_path']
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    networks.load_state_dict(state_dict)

    return networks


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


def get_observations(observation_object, config, stats):
    '''
    TODO: Stream through the intel realsense camera, subscribe to the ros topics
    and get the observations.
    Think about storing a queue and when called, return the last n observations, based on the horizon.
    
    Output:
            Images: (obs_horizon, C, H, W) -> normalized
            Agent State: (obs_horizon, 14) -> normalized

    format:
        {
            'img_front': img_front,
            'img_wrist_thunder': img_wrist_thunder,
            'img_wrist_lightning': img_wrist_lightning,
            'agent_pos': agent_pos,
        }
    '''
    
    obs_dict = observation_object.get_last_n_observations()

    img_front = obs_dict['Images']['img_front']
    img_wrist_thunder = obs_dict['Images']['img_wrist_thunder']
    img_wrist_lightning = obs_dict['Images']['img_wrist_lightning']
    agent_pos = obs_dict['agent_state']

    print("Agent Pos: ", agent_pos.shape)
    print("img_front: ", img_front.shape)
    print("img_wrist_thunder: ", img_wrist_thunder.shape)
    print("img_wrist_lightning: ", img_wrist_lightning.shape)

    # Normalize the images
    img_front = img_front.astype(np.float32) / 255.0
    img_wrist_thunder = img_wrist_thunder.astype(np.float32) / 255.0
    img_wrist_lightning = img_wrist_lightning.astype(np.float32) / 255.0
    # change image axis to (C, H, W)
    img_front = np.moveaxis(img_front, -1, 1)
    img_wrist_thunder = np.moveaxis(img_wrist_thunder, -1, 1)
    img_wrist_lightning = np.moveaxis(img_wrist_lightning, -1, 1)
    
    # Normalize the agent state
    agent_pos = normalize_data(agent_pos, stats = stats['agent_pos'])

    # Change the observations to torch tensors
    img_front = torch.tensor(img_front).float()
    img_wrist_thunder = torch.tensor(img_wrist_thunder).float()
    img_wrist_lightning = torch.tensor(img_wrist_lightning).float()
    agent_pos = torch.tensor(agent_pos).float()

    return {
        'img_front': img_front,
        'img_wrist_thunder': img_wrist_thunder,
        'img_wrist_lightning': img_wrist_lightning,
        'agent_pos': agent_pos,
    }



def run_inference(obs_dict, networks, noise_scheduler, stats, config, device):
    B = 1
    pred_horizon = config['prediction_horizon']
    action_dim = config['action_dim']
    nimage_front = obs_dict['img_front'].to(device) ## (obs_horizon, C, H, W)
    nimage_thunder_wrist = obs_dict['img_wrist_thunder'].to(device) ## (obs_horizon, C, H, W)
    nimage_lightning_wrist = obs_dict['img_wrist_lightning'].to(device) ## (obs_horizon, C, H, W)

    nagent_state = obs_dict['agent_pos'].to(device) ## (obs_horizon, 14)

    with torch.no_grad():
        
        # Get the global condition
        img_front_features = networks['vision_encoder_front'](nimage_front) ## (obs_horizon, D)
        img_thunder_wrist_features = networks['vision_encoder_thunder_wrist'](nimage_thunder_wrist) ## (obs_horizon, D)
        img_lightning_wrist_features = networks['vision_encoder_lightning_wrist'](nimage_lightning_wrist) ## (obs_horizon, D)


        obs_features = torch.cat([img_front_features, img_thunder_wrist_features, img_lightning_wrist_features, nagent_state], dim=-1) ## (obs_horizon, 512 * 3 + 14)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

        # Sample the noisy action
        noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
        naction = noisy_action

        noise_scheduler.set_timesteps(config['num_diffusion_iters'])

        for k in noise_scheduler.timesteps:
            noise_pred = networks['noise_prediction_network'](
                naction, 
                k, 
                global_cond=obs_cond
            )

            naction = noise_scheduler.step(
                model_output = noise_pred,
                timestep = k,
                sample = naction
            ).prev_sample

    naction = naction.detach().to('cpu').numpy()# (B, pred_horizon, action_dim)
    naction = naction[0] ## (pred_horizon, action_dim)
    naction = unnormalize_data(naction, stats = stats['action'])

    # take action horizon number of actions
    start = config['observation_horizon'] - 1
    end = start + config['action_horizon']
    action = naction[start:end, :]

    return action

def init_robot():
    '''
    Initialize the robot
    '''
    thunder_ip = "192.168.0.101"
    lightning_ip = "192.168.0.102"
    arms = ["Thunder", "Lightning"]
    ips = [thunder_ip, lightning_ip]
    enable_control = {
        "Thunder": True,
        "Lightning": True
    }
    URs = UR(arms, ips, enable_grippers=True)
    lightning_io = rtde_io.RTDEIOInterface(lightning_ip)
    thunder_io = rtde_io.RTDEIOInterface(thunder_ip)
    lightning_io.setSpeedSlider(ROBOT_VELOCITY_SCALE)
    thunder_io.setSpeedSlider(ROBOT_VELOCITY_SCALE)

    ## Make ROS publishers for the robot
    # pubs = dict()
    # for arm in arms:
    #     pubs[arm+"_q"] = rospy.Publisher(f"/{arm.lower()}_q", Float32MultiArray, queue_size=10)
    #     pubs[arm+"_gripper"] = rospy.Publisher(f"/{arm.lower()}_gripper", Float32, queue_size=10)

    for arm in arms:
        URs.init_dashboard(arm)
        URs.init_arm(arm, enable_control=enable_control)
    return URs

def perform_action(URs: UR, action: np.ndarray) -> None:
    """
    Perform the current action as given be the model.
    Action is a numpy array of shape (action_horizon, action_dim).
    """
    movement_params = np.array([ROBOT_SPEED, ROBOT_ACCELERATION, ROBOT_BLEND])
    movement_params = np.tile(movement_params, (action.shape[0], 1))
    lightning_actions = np.concatenate((action[:, :6], movement_params), axis=1)
    lightning_actions[-1,-1] = 0
    thunder_actions = np.concatenate((action[:, 7:13], movement_params), axis=1)
    thunder_actions[-1,-1] = 0

    lightning_actions = lightning_actions.tolist()
    thunder_actions = thunder_actions.tolist()
    lightning_grippers = action[:, 6].tolist()
    thunder_grippers = action[:, 13].tolist()

    rate = rospy.rate(10)

    rate.sleep()

    URs.moveJ("Lightning", lightning_actions, asynchronous=True)
    URs.moveJ("Thunder", thunder_actions, asynchronous=False)

# class ObservationSubscriber:
#     def __init__(self, obs_horizon):
#         self.node_name = 'image_subscriber'
#         rospy.init_node(self.node_name, anonymous=True)

#         # Define the observation horizon
#         self.obs_horizon = obs_horizon

#         # Deques to store the last 'n' observations for each image
#         self.img_front_history = deque(maxlen=obs_horizon)
#         self.img_wrist_thunder_history = deque(maxlen=obs_horizon)
#         self.img_wrist_lightning_history = deque(maxlen=obs_horizon)

#         # Deques for agent states
#         self.lightning_angle_history = deque(maxlen=obs_horizon)
#         self.thunder_angle_history = deque(maxlen=obs_horizon)
#         self.lightning_gripper_history = deque(maxlen=obs_horizon)
#         self.thunder_gripper_history = deque(maxlen=obs_horizon)

#         # Temporary storage for incoming messages
#         self.latest_img_front = None
#         self.latest_img_wrist_thunder = None
#         self.latest_img_wrist_lightning = None
#         self.latest_lightning_angle = None
#         self.latest_thunder_angle = None
#         self.latest_lightning_gripper = None
#         self.latest_thunder_gripper = None

#         # Subscribers
#         rospy.Subscriber('/cameras/rgb/thunder/wrist', Image, self.thunder_wrist)
#         rospy.Subscriber('/cameras/rgb/lightning/wrist', Image, self.lightning_wrist)
#         rospy.Subscriber('cameras/rgb/both/front', Image, self.both_front)
#         rospy.Subscriber('/lightning_q', Float32MultiArray, self.lightning_q)
#         rospy.Subscriber('/thunder_q', Float32MultiArray, self.thunder_q)
#         rospy.Subscriber('/thunder_gripper', Float32, self.thunder_gripper)
#         rospy.Subscriber('/lightning_gripper', Float32, self.lightning_gripper)

#     def thunder_wrist(self, msg):
#         # Store the latest wrist image for thunder
#         self.latest_img_wrist_thunder = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

#         # Append all queued data to history when the thunder wrist image arrives
#         if self.latest_img_wrist_thunder is not None:
#             self.img_wrist_thunder_history.append(self.latest_img_wrist_thunder)
#             if self.latest_img_front is not None:
#                 self.img_front_history.append(self.latest_img_front)
#             if self.latest_img_wrist_lightning is not None:
#                 self.img_wrist_lightning_history.append(self.latest_img_wrist_lightning)
#             if self.latest_lightning_angle is not None:
#                 self.lightning_angle_history.append(self.latest_lightning_angle)
#             if self.latest_thunder_angle is not None:
#                 self.thunder_angle_history.append(self.latest_thunder_angle)
#             if self.latest_lightning_gripper is not None:
#                 self.lightning_gripper_history.append(self.latest_lightning_gripper)
#             if self.latest_thunder_gripper is not None:
#                 self.thunder_gripper_history.append(self.latest_thunder_gripper)

#     def lightning_wrist(self, msg):
#         # Store the latest lightning wrist image
#         self.latest_img_wrist_lightning = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

#     def both_front(self, msg):
#         # Store the latest front image
#         self.latest_img_front = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

#     def lightning_q(self, msg):
#         # Store the latest lightning angle (size 6)
#         self.latest_lightning_angle = np.asarray(msg.data)

#     def thunder_q(self, msg):
#         # Store the latest thunder angle (size 6)
#         self.latest_thunder_angle = np.asarray(msg.data)

#     def lightning_gripper(self, msg):
#         # Store the latest lightning gripper state (scalar)
#         self.latest_lightning_gripper = np.asarray(msg.data)

#     def thunder_gripper(self, msg):
#         # Store the latest thunder gripper state (scalar)
#         self.latest_thunder_gripper = np.asarray(msg.data)

#     def get_last_n_observations(self):
#         """Returns the last 'n' observations structured as per the requirement."""
#         images = {
#             'img_front': np.array(self.img_front_history),
#             'img_wrist_thunder': np.array(self.img_wrist_thunder_history),
#             'img_wrist_lightning': np.array(self.img_wrist_lightning_history),
#         }

#         # Agent state is a concatenation of:
#         # - lightning angle (6,)
#         # - lightning gripper (1,)
#         # - thunder angle (6,)
#         # - thunder gripper (1,)
#         agent_state = np.column_stack((
#             np.array(self.lightning_angle_history),        # (obs_horizon, 6)
#             np.array(self.lightning_gripper_history),      # (obs_horizon, 1)
#             np.array(self.thunder_angle_history),         # (obs_horizon, 6)
#             np.array(self.thunder_gripper_history)        # (obs_horizon, 1)
#         ))

#         return {
#             'Images': images,  # (obs_horizon, h, w, c)
#             'agent_state': agent_state  # (obs_horizon, 14) - concatenation of the agent's states
#         }

def main():
    ## Load Config and set device
    config = load_config('config_.yaml')
    device = config['device']

    ## Load dataset Stats
    stats = np.load("dataset/transformed_data/uncork_v2_stats.npy", allow_pickle=True).item()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config['num_diffusion_iters'],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )
    ## Load the pretrained Models
    networks = load_network(config, device)

    ## Subscribe to the ROS topics
    observation_subscriber = ObservationSubscriber(obs_horizon=config['observation_horizon'])
    rate = rospy.Rate(10) # 10hz

    rospy.sleep(2)
    ## Initialize the robot
    URs = init_robot()

    ## Move robot to home position
    # URs.moveJ("Lightning", LIGHTNING_HOME, asynchronous=True)
    input("Press Enter to Move Robot to Home Position...")
    
    while not rospy.is_shutdown():
        obs_dict = get_observations(observation_subscriber, config, stats)
        action = run_inference(obs_dict, networks, noise_scheduler, stats, config, device)
        # print("Action: \n", action)

        # input("Press Enter to Mvoe Robot...")
        perform_action(URs, action)

        ## Break Condition 
        ## TODO: Implement the break condition

        rate.sleep()


if __name__ == '__main__':
    main()
