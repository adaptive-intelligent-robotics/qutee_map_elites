# only required to run python3 examples/cvt_rastrigin.py
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './map_elites')))

import multiprocessing
from multiprocessing import current_process

import numpy as np
import math

import scipy

from . import cvt as cvt_map_elites
from . import common as cm_map_elites

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from qutee_msg.msg import RolloutRes, Weights
import rclpy
from rclpy.node import Node


class QuteeClientAsync(Node):

    def __init__(self, robot_name):
        super().__init__(robot_name + '_client_async')
        # Setting up the Rollout Client
        self.robot_name = robot_name
        self.publisher_ = self.create_publisher(Weights, f'/{self.robot_name}/qutee_weights_to_evaluate', 10)
        self.subscription = self.create_subscription(
            RolloutRes,
            f'/{self.robot_name}/qutee_rollout_results',
            self.rollout_callback,
            10)
        
        
        self.rollout_req = Weights()
        self.layout_dim = MultiArrayDimension()
        self.rollout_req.weights.layout.dim.append(self.layout_dim)
        self.rollout_req.weights.layout.dim[0].label = "weights"
        self.rollout_req.weights.layout.data_offset = 0

        
    def perform_rollout(self, weights):
        print("Setting up messages")
        self.rollout_req.weights.layout.dim[0].size = len(weights)
        self.rollout_req.weights.layout.dim[0].stride = len(weights)
        self.rollout_req.weights.data = weights.data
        
        print(f"Calling {self.robot_name}")
        self.publisher_.publish(self.rollout_req)
        print(f"Waiting for {self.robot_name}'s response")
        rclpy.spin_once(self)
        print(f"Got it from {self.robot_name}")
        return self.last_rollout_results

    def rollout_callback(self, msg):
        self.last_rollout_results = msg



def _multiarray_to_numpy(multiarray):
    dims = tuple(map(lambda x: x.size, multiarray.layout.dim))
    return np.array(multiarray.data, dtype=float).reshape(dims).astype(np.float32)

def __list_process_names(anything):
    time.sleep(1)
    print(current_process().name)

def qutee_rollout(weights):
    print("Start rollout")
    outputs = qutee_client.perform_rollout(weights)
    states = _multiarray_to_numpy(outputs.states)
    actions = _multiarray_to_numpy(outputs.actions)
    received_weights = _multiarray_to_numpy(outputs.weights)

    fitness = - np.sum(np.abs(actions[1:,:] -actions[:-1,:] ))
    
    print(f'fitness = {fitness}')
    accelerations = states[:,3:5]
    velocities = scipy.integrate.cumulative_trapezoid(y=accelerations,dx=1/50.0,axis=0)
    positions = scipy.integrate.cumulative_trapezoid(y=velocities,dx=1/50.0,axis=0)

    features = positions[-1,:]/10+0.5
    print(f'features = {features}')
    return fitness, features


def __assign_robots_to_processes(robot_name):
    time.sleep(1)
    current_process().name=robot_name
    rclpy.init()
    global qutee_client 

    qutee_client = QuteeClientAsync(robot_name)
   
    

def main():

    robot_names = {"qutee_42", "qutee_007", "qutee_airl"}
    # setup the parallel processing pool
    pool = multiprocessing.Pool(len(robot_names))
    print("initial names")
    pool.map(__list_process_names, range(11))
    print("setting new names")
    pool.map(__assign_robots_to_processes, robot_names)
    print("new names names")
    pool.map(__list_process_names, range(11))
    #else: #I don't know yet what to do if we have just one robot, probably that should just work with parallel. 
    #    s_list = map(evaluate_function, to_evaluate)



    # we do 10M evaluations, which takes a while in Python (but it is very fast in the C++ version...)
    px = cm_map_elites.default_params.copy()
    px["dump_period"] = 10
    px["batch_size"] = 5
    px["min"] = -100
    px["max"] = 100
    px["parallel"] = True
    
  



    
    

    gen_dim = 322 ## TO CHANGE IF YOU CHANGE THE NN CONFIGURATION ON THE QUTEEs
    bd_dim = 2
    archive = cvt_map_elites.compute(bd_dim, gen_dim, qutee_rollout, pool = pool, n_niches=400, max_evals=100, log_file=open('cvt.dat', 'w'), params=px)

    qutee_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
