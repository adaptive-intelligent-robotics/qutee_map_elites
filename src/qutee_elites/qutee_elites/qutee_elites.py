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
#from qutee_msg.msg import RolloutRes, Weights
from qutee_interface.srv import Status, Rollout
import rclpy
from rclpy.node import Node


class QuteeClientAsync(Node):

    def __init__(self, robot_names):
        super().__init__('Qutee_manager')
        self.robot_names = robot_names
        self.robots={}
        for name in self.robot_names:
            print(f'Creating client for {name}')
            self.robots[name]={"status":  self.create_client(Status,  f'/{name}/status'),
                               "rollout": self.create_client(Rollout, f'/{name}/rollout')}
        print(f'clients created for {len(self.robots)} robots')
        self.expected_results={} #always use robot names as key


    def collect_results(self, timeout_sec = 2): #TODO careful: this can wait indefinitely if some service never reply
        results = {}
        count = 0
        while len(self.expected_results):# and count<timeout_sec*10: # while there are open requests
            count = count + 1
            completed = []
            
            for name in self.expected_results: # we iterate over the robots with open requests (the remaining keys in the dict)
                if self.expected_results[name][0].done(): # if their request is now done
                    try:
                        results[name]=self.expected_results[name][0].result() # we try to get the result and remove the request from the dict
                        print(f'got results from {name}')
                        completed.append(name)
                    except Exception as e: # if it raised an exception during execution
                        completed.append(name) # still remove result from the dict
                        print("An error occurred:", e)
                        
            for name in completed:                
                del self.expected_results[name] # remove all the completed task during this loop
                
            print(f'{list(self.expected_results.keys())} task(s) are still ongoing') #otherwise we print and wait
            rclpy.spin_once(self,timeout_sec=0.1) # spin ROS once to collect new results
        
        return results

    def get_all_status(self):
        print(f'I now have potentially {len(self.robots)} robots')
        for name in self.robots:
            print(f'requesting status for {name}')
            if self.robots[name]["status"].wait_for_service(1):
                self.expected_results[name] = [self.robots[name]["status"].call_async(Status.Request()), None, None]
            else:
                print(f'status service of {name} not ready')
        return self.collect_results()
    
    def take_attendance(self):
        print(f'start taking attendance')
        status = self.get_all_status()
        attendance=[]
        for name in status:
            #TODO add battery and error checks
            attendance.append(name)
        return attendance

    def request_rollout(self, weights, name, index):
        eval_func = weights[1] # Don't know why
        weights = weights[0]
        print("Setting up messages")
        req = Rollout.Request()
        print(req)
        layout_dim = MultiArrayDimension()
        req.weights.layout.dim.append(layout_dim)
        req.weights.layout.dim[0].label = "weights"
        req.weights.layout.data_offset = 0
        req.weights.layout.dim[0].size = len(weights)
        req.weights.layout.dim[0].stride = len(weights)
        print(weights)
        req.weights.data = set(weights)
        print(req)
        print("calling service")
        if self.robots[name]["rollout"].wait_for_service(1):
            self.expected_results[name] = [self.robots[name]["rollout"].call_async(req), index, req]
            
        return
        

    def map(self, eval_function, to_evaluate): # to be extended soon to provide assigments to specific robots
        # at the moment the eval_function is ignored. Need to change this later
        print(to_evaluate)
        indexes = list(range(len(to_evaluate)))
        full_results = [None] * len(to_evaluate)
        print("start eval loop")
        while(len(to_evaluate)):
            available_robots = self.take_attendance()
            print(f'avail robots: {available_robots}')
            for name in available_robots:
                if(len(to_evaluate)):
                    print(f'request rollout to {name}')
                    self.request_rollout(to_evaluate.pop(0), name,indexes.pop(0))
            print("awaiting results")
            
            full_results.append(self.collect_results(10)) ## we need to process the transitions with the eval_function
        print(full_results)
        return full_results

    
   
    
#    def perform_rollout(self, weights):
#        print("Setting up messages")
#        self.rollout_req.weights.layout.dim[0].size = len(weights)
#        self.rollout_req.weights.layout.dim[0].stride = len(weights)
#        self.rollout_req.weights.data = weights.data
#        
#        print(f"Calling {self.robot_name}")
#        self.publisher_.publish(self.rollout_req)
#        print(f"Waiting for {self.robot_name}'s response")
#        rclpy.spin_once(self)
#        print(f"Got it from {self.robot_name}")
#        return self.last_rollout_results
#
#    def rollout_callback(self, msg):
#        self.last_rollout_results = msg



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


   
    

def main(args=None):
    rclpy.init(args=args)
    
    robot_names = {"qutee_42", "qutee_007", "qutee_airl"}
    qutee_manager = QuteeClientAsync(robot_names)
    print(qutee_manager.take_attendance())
    

    
    # we do 10M evaluations, which takes a while in Python (but it is very fast in the C++ version...)
    px = cm_map_elites.default_params.copy()
    px["dump_period"] = 10
    px["batch_size"] = 2
    px["random_init_batch"] = 2
    px["min"] = -100
    px["max"] = 100
    px["parallel"] = True
    
    gen_dim = 322 ## TO CHANGE IF YOU CHANGE THE NN CONFIGURATION ON THE QUTEEs # This can now be acquired from the Status message
    bd_dim = 2
    archive = cvt_map_elites.compute(bd_dim, gen_dim, qutee_rollout, pool = qutee_manager, n_niches=400, max_evals=100, log_file=open('cvt.dat', 'w'), params=px)
    qutee_manager.destroy_node()
    rclpy.shutdown()
    return

    #qutee_client.destroy_node()
    #rclpy.shutdown()

if __name__ == '__main__':
    main()
