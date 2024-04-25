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


class QuteeManager(Node):

    def __init__(self, robot_names):
        super().__init__('Qutee_manager')
        self.robot_names = robot_names
        self.robots={}
        for name in self.robot_names:
            print(f'Creating client for {name}')
            self.robots[name]={"status":  self.create_client(Status,  f'/{name}/status'),
                               "rollout": self.create_client(Rollout, f'/{name}/rollout')}
        print(f'clients created for {len(self.robots)} robots')
        self.ongoing_requests={} #always use robot names as key


    def send_request(self, name, request_name, request, additional_info):            
        if self.robots[name][request_name].wait_for_service(1):
            self.ongoing_requests[name] = {"future": self.robots[name][request_name ].call_async(request), "request": request,  "info": additional_info}
        else:
            print(f'{request_name} service of {name} not ready')


        
    def collect_results(self, timeout_sec = 2): 
        results = {}
        count = 0
        while len(self.ongoing_requests) and count<timeout_sec*10: # while there are open requests
            count = count + 1
            completed = []
            
            for name in self.ongoing_requests: # we iterate over the robots with open requests (the remaining keys in the dict)
                if self.ongoing_requests[name]["future"].done(): # if their request is now done
                    try:
                        results[name]={"result":self.ongoing_requests[name]["future"].result(), "request":self.ongoing_requests[name]["request"], "info":self.ongoing_requests[name]["info"]} # we try to get the result and remove the request from the dict
                        print(f'got results from {name}')
                        completed.append(name)
                    except Exception as e: # if it raised an exception during execution
                        completed.append(name) # still remove result from the dict
                        print("An error occurred:", e)
                        
            for name in completed:                
                del self.ongoing_requests[name] # remove all the completed task during this loop

            if count % 20 == 0:
                print(f'{list(self.ongoing_requests.keys())} task(s) are still ongoing', end="\r") #otherwise we print and wait
            rclpy.spin_once(self,timeout_sec=0.1) # spin ROS once to collect new results
            
        for name in self.ongoing_requests: # removing any task that did not complete before timeout.
            del self.ongoing_requests[name]
            results[name]={"result":None,"request": self.ongoing_requests[name]["request"], "info": self.ongoing_requwests[name]["info"]}
        return results

    def get_all_status(self):
        print(f'I now have potentially {len(self.robots)} robots')
        for name in self.robots:
            print(f'requesting status for {name}')
            self.send_request(name, "status", Status.Request(), None) 
        return self.collect_results()

                
            
    
    def take_attendance(self):
        print(f'start taking attendance')
        status = self.get_all_status()
        attendance={}
        for name in status:
            #TODO add battery and error checks
            if status[name]["result"]: # is None if robot did not reply
                attendance[name]=status[name]["result"].battery
        return attendance

    def request_rollout(self, inputs, name, index):
        # unpack inputs
        scoring_func = inputs[1] 
        weights = inputs[0]

        # build request message
        req = Rollout.Request()
        
        layout_dim = MultiArrayDimension()
        req.weights.layout.dim.append(layout_dim)
        req.weights.layout.dim[0].label = "weights"
        req.weights.layout.data_offset = 0
        req.weights.layout.dim[0].size = len(weights)
        req.weights.layout.dim[0].stride = len(weights)

        req.weights.data = set(weights)

        # send request
        self.send_request(name, "rollout", req, {"index": index, "scoring_func": scoring_func, "weights": weights}) 
        # Note: here the weights are basically added twice (once in the request message and once in the info. This is done for code clarity, can be remove to reduce RAM usage)
            
        return


    # evaluate a single vector (x) with a function f and return a species
    # t = vector, function
    def __evaluate(t):
        z, f = t  # evaluate z with function f
        fit, desc = f(z)
        return cm.Species(z, desc, fit)


    def map(self, to_evaluate): # to be extended soon to provide assigments to specific robots
        # at the moment the eval_function is ignored. Need to change this later
        indexes = list(range(len(to_evaluate)))
        full_results = [None] * len(to_evaluate)
        
        print("start eval loop")
        while(len(to_evaluate)):
            # Distribute one eval request to each available robot
            available_robots = self.take_attendance()
            print(f'avail robots: {available_robots}')
            for name in available_robots:
                if(len(to_evaluate)):
                    print(f'request rollout to {name}')
                    self.request_rollout(to_evaluate.pop(0), name,indexes.pop(0))

            # Wait for the results
            received_results = self.collect_results(10) #wait up to 10 sec to collect the results

            # Process each result
            for name in received_results:
                res = received_results[name]
                if res["results"]:
                    fit, desc = res["info"]["scoring_func"](res["results"])
                    full_results[res["info"]["index"]] = cm.Species(res["info"]["weights"], desc, fit)
                    # we sort the results via their incoing index and we ask the "evaluat_function" to score the transition and packages everything into a species object
                else:
                    full_results[res["info"]["index"]] = cm.Species(None, None, None) 
                
        print(full_results)
        return full_results

    


def _multiarray_to_numpy(multiarray):
    dims = tuple(map(lambda x: x.size, multiarray.layout.dim))
    return np.array(multiarray.data, dtype=float).reshape(dims).astype(np.float32)

def __list_process_names(anything):
    time.sleep(1)
    print(current_process().name)




def scoring_function(transitions):
    print("Start rollout")
    outputs = transitions
    states = _multiarray_to_numpy(outputs.states)
    actions = _multiarray_to_numpy(outputs.actions)
    #received_weights = _multiarray_to_numpy(outputs.weights)// not provided yet
    print("actions")
    print(actions)
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
    
    robot_names = {"qutee_blue", "qutee_007", "qutee_orange"}
    qutee_manager = QuteeManager(robot_names)
    print(qutee_manager.take_attendance())
    

    
    # we do 10M evaluations, which takes a while in Python (but it is very fast in the C++ version...)
    px = cm_map_elites.default_params.copy()
    px["dump_period"] = 10
    px["batch_size"] = 2
    px["random_init_batch"] = 2
    px["init_range"] = 0.5
    px["min"] = -100
    px["max"] = 100
    px["parallel"] = True
    
    gen_dim = 322 ## TO CHANGE IF YOU CHANGE THE NN CONFIGURATION ON THE QUTEEs # This can now be acquired from the Status message
    bd_dim = 2
    archive = cvt_map_elites.compute(bd_dim, gen_dim, scoring_function, pool = qutee_manager, n_niches=400, max_evals=100, log_file=open('cvt.dat', 'w'), params=px)


    qutee_manager.destroy_node()
    rclpy.shutdown()
    return


if __name__ == '__main__':
    main()
