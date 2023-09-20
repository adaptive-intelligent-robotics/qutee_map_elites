# only required to run python3 examples/cvt_rastrigin.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './map_elites')))

import numpy as np
import math

from . import cvt as cvt_map_elites
from . import common as cm_map_elites

from qutee_msg.srv import Rollout, GetNumParams
import rclpy
from rclpy.node import Node


class QuteeClientAsync(Node):

    def __init__(self):
        super().__init__('qutee_client_async')
        # Setting up the Rollout Client
        self.rollout_client = self.create_client(Rollout, 'rollout')
        while not self.rollout_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Rollout service not available, waiting again...')
        self.rollout_req = Rollout.Request()
        # Setting up the GetNumParams Client
        self.numparams_client = self.create_client(GetNumParams, 'getnumparams')
        while not self.numparams_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('GetNumParams service not available, waiting again...')
        self.numparams_req = GetNumParams.Request()
        
    def get_num_params(self):
        self.future = self.numparams_client.call_async(self.numparams_req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def perform_rollout(self, weights):
        self.rollout_req.layout.dim.append(MultiArrayDimension())
        self.rollout_req.layout.dim[0].label = "width"
        self.rollout_req.layout.dim[0].size = len(weights)
        self.rollout_req.layout.dim[0].stride = len(weights)
        self.rollout_req.layout.data_offset = 0
        self.rollout_req.data = weights
        return self.future.result()


def main():
    
    # we do 10M evaluations, which takes a while in Python (but it is very fast in the C++ version...)
    px = cm_map_elites.default_params.copy()
    px["dump_period"] = 10
    px["batch_size"] = 5
    px["min"] = 0
    px["max"] = 1
    px["parallel"] = False
    
    gen_dim = 42
    bd_dim = 2


    rclpy.init()

    qutee_client = QuteeClientAsync()
    gen_dim = qutee_client.get_num_params().num_params
    qutee_client.get_logger().info(
        'Result of get_num_params: %d' %
        (gen_dim))

    qutee_minimal_client.destroy_node()
    rclpy.shutdown()

    
    def qutee_rollout(weights):
        outputs = qutee_client.perform_rollout(weights)
        x = xx * 10 - 5 # scaling to [-5, 5]
        f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * math.pi * x)).sum()
        return -f, np.array([xx[0], xx[1]])
    
    archive = cvt_map_elites.compute(bd_dim, gen_dim, qutee_rollout, n_niches=400, max_evals=100, log_file=open('cvt.dat', 'w'), params=px)

if __name__ == '__main__':
    main()
