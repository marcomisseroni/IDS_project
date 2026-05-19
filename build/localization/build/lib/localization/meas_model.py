import numpy as np
from localization.agent_type import AgentType

#  __  __                                                    _   __  __           _      _ 
# |  \/  |                                                  | | |  \/  |         | |    | |
# | \  / | ___  __ _ ___ _   _ _ __ ___ _ __ ___   ___ _ __ | |_| \  / | ___   __| | ___| |
# | |\/| |/ _ \/ _` / __| | | | '__/ _ \ '_ ` _ \ / _ \ '_ \| __| |\/| |/ _ \ / _` |/ _ \ |
# | |  | |  __/ (_| \__ \ |_| | | |  __/ | | | | |  __/ | | | |_| |  | | (_) | (_| |  __/ |
# |_|  |_|\___|\__,_|___/\__,_|_|  \___|_| |_| |_|\___|_| |_|\__|_|  |_|\___/ \__,_|\___|_|
                                                                                          
"""
Measurement model is a class that computes the measurement model h and the jacobians:
Zab = h(xa, xb)
Ha = -dh/dxa
Hb = dh/dxb
By calling the proper method we don't need to worry about the difference between measuring a mobile
robot or a person, we can just call h, Ha, Hb using the required parameters.
In the nomenclature used *a* is the agent that takes the measurement and *b* is the agent that get measured.
"""                                                                              

class MeasurementModel:

    def __init__(
            self,
            R_rr: np.ndarray,
            R_rp: np.ndarray):
        
        self.R_rr = R_rr
        self.R_rp = R_rp

    def h_robot_robot(
            self,
            b_state: np.ndarray,
            a_state: np.ndarray,
            ) -> np.ndarray:
            
        delta_x = (b_state[0] - a_state[0]) * np.cos(a_state[2]) + (b_state[1] - a_state[1]) * np.sin(a_state[2])
        delta_y = (b_state[1] - a_state[1]) * np.cos(a_state[2]) + (a_state[0] - b_state[0]) * np.sin(a_state[2])
        delta_theta = b_state[2] - a_state[2]
        return np.array([delta_x, delta_y, delta_theta])
    
    def h_robot_person(
            self, 
            b_state: np.ndarray, 
            a_state: np.ndarray
            ) -> np.ndarray:
        
        delta_x = (b_state[0] - a_state[0]) * np.cos(a_state[2]) + (b_state[1] - a_state[1]) * np.sin(a_state[2])
        delta_y = (b_state[1] - a_state[1]) * np.cos(a_state[2]) + (a_state[0] - b_state[0]) * np.sin(a_state[2])
        return np.array([delta_x, delta_y])

    def h(
            self,
            b_type: AgentType,
            b_state: np.ndarray,
            a_state: np.ndarray
            ) -> np.ndarray:
        
        if b_type == AgentType.ROBOT:
            return self.h_robot_robot(b_state, a_state)
        elif b_type == AgentType.PERSON:
            return self.h_robot_person(b_state, a_state)
        else:
            raise ValueError(f"Unsupported agent_type: {b_type}")
        
    def Ha_robot_robot(
            self,
            b_state: np.ndarray,
            a_state: np.ndarray
            ) -> np.ndarray:
        
        Ha = np.zeros((3, 3))
        Ha[0, 0] = np.cos(a_state[2])
        Ha[0, 1] = np.sin(a_state[2])
        Ha[0, 2] = (a_state[1] - b_state[1]) * np.cos(a_state[2]) + (b_state[0] - a_state[0]) * np.sin(a_state[2])
        Ha[1, 0] = - np.sin(a_state[2])
        Ha[1, 1] = np.cos(a_state[2])
        Ha[1, 2] = (b_state[0] - a_state[0]) * np.cos(a_state[2]) + (b_state[1] - a_state[1]) * np.sin(a_state[2])
        Ha[2, 2] = 1
        return Ha
    
    def Ha_robot_person(
            self, 
            b_state: np.ndarray,
            a_state: np.ndarray
            ) -> np.ndarray:
        
        Ha = np.zeros((2, 3))
        Ha[0, 0] = np.cos(a_state[2])
        Ha[0, 1] = np.sin(a_state[2])
        Ha[0, 2] = ( - b_state[1] + a_state[1]) * np.cos(a_state[2]) + ( - a_state[0] + b_state[0]) * np.sin(a_state[2])
        Ha[1, 0] = - np.sin(a_state[2])
        Ha[1, 1] = np.cos(a_state[2])
        Ha[1, 2] = (b_state[0] - a_state[0]) * np.cos(a_state[2]) + (b_state[1] - a_state[1]) * np.sin(a_state[2])
        return Ha

    def Ha(
            self,
            b_type: AgentType,
            b_state: np.ndarray,
            a_state: np.ndarray
            ) -> np.ndarray:
        
        if b_type == AgentType.ROBOT:
            return self.Ha_robot_robot(b_state, a_state)
        elif b_type == AgentType.PERSON:
            return self.Ha_robot_person(b_state, a_state)
        else:
            raise ValueError(f"Unsupported agent_type: {b_type}")
        
    def Hb_robot_robot(
            self,
            b_state: np.ndarray,
            a_state: np.ndarray
            ) -> np.ndarray:
        
        Hb = np.zeros((3, 3))
        Hb[0, 0] = np.cos(a_state[2])
        Hb[0, 1] = np.sin(a_state[2])
        Hb[1, 0] = - np.sin(a_state[2])
        Hb[1, 1] = np.cos(a_state[2])
        Hb[2, 2] = 1
        return Hb
    
    def Hb_robot_person(
            self, 
            b_state: np.ndarray,
            a_state: np.ndarray
            ) -> np.ndarray:
        
        Hb = np.zeros((2, 4))
        Hb[0, 0] = np.cos(a_state[2])
        Hb[0, 1] = np.sin(a_state[2])
        Hb[1, 0] = - np.sin(a_state[2])
        Hb[1, 1] = np.cos(a_state[2])
        return Hb

    def Hb(
            self,
            b_type: AgentType,
            b_state: np.ndarray,
            a_state: np.ndarray
            ) -> np.ndarray:
        
        if b_type == AgentType.ROBOT:
            return self.Hb_robot_robot(b_state, a_state)
        elif b_type == AgentType.PERSON:
            return self.Hb_robot_person(b_state, a_state)
        else:
            raise ValueError(f"Unsupported agent_type: {b_type}")
        
    def R(
            self,
            b_type: AgentType
            ) -> np.ndarray:
        
        if(b_type == AgentType.ROBOT):
            return self.R_rr
        elif(b_type == AgentType.PERSON):
            return self.R_rp
        else:
            raise ValueError(f"Unsupported agent_type: {b_type}")