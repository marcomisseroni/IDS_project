import numpy as np
from enum import Enum
from abc import ABC, abstractmethod      

#                           _ _______               
#     /\                   | |__   __|              
#    /  \   __ _  ___ _ __ | |_ | |_   _ _ __   ___ 
#   / /\ \ / _` |/ _ \ '_ \| __|| | | | | '_ \ / _ \
#  / ____ \ (_| |  __/ | | | |_ | | |_| | |_) |  __/
# /_/    \_\__, |\___|_| |_|\__||_|\__, | .__/ \___|
#           __/ |                   __/ | |         
#          |___/                   |___/|_|         

"""
Enum class used to identify whether an agent is a ROBOT or a PERSON
"""

class AgentType(Enum):
    ROBOT = 0
    PERSON = 1

#  ____                                          _   __  __           _      _ 
# |  _ \                   /\                   | | |  \/  |         | |    | |
# | |_) | __ _ ___  ___   /  \   __ _  ___ _ __ | |_| \  / | ___   __| | ___| |
# |  _ < / _` / __|/ _ \ / /\ \ / _` |/ _ \ '_ \| __| |\/| |/ _ \ / _` |/ _ \ |
# | |_) | (_| \__ \  __// ____ \ (_| |  __/ | | | |_| |  | | (_) | (_| |  __/ |
# |____/ \__,_|___/\___/_/    \_\__, |\___|_| |_|\__|_|  |_|\___/ \__,_|\___|_|
#                                __/ |                                         
#                               |___/                                          

"""
BaseAgentModel is an abstract class used to derive the MobileRobotModel class and the PersonModel class. 
They contain model-related informations such as system model and jacobians:
x(k+1) = f(x(k), u(k=)) + g(x(k), eta) 
A = df/dx
G = dg/deta
"""

class BaseAgentModel(ABC):
    
    @abstractmethod
    def f(            
            self,
            state: np.ndarray,
            u: np.ndarray,
            dt: float
            ) -> np.ndarray:
        pass 

    @abstractmethod
    def A(            
            self,
            state: np.ndarray,
            u: np.ndarray,
            dt: float
            ) -> np.ndarray:
        pass 

    @abstractmethod
    def G(            
            self,
            state: np.ndarray,
            u: np.ndarray,
            dt: float
            ) -> np.ndarray:
        pass 

#  __  __       _     _ _      _____       _           _   __  __           _      _ 
# |  \/  |     | |   (_) |    |  __ \     | |         | | |  \/  |         | |    | |
# | \  / | ___ | |__  _| | ___| |__) |___ | |__   ___ | |_| \  / | ___   __| | ___| |
# | |\/| |/ _ \| '_ \| | |/ _ \  _  // _ \| '_ \ / _ \| __| |\/| |/ _ \ / _` |/ _ \ |
# | |  | | (_) | |_) | | |  __/ | \ \ (_) | |_) | (_) | |_| |  | | (_) | (_| |  __/ |
# |_|  |_|\___/|_.__/|_|_|\___|_|  \_\___/|_.__/ \___/ \__|_|  |_|\___/ \__,_|\___|_|
                                                                                    
"""
Class that inherits from BaseAgentModel and implements the model of a mobile robot
"""                                                 

class MobileRobotModel(BaseAgentModel):

    def f(
            self, 
            state: np.ndarray, 
            u: np.ndarray,
            dt: float
            ) -> np.ndarray:
                
            v, yaw_rate = u
            x_new = state[0] + dt * v * np.cos(state[2] + dt * yaw_rate / 2)
            y_new = state[1] + dt * v * np.sin(state[2] + dt * yaw_rate / 2)
            theta_new = state[2] + dt * yaw_rate

            return np.array([x_new, y_new, theta_new])
    
    def A(
            self,
            state: np.ndarray,
            u: np.ndarray,
            dt: float
            ) -> np.ndarray:
        
        v, yaw_rate = u
        A = np.identity(3)
        A[0, 2] = - dt * v * np.sin(state[2] + dt * yaw_rate / 2)
        A[1, 2] = dt * v * np.cos(state[2] + dt * yaw_rate / 2)

        return A
    
    def G(
            self,
            state: np.ndarray,
            u: np.ndarray,
            dt: float
            ) -> np.ndarray:
        
        v, yaw_rate = u
        G = np.zeros((3, 3))
        G[0, 0] = dt * np.cos(state[2] + dt * yaw_rate / 2)
        G[0, 1] = - dt ** 2 * v / 2 * np.sin(state[2] + dt * yaw_rate / 2)
        G[1, 0] = dt * np.sin(state[2] + dt * yaw_rate / 2)
        G[1, 1] = dt ** 2 * v / 2 * np.cos(state[2] + dt * yaw_rate / 2)
        G[2, 1] = dt

        return G
    
#  _____                          __  __           _      _ 
# |  __ \                        |  \/  |         | |    | |
# | |__) |__ _ __ ___  ___  _ __ | \  / | ___   __| | ___| |
# |  ___/ _ \ '__/ __|/ _ \| '_ \| |\/| |/ _ \ / _` |/ _ \ |
# | |  |  __/ |  \__ \ (_) | | | | |  | | (_) | (_| |  __/ |
# |_|   \___|_|  |___/\___/|_| |_|_|  |_|\___/ \__,_|\___|_|
                                                           
"""
Class that inherits from BaseAgentModel and implements the model of a person considering a constant velocity scenario
"""                         

class PersonModel(BaseAgentModel):
     
    def A(
            self,
            state: np.ndarray,
            u: np.ndarray,
            dt: float
            ) -> np.ndarray:
        
        A = np.identity(4)
        A[0, 2] = dt
        A[1, 3] = dt

    

        return A
    
    def f(  
            self, 
            state: np.ndarray, 
            u: np.ndarray,
            dt: float
            ) -> np.ndarray:

            state_new = self.A(state, u, dt) @ state            

            return state_new
    
    def G(
            self,
            state: np.ndarray,
            u: np.ndarray,
            dt: float
            ) -> np.ndarray:
        
        G = np.identity(4)

        return G