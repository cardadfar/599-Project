import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


numAgents = 2
maxExposure = 2
eps_f = 0.001

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(numAgents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.sight = 0.2
            agent.size = 0.05
            agent.max_speed = 0.3


            
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
            agent.infected = False

        world.agents[0].infected = True
        world.agents[0].color = np.array([0.75,0.0,0.0])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.dest = np.random.uniform(-1,+1, world.dim_p)

    def L2(self, pos1, pos2):
        return np.sum(np.square(pos1 - pos2))

    def reward(self, agent, world):

        # if off-bounds, modulate to other side
        
        '''
        if agent.state.p_pos[0] < -1:
            agent.state.p_pos[0] = 1
        if agent.state.p_pos[0] > 1:
            agent.state.p_pos[0] = -1
        if agent.state.p_pos[1] < -1:
            agent.state.p_pos[1] = 1
        if agent.state.p_pos[1] > 1:
            agent.state.p_pos[1] = -1
        '''
        
        #if agent.infected:
        #    return -99999

        reward = - self.L2(agent.state.p_pos, agent.dest)
        k = 5
        
        inRange = False
        for i, a in enumerate(world.agents):
            dist     = self.L2(agent.state.p_pos, a.state.p_pos)
            max_dist = agent.sight * agent.sight
            if dist < max_dist:
                reward -= k * (max_dist - dist)
                '''
                inRange = True
                if a.infected:
                    agent.exposure += 1
        
        if np.random.rand() < (agent.exposure / maxExposure):
            agent.infected = True
            #agent.color = np.array([0.75,0.0,0.0])

        if not inRange:
            agent.exposure = 0
        '''
        
        return reward

    def mask(self, vec2, m):
        return np.append(vec2, m)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.agents:
            dist = self.L2(agent.state.p_pos, entity.state.p_pos)
            if dist < agent.sight:
                entity_pos.append(self.mask(agent.state.p_pos - entity.state.p_pos, 1))
            else:
                entity_pos.append([0,0,0])

        self_pos = [agent.state.p_pos]
        self_vel = [agent.state.p_vel]

        self_des = [agent.state.p_pos - agent.dest]

        return np.concatenate(self_des + self_pos + self_vel + entity_pos)
