import math
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class MceEnv(gym.Env):
    def __init__(self):
        self.action_space = [spaces.Discrete(3), spaces.Discrete(3)]
        self.observation_space = [spaces.Box(low=0.0, high=100.0, shape=(11,) ,dtype=np.float32),
                                  spaces.Box(low=0.0, high=100.0, shape=(11,) ,dtype=np.float32)]
        self.obn_space = 1
        self.obe_space = 2
        self.nedges = 6
        self.nnodes = 6
        self._action_set = [(1,1), (2,1), (3,1),
                            (1,2), (2,2), (3,2),
                            (1,3), (2,3), (3,3)]
        self.minigap = 2.5
        self.carlength = 5.0
        self.deltavel = 10
        self.n = 2
        maxacc = 3.0
        desirebrakeacc = 2.0
        self.idmsqrtacc = 2*np.sqrt(maxacc*desirebrakeacc)
        self.seed()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    """
    def convert(self, obs):
        # convert obs in ob_space.shape to graph shape
        ob_ns, ob_nr, ob_e, ob_n = [], [], [], []
        e2ns, e2nr, ns2e, b2e, b2n = [], [], [], [], []
        for i in range(obs.shape[0]):
            #loop over batch
            e2ns += [len(ob_n), len(ob_n), len(ob_n),
                      len(ob_n)+1, len(ob_n)+1, len(ob_n)+1]
            e2nr += [len(ob_n)+2, len(ob_n)+3, len(ob_n)+1,
                      len(ob_n)+3, len(ob_n)+4, len(ob_n)+5]
            ns2e += [[len(ob_e), len(ob_e)+1, len(ob_e)+2],
                     [len(ob_e)+3, len(ob_e)+4, len(ob_e)+5],[],[],[],[]]
            b2e += [range(len(ob_e), len(ob_e)+6)]
            b2n += [range(len(ob_n), len(ob_n)+6)]
            ob_n += [obs[i,0], obs[i,2], obs[i,4], obs[i,6], obs[i,8], obs[i,10]]
            #d = (ob[i,5]*100+50+ob[i,1]*50+25)/100
            d = ob[i,5]+0.5+ob[i,1]*0.5+0.25
            ob_e += [[obs[i,3]+0.5,0.0], [obs[i,5]+0.5,1.0], [-obs[i,1]-0.5,1.0],
                     [d,0.0], [obs[i,7]+0.5,1.0], [-obs[i,9]-0.5,1.0]]
            ob_ns += [obs[i,0], obs[i,0], obs[i,0], obs[i,2], obs[i,2], obs[i,2]]
            ob_nr += [obs[i,4], obs[i,6], obs[i,2], obs[i,6], obs[i,8], obs[i,10]]
        ob_ns, ob_nr, ob_e, ob_n = np.array(ob_ns), np.array(ob_nr), np.array(ob_e), np.array(ob_n)
        e2ns, e2nr, ns2e, b2e, b2n = _multionehot(e2ns, ob_n.shape[0]), _multionehot(e2nr, ob_n.shape[0]),
                                     _multionehot(ns2e, ob_e.shape[0]), _multionehot(b2e, ob_e.shape[0]),
                                     _multionehot(b2n, ob_n.shape[0])
        return (ob_ns, ob_nr, ob_e, ob_n, e2ns, e2nr, ns2e, b2e, b2n)
    """
    def step(self, a):
        if self.done:
            sp = np.zeros(11)
            return [sp, sp], [0.0, 0.0], [self.done, self.done], [{}, {}]
        try:
            action = self._action_set[a]
        except:
            action = (a[0]+1, a[1]+1)
        #print("action: ", action, " a ", a)
        if action[0] == 1:
            sp = np.zeros(11)
            self.done = True
            if action[1] == 1:
                if (self.maintainleftleaderpos-self.maintainerpos-self.carlength-self.minigap)<=0.0 or (self.maintainerpos-self.maintainleftfollowerpos-self.carlength-self.minigap)<=0.0:
                    # maintainer crash
                    R2 = -1
                    if (self.maintainleaderpos-self.carlength-self.minigap)<=0.0 or (-self.maintainerpos-self.carlength-self.minigap) <=0.0:
                        R1 = -1
                    else:
                        # simulate merger merging
                        mergerpos = 0.0
                        iscollided = self.simulate2([mergerpos,self.mergervel],
                         [self.mergeleaderpos, self.mergeleadervel],
                         [self.maintainleaderpos, self.maintainleadervel],
                         [self.maintainerpos, self.maintainervel])
                        if iscollided:
                            R1 = -1
                        else:
                            R1 = 1
                else:
                    if (self.maintainleaderpos-self.carlength-self.minigap)<=0.0:
                        # merger crash maintainer no reward
                        R1 = -1
                        iscollided = self.simulate2([self.maintainerpos,self.maintainervel],
                         [self.maintainleaderpos, self.maintainleadervel],
                         [self.maintainleftleaderpos, self.maintainleftleadervel],
                         [self.maintainleftfollowerpos, self.maintainleftfollowervel])
                        if iscollided:
                            R2 = -1
                        else:
                            R2 = 0

                    else:
                        # simulate maintainer changing & merger merging
                        iscollided = self.simulate2([self.maintainerpos,self.maintainervel],
                         [self.maintainleaderpos, self.maintainleadervel],
                         [self.maintainleftleaderpos, self.maintainleftleadervel],
                         [self.maintainleftfollowerpos, self.maintainleftfollowervel])
                        if iscollided:
                            if (-self.maintainerpos-self.carlength-self.minigap) <= 0.0:
                                R1 = -1
                            else:
                                R1 = 1
                            R2 = -1
                        else:
                            R1 = 1
                            R2 = 0.5

            elif action[1] == 2:
                # maintainer lane keeping
                # simulate maintainer & merger merging
                mergerpos = 0.0
                iscollided = self.simulate2([mergerpos,self.mergervel],
                         [self.mergeleaderpos, self.mergeleadervel],
                         [self.maintainleaderpos, self.maintainleadervel],
                         [self.maintainerpos, self.maintainervel])
                if -self.maintainerpos-self.carlength-self.minigap <=0.0:
                    if iscollided:
                        R1 = R2 = -1
                    else:
                        R1 = R2 = 1
                else:
                    if iscollided:
                        R1 = -1
                        R2 = 0
                    else:
                        R1 = R2 = 1
            else:
                # maintainer openning gap
                # simulate maintainer & merger merging
                if (self.maintainleaderpos-self.carlength-self.minigap)<=0.0:
                    R1 = 0.5
                    R2 = 0.1
                else:
                    R1 = 1
                    R2 = 0
        elif action[0] == 2:
            if action[1] == 1:
                # Maintainer change lane
                iscollided = self.simulate2([self.maintainerpos,self.maintainervel],
                         [self.maintainleaderpos, self.maintainleadervel],
                         [self.maintainleftleaderpos, self.maintainleftleadervel],
                         [self.maintainleftfollowerpos, self.maintainleftfollowervel])
                if iscollided:
                    R2 = -1
                    R1 = 0.1
                else:
                    R2 = 0
                    R1 = 0
                self.done = True
                sp = np.zeros(11)

            elif action[1] == 2:
                # maintainer lane keeping
                mergerpos = 0.0
                sp, self.done = self.simulate(
                         [mergerpos, self.mergervel, 2],
                         [self.maintainerpos,self.maintainervel, 2],
                         [self.mergeleaderpos, self.mergeleadervel],
                         [self.maintainleaderpos, self.maintainleadervel],
                         [self.maintainleftleaderpos, self.maintainleftleadervel],
                         [self.maintainleftfollowerpos, self.maintainleftfollowervel])
                if self.done:
                    R1 = 0.1
                    R2 = 0.1
                else:
                    R1 = 0
                    R2 = 0

            else:
                # maintainer openning gap
                mergerpos = 0.0
                sp, self.done = self.simulate(
                         [mergerpos, self.mergervel, 2],
                         [self.maintainerpos,self.maintainervel, 3],
                         [self.mergeleaderpos, self.mergeleadervel],
                         [self.maintainleaderpos, self.maintainleadervel],
                         [self.maintainleftleaderpos, self.maintainleftleadervel],
                         [self.maintainleftfollowerpos, self.maintainleftfollowervel])
                if self.done:
                    R1 = 0.1
                    R2 = 0
                else:
                    R1 = 0
                    R2 = 0.1


        else:
            if action[1] == 1:
                # Maintainer change lane
                iscollided = self.simulate2([self.maintainerpos,self.maintainervel],
                         [self.maintainleaderpos, self.maintainleadervel],
                         [self.maintainleftleaderpos, self.maintainleftleadervel],
                         [self.maintainleftfollowerpos, self.maintainleftfollowervel])
                if iscollided:
                    R1 = 0.1
                    R2 = -1
                else:
                    R1 = R2 = 0
                self.done = True
                sp = np.zeros(11)

            elif action[1] == 2:
                # maintainer lane keeping
                mergerpos = 0.0
                sp, self.done = self.simulate(
                         [mergerpos, self.mergervel, 3],
                         [self.maintainerpos,self.maintainervel, 2],
                         [self.mergeleaderpos, self.mergeleadervel],
                         [self.maintainleaderpos, self.maintainleadervel],
                         [self.maintainleftleaderpos, self.maintainleftleadervel],
                         [self.maintainleftfollowerpos, self.maintainleftfollowervel])
                if self.done:
                    R1 = 0.1
                    R2 = 0.1
                else:
                    R1 = 0
                    R2 = 0.1

            else:
                # maintainer openning gap
                mergerpos = 0.0
                sp, self.done = self.simulate(
                         [mergerpos, self.mergervel, 3],
                         [self.maintainerpos,self.maintainervel, 3],
                         [self.mergeleaderpos, self.mergeleadervel],
                         [self.maintainleaderpos, self.maintainleadervel],
                         [self.maintainleftleaderpos, self.maintainleftleadervel],
                         [self.maintainleftfollowerpos, self.maintainleftfollowervel])
                if self.done:
                    R1 = R2 = 0
                else:
                    R1 = R2 = -0.1
        #print("a ", a, " ", sp)
        if not self.done:
            #print("here", sp)
            sp[0] = (sp[0]-15)/30
            sp[1] = (sp[1]-25)/50
            sp[2] = (sp[2]-15)/30
            sp[3] = (sp[3]-50)/100
            sp[4] = (sp[4]-15)/30
            sp[5] = (sp[5]-50)/100
            sp[6] = (sp[6]-15)/30
            sp[7] = (sp[7]-50)/100
            sp[8] = (sp[8]-15)/30
            sp[9] = (sp[9]-25)/50
            sp[10] = (sp[10]-15)/30

        return [sp, sp], [R1, R2], [self.done, self.done], [{}, {}]
    def reset(self):
        self.done = False
        disc_state = self.np_random.randint(1, 4, size=11)
        disc_mergervel, disc_maintainerpos, disc_maintainervel,disc_mergeleaderpos, disc_mergeleadervel, disc_maintainleaderpos, disc_maintainleadervel, disc_maintainleftleaderpos, disc_maintainleftleadervel, disc_maintainleftfollowerpos, disc_maintainleftfollowervel = disc_state
        # sample the position and velocity
        self.mergervel = mergervel = (disc_mergervel-1)*self.deltavel + self.np_random.random_sample()*self.deltavel

        # 1: approaching 2: maintain 3: departing
        if disc_maintainervel == 1:
            maxvel = np.minimum(mergervel + 10.0, 30.0)
            minvel = np.minimum(mergervel + 2.0, 30.0)
        elif disc_maintainervel == 2:
            maxvel = np.minimum(mergervel + 2.0, 30.0)
            minvel = np.maximum(mergervel - 2.0, 0.0)
        else:
            maxvel = np.maximum(mergervel - 2.0, 0.0)
            minvel = np.maximum(mergervel - 10.0, 0.0)

        self.maintainervel = maintainervel = self.np_random.random_sample()*(maxvel-minvel) + minvel
        # 1 block behind # 2 not block behind
        if disc_maintainerpos == 1:
            minpos = np.maximum(-50, -self.carlength-self.minigap-np.maximum(0.0, (maintainervel**2-mergervel**2)/16+maintainervel ) )
            maxpos = 0.0
        else:
            maxpos = np.maximum(-50.0,-self.carlength-self.minigap-np.maximum(0.0, (maintainervel**2-mergervel**2)/16+maintainervel ) )
            minpos = -50.0

        self.maintainerpos = maintainerpos = self.np_random.random_sample()*(maxpos-minpos) + minpos

        # 1: approaching 2: maintain 3: departing
        if disc_mergeleadervel == 1:
            maxvel = np.maximum(mergervel - 2.0, 0.0)
            minvel = np.maximum(mergervel - 10.0, 0.0)
        elif disc_mergeleadervel == 2:
            maxvel = np.minimum(mergervel + 2.0, 30.0)
            minvel = np.maximum(mergervel - 2.0, 0.0)
        else:
            maxvel = np.minimum(mergervel + 10.0, 30.0)
            minvel = np.minimum(mergervel + 2.0, 30.0)

        self.mergeleadervel = mergeleadervel = self.np_random.random_sample()*(maxvel-minvel) + minvel
        # 1 block in front # 2 not block in front
        if disc_mergeleaderpos == 1:
            maxpos = np.minimum(100.0, self.carlength+self.minigap+np.maximum(0.0, (mergervel**2-mergeleadervel**2)/16+mergervel ) )
            minpos = self.carlength+self.minigap
        else:
            minpos = np.minimum(100.0, self.carlength+self.minigap+np.maximum(0.0, (mergervel**2-mergeleadervel**2)/16+mergervel ))
            maxpos = 100.0

        self.mergeleaderpos = mergeleaderpos = self.np_random.random_sample()*(maxpos-minpos) + minpos

        # 1: approaching 2: maintain 3: departing
        if disc_maintainleadervel == 1:
            maxvel = np.maximum(mergervel - 2.0, 0.0)
            minvel = np.maximum(mergervel - 10.0, 0.0)
        elif disc_maintainleadervel == 2:
            maxvel = np.minimum(mergervel + 2.0, 30.0)
            minvel = np.maximum(mergervel - 2.0, 0.0)
        else:
            maxvel = np.minimum(mergervel + 10.0, 30.0)
            minvel = np.minimum(mergervel + 2.0, 30.0)

        self.maintainleadervel = maintainleadervel = self.np_random.random_sample()*(maxvel-minvel) + minvel
        # 1 block in front # 2 not block in front
        if disc_maintainleaderpos == 1:
            minpos = np.maximum(0.0, maintainerpos+self.carlength+2*self.minigap)
            maxpos = np.maximum(minpos, np.minimum(100.0, self.carlength+self.minigap+np.maximum(0.0, (mergervel**2-maintainleadervel**2)/16+mergervel ) ) )
        else:
            minpos = np.maximum(maintainerpos+self.carlength+2*self.minigap, np.minimum(100.0, self.carlength+self.minigap+np.maximum(0.0, (mergervel**2-maintainleadervel**2)/16+mergervel ) ))
            maxpos = 100.0

        self.maintainleaderpos = maintainleaderpos = self.np_random.random_sample()*(maxpos-minpos) + minpos

        # 1: approaching 2: maintain 3: departing
        if disc_maintainleftfollowervel == 1:
            maxvel = np.minimum(maintainervel + 10.0, 30.0)
            minvel = np.minimum(maintainervel + 2.0, 30.0)
        elif disc_maintainleftfollowervel == 2:
            maxvel = np.minimum(maintainervel + 2.0, 30.0)
            minvel = np.maximum(maintainervel - 2.0, 0.0)
        else:
            maxvel = np.maximum(maintainervel - 2.0, 0.0)
            minvel = np.maximum(maintainervel - 10.0, 0.0)

        self.maintainleftfollowervel = maintainleftfollowervel = self.np_random.random_sample()*(maxvel-minvel) + minvel
        # 1 block behind # 2 not block behind
        if disc_maintainleftfollowerpos == 1:
            minpos = np.maximum(-50.0, -self.carlength-self.minigap-np.maximum(0.0, (maintainleftfollowervel**2-maintainervel**2)/16+maintainleftfollowervel ) )
            maxpos = 0.0
        else:
            maxpos = np.maximum(-50.0, -self.carlength-self.minigap-np.maximum(0.0, (maintainleftfollowervel**2-maintainervel**2)/16+maintainleftfollowervel ) )
            minpos = -50.0

        self.maintainleftfollowerpos = maintainleftfollowerpos = self.np_random.random_sample()*(maxpos-minpos) + minpos + maintainerpos

        # 1: approaching 2: maintain 3: departing
        if disc_maintainleftleadervel == 1:
            maxvel = np.maximum(maintainervel - 2.0, 0.0)
            minvel = np.maximum(maintainervel - 10.0, 0.0)
        elif disc_maintainleftleadervel == 2:
            maxvel = np.minimum(maintainervel + 2.0, 30.0)
            minvel = np.maximum(maintainervel - 2.0, 0.0)
        else:
            maxvel = np.minimum(maintainervel + 10.0, 30.0)
            minvel = np.minimum(maintainervel + 2.0, 30.0)
        self.maintainleftleadervel = maintainleftleadervel = self.np_random.random_sample()*(maxvel-minvel) + minvel
        # 1 block in front # 2 not block in front
        if disc_maintainleftleaderpos == 1:
            minpos = np.maximum(0.0, (maintainleftfollowerpos-maintainerpos)+self.carlength+2*self.minigap)

            maxpos = np.maximum(minpos, np.minimum(100.0,  self.carlength+self.minigap+np.maximum(0.0, (maintainervel**2-maintainleftleadervel**2)/16+maintainervel ) ) )
        else:
            minpos = np.maximum((maintainleftfollowerpos-maintainerpos)+self.carlength+2*self.minigap,
                          np.minimum(100.0,
                              self.carlength+self.minigap+np.maximum(0.0,
                                                         (maintainervel**2-maintainleftleadervel**2)/16+maintainervel ) ) )
            maxpos = 100.0
        self.maintainleftleaderpos = maintainleftleaderpos = self.np_random.random_sample()*(maxpos-minpos) + minpos + maintainerpos
        #newstate = [mergervel,-maintainerpos, maintainervel,
        #              mergeleaderpos, mergeleadervel,
        #              maintainleaderpos, maintainleadervel,
        #              maintainleftleaderpos-maintainerpos, maintainleftleadervel,
        #              -maintainleftfollowerpos+maintainerpos, maintainleftfollowervel]
        newstate = [(mergervel-15)/30,
                      (-maintainerpos-25)/50, (maintainervel-15)/30,
                      (mergeleaderpos-50)/100, (mergeleadervel-15)/30,
                      (maintainleaderpos-50)/100, (maintainleadervel-15)/30,
                      (maintainleftleaderpos-maintainerpos-50)/100, (maintainleftleadervel-15)/30,
                      (-maintainleftfollowerpos+maintainerpos-25)/50, (maintainleftfollowervel-15)/30]
        #print("state", newstate)
        return [np.array(newstate), np.array(newstate)]

    def simulate(self, merger,
                       maintainer,
                       mergeleader,
                       maintainleader,
                       maintainleftleader,
                       maintainleftfollower):
        mergerpos,mergervel,mergeract = merger
        maintainerpos, maintainervel, maintaineract = maintainer
        mergeleaderpos, mergeleadervel = mergeleader
        maintainleaderpos, maintainleadervel = maintainleader
        maintainleftleaderpos, maintainleftleadervel = maintainleftleader
        maintainleftfollowerpos, maintainleftfollowervel = maintainleftfollower
        k = 0
        tc = 1.0
        steplength = 0.1
        a = 3.0
        tau = 1.0
        while k*steplength <= tc:
            if mergeract == 2:
                if mergeleaderpos - mergerpos <= 50.0:
                    vdes = np.maximum(mergervel, 34.0)
                    sstar = self.minigap+tau*mergervel+(mergervel*(mergervel-mergeleadervel))/self.idmsqrtacc
                    u = a*(1-(mergervel/vdes)**4-(sstar/(mergeleaderpos-mergerpos-self.minigap-self.carlength+1e-7))**2)
                    vel = np.maximum(np.minimum(mergervel+steplength*u, 34.0), 0.0)
                else:
                    u = 2.0
                    vel = np.maximum(np.minimum(mergervel+steplength*u, 34.0), 0.0)
            else:
                u = -2.0
                vel = np.maximum(np.minimum(mergervel+steplength*u, 34.0), 0.0)
            mergerpos += (mergervel+vel)*steplength*0.5
            mergervel = vel
            if maintaineract == 2:
                if maintainleaderpos - maintainerpos <= 50.0:
                    vdes = np.maximum(maintainervel, 34.0)
                    sstar = self.minigap+tau*maintainervel+(maintainervel*(maintainervel-maintainleadervel))/self.idmsqrtacc
                    u = a*(1-(maintainervel/vdes)**4-(sstar/(maintainleaderpos-maintainerpos-self.minigap-self.carlength+1e-7))**2)
                    vel = np.maximum(np.minimum(maintainervel+steplength*u, 34.0), 0.0)
                else:
                    vel = maintainervel
            else:
                u = -2.0
                vel = np.maximum(np.minimum(maintainervel+steplength*u, 34.0), 0.0)
            maintainerpos += (maintainervel+vel)*steplength*0.5
            maintainervel = vel

            mergeleaderpos += mergeleadervel*steplength
            maintainleaderpos += maintainleadervel*steplength
            if maintainleftleaderpos - maintainleftfollowerpos <= 50.0:
                vdes = np.maximum(maintainleftfollowervel, 34.0)
                sstar = self.minigap+tau*maintainleftfollowervel+(maintainleftfollowervel*(maintainleftfollowervel-maintainleftleadervel))/self.idmsqrtacc
                u = a*(1-(maintainleftfollowervel/vdes)**4-(sstar/(maintainleftleaderpos-maintainleftfollowerpos-self.minigap-self.carlength+1e-7))**2)
                vel = np.maximum(np.minimum(maintainleftfollowervel+steplength*u, 34.0), 0.0)
            else:
                vel = maintainleftfollowervel
            maintainleftfollowerpos += (vel + maintainleftfollowervel)*steplength*0.5
            maintainleftfollowervel = vel
            maintainleftleaderpos += maintainleftleadervel*steplength
            k += 1

        if mergerpos < maintainerpos:
            return np.zeros(11), True

        if mergerpos > maintainleaderpos:
            return np.zeros(11), True

        newstate = []
        newstate.append(mergervel)
        newstate.append(mergerpos-maintainerpos)
        newstate.append(maintainervel)
        newstate.append(mergeleaderpos-mergerpos)
        newstate.append(mergeleadervel)
        newstate.append(maintainleaderpos-mergerpos)
        newstate.append(maintainleadervel)

        self.mergervel = mergervel
        self.mergeleaderpos = mergeleaderpos - mergerpos
        self.mergeleadervel = mergeleadervel
        self.maintainerpos = maintainerpos - mergerpos
        self.maintainervel = maintainervel
        self.maintainleaderpos = maintainleaderpos - mergerpos
        self.maintainleadervel = maintainleadervel

        if maintainerpos - maintainleftfollowerpos < 0.0:
            # follower become leader
            newstate.append(maintainleftfollowerpos-maintainerpos)
            newstate.append(maintainleftfollowervel)
            newstate.append(maintainervel+self.carlength+self.minigap)
            newstate.append(maintainervel)
            self.maintainleftleaderpos = maintainleftfollowerpos-mergerpos
            self.maintainleftleadervel = maintainleftfollowervel
            self.maintainleftfollowerpos = self.maintainerpos - (maintainervel+self.carlength+self.minigap)
            self.maintainleftfollowervel = maintainervel

            return np.array(newstate), False

        if maintainleftleaderpos - maintainerpos < 0.0:
            # leader become follower
            newstate.append(maintainervel+self.carlength+self.minigap)
            newstate.append(maintainervel)
            newstate.append(maintainerpos-maintainleftleaderpos)
            newstate.append(maintainleftleadervel)
            self.maintainleftleaderpos = self.maintainerpos + maintainervel + self.carlength + self.minigap
            self.maintainleftleadervel = maintainervel
            self.maintainleftfollowerpos = maintainleftleaderpos - mergerpos
            self.maintainleftfollowervel = maintainleftleadervel

            return np.array(newstate), False

        newstate.append(maintainleftleaderpos-maintainerpos)
        newstate.append(maintainleftleadervel)
        newstate.append(maintainerpos-maintainleftfollowerpos)
        newstate.append(maintainleftfollowervel)
        self.maintainleftleaderpos = maintainleftleaderpos - mergerpos
        self.maintainleftleadervel = maintainleftleadervel
        self.maintainleftfollowerpos = maintainleftfollowerpos - mergerpos
        self.maintainleftfollowervel = maintainleftfollowervel

        return np.array(newstate), False
    def simulate2(self, ego, leader, sideleader, sidefollower):
        egopos, egovel = ego
        leaderpos, leadervel = leader
        sideleaderpos, sideleadervel = sideleader
        sidefollowerpos, sidefollowervel = sidefollower
        k = 0
        tc = 1.8
        steplength = 0.1
        while k * steplength <= tc:
            newleaderpos = leaderpos + leadervel*steplength
            newsideleaderpos = sideleaderpos + sideleadervel*steplength
            if k * steplength < tc/2:
                egopos += egovel*steplength
                if sideleaderpos - sidefollowerpos <= 50.0:
                    vdes = np.maximum(sidefollowervel, 34.0)
                    u = 3.0
                    tau = 1.0
                    sstar = self.minigap+tau*sidefollowervel+(sidefollowervel*(sidefollowervel-sideleadervel))/self.idmsqrtacc
                    u = u*(1-(sidefollowervel/vdes)**4-(sstar/(sideleaderpos-sidefollowerpos-self.minigap-self.carlength+1e-7))**2)
                    vel = np.maximum(np.minimum(sidefollowervel+steplength*u, 34.0), 0.0)
                else:
                    vel = sidefollowervel

                sidefollowerpos += (vel+sidefollowervel)*steplength*0.5
                sidefollowervel = vel
                sideleaderpos = newsideleaderpos
                leaderpos = newleaderpos
            else:
                if egopos - sidefollowerpos <= 50.0:
                    vdes = np.maximum(sidefollowervel, 34.0)
                    u = 3.0
                    tau = 1.0
                    sstar = self.minigap+tau*sidefollowervel+(sidefollowervel*(sidefollowervel-egovel))/self.idmsqrtacc
                    u = u*(1-(sidefollowervel/vdes)**4-(sstar/(egopos-sidefollowerpos-self.minigap-self.carlength+1e-7))**2)
                    vel = np.maximum(np.minimum(sidefollowervel+steplength*u, 34.0), 0.0)
                else:
                    vel = sidefollowervel
                sidefollowerpos += (vel+sidefollowervel)*steplength*0.5
                sidefollowervel = vel
                if k * steplength < tc/3*2:
                    egopos += egovel*steplength
                    sideleaderpos = newsideleaderpos
                    leaderpos = newleaderpos
                else:
                    if sideleaderpos < leaderpos:
                        refpos = sideleaderpos
                        refvel = sideleadervel
                    else:
                        refpos = leaderpos
                        refvel = leadervel

                    if refpos - egopos <= 50.0:
                        vdes = np.maximum(egovel, 34.0)
                        u = 3.0
                        tau = 1.0
                        sstar = self.minigap+tau*egovel+(egovel*(egovel-refvel))/self.idmsqrtacc
                        u = u*(1-(egovel/vdes)**4-(sstar/(refpos-egopos-self.minigap-self.carlength+1e-7))**2)
                        vel = np.maximum(np.minimum(egovel+steplength*u, 34.0), 0.0)
                    else:
                        vel = egovel

                    egopos += (vel+egovel)*steplength*0.5
                    egovel = vel
                    sideleaderpos = newsideleaderpos
                    leaderpos = newleaderpos
            # check collision
            if (leaderpos-egopos-self.carlength-self.minigap)<=0.0:
                return True

            if (sideleaderpos-egopos-self.carlength-self.minigap)<=0.0:
                return True

            if (egopos-sidefollowerpos-self.carlength-self.minigap)<=0.0:
                return True

            k += 1
        return False
def _multionehot(values, depth):
    a = np.zeros([len(values), depth])
    for i in range(len(values)):
        for j in values[i]:
            a[i, j] = 1
    return a
