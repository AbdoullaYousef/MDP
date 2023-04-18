#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class MDP:
    
    def __init__(self,T,R,discount):
        
        assert T.ndim == 3, " Transition Function is invalid, T should have 3 dimensions "
        self.nactions = T.shape[0]
        self.nstates  = T.shape[1]
        assert T.shape == (self.nactions, self.nstates, self.nstates), " Transition Function is invalid "
        self.T = T
        
        assert R.ndim == 2, " Reward Function is invalid, R should have 2 dimensions "
        assert R.shape == (self.nactions, self.nstates), " Reward Function is invalid "
        self.R = R
        
        assert 0 <= discount < 1, " Discount Factor is invalid, Discount should be between [0,1)"
        self.discount = discount
        
    def valueIteration(self, initialV, nIterations=np.inf, tolerance=0.01):
        
        '''Value iteration procedure: V = maxR^a + gamma T^a V '''
        
        V = initialV
        iteration = 0
        
        while iteration < nIterations:
            Q = self.R + self.discount*np.dot(self.T,V)
            newV = Q.max(0)
            epsilon = np.linalg.norm(newV - V, np.inf)
            V = newV
            iteration += 1
            if epsilon <= tolerance: break
            print(V,iteration,epsilon)
        return [V,iteration,epsilon]

        
    def extractPolicy(self,V):
        
        '''Policy extraction procedure: pi = argmax R^a + gamma T^a V '''
                
        Q = self.R + self.discount*np.dot(self.T,V)
        policy = Q.argmax(0)
        print(policy)
        return policy 
    
    def evaluatePolicy(self,policy):
        
        '''Policy evaluation linear equation: V^pi = R^pi + gamma T^pi V^pi '''
        
        T = self.T[policy,np.arange(self.nstates),:]
        R = self.R[policy,np.arange(self.nstates)]
        A = np.identity(self.nstates) - self.discount*T
        V = np.linalg.solve(A,R)
        print(V)
        return V
    
            
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        
        '''Policy iteration alternate between two steps: policy evaluation  V^pi = R^pi + gamma T^pi V^pi 
                                                         policy improvement pi = argmax R^a + gamma T^a V^pi '''
        
        policy = initialPolicy
        V = self.evaluatePolicy(policy)
        iteration = 0
        while iteration < nIterations:
            newPolicy = self.extractPolicy(V)
            iteration += 1
            if (newPolicy == policy).all(): break
            policy = newPolicy
            V = self.evaluatePolicy(policy)
        print(policy,iteration,V)
        return [policy,iteration,V]

 
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        
        '''Partial policy evaluation repeat k times: V^pi = R^pi + gamma T^pi V^pi '''
        
        V = initialV
        T = self.T[policy,np.arange(self.nstates),:]
        R = self.R[policy,np.arange(self.nstates)]
        iteration = 0
        while iteration < nIterations:
            newV = R + self.discount*np.dot(T,V)
            epsilon = np.linalg.norm(newV - V, np.inf)
            V = newV                         
            iteration += 1
            if epsilon <= tolerance: break
        print(V,iteration,epsilon)
        return [V,iteration,epsilon]
    
    
    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        
        ''' Modified policy iteration alternate between two steps: 
                repeat partial policy evaluation k times V^pi = R^pi + gamma T^pi V^pi
                policy improvement pi = argmax R^a + gamma T^a V^pi   '''
        
        policy = initialPolicy
        V = initialV
        iteration = 0
        while iteration < nIterations:
            [V,_,_] = self.evaluatePolicyPartially(policy,V,nIterations=nEvalIterations)
            newPolicy = self.extractPolicy(V)
            [newV,_,_] = self.valueIteration(initialV=V,nIterations=1)
            epsilon = np.linalg.norm(newV - V, np.inf)
            policy = newPolicy
            V = newV
            iteration += 1
            if epsilon < tolerance: break
        print(policy,iteration,epsilon)
        return [policy,iteration,epsilon]
    
    
   


# In[3]:


T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])

R = np.array([[0,0,10,10],[0,0,10,10]])

discount = 0.9        

mdp = MDP(T,R,discount)


# In[4]:


[newV,iteration,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nstates))


# In[5]:


policy = mdp.extractPolicy(newV)


# In[6]:


V = mdp.evaluatePolicy(np.array([0,1,1,1]))


# In[7]:


[policy,iteration,V] = mdp.policyIteration(np.array([0,0,0,0]))


# In[8]:


newV = mdp.evaluatePolicyPartially(np.array([0,1,1,1]),np.array([0,10,0,10]))


# In[9]:


[policy,iteration,epsilon] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,10]))


# # Report the policy, value function, and the number of iterations needed by value iteration when using a tolerance of 0.01 and starting from a value function set to 0 for all states

# In[10]:


[newV,iteration,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nstates), tolerance=0.01)
policy = mdp.extractPolicy(newV)
print("Policy: ", policy)
print("Value function: ", newV)
print("Number of iterations: ", iteration)


# # Report the policy, value function, and the number of iterations needed by policy iteration to find an optimal policy when starting from the policy that chooses action 0 in all states

# In[14]:


[policy,iteration,V] = mdp.policyIteration(np.array([0,0,0,0]))
print("Policy: ", policy)
print("Value function: ", V)
print("Number of iterations: ", iteration)


# # Report the number of iterations needed by modified policy iteration to converge when varying the number of iterations in partial policy evaluation from 1 to 10. Use a tolerance of 0.01, start with the policy that chooses action 0 in all states and start with the value function that assigns 0 to all states.

# In[13]:


[policy,iteration,epsilon] = mdp.modifiedPolicyIteration(np.array([0,0,0,0]),np.array([0,0,0,0]),nEvalIterations=10)
print("Policy: ", policy)
print("Number of iterations: ", iteration)


# # Discuss the impact of the number of iterations in partial policy evaluation on the results and relate the results to value iteration and policy iteration

# In[15]:


[V,iteration,epsilon] = mdp.evaluatePolicyPartially(np.array([0,0,0,0]),np.array([0,0,0,0]))
print("Value function: ", V)
print("Number of iterations: ", iteration)


# Partial policy evaluation is used to estimate the value function of a given policy without evaluating the entire MDP
# 
# The number of iterations in Partial Policy Evaluation has a direct impact on the accuracy of the results obtained.The approximation to the true value function of the policy improves as  the number of iterations increases. As the values of the states are updated based on the expected future rewards.
# 
# Comparing the number of iterations in value iteration, partial policy evaluation and policy iteration. We can see a huge difference between value iteration(57) and partial policy evaluation & policy iteration (2).
