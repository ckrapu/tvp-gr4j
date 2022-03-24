import numpy as np
import os
import pickle
import pymc3 as pm
import theano
import theano.tensor as tt
import time

from pymc3.distributions import Continuous

def static_gr4j_pm_model(p, pet, q, theta_low=[100,0,10,1], theta_high=[1500,10,100,5]):
    routing_speed_limit = 2*theta_high[-1]-1
    
    with pm.Model() as model:
        
        # Hydrology model parameters
        theta = pm.Uniform('theta', lower=theta_low, upper=theta_high, shape=4)
        
        # Initial volume states
        S0 = pm.Uniform('S0',lower = 0, upper = 1000)
        R0 = pm.Uniform('R0',lower = 0, upper = 100)
        Pr0 = np.zeros(routing_speed_limit)
        error_sigma = pm.HalfNormal('error_sigma', 5)

        streamflow = GR4J('streamflow',
                          x1=theta[0],
                          x2=theta[1],
                          x3=theta[2],
                          x4=theta[3],
                          x4_limit=routing_speed_limit,
                          S0=S0,R0=R0,Pr0=Pr0,
                          sd=error_sigma,
                          precipitation=p,
                          evaporation=pet,truncate=-1,
                          observed=q)
        
        varnames = ['theta', 'S0', 'R0', 'error_sigma']      
        return model, varnames
    

def gr4j_transform(x1,x2,x3,x4,x4_limit,S0,R0,precipitation,evaporation,truncate=-1, time_varying=True, dtype='float64'):
    try:
        x1 = tt.as_tensor_variable(x1)
    except TypeError:
        pass
    
    x2 = tt.as_tensor_variable(x2)
    x3 = tt.as_tensor_variable(x3)
    x4 = tt.as_tensor_variable(x4)
    
    S0 = tt.as_tensor_variable(S0)
    R0 = tt.as_tensor_variable(R0)
    Pr0 = tt.as_tensor_variable(np.zeros(2*x4_limit-1).astype(dtype))
    x4_limit = tt.as_tensor_variable(x4_limit)
    
    precipitation = tt.as_tensor_variable(precipitation)
    evaporation   = tt.as_tensor_variable(evaporation)

    UH1,UH2 = hydrograms(x4_limit,x4)

    if time_varying:
        step_fn       = streamflow_step_tv_x1
        sequences     = [precipitation,evaporation,x1]
        non_sequences = [x2,x3,x4,UH1,UH2]

    else:
        step_fn       = streamflow_step
        sequences     = [precipitation,evaporation]
        non_sequences = [x1,x2,x3,x4,UH1,UH2]

    outputs_info  = [None,S0,None,None,None,Pr0,R0,None,None,None]
    
    results, out_dict = theano.scan(fn=step_fn,
                                sequences=sequences,
                                outputs_info=outputs_info,
                                non_sequences=non_sequences)

    streamflow = results[0]
    return streamflow


class GR4J(Continuous):
    def __init__(self,x1,x2,x3,x4,x4_limit,S0,R0,Pr0,sd,precipitation,evaporation,truncate,
                *args,**kwargs):
        super(GR4J, self).__init__(*args,**kwargs)
        
        self.x1 = tt.as_tensor_variable(x1)
        self.x2 = tt.as_tensor_variable(x2)
        self.x3 = tt.as_tensor_variable(x3)
        self.x4 = tt.as_tensor_variable(x4)
        self.x4_limit   = tt.as_tensor_variable(x4_limit)
        
        self.S0 = tt.as_tensor_variable(S0)
        self.R0 = tt.as_tensor_variable(R0)
        self.Pr0 = tt.as_tensor_variable(np.zeros(2*x4_limit-1))
        self.sd = tt.as_tensor_variable(sd)
        
        self.precipitation = tt.as_tensor_variable(precipitation)
        self.evaporation   = tt.as_tensor_variable(evaporation)

        # If we want the autodiff to stop calculating the gradient after
        # some number of chain rule applications, we pass an integer besides
        # -1 here.
        self.truncate   = truncate 

    def get_streamflow(self,precipitation,evaporation,
                      S0,Pr0,R0,x1,x2,x3,x4,x4_limit):
        
        UH1,UH2 = hydrograms(x4_limit,x4)
        results,out_dict = theano.scan(fn = streamflow_step,
                              sequences = [precipitation,evaporation],
                              outputs_info = [None,S0,None,None,None,
                                              Pr0,R0,None,None,None],
                              non_sequences = [x1,x2,x3,x4,UH1,UH2])
        streamflow = results[0]
        return streamflow
    
    def logp(self,observed):
        simulated = self.get_streamflow(self.precipitation,self.evaporation,
                                       self.S0,self.Pr0,self.R0,
                                       self.x1,self.x2,self.x3,self.x4,self.x4_limit)
        sd = self.sd 
        
        return tt.sum(pm.Normal.dist(mu = simulated,sd = sd).logp(observed))

# Determines how much precipitation reaches the production store
def calculate_precip_store(S,precip_net,x1):
    n = x1*(1 - (S / x1)**2) * tt.tanh(precip_net/x1)
    d = 1 + (S / x1) * tt.tanh(precip_net / x1)
    return n/d

# Determines the evaporation loss from the production store
def calculate_evap_store(S,evap_net,x1):
    n = S * (2 - S / x1) * tt.tanh(evap_net/x1)
    d = 1 + (1- S/x1) * tt.tanh(evap_net / x1)
    return n/d

# Determines how much water percolates out of the production store to streamflow
def calculate_perc(current_store,x1):
    return current_store * (1- (1+(4.0/9.0 * current_store / x1)**4)**-0.25)

def streamflow_step_tv_x1(P,E,x1,S,runoff_history,R,x2,x3,x4,UH1,UH2):
     
    # Calculate net precipitation and evapotranspiration
    precip_difference = P-E
    precip_net    = tt.maximum(precip_difference,0)
    evap_net      =  tt.maximum(-precip_difference,0)
    
    # Calculate the fraction of net precipitation that is stored
    precip_store = calculate_precip_store(S,precip_net,x1)
    
    # Calculate the amount of evaporation from storage
    evap_store   = calculate_evap_store(S,evap_net,x1)
    
    # Update the storage by adding effective precipitation and
    # removing evaporation
    S = S - evap_store + precip_store
    
    # Update the storage again to reflect percolation out of the store
    perc = calculate_perc(S,x1)
    S = S  - perc
    
    # The precip. for routing is the sum of the rainfall which
    # did not make it to storage and the percolation from the store
    current_runoff = perc + ( precip_net - precip_store)
    
    # runoff_history keeps track of the recent runoff values in a vector
    # that is shifted by 1 element each timestep.
    runoff_history = tt.roll(runoff_history,1)
    runoff_history = tt.set_subtensor(runoff_history[0],current_runoff)
    
    Q9 = 0.9* tt.dot(runoff_history,UH1)
    Q1 = 0.1* tt.dot(runoff_history,UH2)
    
    F = x2*(R/x3)**3.5
    R = tt.maximum(0,R+Q9+F)
    
    Qr = R * (1-(1+(R/x3)**4)**-0.25)
    R = R-Qr
    
    Qd = tt.maximum(0,Q1+F)
    Q = Qr+Qd
    
    # The order of the returned values is important because it must correspond
    # up with the order of the kwarg list argument 'outputs_info' to theano.scan.
    return Q,S,precip_store,evap_store,perc,runoff_history,R,F,Q9,Q1

def hydrograms(x4_limit,x4):
    timesteps = tt.arange(2*x4_limit)
    
    SH1  = tt.switch(timesteps <= x4,(timesteps/x4)**2.5,1.0)
    SH2A = tt.switch(timesteps <= x4, 0.5 * (timesteps/x4)**2.5,0)
    SH2B = tt.switch(( x4 < timesteps) & (timesteps <= 2*x4),
                     1 - 0.5 * (2 - timesteps/x4)**2.5,0)
    
    # The next step requires taking a fractional power and 
    # an error will be thrown if SH2B_term is negative.
    # Thus, we use only the positive part of it.
    SH2B_term = tt.maximum((2 - timesteps/x4),0)
    SH2B = tt.switch(( x4 < timesteps) & (timesteps <= 2*x4),
                     1 - 0.5 * SH2B_term**2.5,0)
    SH2C = tt.switch( 2*x4 < timesteps,1,0)

    SH2 = SH2A + SH2B + SH2C
    UH1 = SH1[1::] - SH1[0:-1]
    UH2 = SH2[1::] - SH2[0:-1]
    return UH1,UH2

def streamflow_step(P,E,S,runoff_history,R,x1,x2,x3,x4,UH1,UH2):
     
    # Calculate net precipitation and evapotranspiration
    precip_difference = P-E
    precip_net    = tt.maximum(precip_difference,0)
    evap_net      =  tt.maximum(-precip_difference,0)
    
    # Calculate the fraction of net precipitation that is stored
    precip_store = calculate_precip_store(S,precip_net,x1)
    
    # Calculate the amount of evaporation from storage
    evap_store   = calculate_evap_store(S,evap_net,x1)
    
    # Update the storage by adding effective precipitation and
    # removing evaporation
    S = S - evap_store + precip_store
    
    # Update the storage again to reflect percolation out of the store
    perc = calculate_perc(S,x1)
    S = S  - perc
    
    # The precip. for routing is the sum of the rainfall which
    # did not make it to storage and the percolation from the store
    current_runoff = perc + ( precip_net - precip_store)
    
    # runoff_history keeps track of the recent runoff values in a vector
    # that is shifted by 1 element each timestep.
    runoff_history = tt.roll(runoff_history,1)
    runoff_history = tt.set_subtensor(runoff_history[0],current_runoff)
    
    Q9 = 0.9* tt.dot(runoff_history,UH1)
    Q1 = 0.1* tt.dot(runoff_history,UH2)
    
    F = x2*(R/x3)**3.5
    R = tt.maximum(0,R+Q9+F)
    
    Qr = R * (1-(1+(R/x3)**4)**-0.25)
    R = R-Qr
    
    Qd = tt.maximum(0,Q1+F)
    Q = Qr+Qd
    
    # The order of the returned values is important because it must correspond
    # up with the order of the kwarg list argument 'outputs_info' to theano.scan.
    return Q,S,precip_store,evap_store,perc,runoff_history,R,F,Q9,Q1

def streamflow_step_tv_x1(P,E,x1,S,runoff_history,R,x2,x3,x4,UH1,UH2):
     
    # Calculate net precipitation and evapotranspiration
    precip_difference = P-E
    precip_net    = tt.maximum(precip_difference,0)
    evap_net      =  tt.maximum(-precip_difference,0)
    
    # Calculate the fraction of net precipitation that is stored
    precip_store = calculate_precip_store(S,precip_net,x1)
    
    # Calculate the amount of evaporation from storage
    evap_store   = calculate_evap_store(S,evap_net,x1)
    
    # Update the storage by adding effective precipitation and
    # removing evaporation
    S = S - evap_store + precip_store
    
    # Update the storage again to reflect percolation out of the store
    perc = calculate_perc(S,x1)
    S = S  - perc
    
    # The precip. for routing is the sum of the rainfall which
    # did not make it to storage and the percolation from the store
    current_runoff = perc + ( precip_net - precip_store)
    
    # runoff_history keeps track of the recent runoff values in a vector
    # that is shifted by 1 element each timestep.
    runoff_history = tt.roll(runoff_history,1)
    runoff_history = tt.set_subtensor(runoff_history[0],current_runoff)
    
    Q9 = 0.9* tt.dot(runoff_history,UH1)
    Q1 = 0.1* tt.dot(runoff_history,UH2)
    
    F = x2*(R/x3)**3.5
    R = tt.maximum(0,R+Q9+F)
    
    Qr = R * (1-(1+(R/x3)**4)**-0.25)
    R = R-Qr
    
    Qd = tt.maximum(0,Q1+F)
    Q = Qr+Qd
    
    # The order of the returned values is important because it must correspond
    # up with the order of the kwarg list argument 'outputs_info' to theano.scan.
    return Q,S,precip_store,evap_store,perc,runoff_history,R,F,Q9,Q1

class GR4J_tv_x1(Continuous):
    def __init__(self,x1,x2,x3,x4,x4_limit,S0,R0,Pr0,sd,precipitation,evaporation,truncate,
                *args,**kwargs):
        super(GR4J_tv_x1,self).__init__(*args,**kwargs)
        
        self.x1 = tt.as_tensor_variable(x1)
        self.x2 = tt.as_tensor_variable(x2)
        self.x3 = tt.as_tensor_variable(x3)
        self.x4 = tt.as_tensor_variable(x4)
        self.x4_limit   = tt.as_tensor_variable(x4_limit)
        
        self.S0 = tt.as_tensor_variable(S0)
        self.R0 = tt.as_tensor_variable(R0)
        self.Pr0 = tt.as_tensor_variable(np.zeros(2*x4_limit-1))
        self.sd = tt.as_tensor_variable(sd)
        
        self.precipitation = tt.as_tensor_variable(precipitation)
        self.evaporation   = tt.as_tensor_variable(evaporation)

        # If we want the autodiff to stop calculating the gradient after
        # some number of chain rule applications, we pass an integer besides
        # -1 here.
        self.truncate   = truncate 

    def get_streamflow(self,precipitation,evaporation,
                      S0,Pr0,R0,x1,x2,x3,x4,x4_limit):
        
        UH1,UH2 = hydrograms(x4_limit,x4)
        results,out_dict = theano.scan(fn = streamflow_step_tv_x1,
                              sequences = [precipitation,evaporation,x1],
                              outputs_info = [None,S0,None,None,None,
                                              Pr0,R0,None,None,None],
                              non_sequences = [x2,x3,x4,UH1,UH2])
        streamflow = results[0]
        return streamflow
    
    def logp(self,observed):
        simulated = self.get_streamflow(self.precipitation,self.evaporation,
                                       self.S0,self.Pr0,self.R0,
                                       self.x1,self.x2,self.x3,self.x4,self.x4_limit)
        sd = self.sd 
        
        return tt.sum(pm.Normal.dist(mu = simulated,sd = sd).logp(observed))


def constant_prior(model, T, lower=100, upper=1500, varname='theta_1'):
    with model:
        intercept = pm.Uniform('intercept', lower=lower, upper=upper)
        tvp = pm.Deterministic(varname, tt.repeat(intercept, T))
    return tvp

def changepoint_prior(model, T, lower=100, upper=1500, varname='theta_1'):
    with model:
        timesteps   = np.arange(T)
        changepoint = pm.DiscreteUniform('changepoint',lower=1, upper=T-1)
        period_values = pm.Uniform('period_values', lower=lower, upper=upper, shape=2)
        tvp = pm.Deterministic(varname, tt.switch(timesteps<changepoint,
                                                period_values[0], period_values[1]))
    return tvp
    
def linear_basis_prior(model, T, basis_functions=None, lower=100, upper=1500, varname='theta_1'):
    with model:
        intercept = pm.Uniform('intercept', lower=lower, upper=upper)
        T, P = basis_functions.shape
        sigma_beta = pm.HalfNormal('beta_sigma', sd=2.0)
        beta = pm.Normal('beta', sd=sigma_beta, shape=P)
        tvp = pm.Deterministic(varname, intercept + pm.math.dot(basis_functions, beta))
    return tvp

    
def grw_prior(model, T, lower=100, upper=1500, varname='theta_1', sigma_delta=False, resolution=1, prefix_zero=False, grw_intercept=True):
    n_deltas  = int(T/resolution)
    with model:
        if grw_intercept:
            intercept = pm.Uniform('intercept', lower=lower, upper=upper)
        else:
            intercept = 0.

        if not sigma_delta:
            sigma_delta = pm.HalfNormal('sigma_delta', sd=3)
        
        if prefix_zero:
            delta = pm.Normal('delta', sd=sigma_delta, shape=n_deltas-1)
            delta_cumulative = tt.cumsum(tt.concatenate([tt.zeros(1), delta],axis=0))
        else:
            delta = pm.Normal('delta', sd=sigma_delta, shape=n_deltas)
            delta_cumulative = tt.cumsum(delta)

        tvp = pm.Deterministic(varname, intercept + tt.repeat(delta_cumulative, resolution))
    return tvp

def linear_plus_grw(model, T, basis_functions, lower=100, upper=1500, varname='theta_1', sigma_delta=False, resolution=1, prefix_zero=False):
    grw = grw_prior(model, T, lower=lower, upper=upper, varname=f'{varname}_grw',
     sigma_delta=sigma_delta, resolution=resolution, prefix_zero=prefix_zero,
     grw_intercept=False)
    linear = linear_basis_prior(model, T, basis_functions=basis_functions, lower=lower, upper=upper, varname=f'{varname}_linear')
    tvp = pm.Deterministic(varname, grw+linear)
    return tvp

def exp_quad_gp_prior(model, T, lower=100, upper=1500, varname='theta_1', 
                            force_gp_scale=False,resolution=1):
    timesteps = np.linspace(0, T, int(T/resolution))[:, None]
    
    with model:
        rho = pm.Uniform('rho', lower=1, upper=T)

        # Used for testing to see if prior has unintended consequences.
        if force_gp_scale:
            gp_scale = force_gp_scale
        else:
            gp_scale = pm.HalfNormal('gp_scale', sd=100)

        intercept = pm.Uniform('intercept', lower=lower, upper=upper)
        
        cov_func  = pm.gp.cov.ExpQuad(1, ls=rho) * gp_scale**2
        
        mean_func = pm.gp.mean.Constant(intercept)
        gp = pm.gp.Latent(mean_func, cov_func)
        low_res = tt.squeeze(gp.prior('low_resolution', X=timesteps))
        tvp = pm.Deterministic(varname, tt.squeeze(tt.repeat(low_res, resolution)))
        return tvp

def sparse_matern_gp_prior(model, T, n_inducing=10, lower=100, upper=1500, varname='theta_1', sample_prior=False, approx='VFE',
                            force_gp_scale=False,resolution=1):
    timesteps = np.linspace(0, T, int(T/resolution))[:, None]
    inducing  = np.linspace(0, T, n_inducing)[:, None]
    
    with model:
        rho = pm.Uniform('rho', lower=1, upper=T)

        # Used for testing to see if prior has unintended consequences.
        if force_gp_scale:
            gp_scale = force_gp_scale
        else:
            gp_scale = pm.HalfNormal('gp_scale', sd=100)

        intercept = pm.Uniform('intercept', lower=lower, upper=upper)
        
        cov_func  = pm.gp.cov.Matern52(1, ls=rho) * gp_scale**2
        
        if sample_prior:
            mean_func = pm.gp.mean.Constant(intercept)
            gp = pm.gp.Latent(mean_func, cov_func)
            low_res = tt.squeeze(gp.prior('low_resolution', X=timesteps))
        else:
            gp = pm.gp.MarginalSparse(pm.gp.mean.Zero(), cov_func, approx=approx)
            low_res = gp.marginal_likelihood('low_resolution', X=timesteps,
                                         Xu=inducing, y=None, noise=0.1,
                                        is_observed=False)
            low_res = tt.squeeze(low_res) + intercept

        tvp = pm.Deterministic(varname, tt.squeeze(tt.repeat(low_res, resolution)))
        return tvp
    
    
'''def sparse_composite_gp_prior(model, T, n_inducing=10, rho_bounds=[5, 100], lower=100, upper=1500, varname='theta_1', sample_prior=False, approx='VFE'):
    timesteps = np.arange(T)[:, None]
    inducing  = np.linspace(0, T, n_inducing)[:, None]
    
    with model:
        period = pm.Uniform('period', lower=10, upper=365)
        rho = pm.Uniform('rho', lower=10, upper=365, shape=2)
        gp_scale = pm.HalfNormal('gp_scale', sd=np.asarray([100,100]), shape=2)
        intercept = pm.Uniform('intercept', lower=lower, upper=upper)
        
        mean_func = pm.gp.mean.Constant(intercept)

        cov_func  = pm.gp.cov.Matern52(2, ls=rho[0], active_dims=[0])*gp_scale[0]**2 + \
                    pm.gp.cov.Cosine(2, period, active_dims=[0]) * gp_scale[1]**2 
        
        if sample_prior:
            gp = pm.gp.Latent(mean_func, cov_func)
            tvp = gp.prior(varname, X=timesteps)
        else:
            gp = pm.gp.MarginalSparse(mean_func, cov_func, approx=approx)
            tvp = gp.marginal_likelihood(varname, X=timesteps,
                                         Xu=inducing, y=None, noise=0.1,
                                        is_observed=False)
        return tvp'''


def model_wrapper(input_args, x4_limit=5, upper=900, lower=200, return_model=False, save_dir='./',
                  init='jitter+adapt_diag'):
    data, observed_streamflow, prior, name, sampler_kwargs = input_args

    T = len(observed_streamflow.squeeze())
    save_path = save_dir + name + '.pkl'

    if not return_model and os.path.exists(save_path):
        print(f'Skipping {name} as file already exists at {save_path}')
        return
    
    with pm.Model() as model:
        start = time.time()
        theta = pm.Uniform('theta_static', lower=[0,10,1], upper=[10,100,5], shape=3)
        
        err_sd = pm.HalfNormal('err_sd', 5)

        tvp = prior(model, T, upper=upper, lower=lower)
        
        S0_fraction = pm.Uniform('S0_fraction', lower=0.0, upper=0.75)
        S0 = pm.Deterministic('S0', S0_fraction*tvp[0])
        
        R0_fraction = pm.Uniform('R0_fraction', lower=0.0, upper=0.75)
        R0 = pm.Deterministic('R0', R0_fraction*theta[2])

        raw_streamflow = gr4j_transform(tvp,
                       theta[0],
                       theta[1],
                       theta[2],
                       x4_limit,
                       S0,R0, 
                       data['p'], 
                       data['pet'])

        raw_streamflow = pm.Deterministic('raw_streamflow', raw_streamflow)
        streamflow = pm.Normal('streamflow', mu=raw_streamflow, 
                               sd=err_sd, shape=len(observed_streamflow),
                              observed=observed_streamflow)
        if return_model:
            return model
        if sampler_kwargs['step'] == 'metropolis':
            trace = pm.sample(step=pm.Metropolis(), chains=sampler_kwargs['chains'], cores=sampler_kwargs['cores'], tune=sampler_kwargs['tune'],
                          draws=sampler_kwargs['draws'], init=init)
        else:
            trace = pm.sample(cores=sampler_kwargs['cores'], tune=sampler_kwargs['tune'],
                          draws=sampler_kwargs['draws'], init=init,target_accept=sampler_kwargs['tune'],
                          chains=sampler_kwargs['chains'],compute_convergence_checks=False)
        end = time.time()
        
        with open(save_path, 'wb') as dst:
            trace_dict = {n: trace[n] for n in trace.varnames}
            pickle.dump(trace_dict, dst)
            
    return {'trace': trace, 'time' : end - start}
    
