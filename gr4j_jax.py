'''
jax implementation of GR4J.
'''

import jax.numpy as jnp
import jax

from functools import partial

# HARDCODING THESE NUMBERS TO PASS JIT ERRORS
# THIS IS NOT GOOD! PLEASE FIX THIS SOON!
# THESE ARE REFERENCED TO x4_max = 5

num_uh1 = 5
num_uh2 = 9 # should be 2*num_uh1 - 1

@jax.jit
def update_state(state, xs, x1, x2, x3, uh1_ordinates, uh2_ordinates):
    
    '''
    Stateless update function for streamflow and reservoirs in GR4J model.
    '''

    s_store, r_store, uh1, uh2 = state

    # Convention for inputs 
    # is that E is first entry of xs and 
    # P is second
    etp   = xs[0]
    prec  = xs[1]
    
    is_wet = prec >= etp

    p_n_wet = prec - etp

    p_s_wet = ((x1 * (1 - (s_store / x1)**2) * jnp.tanh(p_n_wet/x1)) /
           (1 + s_store / x1 * jnp.tanh(p_n_wet / x1)))

    e_s_wet = 0   

    p_n_dry = 0
    pe_n_dry = etp - prec

    e_s_dry = ((s_store * (2 - s_store/x1) * jnp.tanh(pe_n_dry/x1)) 
           / (1 + (1 - s_store / x1) * jnp.tanh(pe_n_dry / x1)))

    p_s_dry = 0

    p_s     = is_wet*p_s_wet  + (1-is_wet)*p_s_dry
    e_s     = is_wet*e_s_wet  + (1-is_wet)*e_s_dry
    p_n     = is_wet*p_n_wet  + (1-is_wet)*p_n_dry

    s_store = s_store - e_s + p_s

    perc = s_store * (1 - (1 + (4/9 * s_store / x1)**4)**(-0.25))

    s_store = s_store - perc

    p_r = perc + (p_n - p_s)

    p_r_uh1 = 0.9 * p_r 
    p_r_uh2 = 0.1 * p_r

    uh1 = uh1.at[0:num_uh1 - 1].set(uh1[1: num_uh1] + uh1_ordinates[0:num_uh1-1]* p_r_uh1)
    uh1 = uh1.at[-1].set(uh1_ordinates[-1] * p_r_uh1)
    
    uh2 = uh2.at[0:num_uh2 - 1].set(uh2[1:num_uh2] + uh2_ordinates[0:num_uh2 - 1] * p_r_uh2)   
    uh2 = uh2.at[-1].set(uh2_ordinates[-1] * p_r_uh2)

    gw_exchange = x2 * (r_store / x3) ** 3.5

    r_store = jnp.maximum(0, r_store + uh1[0] + gw_exchange)

    q_r = r_store * (1 - (1 + (r_store / x3)**4)**(-0.25))

    r_store = r_store - q_r

    q_d = jnp.maximum(0, uh2[0] + gw_exchange)

    qsim = q_r + q_d
    
    return (s_store, r_store, uh1, uh2), qsim

@jax.jit
def update_state_tv_x1(state, xs, x2, x3, uh1_ordinates, uh2_ordinates):
    
    '''
    Repackages static params and forcings to update_state allowing
    for time-varying x1.
    '''
    # Convention for inputs 
    # is that E is first entry of xs and 
    # P is second and time varying x1 is third
    x1 = xs[2]

    return update_state(state, xs[0:2], x1, x2, x3, uh1_ordinates, uh2_ordinates)
    
@jax.jit
def run_gr4j(prec, etp, params):
    
    # Number of simulation timesteps
    num_timesteps = len(prec)
    
    # Unpack the model parameters
    x1, x2, x3, x4, s_init, r_init = params
    
    if x1.size > 1:
        tv_x1 = True
    else:
        tv_x1 = False
    
    # initialize empty arrays for discharge and all storages
    s_store = jnp.zeros(num_timesteps+1)
    r_store = jnp.zeros(num_timesteps+1)
    qsim    = jnp.zeros(num_timesteps+1)
    
    # set initial values
    if tv_x1:
        s_store = s_store.at[0].set(s_init * x1[0])
    else:
        s_store = s_store.at[0].set(s_init * x1)
        
    r_store = r_store.at[0].set(r_init * x3)
    
    # calculate number of unit hydrograph ordinates
    x4_max = 5
    
    uh1_ordinates, uh2_ordinates = hydrograms(x4_max, x4)
 
    # arrays to store the rain distributed through the unit hydrographs
    uh1 = jnp.zeros(num_uh1)
    uh2 = jnp.zeros(num_uh2)
    if tv_x1:
        forcing = jnp.stack([etp, prec, x1], axis=1)
        update_fn = partial(update_state_tv_x1, x2=x2, x3=x3,
                                uh1_ordinates=uh1_ordinates,
                                uh2_ordinates=uh2_ordinates)
        
    else:
        forcing = jnp.stack([etp,prec], axis=1)
        update_fn = partial(update_state, x1=x1, x2=x2, x3=x3,
                                uh1_ordinates=uh1_ordinates,
                                uh2_ordinates=uh2_ordinates)
    

    init    = (s_init, r_init, uh1, uh2)

    # We don't cary about the unit hydrographs, so throw them
    # away here.
    outs, qsim = jax.lax.scan(update_fn, init, forcing)
    s_store, r_store, uh1, uh2  = outs

    return qsim, outs

def soft_if(x, breakpoint, sharpness=10):
    '''
    Continuous analogue of step function.
    '''
    return jax.nn.sigmoid(sharpness*(x-breakpoint))

def hydrograms(x4_limit, x4, sigmoid_a=5):
    '''
    Calculate streamflow routing hydrograms at model initialization.
    '''
    
    timesteps = jnp.arange(2*x4_limit)
    late_multiplier      = soft_if(timesteps, x4)
    early_multiplier     = 1 - late_multiplier
    recession_multiplier = late_multiplier * (1-soft_if(timesteps, 2*x4)) 
    
    SH1 = early_multiplier * ((timesteps/x4)**2.5) + late_multiplier * 1.
    SH2A = early_multiplier * 0.5 * (timesteps/x4)**2.5
    SH2B = recession_multiplier * (1 - 0.5 * (2 - timesteps/x4)**2.5)

    # The next step requires taking a fractional power and
    # an error will be thrown if SH2B_term is negative.
    # Thus, we use only the positive part of it.
    SH2B_term = jnp.maximum((2 - timesteps/x4),0)
    SH2B      = recession_multiplier*(1 - 0.5 * SH2B_term**2.5)
    SH2C      = jax.nn.sigmoid(-sigmoid_a*timesteps - 2*x4)
    
    SH2 = SH2A + SH2B + SH2C
    UH1 = SH1[1::] - SH1[0:-1]
    UH2 = SH2[1::] - SH2[0:-1]
    return UH1, UH2