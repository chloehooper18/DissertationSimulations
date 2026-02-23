# simulate_cz_signal.py

from neurodsp.sim import set_random_seed, create_times, sim_combined

def generate_cz_signal(seed=256, n_seconds=8, s_rate=250):
    """
    Generate a combined Cz signal with default simulation settings.
    """
    set_random_seed(seed)
    times = create_times(n_seconds, s_rate)

    components = {
        'sim_synaptic_current': {
            'n_neurons': 1000,
            'firing_rate': 2,
            't_ker': 1.0,
            'tau_r': 0.002,
            'tau_d': 0.02
        },
        'sim_bursty_oscillation': {
            'freq': 10
        }
    }

    component_variances = [1, 0.5]
    cz_signal = sim_combined(n_seconds, s_rate, components, component_variances)

    return times, cz_signal