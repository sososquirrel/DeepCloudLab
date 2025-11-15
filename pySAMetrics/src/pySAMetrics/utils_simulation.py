import pySAMetrics

def get_simulation_color(bowen_ratio=1, temperature=300, velocity=0):
    """
    Returns the color of the simulation based on the case.
    
    Parameters:
        bowen_ratio (float): The Bowen ratio of the simulation.
        temperature (float): The temperature of the simulation (in Kelvin).
        velocity (float): The velocity of the simulation (in m/s).
        
    Returns:
        str: The color of the simulation.
    """

    bowen_ratio = float(bowen_ratio)
    temperature = float(temperature)
    velocity = float(velocity)

    # Handle Bowen Ratio
    if bowen_ratio != 1:
        cmap = pySAMetrics.cmap_simu_bowen
        alpha = (1 - bowen_ratio) / 1.3
        alpha = max(0, min(1, alpha))  # Ensure alpha is between 0 and 1
        return cmap(alpha)

    # Handle Temperature
    elif temperature != 300:
        cmap = pySAMetrics.cmap_simu_temp
        alpha = (6 + (300 - temperature)) / 12
        alpha = max(0, min(1, alpha))  # Ensure alpha is between 0 and 1
        return cmap(alpha)

    # Handle Velocity
    elif velocity != 0:
        cmap = pySAMetrics.cmap_simu_shear
        if velocity == 2.5:
            alpha = 0.2
        elif velocity == 5:
            alpha = 0.4
        elif velocity == 10:
            alpha = 0.7
        elif velocity == 20:
            alpha = 0.9
        else:
            alpha = 0  # Default alpha if velocity is unexpected
        return cmap(alpha)

    # Default case
    return 'k'  # Black color for default case
