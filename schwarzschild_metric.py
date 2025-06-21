from manim import *

import numpy as np

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

simulation_fps = 500
display_fps = config.frame_rate
subsample_factor = simulation_fps // display_fps

tau_span = (0, 10)
tau_eval = np.linspace(*tau_span, simulation_fps * (tau_span[1] - tau_span[0]))

G = 1
M = 1
c = 1

r_horizon = 2 * G * M / c ** 2
r_isco = 6 * G * M / c ** 2

lim = 10

r0 = 6.085
dr0 = -0.6
theta0 = np.pi / 4
phi0 = 0
dtheta0 = 0.5
dphi0 = 0

# ISCO
# r0 = r_isco
# dr0 = 0
# theta0 = np.pi / 2
# dtheta0 = 0
# phi0 = 0
# dphi0 = 0.0

# Slingshot
# r0 = 20
# dr0 = -10
# theta0 = np.pi / 4
# phi0 = 0
# dtheta0 = 0.905
# dphi0 = 0

def initial_time(k, r0, dr0, theta0, dtheta0, dphi0):
    L = 1 / (1 - 2 * G * M / (r0 * c ** 2))
    return np.sqrt(k + L ** 2 * dr0 ** 2 + L * r0 ** 2 * dtheta0 ** 2 + L * r0 ** 2 * np.sin(theta0) ** 2 * dphi0 ** 2)

y0 = [0, initial_time(1, r0, dr0, theta0, dtheta0, dphi0), r0, dr0, theta0, dtheta0, phi0, dphi0]
print(f"Using initial conditions {y0}")

def termination_event(tau, X):
    r  = X[2]
    return r - 2 * G * M

def geodesics(tau, X):
    progress = (tau - tau_span[0]) / (tau_span[1] - tau_span[0])
    bar_size = 20
    symbols = int(np.ceil(progress * 20))
    print(f"Simulation progress: [{symbols * '#'}{(bar_size - symbols) * ' '}] {progress * 100:.2f}%", end  = "\r")

    t, dt, r, dr, theta, dtheta, phi, dphi = X

    a = r - 2 * G * M

    ddt = - (2 * G * M) / (r * a) * dr * dt
    ddr = (2 * G * M) / (r * a) * (dr ** 2) + a * (dtheta ** 2 + np.sin(theta) ** 2 * dphi ** 2) - (G * M * c ** 2) / (r * (r * c ** 2 - 2 * G * M)) * (dt ** 2)
    ddtheta = np.sin(theta) * np.cos(theta) * (dphi ** 2) - (2 / r) * dr * dphi
    ddphi = -2 * dr * dphi / r - 2 * dtheta * dphi / np.tan(theta)

    return [
        dt,
        ddt,
        dr,
        ddr,
        dtheta,
        ddtheta,
        dphi,
        ddphi
    ]

sol = solve_ivp(geodesics, tau_span, y0, t_eval = tau_eval, events = termination_event, method = "Radau")
print(f"Simulation status: {sol.status}. {sol.message}")

# Only pick out every few frames
subsampled = np.arange(0, len(sol.t), subsample_factor)

tau = sol.t[subsampled]
t, dt, r, dr, theta, dtheta, phi, dphi = (sol.y[i][subsampled] for i in range(8))

x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

vx = dr * np.sin(theta) * np.cos(phi) + \
    r * dtheta * np.cos(theta) * np.cos(phi) - \
    r * dphi * np.sin(theta) * np.sin(phi)

vy = dr * np.sin(theta) * np.sin(phi) + \
    r * dtheta * np.cos(theta) * np.sin(phi) + \
    r * dphi * np.sin(theta) * np.cos(phi)

vz = dr * np.cos(theta) - r * dtheta * np.sin(theta)

v_mag = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

sphere_r = 2 * G * M
sphere_theta = np.linspace(0, 2 * np.pi, 30)
sphere_phi = np.linspace(0, np.pi, 30)
sphere_theta, sphere_phi = np.meshgrid(sphere_theta, sphere_phi)

sphere_x = sphere_r * np.sin(sphere_phi) * np.cos(sphere_theta)
sphere_y = sphere_r * np.sin(sphere_phi) * np.sin(sphere_theta)
sphere_z = sphere_r * np.cos(sphere_phi)

observer_t = np.linspace(t[0], t[-1])
observer_r = interp1d(t, r, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')(observer_t)
observer_theta = interp1d(t, theta, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')(observer_t)
observer_phi = interp1d(t, phi, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')(observer_t)

observer_x = observer_r * np.sin(observer_theta) * np.cos(observer_phi)
observer_y = observer_r * np.sin(observer_theta) * np.sin(observer_phi)
observer_z = observer_r * np.cos(observer_theta)

normalization = np.sqrt(np.max(r))
print(f"Dividing by {normalization} to normalize coordinates to [0, 1]")

stack_coords = np.column_stack([x, y, z]) / normalization

config.background_color = WHITE

class SchwarzschildGeodesic(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi = 75 * DEGREES, theta = 45 * DEGREES, zoom = 1)

        event_horizon = Sphere(radius = r_horizon / normalization, color = BLACK, fill_opacity = 1)
        self.add(event_horizon)

        trace = VMobject(color = BLUE)
        particle = Dot3D(stack_coords[0], radius = 0.1, color = BLUE)
        self.add(particle, trace)

        for i in range(len(subsampled)):
            particle.move_to(stack_coords[i])
            trace.set_points_as_corners(stack_coords[:i])
            self.wait(1 / display_fps)

