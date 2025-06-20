import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

lim = 20
simulation_fps = 250
display_fps = 30
subsample_factor = simulation_fps // display_fps

tau_span = (0, 20)
tau_eval = np.linspace(*tau_span, simulation_fps * (tau_span[1] - tau_span[0]))

G = 2
M = 1

# r0 = 10
# dr0 = 0
# theta0 = np.pi / 4
# phi0 = 0
# dtheta0 = 0.5
# dphi0 = 0

# Multiple orbits
r0 = 30
dr0 = -20.0
theta0 = np.pi / 4
phi0 = 0
dtheta0 = 1.01244760309
dphi0 = 0

# Slingshot
# r0 = 20
# dr0 = -10
# theta0 = np.pi / 4
# phi0 = 0
# dtheta0 = 0.905
# dphi0 = 0

def initial_time(k, r0, dr0, theta0, dtheta0, dphi0):
    L = 1 / (1 - 2 * G * M / r0)
    return np.sqrt(k + L ** 2 * dr0 ** 2 + L * r0 ** 2 * dtheta0 ** 2 + L * r0 ** 2 * np.sin(theta0) ** 2 * dphi0 ** 2)

y0 = [0, initial_time(1, r0, dr0, theta0, dtheta0, dphi0), r0, dr0, theta0, dtheta0, phi0, dphi0]
print(f"Using initial conditions {y0}")

def termination_event(tau, X):
    r  = X[2]
    return r - 2 * G * M

def geodesics(tau, X):
    t, dt, r, dr, theta, dtheta, phi, dphi = X

    horizon_dist = r - 2 * G * M

    ddt = - (2 * G * M) / (r * horizon_dist) * dr * dt
    ddr = (2 * G * M) / (r * horizon_dist) * (dr ** 2) + horizon_dist * (dtheta ** 2) + horizon_dist * np.sin(theta) ** 2 * dphi ** 2 - (G * M * horizon_dist) / (r ** 3) * (dt ** 2)
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

sol = solve_ivp(geodesics, tau_span, y0, t_eval = tau_eval, events = termination_event)
print(f"Simulation status: {sol.status}. {sol.message}")

# Only pick out every few frames
ts = np.arange(0, len(sol.t), subsample_factor)

tau = sol.t[ts]
t, dt, r, dr, theta, dtheta, phi, dphi = (sol.y[i][ts] for i in range(8))

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

# fig = plt.figure(figsize = (15, 10))
# ax = fig.add_subplot(111, projection = '3d')

# ax.set_xlim([-lim, lim])
# ax.set_ylim([-lim, lim])
# ax.set_zlim([-lim, lim])
# ax.set_box_aspect([1, 1, 1])
# ax.set_title("Geodesic in Schwarzschild spacetime")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.plot(x, y, z, label = "Geodesic path")
# ax.plot_surface(sphere_x, sphere_y, sphere_z, color = 'black', edgecolor = 'gray', alpha = 0.5)

fig = plt.figure(figsize = (10, 7))
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
ax = fig.add_subplot(111, projection = '3d')

ax.set_axis_off()
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([-lim, lim])
ax.set_box_aspect([1, 1, 1])
ax.set_clip_on(False)
ax.set_title("Geodesic in Schwarzschild spacetime")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(elev = 30, azim = 60)
ax.dist = 8

ax.plot_surface(sphere_x, sphere_y, sphere_z, color = 'black', edgecolor = 'gray', alpha = 0.5)

line, = ax.plot([], [], linestyle = 'solid', color = 'orange', label = "Trajectory (proper time, infaller's view)")
line.set_clip_box(None)
observer_line, = ax.plot([], [], linestyle = 'dashed', alpha = 0.5, color = 'blue', label = "Trajectory (observer's viewpoint)")
observer_line.set_clip_box(None)

ax.legend()

time_text = ax.text2D(0.05, 0.95, '', transform = ax.transAxes)
velocity_text = ax.text2D(0.05, 0.90, '', transform = ax.transAxes)

observer_t = np.linspace(t[0], t[-1])
observer_r = interp1d(t, r, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')(observer_t)
observer_theta = interp1d(t, theta, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')(observer_t)
observer_phi = interp1d(t, phi, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')(observer_t)

observer_x = observer_r * np.sin(observer_theta) * np.cos(observer_phi)
observer_y = observer_r * np.sin(observer_theta) * np.sin(observer_phi)
observer_z = observer_r * np.cos(observer_theta)

def update(frame):
    line.set_data(x[:frame], y[:frame])
    line.set_3d_properties(z[:frame])

    # observer_line.set_data(observer_x[:frame], observer_y[:frame])
    # observer_line.set_3d_properties(observer_z[:frame])

    # Proper time is the time experienced by the particle itself
    # Coordinate time is the time experienced by the viewer outside the gravitational field.
    time_text.set_text(f"Proper time: {tau[frame]:.2f}, coordinate time: {t[frame]:.2f}")
    velocity_text.set_text(f"Proper velocity: {v_mag[frame]:.2f}")
    
    return line, observer_line, time_text, velocity_text
    # return line, time_text

animation = anim.FuncAnimation(
    fig, update, frames = len(ts), interval = 1000 / display_fps, blit = True
)

plt.show()