import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

simulation_fps = 500
display_fps = 30
subsample_factor = simulation_fps // display_fps
restart_timeout = 2

tau_span = (0, 10000)
tau_eval = np.linspace(*tau_span, simulation_fps * (tau_span[1] - tau_span[0]))

G = 1
M = 1
c = 1

r_horizon = 2 * G * M / c ** 2
r_isco = 6 * G * M / c ** 2

lim = 10

# r0 = 6.085
# dr0 = -0.5
# theta0 = np.pi / 4
# phi0 = 0
# dtheta0 = 0.5
# dphi0 = 0

# ISCO
r0 = r_isco
dr0 = 0
theta0 = np.pi / 2
dtheta0 = 0
phi0 = 0
dphi0 = 0.0

# Slingshot
# r0 = 20
# dr0 = -10
# theta0 = np.pi / 4
# phi0 = 0
# dtheta0 = 0.905
# dphi0 = 0

TIMELIKE = 1
LIGHTLIKE = 0

def particle_dynamics(kind, initial_conditions):
    r0, dr0, theta0, dtheta0, phi0, dphi0 = initial_conditions

    def time_speed(k, r0, dr0, theta0, dtheta0, dphi0):
        L = 1 / (1 - 2 * G * M / (r0 * c ** 2))
        return np.sqrt(k + L ** 2 * dr0 ** 2 + L * r0 ** 2 * dtheta0 ** 2 + L * r0 ** 2 * np.sin(theta0) ** 2 * dphi0 ** 2)

    y0 = [0, time_speed(kind, r0, dr0, theta0, dtheta0, dphi0), r0, dr0, theta0, dtheta0, phi0, dphi0]
    print(f"Using initial conditions {y0}")

    def termination_event(tau, X):
        r  = X[2]
        return r - 2 * G * M

    def geodesics(tau, X):
        progress = (tau - tau_span[0]) / (tau_span[1] - tau_span[0])
        bar_size = 20
        symbols = int(np.ceil(progress * 20))
        print(f"Simulation progress: [{symbols * '#'}{(bar_size - symbols) * ' '}] {progress * 100:.2f}%", end  = "\r")

        t_, dt, r, dr, theta, dtheta, phi, dphi = X

        

        a = r - 2 * G * M

        # ddt = - (2 * G * M) / (r * a) * dr * dt
        # ddr = (2 * G * M) / (r * a) * (dr ** 2) + a * (dtheta ** 2 + np.sin(theta) ** 2 * dphi ** 2) - (G * M * c ** 2) / (r * (r * c ** 2 - 2 * G * M)) * (dt ** 2)
        # ddtheta = np.sin(theta) * np.cos(theta) * (dphi ** 2) - (2 / r) * dr * dphi
        # ddphi = -2 * dr * dphi / r - 2 * dtheta * dphi / np.tan(theta)

        theta = np.pi / 2
        dtheta = 0
        ddtheta = 0

        ddt = - (2 * G * M) / (r * a) * dr * dt
        ddr = (2 * G * M) / (r * a) * (dr ** 2) + a * (dtheta ** 2 + np.sin(theta) ** 2 * dphi ** 2) - (G * M * c ** 2) / (r * (r * c ** 2 - 2 * G * M)) * (dt ** 2)
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

    return tau, t, x, y, z, v_mag

sphere_r = 2 * G * M
sphere_theta = np.linspace(0, 2 * np.pi, 30)
sphere_phi = np.linspace(0, np.pi, 30)
sphere_theta, sphere_phi = np.meshgrid(sphere_theta, sphere_phi)

sphere_x = sphere_r * np.sin(sphere_phi) * np.cos(sphere_theta)
sphere_y = sphere_r * np.sin(sphere_phi) * np.sin(sphere_theta)
sphere_z = sphere_r * np.cos(sphere_phi)

fig = plt.figure(figsize = (10, 7))

# ax = fig.add_subplot(111, projection = '3d')
ax = fig.add_subplot(111)
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
# ax.set_zlim([-lim, lim])
ax.set_aspect(1)
# ax.set_box_aspect([1, 1, 1])
ax.set_title("Geodesics of null and timelike particles")
ax.set_xlabel("x")
ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.view_init(elev = 90, azim = 0)
# ax.dist = 8

# for dphi0 in np.linspace(0.15425, 0.1543, 10):
dphi0 = 0.15430335
    # p_lambda, p_t, p_x, p_y, p_z, p_v = particle_dynamics(LIGHTLIKE, [6 * G * M, -1, np.pi / 2, 0, 0, dphi0])
s_lambda, s_t, s_x, s_y, s_z, s_v = particle_dynamics(TIMELIKE, [6 * G * M, 0, np.pi / 2, 0, 0, dphi0])

ax.plot(s_x, s_y, label = f"Timelike particle ($d\\phi/d\\tau = {dphi0}$)")
    # # ax.plot(p_x, p_y, label = f"Lightlike particle ($d\\phi/d\\tau = {dphi0}$)")

# ax.plot_surface(sphere_x, sphere_y, color = 'black', edgecolor = 'gray', alpha = 1.0)

hole = plt.Circle((0, 0), 2 * G * M / (c ** 2), color = 'black')
ax.add_patch(hole)

# ax.legend()

# plt.show()

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
ax.view_init(elev = 45, azim = 45)
ax.dist = 8

ax.plot_surface(sphere_x, sphere_y, sphere_z, color = 'black', edgecolor = 'gray', alpha = 0.5)

line, = ax.plot([], [], linestyle = 'solid', color = 'orange', label = "Trajectory (proper time, infaller's view)")
line.set_clip_box(None)
# particle = ax.scatter(0, 0, 0, marker = '^', s = 100)
observer_line, = ax.plot([], [], linestyle = 'dashed', alpha = 0.5, color = 'blue', label = "Trajectory (observer's viewpoint)")
observer_line.set_clip_box(None)

ax.legend()

info_text = ax.text2D(0.05, 0.95, '', transform = ax.transAxes)

# observer_t = np.linspace(t[0], t[-1])
# observer_r = interp1d(t, r, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')(observer_t)
# observer_theta = interp1d(t, theta, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')(observer_t)
# observer_phi = interp1d(t, phi, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')(observer_t)

# observer_x = observer_r * np.sin(observer_theta) * np.cos(observer_phi)
# observer_y = observer_r * np.sin(observer_theta) * np.sin(observer_phi)
# observer_z = observer_r * np.cos(observer_theta)

# frames = len(ts)
def update(frame):
    if frame >= len(s_x):
        # Animation has finished, wait before restarting
        # return particle, observer_line, info_text
        return line, info_text

    line.set_data(s_x[:frame], s_y[:frame])
    line.set_3d_properties(s_z[:frame])
    # particle._offsets3d = ([x[frame]], [y[frame]], [z[frame]])

    # observer_line.set_data(observer_x[:frame], observer_y[:frame])
    # observer_line.set_3d_properties(observer_z[:frame])

    # Proper time is the time experienced by the particle itself
    # Coordinate time is the time experienced by the viewer outside the gravitational field.
    r = np.sqrt(s_x[frame] ** 2 + s_y[frame] ** 2 + s_z[frame] ** 2)
    info_text.set_text(
        f"Proper time: {s_lambda[frame]:.2f}, coordinate time: {s_t[frame]:.2f}\n"
        f"Distance from event horizon: {r - 2 * G * M / c ** 2:.2f}, velocity: {s_v[frame]:.2f}"
    )
    
    return line, info_text
    # return line, time_text

animation = anim.FuncAnimation(
    fig, update, frames = len(s_x) + display_fps * restart_timeout, interval = 1000 / display_fps, blit = True
)

# # from manim import *

# # class SchwarzschildGeodesic(ThreeDScene):
# #     def construct(self):
# #         self.set_camera_orientation(phi = 75 * DEGREES, theta = 45 * DEGREES, distance = 8)

# #         sphere = Sphere(radius = r_horizon, )


plt.show()