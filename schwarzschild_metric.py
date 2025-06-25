import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

dt = 0.001
simulation_fps = int(np.ceil(1 / dt))

# simulation_fps = 500
display_fps = 30
subsample_factor = simulation_fps // display_fps
restart_timeout = 1

tau_span = (0, 25)
tau_eval = np.linspace(*tau_span, simulation_fps * (tau_span[1] - tau_span[0]))

# tau_eval = np.linspace(*tau_span, (tau_span[1] - tau_span[0]) / dt)

G = 1
M = 1
c = 1

r_horizon = 2 * G * M / c ** 2
r_isco = 6 * G * M / c ** 2

lim = 20

initial_conditions = [
[20.000000, 1.570796, 0.000000, -0.889910, -0.016127, -0.016127],
[20.000000, 1.570796, 0.000000, -0.913487, -0.016554, -0.011824],
[20.000000, 1.570796, 0.000000, -0.930288, -0.016859, -0.007225],
[20.000000, 1.570796, 0.000000, -0.939043, -0.017017, -0.002431],
[20.000000, 1.570796, 0.000000, -0.939043, -0.017017, 0.002431],
[20.000000, 1.570796, 0.000000, -0.930288, -0.016859, 0.007225],
[20.000000, 1.570796, 0.000000, -0.913487, -0.016554, 0.011824],
[20.000000, 1.570796, 0.000000, -0.889910, -0.016127, 0.016127],
[20.000000, 1.570796, 0.000000, -0.913487, -0.011824, -0.016554],
[20.000000, 1.570796, 0.000000, -0.939043, -0.012155, -0.012155],
[20.000000, 1.570796, 0.000000, -0.957322, -0.012392, -0.007435],
[20.000000, 1.570796, 0.000000, -0.966871, -0.012515, -0.002503],
[20.000000, 1.570796, 0.000000, -0.966871, -0.012515, 0.002503],
[20.000000, 1.570796, 0.000000, -0.957322, -0.012392, 0.007435],
[20.000000, 1.570796, 0.000000, -0.939043, -0.012155, 0.012155],
[20.000000, 1.570796, 0.000000, -0.913487, -0.011824, 0.016554],
[20.000000, 1.570796, 0.000000, -0.930288, -0.007225, -0.016859],
[20.000000, 1.570796, 0.000000, -0.957322, -0.007435, -0.012392],
[20.000000, 1.570796, 0.000000, -0.976712, -0.007586, -0.007586],
[20.000000, 1.570796, 0.000000, -0.986860, -0.007664, -0.002555],
[20.000000, 1.570796, 0.000000, -0.986860, -0.007664, 0.002555],
[20.000000, 1.570796, 0.000000, -0.976712, -0.007586, 0.007586],
[20.000000, 1.570796, 0.000000, -0.957322, -0.007435, 0.012392],
[20.000000, 1.570796, 0.000000, -0.930288, -0.007225, 0.016859],
[20.000000, 1.570796, 0.000000, -0.939043, -0.002431, -0.017017],
[20.000000, 1.570796, 0.000000, -0.966871, -0.002503, -0.012515],
[20.000000, 1.570796, 0.000000, -0.986860, -0.002555, -0.007664],
[20.000000, 1.570796, 0.000000, -0.997330, -0.002582, -0.002582],
[20.000000, 1.570796, 0.000000, -0.997330, -0.002582, 0.002582],
[20.000000, 1.570796, 0.000000, -0.986860, -0.002555, 0.007664],
[20.000000, 1.570796, 0.000000, -0.966871, -0.002503, 0.012515],
[20.000000, 1.570796, 0.000000, -0.939043, -0.002431, 0.017017],
[20.000000, 1.570796, 0.000000, -0.939043, 0.002431, -0.017017],
[20.000000, 1.570796, 0.000000, -0.966871, 0.002503, -0.012515],
[20.000000, 1.570796, 0.000000, -0.986860, 0.002555, -0.007664],
[20.000000, 1.570796, 0.000000, -0.997330, 0.002582, -0.002582],
[20.000000, 1.570796, 0.000000, -0.997330, 0.002582, 0.002582],
[20.000000, 1.570796, 0.000000, -0.986860, 0.002555, 0.007664],
[20.000000, 1.570796, 0.000000, -0.966871, 0.002503, 0.012515],
[20.000000, 1.570796, 0.000000, -0.939043, 0.002431, 0.017017],
[20.000000, 1.570796, 0.000000, -0.930288, 0.007225, -0.016859],
[20.000000, 1.570796, 0.000000, -0.957322, 0.007435, -0.012392],
[20.000000, 1.570796, 0.000000, -0.976712, 0.007586, -0.007586],
[20.000000, 1.570796, 0.000000, -0.986860, 0.007664, -0.002555],
[20.000000, 1.570796, 0.000000, -0.986860, 0.007664, 0.002555],
[20.000000, 1.570796, 0.000000, -0.976712, 0.007586, 0.007586],
[20.000000, 1.570796, 0.000000, -0.957322, 0.007435, 0.012392],
[20.000000, 1.570796, 0.000000, -0.930288, 0.007225, 0.016859],
[20.000000, 1.570796, 0.000000, -0.913487, 0.011824, -0.016554],
[20.000000, 1.570796, 0.000000, -0.939043, 0.012155, -0.012155],
[20.000000, 1.570796, 0.000000, -0.957322, 0.012392, -0.007435],
[20.000000, 1.570796, 0.000000, -0.966871, 0.012515, -0.002503],
[20.000000, 1.570796, 0.000000, -0.966871, 0.012515, 0.002503],
[20.000000, 1.570796, 0.000000, -0.957322, 0.012392, 0.007435],
[20.000000, 1.570796, 0.000000, -0.939043, 0.012155, 0.012155],
[20.000000, 1.570796, 0.000000, -0.913487, 0.011824, 0.016554],
[20.000000, 1.570796, 0.000000, -0.889910, 0.016127, -0.016127],
[20.000000, 1.570796, 0.000000, -0.913487, 0.016554, -0.011824],
[20.000000, 1.570796, 0.000000, -0.930288, 0.016859, -0.007225],
[20.000000, 1.570796, 0.000000, -0.939043, 0.017017, -0.002431],
[20.000000, 1.570796, 0.000000, -0.939043, 0.017017, 0.002431],
[20.000000, 1.570796, 0.000000, -0.930288, 0.016859, 0.007225],
[20.000000, 1.570796, 0.000000, -0.913487, 0.016554, 0.011824],
[20.000000, 1.570796, 0.000000, -0.889910, 0.016127, 0.016127],
]

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
        return r - 2 * G * M + 0.1

    def geodesics(tau, X):
        progress = (tau - tau_span[0]) / (tau_span[1] - tau_span[0])
        bar_size = 20
        symbols = int(np.ceil(progress * 20))
        print(f"Simulation progress: [{symbols * '#'}{(bar_size - symbols) * ' '}] {progress * 100:.2f}%", end  = "\r")

        t_, dt, r, dr, theta, dtheta, phi, dphi = X

        a = r - 2 * G * M

        # theta = np.pi / 2
        # dtheta = 0
        # ddtheta = 0

        # ddt = - (2 * G * M) / (r * a) * dr * dt
        # ddr = (2 * G * M) / (r * a) * (dr ** 2) + a * (dtheta ** 2 + np.sin(theta) ** 2 * dphi ** 2) - (G * M * c ** 2) / (r * (r * c ** 2 - 2 * G * M)) * (dt ** 2)
        # ddtheta = -2 * dr * dphi / r - np.sin(theta) * np.cos(theta) * dphi * dphi
        # ddphi = -2 * dr * dphi / r - 2 * dtheta * dphi / np.tan(theta)
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

    sol = solve_ivp(geodesics, tau_span, y0, t_eval = tau_eval, events = termination_event, method = "RK45")
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
# dphi0_1 = 0.15430334
# dphi0 = 0.15430335
# dphi0_2 = 0.15430336
    
data = np.zeros((len(initial_conditions), 6), dtype = object)

data = np.loadtxt("trajectories.csv", delimiter = ",")

particle_id = data[:, 0].astype(int)
time = data[:, 1]
r = data[:, 2]
theta = data[:, 3]
phi = data[:, 4]

x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

for particle in range(64):
    steps = np.arange(particle * 20000, particle * 20000 + 20000, 1)

    rp = r[steps]
    thetap = theta[steps]
    phip = phi[steps]

    ax.plot(x, y)

for i, cond in enumerate(data):
    particle_id, step, t, r, theta, phi, dt, dr, dtheta, dphi = cond

    # x = r * np.sin(theta) * np.cos(phi)
    # y = r * np.sin(theta) * np.sin(phi)
    # z = r * np.cos(theta)

    

    # dr0 *= 5
    # dtheta0 *= 5
    # dphi0 *= 5

    # s_lambda, s_t, s_x, s_y, s_z, s_v = particle_dynamics(TIMELIKE, [r0, dr0, theta0, dtheta0, phi0, dphi0])
    # ax.plot(s_x, s_y, label = f"Timelike particle $d\\phi/d\\tau = {dphi0}$)")
    
    # data[i] = [s_lambda, s_t, s_x, s_y, s_z, s_v]

# s1_lambda, s1_t, s1_x, s1_y, s1_z, s1_v = particle_dynamics(TIMELIKE, [6 * G * M, 0, np.pi / 2, 0, 0, dphi0_1])
# ax.plot(s1_x, s1_y, label = f"Timelike particle 1 ($d\\phi/d\\tau = {dphi0}$)")

# s2_lambda, s2_t, s2_x, s2_y, s2_z, s2_v = particle_dynamics(TIMELIKE, [6 * G * M, 0, np.pi / 2, 0, 0, dphi0_2])
# ax.plot(s2_x, s2_y, label = f"Timelike particle 2 ($d\\phi/d\\tau = {dphi0}$)")
    # # ax.plot(p_x, p_y, label = f"Lightlike particle ($d\\phi/d\\tau = {dphi0}$)")

# ax.legend()

ax.plot(sphere_x, sphere_y, color = 'black', alpha = 1.0)

hole = plt.Circle((0, 0), 2 * G * M / (c ** 2), color = 'black')
ax.add_patch(hole)

# ax.legend()

plt.show()

# fig = plt.figure(figsize = (10, 7))
# fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
# ax = fig.add_subplot(111, projection = '3d')

# # ax.set_axis_off()
# ax.set_xlim([-lim, lim])
# ax.set_ylim([-lim, lim])
# ax.set_zlim([-lim, lim])
# ax.set_box_aspect([1, 1, 1])
# ax.set_clip_on(False)
# ax.set_title("Geodesic in Schwarzschild spacetime")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.view_init(elev = 45, azim = 45)
# ax.dist = 8

# ax.plot_surface(sphere_x, sphere_y, sphere_z, color = 'black', edgecolor = 'gray', alpha = 0.5)

# lines = []
# print(f"len = {len(data)}")
# for i in range(len(data)):
#     # print(data)
#     # print(i)
#     line, = ax.plot([], [], linestyle = 'solid', color = 'blue', label = "Trajectory (proper time, infaller's view)", zorder = 1)
#     lines = np.append(lines, line)

# # line1, = ax.plot([], [], linestyle = 'solid', color = 'green', label = "Trajectory 1 (proper time, infaller's view)", zorder = 2)
# # line2, = ax.plot([], [], linestyle = 'solid', color = 'red', label = "Trajectory 2 (proper time, infaller's view)", zorder = 1)

# # ax.legend()

# # info_text = ax.text2D(0.05, 0.95, '', transform = ax.transAxes)
# # info_text1 = ax.text2D(0.05, 0.90, '', transform = ax.transAxes)
# # info_text2 = ax.text2D(0.05, 0.85, '', transform = ax.transAxes)

# # observer_t = np.linspace(t[0], t[-1])
# # observer_r = interp1d(t, r, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')(observer_t)
# # observer_theta = interp1d(t, theta, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')(observer_t)
# # observer_phi = interp1d(t, phi, kind = 'cubic', bounds_error = False, fill_value = 'extrapolate')(observer_t)

# # observer_x = observer_r * np.sin(observer_theta) * np.cos(observer_phi)
# # observer_y = observer_r * np.sin(observer_theta) * np.sin(observer_phi)
# # observer_z = observer_r * np.cos(observer_theta)

# # frames = len(ts)
# def update(frame):
#     frame = 5 * frame

#     min_frame = max(0, frame - 5 * display_fps)

#     for i, data_i in enumerate(data):
#         s_lambda, s_t, s_x, s_y, s_z, s_v = data[i]

#         lines[i].set_data(s_x[min_frame:frame], s_y[min_frame:frame])
#         lines[i].set_3d_properties(s_z[min_frame:frame])

#     return lines

# animation = anim.FuncAnimation(
#     fig, update, frames = len(s_x) + display_fps * restart_timeout, interval = 1000 / display_fps, blit = True
# )

# # # from manim import *

# # # class SchwarzschildGeodesic(ThreeDScene):
# # #     def construct(self):
# # #         self.set_camera_orientation(phi = 75 * DEGREES, theta = 45 * DEGREES, distance = 8)

# # #         sphere = Sphere(radius = r_horizon, )


# plt.show()