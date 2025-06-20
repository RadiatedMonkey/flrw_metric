import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.integrate import solve_ivp

def radius(t):
    return 2 * np.sin(t / 10)
    # return 1

def dradius(t):
    return 2 * np.cos(t / 10) / 10
    # return 0

simulation_fps = 500
display_fps = 30
subsample_factor = simulation_fps // display_fps
tau_span = (0, 10)
tau_eval = np.linspace(*tau_span, simulation_fps * int(np.ceil(tau_span[1] - tau_span[0])))

t0 = 0
theta0 = 0.01
phi0 = 0
dtheta0 = 5
dphi0 = 0

# Computes the initial conditions required for the geodesic
def initial_time(variant, t0, theta0, dtheta0):
    a2 = radius(t0) ** 2
    return np.sqrt(-variant + a2 * dtheta0 ** 2 + a2 * np.sin(theta0) ** 2 * dphi0 ** 2)

y0 = [1, initial_time(-1, t0, theta0, dtheta0), theta0, dtheta0, phi0, dphi0]
print(y0)

def geodesics(tau, X):
    try:
        t, dt, theta, dtheta, phi, dphi = X

        a = radius(t)
        if a == 0:
            return np.zeros_like(X)

        da = dradius(t)

        epsilon = 1e-8
        tan_theta = np.tan(theta) if np.abs(np.cos(theta)) > epsilon else epsilon

        ddt = -a * da * ((dtheta ** 2) + (np.sin(theta) ** 2) * (dphi ** 2))
        ddtheta = np.sin(theta) * np.cos(theta) * (dphi ** 2) - (2 * da) / a * dt * dtheta
        ddphi = -(2 * dtheta * dphi) / tan_theta - (2 * da) / a * dt * dphi

        return [
            dt,
            ddt,
            dtheta,
            ddtheta,
            dphi,
            ddphi
        ]
    except Exception as e:
        print(f"Error at tau={tau}: {e}")
        return np.zeros_like(X)

sol = solve_ivp(geodesics, tau_span, y0, t_eval = tau_eval, method = 'Radau')
print(f"Status: {sol.status}, message: {sol.message}")

ts = np.arange(0, len(sol.t), subsample_factor)

tau = sol.t[ts]
t = sol.y[0][ts]
dt = sol.y[1][ts]
theta = sol.y[2][ts]
dtheta = sol.y[3][ts]
phi = sol.y[4][ts]
dphi = sol.y[5][ts]

print(f"{len(tau)} timesteps in solution")

a = radius(t)
x = a * np.sin(theta) * np.cos(phi)
y = a * np.sin(theta) * np.sin(phi)
z = a * np.cos(theta)

velocity = np.sqrt((a * dtheta) ** 2 + (a * np.sin(theta) * dphi) ** 2)
dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

assert not np.any(np.isnan(x + y + z)), "NaN detected"

fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111, projection = '3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1, 1, 1])
ax.set_title("Geodesic in closed 2+1 FLRW universe")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.plot(x, y, z, label = "Geodesic path")
plt.show()

fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111, projection = '3d')

line, = ax.plot([], [], 'r-', label = "Geodesic")
point, = ax.plot([], [], 'ro')

time_text = ax.text2D(0.05, 0.95, '', transform = ax.transAxes)
velocity_text = ax.text2D(0.05, 0.90, '', transform = ax.transAxes)

# ax.plot_surface(sphere_x, sphere_y, sphere_z, color='lightblue', edgecolor='gray', alpha=0.7)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1, 1, 1])
ax.set_title("Geodesic in closed 2+1 FLRW universe")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# ax.legend()

# sphere = [None]

def update(frame):
    # print(dist[frame])
    # print(dist[frame], tau[frame] / 10 + 3)

    min_frame = int(np.clip(frame - 1000, 0, np.inf))

    line.set_data(x[min_frame:frame], y[min_frame:frame])
    line.set_3d_properties(z[min_frame:frame])

    # sphere_theta = np.linspace(0, 2 * np.pi, 25)
    # sphere_phi = np.linspace(0, np.pi, 25)
    # sphere_theta, sphere_phi = np.meshgrid(sphere_theta, sphere_phi)

    # t_0 = t[frame]
    # a_0 = radius(t_0)
    # sphere_x = a_0 * np.sin(sphere_theta) * np.cos(sphere_phi)
    # sphere_y = a_0 * np.sin(sphere_theta) * np.cos(sphere_phi)
    # sphere_z = a_0 * np.cos(sphere_theta)

    # sphere = ax.plot_surface(
    #     sphere_x, sphere_y, sphere_z,
    #     color = 'lightblue', alpha = 0.3, edgecolor = 'k', linewidth = 0.2
    # )

    time_text.set_text(f"Proper time (tau): {tau[frame]:.2f}, observer time: {t[frame] - t[0]:.2f}")
    velocity_text.set_text(f"Velocity: {velocity[frame]:.2f}")

    return line, point, time_text, velocity_text#, sphere

anim = FuncAnimation(
    fig, update, 
    frames = len(ts), 
    interval = 1000 / display_fps, 
    blit = True
)
plt.show()