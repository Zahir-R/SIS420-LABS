import numpy as np
import matplotlib.pyplot as plt

muestras = 1000
theta0_r = 10.0
theta1_r = 2.5
ruido_scale = 45.0

np.random.seed(67)
x = np.random.uniform(10, 100, muestras)
ruido = np.random.normal(0.0, ruido_scale, muestras)
y = theta0_r + theta1_r * x + ruido

# cm barba maago
# x = [
#     54, 58, 41, 92, 32, 27, 33, 15, 53, 98,
#     35, 78, 32, 17, 24, 55, 29, 85, 35, 45,
#     28, 59, 37, 51, 28, 53, 30, 25, 36, 21,
#     64, 52, 86, 39, 21, 20, 97, 24, 70, 25,
#     40, 38, 33, 28, 33, 46, 74, 56, 58, 26,
#     25, 75, 18, 72, 35, 42, 42, 67, 44, 22,
#     54, 26, 71, 27, 25, 79, 47, 29, 43, 25,
#     42, 16, 24, 49, 31, 25, 17, 28, 30, 43,
#     66, 41, 84, 22, 74, 55, 84, 14, 65, 45,
#     52, 32, 89, 44, 90, 88, 77, 20, 53, 28
# ]
#
# # Dannio magico mago
# y = [
#     289, 408, 198, 451, 102, 348, 240, 429, 155, 485,
#     115, 439, 321, 222, 378, 278, 181, 418, 472, 230,
#     130, 355, 258, 412, 310, 201, 448, 162, 390,  95,
#     332, 284, 435, 225, 366, 141, 480, 251, 403, 208,
#     318, 175, 465, 263, 375, 122, 422, 301, 219, 344,
#     148, 442, 275, 392, 187, 468, 239, 351, 110, 416,
#     312, 212, 458, 169, 382, 269, 433, 134, 330, 292,
#     192, 477, 245, 401, 143, 427, 306, 221, 360, 108,
#     339, 253, 460, 167, 397, 287, 444, 128, 323, 233,
#     371, 199, 491, 156, 407, 261, 347,  92, 420, 309
# ]
#

def quad_err(t0, t1):
    err = 0.0
    for xi, yi in zip(x, y):
        inf = t0 + t1 * xi
        err += (yi - inf) ** 2
    return err


def get_init_bounds(x_vals, y_vals):
    x_arr, y_arr = np.array(x_vals), np.array(y_vals)

    # y = mx + b -> m_est = frac{y_max - y_min}{x_max - x_min}
    range_x = np.ptp(x_arr)
    range_y = np.ptp(y_arr)
    m_est = range_y / range_x

    t1_low, t1_high = -2 * m_est, 2 * m_est

    # y = t0 + t1 * x -> t0 = y - t1 * x
    mean_x, mean_y = np.mean(x_arr), np.mean(y_arr)
    t0_est1 = mean_y - t1_low * mean_x
    t0_est2 = mean_y - t1_high * mean_x

    t0_low = min(t0_est1, t0_est2)
    t0_high = max(t0_est1, t0_est2)

    t0_margin = (t0_high - t0_low) * 0.5
    t0_low -= t0_margin
    t0_high += t0_margin

    return t0_low, t0_high, t1_low, t1_high


def get_best_t(t0_low, t0_high, t1_low, t1_high, steps):
    best_t0, best_t1 = 0.0, 0.0
    min_error = float('inf')

    range_t0 = [t0_low + i * (t0_high - t0_low) / steps for i in range(steps)]
    range_t1 = [t1_low + i * (t1_high - t1_low) / steps for i in range(steps)]

    for t0_test in range_t0:
        for t1_test in range_t1:
            err = quad_err(t0_test, t1_test)
            if err < min_error:
                min_error = err
                best_t0 = t0_test
                best_t1 = t1_test

    return best_t0, best_t1


def search_until_best(x_vals, y_vals, phases=3, steps=100):
    t0_low, t0_high, t1_low, t1_high = get_init_bounds(x_vals, y_vals)

    best_t0, best_t1 = 0.0, 0.0
    for _ in range(phases):
        best_t0, best_t1 = get_best_t(t0_low, t0_high, t1_low, t1_high, steps)

        t0_radius = (t0_high - t0_low) * 0.1
        t1_radius = (t1_high - t1_low) * 0.1

        t0_low, t0_high = best_t0 - t0_radius, best_t0 + t0_radius
        t1_low, t1_high = best_t1 - t1_radius, best_t1 + t1_radius

    return best_t0, best_t1


theta0, theta1 = search_until_best(x, y)


def pred_dmg(cm_barba) -> float:
    return theta0 + theta1 * cm_barba


cm = int(input("Tamannio de la barba del mago: "))
infer = pred_dmg(cm)

print(f"Theta 0 = {theta0:.2f}") 
print(f"Theta 1 = {theta1:.2f}")
print(f"Dannio inferido: {infer:.2f}")

plt.figure(figsize=(9,6))
plt.scatter(x, y, color='blue', alpha=0.6, label='Datos reales (Barba vs Dannio)')

x_line = np.linspace(min(x), max(x), 200)
y_line = theta0 + theta1 * x_line

plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Recta inferida: y = {theta0:.2f} + {theta1:.2f}x')
plt.title('Relacion entre Tamannio de Barba y Dannio Magico')
plt.xlabel('Tamannio de barba del mago (cm)')
plt.ylabel('Dannio magico del mago')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

