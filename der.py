import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(ti.gpu)

#global parameters
E = 3.4e9 # Young's modulus
G = 0.365*E # twist modulus
r = 0.005
dt = 1e-6
rho = 1.3
n_rods = 1
n_vertices = 10

#strand states
total_length = 1
rest_length = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
rest_voronoi_length = ti.field(dtype=float, shape=(n_rods, n_vertices))
rest_kappa = ti.Vector.field(4, dtype=float, shape=(n_rods, n_vertices - 2))
rest_twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))

x = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices)) # position
is_fixed = ti.field(dtype=bool, shape=(n_rods, n_vertices))
v = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices)) # velocity
edge = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
length = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
old_tangent = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
tangent = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
n1_mat = ti.Matrix.field(3, 3, dtype=float, shape=(n_rods, n_vertices - 1))
n2_mat = ti.Matrix.field(3, 3, dtype=float, shape=(n_rods, n_vertices - 1))
n1_ref = ti.Matrix.field(3, 3, dtype=float, shape=(n_rods, n_vertices - 1))
n2_ref = ti.Matrix.field(3, 3, dtype=float, shape=(n_rods, n_vertices - 1))
curvature_binormal = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
kappa = ti.Vector.field(4, dtype=float, shape=(n_rods, n_vertices - 2))
grad_kappa = ti.Matrix.field(4, 3, dtype=float, shape=(n_rods, n_vertices - 2, 3))
grad_theta_kappa = ti.Vector.field(4, dtype=float, shape=(n_rods, n_vertices - 2, 2))
theta = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 2))
grad_twist = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 2, 3))
f_strech = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices))
f_bend = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices))
f_twist = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices))
tau_twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
tau_bend = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))

@ti.kernel
def compute_streching_force():
    for i, j in f_strech:
        f_strech[i, j] = [0, 0, 0]
    for i, j in edge:
        f = np.pi*r*r*E*(length[i, j]/rest_length[i, j] - 1) * tangent[i, j]
        f_strech[i, j] += f
        f_strech[i, j+1] -= f

@ti.kernel
def compute_bending_force():
    for i, j in f_bend:
        f_bend[i, j] = [0, 0, 0]
        tau_bend[i, j] = [0, 0, 0]
    for i, j in kappa:
        b = E * np.pi * r**4 / 8
        kappa_bar = rest_kappa[i, j]
        ilen = 1 / rest_voronoi_length[i, j]
        f_bend[i, j] += -b*ilen*grad_kappa[i, j, 0].transpose()@(kappa[i, j] - kappa_bar)
        f_bend[i, j+1] += -b*ilen*grad_kappa[i, j, 1].transpose()@(kappa[i, j] - kappa_bar)
        f_bend[i, j+2] += -b*ilen*grad_kappa[i, j, 2].transpose()@(kappa[i, j] - kappa_bar)
        tau_bend[i, j] += -b*ilen*tm.dot(grad_kappa[i, j, 0],(kappa[i, j] - kappa_bar))
        tau_bend[i, j+1] += -b*ilen*tm.dot(grad_kappa[i, j, 1],(kappa[i, j] - kappa_bar))

@ti.kernel
def compute_twisting_force():
    for i,j in f_twist:
        f_twist[i, j] = [0, 0, 0]
        tau_twist[i, j] = [0, 0, 0]
    for i, j in kappa:
        b = G * np.pi * r**4 / 4
        twist_bar = rest_twist[i, j]
        ilen = 1 / rest_voronoi_length[i, j]
        f_twist[i, j] += -b*ilen*grad_twist[i, j, 0]*(twist[i, j] - twist_bar)
        f_twist[i, j+1] += -b*ilen*grad_twist[i, j, 1]*(twist[i, j] - twist_bar)
        f_twist[i, j+2] += -b*ilen*grad_twist[i, j, 2]*(twist[i, j] - twist_bar)
        tau_twist[i, j] += -b*ilen*(twist[i, j] - twist_bar)
        tau_twist[i, j+1] += -b*ilen*(twist[i, j] - twist_bar)


@ti.func
def parallelTransport(n, t_pre, t_cur):
    k = tm.cross(t_pre, t_cur).normalized()
    theta = tm.atan2(tm.dot(tm.cross(t_pre, t_cur),k),tm.dot(t_pre, t_cur))
    return n*tm.cos(theta) + tm.cross(k, n)*tm.sin(theta) + k*tm.dot(k, n)*(1-tm.cos(theta))

@ti.kernel
def restore_tangents():
    for i, j in tangent:
        old_tangent[i, j] = tangent[i, j]

@ti.kernel
def update_position():
    for i, j in x:
        if not is_fixed[i, j]:
            x[i, j] += dt * v[i, j]

@ti.kernel
def update_edge_tangent_length():
    for i, j in edge:
        edge[i, j] = x[i, j+1] - x[i, j]
        tangent[i, j] = edge[i, j].normalized()
        length[i, j] = edge[i, j].norm()

@ti.kernel
def update_material_frame():
    for i, j in theta:
        n1_ref[i, j] = parallelTransport(n1_ref[i, j], old_tangent[i, j], tangent[i, j])
        n2_ref[i, j] = parallelTransport(n2_ref[i, j], old_tangent[i, j], tangent[i, j])
        n1_mat[i, j] = tm.cos(theta[i, j])*n1_ref[i, j] + tm.sin(theta[i, j])*n2_ref[i, j]
        n2_mat[i, j] = -tm.sin(theta[i, j])*n1_ref[i, j] + tm.cos(theta[i, j])*n2_ref[i, j]

@ti.kernel
def update_curvature_binormal():
    for i, j in curvature_binormal:
        t1 = tangent[i, j]
        t2 = tangent[i, j+1]
        curvature_binormal[i, j] = 2*tm.cross(t1, t2)/(1+tm.dot(t1, t2))

@ti.kernel
def update_kappa():
    for i, j in kappa:
        kb = curvature_binormal[i, j+1]
        m1e = n1_mat[i, j]
        m2e = n2_mat[i, j]
        m1f = n1_mat[i, j+1]
        m2f = n2_mat[i, j+1]
        kappa[i, j] = ti.Vector([tm.dot(m2e, kb), tm.dot(m2f, kb), -tm.dot(m1e, kb), -tm.dot(m1f, kb)])

@ti.kernel
def update_gradkappa():
    for i, j in grad_kappa:
        norm_e = length[i, j]
        norm_f = length[i, j+1]
        te = tangent[i, j]
        tf = tangent[i, j+1]
        m1e = n1_mat[i, j]
        m2e = n2_mat[i, j]
        m1f = n1_mat[i, j+1]
        m2f = n2_mat[i, j+1]
        chi = 1.0 + tm.dot(te, tf)
        if chi <= 0:
            print("chi = {}, te = {}, tf = {}".format(chi, te, tf))
            chi = 1e-12
        tilde_t = (te + tf) / chi
        k = kappa[i, j+1]
        Dkappa0De = 1.0 / norm_e * (-k[0] * tilde_t + 2*tm.cross(tf, m2e / chi))
        Dkappa0Df = 1.0 / norm_f * (-k[0] * tilde_t - 2*tm.cross(te, m2e / chi))
        Dkappa1De = 1.0 / norm_e * (-k[1] * tilde_t + 2*tm.cross(tf, m2f / chi))
        Dkappa1Df = 1.0 / norm_f * (-k[1] * tilde_t - 2*tm.cross(te, m2f / chi))
        Dkappa2De = 1.0 / norm_e * (-k[2] * tilde_t - 2*tm.cross(tf, m1e / chi))
        Dkappa2Df = 1.0 / norm_f * (-k[2] * tilde_t + 2*tm.cross(te, m1e / chi))
        Dkappa3De = 1.0 / norm_e * (-k[3] * tilde_t - 2*tm.cross(tf, m1f / chi))
        Dkappa3Df = 1.0 / norm_f * (-k[3] * tilde_t + 2*tm.cross(te, m1f / chi))
        grad_kappa[i, j, 0] = ti.Matrix([-Dkappa0De, -Dkappa1De, -Dkappa2De, -Dkappa3De])
        grad_kappa[i, j, 1] = ti.Matrix([Dkappa0De-Dkappa0Df, Dkappa1De-Dkappa1Df, Dkappa2De-Dkappa2Df, Dkappa3De-Dkappa3Df])
        grad_kappa[i, j, 2] = ti.Matrix([Dkappa0Df, Dkappa1Df, Dkappa2Df, Dkappa3Df])
        kb = curvature_binormal[i, j+1]
        grad_theta_kappa[i, j, 0] = ti.Vector([-tm.dot(kb, m1e), 0, -tm.dot(kb, m2e), 0])
        grad_theta_kappa[i, j, 1] = ti.Vector([0, -tm.dot(kb, m1f), 0, -tm.dot(kb, m2f)])

@ti.kernel
def update_twist():
    for i, j in twist:
        v1 = parallelTransport(n2_ref[i, j], old_tangent[i, j], old_tangent[i, j+1])
        v2 = n2_ref[i, j+1]
        n = tm.cross(v1, v2).normalized()
        twist[i, j] = theta[i, j+1] - theta[i, j] + tm.atan2(tm.dot(n, old_tangent[i, j+1]),tm.dot(v1, v2))

@ti.kernel
def update_gradtwist():
    for i, j in grad_twist:
        kb = curvature_binormal[i, j+1]
        grad_twist[i, j, 0] = -0.5 / length[i, j] * kb
        grad_twist[i, j, 2] = 0.5 / length[i, j+1] * kb
        grad_twist[i, j, 1] = -(grad_twist[i, j, 0] + grad_twist[i, j, 2])

@ti.kernel
def update_velocity():
    for i, j in v:
        mass = rho * total_length / n_vertices #TODO: how to set vertices' mass?
        force = f_strech[i, j] + f_bend[i, j] + f_twist[i, j]
        v[i, j] += dt * force / mass

@ti.kernel
def update_theta():
    for i, j in theta:
        theta[i, j] += dt * (tau_bend[i, j] + tau_twist[i, j]) / (0.5 * rho * length[i, j] * r**2)

def explicit_integrator():
    restore_tangents()
    update_position()
    update_edge_tangent_length()
    update_material_frame()
    update_curvature_binormal()
    update_kappa()
    update_gradkappa()
    update_twist()
    update_gradtwist()
    compute_streching_force()
    compute_bending_force()
    compute_twisting_force()
    update_velocity()
    update_theta()

def semiimplicit_integrator():
    pass

def implicit_integrator():
    pass

def initialize():
    for i in ti.static(range(n_rods)):
        for j in ti.static(range(n_vertices)):
            pass

if __name__=="__main__":
    initialize()
    window = ti.ui.Window("Hair DER", (512, 512), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    while window.running:
        # for s in range(int(2e-3 // dt)):
        explicit_integrator()
        camera.position(0.0, 0.0, 3)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(x, radius=r, color=(0.5, 0.42, 0.8))
        scene.lines(x, width=2*r, color=(0.5, 0.42, 0.8))
        canvas.scene(scene)
        window.show()