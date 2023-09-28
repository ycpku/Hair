import taichi as ti
import taichi.math as tm
import numpy as np
import sys
import argparse
import xml.etree.ElementTree as ET

# TODO: encapsulate

ti.init(ti.gpu)

#global parameters
E = 3.9e10 # Young's modulus
G = 3.4e9 # shear modulus
r = 0.0037
dt = 1e-6
rho = 1.32
n_rods = 1
n_vertices = 41

#strand states
rest_length = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
rest_voronoi_length = ti.field(dtype=float, shape=(n_rods, n_vertices))
rest_kappa = ti.Vector.field(4, dtype=float, shape=(n_rods, n_vertices - 2))
rest_twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 2))
ref_twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 2))

x = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices)) # position
is_fixed = ti.field(dtype=bool, shape=(n_rods, n_vertices))
v = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices)) # velocity
edge = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
length = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
old_tangent = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
tangent = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
n1_mat = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
n2_mat = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
n1_ref = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
n2_ref = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
curvature_binormal = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 2))
kappa = ti.Vector.field(4, dtype=float, shape=(n_rods, n_vertices - 2))
grad_kappa = ti.Matrix.field(4, 3, dtype=float, shape=(n_rods, n_vertices - 2, 3))
grad_theta_kappa = ti.Vector.field(4, dtype=float, shape=(n_rods, n_vertices - 2, 2))
theta = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
omega = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 2))
grad_twist = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 2, 3))
f_strech = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices))
f_bend = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices))
f_twist = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices))
tau_twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
tau_bend = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))

#visualization
vertices = ti.Vector.field(3, dtype=float, shape=n_rods*n_vertices)
indices = ti.field(dtype=int, shape=2*(n_vertices-1))

@ti.kernel
def compute_streching_force():
    for i, j in f_strech:
        f_strech[i, j] = [0, 0, 0]
    for i, j in length:
        f = np.pi* r**2 * E *(length[i, j]/rest_length[i, j] - 1) * tangent[i, j]
        f_strech[i, j] += f
        f_strech[i, j+1] -= f

@ti.kernel
def compute_bending_force():
    for i, j in f_bend:
        f_bend[i, j] = [0, 0, 0]
    for i, j in tau_bend:
        tau_bend[i, j] = 0
    for i, j in kappa:
        b = E * np.pi * r**4 / 8
        kappa_bar = rest_kappa[i, j]
        ilen = 1 / rest_voronoi_length[i, j+1]
        f_bend[i, j] += -b*ilen*grad_kappa[i, j, 0].transpose()@(kappa[i, j] - kappa_bar)
        f_bend[i, j+1] += -b*ilen*grad_kappa[i, j, 1].transpose()@(kappa[i, j] - kappa_bar)
        f_bend[i, j+2] += -b*ilen*grad_kappa[i, j, 2].transpose()@(kappa[i, j] - kappa_bar)
        tau_bend[i, j] += -b*ilen * grad_theta_kappa[i, j, 0].dot(kappa[i, j] - kappa_bar)
        tau_bend[i, j+1] += -b*ilen * grad_theta_kappa[i, j, 1].dot(kappa[i, j] - kappa_bar)

@ti.kernel
def compute_twisting_force():
    for i, j in f_twist:
        f_twist[i, j] = [0, 0, 0]
    for i, j in tau_twist:
        tau_twist[i, j] = 0
    for i, j in twist:
        b = G * np.pi * r**4 / 4
        twist_bar = rest_twist[i, j]
        ilen = 1 / rest_voronoi_length[i, j+1]
        f_twist[i, j] += -b*ilen*grad_twist[i, j, 0]*(twist[i, j] - twist_bar)
        f_twist[i, j+1] += -b*ilen*grad_twist[i, j, 1]*(twist[i, j] - twist_bar)
        f_twist[i, j+2] += -b*ilen*grad_twist[i, j, 2]*(twist[i, j] - twist_bar)
        tau_twist[i, j] += b*ilen*(twist[i, j] - twist_bar)
        tau_twist[i, j+1] += -b*ilen*(twist[i, j] - twist_bar)

@ti.func
def parallelTransport(n, t0, t1):
    b = t0.cross(t1)
    ret = ti.Vector([0., 0., 0.])
    if(b.norm()<sys.float_info.epsilon):
        ret = n
    else :
        b = b.normalized()
        n0 = t0.cross(b)
        n1 = t1.cross(b)
        ret = n.dot(n0) * n1 + n.dot(b) * b
    return ret

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
def update_theta():
    for i, j in theta:
        theta[i, j] += dt * omega[i, j]

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
        curvature_binormal[i, j] = 2*t1.cross(t2)/(1+t1.dot(t2))

@ti.kernel
def update_kappa():
    for i, j in kappa:
        kb = curvature_binormal[i, j]
        m1e = n1_mat[i, j]
        m2e = n2_mat[i, j]
        m1f = n1_mat[i, j+1]
        m2f = n2_mat[i, j+1]
        kappa[i, j] = ti.Vector([m2e.dot(kb), m2f.dot(kb), -m1e.dot(kb), -m1f.dot(kb)])

@ti.kernel
def update_gradkappa():
    for i, j in kappa:
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
        k = kappa[i, j]
        Dkappa0De = 1.0 / norm_e * (-k[0] * tilde_t + 2*tm.cross(tf, m2e / chi))
        Dkappa0Df = 1.0 / norm_f * (-k[0] * tilde_t - 2*tm.cross(te, m2e / chi))
        Dkappa1De = 1.0 / norm_e * (-k[1] * tilde_t + 2*tm.cross(tf, m2f / chi))
        Dkappa1Df = 1.0 / norm_f * (-k[1] * tilde_t - 2*tm.cross(te, m2f / chi))
        Dkappa2De = 1.0 / norm_e * (-k[2] * tilde_t - 2*tm.cross(tf, m1e / chi))
        Dkappa2Df = 1.0 / norm_f * (-k[2] * tilde_t + 2*tm.cross(te, m1e / chi))
        Dkappa3De = 1.0 / norm_e * (-k[3] * tilde_t - 2*tm.cross(tf, m1f / chi))
        Dkappa3Df = 1.0 / norm_f * (-k[3] * tilde_t + 2*tm.cross(te, m1f / chi))
        grad_kappa[i, j, 0] = ti.Matrix.rows([-Dkappa0De, -Dkappa1De, -Dkappa2De, -Dkappa3De])
        grad_kappa[i, j, 1] = ti.Matrix.rows([Dkappa0De-Dkappa0Df, Dkappa1De-Dkappa1Df, Dkappa2De-Dkappa2Df, Dkappa3De-Dkappa3Df])
        grad_kappa[i, j, 2] = ti.Matrix.rows([Dkappa0Df, Dkappa1Df, Dkappa2Df, Dkappa3Df])
        kb = curvature_binormal[i, j]
        grad_theta_kappa[i, j, 0] = ti.Vector([0, -tm.dot(kb, m1e), 0, -tm.dot(kb, m2e)])
        grad_theta_kappa[i, j, 1] = ti.Vector([-tm.dot(kb, m1f), 0, -tm.dot(kb, m2f), 0])

@ti.func
def signed_angle(u, v, n):
    w = u.cross(v)
    angle = tm.atan2(w.norm(), u.dot(v))
    ret = angle
    if n.dot(w) < 0:
        ret = -angle
    return ret

@ti.func
def rotateAxisAngle(v, z, theta):
    c = tm.cos(theta)
    s = tm.sin(theta)
    v = c * v + s * z.cross(v) + z.dot(v) * (1.0 - c) * z

@ti.kernel
def update_twist():
    for i, j in twist:
        v1 = parallelTransport(n1_ref[i, j], tangent[i, j], tangent[i, j+1])
        v2 = n1_ref[i, j+1]
        before_twist = ref_twist[i, j]
        rotateAxisAngle(v1, tangent[i, j+1], before_twist)
        ref_twist[i, j] = before_twist + signed_angle(v1, v2, tangent[i, j])
        twist[i, j] = theta[i, j+1] - theta[i, j] + ref_twist[i, j]

@ti.kernel
def update_gradtwist():
    for i, j in kappa:
        kb = curvature_binormal[i, j]
        grad_twist[i, j, 0] = -0.5 / length[i, j] * kb
        grad_twist[i, j, 2] = 0.5 / length[i, j+1] * kb
        grad_twist[i, j, 1] = -(grad_twist[i, j, 0] + grad_twist[i, j, 2])

@ti.kernel
def update_velocity():
    for i, j in v:
        mass = rho * np.pi * r**2 * rest_voronoi_length[i, j]
        force = f_strech[i, j] + f_bend[i, j] + f_twist[i, j]
        v[i, j] += dt * force / mass + dt * ti.Vector([0, -981, 0])

@ti.kernel
def update_omega():
    for i, j in omega:
        mass = rho * np.pi * r**2 * length[i, j]
        omega[i, j] += dt * (tau_bend[i, j] + tau_twist[i, j]) / (0.5 * mass * r**2)

def explicit_integrator():
    restore_tangents()
    update_position()
    update_theta()
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
    update_omega()

@ti.kernel
def init_reference_frame():
    for i, j in n1_ref:
        n1_ref[i, j] = ti.Vector([-tangent[i, j][1], tangent[i, j][0], 0])
        n2_ref[i, j] = tm.cross(tangent[i, j], n1_ref[i, j])
        n1_mat[i, j] = n1_ref[i, j]
        n2_mat[i, j] = n2_ref[i, j]

def initialize():
    for i in ti.static(range(2*(n_vertices-1))):
        indices[i] = (i+1)//2
    # for i in ti.static(range(n_rods)):
    #     for j in ti.static(range(n_vertices)):
    #         x[i, j] = ti.Vector([j/10, 0., 0.])
    # is_fixed[0, 0] = 1
    update_edge_tangent_length()
    for i in ti.static(range(n_rods)):
        for j in ti.static(range(n_vertices - 1)):
            rest_length[i, j] = length[i, j]
            rest_voronoi_length[i, j] += length[i, j] / 2
            rest_voronoi_length[i, j+1] += length[i, j] / 2
    init_reference_frame()
    update_curvature_binormal()
    for i in ti.static(range(n_rods)):
        for j in ti.static(range(n_vertices - 1)):
            rest_twist[i, j] = 0
    update_kappa()
    for i in ti.static(range(n_rods)):
        for j in ti.static(range(n_vertices - 2)):
            rest_kappa[i, j] = kappa[i, j]

@ti.kernel
def update_vertices():
    for i, j in x:
        vertices[i * n_rods + j] = x[i, j]

def initialize_strand(strand, i):
    j = 0
    for p in strand:
        assert(p.tag=='particle')
        x[i, j] = ti.Vector([float(x) for x in p.attrib['x'].split()])
        if p.attrib.get('fixed','0') == '1':
            is_fixed[i, j] = 1
        j += 1

def load_scene(path):
    tree = ET.parse(path)
    root = tree.getroot()
    i = 0
    for child in root:
        if child.tag=='Strand':
            initialize_strand(child, i)
            i += 1

def write_to_file(outfile, frame):
    outfile.write('------frame {}-----\n'.format(frame))
    outfile.write('position:\n{}\n'.format(x))
    outfile.write('velocity:\n{}\n'.format(v))
    outfile.write('theta:\n{}\n'.format(theta))
    outfile.write('twist:\n{}\n'.format(twist))
    outfile.write('streching force:\n{}\n'.format(f_strech))
    outfile.write('bending force:\n{}\n'.format(f_bend))
    outfile.write('twisting force:\n{}\n'.format(f_twist))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scene', type=str, help='input xml file')
    parser.add_argument('-o', '--outfile', type=str)
    args = parser.parse_args()
    load_scene(args.scene)
    initialize()
    window = ti.ui.Window("Hair DER", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, -0.25, 5)
    camera.lookat(0, -0.25, 0)
    frames = 0
    file = open('outfile.txt', 'w')
    while window.running and frames < 20:
        for _ in range(int(1e-6//dt)):
            explicit_integrator()
        frames+=1
        write_to_file(file, frames)
        update_vertices()
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(vertices, radius=r, color=(0, 0, 0))
        scene.lines(vertices, width=4, indices=indices, color=(0, 0, 0)) #TODO: multiple strands
        canvas.scene(scene)
        # window.save_image('output/{}.png'.format(frames))
        window.show()
    file.close()