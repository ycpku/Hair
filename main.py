import taichi as ti
import argparse
from scene import Scene
from der import Simulator
import time

if __name__=="__main__":
    total0 = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scene', type=str, help='XML scene file to load')
    parser.add_argument('-o', '--outfile', type=str, help='Readable file to save simulation state to')
    parser.add_argument('-d', '--display', type=bool, help='Run the simulation with rendering enabled or disabled')
    parser.add_argument('-g', '--generate', type=bool, help='Generate PNG or not')

    arch, default_fp = ti.cpu, ti.f64
    # arch, default_fp = ti.gpu, ti.f32
    ti.init(arch=arch, default_fp=default_fp, debug=False, kernel_profiler=True)
    args = parser.parse_args()

    t1 = time.perf_counter()

    window = ti.ui.Window("Hair DER", (1024, 1024), vsync=True)

    scene = Scene(window.get_scene())
    scene.load_scene(args.scene)
    scene.initialize()

    time_step = 1e-3
    framerate = 1000
    duration = 1

    sim = Simulator(scene.n_rods, scene.n_vertices, scene.params, time_step, default_fp)
    sim.initialize(scene.x, scene.is_fixed, scene.v)

    t2 = time.perf_counter()
    print(f"Initialization time: {t2-t1:.6f} seconds") # time for initialization

    frames = 0
    file = open('outfile.txt', 'w')
    sim.write_to_file(file, frames)

    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    while window.running and frames < duration*framerate:
        for _ in range(int(1e-3//time_step)):
            # sim.explicit_integrator()
            sim.semi_implicit_integrator()
        frames+=1
        sim.write_to_file(file, frames)

        scene.update(sim.x)

        canvas.scene(scene.scene)
        if args.generate:
            window.save_image('output/{}.png'.format(frames))
        window.show()
    # ti.profiler.print_scoped_profiler_info()
    if arch==ti.gpu:
        ti.sync()
    ti.profiler.print_kernel_profiler_info()
    print(f"Matrix solving time: {sim.t_solve:.6f} seconds") # time for solving matrix
    file.close()

    total1 = time.perf_counter()
    print(f"Total execution time: {sim.t_solve:.6f} seconds")