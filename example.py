from adaptive_umbrella import WHAM2DRunner

from copy import deepcopy
import os
import numpy as np
import matplotlib.pyplot as plt


class MyUmbrellaRunner(WHAM2DRunner):
    def after_run_hook(self):
        filename = "pmf_{}.pdf".format(self.num_iterations)
        print("Writing new pmf to {}".format(filename))
        pmf_to_plot = deepcopy(self.pmf.T)
        pmf_to_plot[pmf_to_plot < 0] = None
        plt.figure()
        plt.imshow(pmf_to_plot, origin="lower", cmap='jet')
        cb = plt.colorbar(pad=0.1)
        cb.set_label("kJ/mol")
        plt.savefig(filename)
        os.system("cp {} {}".format(filename, "pmf_current.pdf"))

    def simulate_frames(self, lambdas, frames):
        print("{} new simulations:".format(len(lambdas)))
        counter = 0

        threads = []
        for f in lambdas:
            counter += 1
            if os.path.exists("sim/sim_{}_{}/COLVAR".format(*f)):
                print("{}) Skipping lambdas={}/{}: COLVAR exists".format(counter, *f))
                continue

            print("{}) Simulate lambda1={}, lambda2={}".format(counter, *f))
            command = "bash sim.sh {} {} 2>&1 > run.log".format(*f)
            # print("Running {}".format(command))
            os.system(command)


runner = MyUmbrellaRunner()
runner.WHAM_EXEC = "/opt/wham/wham-2d/wham-2d"
runner.cvs = np.array([
    (-3.1, 3.1, 0.1),
    (-3.1, 3.1, 0.1),
])
runner.cvs_init = (1, -1.4)
runner.E_min = 10
runner.E_max = 200
runner.E_incr = 10
runner.max_iterations = 25

runner.run()
