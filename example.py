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
        plt.imshow(pmf_to_plot, origin="bottom", cmap='jet')
        ticks = [(x,x) for x in [-3, -2, -1, 0, 1, 2, 3]]
        tick_positions = [ self._get_index_for_lambdas(x)[0] for x in ticks ]
        tick_labels = [ str(x[0]) for x in ticks ]
        
        plt.xticks(tick_positions, tick_labels)
        plt.yticks(tick_positions, tick_labels)
        

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
    (-3, 3, 0.2),
    (-3, 3, 0.2),
])
runner.cvs_init = (1.4, -1.4)
runner.E_min = 10
runner.E_max = 100
runner.E_incr = 10
runner.max_iterations = 100

runner.run()
