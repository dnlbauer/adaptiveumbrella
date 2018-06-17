import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('..')
from  adaptiveumbrella.runner import UmbrellaRunner


class WHAM2DRunner(UmbrellaRunner):
    """ Umbrella runner implementation that uses wham-2d to perform 
    the pmf calculation.

    Attributes:
        WHAM_EXEC: path to wham executeable

        """

    def __init__(self):
        UmbrellaRunner.__init__(self)
        self.WHAM_EXEC = 'wham-2d'

    def calculate_new_pmf(self):
        import os
        from shutil import copyfile

        simulation_dir = "tmp/simulations"
        print("Collecting sampling data from simulations folder")

        # collect COLVARs
        wham_dir = "tmp/WHAM/"
        if not os.path.exists(wham_dir):
            os.makedirs(wham_dir)

        for folder in os.listdir(simulation_dir):
            src = os.path.join(simulation_dir, folder, "COLVAR")
            dst = os.path.join(wham_dir, folder + ".xvg")
            copyfile(src, dst)

        # create metadata file
        metadata_file = os.path.join(wham_dir, "{}_metadata.dat".format(self.num_iterations))
        fc_x = 100
        fc_y = 100
        with open(metadata_file, 'w') as out:
            for f in os.listdir(simulation_dir):
                prefix, x, y = f.split("_")
                out.write("{}/{}.xvg {} {} {} {}\n".format(wham_dir, f, x, y, fc_x, fc_y))

        # run WHAM2d
        print("Running WHAM-2d")
        wham_output = os.path.join(wham_dir, "{}_freeenergy.dat".format(self.num_iterations))
        periodicity_x = "pi"
        periodicity_y = "pi"
        tolerance = 0.1
        frames_x, frames_y = 1002, 1002
        min_x = self.cvs[0][0]
        max_x = self.cvs[0][1]
        min_y = self.cvs[1][0]
        max_y = self.cvs[1][1]

        cmd = "{exec} Px={px} {min_x} {max_x} {frames_x} Py={py} {min_y} {max_y} {frames_y} {tol} 298 0 {metafile} {outfile} 0".format(
            exec=self.WHAM_EXEC,
            px=periodicity_x,
            min_x=min_x,
            max_x=max_x,
            frames_x=frames_x,
            py=periodicity_y,
            min_y=min_y,
            max_y=max_y,
            frames_y=frames_y,
            tol=tolerance,
            metafile=metadata_file,
            outfile=wham_output
        )
        print(cmd)
        os.system(cmd)

        # read wham to new pmf
        return self.read_pmf(wham_output)

    def read_pmf(self, pmf_path):
        import pandas as pd
        print("Update PMF from WHAM")
        df = pd.read_csv(pmf_path, delim_whitespace=True, names=['x', 'y', 'e', 'pro'], skiprows=1,
                         index_col=None)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['e'], how='all')
        new_pmf = deepcopy(self.pmf)
        for x in range(new_pmf.shape[0]):
            for y in range(new_pmf.shape[1]):
                lambdax, lambday = self._get_lambdas_for_index((x, y))
                x_selection = (df.x - lambdax).abs() < 0.01
                y_selection = (df.y - lambday).abs() < 0.01
                selected_energies = df[(x_selection) & (y_selection)].e
                if len(selected_energies) == 0:
                    new_pmf[x, y] = -1
                else:
                    new_pmf[x, y] = selected_energies.iloc[0]

        return new_pmf


class MyUmbrellaRunner(WHAM2DRunner):
    def after_run_hook(self):
        filename = "tmp/pmf_{}.pdf".format(self.num_iterations)
        print("Writing new pmf to {}".format(filename))
        pmf_to_plot = deepcopy(self.pmf.T)
        pmf_to_plot[pmf_to_plot < 0] = None
        frames_to_plot = deepcopy(self.sample_list.T)
        frames_to_plot[frames_to_plot == 0] = None

        fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True)
        im = ax0.imshow(pmf_to_plot, origin="bottom", cmap='jet')
        cb = fig.colorbar(im, ax=ax0)
        cb.set_label("kJ/mol")
        im2 = ax1.imshow(frames_to_plot, origin="bottom")
        cb2 = fig.colorbar(im2, ax=ax1, ticks=np.arange(0, self.max_iterations))
        cb.set_label("cycle")
        # ticks = [(x,x) for x in [-3, -2, -1, 0, 1, 2, 3]]
        # tick_positions = [ self._get_index_for_lambdas(x)[0] for x in ticks ]
        # tick_labels = [ str(x[0]) for x in ticks ]

        # ax1.set_xticks(tick_positions, tick_labels)
        # ax1.set_yticks(tick_positions, tick_labels)
        # ax2.set_xticks(tick_positions, tick_labels)
        # ax2.set_yticks(tick_positions, tick_labels)

        plt.savefig(filename)
        os.system("cp {} {}".format(filename, "tmp/pmf_current.pdf"))


    def simulate_frames(self, lambdas, frames):
        print("{} new simulations:".format(len(lambdas)))
        counter = 0

        if not os.path.exists("tmp"):
            os.mkdir('tmp')

        threads = []
        for f in lambdas:
            counter += 1
            print("{}) Simulate lambda1={}, lambda2={}".format(counter, *f))
            command = "bash data/sim.sh {} {} 2>&1 > tmp/run.log".format(*f)
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
runner.max_iterations = 1

runner.run()
