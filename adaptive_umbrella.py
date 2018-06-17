#!/usr/bin/env python3
import numpy as np
import unittest
from copy import deepcopy

#%%
class UmbrellaRunner():
    """
    Runner class for adaptive umbrella sampling
    
    Attributes:
        cvs (numpy array): A multidimensional array of all collective variables (cvs) in the form [cv_min, cv_max, cv_delta]
         Example: to sample a 2 dimensional grid where cv1 and cv2 go from 0 to 3 with stepsize 0.5, use 
         np.array([[0,3,1],[0,3,1]])
        cvs_init (tuple): Starting coordinates for umbrella sampling. Must be a tuple with the same dimension as cvs
        E_min (float, default=0): Starting energy for exporative umbrella sampling
        E_max (float, default=inf): Final energy. Umbrella sampling is stopped if no frames with E < E_max are found
        E_incr (float, default=1): E_min is incremented by this until E_max is reached
        max_iterations (int, default=-1): Max. number of iterations before umbrella sampling stops. -1 for infinite sampling
    
    """

    def __init__(self):
        self.max_iterations = -1
        self.E_min = 0
        self.E_max = np.inf
        self.E_incr = 1

    def _get_pmf_shape(self):
        """ returns the shape of the pmf according to the cvs """
        shape = []
        for dimen in self.cvs:
            windows = np.arange(*dimen)
            size = len(windows)
            if windows[-1] % dimen[-1] == 0:
                size += 1
            shape.append(size)
        return shape


    def _init_pmf(self):
        """ returns an empty matrix where each dimension equals the number of frames along the corresponding reaction
        coordinate """
        shape = self._get_pmf_shape()
        pmf = np.empty(shape)
        pmf.fill(-1)
        return pmf

        
    def _get_lambdas_for_index(self, idx):
        """ takes a coordinate tuple of the pmf and returns corresponding lambda values """

        lambdas = self.cvs.T[0] + idx * self.cvs.T[2]
        return tuple(np.round(lambdas, 10))

    def _get_index_for_lambdas(self, lambdas):
        """ takes a lambda tuple and returns corresponding indexes of the pmf 
        TODO: faster implementation required
        """
        idx = []
        for dimen in range(len(lambdas)):
            cv = self.cvs[dimen]
            r = np.arange(cv[0], cv[1]+cv[2], cv[2])
        
            for i in range(len(r)):
                if abs(r[i]-lambdas[dimen]) < 0.00001:
                    idx.append(i)
                    break
        if len(idx) == len(lambdas):
            return tuple(idx)
        else: # if len differs, theres no index for every dimension
            raise ValueError("{} has no index.".format(lambdas))

    def _get_root_frames(self, pmf, E_max):
        """ returns the index of all positions in the pmf where the energy is
        smaller E_max"""
        selection = np.where((pmf <= E_max) & (pmf >= 0))
        zipped = list(zip(*selection))

        return zipped

    def _generate_neighbor_list(self, root):
        """ builds a list of all direct neighbors of the root coordinate """
        dimens = len(root)
        coords = []
        for i in range(dimens):
            if len(coords) == 0:
                new_coords = []
                new_coords.append([root[i]-1])
                new_coords.append([root[i]])
                new_coords.append([root[i]+1])
                coords = new_coords
            else:
                new_coords = []
                for coord in coords:
                    new_coords.append(coord + [root[i] - 1])
                    new_coords.append(coord + [root[i]])
                    new_coords.append(coord + [root[i] + 1])
                coords = new_coords
        return [tuple(x) for x in coords]


    def _is_in_pmf(self, frame):
        num_dimens = len(self.pmf.shape)
        for dimen in range(num_dimens):
            if frame[dimen] < 0 or frame[dimen] >= self.pmf.shape[dimen]:
                return False
        return True

    def _get_new_frames(self, pmf, root_frames):
        """ returns a dict of all frames surrounding the root_frames
        that have not an assigned energy yet, as well as their corresponding root
        frame in the format {new_frame1: root_frame1, new_frame2: root_frame2} """

        # find all neighboring frames and create a dict that associates them to the root frame with lowest energy
        new_frames = {}
        for frame in root_frames:
            neighbors = self._generate_neighbor_list(frame)

            # remove neighbors that are not inside the pmf
            neighbors = [n for n in neighbors if self._is_in_pmf(n)]

            # for each neighbor, check if its already in the list and compare root frame energy
            for n in neighbors:
                try:
                    root_energy = pmf[frame]
                    old_root = new_frames[n]
                    old_root_energy = pmf[old_root]
                    if root_energy < old_root_energy:
                        new_frames[n] = frame
                except KeyError:
                    new_frames[n] = frame


        # remove already sampled frames (where energy >= 0)
        new_frames_list = list(new_frames.keys())
        for idx in range(len(new_frames_list)):
            new_frame = new_frames_list[idx]
            energy = pmf[new_frame]
            if energy >= 0:
                del(new_frames[new_frame])
                
        return new_frames
    
    
    def _main(self):
        # get the initial simulation and surrounding frames
        root_frames = [self._get_index_for_lambdas(self.cvs_init)]
        new_frames = self._get_new_frames(self.pmf, root_frames)
        
        self.num_iterations = 0
        self.E = self.E_min
        
        # outer main loop: increase E and calculate PMF until E > E_max
        while True:
            
            # stop if max iterations is reached
            self.num_iterations += 1
            if(self.max_iterations > 0 and self.num_iterations > self.max_iterations):
                print("Max iterations reached ({})".format(self.max_iterations))
                return

            # stop if the pmf is sampled 
            if len(np.where(self.pmf < 0)) == 0:
                print("Every window of the PMF appears to be sampled.")
                return
            
            print("~~~~~~~~~~~~~~~ Iteration {}/{} ~~~~~~~~~~~~~~~~".format(self.num_iterations, self.max_iterations))
            lambdas = dict([(self._get_lambdas_for_index(x), self._get_lambdas_for_index(y)) for x,y in new_frames.items()])

            self.pre_run_hook()
            print("Running simulations")
            self.simulate_frames(lambdas, new_frames)
            
            print("Calculating new PMF")
            self.pmf = self.calculate_new_pmf()
            self.after_run_hook()
            
            while self.E <= self.E_max:
                root_frames = self._get_root_frames(self.pmf, self.E)
                new_frames = self._get_new_frames(self.pmf, root_frames)
                
                if len(new_frames) == 0:
                    self.E += self.E_incr
                    print("Max energy increased to {} (max={})".format(self.E, self.E_max))
                else:
                    break


    def run(self):
        # initialize the pmf
        self.pmf = self._init_pmf()

        # start the simulation/evaluation loop
        self._main()
        print("Finished.")
        
    def simulate_frames(self, frames, lambdas):
        """ This should be implemented to run the simulation for each frame in ```frames```
         with ```lambdas```. Both variables are dictionaries where new values are keys and root values are values
         """
        print("New lambda values to simulate: {}".format(lambdas))
        print("(You should implement `simulate_frames` method yourself.)")

    def calculate_new_pmf(self):
        """ This should be overwritten to calculate the PMF based on the new simulations and
        return a pmf of correct shape/spacing according to the cvs"""
        pass


    def after_run_hook(self):
        """ This can be implemented to hook into the simulation cycle after the reevaluation of the pmf """
        pass

    def pre_run_hook(self):
        """ This can be implemented to hook into the simulation cycle before the simulation runs """
        pass

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

        simulation_dir = "simulations"
        print("Collecting sampling data from simulations folder")

        # collect COLVARs
        wham_dir = "WHAM/"
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
                out.write("WHAM/{}.xvg {} {} {} {}\n".format(f, x, y, fc_x, fc_y))

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



class UmbrellaRunnerTest(unittest.TestCase):

    def test_init_pmf_3d(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (-3, 3, 1),
            (-3, 2, 1)
        ])
        pmf = runner._init_pmf()

        expected_shape = (7, 7, 6)
        self.assertEquals(pmf.shape, expected_shape)

    def test_init_pmf_odd(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (-3.5, 3, 1)
        ])
        pmf = runner._init_pmf()
        expected_shape = (7, 7)
        self.assertEquals(pmf.shape, expected_shape)

    def test_get_lambdas_for_index(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (0, 4, 1)
        ])

        lambdas = runner._get_lambdas_for_index((0, 0))
        self.assertAlmostEqual(lambdas, (-3, 0))

        lambdas = runner._get_lambdas_for_index((3, 2))
        self.assertEquals(lambdas, (0, 2))

    def test_get_index_for_lambdas(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (0, 4, 1)
        ])

        index = runner._get_index_for_lambdas((3, 3))
        self.assertEquals(index, (6, 3))

    def test_get_index_for_lambdas_error(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (0, 4, 1)
        ])

        with self.assertRaises(ValueError) as error:
            runner._get_index_for_lambdas((3, 2.5))

    def test_get_root_frames(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (0, 4, 1)
        ])
        runner.pmf = runner._init_pmf()
        runner.pmf[0, 3] = 5
        runner.pmf[0, 2] = 2
        root_frames = runner._get_root_frames(runner.pmf, 3)
        self.assertEquals(1, len(root_frames))
        self.assertEquals((0, 2), root_frames[0])

    def test_get_root_frames_3d(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (0, 4, 1),
            (0, 4, 1)
        ])
        runner.pmf = runner._init_pmf()
        runner.pmf[0, 3, 3] = 5
        runner.pmf[0, 2, 2] = 2
        root_frames = runner._get_root_frames(runner.pmf, 3)
        self.assertEquals(1, len(root_frames))
        self.assertEquals((0, 2, 2), root_frames[0])

    def test_generate_neighbor_list(self):
        runner = UmbrellaRunner()
        root = (1, 3)
        neighbors = runner._generate_neighbor_list(root)
        expected_neighbors = [
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 2),
            (2, 3),
            (2, 4),
        ]
        self.assertEquals(neighbors, expected_neighbors)

    def test_generate_neighbor_list2(self):
        runner = UmbrellaRunner()
        root = (3, 1)
        neighbors = runner._generate_neighbor_list(root)
        expected_neighbors = [
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 1),
            (3, 2),
            (4, 0),
            (4, 1),
            (4, 2),
        ]
        self.assertEquals(neighbors, expected_neighbors)
        pass

    def test_generate_neighbor_list_3d(self):
        runner = UmbrellaRunner()
        root = (2, 2, 2)
        neighbors = runner._generate_neighbor_list(root)
        expected_neighbors = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    expected_neighbors.append((root[0]+x, root[1]+y, root[2]+z))
        self.assertEquals(len(neighbors), len(expected_neighbors))


    def test_is_in_pmf(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-1, 1, 1),
            (-1, 1, 1),
            (-1, 1, 1)
        ])
        runner.pmf = runner._init_pmf()
        self.assertFalse(runner._is_in_pmf((-1, 0, 0)))
        self.assertFalse(runner._is_in_pmf((0, 0, 3)))
        self.assertTrue(runner._is_in_pmf((0, 0, 0)))

    def test_get_new_frames_3d(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-1, 1, 1),
            (-1, 1, 1),
            (-1, 1, 1)
        ])
        runner.pmf = runner._init_pmf()
        runner.pmf[1, 1, 1] = 5
        root_frames = [(1, 1, 1)]
        new_frames = runner._get_new_frames(runner.pmf, root_frames)

        expected_new_frames = {}
        for x in [0, 1, 2]:
            for y in [0, 1, 2]:
                for z in [0, 1, 2]:
                    if (x, y, z) != (1, 1, 1):
                        expected_new_frames[(x, y, z)] = (1, 1, 1)

        self.assertEquals(26, len(new_frames.keys()))
        self.assertDictEqual(expected_new_frames, new_frames)

    def test_get_new_frames(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-2, 2, 1),
            (-2, 2, 1),
        ])
        runner.pmf = np.array([
            [-1, -1, -1, -1, -1],
            [-1, 34, 32, 26, -1],
            [-1, 40, 42, 51, -1],
            [-1, 28, 41, 37, -1],
            [-1, -1, -1, -1, -1]])
        root_frames = [(1, 3), (3, 1)]
        expected_new_frames = {
            (0, 2): (1, 3),
            (0, 3): (1, 3),
            (0, 4): (1, 3),
            (1, 4): (1, 3),
            (2, 4): (1, 3),
            (2, 0): (3, 1),
            (3, 0): (3, 1),
            (4, 0): (3, 1),
            (4, 1): (3, 1),
            (4, 2): (3, 1),
        }
        new_frames = runner._get_new_frames(runner.pmf, root_frames)
        self.assertDictEqual(expected_new_frames, new_frames)



if __name__ == '__main__':
    unittest.main()


