# Useful
from pathlib import Path
import os
from shutil import copy
import pandas as pd
import numpy as np

# Local files to import
from vReconstruction import vReconstruction

from tuners import ADMMoptimizerName

class iComparison(vReconstruction):
    def __init__(self, config):
        print("__init__")

    def runComputation(self, config, fixed_config, hyperparameters_config, root):

        if (self.method == 'AML'):
            self.beta = hyperparameters_config["A_AML"]
            self.A_AML = hyperparameters_config["A_AML"]
        elif (self.method == ADMMoptimizerName):
            self.beta = hyperparameters_config["alpha"]
        elif (self.method == 'BSREM'):
            self.beta = self.rho
        # castor-recon command line
        castor_command_line = self.castor_common_command_line(self.subroot_data, self.PETImage_shape_str, self.phantom,
                                                              self.replicate,
                                                              self.post_smoothing) + self.castor_opti_and_penalty(self.method, self.penalty, self.rho)
        print("commmmmmmmmmmm")
        print(castor_command_line)
        if (self.method == ADMMoptimizerName):
            self.ADMMLim(fixed_config, hyperparameters_config)
        else:
            it = ' -it ' + str(self.max_iter) + ':' + str(fixed_config["nb_subsets"])

            output_path = ' -fout ' + self.subroot + 'Comparison/' + self.method + '/' + self.suffix + '/' + self.method  # Output path for CASTOR framework
            # if (self.method == 'AML' or self.method == 'BSREM'):
            #    output_path += '_beta_' + str(self.beta)
            initialimage = ''

            Path(self.subroot + 'Comparison/' + self.method + '/' + self.suffix).mkdir(parents=True,
                                                                                       exist_ok=True)  # CASToR pat

            print("CASToR command line : ")
            print(castor_command_line + it + output_path + initialimage)
            os.system(castor_command_line + it + output_path + initialimage)

        # NNEPPS
        if (self.method == ADMMoptimizerName):
            max_it = hyperparameters_config["nb_iter_second_admm"]
        else:
            max_it = fixed_config["max_iter"]

        if (hyperparameters_config["NNEPPS"]):
            print("NNEPPS")
            for it in range(1, max_it + 1):
                self.NNEPPS_function(fixed_config, hyperparameters_config, it)

        # Initializing results class
        if ((fixed_config["average_replicates"] and self.replicate == 1) or (
                fixed_config["average_replicates"] == False)):
            from iResults import iResults
            classResults = iResults(config)
            classResults.nb_replicates = self.nb_replicates
            classResults.debug = self.debug
            classResults.rho = self.rho
            classResults.initializeSpecific(fixed_config, hyperparameters_config, root)
            classResults.runComputation(config, fixed_config, hyperparameters_config, root)

    def ADMMLim(self, fixed_config, hyperparameters_config):
        # Path variables
        subroot_output_path = (self.subroot + 'Comparison/' + self.method + '/' + self.suffix)
        subdir = 'ADMM' + '_' + str(fixed_config["nb_threads"])
        Path(self.subroot + 'Comparison/' + self.method + '/').mkdir(parents=True, exist_ok=True)  # CASTor path
        Path(self.subroot + 'Comparison/' + self.method + '/' + self.suffix + '/' + subdir).mkdir(parents=True,
                                                                                                  exist_ok=True)  # CASToR path

        i = 0
        k_init = -1
        full_output_path_k_next = subroot_output_path + '/' + subdir + '/' + format(i) + '_' + format(k_init + 1)

        castor_command_line_x = self.castor_common_command_line(self.subroot_data, self.PETImage_shape_str,
                                                                self.phantom, self.replicate, self.post_smoothing)
        f_mu_for_penalty = ' -multimodal ' + self.subroot_data + 'Data/initialization/1_im_value_cropped.hdr'  # its value is not useful to compute v^0

        # Define command line to run ADMM with CASToR, to compute v^0
        x_for_init_v = ' -img ' + self.subroot_data + 'Data/initialization/' + self.image_init_path_without_extension + '.hdr' if self.image_init_path_without_extension != "" else ''  # initializing CASToR PLL reconstruction with image_init or with CASToR default values

        # Compute one ADMM iteration (x, v, u) when only initializing x to compute v^0. x (i_0_it.img) and u (i_0_u.img) will be computed, but are useless
        x_reconstruction_command_line = castor_command_line_x + self.castor_opti_and_penalty(self.method, self.penalty,
                                                                                             self.rho) + ' -fout ' + full_output_path_k_next + ' -it 1:1' + x_for_init_v + f_mu_for_penalty  # we need f-mu so that ADMM optimizer works, even if we will not use it...
        print('vvvvvvvvvvv0000000000')
        print(x_reconstruction_command_line)
        self.compute_x_v_u_ADMM(x_reconstruction_command_line, full_output_path_k_next, subdir, i, k_init - 1,
                                self.phantom, subroot_output_path, self.subroot_data)
        # Copy u^-1 coming from CASToR to v^0
        copy(full_output_path_k_next + '_u.img', full_output_path_k_next + '_v.img')
        self.write_hdr(self.subroot_data, [i, k_init + 1], subdir, self.phantom, 'v', subroot_output_path,
                       matrix_type='sino')

        # Then initialize u^0 (u^-1 in CASToR)
        copy(self.subroot_data + 'Data/initialization/0_sino_value.img', full_output_path_k_next + '_u.img')
        self.write_hdr(self.subroot_data, [i, k_init + 1], subdir, self.phantom, 'u', subroot_output_path,
                       matrix_type='sino')

        # Compute one ADMM iteration (x, v, u)
        print('xxxxxxxxxxxxxxxxxxxxx')
        for k in range(k_init + 1, hyperparameters_config["nb_iter_second_admm"]):
            # Initialize variables for command line
            if (k == k_init + 1):
                if (i == 0):  # choose initial image for CASToR reconstruction
                    initialimage = ' -img ' + self.subroot_data + 'Data/initialization/' + self.image_init_path_without_extension + '.hdr' if self.image_init_path_without_extension != "" else ''  # initializing CASToR PLL reconstruction with image_init or with CASToR default values
            else:
                initialimage = ' -img ' + subroot_output_path + '/' + subdir + '/' + format(i) + '_' + format(
                    k) + '_it' + str(hyperparameters_config["sub_iter_PLL"]) + '.hdr'

            base_name_k = format(i) + '_' + format(k)
            base_name_k_next = format(i) + '_' + format(k + 1)
            full_output_path_k = subroot_output_path + '/' + subdir + '/' + base_name_k
            full_output_path_k_next = subroot_output_path + '/' + subdir + '/' + base_name_k_next
            v_for_additional_data = ' -additional-data ' + full_output_path_k + '_v.hdr'
            u_for_additional_data = ' -additional-data ' + full_output_path_k + '_u.hdr'

            # Compute one ADMM iteration (x, v, u)
            if ((k == hyperparameters_config["nb_iter_second_admm"] - 1) and fixed_config[
                "post_smoothing"]):  # For last iteration, apply post smoothing for vizualization
                # conv = ''
                conv = ' -conv gaussian,18,1,3.5::post'
            else:
                conv = ''

            # Number of iterations from config dictionnary
            it = ' -it ' + str(hyperparameters_config["sub_iter_PLL"]) + ':1'  # 1 subset

            x_reconstruction_command_line = castor_command_line_x \
                                            + self.castor_opti_and_penalty(self.method, self.penalty, self.rho) \
                                            + ' -fout ' + full_output_path_k_next + it + u_for_additional_data \
                                            + v_for_additional_data + initialimage + f_mu_for_penalty \
                                            + conv # we need f-mu so that ADMM optimizer works, even if we will not use it...

            # x_reconstruction_command_line = castor_command_line_x + ' -fout ' + full_output_path_k_next + it + ' -additional-data /home/meraslia/sgld/hernan_folder/data/Algo/0_8_u.hdr -additional-data /home/meraslia/sgld/hernan_folder/data/Algo/0_8_v.hdr -multimodal /home/meraslia/sgld/hernan_folder/data/Algo/Data/initialization/1_im_value_cropped.hdr'# -img /home/meraslia/sgld/hernan_folder/data/Algo/0_8_it10.hdr'

            print("k = ", k)  # k+1 means the (k+1)th outer iteration
            print(x_reconstruction_command_line)
            self.compute_x_v_u_ADMM(x_reconstruction_command_line, full_output_path_k_next, subdir, i, k, self.phantom,
                                    subroot_output_path, self.subroot_data)

            # -- AdaptiveRho ---- AdaptiveRho ---- AdaptiveRho ---- AdaptiveRho ---- AdaptiveRho ---- AdaptiveRho --
            path_adaptive = subroot_output_path + '/' + subdir + '/' + format(i) + '_' + format(
                k+1) + '_adaptive.log'
            theLog = pd.read_table(path_adaptive)
            theAdaptiveAlphaRow = theLog.loc[[0]]
            theAdaptiveAlphaRowArray = np.array(theAdaptiveAlphaRow)
            theAdaptiveAlphaRowString = theAdaptiveAlphaRowArray[0, 0]
            adaptiveAlpha = float(theAdaptiveAlphaRowString)
            print('*************************************************************************************************************************************', 'k+1 =', k+1, 'adaptive alpha =', adaptiveAlpha)
            self.alpha = adaptiveAlpha
            castor_command_line_x = self.castor_common_command_line(self.subroot_data, self.PETImage_shape_str,
                                                                    self.phantom, self.replicate,
                                                                    self.post_smoothing)


    def NNEPPS_function(self, fixed_config, hyperparameters_config, it):
        executable = 'removeNegativeValues.exe'

        if (self.method == ADMMoptimizerName):
            i = 0
            subdir = 'ADMM' + '_' + str(fixed_config["nb_threads"])
            input_without_extension = self.subroot + 'Comparison/' + self.method + '/' + self.suffix + '/' + subdir + '/' + format(
                i) + '_' + str(it) + '_it' + format(hyperparameters_config["sub_iter_PLL"])
        else:
            input_without_extension = self.subroot + 'Comparison/' + self.method + '/' + self.suffix + '/' + self.method + '_beta_' + str(
                self.beta) + '_it' + format(it)

        input = ' -i ' + input_without_extension + '.img'
        output = ' -o ' + input_without_extension + '_NNEPPS'  # Without extension !

        # The following 9 commands can be used to specify to which part of the image the NNEPPS has to be applied. You can set dim, min, and max as you wish, provided they are consistent. The default value of min is 0. Note that if you specify dim and max, min is automatically set to the correct value.

        dimX = ' -dimX ' + str(self.PETImage_shape[0])
        dimY = ' -dimY ' + str(self.PETImage_shape[1])
        dimZ = ' -dimZ ' + str(self.PETImage_shape[2])

        minX = ''
        minY = ''
        minZ = ''

        maxX = ''
        maxY = ''
        maxZ = ''  # ' -maxZ 3'

        # The two following variables are the full size of the input image. They are important for a correct reading of the data. If unset, they are assumed to be equal to the previous max value.
        inputSizeX = ' -inputSizeX ' + str(self.PETImage_shape[0])
        inputSizeY = ' -inputSizeY ' + str(self.PETImage_shape[1])
        inputSizeZ = ' -inputSizeZ ' + str(self.PETImage_shape[2])

        nbThreads = ''  # '-th 8' Don't use this option if you want to use all threads

        # The 3 following lines give the coefficients assigned to the neighbors in each of the three dimensions (only the 1st-order neighbors are considered). They must sum up to 0.5. If voxels are square, the natural choice is 1/6 for each (default value). In the example, other values are provided to favor close neighbors because voxels are cuboids. See the supplementary material for further explanation of these numbers. Note that the value 0 is forbidden. If you are using 1D or 2D images, provide any value to the unused dimensions, and the code will adapt to the fact that the dimensions do not exist. For example, for square pixels using x and y dimensions, 1/6;1/6;1/6 is equivalent to 0.2;0.2;0.1 and to 0.1;0.1;0.3.
        coeffX = ' -coeffX 0.108882'
        coeffY = ' -coeffY 0.108882'
        coeffZ = ' -coeffZ 0.282236'

        skip_initialization = ' -skip_initialization'  # '-skip_initialization' #Use this option if you want to skip the initialization step.
        critere_stop_init = ''  # '-critere_stop_init 1.0e-4' by default. Criterion used to stop the initialization step, the lower, the longer the initialization step will be. Unused if -skip_initialization is set.
        skip_algebraic = ''  # '-skip_algebraic'#Use this option only if you want to skip the main algebraic part and directly write the image after the initialization step.
        precision = ' -precision -1'  # '-precision 1.0e-3' by default. Use -1 for maximum precision. This is the relative precision used by the main algebraic part to proceed. Unused if -skip_algebraic is set.

        # input and output type. This doesn't affect the precision of the computation, which is always done using doubles. Two possibilities : float or double. Default value: float
        input_type = ''  # -input_type double'
        output_type = ''  # -output_type double'

        # Command line (do not modify):
        NNEPPS_command_line = executable + input + output + dimX + dimY + dimZ + nbThreads + coeffX + coeffY + coeffZ + precision + skip_initialization + critere_stop_init + minX + minY + minZ + maxX + maxY + maxZ + inputSizeX + inputSizeY + input_type + output_type + skip_algebraic
        print(NNEPPS_command_line)
        os.system(NNEPPS_command_line)
