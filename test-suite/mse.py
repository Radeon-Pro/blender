import imageio
import numpy as np
import os
import sys
import subprocess

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare():
    for kernel in ["background", "direct_lighting", "do_volume", "holdout_emission_blurring_pathtermination_ao", "indirect_background", "lamp_emission", "shader_eval", "shadow_blocked_ao", "shadow_blocked_dl", "subsurface_scatter"]:#"background", "direct_lighting", "do_volume", "holdout_emission_blurring_pathtermination_ao", "indirect_background", "lamp_emission", "shader_eval", "shadow_blocked_ao", "shadow_blocked_dl", "subsurface_scatter"
        for blend_project in ["wdas_cloud", "classroom_gpu", "bmw27_gpu", "sponza_cycles_testing", "pavillon_barcelona_gpu", "test-scene"]:
            before_image_filename = blend_project + "-before.png"
            for i in range(0, 7):
                after_image_filename = blend_project + "-after-" + kernel + "-" + str(i) + "-0000.png"
                if os.path.exists(before_image_filename) and os.path.exists(after_image_filename):
                    before = imageio.imread(before_image_filename)
                    after  = imageio.imread(after_image_filename)

                    print(after_image_filename, mse(after, before))

kernels_path = "../build-win64_vc15/bin/RelWithDebInfo/2.92/scripts/addons/cycles/source/kernel/kernels/opencl/"

for kernel in ["background", "direct_lighting", "do_volume", "holdout_emission_blurring_pathtermination_ao", "indirect_background", "lamp_emission", "shader_eval", "shadow_blocked_ao", "shadow_blocked_dl", "subsurface_scatter"]:#"background", "direct_lighting", "do_volume", "holdout_emission_blurring_pathtermination_ao", "indirect_background", "lamp_emission", "shader_eval", "shadow_blocked_ao", "shadow_blocked_dl", "subsurface_scatter"
    kernel_filename = kernels_path + "kernel_" + kernel
    with open(kernel_filename + ".orig", 'r') as original: data = original.read()
    with open(kernel_filename + ".cl", 'w') as modified: modified.write(data)

for blend_project in ["classroom_gpu", "bmw27_gpu", "sponza_cycles_testing" , "pavillon_barcelona_gpu", "wdas_cloud"]:#"wdas_cloud", "classroom_gpu", "bmw27_gpu", "sponza_cycles_testing" , "pavillon_barcelona", "test-scene":
    output_image_filename = blend_project + "-before-"
    if not os.path.exists(output_image_filename + "0000.png"):
        command = ["..\\build-win64_vc15\\bin\\RelWithDebInfo\\blender.exe", "-b", ".\\" + blend_project + "\\" + blend_project +".blend", "-F", "PNG", "-o", "//..\\" + output_image_filename, "-f", "0", "-x", "1"]
        print(command)
        subprocess.run(command, shell=False)

    for kernel in ["background", "direct_lighting", "do_volume", "holdout_emission_blurring_pathtermination_ao", "indirect_background", "lamp_emission", "shader_eval", "shadow_blocked_ao", "shadow_blocked_dl", "subsurface_scatter"]:#"background", "direct_lighting", "do_volume", "holdout_emission_blurring_pathtermination_ao", "indirect_background", "lamp_emission", "shader_eval", "shadow_blocked_ao", "shadow_blocked_dl", "subsurface_scatter"
        kernel_filename = kernels_path + "kernel_" + kernel
        for i in range(0, 7):
            output_image_filename = blend_project + "-after-" + kernel + "-" + str(i) + "-"
            if not os.path.exists(output_image_filename + "0000.png"):
                prepend_text = "#define __SVM_EVAL_NODES_SHADER_TYPE_SURFACE__SKIP__" + str(i) + "\n\n"
                with open(kernel_filename + ".orig", 'r') as original: data = original.read()
                with open(kernel_filename + ".cl", 'w') as modified: modified.write(prepend_text + data)

                command = ["..\\build-win64_vc15\\bin\\RelWithDebInfo\\blender.exe", "-b", ".\\" + blend_project + "\\" + blend_project +".blend", "-F", "PNG", "-o", "//..\\" + output_image_filename, "-f", "0", "-x", "1"]
                print(command)
                subprocess.run(command, shell=False)

                with open(kernel_filename + ".orig", 'r') as original: data = original.read()
                with open(kernel_filename + ".cl", 'w') as modified: modified.write(data)
            else:
                print("SKIP " + output_image_filename + "0000.png")
                sys.stdout.flush()

compare()
