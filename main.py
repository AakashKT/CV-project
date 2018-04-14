import MRF as mrf  
from reprojection import main as reprj

import os, sys, json, re
import numpy as np
from numpy.linalg import inv

from scipy.misc import imread, imsave




if __name__ == '__main__':
	img_dir = sys.argv[1];
	all_images = [];
	print "line 16"
	for i in range(0, 10):
		all_images.append(imread('%s0000000%d.jpg' % (img_dir, i)));

	for i in range(10, 14):
		all_images.append(imread('%s000000%d.jpg' % (img_dir, i)));
	
	ri = reprj.get_reprojection('images_downsampled_2/dense.nvm.cmvs/00/models/option-0000.ply', 'images_downsampled_2/dense.nvm.cmvs/00/cameras_v2.txt', img_dir, '00000013.jpg');
	print "line 24"
	mrf.init(all_images, ri);


	
	# ret_graph = mrf.gridGraph((450, 500));

	# label_assignment = mrf.alpha_expansion_call(ret_graph);
	# final_image = mrf.GetImage(label_assignment);

	# imsave('final.jpg', final_image);