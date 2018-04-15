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

	for i in range(len(reprj.GLOBAL_DICT)):
		all_images.append(None);

	for k, v in reprj.GLOBAL_DICT.items():
		all_images[v] = imread('%s%s' % (img_dir, k));
	
	ri = reprj.get_reprojection('im_2_2/dense.nvm.cmvs/00/models/option-0000.ply', 'im_2_2/dense.nvm.cmvs/00/cameras_v2.txt', img_dir, '00000009.jpg');
	
	mrf.init(all_images, ri);
	#mrf.run_py_max_flow();
	ret_graph = mrf.gridGraph((2424, 2936));

	label_assignment = mrf.alpha_expansion_call(ret_graph);
	final_image = mrf.GetImage(label_assignment);

	imsave('final.jpg', final_image);