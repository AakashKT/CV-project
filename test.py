import os, sys
import numpy as np
from scipy.misc import imsave


if __name__ == '__main__':
	vertices = 0;
	imgs = [];
	for i in range(0, 10):
		imgs.append(np.zeros((1080, 1920, 3), dtype=np.int));

	f = open('raw_images_2/test.nvm', 'r');

	e_lines = 0;
	for l in f:
		if e_lines == 1 and l == '\n':
			break;

		if l == '\n':
			e_lines += 1;

	curr_line = 0;
	for l in f:
		if l == '\n':
			break;
		elif curr_line == 0:
			vertices = int(l.replace('\n', ''));
		else:
			line = l.replace('\n', '').split(' ');
			n_m = int(line[6]);
			r = int(line[3]);
			g = int(line[4]);
			b = int(line[5]);

			for i in range(0, n_m):
				ind = 7 + 2 + (i * 4);

				img_ind = int(line[ind-2])
				x = int(round(float(line[ind]))) + 960;
				y = int(round(float(line[ind+1]))) + 540;

				r_img = imgs[img_ind];
				r_img[y, x, 0] = r;
				r_img[y, x, 1] = g;
				r_img[y, x, 2] = b;

		curr_line += 1;

	for i in range(0, 10):
		r_img = imgs[i];

		imsave('%d_img.png' % i, r_img);