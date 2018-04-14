import os, sys, json, re
import numpy as np
from numpy.linalg import inv

from scipy.misc import imread, imsave, imresize

GLOBAL_DICT = {
	'00000000.jpg':0,
	'00000001.jpg':1,
	'00000002.jpg':2,
	'00000003.jpg':3,
	'00000004.jpg':4,
	'00000005.jpg':5,
	'00000006.jpg':6,
	'00000007.jpg':7,
	'00000008.jpg':8,
	'00000009.jpg':9,
	'00000010.jpg':10,
	'00000011.jpg':11,
	'00000012.jpg':12,
	'00000013.jpg':13,
	'00000014.jpg':14,
	'00000015.jpg':15,
	'00000016.jpg':16,
	'00000017.jpg':17,
	'00000018.jpg':18,
	'00000019.jpg':19,
	'00000020.jpg':20,
	'00000021.jpg':21,
	'00000022.jpg':22
}

ROWS = 900;
COLS = 1000;

class Camera():

	def __init__(self, focal_length, r_mat, t_vec, pos_vec):
		self.k = np.zeros((3, 3), dtype=np.float);
		self.c = np.zeros((3, 1), dtype=np.float);
		self.intr = np.zeros((3, 4), dtype=np.float);

		self.k[0, 0] = focal_length;
		self.k[1, 1] = focal_length;
		self.k[2, 2] = 1;

		self.intr[0:3, 0:3] = r_mat;
		self.intr[0:3, 3] = t_vec;

		self.c[0:, 0] = pos_vec;

		self.cam_mat = np.matmul(self.k, self.intr);

	def get_img_coord(self, vec):
		vec = np.append(vec, [[1]], axis=0);
		img_coord = np.matmul(self.cam_mat, vec);
		return img_coord[0:2, :] / float(img_coord[2, 0]);

	def get_depth_val(self, vec):
		diff = self.c - vec;
		return np.linalg.norm(diff);

	def get_p_far(self, vec):
		diff = self.c - vec;
		diff = diff + 0.01 * diff;

		return diff;

	def get_p_near(self, vec):
		diff = self.c - vec;
		diff = diff = 0.01 * diff;

		return diff;


def get_3d_points(model_file):
	points = [];
	f = open(model_file, 'r');

	l_start = False;
	for line in f:
		if l_start and line:
			sp = line.replace('\n', '').split(' ');

			x = float(sp[0]);
			y = float(sp[1]);
			z = float(sp[2]);

			r = int(sp[6]);
			g = int(sp[7]);
			b = int(sp[8]);

			points.append([x, y, z, r, g, b]);
		else:
			if line == 'end_header\n':
				l_start = True; 

	return points;


def get_cameras(camera_file):
	cameras = {};
	n_cam = 0;
	f = open(camera_file);

	l_start = False;
	for line in f:
		if l_start:
			for cam in range(n_cam):
				curr_line = 0;
				img_name = '';
				focal_length = 0.0;
				r_mat = [];
				t_vec = [];
				pos_vec = [];

				for cam_line in f:
					f_line = cam_line.replace('\n', '');

					if cam_line == '\n':
						break;
					elif curr_line == 0:
						img_name = f_line;
					elif curr_line == 2:
						focal_length = float(f_line);
					elif curr_line == 4:
						sp = f_line.split(' ');

						t_vec.append(float(sp[0]));
						t_vec.append(float(sp[1]));
						t_vec.append(float(sp[2]));
					elif curr_line == 5:
						sp = f_line.split(' ');

						pos_vec.append(float(sp[0]));
						pos_vec.append(float(sp[1]));
						pos_vec.append(float(sp[2]));
					elif curr_line == 8 or curr_line == 9 or curr_line == 10:
						sp = f_line.split(' ');

						temp = [];
						temp.append(float(sp[0]));
						temp.append(float(sp[1]));
						temp.append(float(sp[2]));

						r_mat.append(temp);

					curr_line += 1;

				cameras[img_name] = Camera(focal_length, r_mat, t_vec, pos_vec);

			break;
		else:
			if line.split(' ')[0] == '#' or line == '\n':
				continue;
			else:
				n_cam = int(line);
				l_start = True;

	return cameras;


def shift_coords(coord, size_r, size_c):
	r = coord[0] + size_r / 2;
	c = coord[1] + size_c / 2;

	return [r, c];


def temp_func(t_image, cameras, points, target_image):

	t_cam = cameras[target_image];
	for point in points:
		vec = np.array([[point[0]], [point[1]], [point[2]]]);
		img_coord = t_cam.get_img_coord(vec);

		x, y = shift_coords((int(round(img_coord[1])), int(round(img_coord[0]))), t_image.shape[0], t_image.shape[1]);

		t_image[x:x+11, y:y+11, 0] = int(point[3]);
		t_image[x:x+11, y:y+11, 1] = int(point[4]);
		t_image[x:x+11, y:y+11, 2] = int(point[5]);

		# t_image[x-5:x, y-5:y, 0] = int(point[3]);
		# t_image[x-5:x, y-5:y, 1] = int(point[4]);
		# t_image[x-5:x, y-5:y, 2] = int(point[5]);

	imsave('aakash.jpg', t_image);

def get_reprojection(model_file, camera_file, image_dir, target_image):
	extent_r = 300;
	extent_c = 300;

	points = get_3d_points(model_file);
	cameras = get_cameras(camera_file);

	og_t_image = imread(image_dir+target_image);
	(r, c, a) = og_t_image.shape;

	t_image = np.zeros((r+extent_r*2, c+extent_c*2, a), dtype=np.uint8);
	t_image[extent_r:extent_r+r, extent_c:extent_c+c, :] = og_t_image;

	temp_func(t_image, cameras, points, target_image);

	final_projection = [];
	for i in range(t_image.shape[0]):
		zero_pix = [];
		for i in range(t_image.shape[1]):
			temp = [];
			zero_pix.append(temp);

		final_projection.append(zero_pix);

	t_cam = cameras[target_image];
	for point in points:
		vec = np.array([[point[0]], [point[1]], [point[2]]]);
		img_coord = t_cam.get_img_coord(vec);

		x, y = shift_coords((int(round(img_coord[1])), int(round(img_coord[0]))), t_image.shape[0], t_image.shape[1]);

		for img, cam in cameras.items():
			temp = {};

			og_location = cam.get_img_coord(vec);
			og_location = shift_coords((int(round(og_location[1])), int(round(og_location[0]))), ROWS, COLS);
			if og_location[0] >= ROWS:
				og_location[0] = ROWS-1;
			if og_location[1] >= COLS:
				og_location[1] = COLS-1;

			p_near = cam.get_img_coord(cam.get_p_near(vec));
			p_near = shift_coords((int(round(p_near[1])), int(round(p_near[0]))), ROWS, COLS);

			p_far = cam.get_img_coord(cam.get_p_far(vec));
			p_far = shift_coords((int(round(p_far[1])), int(round(p_far[0]))), ROWS, COLS);

			temp['original_location'] = np.array(og_location, dtype=np.float);
			temp['p_location'] = np.array((x, y), dtype=np.float);
			temp['pfar_location'] = np.array(p_far, dtype=np.float);
			temp['pnear_location'] = np.array(p_near, dtype=np.float);
			temp['color'] = np.array((int(point[3]), int(point[4]), int(point[5])), dtype=np.float);
			temp['label'] = GLOBAL_DICT[img];

			final_projection[x][y].append(temp);
			final_projection[x+1][y].append(temp);
			final_projection[x-1][y].append(temp);
			final_projection[x][y+1].append(temp);
			final_projection[x][y-1].append(temp);

	for i in range(extent_r, extent_r+r):
		for j in range(extent_c, extent_c+c):
			temp = {};

			temp['original_location'] = np.array((i-extent_r, j-extent_c), dtype=np.float);
			temp['p_location'] = np.array((i, j), dtype=np.float);
			temp['pfar_location'] = np.array((i, j), dtype=np.float);
			temp['pnear_location'] = np.array((i, j), dtype=np.float);
			temp['color'] = t_image[i, j, :];
			temp['label'] = GLOBAL_DICT[target_image];

			final_projection[i][j].append(temp);
			final_projection[i+1][j].append(temp);
			final_projection[i-1][j].append(temp);
			final_projection[i][j+1].append(temp);
			final_projection[i][j-1].append(temp);

	return final_projection;




if __name__ == '__main__':
	a1 = sys.argv[1];
	a2 = sys.argv[2];
	a3 = sys.argv[3];
	a4 = sys.argv[4];

	get_reprojection(a1, a2, a3, a4);