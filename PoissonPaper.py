import numpy as np
import cv2

#img = cv2.imread("./sharkBefore.png");
#cv2.imshow('image',img);

class PoissonBlending:
	def __init__(self, Labels):
		self.Labels = Labels;
		self.img = cv2.imread("./newImage.png");
		# [row, column, depth] = np.shape(self.img);
		# for k in range(depth):
		# 	for i in range(row):
		# 		for j in range(column):
		# 			print(self.img[i][j][k]);
		self.gradX = np.empty(np.shape(self.img), dtype=np.int32);
		self.gradY = np.empty(np.shape(self.img), dtype=np.int32);


	def gradientX(self, sourceLabel):
		print (np.shape(self.gradX));
		[row, column, depth] = np.shape(self.gradX);
		for k in range(depth):
			for i in range(row):
				for j in range(column):
					if self.Labels[i][j][k] == sourceLabel:
						self.gradX[i][j][k] = 1000;
					else:
						if j == column-1:
							self.gradX[i][j][k] = 0;
						else:
							if self.Labels[i][j][k] == -1 or self.Labels[i][j][k] != self.Labels[i][j+1][k]: 
								self.gradX[i][j][k] = 0;
							else:
								self.gradX[i][j][k] = (int)(self.img[i][j+1][k]) - (int)(self.img[i][j][k]);

	def gradientY(self, sourceLabel):
		[row, column, depth] = np.shape(self.gradY);
		for k in range(depth):
			for i in range(row):
				for j in range(column):
					if self.Labels[i][j][k] == sourceLabel:
						self.gradY[i][j][k] = 1000;
					else:
						if i == row-1:
							self.gradY[i][j][k] = 0;
						else:
							if self.Labels[i][j][k] == -1 or self.Labels[i][j][k] != self.Labels[i+1][j][k]:
								self.gradY[i][j][k] = 0;
							else:
								self.gradY[i][j][k] = (int)(self.img[i+1][j][k]) - (int)(self.img[i][j][k]);


	def calculate(self):
		self.newImageGradx = np.empty(np.shape(self.img), dtype=np.int32);
		[row, column, depth] = np.shape(self.newImageGradx);

		# For gradient along x-axis
		for k in range(depth):
			for i in range(row):
				for j in range(column-1, -1, -1):
					if j == column-1:
						self.newImageGradx[i][j][k] = -1;
					else:
						if self.gradX[i][j][k] == 1000:
							self.newImageGradx[i][j][k] = self.img[i][j][k];
						else:
							if self.newImageGradx[i][j+1][k] == -1:
								self.newImageGradx[i][j][k] = -1;
							else:
								self.newImageGradx[i][j][k] = self.newImageGradx[i][j+1][k] - self.gradX[i][j][k];

		for k in range(depth):
			for i in range(row):
				for j in range(column):
					if j == 0:
						continue;
					else:
						if self.newImageGradx[i][j-1][k] >= 0:
							self.newImageGradx[i][j][k] = self.newImageGradx[i][j-1][k] + self.gradX[i][j-1][k];
						else:
							self.newImageGradx[i][j][k] = 0;

		for k in range(depth):
			for i in range(row):
				for j in range(column):
					if self.newImageGradx[i][j][k] == -1:
						self.newImageGradx[i][j][k] = 0;

		
		# For gradient along y-axis
		self.newImageGrady = np.empty(np.shape(self.img), dtype=np.int32);
		[row, column, depth] = np.shape(self.newImageGrady);

		for k in range(depth):
			for j in range(column):
				for i in range(row):
					if i == 0:
						self.newImageGrady[i][j][k] = -1;
					else:
						if self.gradY[i][j][k] == 1000:
							self.newImageGrady[i][j][k] = self.img[i][j][k];
						else:
							if self.newImageGrady[i-1][j][k] >= 0:
								self.newImageGrady[i][j][k] = self.newImageGrady[i-1][j][k] + self.gradY[i-1][j][k];
							else:
								self.newImageGrady[i][j][k] = -1;

		for k in range(depth):
			for j in range(column):
				for i in range(row-1, -1, -1):
					if i == row-1:
						continue;
					else:
						if self.newImageGrady[i+1][j][k] >= 0:
							self.newImageGrady[i][j][k] = self.newImageGrady[i+1][j][k] - self.gradY[i][j][k];
						else:
							self.newImageGrady[i][j][k] = 0;

		for k in range(depth):
			for i in range(row):
				for j in range(column):
					if self.newImageGrady[i][j][k] == -1:
						self.newImageGrady[i][j][k] = 0;


		self.finalImage = np.empty(np.shape(self.img), dtype=np.int32);
		[row, column, depth] = np.shape(self.finalImage);
		
		for k in range(depth):
			for i in range(row):
				for j in range(column):
					self.finalImage[i][j][k] = (self.newImageGradx[i][j][k] + self.newImageGrady[i][j][k])//2;

		Max = np.amax(self.finalImage, axis=0);
		#print(Max);
		Max = np.amax(Max, axis=0);
		#print(Max);
		Max = np.amax(Max, axis=0);


		for k in range(depth):
			for i in range(row):
				for j in range(column):
					self.finalImage[i][j][k] = (int)((self.finalImage[i][j][k]/Max)*255);


		

		# for k in range(depth):
		# 	for i in range(row):
		# 		for j in range(column):
		# 			print(self.finalImage[i][j][k]);
		print (self.finalImage.dtype);
		print (np.shape(self.finalImage));
		cv2.imshow('image',self.finalImage.astype(int));
		cv2.waitKey(0);

	def apply(self):
		self.gradientX(0);
		self.gradientY(0);
		self.calculate();



if __name__ == "__main__":
	Labels = cv2.imread("./mask.png");
	poissonBlendingObj = PoissonBlending(Labels);
	poissonBlendingObj.apply();


#cv2.waitKey(0);
#print(np.shape(im));
