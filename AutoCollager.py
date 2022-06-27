import numpy as np
from PIL import Image
from glob import glob
import random
import os

class AutoCollager:
    def __init__(self, target_img_path, sample_dir, save_dir, n_samples=50, square_size=32,
                random_square_sizes=True,square_sizes=[8,16,32,]):
        self.target_img_path = target_img_path
        self.square_size = square_size
        self.shape_validated = False
        while not self.shape_validated:
            self.open_target()
            self.validate_shape()
        self.create_canvas()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.sample_dir = sample_dir
        self.image_paths = glob(os.path.join(sample_dir, '*'))
        
        self.n_samples = n_samples
        
        
        self.samples = [0 for i in range(n_samples)]
        self.best_sample = 0
        
        self.last_frame = 0
        
    def collage(self, n_iters=1000):
        for i in range(n_iters):
            print("Starting iter",i)
            self.sample_target()
            self.gather_samples()
            self.compare_samples_to_target()
            if self.compare_sample_to_canvas():
                self.paste_sample()
                self.save_progress()
            
    def open_target(self):
        self.target_img = Image.open(self.target_img_path).convert("RGB")
        self.imgdims = self.target_img.size
        
    def validate_shape(self):
        if self.imgdims[0] < self.square_size \
            or self.imgdims[1] < self.square_size:
            self.shape_validated = False
        else:
            self.shape_validated = True
            
    def create_canvas(self):
        self.canvas = np.array(Image.new('RGB', self.imgdims))
    
    def sample_target(self):
        self.current_top = random.randint(0, self.imgdims[1]-self.square_size-1)
        self.current_left = random.randint(0, self.imgdims[0]-self.square_size-1)
        data = np.array(self.target_img)[self.current_top:self.current_top+self.square_size,\
                                       self.current_left:self.current_left+self.square_size,:]
        self.target_sample = Image.fromarray(data.astype('uint8'))
        try:
            assert data.shape == (self.square_size, self.square_size, 3)
        except:
            print(np.array(self.target_img).shape, data.shape, self.current_top, self.current_left)
            print("imgdims:", self.imgdims)
        
    def gather_samples(self):
        for i in range(self.n_samples):
            
            sample_dims = (0,0,0)
            while sample_dims[0] < self.square_size+1 or sample_dims[1] < self.square_size+1:
                image_to_sample_path = self.image_paths[random.randint(0,len(self.image_paths)-1)]
                image_to_sample = np.array(Image.open(image_to_sample_path).convert("RGB"))
                sample_dims = image_to_sample.shape

            
            
            x1 = random.randint(0, (sample_dims[0]-self.square_size)-1)
            y1 = random.randint(0, (sample_dims[1]-self.square_size)-1)
            
            sample = image_to_sample[x1:x1+self.square_size,\
                                    y1:y1+self.square_size,:]
            #print(sample_dims, x1, y1, sample.shape)
            assert sample.shape == (self.square_size, self.square_size, 3)

            self.samples[i] = sample
            
    
    def compare_samples_to_target(self):
        scores = []
        for i, sample in enumerate(self.samples):
            scores.append(self.comparison_function(np.array(self.target_sample),sample))
        self.best_idx = np.argmin(scores)
            
            
    def compare_sample_to_canvas(self):
        best_sample = self.samples[self.best_idx]
        sampled_canvas = self.canvas[self.current_top:self.current_top+self.square_size,
                    self.current_left:self.current_left+self.square_size,:]
        
        sample_score = self.comparison_function(np.array(self.target_sample),best_sample)
        canvas_score = self.comparison_function(np.array(self.target_sample),sampled_canvas)
        
        # Returns true if sample is better than canvas
        return sample_score < canvas_score
    
    def paste_sample(self):
        assert self.canvas[self.current_top:self.current_top+self.square_size,
                    self.current_left:self.current_left+self.square_size,:].shape == (self.square_size, self.square_size, 3)
        assert self.samples[self.best_idx].shape == (self.square_size, self.square_size, 3)
        self.canvas[self.current_top:self.current_top+self.square_size,
                    self.current_left:self.current_left+self.square_size,:] =\
                    self.samples[self.best_idx]
    
    def save_progress(self, i=None):
        if i is not None:
            Image.fromarray(self.canvas).save(os.path.join(self.save_dir, 'test%04d.jpg'%i))
        else:
            Image.fromarray(self.canvas).save(os.path.join(self.save_dir, 'test%04d.jpg'%self.last_frame))
            self.last_frame += 1
    
    def comparison_function(self, i1, i2):
        try:
            #mse = np.mean((i1 - i2)**2)
            mse = np.sum(i1-i2)
        except:
            mse = 0
        return mse
