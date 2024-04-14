# In order to get everything to build and setup, first we need a GPU. Otherwise, some of the pip installations will fail.
# We don't need a very powerful GPU for this step so in order to reduce queue time, we'll request any GPU
# This, the below command requests 1 GPU with cuda capability above 3.5 (lower) for at most 3 hours with 1 CPU and queue timeouts disabled
1. `qrsh -l gpus=1 -l gpu_c=3.5 -l h_rt=3:00:00 -pe omp 1 -now n`

# Next lets load all the modules wel'll need in.
2. `module load ninja`
3. `module load miniconda`
4. `module load cuda/11.8`
# (These steps are only to load ImageMagick and its submodules which is only needed if resizing images)
5. `module load fftw/3.3.4`
6. `module load tiff/4.0.6`
7. `module load openjpeg/2.1.2`
8. `module load imagemagick`

# Lets now setup our conda environment
# First we need a directory in projectnb to work in. Our home directory is limited to just 10 GB so that won't do.
# From here onwards, path inside this projectnb directory will be referred to as "$WRK_DIR"
# Might as well just export a env variable to make like easier.
9. `export WRK_DIR=/projectnb/path/that/you/figure/out`

# By default, when you install conda packages, they land in ~/.conda. This won't do due to the 10 GB cap. So, we need to relocate our conda directory using a special file called the .condarc file.
10. `mkdir $WRK_DIR/.conda $WRK_DIR/.conda/pkgs $WRK_DIR/.conda/envs`
11. `nano ~/.condarc` or `vim ~/.condarc`
12. Paste in the following text (modifying the $WRK_DIR ofcourse):
```
pkgs_dirs:
  - /projectnb/cs585/aseef/.conda/pkgs
envs_dirs:
  - /projectnb/cs585/aseef/.conda/envs
```

# Clone the our fork of the gaussing splatting repo from with in $WRK_DIR
# The recursive clonning will also clone subprojects used including: https://gitlab.inria.fr/sibr/sibr_core, https://github.com/graphdeco-inria/diff-gaussian-rasterization, https://gitlab.inria.fr/bkerbl/simple-knn, https://github.com/g-truc/glm
13. `git clone --recursive https://github.com/F1TenthBU/gaussian-splatting.git`

# Create your initial conda enviornment with cuda, torch, and colmap installed. On the SCC, higher cuda versions do not seem work.
# One you run this command, enter 'y' to confirm creating the environment.
# Once finished, this should have created a conda environment in $WRK_DIR/.conda/envs and NOT ~/.conda/envs.
14. `conda create --name venv python=3.8 colmap=3.8 pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia -c conda-forge`

# Activate the conda environment you created
15. `conda activate venv`

# We aren't done just yet! From inside the conda environment, we need to pip install a few more things!
16. `pip install plyfile tqdm $WRK_DIR/gaussian-splatting/submodules/simple-knn $WRK_DIR/gaussian-splatting/submodules/diff-gaussian-rasterization`

# Assuming, everything worked, we are now almost ready to train!
# First, lets create a datasets folder to put all the training images in.
17. mkdir $WRK_DIR/gaussian-splatting/datasets

# Now upload to the datasets a subfolder with the training images.
18. [Upload to $WRK_DIR/gaussian-splatting/datasets/your_dataset training data. The images should be in $WRK_DIR/gaussian-splatting/datasets/your_dataset/input and should be numbered. Also make sure the images are not blurry. To achieve this, consider using https://github.com/F1TenthBU/Video2Image4Colmap which attempts to convert a video stream to images using the least blurry frames. Ofcourse the input video itself still should be of as high quality as possible capturing multiple angles of the same location, avoids capturing just plain objects like a blank white wall, and has minimal motion blur.]

19. Modify `$WRK_DIR/gaussian-splatting/colmap_script` to fit your needs. This script runs the convert.py file which runs colmap. Here are some things to keep in mind:
* In the `-P` flag you must set the SCC project to use for the job queue. For example if you have access to the SCC through CS454 and thus you have access to /projectnb/cs454/, in the `-P` flag, you would put CS454.
* `-l h_rt=` is the hard max limit for the job runtime. If colmap takes longer than this, it will be killed.
* See https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/#job-options for all job options
* Make sure to look at https://github.com/F1TenthBU/gaussian-splatting/ for further params you can pass to the convert.py script

# When ready, submit the job using the following command
20. `qsub $WRK_DIR/gaussian-splatting/colmap_script`

21. Next, we need to modify `$WRK_DIR/gaussian-splatting/splatting_script` to fit your needs. This script runs train.py which runs the actual gaussian splatting training. Here are some things to keep in mind:
* Gaussian splatting requires 24G+ VRam to train. So avoid changing the `-l gpu_memory=24G` option unless you know what you are doing.
* `-l h_rt=` is the hard max limit for the job runtime. If training takes longer than this, it will be killed.
* See https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/#job-options for all job options
* Make sure to look at https://github.com/F1TenthBU/gaussian-splatting/ for further params you can pass to the train.py script
