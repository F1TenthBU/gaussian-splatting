# Requirements
- In order to get everything to build and setup, first we need a GPU. Otherwise, some of the pip installations will fail.
- We don't need a very powerful GPU for this step so in order to reduce queue time, we'll request any GPU.
- This, the below command requests 1 GPU with cuda capability above 3.5 (lower) for at most 3 hours with 1 CPU and queue timeouts disabled. Please modify hours required (`h_rt`) base on your training estimation.
  
   `qrsh -l gpus=1 -l gpu_c=3.5 -l h_rt=3:00:00 -pe omp 1 -now n`

# Load Required Modules
- `module load ninja`
- `module load miniconda`
- `module load cuda/11.8`
  
# Load Optional Modules

(These steps are only to load ImageMagick and its submodules which is only needed if resizing images)
-  `module load fftw/3.3.4`
-  `module load tiff/4.0.6`
-  `module load openjpeg/2.1.2`
-  `module load imagemagick`

# Setup Conda Environment
## Find Suitable working Directory
- We need a directory in **projectnb** to work in. Our home directory is limited to just *10 GB* so that won't do.
- From here onwards, path inside this projectnb directory will be referred to as `$WRK_DIR`
- Export the environment variable to make it easier:

  `export WRK_DIR=/projectnb/path/that/you/figure/out`

# Re-run
If it's not your first time running gaussian-splatting and **have** installed all packages, you can run the script `initialize_script.sh` by `. initialize_script.sh` to load modules and export WRK_DIR, then jump to submit jobs.

Make sure you have set the script executable by `chmod +x initialize_script.sh` before running.

## Install Conda Packages
### Relocate Conda Directory
By default, when you install conda packages, they land in `~/.conda`. This won't do due to the 10 GB cap. So, we need to relocate our conda directory using a special file called the .condarc file.
- `mkdir $WRK_DIR/.conda $WRK_DIR/.conda/pkgs $WRK_DIR/.conda/envs`
- `nano ~/.condarc` or `vim ~/.condarc`

Paste in the following text (modifying the `$WRK_DIR`):
```
pkgs_dirs:
  - /projectnb/cs585/aseef/.conda/pkgs
envs_dirs:
  - /projectnb/cs585/aseef/.conda/envs
```

### Clone Gaussing Splatting Repo
- Clone the our fork of the gaussing splatting repo from with in `$WRK_DIR`
- The recursive clonning will also clone subprojects used including: [SIBR Core](https://gitlab.inria.fr/sibr/sibr_core), [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization), [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn), [glm
](https://github.com/g-truc/glm)

  `git clone --recursive https://github.com/F1TenthBU/gaussian-splatting.git`

### Create your initial conda enviornment with cuda, torch, and colmap installed. 
  `conda create --name venv python=3.8 colmap=3.8 pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia -c conda-forge`

- On the SCC, higher cuda versions do not seem work.
- One you run this command, enter 'y' to confirm creating the environment.
- Once finished, this should have created a conda environment in $WRK_DIR/.conda/envs and NOT ~/.conda/envs.
 

### Activate Conda Environment
  `conda activate venv`

### Pip Modules Installation
We aren't done just yet! From inside the conda environment, we need to pip install a few more things!

   `pip install plyfile tqdm $WRK_DIR/gaussian-splatting/submodules/simple-knn $WRK_DIR/gaussian-splatting/submodules/diff-gaussian-rasterization`

- If running into error like this:
  
  ```shell
  File "/projectnb/ec504bk/students/alinajw/.conda/envs/Gaussians4D/lib/python3.7/site-packages/torch/utils/cpp_extension.py", line 1780, in _get_cuda_arch_flags
        arch_list[-1] += '+PTX'
    IndexError: list index out of range
  ```

  Run the following command to manually Set CUDA Architecture Flags, and then rerun pip install.

  ```shell
  export TORCH_CUDA_ARCH_LIST="6.0;7.5;8.0"
  ```
  
# Assuming, everything worked, we are now almost ready to train!
# Create a datasets folder to put all the training images in.
  `mkdir $WRK_DIR/gaussian-splatting/dataset`

# Upload to the dataset a subfolder with the training images.
- Upload to `$WRK_DIR/gaussian-splatting/dataset/[your_dataset]` training data.
- The images should be in `$WRK_DIR/gaussian-splatting/dataset/[your_dataset]/input` and should be numbered.
- Structure of foldering:
  ```
   ---gaussian-splatting
    |---dataset
    |  |---[your_dataset]
    |    |---input
    |---colmap_script
    |---splatting_script
  ```
- Also make sure the images are not blurry. To achieve this, consider using https://github.com/F1TenthBU/Video2Image4Colmap which attempts to convert a video stream to images using the least blurry frames. Ofcourse the input video itself still should be of as high quality as possible capturing multiple angles of the same location, avoids capturing just plain objects like a blank white wall, and has minimal motion blur.

# Modify `colmap_script` 
- Modify `$WRK_DIR/gaussian-splatting/colmap_script` to fit your needs. This script runs the `convert.py` file which runs colmap. Here are some things to keep in mind:
  -  In the `-P` flag you must set the SCC project to use for the job queue. For example if you have access to the SCC through CS454 and thus you have access to `/projectnb/cs454/`, in the `-P` flag, you would put CS454.
  -   `-l h_rt=` is the hard max limit for the job runtime. If colmap takes longer than this, it will be killed.
  -   See https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/#job-options for all job options
  -   Make sure to look at https://github.com/F1TenthBU/gaussian-splatting/ for further params you can pass to the convert.py script
-   Sample colmap_script based on the above structure
  ```shell
  #!/bin/bash -l

  #$ -P [to be replaced by your won project folder name]
  #$ -l h_rt=8:00:00
  #$ -m ea
  #$ -N colmap
  #$ -j y
  #$ -o colmap.logs
  #$ -pe omp 32
  #$ -l gpus=1
  #$ -l gpu_c=3.5
  
  module load miniconda
  conda activate venv
  python convert.py -s ./dataset/[your_dataset]/
  ```

# Submit colmap job
  `qsub $WRK_DIR/gaussian-splatting/colmap_script` or `qsub ./colmap_script` in `gaussian-splatting` directory.

  - If encountered error  like this
    ```shell
    qt.qpa.xcb: could not connect to display
    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
    ```

    Update this line in colmap_script: `python convert.py --no_gpu -s ./dataset/[your_dataset]/`

# Check colmap job output
- After colmap finish running, check output by doing `cat colmap.logs` to make sure everything ran smoothly.
- You should also head on over to `$WRK_DIR/gaussian-splatting//dataset/[your_dataset]` to check out the output of colmap. You should have a directory layout that looks like this:
```
<location>
|---<run-colmap-photometric.sh>
|---<run-colmap-geometric.sh>
|---<database.db-wal>
|---<database.db-shm>
|---<database.db>
|---stereo
    |---<patch-match.cfg>
    |---<fusion.cfg>
    |---normal_maps
    |---depth_maps
    |---consistency_graphs
|---input
    |---<image 0>
    |---<image 1>
    |---...
|---images
    |---<image 0>
    |---<image 1>
    |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
|---distorted
    |---sparse
        |---0
            |---cameras.bin
            |---images.bin
            |---points3D.bin
```
We are most interested in the `distorted` directory. Take a look in there to make sure `./distorted/sparse` has a directory called `0`. You may also want to visualize the colmap output on your local computer by downloading the output directory or directories in `./distorted/sparse/`. Note that it is possible to have other directories numbered `1`, `2`, and so on. Each directory here represents a set of points that colmap discovered are related to another. However, if you end up with more than one directory (i.e. a directory other than just `0`) *especially* if they contain files of similar sizes, it indicates colormap discovered 2 or more trees of related points but could not figure out how those sets of points relate with each other (in terms of their relative location). This indicates an issue with the original input video and may be a problem if the relevant trees of points are large.
  
# Modify `splatting_script` 
- Now, we need to modify `$WRK_DIR/gaussian-splatting/splatting_script` to fit your needs. This script runs `train.py` which runs the actual gaussian splatting training. Here are some things to keep in mind:
  - Gaussian splatting requires 24G+ VRam to train. So avoid changing the `-l gpu_memory=24G` option unless you know what you are doing.
  - `-l h_rt=` is the hard max limit for the job runtime. If training takes longer than this, it will be killed.
  - See https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/#job-options for all job options
  - Make sure to look at https://github.com/F1TenthBU/gaussian-splatting/ for further params you can pass to the train.py script

- Sample splatting_script based on the above structure
  ```shell
  #!/bin/bash -l

  #$ -P [to be replaced by your won project folder name]
  #$ -l h_rt=4:00:00
  #$ -m ea
  #$ -N splatting
  #$ -j y
  #$ -o splatting.logs
  #$ -pe omp 8
  #$ -l gpus=1
  #$ -l gpu_c=6.0
  #$ -l gpu_memory=24G
  
  module load miniconda/23.11.0
  module load cuda/11.8
  conda activate venv
  
  python train.py -s ./dataset/[your_dataset] --model_path ./output/[your_own_folder_name] --iterations 60000 --test_iterations 1000 7000 30000 45000 60000 --save_iterations 1000 7000 30000 45000 60000
  ```
# Submit splatting job
  `qsub $WRK_DIR/gaussian-splatting/splatting_script` or `qsub ./splatting_script` in `gaussian-splatting` directory.

# Visualize splatting results
- In `$WRK_DIR/gaussian-splatting/output/[your_own_folder_name]/point_cloud/iteration_[iteration_number]`, use the polygon file (`.ply`) for visualization on [WebGL 3D Gaussian Splat Viewer](https://antimatter15.com/splat/).

# Track current job
- Use `qstat -u <your_scc_username>` to keep track of the submitted job
  - If state is `qw`, then current job is queued. If state is `r`, then current job is running.
