#I. File list
```
.
|    optimization.py - Python script of phase 1 optimization
|    Dockerfile - Docker file to run python script
|    borg_files - Wrapper Files for the Borg MOEA*
     |    liborg.so - Borg MOEA shared object
     |    borg.py - Borg MOEA python wrapper

*The Borg MOEA is freely available to academic and non-commercial users at http://borgmoea.org/. Please download and
place the above files in the borg_files directory.
```

#II. How to Run
1. Download the associated data at <insert url later>. Note, that this already contains both the inputs and outputs 
of the phase 1 optimization.
2. Download and Run Docker Desktop. For more information on Docker visit: https://docs.docker.com/desktop/. To ensure 
that it is installed correctly go to the command prompt/terminal and enter $ docker --version
3. Change to the current working directory using command prompt/terminal $ cd <insert_path_to_\phase_1_optimization>
4. Build the docker image by running $ docker build --tag p1_optimization .
5. Run the image and mount the associated data you downloaded in step 1 by running (Note, this will take a long time)
$docker run -v <path_to_associated_data>\multiobjective_dam_hazard_io:/app_io p1_optimization 