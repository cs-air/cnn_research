# Links



*   [Overview of Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) 
*   [Anaconda and TensorFlow installation (Don't use the "conventional approach")](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc)
*   [TensorFlow installation (Hard mode)](https://medium.com/intel-student-ambassadors/installing-tensorflow-on-windows-with-anaconda-af6fa6280a4b) - Don’t use
*   [Basic TF project tutorial](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/)
*   [How CNNs work](https://www.youtube.com/watch?v=FmpDIaiMIeA)
*   [CNNs 2](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)
*   [CNNs 3](https://www.wandb.com/tutorial/convolutional-neural-networks)
*   [One hot encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
*   [cross-validation - k fold](https://machinelearningmastery.com/k-fold-cross-validation/)
*   [Convolutional Layers](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)
*   [Pooling](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)
*   [3blue video 1 - What is a Nerual Network?](https://www.youtube.com/watch?time_continue=261&v=aircAruvnKk&feature=emb_logo)
*   [3blue video 2 - Gradient Descent](https://www.youtube.com/watch?time_continue=1&v=IHZwWFHWa-w&feature=emb_logo)
*   [Online Book - Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
*   [Tutorial - Cat and Dog Recognition](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/)
*   [VGG model](https://arxiv.org/pdf/1409.1556.pdf)

A conda environment is a directory that contains a specific collection of conda packages that you have installed. For example, you may have one environment with NumPy 1.7 and its dependencies, and another environment with NumPy 1.6 for legacy testing. If you change one environment, your other environments are not affected. You can easily activate or deactivate environments, which is how you switch behttps://www.youtube.com/watch?v=FmpDIaiMIeAtween them. You can also share your environment with someone by giving them a copy of your environment.yaml file.


# Setup



*   Install Anaconda [from here](https://docs.anaconda.com/anaconda/install/)
    *   I used the python 3.7 version, the 64- Bit Graphical Installer and didn’t change any install settings
*   Open Anaconda Powershell Prompt or Anaconda Prompt
*   Update conda and anaconda
    *   **conda update conda **and type y if asked to proceed
    *   **conda update anaconda **and type y if asked to proceed
    *   **conda install keras **and type y if asked to proceed


# Making Keras and Tensorflow run on your GPU (optional but recommended)



*   [build configurations for linux and mac](https://www.tensorflow.org/install/source#tested_build_configurations)
*   [build configurations for windows](https://www.tensorflow.org/install/source_windows#gpu)

Easy way



*   Open conda prompt and type **conda create --name [env name of your choice] tensorflow-gpu**
*   **Conda activate [your env name]**
*   **conda install ipykernel scikit-learn matplotlib keras pillow**
*   Depending on the project, you may need to install other packages
*   Type **python**
*   Type **import tensorflow as tf**
*   Type **sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))**
*   The last few lines should display some information about your gpu
*   Type** print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))<code>)</code></strong>
*   The output should read<strong> Num GPUs Available: 1</strong>
*   I recommend making a new environment for each new project and only adding the packages you need. Our current environment has everything we need for the MNIST handwriting project below

Difficult way 



*   This process requires critical thinking skills! Compatibility may have changed since the time of this writing. Don’t blindly follow these steps, read the linked website to ensure everything will work!
*   Go to the [Tensorflow website](https://www.tensorflow.org/install/gpu) and check the hardware and software requirements
*   Update you [nvidia graphics card drivers](https://www.nvidia.com/Download/index.aspx)
*   [Download and install CUDA ](https://developer.nvidia.com/cuda-toolkit-archive)(I used 10.1 update 2)
*   [Download cuDNN](https://developer.nvidia.com/cudnn) (a free account is required) ([install instructions](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#abstract))
*   In anaconda prompt, type **conda install keras-gpu **(This can be in your base environment a custom one. Remember to activate this environment or install keras-gpu to any environments for projects you want to use your GPU in)
*   To verify the installation, open anaconda prompt and type **python**
*   Type **import tensorflow as tf**
*   Type** print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))<code>)</code></strong>
*   The output should read<strong> Num GPUs Available: 1</strong>
*   [semi outdated tutorial](https://www.codingforentrepreneurs.com/blog/install-tensorflow-gpu-windows-cuda-cudnn/)


# <strong><span style="text-decoration:underline;">Important Notes for the Following Projects</span></strong>



*   If you run into problems, make your code into a python script and run it from the command line or terminal. Jupyter notebook gives extremely little feedback in error messages! 
*   If you receive the following error :

        **_ValueError: Only know how to handle extensions: ['png']; with Pillow installed matplotlib can handle more images_**


    Check your environment and apply pillow to your environment. Matplotlib requires PIL(Python Imaging Library) to work with .jpg format. Install pillow , if using conda in command line try conda install pillow, if using Anaconda Navigator go to your environment and type pillow and then press apply. 

*   Make sure your train folder is in the same folder as your juypter notebook .ipynb file
*   If you have an older graphics card and you receive an error messages that says your “compute capability is X, and the required compute capability is  >X” (or your GPU just isn’t running when the program is training) try  
    *   **conda install cudatoolkit=9.2**
    *   **Conda install tensorflow-gpu=1.12**
    *   **Conda install keras=2.2.4**
    *   These steps allowed the dog cat recognition program to run on my GTX 770, which has a terrible compute capability of 3.0. 
*   If you get OOM (out of memory) errors, reduce the batch size. I had to go from a batch size of 64 to 32 in the dog cat recognition program


# Preparing for the MNIST Handwritten Digit Classification Tutorial - Using Command Line



*   Create an environment (we will name ours **handwriting) **You can create an environment by typing **conda create -n [your environment name] [any packages you want to include]. **However we will be cloning our base environment using **conda create -n  [your environment name] --clone [name of environment being cloned].**
*   **conda create -n handwriting --clone base**
    *   This will take a while
*   **cd **to the location you want your project to be and create a directory there
*   open jupyter notebook by typing **jupyter notebook**
*   In jupyter notebook, click new, and then python3
*   Paste your code in cells as you prefer, following the tutorial. More information about how to use jupyter notebook can be found **[here](https://www.dataquest.io/blog/jupyter-notebook-tutorial/).**

A validation dataset is a sample of data held back from training your model that is used to give an estimate of model skill while tuning model’s hyperparameters.

The validation dataset is different from the test dataset that is also held back from the training of the model, but is instead used to give an unbiased estimate of the skill of the final tuned model when comparing or selecting between final models.


# Preparing for the MNIST Handwritten Digit Classification Tutorial - Using Anaconda Navigator



*   After installing Anaconda Navigator you will open the application. 
*   Go to **Environments**, press** create** and place the name you want for the environment and select the python you will use with it.
*   Next, you will click on the environment you created, where it says installed press not installed. Then, where it has search packages type in TensorFlow. Locate the package that says TensorFlow and press it and apply it. (wait until the packages have fully imported and then proceed to repeat the step with any other packages you would like in your environment. 
*   Example. Tensorflow, Keras, matplotlib, pandas, statsmodels, scikit-learn etc. 
*   When applying these packages it is important to keep checking what you already have installed because some that you may not see pop up will already be applied with other packages. 
*   Once, all the packages you wanted to add to the environment are added then you press Home and where it says applications you look for the environment you created. 
*   When you press the environment you will see Jupyter notebook needs to be installed just press install and wait for it to install.
*   Once, everything has completed. You will check by going to your start menu and find anaconda  and scroll down to the Jupyter notebook that  has your environment name and now you can use it. 
*   **Check: **to ensure everything has been properly imported in jupyter notebook type in the two lines below and then press run. 

    **Import TensorFlow as tf **


    **From TensorFlow import Keras**

*   **Note: **if TensorFlow is not able to install on your computer for some reason. Then you will install **Theano **instead and that works with Keras.

**<span style="text-decoration:underline;">CNN coding info:</span>**

Train x = data set

Train y = the labels of train x

Test x = data set

Test y = labels of test x



*   Cross validation or k validation : [cross validation - k fold](https://machinelearningmastery.com/k-fold-cross-validation/)

Information of using Juypter notebook on VsCode

[https://github.com/Microsoft/vscode-python/blob/master/PYTHON_INTERACTIVE_TROUBLESHOOTING.md](https://github.com/Microsoft/vscode-python/blob/master/PYTHON_INTERACTIVE_TROUBLESHOOTING.md)



*   Data loading and reshaping to change to binary vectors
*   Tensorflow Core 2.20 v : [tf.keras.utils.to_categorical](https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical)
*   
*   


# Dog and Cat Recognition



*   [Tutorial - Cat and Dog Recognition](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/)
*   [VGG model](https://arxiv.org/pdf/1409.1556.pdf)
*   Make a new environment, we will clone the environment we used for digit classification
*   Download [the dataset](https://www.kaggle.com/c/dogs-vs-cats/rules) by following the link and clicking “download all” (you need to create an account)
*   Save file
*   Unzip the train.zip dataset, this is the only one we need
*   Only do the optional photo pre-processing if you have over 12GB of memory


# **<span style="text-decoration:underline;">Important Notes</span>**



*   If you run into problems, make your code into a python script and run it from the command line or terminal. Jupyter notebook gives extremely little feedback in error messages! 
*   If you receive the following error :

        **_ValueError: Only know how to handle extensions: ['png']; with Pillow installed matplotlib can handle more images_**


    Check your environment and apply pillow to your environment. Matplotlib requires PIL(Python Imaging Library) to work with .jpg format. Install pillow , if using conda in command line try conda install pillow, if using Anaconda Navigator go to your environment and type pillow and then press apply. 

*   Make sure your train folder is in the same folder as your juypter notebook .ipynb file
*   If you have an older graphics card and you receive an error messages that says your “compute capability is X, and the required compute capability is  >X” (or your GPU just isn’t running when the program is training) try  
    *   **conda install cudatoolkit=9.2**
    *   **Conda install tensorflow-gpu=1.12**
    *   **Conda install keras=2.2.4**
    *   These steps allowed the dog cat recognition program to run on my GTX 770, which has a terrible compute capability of 3.0. 
*   If you get OOM (out of memory) errors, reduce the batch size. I had to go from a batch size of 64 to 32 in the dog cat recognition program

**Sites:**

[Python downloads](https://www.python.org/downloads/)

[Installing Homebrew](https://osxdaily.com/2018/03/07/how-install-homebrew-mac-os/)

[Mac OS - Homebrew - Python](https://osxdaily.com/2018/06/13/how-install-update-python-3x-mac/)

[Mac OS - Command Line- Anaconda](https://towardsdatascience.com/how-to-successfully-install-anaconda-on-a-mac-and-actually-get-it-to-work-53ce18025f97)

[Setting up MacOs-Keras-TensorFlow](https://www.dataweekends.com/blog/2017/03/09/set-up-your-mac-for-deep-learning-with-python-keras-and-tensorflow)

**Mac OS system updates**:

Check terminal for the type of python you have by typing 

**python --version**

Next, click the link below

[Anaconda & python](https://www.anaconda.com/products/individual)

I clicked 64-bit graphical installer (442 MB) but, you can use the 64-command line installer (430 MB)

I used Homebrew to install python 3.8.

**<span style="text-decoration:underline;">HOMEBREW</span>**

Simplifies the process of installing command-line software and tools on a Mac. 

**Requirements for installing Homebrew on Mac OS: **



*   Mac running Mac OS X 10.10 or later.
*   Command Line Tools have to be installed on the Mac ( independently or through Xcode)
*   Knowledge of the command line and using bash zsh

**How to install Homebrew on Mac OS: **



*   Install through ruby and curl
1. Open the “Terminal” application, found in / Applications/ Utilities/ or type in the search box “terminal”, then click Terminal.app
2. Enter the following command into a single line of the terminal:

    **[For MacOS Catalina, macOS Mojave, and MacOS Big Sur**]


    /bin/bash -c “$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/master/install.sh](https://raw.githubusercontent.com/Homebrew/install/master/install.sh))”


    **[For MacOS High Sierra, Sierra, El Capitan, and earlier**]


    /usr/bin/ruby -e “$ (curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/master/install](https://raw.githubusercontent.com/Homebrew/install/master/install))”

3. Hit enter and there should be a lot of lines about what the script will install & where it will install. Hit enter again to agree or hit escape to cancel.
4. Enter the administrator password (required by sudo execution) to begin the installation. 

    The installation of Homebrew will take a little while (depending on the speed of the Mac & internet)

5. When done there will be an “Installation Successful” message.
6. Read the help documentation if needed by the following command. 

    brew help

7. Install software packages through Homebrew

**How to install software packages through Homebrew on Mac**:



1. Type 

    **brew install [package name]**

2. Using python as an example:

	**brew install python3**



3. After python3 installs, you can run it by using

    **python3**

4. The default version of  python 2.7 is preinstalled with Mac OS and Mac OS X and is still installed and can run by using 

    **python**

5. After python has been installed with Homebrew or with the package installer, you can check the updated version of python by using

    **python3 --version**


    **[side note: both Python will coexist without conflict]**


**How to install anaconda by Homebrew**



1. Install anaconda by using [ install it in /usr/local aka $HOME ]

    **brew cask install anaconda**

2. Copy the path of anaconda

    Something like this will show 


    **/usr/local/anaconda3**

3. Setup the environment path 

**How to install anaconda using the command line:**



1. Depending on your version of python you can install (if using python 3.7 or 2.7)

    For **python 2.7**


    **bash ~/Downloads/Anaconda2-2019.03-MacOSX-x86_64.sh**


    For** python 3.7**


    **Bash ~/Downloads/Anaconda3-2019.03-MacOSX-x86_64.sh**

2. Review the license agreement accept it by pressing “Enter” until you get to the end then type “yes”
3. If you are happy with the location you will be prompted to hit “Enter”.

    Keep in mind you can change the location or cancel the installation by entering CTRL-C.

4. The installation will take some time but will prompt “Do You wish the installer to initialize Anaconda3 by running conda init?” and you will type “yes” when asked.
5. When finished close out of the terminal and open a new one 
6. Next, to check if it works type 

    **conda list**


    A list should pop up with packages and versions that are installed in your environment. 



# Turing



*   Log in and type **ssh gpu**
*   **conda create -n tf-gpu2 tensorflow=2.0.0 matplotlib scikit-learn keras pillow tensorflow-gpu=2.0.0**
*   Use filezilla to upload your data to turing
*   To run your code, navigate to it on turing and type **python [yourcode.py]**


# Object Detection



*   [darknet wiki](https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection)
*   If you don’t have Ubuntu, [install the ubuntu shell on windows](https://ubuntu.com/tutorials/ubuntu-on-windows#1-overview)
*   [install opencv](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/#installing-opencv-from-the-source)
*   [install CUDA (if you haven't already)](https://developer.nvidia.com/cuda-downloads)
*   NOTE THAT IF YOU HAVE ISSUES WITH OPENCV OR CUDA, YOU DON’T HAVE TO INSTALL THEM
    *   Opencv allows darknet to display detections on-screen, instead of creating a .png you have to manually open (predictions.png). You also won’t be able to detect objects in a video stream
    *   CUDA allows darknet to use your GPU, which significantly increases performance.
*   [install darknet](https://pjreddie.com/darknet/install/)        [using this repository!!!](https://github.com/AlexeyAB/darknet)
    *   **git clone [https://github.com/AlexeyAB/darknet.git](https://github.com/AlexeyAB/darknet.git)**
    *   **cd darknet**
    *   Open Makefile and set GPU=1 and OPENCV=1 in the first few lines
        *   If you didn’t install CUDA,  GPU=0
        *   If you didn’t install opencv, OPENCV=0
    *   **Make **[(more info can be found in the readme)](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-cmake)
        *   It will look like there ar lots of errors, ignore them unless it gives you an error at the end.
    *   Test with  **./darknet imtest data/eagle.jpg**
        *   There should be several pictures of an eagle
*   Set up [YOLO](https://pjreddie.com/darknet/yolo/)
    *   **cd darknet**
    *   Download training data **wget https://pjreddie.com/media/files/yolov3.weights**
    *   To detect objects in a video stream, use **./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights**
        *   Objects in your webcam should be labeled in realtime!
        *   You  can also do this with a video with **./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights &lt;video file>**
        *   This won’t work if you didn’t install opencv
        *   Your “video will only update once every 20 seconds or so if you didn’t install CUDA
    *   To detect objects in a still image, use **./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg**
        *   The data folder has other .jpgs you can test. You can add your own.
*   Set up [Nightmare](https://pjreddie.com/darknet/nightmare/) (don’t do this without a massive amount of VRAM)
    *   Nightmare is just for fun -- It makes your images look like Google’s deep dream.
    *   **cd darknet**
    *   **wget [http://pjreddie.com/media/files/vgg-conv.weights](http://pjreddie.com/media/files/vgg-conv.weights)**
    *   Test with **./darknet nightmare cfg/vgg-conv.cfg vgg-conv.weights data/dog.jpg 7**
    *   Other tweaks and settings can be found on the Nightmare page
    *   

Redmon, J. and Farhadi, A., 2018. _Yolov3: An Incremental Improvement_. [online] arXiv.org. Available at: &lt;[https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)>.
