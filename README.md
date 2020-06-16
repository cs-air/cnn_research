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



*   Open conda prompt and type **conda create --name tf_env tensorflow**
*   **conda install tensorflow-gpu ipykernel scikit-learn matplotlib keras**
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
*   Make a new environment, we will use a clone of the base environment
*   Download [the dataset](https://www.kaggle.com/c/dogs-vs-cats/rules) by following the link and clicking “download all” (you need to create an account)
*   Save file
*   Unzip the train.zip dataset, this is the only one we need
*   Only do the optional photo pre-processing if you have over 12GB of memory

**<span style="text-decoration:underline;">Side note</span>**: 



*   If you receive the following error :

        **_ValueError: Only know how to handle extensions: ['png']; with Pillow installed matplotlib can handle more images_**


    Check your environment and apply pillow to your environment. Matplotlib requires PIL(Python Imaging Library) to work with .jpg format. Install pillow , if using conda in command line try conda install pillow, if using Anaconda Navigator go to your environment and type pillow and then press apply. 

*   Make sure your train folder is in the same folder as your juypter notebook .ipynb file

Python 3.7.6 64-bit ('tf-env4':conda)
