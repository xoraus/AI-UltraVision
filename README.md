
# Using The Super-Resolution Convolutional Neural Network for Image Restoration

Welcome to this tutorial on single-image super-resolution. The goal of super-resolution (SR) is to recover a high-resolution image from a low-resolution input, or as they might say on any modern crime show,  **enhance!**

To accomplish this goal, we will be deploying the super-resolution convolution neural network (SRCNN) using Keras. This network was published in the paper, “Image Super-Resolution Using Deep Convolutional Networks” by Chao Dong, et al. in 2014. You can read the full paper at  [https://arxiv.org/abs/1501.00092](https://arxiv.org/abs/1501.00092).

[](https://www.datadriveninvestor.com/2019/03/03/editors-pick-5-machine-learning-books/?source=post_page-----ff1e8420d846----------------------)


As the title suggests, the SRCNN is a deep convolutional neural network that learns the end-to-end mapping of low-resolution to high-resolution images. As a result, we can use it to improve the image quality of low-resolution images. To evaluate the performance of this network, we will be using three image quality metrics: peak signal to noise ratio (PSNR), mean squared error (MSE), and the structural similarity (SSIM) index.

In brief, with better SR approach, we can get a better quality of a larger image even we only get a small image originally.

![](https://miro.medium.com/max/30/1*FzN1KFBv_q0IramC4nxHRw.png?q=20)

![](https://miro.medium.com/max/700/1*FzN1KFBv_q0IramC4nxHRw.png)

Furthermore, we will be using OpenCV, the Open Source Computer Vision Library. OpenCV was originally developed by Intel and is used for many real-time computer vision applications. In this particular project, we will be using it to pre and post process our images. As you will see later, we will frequently be converting our images back and forth between the RGB, BGR, and YCrCb color spaces. This is necessary because the SRCNN network was trained on the luminance (Y) channel in the YCrCb color space.

During this project, you will learn how to:

-   use the PSNR, MSE, and SSIM image quality metrics,
-   process images using OpenCV,
-   convert between the RGB, BGR, and YCrCb color spaces,
-   build deep neural networks in Keras,
-   deploy and evaluate the SRCNN network

# The SRCNN Network

![](https://miro.medium.com/max/30/1*mZJO-i6ImYyXHorv4H1q_Q.png?q=20)

![](https://miro.medium.com/max/700/1*mZJO-i6ImYyXHorv4H1q_Q.png)

# 1. Importing Packages

Let’s dive right in! In this first cell, we will import the libraries and packages we will be using in this project and print their version numbers. This is an important step to make sure we are all on the same page; furthermore, it will help others reproduce the results we obtain.

_# check package versions_  
**import** **sys**  
**import** **keras**  
**import** **cv2**  
**import** **numpy**  
**import** **matplotlib**  
**import** **skimage**  
  
print('Python: **{}**'.format(sys.version))  
print('Keras: **{}**'.format(keras.__version__))  
print('OpenCV: **{}**'.format(cv2.__version__))  
print('NumPy: **{}**'.format(numpy.__version__))  
print('Matplotlib: **{}**'.format(matplotlib.__version__))  
print('Scikit-Image: **{}**'.format(skimage.__version__))

## _Import the necessary packages_

**from** **keras.models** **import** Sequential  
**from** **keras.layers** **import** Conv2D  
**from** **keras.optimizers** **import** Adam  
**from** **skimage.measure** **import** compare_ssim **as** ssim  
**from** **matplotlib** **import** pyplot **as** plt  
**import** **cv2**  
**import** **numpy** **as** **np**  
**import** **math**  
**import** **os**  
  
_# python magic function, displays pyplot figures in the notebook_  
%matplotlib inline

# 2. Image Quality Metrics

To start, let's define a couple of functions that we can use to calculate the PSNR, MSE, and SSIM. The structural similarity (SSIM) index was imported directly from the scikit-image library; however, we will have to define our own functions for the PSNR and MSE. Furthermore, we will wrap all three of these metrics into a single function that we can call later.

_# define a function for peak signal-to-noise ratio (PSNR)_  
**def** psnr(target, ref):  
           
    _# assume RGB image_  
    target_data = target.astype(float)  
    ref_data = ref.astype(float)  
  
    diff = ref_data - target_data  
    diff = diff.flatten('C')  
  
    rmse = math.sqrt(np.mean(diff ** 2.))  
  
    **return** 20 * math.log10(255. / rmse)  
  
_# define function for mean squared error (MSE)_  
**def** mse(target, ref):  
    _# the MSE between the two images is the sum of the squared difference between the two images_  
    err = np.sum((target.astype('float') - ref.astype('float')) ** 2)  
    err /= float(target.shape[0] * target.shape[1])  
      
    **return** err  
  
_# define function that combines all three image quality metrics_  
**def** compare_images(target, ref):  
    scores = []  
    scores.append(psnr(target, ref))  
    scores.append(mse(target, ref))  
    scores.append(ssim(target, ref, multichannel =**True**))  
      
    **return** scores

# 3. Preparing Images

For this project, we will be using the same images that were used in the original SRCNN paper. We can download these images from  [http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html). The .zip file identified as the MATLAB code contains the images we want. Copy both the Set5 and Set14 datasets into a new folder called ‘source’.

Now that we have some images, we want to produce low-resolution versions of these same images. We can accomplish this by resizing the images, both downwards and upwards, using OpeCV. There are several interpolation methods that can be used to resize images; however, we will be using bilinear interpolation.

Once we produce these low-resolution images, we can save them in a new folder.

_# prepare degraded images by introducing quality distortions via resizing_  
  
**def** prepare_images(path, factor):  
      
    _# loop through the files in the directory_  
    **for** file **in** os.listdir(path):  
          
        _# open the file_  
        img = cv2.imread(path + '/' + file)  
          
        _# find old and new image dimensions_  
        h, w, _ = img.shape  
        new_height = h / factor  
        new_width = w / factor  
          
        _# resize the image - down_  
        img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR)  
          
        _# resize the image - up_  
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR)  
          
        _# save the image_  
        print('Saving **{}**'.format(file))  
        cv2.imwrite('images/**{}**'.format(file), img)prepare_images('source/', 2)

# 3. Testing Low-Resolution Images

To ensure that our image quality metrics are being calculated correctly and that the images were effectively degraded, let's calculate the PSNR, MSE, and SSIM between our reference images and the degraded images that we just prepared.

_# test the generated images using the image quality metrics_  
  
**for** file **in** os.listdir('images/'):  
      
    _# open target and reference images_  
    target = cv2.imread('images/**{}**'.format(file))  
    ref = cv2.imread('source/**{}**'.format(file))  
      
    _# calculate score_  
    scores = compare_images(target, ref)  
  
    _# print all three scores with new line characters (\n)_   
    print('**{}\n**PSNR: **{}\n**MSE: **{}\n**SSIM: **{}\n**'.format(file, scores[0], scores[1], scores[2]))

# 4. Building the SRCNN Model

Now that we have our low-resolution images and all three image quality metrics functioning properly, we can start building the SRCNN. In Keras, it’s as simple as adding layers one after the other. The architecture and hyperparameters of the SRCNN network can be obtained from the publication referenced above.

_# define the SRCNN model_  
**def** model():  
      
    _# define model type_  
    SRCNN = Sequential()  
      
    _# add model layers_  
    SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',  
                     activation='relu', padding='valid', use_bias=**True**, input_shape=(**None**, **None**, 1)))  
    SRCNN.add(Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',  
                     activation='relu', padding='same', use_bias=**True**))  
    SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',  
                     activation='linear', padding='valid', use_bias=**True**))  
      
    _# define optimizer_  
    adam = Adam(lr=0.0003)  
      
    _# compile model_  
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])  
      
    **return** SRCNN

# 5. Deploying the SRCNN

Now that we have defined our model, we can use it for single-image super-resolution. However, before we do this, we will need to define a couple of image processing functions. Furthermore, it will be necessary to preprocess the images extensively before using them as inputs to the network. This processing will include cropping and color space conversions.

Additionally, to save us the time it takes to train a deep neural network, we will be loading pre-trained weights for the SRCNN. These weights can be found at the following GitHub page:  [https://github.com/MarkPrecursor/SRCNN-keras](https://github.com/MarkPrecursor/SRCNN-keras)

Once we have tested our network, we can perform single-image super-resolution on all of our input images. Furthermore, after processing, we can calculate the PSNR, MSE, and SSIM on the images that we produce. We can save these images directly or create subplots to conveniently display the original, low-resolution, and high-resolution images side by side.

_# define necessary image processing functions_  
  
**def** modcrop(img, scale):  
    tmpsz = img.shape  
    sz = tmpsz[0:2]  
    sz = sz - np.mod(sz, scale)  
    img = img[0:sz[0], 1:sz[1]]  
    **return** img  
  
  
**def** shave(image, border):  
    img = image[border: -border, border: -border]  
    **return** img_# define main prediction function_  
  
**def** predict(image_path):  
      
    _# load the srcnn model with weights_  
    srcnn = model()  
    srcnn.load_weights('3051crop_weight_200.h5')  
      
    _# load the degraded and reference images_  
    path, file = os.path.split(image_path)  
    degraded = cv2.imread(image_path)  
    ref = cv2.imread('source/**{}**'.format(file))  
      
    _# preprocess the image with modcrop_  
    ref = modcrop(ref, 3)  
    degraded = modcrop(degraded, 3)  
      
    _# convert the image to YCrCb - (srcnn trained on Y channel)_  
    temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)  
      
    _# create image slice and normalize_   
    Y = numpy.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)  
    Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255  
      
    _# perform super-resolution with srcnn_  
    pre = srcnn.predict(Y, batch_size=1)  
      
    _# post-process output_  
    pre *= 255  
    pre[pre[:] > 255] = 255  
    pre[pre[:] < 0] = 0  
    pre = pre.astype(np.uint8)  
      
    _# copy Y channel back to image and convert to BGR_  
    temp = shave(temp, 6)  
    temp[:, :, 0] = pre[0, :, :, 0]  
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)  
      
    _# remove border from reference and degraged image_  
    ref = shave(ref.astype(np.uint8), 6)  
    degraded = shave(degraded.astype(np.uint8), 6)  
      
    _# image quality calculations_  
    scores = []  
    scores.append(compare_images(degraded, ref))  
    scores.append(compare_images(output, ref))  
      
    _# return images and scores_  
    **return** ref, degraded, output, scoresref, degraded, output, scores = predict('images/flowers.bmp')  
  
_# print all scores for all images_  
print('Degraded Image: **\n**PSNR: **{}\n**MSE: **{}\n**SSIM: **{}\n**'.format(scores[0][0], scores[0][1], scores[0][2]))  
print('Reconstructed Image: **\n**PSNR: **{}\n**MSE: **{}\n**SSIM: **{}\n**'.format(scores[1][0], scores[1][1], scores[1][2]))  
  
  
_# display images as subplots_  
fig, axs = plt.subplots(1, 3, figsize=(20, 8))  
axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))  
axs[0].set_title('Original')  
axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))  
axs[1].set_title('Degraded')  
axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))  
axs[2].set_title('SRCNN')  
  
_# remove the x and y ticks_  
**for** ax **in** axs:  
    ax.set_xticks([])  
    ax.set_yticks([])

![](https://miro.medium.com/max/30/1*j9a0kGhWcG8lEDLf5eqcfg.png?q=20)

![](https://miro.medium.com/max/1137/1*j9a0kGhWcG8lEDLf5eqcfg.png)

**for** file **in** os.listdir('images'):  
      
    _# perform super-resolution_  
    ref, degraded, output, scores = predict('images/**{}**'.format(file))  
      
    _# display images as subplots_  
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))  
    axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))  
    axs[0].set_title('Original')  
    axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))  
    axs[1].set_title('Degraded')  
    axs[1].set(xlabel = 'PSNR: **{}\n**MSE: **{}** **\n**SSIM: **{}**'.format(scores[0][0], scores[0][1], scores[0][2]))  
    axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))  
    axs[2].set_title('SRCNN')  
    axs[2].set(xlabel = 'PSNR: **{}** **\n**MSE: **{}** **\n**SSIM: **{}**'.format(scores[1][0], scores[1][1], scores[1][2]))  
  
    _# remove the x and y ticks_  
    **for** ax **in** axs:  
        ax.set_xticks([])  
        ax.set_yticks([])  
        
    print('Saving **{}**'.format(file))  
    fig.savefig('output/**{}**.png'.format(os.path.splitext(file)[0]))   
    plt.close()

![](https://miro.medium.com/max/30/1*8lpeTi2p_F7AhE2o_7tJ9Q.png?q=20)

![](https://miro.medium.com/max/845/1*8lpeTi2p_F7AhE2o_7tJ9Q.png)

References:  
[1]  [https://en.wikipedia.org/wiki/Convolutional_neural_network](https://en.wikipedia.org/wiki/Convolutional_neural_network)

[2]  [http://keras.io/examples/cifar10_cnn/](http://keras.io/examples/cifar10_cnn/)

[3]  [http://keras.io/layers/convolutional/](http://keras.io/layers/convolutional/)

[4]  [https://arxiv.org/abs/1501.00092](https://arxiv.org/abs/1501.00092)

[5]  [http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html).

[6] [2014 ECCV] [SRCNN]  
[Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/pdf/1501.00092)
