{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Import"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Successful imported...\n"
     ]
    }
   ],
   "source": [
    "from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D\n",
    "from keras.layers.advanced_activations import LeakyReLU\n",
    "from keras.layers.convolutional import UpSampling2D, Conv2D\n",
    "from keras.models import Sequential, Model, load_model\n",
    "from keras.optimizers import Adam\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "import os\n",
    "print(\"Successful imported...\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Definition of parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Preview image Frame\n",
    "PREVIEW_ROWS = 4\n",
    "PREVIEW_COLS = 7\n",
    "PREVIEW_MARGIN = 4\n",
    "SAVE_FREQ = 100\n",
    "# Size vector to generate images from\n",
    "NOISE_SIZE = 100\n",
    "# Configuration\n",
    "EPOCHS = 1000 # number of iterations\n",
    "BATCH_SIZE = 32\n",
    "GENERATE_RES = 3\n",
    "IMAGE_SIZE = 128 # rows/cols\n",
    "IMAGE_CHANNELS = 3"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Import training_data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data imported\n"
     ]
    }
   ],
   "source": [
    "training_data = np.load(\"vangogh_data.npy\")\n",
    "print(\"Data imported\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Function build_discriminator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_discriminator(image_shape):\n",
    "    model = Sequential()\n",
    "    model.add(Conv2D(32, kernel_size=3, strides=2,\n",
    "    input_shape=image_shape, padding='same'))\n",
    "    model.add(LeakyReLU(alpha=0.2))\n",
    "    model.add(Dropout(0.25))\n",
    "    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))\n",
    "    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))\n",
    "    model.add(BatchNormalization(momentum=0.8))\n",
    "    model.add(LeakyReLU(alpha=0.2))\n",
    "    model.add(Dropout(0.25))\n",
    "    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))\n",
    "    model.add(BatchNormalization(momentum=0.8))\n",
    "    model.add(LeakyReLU(alpha=0.2))\n",
    "    model.add(Dropout(0.25))\n",
    "    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))\n",
    "    model.add(BatchNormalization(momentum=0.8))\n",
    "    model.add(LeakyReLU(alpha=0.2))\n",
    "    model.add(Dropout(0.25))\n",
    "    model.add(Conv2D(512, kernel_size=3, strides=1, padding='same'))\n",
    "    model.add(BatchNormalization(momentum=0.8))\n",
    "    model.add(LeakyReLU(alpha=0.2))\n",
    "    model.add(Dropout(0.25))\n",
    "    model.add(Flatten())\n",
    "    model.add(Dense(1, activation=\"sigmoid\"))\n",
    "    input_image = Input(shape=image_shape)\n",
    "    validity = model(input_image)\n",
    "    return Model(input_image, validity)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Function build_generator "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_generator(noise_size, channels):\n",
    "    model = Sequential()\n",
    "    model.add(Dense(4 * 4 * 256, activation='relu',       input_dim=noise_size))\n",
    "    model.add(Reshape((4, 4, 256)))\n",
    "    model.add(UpSampling2D())\n",
    "    model.add(Conv2D(256, kernel_size=3, padding='same'))\n",
    "    model.add(BatchNormalization(momentum=0.8))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(UpSampling2D())\n",
    "    model.add(Conv2D(256, kernel_size=3, padding='same'))\n",
    "    model.add(BatchNormalization(momentum=0.8))\n",
    "    model.add(Activation('relu'))\n",
    "    for i in range(GENERATE_RES):\n",
    "         model.add(UpSampling2D())\n",
    "         model.add(Conv2D(256, kernel_size=3, padding='same'))\n",
    "         model.add(BatchNormalization(momentum=0.8))\n",
    "         model.add(Activation('relu'))\n",
    "    model.summary()\n",
    "    model.add(Conv2D(channels, kernel_size=3, padding='same'))\n",
    "    model.add(Activation('tanh'))\n",
    "    input = Input(shape=(noise_size,))\n",
    "    generated_image = model(input)\n",
    "    \n",
    "    return Model(input, generated_image)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## helper_function save_images "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def save_images(cnt, noise):\n",
    "    image_array = np.full((\n",
    "        PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE + PREVIEW_MARGIN)),\n",
    "        PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE + PREVIEW_MARGIN)), 3),\n",
    "        255, dtype=np.uint8)\n",
    "    generated_images = generator.predict(noise)\n",
    "    generated_images = 0.5 * generated_images + 0.5\n",
    "    image_count = 0\n",
    "    for row in range(PREVIEW_ROWS):\n",
    "        for col in range(PREVIEW_COLS):\n",
    "            r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN\n",
    "            c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN\n",
    "            image_array[r:r + IMAGE_SIZE, c:c +\n",
    "                        IMAGE_SIZE] = generated_images[image_count] * 255\n",
    "            image_count += 1\n",
    "    output_path = 'output'\n",
    "    if not os.path.exists(output_path):\n",
    "        os.makedirs(output_path)\n",
    "    filename = os.path.join(output_path, f\"trained-{cnt}.png\")\n",
    "    im = Image.fromarray(image_array)\n",
    "    im.save(filename)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Compile models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_3\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_3 (Dense)              (None, 4096)              413696    \n",
      "_________________________________________________________________\n",
      "reshape_1 (Reshape)          (None, 4, 4, 256)         0         \n",
      "_________________________________________________________________\n",
      "up_sampling2d_5 (UpSampling2 (None, 8, 8, 256)         0         \n",
      "_________________________________________________________________\n",
      "conv2d_16 (Conv2D)           (None, 8, 8, 256)         590080    \n",
      "_________________________________________________________________\n",
      "batch_normalization_13 (Batc (None, 8, 8, 256)         1024      \n",
      "_________________________________________________________________\n",
      "activation_6 (Activation)    (None, 8, 8, 256)         0         \n",
      "_________________________________________________________________\n",
      "up_sampling2d_6 (UpSampling2 (None, 16, 16, 256)       0         \n",
      "_________________________________________________________________\n",
      "conv2d_17 (Conv2D)           (None, 16, 16, 256)       590080    \n",
      "_________________________________________________________________\n",
      "batch_normalization_14 (Batc (None, 16, 16, 256)       1024      \n",
      "_________________________________________________________________\n",
      "activation_7 (Activation)    (None, 16, 16, 256)       0         \n",
      "_________________________________________________________________\n",
      "up_sampling2d_7 (UpSampling2 (None, 32, 32, 256)       0         \n",
      "_________________________________________________________________\n",
      "conv2d_18 (Conv2D)           (None, 32, 32, 256)       590080    \n",
      "_________________________________________________________________\n",
      "batch_normalization_15 (Batc (None, 32, 32, 256)       1024      \n",
      "_________________________________________________________________\n",
      "activation_8 (Activation)    (None, 32, 32, 256)       0         \n",
      "_________________________________________________________________\n",
      "up_sampling2d_8 (UpSampling2 (None, 64, 64, 256)       0         \n",
      "_________________________________________________________________\n",
      "conv2d_19 (Conv2D)           (None, 64, 64, 256)       590080    \n",
      "_________________________________________________________________\n",
      "batch_normalization_16 (Batc (None, 64, 64, 256)       1024      \n",
      "_________________________________________________________________\n",
      "activation_9 (Activation)    (None, 64, 64, 256)       0         \n",
      "_________________________________________________________________\n",
      "up_sampling2d_9 (UpSampling2 (None, 128, 128, 256)     0         \n",
      "_________________________________________________________________\n",
      "conv2d_20 (Conv2D)           (None, 128, 128, 256)     590080    \n",
      "_________________________________________________________________\n",
      "batch_normalization_17 (Batc (None, 128, 128, 256)     1024      \n",
      "_________________________________________________________________\n",
      "activation_10 (Activation)   (None, 128, 128, 256)     0         \n",
      "=================================================================\n",
      "Total params: 3,369,216\n",
      "Trainable params: 3,366,656\n",
      "Non-trainable params: 2,560\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "image_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)\n",
    "optimizer = Adam(1.5e-4, 0.5)\n",
    "discriminator = build_discriminator(image_shape)\n",
    "discriminator.compile(loss='binary_crossentropy',\n",
    "optimizer=optimizer, metrics=['accuracy'])\n",
    "generator = build_generator(NOISE_SIZE, IMAGE_CHANNELS)\n",
    "random_input = Input(shape=(NOISE_SIZE,))\n",
    "generated_image = generator(random_input)\n",
    "discriminator.trainable = False\n",
    "validity = discriminator(generated_image)\n",
    "combined = Model(random_input, validity)\n",
    "combined.compile(loss='binary_crossentropy',\n",
    "optimizer=optimizer, metrics=['accuracy'])\n",
    "y_real = np.ones((BATCH_SIZE, 1))\n",
    "y_fake = np.zeros((BATCH_SIZE, 1))\n",
    "fixed_noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Train the model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 epoch, Discriminator accuracy: 18.75, Generator accuracy: 100.0\n",
      "100 epoch, Discriminator accuracy: 100.0, Generator accuracy: 96.875\n",
      "200 epoch, Discriminator accuracy: 100.0, Generator accuracy: 100.0\n",
      "300 epoch, Discriminator accuracy: 100.0, Generator accuracy: 100.0\n",
      "400 epoch, Discriminator accuracy: 100.0, Generator accuracy: 100.0\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-14-20da49fc268e>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     12\u001b[0m     \u001b[0mdiscriminator_metric\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m0.5\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0madd\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdiscriminator_metric_real\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdiscriminator_metric_generated\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     13\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 14\u001b[0;31m     \u001b[0mgenerator_metric\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcombined\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtrain_on_batch\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnoise\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_real\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     15\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mepoch\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0mSAVE_FREQ\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     16\u001b[0m         \u001b[0msave_images\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcnt\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfixed_noise\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py\u001b[0m in \u001b[0;36mtrain_on_batch\u001b[0;34m(self, x, y, sample_weight, class_weight, reset_metrics, return_dict)\u001b[0m\n\u001b[1;32m   1725\u001b[0m                                                     class_weight)\n\u001b[1;32m   1726\u001b[0m       \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtrain_function\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmake_train_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1727\u001b[0;31m       \u001b[0mlogs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtrain_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1728\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1729\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mreset_metrics\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *args, **kwds)\u001b[0m\n\u001b[1;32m    826\u001b[0m     \u001b[0mtracing_count\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexperimental_get_tracing_count\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    827\u001b[0m     \u001b[0;32mwith\u001b[0m \u001b[0mtrace\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mTrace\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_name\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mtm\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 828\u001b[0;31m       \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_call\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    829\u001b[0m       \u001b[0mcompiler\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m\"xla\"\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_experimental_compile\u001b[0m \u001b[0;32melse\u001b[0m \u001b[0;34m\"nonXla\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    830\u001b[0m       \u001b[0mnew_tracing_count\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexperimental_get_tracing_count\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py\u001b[0m in \u001b[0;36m_call\u001b[0;34m(self, *args, **kwds)\u001b[0m\n\u001b[1;32m    853\u001b[0m       \u001b[0;31m# In this case we have created variables on the first call, so we run the\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    854\u001b[0m       \u001b[0;31m# defunned version which is guaranteed to never create variables.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 855\u001b[0;31m       \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_stateless_fn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# pylint: disable=not-callable\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    856\u001b[0m     \u001b[0;32melif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_stateful_fn\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    857\u001b[0m       \u001b[0;31m# Release the lock early so that multiple threads can perform the call\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.8/site-packages/tensorflow/python/eager/function.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   2940\u001b[0m       (graph_function,\n\u001b[1;32m   2941\u001b[0m        filtered_flat_args) = self._maybe_define_function(args, kwargs)\n\u001b[0;32m-> 2942\u001b[0;31m     return graph_function._call_flat(\n\u001b[0m\u001b[1;32m   2943\u001b[0m         filtered_flat_args, captured_inputs=graph_function.captured_inputs)  # pylint: disable=protected-access\n\u001b[1;32m   2944\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.8/site-packages/tensorflow/python/eager/function.py\u001b[0m in \u001b[0;36m_call_flat\u001b[0;34m(self, args, captured_inputs, cancellation_manager)\u001b[0m\n\u001b[1;32m   1916\u001b[0m         and executing_eagerly):\n\u001b[1;32m   1917\u001b[0m       \u001b[0;31m# No tape is watching; skip to running the function.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1918\u001b[0;31m       return self._build_call_outputs(self._inference_function.call(\n\u001b[0m\u001b[1;32m   1919\u001b[0m           ctx, args, cancellation_manager=cancellation_manager))\n\u001b[1;32m   1920\u001b[0m     forward_backward = self._select_forward_and_backward_functions(\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.8/site-packages/tensorflow/python/eager/function.py\u001b[0m in \u001b[0;36mcall\u001b[0;34m(self, ctx, args, cancellation_manager)\u001b[0m\n\u001b[1;32m    553\u001b[0m       \u001b[0;32mwith\u001b[0m \u001b[0m_InterpolateFunctionError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    554\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mcancellation_manager\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 555\u001b[0;31m           outputs = execute.execute(\n\u001b[0m\u001b[1;32m    556\u001b[0m               \u001b[0mstr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msignature\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    557\u001b[0m               \u001b[0mnum_outputs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_num_outputs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.8/site-packages/tensorflow/python/eager/execute.py\u001b[0m in \u001b[0;36mquick_execute\u001b[0;34m(op_name, num_outputs, inputs, attrs, ctx, name)\u001b[0m\n\u001b[1;32m     57\u001b[0m   \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     58\u001b[0m     \u001b[0mctx\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mensure_initialized\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 59\u001b[0;31m     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,\n\u001b[0m\u001b[1;32m     60\u001b[0m                                         inputs, attrs, num_outputs)\n\u001b[1;32m     61\u001b[0m   \u001b[0;32mexcept\u001b[0m \u001b[0mcore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_NotOkStatusException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "cnt = 1\n",
    "for epoch in range(EPOCHS):\n",
    "    idx = np.random.randint(0, training_data.shape[0], BATCH_SIZE)\n",
    "    x_real = training_data[idx]\n",
    "    noise= np.random.normal(0, 1, (BATCH_SIZE, NOISE_SIZE))\n",
    "    x_fake = generator.predict(noise)\n",
    "    \n",
    "    discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)\n",
    "\n",
    "    discriminator_metric_generated = discriminator.train_on_batch(x_fake, y_fake)\n",
    " \n",
    "    discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)\n",
    "    \n",
    "    generator_metric = combined.train_on_batch(noise, y_real)\n",
    "    if epoch % SAVE_FREQ == 0:\n",
    "        save_images(cnt, fixed_noise)\n",
    "        cnt += 1\n",
    "        print(f'{epoch} epoch, Discriminator accuracy: {100*  discriminator_metric[1]}, Generator accuracy: {100 * generator_metric[1]}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
