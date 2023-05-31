## -----------------------------------------------------------------------
knitr::purl("ConvolutionalNeuralNetworkChest-XRay.Rmd")


## -----------------------------------------------------------------------
pacman::p_load(tidyverse, keras, tensorflow, EBImage)


## -----------------------------------------------------------------------
base_dir <- list.dirs(path = "./data/chest_xray", recursive = T)

base_dir
sapply(base_dir, function(dir){length(list.files(dir))})



## -----------------------------------------------------------------------
train_dir <- file.path("./data/chest_xray/train")
validation_dir <- file.path("./data/chest_xray/val")

train_normal <- list.dirs(train_dir, recursive = T)[2]
train_pneumonia <- list.dirs(train_dir, recursive = T)[3]

# validation
val_normal <- list.dirs(validation_dir, recursive = T)[2]
val_pneumonia <- list.dirs(validation_dir, recursive = T)[3]

# testing 
test_dir <- file.path("./data/chest_xray/test")


## -----------------------------------------------------------------------
normal_lung <- list.files(path = train_normal, full.names = T) |> 
  sample(size = 10, replace = FALSE)

pneumonia_lung <- list.files(path = train_pneumonia, full.names = T) |> 
  sample(size = 10, replace = FALSE)


images_store <- c(normal_lung, pneumonia_lung)

for (i in seq.int(images_store)) {
  readImage(images_store[i]) |> 
    resize(w = 300, h = 300) |> 
    writeImage(images_store[i])
}



EBImage::display(
  readImage(images_store),
  method = 'raster',
  all = T,
  nx = 5, # number of columns
  spacing = c(5,5) # white space between images (a,b), a gives space for vertical and b gives space for horizontal 
)


## -----------------------------------------------------------------------
resize = c(450,450)

train_generator <- flow_images_from_directory(
 directory = train_dir,
 generator = image_data_generator(rescale = 1/255),
 # resizing the images to the same dimensions 
 target_size = resize,
 # number of images to be fed to CNN
 batch_size = 32,
 # only two classes
 class_mode = "binary"
)




## -----------------------------------------------------------------------
validation_generator <- flow_images_from_directory(
 directory = validation_dir,
 # scale pixels
 generator = image_data_generator(rescale = 1/255),
  # resizing the images to the same dimensions
 target_size = resize,
 # number of images to be fed to CNN
 batch_size = 32,
 # only two classes
 class_mode = "binary"
)


## -----------------------------------------------------------------------
model <- keras_model_sequential() |> 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), 
                padding = "same", activation = "relu",
                input_shape = c(resize,3)) |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 128, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 256, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_flatten() |> 
  layer_dropout(rate=.7) |> 
  layer_dense(units = 512, activation = "relu") |> 
  layer_dense(units = 1, activation = "sigmoid")
  


## -----------------------------------------------------------------------
model |> 
  compile(loss = "binary_crossentropy",
          optimizer = optimizer_rmsprop(learning_rate = 0.0001),
          metrics = "accuracy")


## -----------------------------------------------------------------------
his <- model |> 
  keras::fit(
    train_generator,
    # sample size / batch size = (1341+3875)/32 = 136
    #steps_per_epoch = 136,
    epoch = 10,
    validation_data = validation_generator
  )


## -----------------------------------------------------------------------
his


## -----------------------------------------------------------------------
model %>% save_model_hdf5("./model/normal_abnormal_lungs.h5") 



## -----------------------------------------------------------------------
test_generator <- flow_images_from_directory(
  directory = test_dir,
  target_size = resize,
  class_mode = "binary",
  shuffle = FALSE,
  seed = 1,
  generator =image_data_generator(rescale = 1/255)
)


## -----------------------------------------------------------------------
fit1 <- load_model_hdf5("./model/normal_abnormal_lungs.h5")
evaluate(fit1, test_generator)


## -----------------------------------------------------------------------
predict(fit1, test_generator)


## -----------------------------------------------------------------------
train1_dir <- file.path("./data/chest_xray/train1")
validation1_dir <- file.path("./data/chest_xray/val1")

train1_normal <- list.dirs(train1_dir, recursive = T)[2]
train1_pneumonia <- list.dirs(train1_dir, recursive = T)[3]



## ----echo = FALSE, eval=FALSE-------------------------------------------
## 
## # index_normal <- list.files(path = train1_normal, full.names = T) |>
## #   sample(size = length(list.files(train1_normal)) - 50 , replace = FALSE)
## 
## 
## # files_normal <- list.files(path = train1_normal, full.names = T)
## # random_normal <- files_normal[!files_normal %in% index_normal]
## 
## # file.copy(random_normal, "./data/chest_xray/val1/NORMAL/")
## # file.remove(random_normal)
## 
## # list.files(train1_normal) |> length()
## # list.files(train_normal) |> length()


## ----echo = FALSE, eval=FALSE-------------------------------------------
## # index_pneumonia <- list.files(path = train1_pneumonia, full.names = T) |>
## #   sample(size = length(list.files(train1_pneumonia)) - 50 , replace = FALSE)
## 
## 
## # files_pneumonia <- list.files(path = train1_pneumonia, full.names = T)
## # random_pneumonia <- files_pneumonia[!files_pneumonia %in% index_pneumonia]
## 
## # file.copy(random_pneumonia, "./data/chest_xray/val1/PNEUMONIA/")
## # file.remove(random_pneumonia)
## 
## # list.files(train1_pneumonia) |> length()
## # list.files(train_pneumonia) |> length()


## -----------------------------------------------------------------------
resize = c(500,500)

train1_generator <- flow_images_from_directory(
 directory = train1_dir,
 generator = image_data_generator(rescale = 1/255),
 # resizing the images to the same dimensions 
 target_size = resize,
 # number of images to be fed to CNN
 batch_size = 32,
 # only two classes
 class_mode = "binary"
)




## -----------------------------------------------------------------------
validation1_generator <- flow_images_from_directory(
 directory = validation1_dir,
 # scale pixels
 generator = image_data_generator(rescale = 1/255),
  # resizing the images to the same dimensions
 target_size = resize,
 # number of images to be fed to CNN
 batch_size = 32,
 # only two classes
 class_mode = "binary"
)


## -----------------------------------------------------------------------
model2 <- keras_model_sequential() |> 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), 
                padding = "same", activation = "relu",
                input_shape = c(resize,3)) |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 128, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 256, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_flatten() |> 
  layer_dropout(rate=.7) |> 
  layer_dense(units = 512, activation = "relu") |> 
  layer_dense(units = 1, activation = "sigmoid")
  


## -----------------------------------------------------------------------
model2 |> 
  compile(loss = "binary_crossentropy",
          optimizer = optimizer_rmsprop(learning_rate = 0.0001),
          metrics = "accuracy")


## -----------------------------------------------------------------------
his2 <- model2 |> 
  keras::fit(
    train1_generator,
    # sample size / batch size = (1341+3875)/32 = 136
    #steps_per_epoch = 136,
    epoch = 10,
    validation_data = validation1_generator
  )


## -----------------------------------------------------------------------
his2


## -----------------------------------------------------------------------
model2 %>% save_model_hdf5("./model/normal_abnormal_lungs_add_val.h5") 



## -----------------------------------------------------------------------
test_generator <- flow_images_from_directory(
  directory = test_dir,
  target_size = resize,
  class_mode = "binary",
  shuffle = FALSE,
  seed = 1,
  generator = image_data_generator(rescale = 1/255)
)


## -----------------------------------------------------------------------
fit2 <- load_model_hdf5("./model/normal_abnormal_lungs_add_val.h5")
fit2


evaluate(fit2, test_generator)


## -----------------------------------------------------------------------
train_aug <- image_data_generator(
  rescale = 1/255,
  rotation_range = 50, 
  zoom_range = 0.3,
  shear_range = 0.1,
  width_shift_range = 0.3,
  height_shift_range = 0.25,
  horizontal_flip = T,
  fill_mode = "nearest"
)

augmented_train_generator <- flow_images_from_directory(
  directory = train_dir,
  generator = train_aug,
  target_size = resize,
  batch_size = 32,
  class_mode = "binary",
  shuffle = TRUE          
)

# validation set
validation_generator


## -----------------------------------------------------------------------
model1 <- keras_model_sequential() |> 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), 
                padding = "same", activation = "relu",
                input_shape = c(resize,3)) |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 128, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 256, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_flatten() |> 
  layer_dropout(rate=.6) |> 
  layer_dense(units = 512, activation = "relu") |> 
  layer_dense(units = 1, activation = "sigmoid")
  


## -----------------------------------------------------------------------
model1 |> 
  compile(loss = "binary_crossentropy",
          optimizer = optimizer_rmsprop(learning_rate = 0.0001),
          metrics = "accuracy")


## -----------------------------------------------------------------------
his <- model1 |> 
  keras::fit(
    augmented_train_generator,
    # sample size / batch size = (1341+3875)/32 = 136
    #steps_per_epoch = 136,
    epoch = 10,
    validation_data = validation_generator
  )


## -----------------------------------------------------------------------
model1 %>% save_model_hdf5("./model/normal_abnormal_lungs_augmentation.h5") 



## -----------------------------------------------------------------------
his
model |>
  evaluate(test_x, test_y)


## -----------------------------------------------------------------------
his1 <- model |> 
  keras::fit(
    generator = augmented_train_generator,
    # sample size / batch size = (1341+3875)/32 = 136
    #steps_per_epoch = 136,
    epoch = 10,
    validation_data = validation_generator
  )

