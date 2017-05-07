import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

filepath = "../FallPredictor.hd5"
test_data_dir = "../data/Testing"

# dimensions of our images.
img_width, img_height = 400, 266
batch_size = 16

nb_validation_samples = 29
steps = 1

model = keras.models.load_model(filepath)

datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')


# -- Evaluate generator -- #
result = model.evaluate_generator(
	generator=validation_generator,
	steps=nb_validation_samples)
print("Model [loss, accuracy]: {0}".format(result))


# -- Predict generator -- #
predict = model.predict_generator(
    generator=validation_generator,
    steps=nb_validation_samples)

print("model predictions: {0}".format(predict))
