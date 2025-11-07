from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_test_generator(config):
    test_dir = config["data"]["test_dir"]
    image_size = tuple(config["data"]["image_size"])
    batch_size = config["data"]["batch_size"]

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return test_generator
