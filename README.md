# ExamenTD
Examen Tratamientos de Datos
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator  


width_shape = 300
height_shape = 300


names = ['CLASS_01', 'CLASS_02', 'CLASS_03', 'CLASS_04', 'CLASS_05', 'CLASS_06', 'CLASS_07', 'CLASS_08']


model = Sequential()


model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width_shape, height_shape, 3)))


model.add(MaxPooling2D((2, 2)))


model.add(Conv2D(64, (3, 3), activation='relu'))


model.add(MaxPooling2D((2, 2)))


model.add(Conv2D(128, (3, 3), activation='relu'))


model.add(MaxPooling2D((2, 2)))


model.add(Flatten())


model.add(Dense(128, activation='relu'))


model.add(Dense(len(names), activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


train_data_dir = 'C:/Users/victo/Documents/Examen/test'
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(width_shape, height_shape),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(width_shape, height_shape),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)


model.save('modelo.h5')

model.summary()


from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_data_dir = 'C:/Users/victo/Documents/Examen/train'


width_shape = 300
height_shape = 300


train_datagen = ImageDataGenerator(
    rescale=1.0/255,             
    validation_split=0.2,        
    rotation_range=20,           
    width_shift_range=0.2,       
    height_shift_range=0.2,      
    shear_range=0.2,             
    zoom_range=0.2,              
    horizontal_flip=True,        
    fill_mode='nearest'          
)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(width_shape, height_shape),
    batch_size=32,               
    class_mode='categorical',    
    subset='training'            
)


validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(width_shape, height_shape),
    batch_size=32,               
    class_mode='categorical',    
    subset='validation'          
)


import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

custom_Model = tf.keras.models.load_model('modelo.h5')

class_names = ['CLASS_01', 'CLASS_02', 'CLASS_03', 'CLASS_04', 'CLASS_05', 'CLASS_06', 'CLASS_07', 'CLASS_08']

image_path = 'C:/Users/victo/Documents/Examen/test/CLASS_01/14-CAPTURE_20220523_141530_080.png'

img = image.load_img(image_path, target_size=(300, 300))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  
img_array /= 255.0  

predictions = custom_Model.predict(img_array)

predicted_class_index = np.argmax(predictions[0])
predicted_class_name = class_names[predicted_class_index]
accuracy_percentage = 100 * predictions[0][predicted_class_index]

print('Esta imagen parece ser {} con un {:.2f}% de exactitud.'.format(predicted_class_name, accuracy_percentage))

model.complice(optimizer='adam'
              loss=tf.keras.losses.SparseCategoricalCrossentropy[from_logits=True],
              metrics={'accurancy'})
