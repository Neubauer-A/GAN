import os
import tensorflow as tf

from PIL import Image
from numpy.random import randint
from tensorflow.keras.utils import to_categorical

def gan_model_check(epoch, g_model, latent_dim, n_cats, images):
    os.system('mkdir %s' %(epoch))
    random_vectors, _ = cat_gen_input(images, latent_dim, n_cats)
    generated_images = g_model(random_vectors)
    generated_images = generated_images.numpy()
    generated_images = (generated_images+1)*127.5
    generated_images = generated_images.astype('uint8')
    for i in range(images):
        image = Image.fromarray(generated_images[i])
        image.save(str(epoch)+'/'+str(epoch)+str(i)+'.png')
    g_model.save(str(epoch)+'/generator.h5')

def can_model_check(epoch, g_model, latent_dim, n_cats, images):
    os.system('mkdir %s' %(epoch))
    random_vectors, _ = ambiguous_gen_input(images, latent_dim, n_cats)
    generated_images = g_model(random_vectors)
    generated_images = generated_images.numpy()
    generated_images = (generated_images+1)*127.5
    generated_images = generated_images.astype('uint8')
    for i in range(images):
        image = Image.fromarray(generated_images[i])
        image.save(str(epoch)+'/'+str(epoch)+str(i)+'.png')
    g_model.save(str(epoch)+'/generator.h5')

def cat_gen_input(batch_size, latent_dim, n_cats):
    control_codes = randint(0, n_cats, batch_size)
    control_codes = to_categorical(control_codes, n_cats)
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    return tf.concat([random_latent_vectors, control_codes], axis=1), control_codes

def ambiguous_gen_input(batch_size, latent_dim, n_cats):
    control_codes = tf.ones((batch_size, n_cats)) / n_cats
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    return tf.concat([random_latent_vectors, control_codes], axis=1), control_codes

def train(g_model, d_model, gan_model, can_model, processor, data, shuffle=90000, latent_dim=100, 
          n_cats=10, batch_size=64, epochs=100, checkpoint=5, num_images=8, min_p1_epochs=10):
    
    dataset = processor.load_dataset(data, batch_size=batch_size, shuffle=shuffle)
    
    def train_phase_1(epoch):
        print('train phase 1:')
        step = 0
        for batch in dataset:
            batch_size = tf.shape(batch)[0]
            # train discrimininator on real images
            labels = tf.zeros((batch_size, 1))
            labels += 0.05 * tf.random.uniform(tf.shape(labels))
            d_loss_1 = d_model.train_on_batch(batch, labels)
            # train discriminator on generated images
            random_vectors, _ = cat_gen_input(batch_size, latent_dim, n_cats)
            images = g_model(random_vectors)
            labels = tf.ones((batch_size, 1))
            labels += 0.05 * tf.random.uniform(tf.shape(labels))
            d_loss_2 = d_model.train_on_batch(images, labels)
            # train gan model
            random_vectors, cat_codes = cat_gen_input(batch_size, latent_dim, n_cats)
            misleading_labels = tf.zeros((batch_size, 1)) 
            _, gan_loss_1, gan_loss_2 = gan_model.train_on_batch(random_vectors, [misleading_labels, cat_codes])
            step += 1
            print('>%d: %d, d_loss_1=%.3f, d_loss_2=%.3f, gan_loss_1=%.3f, gan_loss_2=%.3f' \
                  %(epoch+1, step, d_loss_1, d_loss_2, gan_loss_1, gan_loss_2))
        if (epoch+1) % checkpoint == 0:
            gan_model_check(epoch+1, g_model, latent_dim, n_cats, num_images)
        if gan_loss_2 < 0.1 and epoch+1 >= min_p1_epochs:
            return True
        return False

    def train_phase_2(epoch):
        print('train phase 2:')
        step = 0
        for batch in dataset:
            batch_size = tf.shape(batch)[0]
            # train discrimininator on real images
            labels = tf.zeros((batch_size, 1))
            labels += 0.05 * tf.random.uniform(tf.shape(labels))
            d_loss_1 = d_model.train_on_batch(batch, labels)
            # train discriminator on generated images
            random_vectors, _ = cat_gen_input(batch_size, latent_dim, n_cats)
            images = g_model(random_vectors)
            labels = tf.ones((batch_size, 1))
            labels += 0.05 * tf.random.uniform(tf.shape(labels))
            d_loss_2 = d_model.train_on_batch(images, labels)
            # train can model
            random_vectors, amb_codes = ambiguous_gen_input(batch_size, latent_dim, n_cats)
            misleading_labels = tf.zeros((batch_size, 1)) 
            _, can_loss_1, can_loss_2 = can_model.train_on_batch(random_vectors, [misleading_labels, amb_codes])
            step += 1
            print('>%d: %d, d_loss_1=%.3f, d_loss_2=%.3f, can_loss_1=%.3f, can_loss_2=%.3f' \
                  %(epoch+1, step, d_loss_1, d_loss_2, can_loss_1, can_loss_2))
            if (epoch+1) % checkpoint == 0:
                can_model_check(epoch+1, g_model, latent_dim, n_cats, num_images)        
    
    phase_1_done = False
    for epoch in range(epochs):
        if not phase_1_done:
            phase_1_done = train_phase_1(epoch)
        else:
            train_phase_2(epoch)
